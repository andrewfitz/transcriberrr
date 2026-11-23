import AppKit
import Darwin
import Security
import SwiftUI

enum ServerEngine: String, CaseIterable, Identifiable {
    case qwen = "qwen3"
    case parakeet = "parakeet-v3"
    case apple = "apple-native"

    var id: String { rawValue }
    var pass: TranscriptionPass { TranscriptionPass(engine: self == .qwen ? "qwen" : rawValue) }
    var defaultModel: String { self == .qwen ? "Qwen/Qwen3-ASR-0.6B" : pass.defaultModel }
}

struct ServerOptions {
    let maxSessions: String
    let maxTracks: String
    let originList: String
    let diarize: Bool
    let alignWords: Bool
}

/// Keychain calls can block, for example while macOS asks whether to allow access.
/// They run one at a time on their own queue instead of the main actor.
private enum ServerKeychain {
    private static let queue = DispatchQueue(label: "com.transcriberrr.server-keychain")

    private static var identity: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.transcriberrr.realtime-server",
            kSecAttrAccount as String: "api-key"
        ]
    }

    struct Failure: LocalizedError {
        let status: OSStatus
        var errorDescription: String? {
            let detail = SecCopyErrorMessageString(status, nil) as String? ?? "Error \(status)"
            return "Couldn’t access the server key in Keychain. \(detail)"
        }
    }

    static func read() async throws -> String {
        try await perform { try readNow() }
    }

    /// Stores the key unless it is already stored. A failed read stores nothing, so
    /// a key that could not be read is never replaced.
    static func save(_ key: String) async throws {
        try await perform {
            if try readNow() != key { try storeNow(key) }
        }
    }

    private static func perform<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { continuation.resume(with: Result { try body() }) }
        }
    }

    private static func readNow() throws -> String {
        var query = identity
        query[kSecReturnData as String] = true
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return "" }
        guard status == errSecSuccess else { throw Failure(status: status) }
        guard let data = result as? Data, let key = String(data: data, encoding: .utf8) else {
            throw Failure(status: errSecDecode)
        }
        return key
    }

    private static func storeNow(_ key: String) throws {
        if key.isEmpty {
            let status = SecItemDelete(identity as CFDictionary)
            guard status == errSecSuccess || status == errSecItemNotFound else {
                throw Failure(status: status)
            }
            return
        }
        let keyData = [kSecValueData as String: Data(key.utf8)]
        var status = SecItemUpdate(identity as CFDictionary, keyData as CFDictionary)
        if status == errSecItemNotFound {
            status = SecItemAdd(identity.merging(keyData) { _, value in value } as CFDictionary, nil)
        }
        guard status == errSecSuccess else { throw Failure(status: status) }
    }
}

@MainActor
final class RealtimeServer: ObservableObject {
    enum Phase: String {
        case stopped = "Stopped"
        case starting = "Starting"
        case loading = "Loading model"
        case ready = "Ready"
        case draining = "Stopping"
        case failed = "Failed"

        var isActive: Bool {
            switch self {
            case .stopped, .failed: false
            case .starting, .loading, .ready, .draining: true
            }
        }
    }

    private struct Event: Decodable {
        let event: String
        let address: String?
        let message: String?
    }

    @Published private(set) var phase = Phase.stopped
    @Published private(set) var address: String?
    @Published private(set) var endpoint: URL?
    @Published private(set) var errorMessage: String?
    var healthURL: URL? {
        guard phase == .loading || phase == .ready, let address else { return nil }
        return Self.clientURL(address)?.appending(path: "healthz")
    }

    private(set) var activeAPIKey: String?
    private var process: Process?
    private var input: FileHandle?
    private var outputBuffer = Data()
    private var shutdownTask: Task<Void, Never>?
    // Saving the key to Keychain before the process launches.
    private var launchTask: Task<Void, Never>?

    func start(cliURL: URL?, nativeURL: URL?, bind: String, portText: String, engine: ServerEngine, model: String,
               offline: Bool, apiKey: String, options: ServerOptions) {
        guard process == nil, launchTask == nil else { return }
        errorMessage = nil
        address = nil
        endpoint = nil

        let bind = bind.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !bind.utf8.contains(0) else {
            return fail("Enter a numeric IPv4 or IPv6 bind address.")
        }
        var ipv4 = in_addr()
        var ipv6 = in6_addr()
        let isIPv4 = inet_pton(AF_INET, bind, &ipv4) == 1
        guard isIPv4 || inet_pton(AF_INET6, bind, &ipv6) == 1 else {
            return fail("Enter a numeric IPv4 or IPv6 bind address.")
        }
        guard let port = UInt16(portText.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return fail("Enter a port from 0 to 65535.")
        }
        guard let maxSessions = Int(options.maxSessions.trimmingCharacters(in: .whitespacesAndNewlines)),
              (1...1024).contains(maxSessions) else {
            return fail("Enter a session limit from 1 to 1024.")
        }
        guard let maxTracks = Int(options.maxTracks.trimmingCharacters(in: .whitespacesAndNewlines)),
              (1...64).contains(maxTracks) else {
            return fail("Enter a track limit from 1 to 64.")
        }
        let isLoopback = isIPv4 ? (UInt32(bigEndian: ipv4.s_addr) >> 24 == 127) :
            withUnsafeBytes(of: ipv6) { bytes in
                withUnsafeBytes(of: in6addr_loopback) { bytes.elementsEqual($0) }
            }
        guard isLoopback || !apiKey.isEmpty else {
            return fail("A bind outside loopback requires a bearer key.")
        }
        guard apiKey.utf8.allSatisfy({ (33...126).contains($0) }) else {
            return fail("Use printable ASCII characters without spaces in the bearer key.")
        }
        if engine == .apple {
            guard #available(macOS 26, *) else {
                return fail("Apple SpeechTranscriber requires macOS 26 or later.")
            }
        }
        guard let cliURL else { return fail("Couldn’t find the Transcriberrr CLI.") }

        var arguments = ["serve", "--bind", bind, "--port", String(port), "--engine", engine.rawValue, "--exit-on-stdin-eof"]
        arguments += ["--max-sessions", String(maxSessions), "--max-tracks", String(maxTracks)]
        arguments += options.originList.split { $0.isWhitespace || $0 == "," }.map { "--allow-origin=\($0)" }
        if !model.isEmpty, engine != .apple { arguments.append("--model=\(model)") }
        if offline { arguments.append("--offline") }
        if options.diarize { arguments += ["--diarization-model", "sortformer"] }
        if options.alignWords, engine == .qwen { arguments += ["--forced-aligner", "forced-aligner"] }

        // Active from here, so a second Start cannot launch another process and Stop
        // can cancel the launch.
        phase = .starting
        launchTask = Task { [weak self] in
            do {
                try await ServerKeychain.save(apiKey)
            } catch {
                guard let self, !Task.isCancelled else { return }
                self.launchTask = nil
                return self.fail(error.localizedDescription)
            }
            guard let self, !Task.isCancelled else { return }
            self.launchTask = nil
            self.launch(cliURL: cliURL, nativeURL: nativeURL, arguments: arguments, apiKey: apiKey)
        }
    }

    private func launch(cliURL: URL, nativeURL: URL?, arguments: [String], apiKey: String) {
        let process = Process()
        process.executableURL = cliURL
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        environment["TRANSCRIBERRR_API_KEY"] = apiKey.isEmpty ? nil : apiKey
        if let nativeURL { environment["TRANSCRIBERRR_NATIVE"] = nativeURL.path }
        process.environment = environment

        let inputPipe = Pipe()
        process.standardInput = inputPipe
        let output = PipeCollector { [weak self, weak process] data in
            DispatchQueue.main.async {
                guard let self, let process, self.process === process else { return }
                self.receive(data, apiKey: apiKey)
            }
        }
        let errors = PipeCollector(limit: 16 * 1024)
        process.standardOutput = output.pipe
        process.standardError = errors.pipe
        process.terminationHandler = { [weak self] process in
            let events = output.finish(waitForEOF: false)
            let detail = errors.finish(waitForEOF: false)
            DispatchQueue.main.async {
                self?.processExited(process, events: events, detail: detail, apiKey: apiKey)
            }
        }

        outputBuffer.removeAll()
        self.process = process
        input = inputPipe.fileHandleForWriting
        activeAPIKey = apiKey
        phase = .starting
        do {
            try process.run()
        } catch {
            process.terminationHandler = nil
            _ = output.finishData(waitForEOF: false)
            _ = errors.finishData(waitForEOF: false)
            try? input?.close()
            input = nil
            self.process = nil
            activeAPIKey = nil
            fail(redact(error.localizedDescription, key: apiKey))
        }
    }

    func stop() {
        if let launchTask {
            // Still saving the key. No process has been launched.
            launchTask.cancel()
            self.launchTask = nil
            phase = .stopped
            return
        }
        guard let process, shutdownTask == nil else { return }
        phase = .draining
        // EOF follows the same bounded drain as a signal. The app also releases this
        // sole write end when force-quit, even when it cannot run a termination handler.
        try? input?.close()
        input = nil
        shutdownTask = Task { [weak self, weak process] in
            do { try await Task.sleep(for: .seconds(20)) } catch { return }
            guard let self, let process, self.process === process, process.isRunning else { return }
            Darwin.kill(process.processIdentifier, SIGKILL)
        }
    }

    private func receive(_ data: Data, apiKey: String) {
        outputBuffer.append(data)
        while let newline = outputBuffer.firstIndex(of: 10) {
            let line = Data(outputBuffer[..<newline])
            outputBuffer.removeSubrange(...newline)
            guard let event = try? JSONDecoder().decode(Event.self, from: line) else { continue }
            switch event.event {
            case "listening":
                guard phase != .draining, let address = event.address else { continue }
                self.address = address
                phase = .loading
                if let base = Self.clientURL(address) {
                    endpoint = Self.realtimeURL(base)
                }
            case "ready":
                if phase != .draining { phase = .ready }
            case "draining":
                stop()
            case "error":
                errorMessage = redact(event.message ?? "The server reported an error.", key: apiKey)
            default: break
            }
        }
        if outputBuffer.count > 256 * 1024 { outputBuffer.removeAll() }
    }

    private func processExited(_ process: Process, events: String, detail: String, apiKey: String) {
        guard self.process === process else { return }
        let wasStopping = phase == .draining
        try? input?.close()
        input = nil
        shutdownTask?.cancel()
        shutdownTask = nil
        self.process = nil
        activeAPIKey = nil
        address = nil
        endpoint = nil
        outputBuffer.removeAll()

        if wasStopping && process.terminationReason == .exit && process.terminationStatus == 0 {
            errorMessage = nil
            phase = .stopped
        } else {
            let reported = events.split(whereSeparator: \.isNewline).reversed().lazy.compactMap { line in
                try? JSONDecoder().decode(Event.self, from: Data(line.utf8))
            }.first { $0.event == "error" }?.message
            let lastError = Self.lastLines(of: redact(detail, key: apiKey))
            let fallback: String
            if wasStopping {
                fallback = "The server stopped before finishing its shutdown."
            } else if process.terminationReason == .uncaughtSignal {
                fallback = "The server was terminated by signal \(process.terminationStatus)."
            } else {
                fallback = "The server exited (status \(process.terminationStatus))."
            }
            fail(reported.map { redact($0, key: apiKey) } ?? errorMessage ?? lastError ?? fallback)
        }
    }

    /// The last few non-empty lines of the server's standard error. A single last
    /// line is often only a hint, such as clap's "For more information, try '--help'."
    private static func lastLines(of text: String, count: Int = 4, limit: Int = 1_000) -> String? {
        let lines = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .suffix(count)
        guard !lines.isEmpty else { return nil }
        let joined = lines.joined(separator: "\n")
        return joined.count > limit ? "…" + joined.suffix(limit - 1) : joined
    }

    private func fail(_ message: String) {
        errorMessage = message
        phase = .failed
    }

    private func redact(_ text: String, key: String) -> String {
        key.isEmpty ? text : text.replacingOccurrences(of: key, with: "[redacted]")
    }

    private static func clientURL(_ address: String) -> URL? {
        guard var parts = URLComponents(string: "http://\(address)") else { return nil }
        switch parts.host {
        case "0.0.0.0": parts.host = "127.0.0.1"
        case "[::]": parts.host = "[::1]"
        default: break
        }
        return parts.url
    }

    private static func realtimeURL(_ base: URL) -> URL? {
        var parts = URLComponents(url: base, resolvingAgainstBaseURL: false)
        parts?.scheme = "ws"
        parts?.path = "/v1/realtime"
        parts?.queryItems = [URLQueryItem(name: "intent", value: "transcription")]
        return parts?.url
    }
}

@MainActor
struct ServerPanel: View {
    @ObservedObject var server: RealtimeServer
    @ObservedObject var state: AppState

    @Environment(\.dismiss) private var dismiss
    @AppStorage("serverBind") private var bind = "127.0.0.1"
    @AppStorage("serverPort") private var port = "0"
    @AppStorage("serverEngine") private var engine = ServerEngine.parakeet
    @AppStorage("serverQwenModel") private var qwenModel = ""
    @AppStorage("serverParakeetModel") private var parakeetModel = ""
    @AppStorage("serverOffline") private var offline = false
    @AppStorage("serverMaxSessions") private var maxSessions = "2"
    @AppStorage("serverMaxTracks") private var maxTracks = "2"
    @AppStorage("serverAllowedOrigins") private var originList = ""
    @AppStorage("serverDiarizeMixedTracks") private var diarize = false
    @AppStorage("serverAlignWords") private var alignWords = false
    private struct Health: Decodable, Equatable {
        let activeSessions: Int
        let maxSessions: Int
        let backlog: Int

        enum CodingKeys: String, CodingKey {
            case activeSessions = "active_sessions"
            case maxSessions = "max_sessions"
            case backlog
        }
    }

    @State private var health: Health?
    @State private var apiKey = ""
    @State private var keyError: String?
    @State private var needsInitialKey = true
    @State private var keyLoadFailed = false
    // Until the first read finishes, the field is empty because nothing has loaded.
    @State private var keyLoading = true
    @State private var keyRead = 0

    private var model: Binding<String> { engine == .qwen ? $qwenModel : $parakeetModel }
    private var displayedKey: Binding<String> {
        if let key = server.activeAPIKey { return .constant(key) }
        return $apiKey
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Realtime Server").font(.title2.bold())
                Spacer()
                if server.phase == .starting || server.phase == .loading || server.phase == .draining {
                    ProgressView().controlSize(.small)
                }
                Text(server.phase.rawValue)
                    .foregroundStyle(server.phase == .ready ? Color.green : .secondary)
                    .accessibilityLabel("Server status, \(server.phase.rawValue)")
            }

            if let error = server.errorMessage ?? keyError {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }

            if server.phase.isActive {
                Text("Batch transcription is paused until the server stops.")
                    .foregroundStyle(.secondary)
            } else if state.isRunning {
                Text("The server can start once the current batch finishes.")
                    .foregroundStyle(.secondary)
            } else if state.downloadingLibraryModel != nil {
                Text("The server can start once the model download finishes.")
                    .foregroundStyle(.secondary)
            }

            if let endpoint = server.endpoint {
                LabeledContent("Endpoint") {
                    HStack {
                        Text(endpoint.absoluteString).font(.system(.body, design: .monospaced))
                            .textSelection(.enabled)
                        Button("Copy", systemImage: "document.on.document") { copy(endpoint.absoluteString) }
                            .labelStyle(.iconOnly)
                            .help("Copy server endpoint")
                    }
                }
            }

            if let address = server.address { LabeledContent("Listening on", value: address) }
            if let health {
                LabeledContent("Session slots", value: "\(health.activeSessions) of \(health.maxSessions)")
                LabeledContent("Backlog", value: health.backlog.formatted())
            }

            Form {
                Group {
                    TextField("Bind address", text: $bind, prompt: Text("127.0.0.1"))
                    TextField("Port", text: $port, prompt: Text("0"))
                        .help("Use 0 to pick an available port")
                    Picker("Engine", selection: $engine) {
                        ForEach(ServerEngine.allCases) { engine in
                            Text(engine.pass.displayName).tag(engine)
                        }
                    }
                    if engine != .apple {
                        TextField("Model", text: model, prompt: Text(engine.defaultModel))
                    }
                    Toggle("Use cached models only", isOn: $offline)
                }
                .disabled(server.phase.isActive)
                HStack {
                    SecureField("Bearer key", text: displayedKey)
                        .help("Stored in Keychain when you start the server")
                        .disabled(server.phase.isActive)
                    Button("Generate Key") { apiKey = UUID().uuidString + UUID().uuidString }
                        .disabled(server.phase.isActive)
                    Button("Copy Key", systemImage: "document.on.document") { copy(displayedKey.wrappedValue) }
                        .labelStyle(.iconOnly)
                        .disabled(displayedKey.wrappedValue.isEmpty)
                        .help("Copy bearer key")
                }
                advancedOptions
            }

            HStack {
                Button("Done") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                if server.phase.isActive {
                    Button("Stop Server") { server.stop() }
                        .disabled(server.phase == .draining)
                } else {
                    Button("Start Server") {
                        // The field is empty because the stored key could not be read.
                        // Starting now would replace that key with no key.
                        if keyLoadFailed && apiKey.isEmpty {
                            Task {
                                await loadKey()
                                if !keyLoadFailed { startServer() }
                            }
                            return
                        }
                        startServer()
                    }
                    .buttonStyle(.borderedProminent)
                    // While the stored key loads, the field is empty for the same reason.
                    .disabled(!state.canStartRealtimeServer || keyLoading)
                }
            }
        }
        .padding(24)
        .frame(width: 550)
        .task(id: server.healthURL) {
            await pollHealth(server.healthURL)
        }
        .onAppear {
            loadInitialKeyIfIdle()
        }
        .onChange(of: server.phase.isActive) { _, active in
            if !active { loadInitialKeyIfIdle() }
        }
    }

    private func loadInitialKeyIfIdle() {
        // A sheet opened while the server runs displays its active key. Read the
        // stored key when it stops. Preserve edits across failed or canceled starts.
        guard needsInitialKey, !server.phase.isActive else { return }
        needsInitialKey = false
        Task { await loadKey() }
    }

    private func startServer() {
        keyError = nil
        state.startRealtimeServer(
            bind: bind, port: port, engine: engine,
            model: model.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines),
            offline: offline, apiKey: apiKey,
            options: ServerOptions(maxSessions: maxSessions, maxTracks: maxTracks,
                                   originList: originList, diarize: diarize, alignWords: alignWords)
        )
    }

    private var advancedOptions: some View {
        DisclosureGroup("Advanced") {
            VStack(alignment: .leading, spacing: 8) {
                LabeledContent("Maximum sessions") {
                    TextField("Maximum sessions", text: $maxSessions)
                        .labelsHidden()
                        .frame(width: 56)
                }
                LabeledContent("Maximum tracks per session") {
                    TextField("Maximum tracks per session", text: $maxTracks)
                        .labelsHidden()
                        .frame(width: 56)
                }
                LabeledContent("Allowed browser origins") {
                    TextField("Allowed browser origins", text: $originList,
                              prompt: Text("https://example.com"), axis: .vertical)
                        .labelsHidden()
                        .lineLimit(1...2)
                        .frame(width: 230)
                        .help("Enter exact origins separated by spaces or commas")
                }
                Toggle("Detect speakers on mixed tracks (Sortformer)", isOn: $diarize)
                if engine == .qwen {
                    Toggle("Align final Qwen words to audio", isOn: $alignWords)
                }
            }
            .padding(.top, 6)
            .disabled(server.phase.isActive)
        }
    }

    // Health is displayed only in this sheet. SwiftUI cancels its request and timer
    // when the sheet closes, the server stops, or a new address is announced.
    private func pollHealth(_ url: URL?) async {
        health = nil
        guard let url else { return }
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 2
        config.timeoutIntervalForResource = 2
        config.connectionProxyDictionary = [:]
        config.urlCache = nil
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        while !Task.isCancelled {
            let result = try? await session.data(from: url)
            guard !Task.isCancelled else { return }
            let fresh: Health?
            if let (data, response) = result, (response as? HTTPURLResponse)?.statusCode == 200 {
                fresh = try? JSONDecoder().decode(Health.self, from: data)
            } else {
                fresh = nil
            }
            if health != fresh { health = fresh }
            do { try await Task.sleep(for: .seconds(1)) } catch { return }
        }
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    /// Reads the stored key off the main thread.
    private func loadKey() async {
        keyRead += 1
        let read = keyRead
        let shown = apiKey
        keyLoading = true
        let result: Result<String, any Error>
        do {
            result = .success(try await ServerKeychain.read())
        } catch {
            result = .failure(error)
        }
        // A newer read owns the outcome.
        guard read == keyRead else { return }
        keyLoading = false
        switch result {
        case .success(let key):
            // Keep a key typed or generated while the read ran.
            if apiKey == shown { apiKey = key }
            keyError = nil
            keyLoadFailed = false
        case .failure(let error):
            keyError = error.localizedDescription
            keyLoadFailed = true
        }
    }
}
