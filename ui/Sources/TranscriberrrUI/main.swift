import AVFoundation
import AppKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct TranscriberrrUIApp: App {
    var body: some Scene {
        WindowGroup("Transcriberrr") {
            ContentView()
                .frame(minWidth: 780, minHeight: 560)
        }
        .windowResizability(.contentSize)
    }
}

@MainActor
final class AudioItem: ObservableObject, Identifiable {
    let id: UUID
    let url: URL
    @Published var duration: TimeInterval
    @Published var state: ItemState = .queued
    @Published var progress: Double = 0
    @Published var selected = true
    @Published var diarize = false
    /// Nil asks the diarizer to infer speakers. When set, it is passed per item.
    @Published var speakerCount: Int?
    @Published var outputURL: URL?
    @Published var resultSummary = ""

    init(url: URL, id: UUID = UUID(), duration: TimeInterval = 0, diarize: Bool = false, speakerCount: Int? = nil) {
        self.id = id
        self.url = url
        self.duration = duration
        self.diarize = diarize
        self.speakerCount = speakerCount
    }
}

enum ItemState: Equatable {
    case queued
    case processing
    case done
    case failed(String)

    var label: String {
        switch self {
        case .queued: "Queued"
        case .processing: "Transcribing"
        case .done: "Complete"
        case .failed(let message): message
        }
    }

    var stateName: String {
        switch self {
        case .queued: "queued"
        case .processing: "processing"
        case .done: "done"
        case .failed: "failed"
        }
    }
}

enum OutputDestination: String, CaseIterable, Identifiable {
    case folder
    case besideOriginal

    var id: String { rawValue }
    var title: String {
        switch self {
        case .folder: "Destination folder"
        case .besideOriginal: "Next to original"
        }
    }
}

private struct SavedItem: Codable {
    var id: UUID
    var path: String
    var duration: TimeInterval
    var state: String
    var error: String?
    var progress: Double
    var selected: Bool
    var diarize: Bool
    var speakerCount: Int?
    var outputPath: String?
    var resultSummary: String
}

private struct SavedQueue: Codable {
    var items: [SavedItem]
    var outputFolder: String
    var outputDestination: String
    var modelID: String
    var diarizationModel: String
    var device: String
    var format: String
    var timestamps: String
    var diarizeByDefault: Bool
    var engine: String?
}

/// The Rust CLI manifest is intentionally small and stable.  Keeping a typed
/// representation here lets the UI launch one process for an entire queue,
/// so the model is loaded once instead of once per track.
private struct ManifestItem: Codable {
    var input: String
    var diarize: Bool
    var speakers: Int?
    var output: String?
    var outputDir: String
    var outputLocation: String

    enum CodingKeys: String, CodingKey {
        case input
        case diarize
        case speakers
        case output
        case outputDir = "output_dir"
        case outputLocation = "output_location"
    }
}

private struct JobManifest: Codable {
    var items: [ManifestItem]
}

private struct ProgressEvent: Decodable {
    var input: String
    var status: String
    var output: String?
    var error: String?
}

/// Incrementally consumes the CLI's JSONL progress file. Reading the complete
/// file on every poll turns a 5,000-item queue into quadratic I/O and can make
/// the main actor visibly stutter.
@MainActor
private final class ProgressReader {
    private var offset: UInt64 = 0
    private var remainder = ""
    private(set) var events: [String: ProgressEvent] = [:]

    func read(from url: URL) -> [String: ProgressEvent] {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return events }
        defer { try? handle.close() }
        try? handle.seek(toOffset: offset)
        let data = handle.readDataToEndOfFile()
        offset = handle.offsetInFile
        guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return events }

        let combined = remainder + text
        let lines = combined.split(separator: "\n", omittingEmptySubsequences: false)
        remainder = combined.hasSuffix("\n") ? "" : String(lines.last ?? "")
        let completeLines = combined.hasSuffix("\n") ? lines : lines.dropLast()
        for line in completeLines {
            guard let event = try? JSONDecoder().decode(ProgressEvent.self, from: Data(line.utf8)) else { continue }
            let key = URL(fileURLWithPath: event.input).standardizedFileURL.resolvingSymlinksInPath().path
            events[key] = event
        }
        return events
    }
}

private final class URLCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var urls: [URL] = []

    func append(_ url: URL) {
        lock.lock()
        urls.append(url)
        lock.unlock()
    }

    func values() -> [URL] {
        lock.lock()
        defer { lock.unlock() }
        return urls
    }
}

@MainActor
final class AppState: ObservableObject {
    @Published var items: [AudioItem] = []
    @Published var outputFolder = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Transcripts")
    @Published var outputDestination: OutputDestination = .folder
    @Published var modelID = "Qwen/Qwen3-ASR-1.7B"
    /// `qwen` uses the Rust Metal pipeline; `apple-native` uses the bundled
    /// SpeechTranscriber helper on macOS 26 and newer.
    @Published var engine = "qwen"
    // Use the CLI alias so a fresh install downloads Sortformer into the shared model cache.
    // Users may replace this with a local ONNX file from the picker.
    @Published var diarizationModel = URL(fileURLWithPath: "sortformer")
    @Published var device = "metal"
    @Published var format = "json"
    @Published var timestamps = "words"
    @Published var diarize = false
    @Published var isRunning = false

    private let supportedExtensions: Set<String> = ["aac", "aif", "aiff", "caf", "flac", "m4a", "mp3", "ogg", "wav", "webm"]
    private var pendingSave: Task<Void, Never>?
    private var cachedModelStatusValue: String?

    private var queueFile: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Transcriberrr", isDirectory: true)
            .appendingPathComponent("queue.json")
    }

    init() {
        loadQueue()
    }

    var completedCount: Int { items.filter { if case .done = $0.state { return true }; return false }.count }
    var totalDuration: TimeInterval { items.reduce(0) { $0 + $1.duration } }

    var modelStatus: String {
        if let cachedModelStatusValue { return cachedModelStatusValue }
        let value: String
        switch engine {
        case "apple-native": value = "Apple SpeechTranscriber (system model; macOS 26+)"
        case "parakeet-v3":
            if modelID.hasPrefix("/") && FileManager.default.fileExists(atPath: modelID) {
                value = "Local model ready"
            } else {
                value = cachedModel(modelID) ? "Downloaded" : "Downloads on first use"
            }
        default:
            if modelID.hasPrefix("/") && FileManager.default.fileExists(atPath: modelID) {
                value = "Local model ready"
            } else {
                value = cachedModel(modelID) ? "Downloaded" : "Downloads on first use"
            }
        }
        cachedModelStatusValue = value
        return value
    }

    func selectEngine(_ value: String) {
        if value == "parakeet-v3" && modelID == "Qwen/Qwen3-ASR-1.7B" {
            modelID = "istupakov/parakeet-tdt-0.6b-v3-onnx"
        } else if value == "qwen" && modelID == "istupakov/parakeet-tdt-0.6b-v3-onnx" {
            modelID = "Qwen/Qwen3-ASR-1.7B"
        }
        if value == "apple-native" {
            // SpeechTranscriber has no speaker labels. Clear stale per-item
            // diarization flags when switching engines so a queued job cannot
            // fail later with an option the current engine cannot honor.
            diarize = false
            for item in items {
                item.diarize = false
                item.speakerCount = nil
            }
        }
        engine = value
        cachedModelStatusValue = nil
    }

    func invalidateModelStatus() {
        cachedModelStatusValue = nil
    }

    private func cachedModel(_ id: String) -> Bool {
        guard !id.isEmpty, !id.hasPrefix("/") else { return false }
        let root = ProcessInfo.processInfo.environment["TRANSCRIBERRR_MODEL_CACHE"]
            .map(URL.init(fileURLWithPath:))
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support/Transcriberrr/models/hub")
        let encoded = id.replacingOccurrences(of: "/", with: "--")
        let snapshots = root.appendingPathComponent("models--\(encoded)/snapshots", isDirectory: true)
        guard let revisions = try? FileManager.default.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return false }
        // Match the CLI's completeness check: a created snapshots directory alone can be left
        // behind by an interrupted download and must not be shown as ready.
        return revisions.contains { revision in
            guard (try? revision.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { return false }
            let config = revision.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: config.path) else { return false }
            let markers = [
                "model.safetensors",
                "model.safetensors.index.json",
                "model.onnx",
                "encoder-model.onnx",
                "diar_streaming_sortformer_4spk-v2.1.nemo"
            ]
            return markers.contains { FileManager.default.fileExists(atPath: revision.appendingPathComponent($0).path) }
        }
    }

    func add(urls: [URL]) {
        var discovered: [URL] = []
        for url in urls {
            if isDirectory(url) {
                if let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
                    for case let child as URL in enumerator where isSupported(child) {
                        discovered.append(child)
                    }
                }
            } else if isSupported(url) {
                discovered.append(url)
            }
        }
        // Keep one queue entry per path, including when a dropped folder contains
        // the same file as another dropped folder or an existing queue item.
        var known = Set(items.map { $0.url.standardizedFileURL })
        let newURLs = discovered
            .filter { known.insert($0.standardizedFileURL).inserted }
            .sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
        items.append(contentsOf: newURLs.map { AudioItem(url: $0, diarize: diarize) })
        saveQueue()
        loadDurations(for: newURLs)
    }

    func remove(_ item: AudioItem) {
        items.removeAll { $0.id == item.id }
        saveQueue()
    }

    func clearCompleted() {
        items.removeAll { if case .done = $0.state { return true }; return false }
        saveQueue()
    }

    func setSelection(for item: AudioItem, _ selected: Bool) {
        item.selected = selected
        saveQueue()
    }

    func setDiarization(for item: AudioItem, enabled: Bool) {
        item.diarize = enabled
        saveQueue()
    }

    func setSpeakerCount(for item: AudioItem, _ value: Int?) {
        item.speakerCount = value
        saveQueue()
    }

    /// AVAudioFile can spend noticeable time probing compressed media. Do that
    /// work off the main actor so dropping a folder with thousands of tracks
    /// keeps the queue responsive; publish all discovered durations as one batch.
    private func loadDurations(for urls: [URL]) {
        guard !urls.isEmpty else { return }
        Task.detached(priority: .utility) { [weak self] in
            let durations: [(String, TimeInterval)] = urls.compactMap { url in
                guard let file = try? AVAudioFile(forReading: url), file.processingFormat.sampleRate > 0 else { return nil }
                let duration = Double(file.length) / file.processingFormat.sampleRate
                return duration.isFinite ? (url.standardizedFileURL.path, duration) : nil
            }
            await MainActor.run {
                guard let self else { return }
                let values = Dictionary(uniqueKeysWithValues: durations)
                for item in self.items {
                    if let duration = values[item.url.standardizedFileURL.path] {
                        item.duration = duration
                    }
                }
                self.saveQueue()
            }
        }
    }

    func selectAll(_ visibleItems: [AudioItem], selected: Bool) {
        visibleItems.forEach { $0.selected = selected }
        saveQueue()
    }

    func chooseFolder(_ current: URL, completion: @escaping (URL) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = current
        if panel.runModal() == .OK, let url = panel.url { completion(url) }
    }

    func chooseDiarizationModel(_ current: URL, completion: @escaping (URL) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowsOtherFileTypes = true
        panel.directoryURL = current.deletingLastPathComponent()
        if panel.runModal() == .OK, let url = panel.url { completion(url) }
    }

    func transcribe() {
        guard !items.isEmpty, !isRunning else { return }
        isRunning = true
        Task { @MainActor in
            if outputDestination == .folder {
                try? FileManager.default.createDirectory(at: outputFolder, withIntermediateDirectories: true)
            }
            let pending = items.filter { $0.selected && $0.state != .done }
            pending.forEach {
                $0.state = .processing
                $0.progress = 0.05
                $0.resultSummary = "Queued…"
            }
            saveQueue()

            // Qwen and Parakeet keep their weights in process memory. Run the
            // whole queue through one manifest so a 5,000-track queue does not
            // repeatedly pay model startup and Metal pipeline compilation.
            if engine == "apple-native" {
                var reservedOutputs = Set<String>()
                for item in pending {
                    let output = uniqueOutput(expectedOutput(for: item), reserved: &reservedOutputs)
                    let result = await runCLI(for: item, outputOverride: output)
                    apply(result, to: item)
                    saveQueue()
                }
            } else if !pending.isEmpty {
                let result = await runBulkCLI(for: pending)
                applyBulk(result, to: pending)
                saveQueue()
            }
            isRunning = false
            saveQueue()
        }
    }

    private func apply(_ result: (status: Int32, message: String, outputURL: URL?), to item: AudioItem) {
        if result.status == 0 {
            item.progress = 1
            item.state = .done
            item.outputURL = result.outputURL
            item.resultSummary = result.outputURL.map { "Saved to \($0.path)" } ?? "Transcription complete"
        } else {
            item.state = .failed(result.message.isEmpty ? "Transcription failed" : result.message)
            item.resultSummary = item.state.label
        }
    }

    private func applyBulk(_ result: (status: Int32, message: String, progress: [String: ProgressEvent]), to items: [AudioItem]) {
        let fallback = result.message.trimmingCharacters(in: .whitespacesAndNewlines)
        for item in items {
            let key = progressKey(item.url.path)
            let event = result.progress[key]
                ?? result.progress[item.url.path]
            if let event, (event.status == "completed" || event.status == "skipped") {
                let output = event.output.map(URL.init(fileURLWithPath:))
                item.progress = 1
                item.state = .done
                item.outputURL = output
                item.resultSummary = output.map { "Saved to \($0.path)" } ?? "Transcription complete"
            } else if result.status == 0 {
                let output = expectedOutput(for: item)
                item.progress = 1
                item.state = .done
                item.outputURL = output
                item.resultSummary = "Saved to \(output.path)"
            } else {
                let message = event?.error ?? (fallback.isEmpty ? "Transcription failed" : fallback)
                item.state = .failed(message)
                item.resultSummary = message
            }
        }
    }

    private func expectedOutput(for item: AudioItem) -> URL {
        let destination = outputDestination == .folder ? outputFolder : item.url.deletingLastPathComponent()
        return destination
            .appendingPathComponent(item.url.deletingPathExtension().lastPathComponent)
            .appendingPathExtension(format)
    }

    private func uniqueOutput(_ candidate: URL, reserved: inout Set<String>) -> URL {
        let key = candidate.standardizedFileURL.path.lowercased()
        if reserved.insert(key).inserted { return candidate }
        let stem = candidate.deletingPathExtension().lastPathComponent
        let ext = candidate.pathExtension
        var suffix = 2
        while true {
            let numbered = candidate.deletingLastPathComponent()
                .appendingPathComponent("\(stem)-\(suffix)")
                .appendingPathExtension(ext)
            if reserved.insert(numbered.standardizedFileURL.path.lowercased()).inserted {
                return numbered
            }
            suffix += 1
        }
    }

    private func runCLI(for item: AudioItem, outputOverride: URL? = nil) async -> (status: Int32, message: String, outputURL: URL?) {
        let process = Process()
        let isNative = engine == "apple-native"
        if isNative && item.diarize {
            return (1, "Apple SpeechTranscriber does not provide diarization; choose Qwen3 or Parakeet v3 for speaker labels.", nil)
        }
        let executableURL = isNative ? nativeURL : cliURL
        guard let executableURL else {
            let name = isNative ? "transcriberrr-native" : "transcriberrr"
            return (1, "The \(name) executable is not bundled. Rebuild the app with scripts/build-macos-app.sh.", nil)
        }
        process.executableURL = executableURL
        let destination = outputDestination == .folder ? outputFolder : item.url.deletingLastPathComponent()
        let outputExtension = format
        let output = outputOverride
            ?? destination.appendingPathComponent(item.url.deletingPathExtension().lastPathComponent).appendingPathExtension(outputExtension)
        var arguments: [String]
        if isNative {
            arguments = [item.url.path, "--output", output.path, "--format", format, "--timestamps", timestamps]
            if item.diarize {
                arguments.append("--diarize")
                if let speakerCount = item.speakerCount { arguments += ["--speakers", String(speakerCount)] }
            }
        } else {
            let backend = engine == "parakeet-v3" ? "parakeet-v3" : "qwen3"
            arguments = [item.url.path, "--output-dir", destination.path, "--engine", backend, "--model", modelID, "--device", device, "--format", format, "--timestamps", timestamps, "--overwrite"]
            if item.diarize {
                arguments += ["--diarize", "--diarization-model", diarizationModel.path]
                if let speakerCount = item.speakerCount {
                    arguments += ["--speakers", String(speakerCount)]
                }
            }
        }
        process.arguments = arguments
        let errorPipe = Pipe()
        process.standardError = errorPipe
        return await withCheckedContinuation { continuation in
            process.terminationHandler = { process in
                let data = errorPipe.fileHandleForReading.readDataToEndOfFile()
                let message = String(data: data, encoding: .utf8) ?? ""
                continuation.resume(returning: (process.terminationStatus, message, output))
            }
            do { try process.run() } catch {
                process.terminationHandler = nil
                continuation.resume(returning: (1, error.localizedDescription, nil))
            }
        }
    }

    private func runBulkCLI(for items: [AudioItem]) async -> (status: Int32, message: String, progress: [String: ProgressEvent]) {
        guard let executableURL = cliURL else {
            return (1, "The transcriberrr executable is not bundled. Rebuild the app with scripts/build-macos-app.sh.", [:])
        }
        let temporaryRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("Transcriberrr", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let manifestURL = temporaryRoot.appendingPathComponent("jobs.json")
        let progressURL = temporaryRoot.appendingPathComponent("progress.jsonl")
        do {
            try FileManager.default.createDirectory(at: temporaryRoot, withIntermediateDirectories: true)
            let destination = outputDestination == .folder ? outputFolder : items[0].url.deletingLastPathComponent()
            let location = outputDestination == .folder ? "folder" : "next-to-source"
            // Explicit output paths keep duplicate basenames from different source folders from
            // overwriting each other when a large queue is written to one destination folder.
            var reserved = Set<String>()
            let manifestItems = items.map { item -> ManifestItem in
                let output = uniqueOutput(expectedOutput(for: item), reserved: &reserved)
                return ManifestItem(input: item.url.standardizedFileURL.path, diarize: item.diarize,
                                    speakers: item.speakerCount, output: output.path,
                                    outputDir: destination.path, outputLocation: location)
            }
            let manifest = JobManifest(items: manifestItems)
            let data = try JSONEncoder().encode(manifest)
            try data.write(to: manifestURL, options: .atomic)
            var arguments = ["--manifest", manifestURL.path, "--output-dir", destination.path,
                             "--engine", engine == "parakeet-v3" ? "parakeet-v3" : "qwen3",
                             "--model", modelID, "--device", device, "--format", format,
                             "--timestamps", timestamps, "--progress-file", progressURL.path, "--overwrite"]
            if items.contains(where: { $0.diarize }) {
                arguments += ["--diarization-model", diarizationModel.path]
            }
            let process = Process()
            process.executableURL = executableURL
            process.arguments = arguments
            let errorPipe = Pipe()
            process.standardError = errorPipe
            let progressReader = ProgressReader()
            let progressPoller = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(500))
                    guard !Task.isCancelled else { break }
                    let events = progressReader.read(from: progressURL)
                    for item in items {
                        let key = self?.progressKey(item.url.path) ?? item.url.path
                        if let event = events[key] {
                            if event.status == "running" {
                                item.progress = max(item.progress, 0.1)
                                item.resultSummary = "Transcribing…"
                            } else if event.status == "completed" {
                                item.progress = 0.95
                                item.resultSummary = "Transcription complete; saving…"
                            }
                        }
                    }
                }
            }
            let result = await withCheckedContinuation { continuation in
                process.terminationHandler = { process in
                    let data = errorPipe.fileHandleForReading.readDataToEndOfFile()
                    continuation.resume(returning: (process.terminationStatus, String(data: data, encoding: .utf8) ?? ""))
                }
                do { try process.run() } catch {
                    process.terminationHandler = nil
                    continuation.resume(returning: (1, error.localizedDescription))
                }
            }
            progressPoller.cancel()
            await progressPoller.value
            let events = progressReader.read(from: progressURL)
            try? FileManager.default.removeItem(at: temporaryRoot)
            return (result.0, result.1, events)
        } catch {
            try? FileManager.default.removeItem(at: temporaryRoot)
            return (1, error.localizedDescription, [:])
        }
    }

    private func progressKey(_ path: String) -> String {
        URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path
    }

    private func loadQueue() {
        guard let data = try? Data(contentsOf: queueFile), let saved = try? JSONDecoder().decode(SavedQueue.self, from: data) else { return }
        outputFolder = URL(fileURLWithPath: saved.outputFolder)
        outputDestination = OutputDestination(rawValue: saved.outputDestination) ?? .folder
        modelID = saved.modelID
        engine = saved.engine ?? "qwen"
        let savedDiarizationModel = URL(fileURLWithPath: saved.diarizationModel)
        diarizationModel = (!FileManager.default.fileExists(atPath: savedDiarizationModel.path)
                            && savedDiarizationModel.pathExtension.lowercased() == "onnx")
            ? URL(fileURLWithPath: "sortformer")
            : savedDiarizationModel
        device = saved.device
        format = saved.format
        timestamps = saved.timestamps
        diarize = saved.diarizeByDefault
        items = saved.items.compactMap { value in
            let url = URL(fileURLWithPath: value.path)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            let item = AudioItem(url: url, id: value.id, duration: value.duration, diarize: value.diarize, speakerCount: value.speakerCount)
            item.state = value.state == "done" ? .done : value.state == "processing" ? .queued : value.state == "failed" ? .failed(value.error ?? "Transcription failed") : .queued
            item.progress = value.progress
            item.selected = value.selected
            item.outputURL = value.outputPath.map(URL.init(fileURLWithPath:))
            item.resultSummary = value.resultSummary
            return item
        }
        if engine == "apple-native" {
            diarize = false
            items.forEach {
                $0.diarize = false
                $0.speakerCount = nil
            }
        }
    }

    func saveQueue() {
        // Coalesce bursts from selecting thousands of rows. The last state is
        // durable within a short interval while avoiding O(n²) JSON rewrites.
        pendingSave?.cancel()
        pendingSave = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(150))
            guard !Task.isCancelled else { return }
            self?.writeQueue()
        }
    }

    private func writeQueue() {
        let saved = SavedQueue(
            items: items.map { item in
                let error: String? = if case .failed(let message) = item.state { message } else { nil }
                return SavedItem(id: item.id, path: item.url.path, duration: item.duration, state: item.state.stateName, error: error, progress: item.progress, selected: item.selected, diarize: item.diarize, speakerCount: item.speakerCount, outputPath: item.outputURL?.path, resultSummary: item.resultSummary)
            },
            outputFolder: outputFolder.path,
            outputDestination: outputDestination.rawValue,
            modelID: modelID,
            diarizationModel: diarizationModel.path,
            device: device,
            format: format,
            timestamps: timestamps,
            diarizeByDefault: diarize,
            engine: engine
        )
        guard let data = try? JSONEncoder().encode(saved) else { return }
        try? FileManager.default.createDirectory(at: queueFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: queueFile, options: .atomic)
    }

    private var cliURL: URL? {
        let environment = ProcessInfo.processInfo.environment
        if let override = environment["TRANSCRIBERRR_CLI"], !override.isEmpty {
            let url = URL(fileURLWithPath: override)
            if FileManager.default.isExecutableFile(atPath: url.path) { return url }
        }
        if let bundled = Bundle.main.url(forResource: "transcriberrr", withExtension: nil),
           FileManager.default.isExecutableFile(atPath: bundled.path) {
            return bundled
        }
        // This fallback makes `swift run` and local development convenient while
        // keeping Finder-launched apps independent of the user's shell PATH.
        let repositoryCLI = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Sources/TranscriberrrUI
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // ui
            .deletingLastPathComponent() // repository root
            .appendingPathComponent("target/release/transcriberrr")
        return FileManager.default.isExecutableFile(atPath: repositoryCLI.path) ? repositoryCLI : nil
    }

    private var nativeURL: URL? {
        let environment = ProcessInfo.processInfo.environment
        if let override = environment["TRANSCRIBERRR_NATIVE"], !override.isEmpty {
            let url = URL(fileURLWithPath: override)
            if FileManager.default.isExecutableFile(atPath: url.path) { return url }
        }
        if let bundled = Bundle.main.url(forResource: "transcriberrr-native", withExtension: nil),
           FileManager.default.isExecutableFile(atPath: bundled.path) { return bundled }
        let repositoryHelper = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("ui/.build/out/Products/Release/transcriberrr-native")
        return FileManager.default.isExecutableFile(atPath: repositoryHelper.path) ? repositoryHelper : nil
    }

    private func isSupported(_ url: URL) -> Bool {
        supportedExtensions.contains(url.pathExtension.lowercased())
    }

    private func isDirectory(_ url: URL) -> Bool {
        if url.hasDirectoryPath { return true }
        return (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }
}

struct ContentView: View {
    @StateObject private var state = AppState()
    @State private var dropTargeted = false
    @State private var queueFilter = ""
    @State private var compiledFilter: NSRegularExpression?
    @State private var filterErrorMessage: String?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            dropZone
            if state.items.isEmpty { emptyState } else { queue }
            Divider()
            footer
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
            loadDroppedItems(providers)
            return true
        }
        .onChange(of: state.outputDestination) { _, _ in state.saveQueue() }
        .onChange(of: state.outputFolder) { _, _ in state.saveQueue() }
        .onChange(of: state.engine) { _, value in state.selectEngine(value); state.saveQueue() }
        .onChange(of: state.modelID) { _, _ in state.invalidateModelStatus(); state.saveQueue() }
        .onChange(of: state.diarizationModel) { _, _ in state.saveQueue() }
        .onChange(of: state.device) { _, _ in state.saveQueue() }
        .onChange(of: state.format) { _, _ in state.saveQueue() }
        .onChange(of: state.timestamps) { _, _ in state.saveQueue() }
        .onAppear { updateFilter(queueFilter) }
        .onChange(of: queueFilter) { _, value in updateFilter(value) }
    }

    private var header: some View {
        HStack {
            Image(systemName: "waveform.and.mic")
                .font(.system(size: 24, weight: .semibold))
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 2) {
                Text("Transcriberrr").font(.title2.weight(.semibold))
                Text("Private, offline transcription on Apple Silicon").foregroundStyle(.secondary)
            }
            Spacer()
            Button("Choose files…") { chooseFiles() }
                .keyboardShortcut("o")
                .disabled(state.isRunning)
        }
        .padding(20)
    }

    private var dropZone: some View {
        VStack(spacing: 8) {
            Image(systemName: dropTargeted ? "arrow.down.circle.fill" : "arrow.down.doc")
                .font(.system(size: 32))
                .foregroundStyle(dropTargeted ? Color.accentColor : Color.secondary)
            Text(dropTargeted ? "Release to add audio" : "Drop audio files or folders here")
                .font(.headline)
            Text("WAV, AIFF, CAF, M4A, MP3, FLAC, AAC, OGG, or WebM")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 116)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(dropTargeted ? Color.accentColor : .secondary.opacity(0.25), style: StrokeStyle(lineWidth: 2, dash: [8])))
        .padding(.horizontal, 20)
        .padding(.top, 18)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(dropTargeted ? "Release to add audio" : "Drop audio files or folders here")
        .accessibilityHint("You can also use the Choose files button to select audio.")
    }

    private var emptyState: some View {
        ContentUnavailableView("No audio queued", systemImage: "music.note.list", description: Text("Add a file or folder to begin."))
            .frame(maxHeight: .infinity)
    }

    private var queue: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Queue").font(.headline)
                Text("\(state.items.count) file\(state.items.count == 1 ? "" : "s")").foregroundStyle(.secondary)
                Text("• \(visibleItems.count) visible").foregroundStyle(.secondary)
                Spacer()
                TextField("Regex filter", text: $queueFilter)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 180)
                Button("Check all") { state.selectAll(visibleItems, selected: true) }
                    .buttonStyle(.borderless)
                Button("Check none") { state.selectAll(visibleItems, selected: false) }
                    .buttonStyle(.borderless)
                Button("Clear completed") { state.clearCompleted() }
                    .buttonStyle(.borderless)
                    .disabled(state.completedCount == 0 || state.isRunning)
            }
            .padding(.horizontal, 20)
            .padding(.top, 18)
            if let regexError = filterError {
                Text("Invalid regex: \(regexError)").font(.caption).foregroundStyle(.red).padding(.horizontal, 20).padding(.top, 4)
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(visibleItems) { item in queueRow(item) }
                }
                .padding(.horizontal, 20)
            }
        }
        .frame(maxHeight: .infinity)
    }

    private var visibleItems: [AudioItem] {
        guard let regex = compiledFilter else { return state.items }
        return state.items.filter { item in
            let value = "\(item.url.lastPathComponent) \(item.url.path)"
            return regex.firstMatch(in: value, range: NSRange(value.startIndex..., in: value)) != nil
        }
    }

    private var filterError: String? {
        filterErrorMessage
    }

    private func updateFilter(_ value: String) {
        guard !value.isEmpty else {
            compiledFilter = nil
            filterErrorMessage = nil
            return
        }
        do {
            compiledFilter = try NSRegularExpression(pattern: value, options: [.caseInsensitive])
            filterErrorMessage = nil
        } catch {
            compiledFilter = try! NSRegularExpression(pattern: "(?!)")
            filterErrorMessage = error.localizedDescription
        }
    }

    @ViewBuilder
    private func queueRow(_ item: AudioItem) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Button {
                    state.setSelection(for: item, !item.selected)
                } label: {
                    Image(systemName: item.selected ? "checkmark.square.fill" : "square")
                }
                    .buttonStyle(.borderless)
                    .foregroundStyle(item.selected ? .primary : .secondary)
                    .accessibilityLabel("Include \(item.url.lastPathComponent) in transcription")
                    .accessibilityValue(item.selected ? "On" : "Off")
                    .disabled(state.isRunning)
                Image(systemName: item.diarize ? "person.2.wave.2" : "waveform")
                    .foregroundStyle(.tint).frame(width: 22)
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.url.lastPathComponent).lineLimit(1)
                    Text("\(formatDuration(item.duration))  •  \(item.url.deletingLastPathComponent().path)")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    if !item.resultSummary.isEmpty { Text(item.resultSummary).font(.caption2).foregroundStyle(.secondary).lineLimit(1) }
                }
                Spacer(minLength: 8)
                if item.state == .processing {
                    Group {
                        if state.engine == "apple-native" {
                            ProgressView()
                        } else {
                            ProgressView(value: item.progress)
                        }
                    }
                        .frame(width: 72)
                        .accessibilityLabel("Transcribing \(item.url.lastPathComponent)")
                }
                Text(item.state.label).font(.caption).foregroundStyle(color(for: item.state)).frame(minWidth: 82, alignment: .trailing).lineLimit(1)
                Button { state.remove(item) } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary) }
                    .buttonStyle(.borderless)
                    .accessibilityLabel("Remove \(item.url.lastPathComponent) from queue")
                    .disabled(state.isRunning)
            }
            HStack(spacing: 14) {
                Button {
                    state.setDiarization(for: item, enabled: !item.diarize)
                } label: {
                    Label("Diarize this track", systemImage: item.diarize ? "checkmark.square.fill" : "square")
                }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .foregroundStyle(item.diarize ? .primary : .secondary)
                    .accessibilityLabel("Diarize this track")
                    .accessibilityValue(item.diarize ? "On" : "Off")
                    .disabled(state.isRunning)
                if item.diarize && state.engine != "apple-native" {
                    Picker("Speakers", selection: Binding<Int>(get: { item.speakerCount ?? 0 }, set: { state.setSpeakerCount(for: item, $0 == 0 ? nil : $0) })) {
                        Text("Auto").tag(0)
                        ForEach(1...8, id: \.self) { count in Text("\(count)").tag(count) }
                    }
                    .pickerStyle(.menu).font(.caption).frame(width: 140)
                }
                Spacer()
            }
            Divider()
        }
        .padding(.vertical, 7)
    }

    private var footer: some View {
        VStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 16) {
                    setting("Engine") {
                        Picker("", selection: $state.engine) {
                            Text("Qwen3 ASR (Metal)").tag("qwen")
                            Text("Apple SpeechTranscriber").tag("apple-native")
                            Text("Parakeet v3").tag("parakeet-v3")
                        }
                        .labelsHidden().pickerStyle(.menu).frame(width: 190)
                    }
                    setting("Model") {
                        TextField("Hugging Face ID or local path", text: $state.modelID)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 300)
                    }
                    Text(state.modelStatus).font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                }
                HStack(spacing: 16) {
                    setting("Output") {
                        Picker("", selection: $state.outputDestination) {
                            ForEach(OutputDestination.allCases) { destination in Text(destination.title).tag(destination) }
                        }
                        .labelsHidden().pickerStyle(.menu).frame(width: 170)
                    }
                    if state.outputDestination == .folder {
                        Button(state.outputFolder.path) { state.chooseFolder(state.outputFolder) { state.outputFolder = $0; state.saveQueue() } }
                            .buttonStyle(.borderless).lineLimit(1).frame(maxWidth: 300, alignment: .leading)
                    }
                    Spacer()
                }
                HStack(spacing: 16) {
                    setting("Acceleration") {
                        Picker("", selection: $state.device) { Text("Metal GPU").tag("metal"); Text("CPU").tag("cpu") }
                            .labelsHidden().pickerStyle(.menu).frame(width: 130)
                    }
                    setting("Format") {
                        Picker("", selection: Binding(
                            get: { state.format },
                            set: { format in
                                state.format = format
                                if format == "txt" { state.timestamps = "none" }
                            }
                        )) {
                            Text("JSON").tag("json")
                            Text("Text").tag("txt")
                            Text("SRT").tag("srt")
                            Text("VTT").tag("vtt")
                        }
                        .labelsHidden().pickerStyle(.menu).frame(width: 100)
                    }
                    Toggle("Include timestamps", isOn: Binding(
                        get: { state.timestamps != "none" },
                        set: { enabled in
                            if enabled {
                                if state.format == "txt" { state.format = "json" }
                                state.timestamps = "words"
                            } else {
                                state.timestamps = "none"
                            }
                        }
                    ))
                    .toggleStyle(.checkbox)
                    if state.timestamps != "none" {
                        setting("Timestamp level") {
                            Picker("", selection: $state.timestamps) {
                                Text("Word level").tag("words")
                                Text("Sentence level").tag("sentences")
                            }
                            .labelsHidden().pickerStyle(.menu).frame(width: 150)
                        }
                    }
                    Button {
                        state.diarize.toggle()
                        state.saveQueue()
                    } label: {
                        Label("Diarization default", systemImage: state.diarize ? "checkmark.square.fill" : "square")
                    }
                        .buttonStyle(.borderless)
                        .accessibilityValue(state.diarize ? "On" : "Off")
                    Spacer()
                }
            }
            .disabled(state.isRunning)
            if state.diarize {
                HStack {
                    Text("Diarization model").foregroundStyle(.secondary)
                    Text(state.diarizationModel.path).lineLimit(1).foregroundStyle(.secondary)
                    Button("Choose…") { state.chooseDiarizationModel(state.diarizationModel) { state.diarizationModel = $0; state.saveQueue() } }
                        .disabled(state.isRunning)
                }
                .font(.caption)
            }
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(state.completedCount) of \(state.items.count) complete")
                    Text("\(formatDuration(state.totalDuration)) queued")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if state.isRunning { ProgressView(value: Double(state.completedCount), total: Double(max(state.items.count, 1))).frame(width: 130) }
                Button(state.isRunning ? "Transcribing…" : "Transcribe") { state.transcribe() }
                    .buttonStyle(.borderedProminent)
                    .disabled(state.items.isEmpty || state.isRunning)
            }
            if state.timestamps != "none" {
                Text("JSON stores each word with start and end seconds; SRT and VTT write one cue per word.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(20)
    }

    private func setting<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 3) { Text(title).font(.caption).foregroundStyle(.secondary); content() }
    }

    private func loadDroppedItems(_ providers: [NSItemProvider]) {
        guard !state.isRunning else { return }
        guard !providers.isEmpty else { return }
        let group = DispatchGroup()
        let collector = URLCollector()
        for provider in providers {
            group.enter()
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                let url: URL?
                if let data = item as? Data { url = URL(dataRepresentation: data, relativeTo: nil) }
                else if let value = item as? URL { url = value }
                else if let value = item as? NSURL { url = value as URL }
                else { url = nil }
                if let url {
                    collector.append(url)
                }
                group.leave()
            }
        }
        // A single add call deduplicates the whole drop, schedules one duration
        // probe, and coalesces persistence for folders containing thousands of files.
        DispatchQueue.global(qos: .userInitiated).async {
            group.wait()
            let urls = collector.values()
            DispatchQueue.main.async {
                guard !self.state.isRunning else { return }
                self.state.add(urls: urls)
            }
        }
    }

    private func chooseFiles() {
        guard !state.isRunning else { return }
        let panel = NSOpenPanel()
        // Filter supported extensions after selection so folders and formats such
        // as FLAC, OGG, AAC, and WebM can be selected consistently.
        panel.allowedContentTypes = []
        panel.allowsOtherFileTypes = true
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { state.add(urls: panel.urls) }
    }

    private func color(for itemState: ItemState) -> Color {
        switch itemState { case .done: .green; case .failed: .red; default: .secondary }
    }
}

private func formatDuration(_ duration: TimeInterval) -> String {
    guard duration.isFinite, duration > 0 else { return "Unknown duration" }
    let total = Int(duration.rounded())
    return total >= 3600 ? String(format: "%d:%02d:%02d", total / 3600, (total % 3600) / 60, total % 60) : String(format: "%d:%02d", total / 60, total % 60)
}
