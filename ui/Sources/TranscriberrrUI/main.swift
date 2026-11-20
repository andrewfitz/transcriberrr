import AVFoundation
import AppKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct TranscriberrrUIApp: App {
    // One queue per app: it owns queue.json and the running CLI, and every window shows it,
    // so windows cannot overwrite each other's queue file. A WindowGroup (not `Window`)
    // keeps the app running when its window closes, so a run continues and the Dock
    // reopens the window; as the only scene, `Window` quits the app on close.
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup("Transcriberrr") {
            QueueWindow(state: state)
                .frame(minWidth: 660, minHeight: 460)
        }
        .defaultSize(width: 1050, height: 720)
        .windowResizability(.contentSize)
        .commands {
            // A second window would only mirror the same queue. The group must not be empty
            // (that removes the whole File menu, including Close / Cmd-W).
            CommandGroup(replacing: .newItem) {
                Button("Add Files…") { state.chooseFiles() }
                    .keyboardShortcut("o")
                    .disabled(state.isRunning)
            }
        }
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
    /// One speaker's track of a multi-track recording (a Zoom participant file, say): its
    /// session's tracks are transcribed together with bleed muted, then merged.
    @Published var isoTrack = false
    var isoOverride: Bool?
    /// The speaker the CLI named this track after.
    @Published var speakerName: String?
    /// The session the CLI grouped this track into.
    @Published var sessionLabel: String?
    var zoomMeeting: String?
    var zoomRecording = false
    var skipReason: String?
    var coveredBy: [String]?
    @Published var outputURL: URL?
    var modelOutputs: [ModelOutput] = []
    @Published var resultSummary = ""
    @Published var elapsedSeconds: TimeInterval = 0
    @Published var estimatedRemainingSeconds: TimeInterval?

    init(url: URL, id: UUID = UUID(), duration: TimeInterval = 0, diarize: Bool = false, speakerCount: Int? = nil, isoTrack: Bool = false) {
        self.id = id
        self.url = url
        self.duration = duration
        self.diarize = diarize && !isoTrack
        self.speakerCount = diarize && !isoTrack ? speakerCount : nil
        self.isoTrack = isoTrack
    }
}

enum ItemState: Equatable {
    case queued
    case processing
    case done
    case canceled
    case failed(String)

    var label: String {
        switch self {
        case .queued: "Queued"
        case .processing: "Transcribing"
        case .done: "Complete"
        case .canceled: "Canceled"
        case .failed(let message): message
        }
    }

    var stateName: String {
        switch self {
        case .queued: "queued"
        case .processing: "processing"
        case .done: "done"
        case .canceled: "canceled"
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
    /// Optional so queue files from older versions still decode.
    var isoTrack: Bool?
    var isoOverride: Bool?
    var speakerName: String?
    var sessionLabel: String?
    var zoomMeeting: String?
    var zoomRecording: Bool?
    var skipReason: String?
    var coveredBy: [String]?
    var modelOutputs: [String: String]?
    var modelOutputHistory: [ModelOutput]?
}

struct ModelOutput: Codable, Identifiable, Equatable {
    var tag: String
    var path: String
    var id: String { path }
}

struct TranscriptionPass: Codable, Identifiable, Equatable {
    var id = UUID()
    var engine: String
    var model = ""

    var usesApple: Bool { engine == "apple-native" }
    var cliEngine: String { engine == "qwen" ? "qwen3" : engine }

    var defaultModel: String {
        switch engine {
        case "parakeet-v3": "eschmidbauer/parakeet-unified-en-0.6b-onnx"
        case "apple-native": "Apple SpeechTranscriber"
        default: "Qwen/Qwen3-ASR-1.7B"
        }
    }

    var modelArgument: String {
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return model.isEmpty || usesApple ? defaultModel : model
    }

    var arguments: [String] { ["--engine", cliEngine, "--model", modelArgument] }
}

/// A session's merged transcript, reported by the CLI.
struct MergedSession: Codable, Identifiable, Equatable {
    /// The merged transcript's path.
    var id: String
    var label: String
    var state: String
    var message: String
    var meetingID: String?
    var modelTag: String?
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
    /// Optional so queue files from older versions still decode.
    var speakerTurns: Bool?
    var mergedSessions: [MergedSession]?
    var zoomMeetings: [ZoomMeeting]?
    var removedZoomInputs: Set<String>?
    var additionalPasses: [TranscriptionPass]?
    var appendModelID: Bool?
    var passOutcomes: [AppState.RunTarget: [AppState.PassOutcome]]?
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
    var isoTrack: Bool?

    enum CodingKeys: String, CodingKey {
        case input
        case diarize
        case speakers
        case output
        case outputDir = "output_dir"
        case outputLocation = "output_location"
        case isoTrack = "iso_track"
    }
}

private struct JobManifest: Codable {
    var items: [ManifestItem]
    var discoverZoom: Bool? = nil
    var excludedInputs: [String]? = nil

    enum CodingKeys: String, CodingKey {
        case items
        case discoverZoom = "discover_zoom"
        case excludedInputs = "excluded_inputs"
    }
}

struct ZoomMeeting: Codable, Identifiable {
    var id: String
    var label: String
    var output: String
    var done: Bool
    var participants: Int
}

private struct QueuePlan: Decodable {
    struct File: Decodable {
        var path: String
        var meeting: String?
        var speaker: String?
        var output: String?
        var iso: Bool?
        var diarize: Bool?
        var done: Bool?
        var zoomRecording: Bool?
        var skip: String?
        var coveredBy: [String]?
        var mergeOutput: String?
        var group: String?
        var error: String?

        enum CodingKeys: String, CodingKey {
            case path, meeting, speaker, output, iso, diarize, done, skip, error, group
            case zoomRecording = "zoom_recording"
            case coveredBy = "covered_by"
            case mergeOutput = "merge_output"
        }
    }
    var files: [File]
    var meetings: [ZoomMeeting]
}

private struct ProgressEvent: Decodable {
    var input: String
    var status: String
    var output: String?
    var error: String?
    var progress: Double?
    var phase: String?
    var elapsedSeconds: TimeInterval?
    /// Set on a finished track that needs a look, e.g. an empty transcript.
    var warning: String?
    /// Speaker tracks: the session and the speaker the CLI named.
    var group: String?
    var speaker: String?
    /// `merge` for a session's merged transcript; `input` is its path.
    var kind: String?

    enum CodingKeys: String, CodingKey {
        case input, status, output, error, progress, phase, warning, group, speaker, kind
        case elapsedSeconds = "elapsed_seconds"
    }
}

/// Incrementally consumes the CLI's JSONL progress file. Reading the complete
/// file on every poll turns a 5,000-item queue into quadratic I/O and can make
/// the main actor visibly stutter.
@MainActor
private final class ProgressReader {
    private var offset: UInt64 = 0
    // Keep undecoded bytes rather than a String. FileHandle reads can split a
    // multibyte UTF-8 character, and decoding each chunk independently would
    // silently discard that partial event.
    private var remainder = Data()
    private(set) var events: [String: ProgressEvent] = [:]

    func read(from url: URL) -> [String: ProgressEvent] {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return events }
        defer { try? handle.close() }
        try? handle.seek(toOffset: offset)
        let data = handle.readDataToEndOfFile()
        offset = handle.offsetInFile
        guard !data.isEmpty else { return events }

        let combined = remainder + data
        var lineStart = combined.startIndex
        for index in combined.indices where combined[index] == 0x0A {
            let line = combined[lineStart..<index]
            lineStart = combined.index(after: index)
            guard let event = try? JSONDecoder().decode(ProgressEvent.self, from: line) else { continue }
            events[canonicalPath(event.input)] = event
        }
        remainder = Data(combined[lineStart...])
        return events
    }
}

/// Drains a child process pipe while it runs. A pipe holds only ~64 KB; the CLI
/// writes a line per finished track, so reading only after exit deadlocks a large
/// queue (the CLI blocks on write and never exits). Keeps a bounded tail for errors.
private final class PipeCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()
    private let limit: Int?
    private let onData: (@Sendable (Data) -> Void)?
    let pipe = Pipe()

    init(limit: Int? = 256 * 1024, onData: (@Sendable (Data) -> Void)? = nil) {
        self.limit = limit
        self.onData = onData
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            // Read under the lock, so `finish`'s read cannot overtake a chunk read here and
            // append the end of the output before it.
            guard let self else { return }
            self.lock.lock()
            defer { self.lock.unlock() }
            let chunk = handle.availableData
            // Empty data means EOF; without clearing, the handler is re-invoked in a busy loop.
            if chunk.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            self.append(chunk)
            onData?(chunk)
        }
    }

    /// With the lock held.
    private func append(_ chunk: Data) {
        data.append(chunk)
        if let limit, data.count > limit { data.removeFirst(data.count - limit) }
    }

    /// Call from the termination handler (or after a failed launch): stops the reader and
    /// collects anything left.
    func finish() -> String {
        pipe.fileHandleForReading.readabilityHandler = nil
        // A process that never started leaves this the only write end, and reading to EOF
        // would wait forever; after a launch, closing this copy changes nothing.
        try? pipe.fileHandleForWriting.close()
        lock.lock()
        defer { lock.unlock() }
        let rest = pipe.fileHandleForReading.readDataToEndOfFile()
        if !rest.isEmpty {
            append(rest)
            onData?(rest)
        }
        return String(decoding: data, as: UTF8.self)
    }
}

/// Matches Rust's `fs::canonicalize` (realpath). `resolvingSymlinksInPath` drops
/// `/private`, so files under /tmp or /var never matched their progress events.
private func canonicalPath(_ path: String) -> String {
    guard let resolved = realpath(path, nil) else {
        return URL(fileURLWithPath: path).standardizedFileURL.path
    }
    defer { free(resolved) }
    return String(cString: resolved)
}

extension AppState {
    enum RunTarget: Hashable, Codable {
        case recording(UUID)
        case meeting(String)
    }

    struct PassOutcome: Codable, Equatable, Identifiable {
        enum Phase: String, Codable {
            case waiting, running, completed, failed, canceled
        }

        let id: String
        let label: String
        let modelTag: String?
        var phase: Phase
        var message: String?
        var output: URL?
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
    @Published var additionalPasses: [TranscriptionPass] = []
    @Published var appendModelID = false
    @Published private(set) var activePassLabel: String?
    @Published private(set) var lastRunSummary: String?
    // Use an absolute sentinel for the CLI alias. URL(fileURLWithPath:) resolves
    // relative paths against the current working directory, which would turn
    // `sortformer` into a nonexistent local model path and skip auto-download.
    // Users may replace this with a local ONNX file from the picker.
    @Published var diarizationModel = URL(fileURLWithPath: "/sortformer")
    @Published var device = "metal"
    @Published var format = "json"
    @Published var timestamps = "words"
    @Published var diarize = false
    /// Text output only: label each speaker turn instead of writing the exact transcript.
    @Published var speakerTurns = false
    @Published var isRunning = false
    @Published var showModelActivity = false
    @Published private(set) var downloadingLibraryModel: String?
    @Published var modelActivityTitle = "Preparing models"
    @Published var modelActivityMessage = "Checking the local model cache…"
    @Published var modelActivityLog: [String] = []
    @Published var modelActivityProgress: Double?
    @Published var modelActivityCanDismiss = false
    @Published var isCancelling = false
    @Published private var mergeResults: [String: MergedSession] = [:]
    var mergedSessions: [MergedSession] { mergeResults.values.sorted { $0.id < $1.id } }
    @Published var zoomMeetings: [ZoomMeeting] = []
    private var removedZoomInputs: Set<String> = []
    @Published var planningError: String?
    @Published private(set) var addOperations = 0

    private var pendingSave: Task<Void, Never>?
    private var activeProcess: Process?
    private var queueStartedAt: Date?
    private var storedQueueElapsedSeconds: TimeInterval = 0
    @Published private var clock = Date()
    private var clockTask: Task<Void, Never>?

    private var queueFile: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Transcriberrr", isDirectory: true)
            .appendingPathComponent("queue.json")
    }

    init() {
        loadQueue()
        // saveQueue() is debounced, so flush on quit. Child processes also outlive the
        // app; stop the CLI rather than leave it holding the GPU with nobody watching.
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.pendingSave?.cancel()
                self.activeProcess?.terminate()
                self.writeQueue()
            }
        }
    }

    var selectedItems: [AudioItem] { items.filter(\.selected) }
    var supportsDiarization: Bool { engine != "apple-native" || additionalPasses.contains { !$0.usesApple } }
    var selectedCount: Int {
        let selected = selectedItems
        return selected.filter { $0.zoomMeeting == nil }.count + Set(selected.compactMap(\.zoomMeeting)).count
    }
    private var completedMeetingIDs: Set<String> {
        let incomplete = Set(items.compactMap { item in
            switch item.state {
            case .failed, .canceled: item.zoomMeeting
            default: nil
            }
        })
        return Set(zoomMeetings.filter { !incomplete.contains($0.id) && meetingResult($0)?.state == "done" }.map(\.id))
    }
    var completedItemCount: Int {
        let finished = completedMeetingIDs
        return items.filter { $0.zoomMeeting.map(finished.contains) ?? ($0.state == .done) }.count
    }
    var completedCount: Int {
        let selected = selectedItems
        let groups = Set(selected.compactMap(\.zoomMeeting))
        return selected.filter { $0.zoomMeeting == nil && $0.state == .done }.count
            + groups.intersection(completedMeetingIDs).count
    }
    var activeStatus: String {
        if isCancelling { return "Stopping safely…" }
        return activeRunItem?.resultSummary ?? activeMerge?.message ?? "Preparing…"
    }
    /// Items in the current run. Progress and ETA cover only these: counting tracks finished
    /// in earlier runs as done made the estimate start near zero.
    private var runItemIDs: Set<UUID> = []
    private var runScopeIDs: Set<UUID> = []
    private var runTargets: Set<RunTarget> = []
    private var passIndex = 0
    private var passCount = 1
    private var activeModelTag: String?
    private var passMeetingIDsByOutput: [String: String] = [:]
    @Published private var outcomes: [RunTarget: [PassOutcome]] = [:]
    @Published private(set) var activePassUsesApple = false

    private struct PassWork {
        var items: [AudioItem]
        var output: URL?
        var mergePath: String?
    }
    private var currentPassWork: [RunTarget: PassWork] = [:]

    func isInCurrentRun(_ target: RunTarget) -> Bool {
        isRunning && runTargets.contains(target)
    }

    func passOutcomes(for target: RunTarget) -> [PassOutcome] {
        outcomes[target] ?? []
    }

    var activeRecordingLabel: String? {
        guard isRunning else { return nil }
        if let item = activeRunItem {
            return item.sessionLabel ?? item.url.lastPathComponent
        }
        return activeMerge?.label
    }

    private var activeRunItem: AudioItem? {
        items.first { runItemIDs.contains($0.id) && $0.state == .processing }
    }

    private var activeMerge: MergedSession? {
        currentPassWork.values.lazy.compactMap(\.mergePath)
            .compactMap { self.mergeResults[$0] }.first { $0.state == "running" }
    }

    var queueProgress: Double {
        let tracked = isRunning
            ? items.filter { runItemIDs.contains($0.id) }
            : selectedItems
        let current = tracked.reduce(0.0) { total, item in
            if item.state == .done { return total + 1 }
            return total + min(max(item.progress, 0), 1)
        } / Double(max(1, tracked.count))
        let value = activeMerge != nil ? min(current, 0.99) : current
        return isRunning ? (Double(passIndex) + value) / Double(passCount) : current
    }

    var queueElapsedSeconds: TimeInterval {
        guard let queueStartedAt else { return storedQueueElapsedSeconds }
        return max(0, clock.timeIntervalSince(queueStartedAt))
    }

    var queueEstimatedRemainingSeconds: TimeInterval? {
        let progress = queueProgress
        let elapsed = queueElapsedSeconds
        guard isRunning, progress > 0.01, elapsed > 0 else { return nil }
        return max(0, elapsed * (1 / progress - 1))
    }

    /// Model families, so switching engines swaps any model of the other family (sizes,
    /// aliases), not just the two defaults.
    /// Mirrors the CLI (main.rs): Qwen3-ASR Hub IDs and their aliases, not local paths.
    private func isQwenModel(_ id: String) -> Bool {
        id.hasPrefix("Qwen/Qwen3-ASR") || ["qwen3-asr-1.7b", "qwen-1.7b", "qwen3-asr-0.6b", "qwen-0.6b"].contains(id)
    }
    private func isParakeetModel(_ id: String) -> Bool { !isLocalModelPath(id) && id.lowercased().contains("parakeet") }

    func add(urls: [URL]) {
        guard !urls.isEmpty else { return }
        let addFormat = format
        let manifest = JobManifest(items: urls.map {
            ManifestItem(input: $0.standardizedFileURL.path, diarize: diarize, outputDir: outputFolder.path,
                         outputLocation: cliOutputLocation)
        })
        addOperations += 1
        Task { @MainActor in
            defer { addOperations -= 1 }
            do {
                var plan = try await preview(manifest, format: addFormat, inventoryOnly: true)
                let overrides = itemsByPath(items.filter { $0.isoOverride != nil })
                if plan.files.contains(where: { overrides[$0.path] != nil }), let defaults = manifest.items.first {
                    let files = plan.files.map { file in
                        var settings = defaults
                        settings.input = file.path
                        if let item = overrides[file.path] {
                            settings.diarize = item.diarize
                            settings.speakers = item.speakerCount
                            settings.isoTrack = item.isoOverride
                        }
                        return settings
                    }
                    plan = try await preview(JobManifest(items: files, discoverZoom: false),
                                             format: addFormat, inventoryOnly: true)
                }
                removedZoomInputs.subtract(plan.files.map(\.path))
                applyPlan(plan)
            } catch {
                planningError = error.localizedDescription
            }
        }
    }

    /// Read the full plan even for large libraries. stderr remains bounded.
    private func preview(_ manifest: JobManifest, format: String, inventoryOnly: Bool = false,
                         pass: TranscriptionPass? = nil, tag: String? = nil, trackProcess: Bool = true) async throws -> QueuePlan {
        guard let executableURL = cliURL else {
            throw PlanningFailure(message: "The transcriberrr executable is unavailable. Rebuild the app bundle.")
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        try JSONEncoder().encode(manifest).write(to: url, options: .atomic)
        let process = Process()
        let stdout = PipeCollector(limit: nil)
        let stderr = PipeCollector()
        process.executableURL = executableURL
        var arguments = ["--manifest", url.path, "--format", format, "--plan"]
        if inventoryOnly { arguments.append("--inventory-only") }
        if let pass { arguments += pass.arguments }
        if tag != nil { arguments += ["--model-suffix", "--skip-existing"] }
        process.arguments = arguments
        process.standardOutput = stdout.pipe
        process.standardError = stderr.pipe
        let tracked = trackProcess && !inventoryOnly
        let result: (Int32, String, String) = try await withCheckedThrowingContinuation { continuation in
            process.terminationHandler = { process in
                if tracked {
                    Task { @MainActor in
                        if self.activeProcess === process { self.activeProcess = nil }
                    }
                }
                continuation.resume(returning: (process.terminationStatus, stdout.finish(), stderr.finish()))
            }
            let failedToLaunch: (Error) -> Void = { error in
                _ = stdout.finish()
                _ = stderr.finish()
                continuation.resume(throwing: error)
            }
            if !tracked {
                do { try process.run() }
                catch { failedToLaunch(error) }
            } else {
                launch(process, onFailure: failedToLaunch)
            }
        }
        guard result.0 == 0 else {
            throw PlanningFailure(message: result.2.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return try JSONDecoder().decode(QueuePlan.self, from: Data(result.1.utf8))
    }

    private struct PlanningFailure: LocalizedError {
        var message: String
        var errorDescription: String? { message }
    }

    @discardableResult
    private func applyPlan(_ plan: QueuePlan, selectingMeetings: Bool = false, tag: String? = nil,
                           preserveDiarization: Bool = false, preserveInclusion: Bool = false) -> [AudioItem] {
        var known = itemsByPath(items)
        let unchecked = Set(items.filter { !$0.selected }.compactMap(\.zoomMeeting))
        let includedMeetings = Set(plan.files.compactMap { file in
            known[file.path]?.selected == true ? file.meeting : nil
        })
        var updates = Dictionary(uniqueKeysWithValues: plan.meetings.map { ($0.id, $0) })
        if !selectingMeetings {
            for previous in zoomMeetings where mergeResults[previous.output] != nil || outputExists(previous.output) {
                guard var meeting = updates[previous.id] else { continue }
                meeting.output = previous.output
                meeting.done = previous.done
                updates[meeting.id] = meeting
            }
        }
        if selectingMeetings {
            passMeetingIDsByOutput = Dictionary(plan.meetings.map { ($0.output, $0.id) }, uniquingKeysWith: { first, _ in first })
        }
        let completed = Set(plan.files.filter { $0.done == true }.map(\.path))
        let affected = Set(plan.files.compactMap { known[$0.path]?.zoomMeeting })
        for index in zoomMeetings.indices {
            let meeting = zoomMeetings[index]
            if updates[meeting.id] == nil, affected.contains(meeting.id), !outputExists(meeting.output) {
                zoomMeetings[index].done = false
                mergeResults.removeValue(forKey: meeting.output)
            }
        }
        for old in zoomMeetings {
            if tag == nil, let update = updates[old.id], update.output != old.output,
               mergeResults[old.output]?.modelTag == nil {
                mergeResults.removeValue(forKey: old.output)
            }
        }
        for meeting in plan.meetings where selectingMeetings {
            if let result = mergeResults[meeting.output], result.state != "done" || !meeting.done {
                mergeResults.removeValue(forKey: meeting.output)
            }
            if meeting.done, mergeResults[meeting.output] == nil {
                mergeResults[meeting.output] = MergedSession(
                    id: meeting.output, label: meeting.label, state: "done", message: "Transcript already exists",
                    meetingID: meeting.id, modelTag: tag)
            }
        }
        zoomMeetings.removeAll { updates[$0.id] != nil }
        zoomMeetings.append(contentsOf: plan.meetings.compactMap { updates[$0.id] })

        var added: [URL] = []
        let planned = plan.files.map { file in
            let item: AudioItem
            let previous = known[file.path]
            if let previous {
                item = previous
            } else {
                let url = URL(fileURLWithPath: file.path)
                item = AudioItem(url: url, diarize: file.diarize ?? diarize, isoTrack: file.iso ?? false)
                items.append(item)
                known[file.path] = item
            }
            let preserveState = !selectingMeetings && file.meeting != nil && item.zoomMeeting == file.meeting
            if file.error == nil {
                if item.duration == 0 && file.skip == nil { added.append(item.url) }
                item.zoomMeeting = file.meeting
                if preserveInclusion {
                    if previous == nil {
                        item.selected = file.meeting.map(includedMeetings.contains) ?? false
                    }
                } else if let meeting = file.meeting {
                    item.selected = selectingMeetings || !unchecked.contains(meeting)
                }
            }
            if let zoomRecording = file.zoomRecording { item.zoomRecording = zoomRecording }
            item.skipReason = file.skip
            item.coveredBy = file.coveredBy
            if let speaker = file.speaker {
                item.isoTrack = true
                item.diarize = false
                item.speakerCount = nil
                item.speakerName = speaker
                item.sessionLabel = file.meeting.flatMap { updates[$0]?.label }
                item.outputURL = nil
            } else if file.error == nil && file.skip == nil {
                if let iso = file.iso, selectingMeetings || item.isoOverride == nil { item.isoTrack = iso }
                if selectingMeetings && !preserveDiarization, let diarize = file.diarize { item.diarize = diarize }
            }
            if let tag, let output = file.output, file.error == nil && file.skip == nil {
                let history = ModelOutput(tag: tag, path: output)
                if !item.modelOutputs.contains(history) { item.modelOutputs.append(history) }
            }
            if selectingMeetings, file.done == true, let merge = file.mergeOutput,
               mergeResults[merge]?.state != "done" {
                mergeResults[merge] = MergedSession(
                    id: merge, label: file.group ?? URL(fileURLWithPath: merge).lastPathComponent,
                    state: "done", message: "Transcript already exists", modelTag: tag)
            }
            if let error = file.error {
                item.state = .failed(error)
                item.progress = 0
                item.resultSummary = error
            } else if let skip = file.skip, !preserveState {
                let covered = file.coveredBy?.contains(where: completed.contains) == true
                    || hasSelectedCover(item, in: known, scope: selectingMeetings ? runScopeIDs : nil)
                item.state = covered ? .done : .queued
                item.progress = covered ? 1 : 0
                item.outputURL = nil
                item.resultSummary = skip
            } else if file.done == true, !preserveState {
                item.state = .done
                item.progress = 1
                item.outputURL = file.output.map(URL.init(fileURLWithPath:))
                item.resultSummary = "Transcript already exists"
            } else if !preserveState && (file.meeting != nil || file.zoomRecording == true) {
                item.state = .queued
                item.progress = 0
                item.outputURL = nil
                item.resultSummary = ""
            }
            return item
        }
        let represented = Set(items.compactMap(\.zoomMeeting))
        zoomMeetings.removeAll { !represented.contains($0.id) }
        pruneOutcomes()
        objectWillChange.send()
        saveQueue()
        loadDurations(for: added)
        return planned
    }

    private func itemsByPath(_ items: [AudioItem]) -> [String: AudioItem] {
        Dictionary(items.map { (canonicalPath($0.url.path), $0) }, uniquingKeysWith: { first, _ in first })
    }

    private func hasSelectedCover(_ item: AudioItem, in known: [String: AudioItem], scope: Set<UUID>? = nil) -> Bool {
        var paths = item.coveredBy ?? []
        var visited: Set<String> = []
        while let path = paths.popLast() {
            guard visited.insert(path).inserted, let cover = known[path],
                  scope?.contains(cover.id) ?? cover.selected else { continue }
            if cover.state == .done, let output = cover.outputURL, outputExists(output.path) { return true }
            paths.append(contentsOf: cover.coveredBy ?? [])
        }
        return false
    }

    private func outputExists(_ path: String) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &directory) && !directory.boolValue
    }

    private func reconcileSkippedRows(_ tracked: [AudioItem], scope: Set<UUID>) {
        let known = itemsByPath(items)
        for item in tracked where item.state == .done && item.zoomMeeting == nil {
            if let reason = item.skipReason, item.outputURL == nil, !hasSelectedCover(item, in: known, scope: scope) {
                item.state = .queued
                item.progress = 0
                item.resultSummary = reason
            }
        }
    }

    func remove(_ item: AudioItem) {
        if item.zoomRecording { removedZoomInputs.insert(canonicalPath(item.url.path)) }
        items.removeAll { $0.id == item.id }
        pruneOutcomes()
        saveQueue()
    }

    func removeMeeting(_ meeting: ZoomMeeting) {
        items.removeAll { $0.zoomMeeting == meeting.id }
        zoomMeetings.removeAll { $0.id == meeting.id }
        mergeResults = mergeResults.filter { $0.key != meeting.output && $0.value.meetingID != meeting.id }
        pruneOutcomes()
        saveQueue()
    }

    func meetingResult(_ meeting: ZoomMeeting) -> MergedSession? {
        let result = mergeResults[meeting.output]
            ?? (meeting.done ? MergedSession(id: meeting.output, label: meeting.label, state: "done",
                                              message: "Transcript already exists") : nil)
        if result?.state == "done" && !outputExists(meeting.output) { return nil }
        return result
    }

    func meetingResults(_ meeting: ZoomMeeting, previous: [MergedSession]) -> [MergedSession] {
        var results = previous.filter { $0.id != meeting.output }
        if let current = meetingResult(meeting) { results.append(current) }
        return results.filter { $0.state != "done" || outputExists($0.id) }
            .sorted { ($0.modelTag ?? "", $0.id) < ($1.modelTag ?? "", $1.id) }
    }

    func modelOutputs(for item: AudioItem) -> [ModelOutput] {
        item.modelOutputs.filter { outputExists($0.path) }.sorted { ($0.tag, $0.path) < ($1.tag, $1.path) }
    }

    func clearCompleted() {
        let completed = completedMeetingIDs
        items.removeAll { item in
            item.zoomMeeting.map(completed.contains) ?? (item.state == .done)
        }
        zoomMeetings.removeAll { completed.contains($0.id) }
        mergeResults = mergeResults.filter {
            $0.value.meetingID.map { !completed.contains($0) } ?? ($0.value.state != "done")
        }
        pruneOutcomes()
        saveQueue()
    }

    func setSelection(for item: AudioItem, _ selected: Bool) {
        item.selected = selected
        objectWillChange.send()
        saveQueue()
    }

    func setDiarization(for item: AudioItem, enabled: Bool) {
        item.diarize = enabled
        if enabled {
            item.isoTrack = false
            item.isoOverride = false
        }
        if !enabled { item.speakerCount = nil }
        objectWillChange.send()
        saveQueue()
    }

    /// A speaker track holds one speaker, so it is never diarized.
    func setIsoTrack(for item: AudioItem, enabled: Bool) {
        item.isoTrack = enabled
        item.isoOverride = enabled
        if enabled {
            item.diarize = false
            item.speakerCount = nil
        }
        objectWillChange.send()
        saveQueue()
    }

    /// Marks the visible rows as speaker tracks, or unmarks them if all already are.
    func toggleIsoTracks(_ visibleItems: [AudioItem]) {
        let editable = visibleItems.filter { $0.zoomMeeting == nil }
        let enable = !editable.allSatisfy(\.isoTrack)
        for item in editable {
            item.isoTrack = enable
            item.isoOverride = enable
            if enable {
                item.diarize = false
                item.speakerCount = nil
            }
        }
        objectWillChange.send()
        saveQueue()
    }

    func revealMerged(_ session: MergedSession) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: session.id)])
    }

    /// Non-Zoom speaker tracks are grouped by source folder.
    private func sessionKey(_ item: AudioItem) -> String? {
        guard item.isoTrack && item.zoomMeeting == nil else { return nil }
        return canonicalPath(item.url.deletingLastPathComponent().path)
    }

    func setSpeakerCount(for item: AudioItem, _ value: Int?) {
        item.speakerCount = value
        objectWillChange.send()
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
                // Distinct URLs can share a path; uniqueKeysWithValues would trap.
                let values = Dictionary(durations, uniquingKeysWith: { first, _ in first })
                for item in self.items {
                    if let duration = values[item.url.standardizedFileURL.path] {
                        item.duration = duration
                    }
                }
                // Rows observe AppState, not each AudioItem; publish so durations appear.
                self.objectWillChange.send()
                self.saveQueue()
            }
        }
    }

    func selectAll(_ visibleItems: [AudioItem], selected: Bool) {
        let meetings = Set(visibleItems.compactMap(\.zoomMeeting))
        visibleItems.forEach { $0.selected = selected }
        items.filter { $0.zoomMeeting.map(meetings.contains) == true }.forEach { $0.selected = selected }
        objectWillChange.send()
        saveQueue()
    }

    func chooseFiles() {
        guard !isRunning else { return }
        let panel = NSOpenPanel()
        // Filter supported extensions after selection so folders and formats such
        // as FLAC, OGG, AAC, and WebM can be selected consistently.
        panel.allowedContentTypes = []
        panel.allowsOtherFileTypes = true
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { add(urls: panel.urls) }
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
        panel.directoryURL = current.path == "/sortformer"
            ? FileManager.default.homeDirectoryForCurrentUser
            : current.deletingLastPathComponent()
        if panel.runModal() == .OK, let url = panel.url { completion(url) }
    }

    func cancelTranscription() {
        guard isRunning, !isCancelling else { return }
        isCancelling = true
        modelActivityMessage = "Stopping safely and cleaning temporary queue files…"
        guard let process = activeProcess else { return }
        // SIGTERM lets the CLI stop at the next chunk, stop its helper, and remove temp
        // files; before its first job (model loading) it exits at once. A long step without a
        // checkpoint (decoding hours of audio, diarizing) can take a while, and killing it
        // then orphans the helper, so force-stop only a CLI that seems stuck.
        process.terminate()
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(120))
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        }
    }

    private func markCanceled(_ pending: [AudioItem]) {
        // Finished tracks keep their result; only unfinished ones become canceled.
        for item in pending where item.state != .done {
            if case .failed = item.state { continue }
            item.state = .canceled
            item.progress = 0
            item.resultSummary = "Canceled"
        }
        objectWillChange.send()
        saveQueue()
    }

    func transcribe() {
        startRun(Set(selectedItems.map(runTarget)), scopedRetry: false)
    }

    func retry(_ target: RunTarget) {
        startRun([target], scopedRetry: true)
    }

    private func runTarget(for item: AudioItem) -> RunTarget {
        item.zoomMeeting.map(RunTarget.meeting) ?? .recording(item.id)
    }

    private func startRun(_ targets: Set<RunTarget>, scopedRetry: Bool) {
        guard !isRunning, downloadingLibraryModel == nil, addOperations == 0, !targets.isEmpty else { return }
        runScopeIDs = Set(items.filter { targets.contains(runTarget(for: $0)) }.map(\.id))
        guard !runScopeIDs.isEmpty else { return }
        runTargets = targets
        let passes = configuredPasses
        lastRunSummary = nil
        isRunning = true
        isCancelling = false
        passCount = passes.count
        passIndex = 0
        runItemIDs = []
        startQueueClock()
        Task { @MainActor in
            defer {
                finishQueueClock()
                activeModelTag = nil
                activePassLabel = nil
                activePassUsesApple = false
                passMeetingIDsByOutput = [:]
                currentPassWork = [:]
                runScopeIDs = []
                runTargets = []
                activeProcess = nil
                isRunning = false
                isCancelling = false
                saveQueue()
            }
            let tags: [String?]
            do {
                tags = try await validateModelPasses(passes)
            } catch {
                if !isCancelling { planningError = error.localizedDescription }
                lastRunSummary = isCancelling ? "Run canceled" : "Couldn’t start transcription"
                return
            }
            guard !isCancelling else {
                lastRunSummary = "Run canceled"
                return
            }
            let pendingOutcomes = zip(passes, tags).enumerated().map { index, entry in
                let (pass, tag) = entry
                let model = pass.modelArgument
                let label = isLocalModelPath(model) ? URL(fileURLWithPath: model).lastPathComponent : model
                return PassOutcome(id: tag ?? String(index), label: label, modelTag: tag, phase: .waiting)
            }
            for target in runTargets { outcomes[target] = pendingOutcomes }

            for (index, pass) in passes.enumerated() {
                if isCancelling { break }
                let tag = tags[index]
                passIndex = index
                runItemIDs = []
                activeModelTag = tag
                activePassUsesApple = pass.usesApple
                activePassLabel = "Pass \(index + 1) of \(passes.count) · \(pendingOutcomes[index].label)"
                await transcribePass(pass, tag: tag, scopedRetry: scopedRetry, pendingOutcomes: pendingOutcomes)
                if runTargets.contains(where: { outcomes[$0]?[index].phase == .canceled }) { break }
            }
            finishRunOutcomes()
        }
    }

    private func transcribePass(_ pass: TranscriptionPass, tag: String?, scopedRetry: Bool,
                                pendingOutcomes: [PassOutcome]) async {
        let known = itemsByPath(items)
        let selected = items.filter { runScopeIDs.contains($0.id) || runTargets.contains(runTarget(for: $0)) }
        runScopeIDs.formUnion(selected.map(\.id))
        currentPassWork = Dictionary(grouping: selected, by: runTarget)
            .mapValues { PassWork(items: $0, output: nil, mergePath: nil) }
        defer {
            updatePassOutcomes(ending: true)
            currentPassWork = [:]
        }
        let coveredSkips = selected.filter {
            tag == nil &&
            $0.zoomMeeting == nil && !$0.zoomRecording && $0.skipReason != nil &&
            hasSelectedCover($0, in: known, scope: runScopeIDs)
        }
        let coveredIDs = Set(coveredSkips.map(\.id))
        let newlyCovered = coveredSkips.filter { $0.state != .done }
        for item in newlyCovered {
            item.state = .done
            item.progress = 1
        }
        if !newlyCovered.isEmpty {
            objectWillChange.send()
            saveQueue()
        }
        // Check Zoom destinations on every run, even if the queue previously marked them done.
        var requested = selected.filter { item in
            // A covered export can be renamed while its saved queue row still points at the
            // old path. The selected participant tracks will rediscover the renamed file.
            if item.zoomMeeting != nil && !item.isoTrack && item.skipReason != nil
                && !FileManager.default.fileExists(atPath: item.url.path) { return false }
            return !coveredIDs.contains(item.id)
                && (tag != nil || item.state != .done || item.zoomMeeting != nil || item.zoomRecording || item.skipReason != nil
                    || reusableUntaggedOutput(for: item) == nil)
        }
        guard !requested.isEmpty else {
            for (target, work) in currentPassWork {
                currentPassWork[target]?.output = work.items.first?.outputURL
            }
            return
        }
        let sessions = Set(requested.compactMap(sessionKey))
        let requestedIDs = Set(requested.map(\.id))
        requested += selected.filter {
            !requestedIDs.contains($0.id) && $0.skipReason == nil && sessionKey($0).map(sessions.contains) == true
        }
        let selectedIDs = Set(selected.map(\.id))
        let excluded = Set(known.compactMap { path, item in selectedIDs.contains(item.id) ? nil : path })
            .union(removedZoomInputs)
        let manifest = makeRunManifest(requested, excluding: excluded.sorted(), pass: pass, tag: tag)
        let planned: [AudioItem]
        let pending: [AudioItem]
        let tracked: [AudioItem]
        let outputs: [UUID: URL]
        let mergeOutputs: [UUID: String]
        let frozen: JobManifest
        do {
            let plan = try await preview(manifest, format: format, pass: pass, tag: tag)
            if isCancelling { throw CancellationError() }
            planned = applyPlan(plan, selectingMeetings: true, tag: tag, preserveDiarization: pass.usesApple,
                                preserveInclusion: scopedRetry)
            runScopeIDs.formUnion(planned.map(\.id))
            for item in planned {
                let target = runTarget(for: item)
                if runTargets.insert(target).inserted { outcomes[target] = pendingOutcomes }
            }
            let accepted = zip(planned, plan.files).filter { $0.1.error == nil }
            tracked = accepted.map(\.0)
            pending = accepted.compactMap { item, file in
                file.done == true || file.skip != nil ? nil : item
            }
            outputs = Dictionary(accepted.compactMap { item, file in
                file.output.map { (item.id, URL(fileURLWithPath: $0)) }
            }, uniquingKeysWith: { first, _ in first })
            let meetings = Dictionary(uniqueKeysWithValues: plan.meetings.map { ($0.id, $0.output) })
            mergeOutputs = Dictionary(accepted.compactMap { item, file in
                guard file.skip == nil,
                      let output = file.mergeOutput ?? file.meeting.flatMap({ meetings[$0] }) else { return nil }
                return (item.id, output)
            }, uniquingKeysWith: { first, _ in first })
            let plannedIDs = Set(planned.map(\.id))
            currentPassWork = Dictionary(grouping: selected + planned, by: runTarget).mapValues { group in
                let tracks = Dictionary(group.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first }).values
                    .filter { $0.skipReason == nil }
                let candidates = tracks.isEmpty ? [group[0]] : Array(tracks)
                let mergePath = candidates.lazy.compactMap { mergeOutputs[$0.id] }.first
                let output = group[0].zoomMeeting == nil
                    ? candidates.lazy.compactMap { item in
                        outputs[item.id] ?? (!plannedIDs.contains(item.id) && item.state == .done ? item.outputURL : nil)
                    }.first
                    : mergePath.map(URL.init(fileURLWithPath:))
                return PassWork(items: candidates, output: output, mergePath: mergePath)
            }
            let original = Dictionary(manifest.items.enumerated().map { index, item in
                (canonicalPath(item.input), (index, item.output))
            }, uniquingKeysWith: { first, _ in first })
            // Include discovered tracks and skipped continuations explicitly. If a
            // track disappears during model setup, the CLI can reject the whole meeting.
            let inputs = accepted.enumerated().map { index, entry in
                let (item, file) = entry
                let (order, output) = original[file.path] ?? (manifest.items.count + index, nil)
                return (order, manifestItem(for: item, input: file.path, output: file.output.map { outputBase($0, tag: tag) } ?? output, pass: pass))
            }.sorted { $0.0 < $1.0 }.map(\.1)
            frozen = JobManifest(items: inputs, discoverZoom: false)
        } catch {
            if isCancelling {
                for target in currentPassWork.keys {
                    outcomes[target]?[passIndex].phase = .canceled
                    outcomes[target]?[passIndex].message = "Canceled"
                }
            } else {
                let prefix = tag.map { "\($0) · " } ?? ""
                planningError = prefix + error.localizedDescription
                for item in requested {
                    item.state = .failed(error.localizedDescription)
                    item.progress = 0
                    item.resultSummary = error.localizedDescription
                }
            }
            return
        }
        runItemIDs = Set(pending.map(\.id))
        guard !pending.isEmpty, !isCancelling else {
            if isCancelling { markCanceled(pending) }
            saveQueue()
            return
        }
        for (index, item) in pending.enumerated() {
            item.state = .queued
            item.progress = 0
            item.elapsedSeconds = 0
            item.estimatedRemainingSeconds = nil
            item.resultSummary = index == 0 ? "Waiting for model setup…" : "Waiting…"
        }
        updatePassOutcomes()
        objectWillChange.send()
        saveQueue()

        let modelDownloads = await requiredModelDownloads(for: pending, pass: pass)
        if !modelDownloads.isEmpty, !isCancelling {
            beginModelActivity(title: "Preparing models", message: "Checking and downloading model files…")
            modelActivityLog = modelDownloads.map { "Model: \($0)" }
            let result = await downloadModels(modelDownloads)
            if result.status != 0 {
                if isCancelling {
                    markCanceled(pending)
                } else {
                    let detail = result.message.isEmpty ? "Could not prepare model files." : result.message
                    pending.forEach {
                        $0.state = .failed(detail)
                        $0.resultSummary = detail
                    }
                }
                modelActivityMessage = isCancelling
                    ? "Model download canceled."
                    : result.message.isEmpty ? "Model setup failed." : result.message
                modelActivityCanDismiss = true
                saveQueue()
                return
            }
            // Get out of the way of the run's progress once the models are ready.
            showModelActivity = false
        }

        if let first = pending.first {
            first.state = .processing
            first.progress = 0
            first.resultSummary = "Starting transcription…"
            updatePassOutcomes()
            objectWillChange.send()
            saveQueue()
        }

        // Qwen and Parakeet keep their weights in process memory. Run the
        // whole queue through one manifest so a 5,000-track queue does not
        // repeatedly pay model startup and Metal pipeline compilation. The
        // Apple engine goes through the CLI too, which converts formats
        // AVFoundation cannot open (WebM) and validates the transcript.
        if !pending.isEmpty, !isCancelling {
            let result = await runBulkCLI(frozen, for: tracked, outputs: outputs, pass: pass, tag: tag)
            if isCancelling {
                applyFinished(result, to: tracked)
                markCanceled(pending)
            } else {
                applyBulk(result, to: tracked)
            }
            for item in tracked {
                guard let output = mergeOutputs[item.id], let merge = mergeResults[output] else { continue }
                if case .failed = item.state { continue }
                if merge.state == "failed" {
                    item.state = .failed(merge.message)
                    item.resultSummary = merge.message
                } else if merge.state == "canceled" {
                    item.state = .canceled
                    item.progress = 0
                    item.resultSummary = "Canceled"
                }
            }
            reconcileSkippedRows(tracked, scope: runScopeIDs)
            objectWillChange.send()
            saveQueue()
        }
        if isCancelling { markCanceled(pending) }
    }

    @discardableResult
    private func updatePassOutcomes(ending: Bool = false) -> Bool {
        var next = outcomes
        var finished = false
        for (target, work) in currentPassWork {
            guard var outcome = next[target]?[passIndex] else { continue }
            let prior = outcome
            let merge = work.mergePath.flatMap { mergeResults[$0] }
            let failure = work.items.compactMap { item -> String? in
                guard case .failed(let error) = item.state else { return nil }
                return error
            }.first
            let tracksDone = work.items.allSatisfy { $0.state == .done }
            if let url = work.output,
               outcome.output == url || (ending || merge?.state == "done" || tracksDone) && outputExists(url.path) {
                outcome.output = url
            }
            if prior.phase == .canceled {
                continue
            } else if let message = merge?.state == "failed" ? merge?.message : failure {
                outcome.phase = .failed
                outcome.message = message
            } else if merge?.state == "canceled" || work.items.contains(where: { $0.state == .canceled }) {
                outcome.phase = .canceled
                outcome.message = "Canceled"
            } else if merge?.state == "done" || tracksDone && work.mergePath == nil {
                outcome.phase = .completed
                outcome.message = merge?.message ?? work.items.first?.resultSummary
            } else if ending {
                outcome.phase = isCancelling ? .canceled : .failed
                outcome.message = isCancelling ? "Canceled" : "Transcription did not finish"
            } else if merge?.state == "running" || work.items.contains(where: { $0.state == .processing }) {
                outcome.phase = .running
                outcome.message = merge?.message ?? work.items.first(where: { $0.state == .processing })?.resultSummary
            } else {
                outcome.phase = .waiting
                outcome.message = nil
            }
            if outcome != prior {
                next[target]?[passIndex] = outcome
                finished = finished || [.completed, .failed, .canceled].contains(outcome.phase)
            }
        }
        if next != outcomes { outcomes = next }
        return finished
    }

    private func finishRunOutcomes() {
        let scoped = items.filter { runScopeIDs.contains($0.id) }
        let targets = Set(scoped.map(runTarget))
        let stopped = isCancelling || targets.contains { target in
            outcomes[target]?.contains { $0.phase == .canceled } == true
        }
        var completed = 0
        var failed = 0
        var saved = Set<URL>()
        for target in targets {
            guard var results = outcomes[target] else { continue }
            for index in results.indices where results[index].phase == .waiting || results[index].phase == .running {
                results[index].phase = stopped ? .canceled : .failed
                results[index].message = stopped ? "Canceled" : "Transcription did not finish"
            }
            outcomes[target] = results
            saved.formUnion(results.compactMap(\.output))
            if results.allSatisfy({ $0.phase == .completed }) { completed += 1 }
            if results.contains(where: { $0.phase == .failed }) { failed += 1 }
        }
        for item in scoped where item.skipReason == nil {
            guard let results = outcomes[runTarget(for: item)] else { continue }
            let failures = results.filter { $0.phase == .failed }
            if !failures.isEmpty {
                let message = failures.map { outcome in
                    let failure = outcome.message ?? "Transcription failed"
                    return results.count > 1 ? "\(outcome.label) · \(failure)" : failure
                }.joined(separator: "\n")
                item.state = .failed(message)
                item.resultSummary = message
                item.estimatedRemainingSeconds = nil
            } else if results.contains(where: { $0.phase == .canceled }) {
                item.state = .canceled
                item.progress = 0
                item.resultSummary = "Canceled"
                item.estimatedRemainingSeconds = nil
            }
        }
        if stopped {
            lastRunSummary = saved.isEmpty ? "Run canceled" : "Run canceled · \(saved.count) saved"
        } else if failed > 0 {
            lastRunSummary = "Run finished · \(completed) complete · \(failed) failed"
        } else {
            lastRunSummary = completed == 1 ? "Run complete · 1 recording" : "Run complete · \(completed) recordings"
        }
        pruneOutcomes()
    }

    private func pruneOutcomes() {
        let live = Set(items.map(runTarget))
        if !Set(outcomes.keys).isSubset(of: live) {
            outcomes = outcomes.filter { live.contains($0.key) }
        }
    }

    private func startQueueClock() {
        clockTask?.cancel()
        let now = Date()
        clock = now
        queueStartedAt = now
        storedQueueElapsedSeconds = 0
        clockTask = Task { @MainActor [weak self] in
            while let self, !Task.isCancelled, self.isRunning {
                self.clock = Date()
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }

    private func finishQueueClock() {
        clock = Date()
        storedQueueElapsedSeconds = queueElapsedSeconds
        queueStartedAt = nil
        clockTask?.cancel()
        clockTask = nil
    }

    private func requiredModelDownloads(for pending: [AudioItem], pass: TranscriptionPass) async -> [String] {
        guard !pass.usesApple else { return [] }
        var candidates: [String] = []
        func append(_ value: String) {
            // Local paths cannot be downloaded; the CLI reports a missing one itself.
            guard !value.isEmpty, !isLocalModelPath(value), !candidates.contains(value) else { return }
            candidates.append(value)
        }
        let model = pass.modelArgument
        append(pass.engine == "parakeet-v3" && isQwenModel(model) ? pass.defaultModel : model)
        let diarizing = pending.contains(where: { $0.diarize || $0.speakerCount != nil })
        // The CLI loads the aligner for diarization too: speakers attach to word timings.
        // Subtitles always get sentence timing (main.rs), which needs the aligner too, and
        // merging speaker tracks needs every track's word timings.
        let merging = pending.contains(where: \.isoTrack)
        if pass.engine == "qwen" && (timestamps != "none" || diarizing || merging || format == "srt" || format == "vtt") {
            append("Qwen/Qwen3-ForcedAligner-0.6B")
        }
        if diarizing {
            append(diarizationModelArgument)
        }
        guard !candidates.isEmpty else { return [] }
        // If the query fails, let the download step decide; it is a no-op for
        // files already in the cache.
        guard let cached = await cachedModels(candidates) else { return candidates }
        return candidates.filter { cached[$0] != true }
    }

    /// Asks the CLI (`--cached-model`) which models are usable offline, so the
    /// app never re-implements aliases, completeness markers, or snapshot rules.
    private func cachedModels(_ ids: [String]) async -> [String: Bool]? {
        struct Entry: Decodable { var id: String; var cached: Bool }
        guard let result = await readCLI(ids.flatMap({ ["--cached-model", $0] })),
              let entries = try? JSONDecoder().decode([Entry].self, from: Data(result.utf8)) else { return nil }
        return Dictionary(entries.map { ($0.id, $0.cached) }, uniquingKeysWith: { first, _ in first })
    }

    private func readModelTag(_ pass: TranscriptionPass) async -> String? {
        guard let result = await readCLI(["--print-model-tag"] + pass.arguments) else { return nil }
        let tag = result.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !tag.isEmpty, tag.utf8.allSatisfy({ (48...57).contains($0) || (97...122).contains($0) }) else { return nil }
        return tag
    }

    private func readCLI(_ arguments: [String]) async -> String? {
        guard let executableURL = cliURL else { return nil }
        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        let output = PipeCollector()
        process.standardOutput = output.pipe
        process.standardError = FileHandle.nullDevice
        let result: (Int32, String)? = await withCheckedContinuation { continuation in
            process.terminationHandler = { process in
                continuation.resume(returning: (process.terminationStatus, output.finish()))
            }
            do {
                try process.run()
            } catch {
                process.terminationHandler = nil
                _ = output.finish()
                continuation.resume(returning: nil)
            }
        }
        guard let result, result.0 == 0 else { return nil }
        return result.1
    }

    var diarizationModelArgument: String {
        // URL(fileURLWithPath: "sortformer") normalizes the alias to "/sortformer".
        // Preserve the CLI alias so model_cache canonicalizes it to the compatible ONNX export.
        diarizationModel.path == "/sortformer" ? "sortformer" : diarizationModel.path
    }

    private func beginModelActivity(title: String, message: String) {
        modelActivityTitle = title
        modelActivityMessage = message
        modelActivityLog = []
        modelActivityProgress = nil
        modelActivityCanDismiss = false
        showModelActivity = true
    }

    private func appendModelActivity(_ text: String) {
        let normalized = text.replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.split(whereSeparator: { $0 == "\n" }).map(String.init)
        guard !lines.isEmpty else { return }
        modelActivityLog.append(contentsOf: lines.suffix(80))
        if modelActivityLog.count > 160 {
            modelActivityLog.removeFirst(modelActivityLog.count - 160)
        }
        if let last = lines.last, !last.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            modelActivityMessage = last.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // No trailing \b: after "%" it requires a word character, so "45% " never matched.
        // The last percentage in the chunk is the current one.
        if let regex = try? NSRegularExpression(pattern: #"\b(\d{1,3}(?:\.\d+)?)%"#),
           let match = regex.matches(in: normalized, range: NSRange(normalized.startIndex..., in: normalized)).last,
           let range = Range(match.range(at: 1), in: normalized),
           let percent = Double(normalized[range]) {
            modelActivityProgress = min(max(percent / 100, 0), 1)
        }
    }

    private func downloadModels(_ ids: [String]) async -> (status: Int32, message: String) {
        guard let executableURL = cliURL else {
            return (1, "The transcriberrr executable is not bundled. Rebuild the app with scripts/build-macos-app.sh.")
        }
        let process = Process()
        process.executableURL = executableURL
        process.arguments = ids.flatMap { ["--download-model", $0] }
        let output = PipeCollector { [weak self] data in
            let text = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self?.appendModelActivity(text) }
        }
        process.standardOutput = output.pipe
        process.standardError = output.pipe
        return await withCheckedContinuation { continuation in
            process.terminationHandler = { process in
                let transcript = output.finish()
                Task { @MainActor in
                    if self.activeProcess === process { self.activeProcess = nil }
                    // The streamed log already holds the output; report the last line
                    // (usually the error) instead of only whatever arrived after exit.
                    let message = transcript.split(whereSeparator: \.isNewline).last.map(String.init) ?? ""
                    continuation.resume(returning: (process.terminationStatus, message))
                }
            }
            launch(process) { error in
                _ = output.finish()
                continuation.resume(returning: (1, error.localizedDescription))
            }
        }
    }

    /// Starts `process` as the active (cancellable) process. A cancel that arrived
    /// while no process was running would otherwise be lost and let a whole queue run.
    private func launch(_ process: Process, onFailure: (Error) -> Void) {
        do {
            activeProcess = process
            try process.run()
            if isCancelling { process.terminate() }
        } catch {
            process.terminationHandler = nil
            activeProcess = nil
            onFailure(error)
        }
    }

    /// Applies one CLI progress event. The live poller and the end-of-run pass
    /// share this, so a track is marked done as soon as the CLI finishes it and
    /// stays done if the rest of the run is canceled or the app quits.
    private func apply(_ event: ProgressEvent, to item: AudioItem) {
        if let speaker = event.speaker { item.speakerName = speaker }
        if let group = event.group { item.sessionLabel = group }
        switch event.status {
        case "running":
            item.state = .processing
            item.progress = event.progress ?? item.progress
            if let elapsed = event.elapsedSeconds {
                item.elapsedSeconds = elapsed
                item.estimatedRemainingSeconds = item.progress > 0.01
                    ? max(0, elapsed * (1 / item.progress - 1))
                    : nil
            }
            item.resultSummary = event.phase ?? "Transcribing…"
        case "completed", "skipped":
            let output = event.output.map(URL.init(fileURLWithPath:))
            item.progress = 1
            item.elapsedSeconds = event.elapsedSeconds ?? item.elapsedSeconds
            item.estimatedRemainingSeconds = nil
            item.state = .done
            item.outputURL = output
            item.resultSummary = output.map { "Saved to \($0.path)" } ?? (event.phase ?? "Transcription complete")
            if let warning = event.warning {
                // A file the CLI planned out (a meeting's video, say) has no output of its own.
                item.resultSummary = output == nil ? warning : "\(warning). \(item.resultSummary)"
            }
        case "canceled":
            item.estimatedRemainingSeconds = nil
            item.state = .canceled
            item.progress = 0
            item.resultSummary = "Canceled"
        case "failed":
            let message = event.error ?? "Transcription failed"
            item.elapsedSeconds = event.elapsedSeconds ?? item.elapsedSeconds
            item.estimatedRemainingSeconds = nil
            item.state = .failed(message)
            item.resultSummary = message
        default:
            break
        }
    }

    private static let finalStatuses: Set<String> = ["completed", "skipped", "failed", "canceled"]

    /// Shows a session's merged transcript from its CLI event.
    private func applyMerge(_ event: ProgressEvent) {
        let path = event.input
        let state = switch event.status {
        case "completed", "skipped": "done"
        case "failed": "failed"
        case "canceled": "canceled"
        default: "running"
        }
        let message = switch state {
        case "done": event.warning ?? (event.status == "skipped" ? "Transcript already exists" : "Merged")
        case "failed": event.error ?? "Merge failed"
        case "canceled": "Canceled"
        default: event.phase ?? "Waiting for its tracks…"
        }
        let session = MergedSession(id: path, label: event.group ?? URL(fileURLWithPath: path).lastPathComponent,
                                    state: state, message: message,
                                    meetingID: passMeetingIDsByOutput[path],
                                    modelTag: activeModelTag)
        if mergeResults[path] != session { mergeResults[path] = session }
    }

    private func applyMerges(_ events: [String: ProgressEvent]) {
        for event in events.values where event.kind == "merge" { applyMerge(event) }
    }

    /// A canceled or interrupted run leaves no session merging.
    private func stopRunningMerges() {
        for (path, var result) in mergeResults where result.state == "running" {
            result.state = "canceled"
            result.message = "Canceled"
            mergeResults[path] = result
        }
    }

    /// Applies only finished-track events; used when a run is canceled.
    private func applyFinished(_ result: BulkResult, to items: [AudioItem]) {
        applyMerges(result.progress)
        stopRunningMerges()
        for item in items {
            if let event = result.progress[progressKey(item.url.path)], Self.finalStatuses.contains(event.status) {
                apply(event, to: item)
            }
        }
    }

    private func applyBulk(_ result: BulkResult, to items: [AudioItem]) {
        applyMerges(result.progress)
        // stderr holds a line per finished track; the last line is the run's error.
        let fallback = result.message.split(whereSeparator: \.isNewline).last
            .map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
        // The CLI was stopped from outside the app (Activity Monitor, `kill`): tracks it never
        // reached were canceled, not failed.
        // Before its first job, the CLI answers SIGTERM/SIGINT by exiting 143/130 at once,
        // with no event written.
        let canceled = result.progress.values.contains { $0.status == "canceled" }
            || result.status == 143 || result.status == 130
        if canceled {
            stopRunningMerges()
        } else {
            // The CLI died mid-session (crash, out of memory): the merge failed, not canceled.
            let message = fallback.isEmpty ? "Transcription did not finish" : fallback
            for (path, var merge) in mergeResults where merge.state == "running" {
                merge.state = "failed"
                merge.message = message
                mergeResults[path] = merge
            }
        }
        for item in items {
            let event = result.progress[progressKey(item.url.path)]
            if let event, Self.finalStatuses.contains(event.status) {
                apply(event, to: item)
            } else if item.state == .done {
                continue
            } else if canceled {
                item.estimatedRemainingSeconds = nil
                item.state = .canceled
                item.progress = 0
                item.resultSummary = "Canceled"
            } else if result.status == 0, let output = result.outputs[item.id], outputExists(output.path) {
                item.progress = 1
                item.estimatedRemainingSeconds = nil
                item.state = .done
                item.outputURL = output
                item.resultSummary = "Saved to \(output.path)"
            } else {
                let message = event?.error ?? (fallback.isEmpty ? "Transcription failed" : fallback)
                item.elapsedSeconds = event?.elapsedSeconds ?? item.elapsedSeconds
                item.estimatedRemainingSeconds = nil
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

    private func reusableUntaggedOutput(for item: AudioItem) -> URL? {
        guard let output = item.outputURL, outputExists(output.path),
              !item.modelOutputs.contains(where: { canonicalPath($0.path) == canonicalPath(output.path) }) else { return nil }
        let expected = expectedOutput(for: item)
        guard output.pathExtension.caseInsensitiveCompare(expected.pathExtension) == .orderedSame,
              canonicalPath(output.deletingLastPathComponent().path) ==
                canonicalPath(expected.deletingLastPathComponent().path) else { return nil }
        return output
    }

    /// Picks an output path that no other row uses and that does not replace an existing
    /// file such as notes.txt beside notes.m4a.
    private func uniqueOutput(_ candidate: URL, tag: String?, reserved: inout Set<String>) -> URL {
        func available(_ url: URL) -> Bool {
            let target = tag.map { url.deletingPathExtension().appendingPathExtension($0).appendingPathExtension(url.pathExtension) } ?? url
            let key = target.standardizedFileURL.path.lowercased()
            guard !reserved.contains(key) else { return false }
            guard !FileManager.default.fileExists(atPath: target.path) else { return false }
            reserved.insert(key)
            return true
        }
        if available(candidate) { return candidate }
        let stem = candidate.deletingPathExtension().lastPathComponent
        let ext = candidate.pathExtension
        var suffix = 2
        while true {
            let numbered = candidate.deletingLastPathComponent()
                .appendingPathComponent("\(stem)-\(suffix)")
                .appendingPathExtension(ext)
            if available(numbered) { return numbered }
            suffix += 1
        }
    }

    private struct BulkResult {
        var status: Int32
        var message: String
        var progress: [String: ProgressEvent]
        var outputs: [UUID: URL] = [:]
    }

    private var cliOutputLocation: String {
        outputDestination == .folder ? "folder" : "next-to-source"
    }

    private func outputBase(_ path: String, tag: String?) -> String {
        guard let tag else { return path }
        let url = URL(fileURLWithPath: path)
        let stem = url.deletingPathExtension()
        guard stem.pathExtension == tag else { return path }
        return stem.deletingPathExtension().appendingPathExtension(url.pathExtension).path
    }

    private func manifestItem(for item: AudioItem, input: String, output: String?, pass: TranscriptionPass) -> ManifestItem {
        ManifestItem(input: input, diarize: !pass.usesApple && item.diarize, speakers: pass.usesApple ? nil : item.speakerCount, output: output,
                     outputDir: outputFolder.path, outputLocation: cliOutputLocation, isoTrack: item.isoOverride)
    }

    private func makeRunManifest(_ requested: [AudioItem], excluding: [String], pass: TranscriptionPass, tag: String?) -> JobManifest {
        var reserved = Set(items.flatMap { item in
            (item.modelOutputs.map(\.path) + [item.outputURL?.path].compactMap { $0 }).map {
                URL(fileURLWithPath: $0).standardizedFileURL.path.lowercased()
            }
        })
        return JobManifest(items: requested.map { item in
            var output: String?
            if item.zoomMeeting == nil && !item.zoomRecording {
                let candidate = expectedOutput(for: item)
                let destination = canonicalPath(candidate.deletingLastPathComponent().path)
                if let tag, let recorded = item.modelOutputs.last(where: { entry in
                    guard entry.tag == tag else { return false }
                    let prior = URL(fileURLWithPath: entry.path)
                    return prior.pathExtension == candidate.pathExtension
                        && canonicalPath(prior.deletingLastPathComponent().path) == destination
                }) {
                    output = outputBase(recorded.path, tag: tag)
                }
                if output == nil { output = uniqueOutput(candidate, tag: tag, reserved: &reserved).path }
            }
            return manifestItem(for: item, input: item.url.standardizedFileURL.path, output: output, pass: pass)
        }, excludedInputs: excluding.isEmpty ? nil : excluding)
    }

    private func runBulkCLI(_ manifest: JobManifest, for items: [AudioItem], outputs: [UUID: URL],
                            pass: TranscriptionPass, tag: String?) async -> BulkResult {
        guard let executableURL = cliURL else {
            return BulkResult(status: 1, message: "The transcriberrr executable is not bundled. Rebuild the app with scripts/build-macos-app.sh.", progress: [:])
        }
        let temporaryRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("Transcriberrr", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let manifestURL = temporaryRoot.appendingPathComponent("jobs.json")
        let progressURL = temporaryRoot.appendingPathComponent("progress.jsonl")
        do {
            try FileManager.default.createDirectory(at: temporaryRoot, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(manifest)
            try data.write(to: manifestURL, options: .atomic)
            var arguments = ["--manifest", manifestURL.path, "--output-dir", outputFolder.path,
                             "--device", device, "--format", format,
                             "--timestamps", timestamps, "--progress-file", progressURL.path, "--resume"]
            arguments += pass.arguments
            if tag != nil { arguments += ["--model-suffix", "--skip-existing"] }
            if format == "txt" && speakerTurns {
                arguments.append("--speaker-turns")
            }
            if !pass.usesApple && items.contains(where: { $0.diarize || $0.speakerCount != nil }) {
                arguments += ["--diarization-model", diarizationModelArgument]
            }
            let process = Process()
            process.executableURL = executableURL
            process.arguments = arguments
            // TRANSCRIBERRR_PARENT_PID: the CLI stops if this app crashes or is force-quit.
            // TRANSCRIBERRR_NATIVE: lets a development CLI (target/release) find the helper.
            var environment = ProcessInfo.processInfo.environment
            environment["TRANSCRIBERRR_PARENT_PID"] = String(ProcessInfo.processInfo.processIdentifier)
            if let nativeURL { environment["TRANSCRIBERRR_NATIVE"] = nativeURL.path }
            process.environment = environment
            let errors = PipeCollector()
            process.standardError = errors.pipe
            let qwenEngine = pass.engine == "qwen"
            let progressReader = ProgressReader()
            // Resolve once: realpath for every row every poll is thousands of syscalls
            // per second on the main actor for a large queue.
            let keys = items.map { progressKey($0.url.path) }
            let progressPoller = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(500))
                    guard !Task.isCancelled else { break }
                    let events = progressReader.read(from: progressURL)
                    guard let self else { break }
                    var finished = false
                    for (item, key) in zip(items, keys) {
                        guard let event = events[key] else { continue }
                        let wasDone = item.state == .done
                        self.apply(event, to: item)
                        finished = finished || (!wasDone && item.state == .done)
                    }
                    self.applyMerges(events)
                    finished = self.updatePassOutcomes() || finished
                    self.objectWillChange.send()
                    // Persist finished rows now: after a crash or force quit they would
                    // otherwise reload as queued and be transcribed again as `name-2`.
                    if finished { self.saveQueue() }
                }
            }
            let result = await withCheckedContinuation { continuation in
                process.terminationHandler = { process in
                    let message = errors.finish()
                    Task { @MainActor in
                        if self.activeProcess === process { self.activeProcess = nil }
                    }
                    // Killed by a signal (out-of-memory, crash): stderr's last line is usually the
                    // previous track's success line, which made a misleading failure reason.
                    let advice = qwenEngine ? " If this repeats, try the smaller Qwen3-ASR-0.6B model." : ""
                    let reason = process.terminationReason == .uncaughtSignal
                        ? "Transcriber stopped unexpectedly (signal \(process.terminationStatus))." + advice
                        : message
                    continuation.resume(returning: (process.terminationStatus, reason))
                }
                launch(process) { error in
                    _ = errors.finish()
                    continuation.resume(returning: (1, error.localizedDescription))
                }
            }
            progressPoller.cancel()
            await progressPoller.value
            let events = progressReader.read(from: progressURL)
            // The parent is shared by other app instances; remove only this
            // job's directory so their active manifests remain intact.
            try? FileManager.default.removeItem(at: temporaryRoot)
            return BulkResult(status: result.0, message: result.1, progress: events, outputs: outputs)
        } catch {
            try? FileManager.default.removeItem(at: temporaryRoot)
            return BulkResult(status: 1, message: error.localizedDescription, progress: [:])
        }
    }

    private func progressKey(_ path: String) -> String {
        canonicalPath(path)
    }

    private func loadQueue() {
        guard let data = try? Data(contentsOf: queueFile), let saved = try? JSONDecoder().decode(SavedQueue.self, from: data) else { return }
        outputFolder = URL(fileURLWithPath: saved.outputFolder)
        outputDestination = OutputDestination(rawValue: saved.outputDestination) ?? .folder
        modelID = saved.modelID
        engine = saved.engine ?? "qwen"
        additionalPasses = saved.additionalPasses ?? []
        appendModelID = saved.appendModelID ?? false
        speakerTurns = saved.speakerTurns ?? false
        zoomMeetings = saved.zoomMeetings ?? []
        removedZoomInputs = saved.removedZoomInputs ?? []
        for var session in saved.mergedSessions ?? [] {
            // The app quit mid-merge.
            if session.state == "running" { session.state = "canceled"; session.message = "Canceled" }
            mergeResults[session.id] = session
        }
        let savedDiarizationPath = saved.diarizationModel
        let isLegacySortformerAlias = savedDiarizationPath == "sortformer"
            || savedDiarizationPath == "/sortformer"
            || (savedDiarizationPath.hasSuffix("/sortformer")
                && !FileManager.default.fileExists(atPath: savedDiarizationPath))
        if isLegacySortformerAlias {
            // Preserve the alias when loading both new and older queue files;
            // passing "sortformer" through URL(fileURLWithPath:) makes it cwd-relative.
            diarizationModel = URL(fileURLWithPath: "/sortformer")
        } else {
            let savedDiarizationModel = URL(fileURLWithPath: saved.diarizationModel)
            diarizationModel = (!FileManager.default.fileExists(atPath: savedDiarizationModel.path)
                                && savedDiarizationModel.pathExtension.lowercased() == "onnx")
                ? URL(fileURLWithPath: "/sortformer")
                : savedDiarizationModel
        }
        device = saved.device
        format = saved.format
        // The format picker keeps Text untimed; enforce that for older queue files too,
        // otherwise engines disagree (the Apple helper ignores TXT timestamps).
        timestamps = saved.format == "txt" ? "none" : saved.timestamps
        diarize = saved.diarizeByDefault
        items = saved.items.compactMap { value in
            let url = URL(fileURLWithPath: value.path)
            // Keep rows whose file is missing right now (an unplugged drive, say): dropping
            // them here would erase them from queue.json on the next save. transcribe()
            // checks each file when a run starts.
            let item = AudioItem(url: url, id: value.id, duration: value.duration, diarize: value.diarize,
                                 speakerCount: value.diarize ? value.speakerCount : nil,
                                 isoTrack: value.isoTrack ?? false)
            item.zoomMeeting = value.zoomMeeting
            item.isoOverride = value.isoOverride ?? (saved.zoomMeetings == nil ? value.isoTrack : nil)
            item.zoomRecording = value.zoomRecording ?? false
            item.skipReason = value.skipReason
            item.coveredBy = value.coveredBy
            item.speakerName = value.speakerName
            item.sessionLabel = value.sessionLabel
            item.state = value.state == "done" ? .done : value.state == "processing" ? .queued : value.state == "canceled" ? .canceled : value.state == "failed" ? .failed(value.error ?? "Transcription failed") : .queued
            item.progress = value.progress
            item.selected = value.selected
            item.outputURL = value.outputPath.map(URL.init(fileURLWithPath:))
            item.modelOutputs = value.modelOutputHistory ?? (value.modelOutputs ?? [:]).map { ModelOutput(tag: $0.key, path: $0.value) }
            item.resultSummary = value.resultSummary
            return item
        }
        outcomes = (saved.passOutcomes ?? [:]).mapValues { passes in
            passes.map { outcome in
                guard outcome.phase == .running || outcome.phase == .waiting else { return outcome }
                var interrupted = outcome
                interrupted.phase = .canceled
                interrupted.message = "Interrupted before completion"
                return interrupted
            }
        }
        pruneOutcomes()
        if !supportsDiarization {
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
                return SavedItem(id: item.id, path: item.url.path, duration: item.duration, state: item.state.stateName, error: error, progress: item.progress, selected: item.selected, diarize: item.diarize, speakerCount: item.speakerCount, outputPath: item.outputURL?.path, resultSummary: item.resultSummary, isoTrack: item.isoTrack, isoOverride: item.isoOverride, speakerName: item.speakerName, sessionLabel: item.sessionLabel, zoomMeeting: item.zoomMeeting, zoomRecording: item.zoomRecording, skipReason: item.skipReason, coveredBy: item.coveredBy, modelOutputHistory: item.modelOutputs)
            },
            outputFolder: outputFolder.path,
            outputDestination: outputDestination.rawValue,
            modelID: modelID,
            diarizationModel: diarizationModelArgument,
            device: device,
            format: format,
            timestamps: timestamps,
            diarizeByDefault: diarize,
            engine: engine,
            speakerTurns: speakerTurns,
            mergedSessions: mergedSessions,
            zoomMeetings: zoomMeetings,
            removedZoomInputs: removedZoomInputs,
            additionalPasses: additionalPasses,
            appendModelID: appendModelID,
            passOutcomes: outcomes
        )
        guard let data = try? JSONEncoder().encode(saved) else { return }
        try? FileManager.default.createDirectory(at: queueFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: queueFile, options: .atomic)
    }

    private var cliURL: URL? {
        executable("transcriberrr", override: "TRANSCRIBERRR_CLI", development: ["../target/release/transcriberrr"])
    }

    private var nativeURL: URL? {
        executable("transcriberrr-native", override: "TRANSCRIBERRR_NATIVE", development: [
            ".build/out/Products/Release/transcriberrr-native", // Swift Build (`--show-bin-path`)
            ".build/release/transcriberrr-native" // classic SwiftPM layout
        ])
    }

    /// Finds a bundled executable: the environment override, then `Contents/Helpers`
    /// (where nested code belongs for signing and notarization), then development
    /// paths relative to `ui/`, so `swift run` works without a PATH lookup.
    private func executable(_ name: String, override: String, development: [String]) -> URL? {
        let fileManager = FileManager.default
        if let value = ProcessInfo.processInfo.environment[override], !value.isEmpty,
           fileManager.isExecutableFile(atPath: value) {
            return URL(fileURLWithPath: value)
        }
        let helper = Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/\(name)")
        if fileManager.isExecutableFile(atPath: helper.path) { return helper }
        // Three components up from this file is `ui/`.
        let ui = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return development
            .map { ui.appendingPathComponent($0).standardizedFileURL }
            .first { fileManager.isExecutableFile(atPath: $0.path) }
    }

    private func isLocalModelPath(_ value: String) -> Bool {
        value == "~" || value.hasPrefix("~/") || value.hasPrefix("/")
    }


}

struct PlannedTranscript: Identifiable, Equatable {
    let path: String
    let model: String
    let tag: String?
    let exists: Bool
    var id: String { "\(tag ?? "")|\(path)" }
}

struct LibraryModel: Decodable, Identifiable, Equatable {
    let id: String
    let description: String
    let cached: Bool
    let path: String?
    var isLocal: Bool { id.hasPrefix("/") || id.hasPrefix("~/") || id.hasPrefix("./") || id.hasPrefix("../") }
}

extension TranscriptionPass {
    var displayName: String {
        switch engine {
        case "parakeet-v3": "Parakeet"
        case "apple-native": "Apple SpeechTranscriber"
        default: "Qwen3 ASR"
        }
    }
}

extension AppState {
    var configuredPasses: [TranscriptionPass] {
        [TranscriptionPass(engine: engine, model: modelID)] + additionalPasses
    }

    func replaceModelPasses(_ passes: [TranscriptionPass], appendSuffix: Bool) {
        guard let first = passes.first, !isRunning, downloadingLibraryModel == nil else { return }
        engine = first.engine
        modelID = first.model
        additionalPasses = Array(passes.dropFirst())
        appendModelID = appendSuffix
        saveQueue()
    }

    func validateModelPasses(_ passes: [TranscriptionPass], appendSuffix: Bool? = nil) async throws -> [String?] {
        let tagged = passes.count > 1 || (appendSuffix ?? appendModelID)
        var tags: [String?] = []
        var used: [String: Int] = [:]
        for (index, pass) in passes.enumerated() {
            try Task.checkCancellation()
            if pass.usesApple {
                guard #available(macOS 26, *) else {
                    throw PlanningFailure(message: "Apple SpeechTranscriber requires macOS 26 or later.")
                }
            }
            let model = pass.modelArgument
            if (pass.engine == "parakeet-v3" && isQwenModel(model)) || (pass.engine == "qwen" && isParakeetModel(model)) {
                throw PlanningFailure(message: "Pass \(index + 1) needs a \(pass.displayName) model.")
            }
            guard tagged else {
                tags.append(nil)
                continue
            }
            guard let tag = await readModelTag(pass) else {
                throw PlanningFailure(message: "Couldn’t determine the model identifier for pass \(index + 1).")
            }
            if let previous = used.updateValue(index + 1, forKey: tag) {
                throw PlanningFailure(message: "Passes \(previous) and \(index + 1) both use .\(tag). Choose a different model or remove one pass.")
            }
            tags.append(tag)
        }
        return tags
    }

    func previewOutputs(for selection: QueueSelection, passes: [TranscriptionPass], tags: [String?]) async throws -> [PlannedTranscript] {
        let inspected: [AudioItem]
        switch selection {
        case .recording(let id):
            guard let item = items.first(where: { $0.id == id }) else { return [] }
            if let meeting = item.zoomMeeting { inspected = items.filter { $0.zoomMeeting == meeting } }
            else { inspected = [item] }
        case .meeting(let id):
            inspected = items.filter { $0.zoomMeeting == id }
        case .merge:
            return []
        }
        guard !inspected.isEmpty else { return [] }
        let targetIDs = Set(inspected.map(\.id))
        let targetPaths = Set(inspected.map { canonicalPath($0.url.path) })
        let includedMeetings = Set(selectedItems.compactMap(\.zoomMeeting) + inspected.compactMap(\.zoomMeeting))
        let requested = items.filter { item in
            guard item.selected || targetIDs.contains(item.id) || item.zoomMeeting.map(includedMeetings.contains) == true else { return false }
            return item.zoomMeeting == nil || item.isoTrack || item.skipReason == nil || FileManager.default.fileExists(atPath: item.url.path)
        }
        let requestedIDs = Set(requested.map(\.id))
        let excluded = itemsByPath(items).compactMap { path, item in requestedIDs.contains(item.id) ? nil : path }
        let format = self.format
        let plans = zip(passes, tags).map { pass, tag in
            (pass, tag, makeRunManifest(requested, excluding: Set(excluded).union(removedZoomInputs).sorted(), pass: pass, tag: tag))
        }
        var outputs: [PlannedTranscript] = []
        var errors: [String] = []
        for (pass, tag, manifest) in plans {
            try Task.checkCancellation()
            if tag == nil, inspected.count == 1, let item = inspected.first,
               item.zoomMeeting == nil, !item.zoomRecording, !item.isoTrack, item.skipReason == nil,
               item.state == .done, let saved = reusableUntaggedOutput(for: item) {
                outputs.append(PlannedTranscript(path: saved.path, model: pass.modelArgument, tag: nil, exists: true))
                continue
            }
            let plan = try await preview(manifest, format: format, pass: pass, tag: tag, trackProcess: false)
            try Task.checkCancellation()
            let matched = plan.files.filter { targetPaths.contains($0.path) }
            errors.append(contentsOf: matched.compactMap(\.error))
            let coveringPaths = Set(matched.flatMap { $0.coveredBy ?? [] })
            let related = plan.files.filter { targetPaths.contains($0.path) || coveringPaths.contains($0.path) }
            let meetingIDs = Set(related.compactMap(\.meeting))
            let paths = plan.meetings.filter { meetingIDs.contains($0.id) }.map(\.output)
                + related.flatMap { [$0.output, $0.mergeOutput].compactMap { $0 } }
            var seen: Set<String> = []
            for path in paths where seen.insert(path).inserted {
                outputs.append(PlannedTranscript(path: path, model: pass.modelArgument, tag: tag, exists: outputExists(path)))
            }
        }
        if outputs.isEmpty, let error = errors.first { throw PlanningFailure(message: error) }
        return outputs
    }

    func modelLibrary() async throws -> [LibraryModel] {
        guard let report = await readCLI(["--model-status"]),
              var models = try? JSONDecoder().decode([LibraryModel].self, from: Data(report.utf8)) else {
            throw PlanningFailure(message: "Couldn’t read the local model cache.")
        }
        let configured = configuredPasses.filter { !$0.usesApple }.map(\.modelArgument) + [diarizationModelArgument]
        var seen = Set(models.map(\.id) + ["sortformer"])
        let extra = configured.filter { seen.insert($0).inserted }
        if !extra.isEmpty {
            let cached = await cachedModels(extra)
            models += extra.map { LibraryModel(id: $0, description: $0, cached: cached?[$0] == true, path: nil) }
        }
        return models
    }

    func fetchLibraryModel(_ model: LibraryModel) async {
        guard !isRunning, downloadingLibraryModel == nil, !model.cached, !model.isLocal else { return }
        downloadingLibraryModel = model.id
        isCancelling = false
        beginModelActivity(title: "Downloading Model", message: model.description)
        let result = await downloadModels([model.id])
        modelActivityCanDismiss = true
        modelActivityMessage = isCancelling ? "Download canceled." : result.status == 0 ? "Model downloaded."
            : result.message.isEmpty ? "Model download failed." : result.message
        downloadingLibraryModel = nil
        isCancelling = false
    }

    func cancelLibraryDownload() {
        guard downloadingLibraryModel != nil, !isCancelling else { return }
        isCancelling = true
        modelActivityMessage = "Stopping download…"
        activeProcess?.terminate()
    }
}

func formatElapsed(_ duration: TimeInterval) -> String {
    guard duration.isFinite, duration >= 0 else { return "--:--" }
    let total = Int(duration.rounded())
    return total >= 3600
        ? String(format: "%d:%02d:%02d", total / 3600, (total % 3600) / 60, total % 60)
        : String(format: "%d:%02d", total / 60, total % 60)
}
