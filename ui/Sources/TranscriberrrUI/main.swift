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
            ContentView(state: state)
                .frame(minWidth: 780, minHeight: 640)
        }
        .defaultSize(width: 900, height: 720)
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
}

/// A session's merged transcript, reported by the CLI.
struct MergedSession: Codable, Identifiable, Equatable {
    /// The merged transcript's path.
    var id: String
    var label: String
    var state: String
    var message: String
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
        var error: String?

        enum CodingKeys: String, CodingKey {
            case path, meeting, speaker, output, iso, diarize, done, skip, error
            case zoomRecording = "zoom_recording"
            case coveredBy = "covered_by"
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

@MainActor
final class AppState: ObservableObject {
    @Published var items: [AudioItem] = []
    @Published var outputFolder = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Transcripts")
    @Published var outputDestination: OutputDestination = .folder
    @Published var modelID = "Qwen/Qwen3-ASR-1.7B"
    /// `qwen` uses the Rust Metal pipeline; `apple-native` uses the bundled
    /// SpeechTranscriber helper on macOS 26 and newer.
    @Published var engine = "qwen"
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
    @Published var modelActivityTitle = "Preparing models"
    @Published var modelActivityMessage = "Checking the local model cache…"
    @Published var modelActivityLog: [String] = []
    @Published var modelActivityProgress: Double?
    @Published var modelActivityCanDismiss = false
    @Published var modelActivitySucceeded = false
    @Published var isCancelling = false
    @Published private var mergeResults: [String: MergedSession] = [:]
    var mergedSessions: [MergedSession] { mergeResults.values.sorted { $0.id < $1.id } }
    @Published var zoomMeetings: [ZoomMeeting] = []
    private var removedZoomInputs: Set<String> = []
    @Published var planningError: String?
    @Published private(set) var addOperations = 0

    private var pendingSave: Task<Void, Never>?
    @Published private(set) var modelStatus = "Checking the local model cache…"
    private var modelStatusTask: Task<Void, Never>?
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
        invalidateModelStatus()
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
    var selectedCount: Int {
        let selected = selectedItems
        return selected.filter { $0.zoomMeeting == nil }.count + Set(selected.compactMap(\.zoomMeeting)).count
    }
    var completedItemCount: Int {
        let finished = Set(zoomMeetings.filter { meetingResult($0)?.state == "done" }.map(\.id))
        return items.filter { $0.zoomMeeting.map(finished.contains) ?? ($0.state == .done) }.count
    }
    var completedCount: Int {
        let selected = selectedItems
        let groups = Set(selected.compactMap(\.zoomMeeting))
        return selected.filter { $0.zoomMeeting == nil && $0.state == .done }.count
            + zoomMeetings.filter { groups.contains($0.id) && meetingResult($0)?.state == "done" }.count
    }
    var totalDuration: TimeInterval { selectedItems.reduce(0) { $0 + $1.duration } }
    var activeStatus: String {
        if isCancelling { return "Stopping safely…" }
        return items.first(where: { $0.state == .processing })?.resultSummary ?? "Preparing…"
    }
    /// Items in the current run. Progress and ETA cover only these: counting tracks finished
    /// in earlier runs as done made the estimate start near zero.
    private var runItemIDs: Set<UUID> = []

    var queueProgress: Double {
        let tracked = isRunning && !runItemIDs.isEmpty
            ? items.filter { runItemIDs.contains($0.id) }
            : selectedItems
        guard !tracked.isEmpty else { return 0 }
        return tracked.reduce(0) { total, item in
            if item.state == .done { return total + 1 }
            return total + min(max(item.progress, 0), 1)
        } / Double(tracked.count)
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

    func selectEngine(_ value: String) {
        if value == "parakeet-v3" && isQwenModel(modelID) {
            modelID = "eschmidbauer/parakeet-unified-en-0.6b-onnx"
        } else if value == "qwen" && isParakeetModel(modelID) {
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
        invalidateModelStatus()
    }

    /// Re-query the CLI for the selected model's cache state. Called on launch
    /// and whenever the engine or model changes.
    func invalidateModelStatus() {
        modelStatusTask?.cancel()
        guard engine != "apple-native" else {
            modelStatus = "Apple SpeechTranscriber (system model; macOS 26+)"
            return
        }
        let model = effectiveModelID
        modelStatus = "Checking the local model cache…"
        modelStatusTask = Task { @MainActor [weak self] in
            // Debounce typing in the model field: one CLI query after a pause, not per key.
            try? await Task.sleep(for: .milliseconds(400))
            guard let self, !Task.isCancelled else { return }
            let cached = await self.cachedModels([model])?[model]
            guard !Task.isCancelled else { return }
            let local = self.isLocalModelPath(model)
            self.modelStatus = switch cached {
            case true?: local ? "Local model ready" : "Downloaded"
            case false?: local ? "Local model not found" : "Downloads on first use"
            case nil: "Model status unavailable"
            }
        }
    }

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
    private func preview(_ manifest: JobManifest, format: String, inventoryOnly: Bool = false) async throws -> QueuePlan {
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
        process.arguments = ["--manifest", url.path, "--format", format, "--plan"]
        if inventoryOnly { process.arguments?.append("--inventory-only") }
        process.standardOutput = stdout.pipe
        process.standardError = stderr.pipe
        let result: (Int32, String, String) = try await withCheckedThrowingContinuation { continuation in
            process.terminationHandler = { process in
                if !inventoryOnly {
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
            if inventoryOnly {
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
    private func applyPlan(_ plan: QueuePlan, selectingMeetings: Bool = false) -> [AudioItem] {
        var known = itemsByPath(items)
        let unchecked = Set(items.filter { !$0.selected }.compactMap(\.zoomMeeting))
        let updates = Dictionary(uniqueKeysWithValues: plan.meetings.map { ($0.id, $0) })
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
            if let update = updates[old.id], update.output != old.output {
                mergeResults.removeValue(forKey: old.output)
            }
        }
        for meeting in plan.meetings {
            if let result = mergeResults[meeting.output], result.state != "done" || !meeting.done {
                mergeResults.removeValue(forKey: meeting.output)
            }
        }
        zoomMeetings.removeAll { updates[$0.id] != nil }
        zoomMeetings.append(contentsOf: plan.meetings)

        var added: [URL] = []
        let planned = plan.files.map { file in
            let item: AudioItem
            if let previous = known[file.path] {
                item = previous
            } else {
                let url = URL(fileURLWithPath: file.path)
                item = AudioItem(url: url, diarize: file.diarize ?? diarize, isoTrack: file.iso ?? false)
                items.append(item)
                known[file.path] = item
            }
            if file.error == nil {
                if item.duration == 0 && file.skip == nil { added.append(item.url) }
                item.zoomMeeting = file.meeting
                if let meeting = file.meeting { item.selected = selectingMeetings || !unchecked.contains(meeting) }
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
                if selectingMeetings, let diarize = file.diarize { item.diarize = diarize }
            }
            if let error = file.error {
                item.state = .failed(error)
                item.progress = 0
                item.resultSummary = error
            } else if let skip = file.skip {
                let covered = file.coveredBy?.contains(where: completed.contains) == true
                    || hasSelectedCover(item, in: known)
                item.state = covered ? .done : .queued
                item.progress = covered ? 1 : 0
                item.outputURL = nil
                item.resultSummary = skip
            } else if file.done == true {
                item.state = .done
                item.progress = 1
                item.outputURL = file.output.map(URL.init(fileURLWithPath:))
                item.resultSummary = "Transcript already exists"
            } else if file.meeting != nil || file.zoomRecording == true {
                item.state = .queued
                item.progress = 0
                item.outputURL = nil
                item.resultSummary = ""
            }
            return item
        }
        let represented = Set(items.compactMap(\.zoomMeeting))
        zoomMeetings.removeAll { !represented.contains($0.id) }
        objectWillChange.send()
        saveQueue()
        loadDurations(for: added)
        return planned
    }

    private func itemsByPath(_ items: [AudioItem]) -> [String: AudioItem] {
        Dictionary(items.map { (canonicalPath($0.url.path), $0) }, uniquingKeysWith: { first, _ in first })
    }

    private func hasSelectedCover(_ item: AudioItem, in known: [String: AudioItem]) -> Bool {
        var paths = item.coveredBy ?? []
        var visited: Set<String> = []
        while let path = paths.popLast() {
            guard visited.insert(path).inserted, let cover = known[path], cover.selected else { continue }
            if cover.state == .done, let output = cover.outputURL, outputExists(output.path) { return true }
            paths.append(contentsOf: cover.coveredBy ?? [])
        }
        return false
    }

    private func outputExists(_ path: String) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &directory) && !directory.boolValue
    }

    private func reconcileSkippedRows(_ tracked: [AudioItem]) {
        let known = itemsByPath(items)
        for item in tracked where item.state == .done && item.zoomMeeting == nil {
            if let reason = item.skipReason, item.outputURL == nil, !hasSelectedCover(item, in: known) {
                item.state = .queued
                item.progress = 0
                item.resultSummary = reason
            }
        }
    }

    func remove(_ item: AudioItem) {
        if item.zoomRecording { removedZoomInputs.insert(canonicalPath(item.url.path)) }
        items.removeAll { $0.id == item.id }
        saveQueue()
    }

    func removeMeeting(_ meeting: ZoomMeeting) {
        items.removeAll { $0.zoomMeeting == meeting.id }
        zoomMeetings.removeAll { $0.id == meeting.id }
        mergeResults.removeValue(forKey: meeting.output)
        saveQueue()
    }

    func meetingResult(_ meeting: ZoomMeeting) -> MergedSession? {
        mergeResults[meeting.output]
            ?? (meeting.done ? MergedSession(id: meeting.output, label: meeting.label, state: "done",
                                              message: "Transcript already exists") : nil)
    }

    func clearCompleted() {
        let completed = Set(zoomMeetings.filter { meetingResult($0)?.state == "done" }.map(\.id))
        items.removeAll { item in
            item.zoomMeeting.map(completed.contains) ?? (item.state == .done)
        }
        zoomMeetings.removeAll { completed.contains($0.id) }
        mergeResults = mergeResults.filter { $0.value.state != "done" }
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

    func revealOutput(for item: AudioItem) {
        guard let outputURL = item.outputURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([outputURL])
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
        guard isRunning else { return }
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
        guard !isRunning, addOperations == 0 else { return }
        let known = itemsByPath(items)
        let selectedMeetings = Set(selectedItems.compactMap(\.zoomMeeting))
        let selected = items.filter { $0.selected || $0.zoomMeeting.map(selectedMeetings.contains) == true }
        let coveredSkips = selected.filter {
            $0.zoomMeeting == nil && !$0.zoomRecording && $0.skipReason != nil && hasSelectedCover($0, in: known)
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
        var requested = selected.filter {
            !coveredIDs.contains($0.id)
                && ($0.state != .done || $0.zoomMeeting != nil || $0.zoomRecording || $0.skipReason != nil)
        }
        guard !requested.isEmpty else { return }
        let sessions = Set(requested.compactMap(sessionKey))
        let requestedIDs = Set(requested.map(\.id))
        requested += selected.filter {
            !requestedIDs.contains($0.id) && $0.skipReason == nil && sessionKey($0).map(sessions.contains) == true
        }
        let selectedIDs = Set(selected.map(\.id))
        let excluded = Set(known.compactMap { path, item in selectedIDs.contains(item.id) ? nil : path })
            .union(removedZoomInputs)
        let manifest = makeRunManifest(requested, excluding: excluded.sorted())
        isRunning = true
        isCancelling = false
        startQueueClock()
        Task { @MainActor in
            let pending: [AudioItem]
            let tracked: [AudioItem]
            let outputs: [UUID: URL]
            let frozen: JobManifest
            do {
                let plan = try await preview(manifest, format: format)
                if isCancelling { throw CancellationError() }
                let planned = applyPlan(plan, selectingMeetings: true)
                let accepted = zip(planned, plan.files).filter { $0.1.error == nil }
                tracked = accepted.map(\.0)
                pending = accepted.compactMap { item, file in
                    file.done == true || file.skip != nil ? nil : item
                }
                outputs = Dictionary(accepted.compactMap { item, file in
                    file.output.map { (item.id, URL(fileURLWithPath: $0)) }
                }, uniquingKeysWith: { first, _ in first })
                let original = Dictionary(manifest.items.enumerated().map { index, item in
                    (canonicalPath(item.input), (index, item.output))
                }, uniquingKeysWith: { first, _ in first })
                // Include discovered tracks and skipped continuations explicitly. If a
                // track disappears during model setup, the CLI can reject the whole meeting.
                let inputs = accepted.enumerated().map { index, entry in
                    let (item, file) = entry
                    let (order, output) = original[file.path] ?? (manifest.items.count + index, nil)
                    return (order, manifestItem(for: item, input: file.path, output: file.output ?? output))
                }.sorted { $0.0 < $1.0 }.map(\.1)
                frozen = JobManifest(items: inputs, discoverZoom: false)
            } catch {
                if !isCancelling { planningError = error.localizedDescription }
                finishQueueClock()
                isRunning = false
                isCancelling = false
                return
            }
            runItemIDs = Set(pending.map(\.id))
            guard !pending.isEmpty, !isCancelling else {
                if isCancelling { markCanceled(pending) }
                finishQueueClock()
                isRunning = false
                isCancelling = false
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
            objectWillChange.send()
            saveQueue()

            let modelDownloads = await requiredModelDownloads(for: pending)
            if !modelDownloads.isEmpty, !isCancelling {
                beginModelActivity(title: "Preparing models", message: "Checking and downloading model files…")
                modelActivityLog = modelDownloads.map { "Model: \($0)" }
                let result = await downloadModels(modelDownloads)
                if result.status != 0 {
                    if isCancelling {
                        markCanceled(pending)
                    } else {
                        pending.forEach {
                            $0.state = .failed("Model setup failed")
                            $0.resultSummary = "Model setup failed; see the model activity window."
                        }
                    }
                    modelActivityMessage = isCancelling
                        ? "Model download canceled."
                        : result.message.isEmpty ? "Model setup failed." : result.message
                    modelActivityCanDismiss = true
                    modelActivitySucceeded = false
                    finishQueueClock()
                    isRunning = false
                    isCancelling = false
                    saveQueue()
                    return
                }
                invalidateModelStatus()
                // Get out of the way of the run's progress once the models are ready.
                showModelActivity = false
            }

            if let first = pending.first {
                first.state = .processing
                first.progress = 0.05
                first.resultSummary = "Starting transcription…"
                objectWillChange.send()
                saveQueue()
            }

            // Qwen and Parakeet keep their weights in process memory. Run the
            // whole queue through one manifest so a 5,000-track queue does not
            // repeatedly pay model startup and Metal pipeline compilation. The
            // Apple engine goes through the CLI too, which converts formats
            // AVFoundation cannot open (WebM) and validates the transcript.
            if !pending.isEmpty, !isCancelling {
                let result = await runBulkCLI(frozen, for: tracked, outputs: outputs)
                if isCancelling {
                    applyFinished(result, to: tracked)
                    markCanceled(pending)
                } else {
                    applyBulk(result, to: tracked)
                }
                reconcileSkippedRows(tracked)
                objectWillChange.send()
                saveQueue()
            }
            if isCancelling { markCanceled(pending) }
            finishQueueClock()
            isRunning = false
            isCancelling = false
            activeProcess = nil
            saveQueue()
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

    /// The model the CLI will load: with --engine parakeet-v3 it replaces any Qwen
    /// model with the Parakeet default (main.rs).
    private var effectiveModelID: String {
        engine == "parakeet-v3" && isQwenModel(modelID)
            ? "eschmidbauer/parakeet-unified-en-0.6b-onnx" : modelID
    }

    private func requiredModelDownloads(for pending: [AudioItem]) async -> [String] {
        guard engine != "apple-native" else { return [] }
        var candidates: [String] = []
        func append(_ value: String) {
            // Local paths cannot be downloaded; the CLI reports a missing one itself.
            guard !value.isEmpty, !isLocalModelPath(value), !candidates.contains(value) else { return }
            candidates.append(value)
        }
        append(effectiveModelID)
        let diarizing = pending.contains(where: { $0.diarize || $0.speakerCount != nil })
        // The CLI loads the aligner for diarization too: speakers attach to word timings.
        // Subtitles always get sentence timing (main.rs), which needs the aligner too, and
        // merging speaker tracks needs every track's word timings.
        let merging = pending.contains(where: \.isoTrack)
        if engine == "qwen" && (timestamps != "none" || diarizing || merging || format == "srt" || format == "vtt") {
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
        guard let executableURL = cliURL else { return nil }
        let process = Process()
        process.executableURL = executableURL
        process.arguments = ids.flatMap { ["--cached-model", $0] }
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
        struct Entry: Decodable { var id: String; var cached: Bool }
        guard let result, result.0 == 0,
              let entries = try? JSONDecoder().decode([Entry].self, from: Data(result.1.utf8)) else { return nil }
        return Dictionary(entries.map { ($0.id, $0.cached) }, uniquingKeysWith: { first, _ in first })
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
        modelActivitySucceeded = false
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
            item.progress = event.progress ?? max(item.progress, 0.05)
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
                                    state: state, message: message)
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
        stopRunningMerges()
        // stderr holds a line per finished track; the last line is the run's error.
        let fallback = result.message.split(whereSeparator: \.isNewline).last
            .map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
        // The CLI was stopped from outside the app (Activity Monitor, `kill`): tracks it never
        // reached were canceled, not failed.
        // Before its first job, the CLI answers SIGTERM/SIGINT by exiting 143/130 at once,
        // with no event written.
        let canceled = result.progress.values.contains { $0.status == "canceled" }
            || result.status == 143 || result.status == 130
        for item in items {
            let event = result.progress[progressKey(item.url.path)]
            if let event, Self.finalStatuses.contains(event.status) {
                apply(event, to: item)
            } else if canceled {
                item.estimatedRemainingSeconds = nil
                item.state = .canceled
                item.progress = 0
                item.resultSummary = "Canceled"
            } else if result.status == 0, let output = result.outputs[item.id] {
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

    /// Picks an output path that no other row uses and that does not replace an existing
    /// file such as notes.txt beside notes.m4a.
    private func uniqueOutput(_ candidate: URL, reserved: inout Set<String>) -> URL {
        func available(_ url: URL) -> Bool {
            let key = url.standardizedFileURL.path.lowercased()
            guard !reserved.contains(key) else { return false }
            guard !FileManager.default.fileExists(atPath: url.path) else { return false }
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

    private func manifestItem(for item: AudioItem, input: String, output: String?) -> ManifestItem {
        ManifestItem(input: input, diarize: item.diarize, speakers: item.speakerCount, output: output,
                     outputDir: outputFolder.path, outputLocation: cliOutputLocation, isoTrack: item.isoOverride)
    }

    private func makeRunManifest(_ requested: [AudioItem], excluding: [String]) -> JobManifest {
        var reserved = Set(items.compactMap { $0.outputURL?.standardizedFileURL.path.lowercased() })
        return JobManifest(items: requested.map { item in
            let output = item.zoomMeeting == nil && !item.zoomRecording
                ? uniqueOutput(expectedOutput(for: item), reserved: &reserved).path : nil
            return manifestItem(for: item, input: item.url.standardizedFileURL.path, output: output)
        }, excludedInputs: excluding.isEmpty ? nil : excluding)
    }

    private func runBulkCLI(_ manifest: JobManifest, for items: [AudioItem], outputs: [UUID: URL]) async -> BulkResult {
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
                             "--engine", cliEngine,
                             "--model", modelID, "--device", device, "--format", format,
                             "--timestamps", timestamps, "--progress-file", progressURL.path, "--resume"]
            if format == "txt" && speakerTurns {
                arguments.append("--speaker-turns")
            }
            if items.contains(where: { $0.diarize || $0.speakerCount != nil }) {
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
            let qwenEngine = engine == "qwen"
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

    private var cliEngine: String {
        switch engine {
        case "parakeet-v3": "parakeet-v3"
        case "apple-native": "apple-native"
        default: "qwen3"
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
                return SavedItem(id: item.id, path: item.url.path, duration: item.duration, state: item.state.stateName, error: error, progress: item.progress, selected: item.selected, diarize: item.diarize, speakerCount: item.speakerCount, outputPath: item.outputURL?.path, resultSummary: item.resultSummary, isoTrack: item.isoTrack, isoOverride: item.isoOverride, speakerName: item.speakerName, sessionLabel: item.sessionLabel, zoomMeeting: item.zoomMeeting, zoomRecording: item.zoomRecording, skipReason: item.skipReason, coveredBy: item.coveredBy)
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
            removedZoomInputs: removedZoomInputs
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

private struct ModelActivitySheet: View {
    @ObservedObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                Image(systemName: state.modelActivitySucceeded ? "checkmark.circle.fill" : (state.modelActivityCanDismiss ? "exclamationmark.triangle.fill" : "arrow.down.circle.fill"))
                    .font(.system(size: 28))
                    .foregroundStyle(state.modelActivitySucceeded ? .green : (state.modelActivityCanDismiss ? .orange : Color.accentColor))
                VStack(alignment: .leading, spacing: 3) {
                    Text(state.modelActivityTitle).font(.title3.weight(.semibold))
                    Text(state.modelActivityMessage).font(.callout).foregroundStyle(.secondary).lineLimit(2)
                }
                Spacer()
            }
            if let progress = state.modelActivityProgress {
                ProgressView(value: progress)
                    .tint(.accentColor)
            } else {
                ProgressView().controlSize(.small)
            }
            Text("Models are stored in the local Transcriberrr cache. This may take several minutes on a first run.")
                .font(.caption)
                .foregroundStyle(.secondary)
            ScrollViewReader { proxy in
                ScrollView {
                    Text(state.modelActivityLog.isEmpty ? "Waiting for the model manager…" : state.modelActivityLog.joined(separator: "\n"))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .id("model-log")
                }
                .frame(minHeight: 180, maxHeight: 300)
                .padding(10)
                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                .onChange(of: state.modelActivityLog.count) { _, _ in
                    withAnimation { proxy.scrollTo("model-log", anchor: .bottom) }
                }
            }
            HStack {
                Spacer()
                if state.isRunning {
                    // No Esc shortcut: pressing Esc to dismiss the panel canceled the run.
                    Button(state.isCancelling ? "Stopping…" : "Cancel") {
                        state.cancelTranscription()
                    }
                    .disabled(state.isCancelling)
                }
                if state.modelActivityCanDismiss {
                    Button("Close") {
                        state.showModelActivity = false
                    }
                    .keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(24)
        .frame(width: 620, height: 460)
    }
}

struct ContentView: View {
    @ObservedObject var state: AppState
    @State private var dropTargeted = false
    @State private var queueFilter = ""
    @State private var compiledFilter: NSRegularExpression?
    @State private var filterErrorMessage: String?
    @State private var expandedMeetings: Set<String> = []

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
        .alert("Couldn’t plan the queue", isPresented: Binding(
            get: { state.planningError != nil },
            set: { if !$0 { state.planningError = nil } }
        )) {
            Button("OK") { state.planningError = nil }
        } message: {
            Text(state.planningError ?? "")
        }
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
            // Refuse (so Finder animates the files back) rather than silently drop them.
            guard !state.isRunning else { return false }
            loadDroppedItems(providers)
            return true
        }
        .onChange(of: state.outputDestination) { _, _ in state.saveQueue() }
        .onChange(of: state.outputFolder) { _, _ in state.saveQueue() }
        .onChange(of: state.speakerTurns) { _, _ in state.saveQueue() }
        .onChange(of: state.engine) { _, value in state.selectEngine(value); state.saveQueue() }
        .onChange(of: state.modelID) { _, _ in state.invalidateModelStatus(); state.saveQueue() }
        .onChange(of: state.diarizationModel) { _, _ in state.saveQueue() }
        .onChange(of: state.device) { _, _ in state.saveQueue() }
        .onChange(of: state.format) { _, _ in state.saveQueue() }
        .onChange(of: state.timestamps) { _, _ in state.saveQueue() }
        .onAppear { updateFilter(queueFilter) }
        .onChange(of: queueFilter) { _, value in updateFilter(value) }
        // An overlay rather than a sheet: macOS refuses to quit (Cmd-Q, Dock, logout) while a
        // sheet is open, which blocked quitting for an entire first-run model download.
        .overlay {
            if state.showModelActivity {
                ZStack {
                    Color.black.opacity(0.25).ignoresSafeArea()
                    ModelActivitySheet(state: state)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                        .shadow(radius: 20)
                }
            }
        }
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
        Group {
            if state.items.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: dropTargeted ? "arrow.down.circle.fill" : "arrow.down.doc")
                        .font(.system(size: 32))
                        .foregroundStyle(dropTargeted ? Color.accentColor : Color.secondary)
                    Text(dropTargeted ? "Release to add audio" : "Drop audio files or folders here")
                        .font(.headline)
                    Text("WAV, AIFF, CAF, M4A, M4B, MP3, MP4, FLAC, AAC, OGG, Opus, or WebM")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            } else {
                Label(dropTargeted ? "Release to add audio" : "Drop more files or folders",
                      systemImage: dropTargeted ? "arrow.down.circle.fill" : "arrow.down.doc")
                    .font(.callout)
                    .foregroundStyle(dropTargeted ? Color.accentColor : Color.secondary)
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: state.items.isEmpty ? 116 : 44)
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
        let visible = visibleItems
        let visibleMeetings = Set(visible.compactMap(\.zoomMeeting))
        let children = Dictionary(grouping: state.items.filter { $0.zoomMeeting != nil }, by: { $0.zoomMeeting ?? "" })
        let meetingOutputs = Set(state.zoomMeetings.map(\.output))
        let otherMerges = state.mergedSessions.filter { !meetingOutputs.contains($0.id) }
        return VStack(spacing: 0) {
            HStack {
                Text("Queue").font(.headline)
                let count = state.zoomMeetings.count
                if count > 0 {
                    Text("\(count) \(count == 1 ? "meeting" : "meetings")").foregroundStyle(.secondary)
                }
                if count == 0 || !queueFilter.isEmpty {
                    Text("\(visible.count) \(visible.count == 1 ? "file" : "files")").foregroundStyle(.secondary)
                }
                Spacer()
                TextField("Regex filter", text: $queueFilter)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 180)
                // Selection drives queue progress and ETA; changing it mid-run skews both.
                Button("Check all") { state.selectAll(visibleItems, selected: true) }
                    .buttonStyle(.borderless)
                    .disabled(state.isRunning)
                Button("Check none") { state.selectAll(visibleItems, selected: false) }
                    .buttonStyle(.borderless)
                    .disabled(state.isRunning)
                Button("Speaker tracks") { state.toggleIsoTracks(visibleItems) }
                    .buttonStyle(.borderless)
                    .disabled(state.isRunning || !visible.contains { $0.zoomMeeting == nil })
                    .help("Mark the visible files as one speaker's track each (or unmark them). Tracks of one session are merged.")
                Button("Clear completed") { state.clearCompleted() }
                    .buttonStyle(.borderless)
                .disabled((state.completedItemCount == 0 && !state.mergedSessions.contains { $0.state == "done" }) || state.isRunning)
            }
            .padding(.horizontal, 20)
            .padding(.top, 18)
            if let regexError = filterError {
                Text("Invalid regex: \(regexError)").font(.caption).foregroundStyle(.red).padding(.horizontal, 20).padding(.top, 4)
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(state.zoomMeetings.filter { visibleMeetings.contains($0.id) }) { meeting in
                        meetingRow(meeting, files: children[meeting.id] ?? [])
                    }
                    ForEach(visible.filter { $0.zoomMeeting == nil }) { item in queueRow(item) }
                    if !otherMerges.isEmpty {
                        Text("Merged transcripts").font(.headline).padding(.top, 14).padding(.bottom, 4)
                        ForEach(otherMerges) { session in mergedRow(session) }
                    }
                }
                .padding(.horizontal, 20)
            }
        }
        .frame(maxHeight: .infinity)
    }

    private func meetingRow(_ meeting: ZoomMeeting, files: [AudioItem]) -> some View {
        let tracks = files.filter { $0.skipReason == nil }
        let result = state.meetingResult(meeting)
        let running = result?.state == "running" || tracks.contains { $0.state == .processing }
        let failed = tracks.first { if case .failed = $0.state { return true }; return false }
        let canceled = tracks.contains { $0.state == .canceled }
        let included = files.contains(where: \.selected)
        let progress = tracks.reduce(0.0) { $0 + $1.progress } / Double(max(1, tracks.count))
        let status = result?.state == "done" ? "Merged"
            : result?.state == "failed" || failed != nil ? "Failed"
            : result?.state == "canceled" || canceled ? "Canceled"
            : running ? "Transcribing" : "Queued"
        let participants = meeting.participants == 1 ? "1 participant’s audio"
            : "\(meeting.participants) participants"
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 10) {
                Button {
                    state.selectAll(files, selected: !included)
                } label: {
                    Image(systemName: included ? "checkmark.square.fill" : "square")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Include \(meeting.label) in transcription")
                .accessibilityValue(included ? "On" : "Off")
                .disabled(state.isRunning)
                DisclosureGroup(isExpanded: Binding(
                    get: { expandedMeetings.contains(meeting.id) },
                    set: { expanded in
                        if expanded { expandedMeetings.insert(meeting.id) }
                        else { expandedMeetings.remove(meeting.id) }
                    }
                )) {
                    ForEach(files) { item in
                        HStack(alignment: .firstTextBaseline) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.speakerName ?? item.url.lastPathComponent)
                                if item.speakerName != nil {
                                    Text(item.url.lastPathComponent).font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                            Spacer()
                            Text(item.skipReason ?? (item.resultSummary.isEmpty ? item.state.label : item.resultSummary))
                                .foregroundStyle(.secondary)
                        }
                        .font(.caption)
                        .padding(.vertical, 3)
                    }
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Label(meeting.label, systemImage: "video")
                            .font(.headline).lineLimit(1)
                        Text("\(participants) • \(tracks.count) \(tracks.count == 1 ? "track" : "tracks")")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                if running {
                    ProgressView(value: min(progress, 0.99)).frame(width: 72)
                        .accessibilityLabel("Progress for \(meeting.label)")
                }
                Text(status).font(.caption).foregroundStyle(status == "Merged" ? .green : status == "Failed" ? .red : .secondary)
                    .frame(minWidth: 82, alignment: .trailing)
                Button { state.removeMeeting(meeting) } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Remove \(meeting.label) from queue")
                .disabled(state.isRunning)
            }
            if let result, result.state == "done" {
                Button { state.revealMerged(result) } label: {
                    Text(URL(fileURLWithPath: result.id).lastPathComponent).lineLimit(1)
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.tint)
                .accessibilityLabel("Show merged transcript in Finder")
                .help(result.id)
            }
            if let message = result?.message ?? failed?.resultSummary {
                Text(message).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Divider()
        }
        .padding(.vertical, 7)
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
                Image(systemName: item.diarize ? "person.2.wave.2" : item.isoTrack ? "person.wave.2" : "waveform")
                    .foregroundStyle(.tint).frame(width: 22)
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.url.lastPathComponent).lineLimit(1)
                    Text("\(formatDuration(item.duration))  •  \(item.url.deletingLastPathComponent().path)")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    if item.state == .processing || item.elapsedSeconds > 0 {
                        Text(itemTimingLabel(item))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    if let outputURL = item.outputURL {
                        Button {
                            state.revealOutput(for: item)
                        } label: {
                            Text(item.resultSummary.isEmpty ? outputURL.path : item.resultSummary)
                                .font(.caption2)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.tint)
                        .accessibilityLabel("Show saved transcript in Finder")
                        .help("Reveal \(outputURL.path) in Finder")
                    } else if !item.resultSummary.isEmpty {
                        Text(item.resultSummary).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    }
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
                    // SpeechTranscriber has no speaker labels; enabling this only fails the item.
                    .disabled(state.isRunning || state.engine == "apple-native")
                if item.diarize && state.engine != "apple-native" {
                    Picker("Speakers", selection: Binding<Int>(get: { item.speakerCount ?? 0 }, set: { state.setSpeakerCount(for: item, $0 == 0 ? nil : $0) })) {
                        Text("Auto").tag(0)
                        ForEach(1...8, id: \.self) { count in Text("\(count)").tag(count) }
                    }
                    .pickerStyle(.menu).font(.caption).frame(width: 140)
                    .disabled(state.isRunning)
                }
                Button {
                    state.setIsoTrack(for: item, enabled: !item.isoTrack)
                } label: {
                    Label("Speaker track", systemImage: item.isoTrack ? "checkmark.square.fill" : "square")
                }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .foregroundStyle(item.isoTrack ? .primary : .secondary)
                    .accessibilityLabel("One speaker's track, merged with its session")
                    .accessibilityValue(item.isoTrack ? "On" : "Off")
                    .help("This file holds one speaker. Tracks from one Zoom session or folder are transcribed together with the other speakers' bleed muted, then merged.")
                    .disabled(state.isRunning)
                if item.isoTrack, let speaker = item.speakerName {
                    Text(item.sessionLabel.map { "\(speaker) • \($0)" } ?? speaker)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
            }
            Divider()
        }
        .padding(.vertical, 7)
    }

    @ViewBuilder
    private func mergedRow(_ session: MergedSession) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Image(systemName: "rectangle.stack.badge.person.crop").foregroundStyle(.tint).frame(width: 22)
                VStack(alignment: .leading, spacing: 2) {
                    Text(session.label).lineLimit(1)
                    if session.state == "done" {
                        Button {
                            state.revealMerged(session)
                        } label: {
                            Text(session.id).font(.caption2).lineLimit(1).truncationMode(.middle)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.tint)
                        .accessibilityLabel("Show merged transcript in Finder")
                        .help("Reveal \(session.id) in Finder")
                    }
                    Text(session.message).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
                }
                Spacer(minLength: 8)
                if session.state == "running" { ProgressView().controlSize(.small) }
                Text(session.state == "done" ? "Merged" : session.state == "failed" ? "Failed" : session.state == "canceled" ? "Canceled" : "Merging")
                    .font(.caption)
                    .foregroundStyle(session.state == "done" ? .green : session.state == "failed" ? .red : .secondary)
                    .frame(minWidth: 82, alignment: .trailing)
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
                            Text("Parakeet Unified EN (latest)").tag("parakeet-v3")
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
                    // Text is always untimed; speaker turns only change diarized tracks.
                    if state.format == "txt" {
                        Toggle("Speaker turns", isOn: $state.speakerTurns)
                            .toggleStyle(.checkbox)
                            .help("Start a [Speaker N] paragraph at each speaker change in diarized tracks.")
                            .disabled(state.engine == "apple-native")
                    }
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
                        .disabled(state.engine == "apple-native")
                    Spacer()
                }
            }
            .disabled(state.isRunning)
            // Also when only individual tracks diarize: they use this model too.
            if state.diarize || state.items.contains(where: \.diarize) {
                HStack {
                    Text("Diarization model").foregroundStyle(.secondary)
                    Text(state.diarizationModelArgument).lineLimit(1).foregroundStyle(.secondary)
                    Button("Choose…") { state.chooseDiarizationModel(state.diarizationModel) { state.diarizationModel = $0; state.saveQueue() } }
                        .disabled(state.isRunning)
                }
                .font(.caption)
            }
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(state.completedCount) of \(state.selectedCount) selected complete")
                    Text("\(formatDuration(state.totalDuration)) queued")
                        .font(.caption).foregroundStyle(.secondary)
                    if state.isRunning || state.queueElapsedSeconds > 0 {
                        HStack(spacing: 6) {
                            Text("\(formatElapsed(state.queueElapsedSeconds)) elapsed")
                            if let remaining = state.queueEstimatedRemainingSeconds {
                                Text("• ~\(formatElapsed(remaining)) remaining")
                            }
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if state.isRunning {
                    HStack(spacing: 8) {
                        ProgressView(value: state.queueProgress).frame(width: 130)
                        Text(state.activeStatus).font(.caption).foregroundStyle(.secondary)
                    }
                }
                Button(state.isRunning ? (state.isCancelling ? "Stopping…" : "Cancel") : "Transcribe") {
                    if state.isRunning { state.cancelTranscription() } else { state.transcribe() }
                }
                // Not Return/Esc: text fields (model ID, regex filter) pass those through, so
                // confirming a field started the queue and Esc in the filter canceled a run.
                .keyboardShortcut(state.isRunning ? "." : .return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .disabled(state.items.isEmpty || state.isCancelling || state.addOperations > 0)
            }
            if state.timestamps != "none" {
                Text(state.timestamps == "words"
                     ? "JSON stores each word with start and end seconds; SRT and VTT write one cue per word."
                     : "JSON, SRT, and VTT store one timed entry per sentence.")
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

    private func chooseFiles() { state.chooseFiles() }

    private func color(for itemState: ItemState) -> Color {
        switch itemState { case .done: .green; case .failed: .red; default: .secondary }
    }
}

private func formatDuration(_ duration: TimeInterval) -> String {
    guard duration.isFinite, duration > 0 else { return "Unknown duration" }
    let total = Int(duration.rounded())
    return total >= 3600 ? String(format: "%d:%02d:%02d", total / 3600, (total % 3600) / 60, total % 60) : String(format: "%d:%02d", total / 60, total % 60)
}

private func formatElapsed(_ duration: TimeInterval) -> String {
    guard duration.isFinite, duration >= 0 else { return "--:--" }
    let total = Int(duration.rounded())
    return total >= 3600
        ? String(format: "%d:%02d:%02d", total / 3600, (total % 3600) / 60, total % 60)
        : String(format: "%d:%02d", total / 60, total % 60)
}

@MainActor
private func itemTimingLabel(_ item: AudioItem) -> String {
    var label = "\(formatElapsed(item.elapsedSeconds)) elapsed"
    if let remaining = item.estimatedRemainingSeconds {
        label += " • ~\(formatElapsed(remaining)) remaining"
    }
    return label
}
