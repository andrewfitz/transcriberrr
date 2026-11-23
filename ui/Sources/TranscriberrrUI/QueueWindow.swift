import AppKit
import SwiftUI

/// Run actions the focused window publishes to the menu bar.
struct QueueActions {
    var transcribe: (() -> Void)?
    var cancel: (() -> Void)?
    var showRunSettings: () -> Void
    var showModelLibrary: () -> Void
}

private struct QueueActionsKey: FocusedValueKey {
    typealias Value = QueueActions
}

extension FocusedValues {
    var queueActions: QueueActions? {
        get { self[QueueActionsKey.self] }
        set { self[QueueActionsKey.self] = newValue }
    }
}

struct QueueCommands: Commands {
    @FocusedValue(\.queueActions) private var actions

    var body: some Commands {
        CommandMenu("Queue") {
            Button("Transcribe") { actions?.transcribe?() }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(actions?.transcribe == nil)
            Button("Cancel Run") { actions?.cancel?() }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(actions?.cancel == nil)
            Divider()
            Button("Run Settings") { actions?.showRunSettings() }
                .keyboardShortcut(",", modifiers: [.command, .shift])
                .disabled(actions == nil)
            Button("Model Library…") { actions?.showModelLibrary() }
                .keyboardShortcut("l", modifiers: [.command, .shift])
                .disabled(actions == nil)
        }
    }
}

@MainActor
struct QueueWindow: View {
    @ObservedObject var state: AppState
    @State private var selection: QueueSelection?
    @State private var scope = QueueScope()
    @State private var inspectorPane: InspectorPane = .recording
    @State private var showsInspector = true
    @State private var sheet: WindowSheet?
    @State private var validation: PassValidation?
    @State private var completedPreview: PreviewResult?
    @State private var meetingToRemove: ZoomMeeting?

    private enum WindowSheet: String, Identifiable {
        case passes, library, server
        var id: String { rawValue }
    }

    private struct PassOptions: Equatable {
        let arguments: [[String]]
        let suffix: Bool
    }

    private struct PassValidation {
        let options: PassOptions
        let tags: [String?]
        let error: String?
    }

    private struct OutputOptions: Equatable {
        let destination: OutputDestination
        let folder: URL
        let format: String
        let timestamps: String
        let turns: Bool
        let device: String
        let diarize: Bool
        let diarizationModel: URL
    }

    private struct InputOptions: Equatable {
        let id: UUID
        let included: Bool
        let iso: Bool?
        let diarize: Bool
        let speakers: Int?
        let meeting: String?
        let outputs: [ModelOutput]
        let output: URL?
    }

    private struct PreviewRequest: Equatable {
        let selection: QueueSelection
        let passes: PassOptions
        let output: OutputOptions
        let inputs: [InputOptions]
    }

    private struct PreviewResult {
        let request: PreviewRequest
        let outputs: [PlannedTranscript]
        let error: String?
    }

    private var passOptions: PassOptions {
        PassOptions(arguments: state.configuredPasses.map(\.arguments), suffix: state.appendModelID)
    }

    private var outputOptions: OutputOptions {
        OutputOptions(destination: state.outputDestination, folder: state.outputFolder, format: state.format,
                      timestamps: state.timestamps, turns: state.speakerTurns, device: state.device,
                      diarize: state.diarize, diarizationModel: state.diarizationModel)
    }

    private var currentValidation: PassValidation? {
        validation?.options == passOptions ? validation : nil
    }

    private var previewRequest: PreviewRequest? {
        guard showsInspector, inspectorPane == .recording, let selection, selection.runTarget != nil,
              !state.isRunning, state.addOperations == 0,
              let validation = currentValidation, validation.error == nil else { return nil }
        return PreviewRequest(selection: selection, passes: validation.options, output: outputOptions,
                              inputs: state.items.map { item in
            InputOptions(id: item.id, included: item.selected, iso: item.isoOverride, diarize: item.diarize,
                         speakers: item.speakerCount, meeting: item.zoomMeeting, outputs: item.modelOutputs,
                         output: item.outputURL)
        })
    }

    private var canStart: Bool {
        state.canStartBatch && state.selectedCount > 0
            && currentValidation?.error == nil && currentValidation != nil
    }

    var body: some View {
        let request = previewRequest
        let preview = completedPreview.flatMap { $0.request == request ? $0 : nil }
        let passes = passOptions
        let queue = QueueSnapshot(state: state, scope: scope)
        // The menu bar stays live under a sheet or alert. ⌘↩ must not start a run behind
        // the pass editor, ⇧⌘L must not swap it for the library and drop its draft, and
        // ⌘. belongs to the sheet (it is also the system's Cancel key).
        let modal = sheet != nil || meetingToRemove != nil || state.planningError != nil
        VStack(spacing: 0) {
            QueueView(state: state, queue: queue, selection: $selection, scope: $scope,
                      retry: retry, remove: remove)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .inspector(isPresented: $showsInspector) {
                    RecordingInspector(state: state, selection: selection, status: queue.status(of: selection),
                                       pane: $inspectorPane,
                                       editPasses: { sheet = .passes },
                                       plannedOutputs: preview?.outputs ?? [],
                                       planningOutputs: request != nil && preview == nil,
                                       planError: preview?.error,
                                       retry: retry, remove: remove)
                        .inspectorColumnWidth(min: 250, ideal: 325, max: 460)
                }
            Divider()
            runBar(queue)
        }
        .navigationSubtitle(queue.counts)
        .focusedSceneValue(\.queueActions, modal ? nil : QueueActions(
            transcribe: canStart ? { state.transcribe() } : nil,
            cancel: state.isRunning && !state.isCancelling ? { state.cancelTranscription() } : nil,
            showRunSettings: {
                inspectorPane = .runSettings
                showsInspector = true
            },
            showModelLibrary: { sheet = .library }
        ))
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button("Server", systemImage: "dot.radiowaves.left.and.right") { sheet = .server }
                    .help("Manage the realtime transcription server")
                Button("Model Library", systemImage: "square.stack.3d.up") { sheet = .library }
                    .help("Download and manage transcription models (⇧⌘L)")
                Button("Add Recordings", systemImage: "plus") { state.chooseFiles() }
                    .labelStyle(.titleAndIcon)
                    .help("Add recordings, folders, or a Zoom library (⌘O)")
                    .disabled(state.isRunning)
            }
            ToolbarItem(placement: .automatic) {
                Button("Inspector", systemImage: "sidebar.right") { showsInspector.toggle() }
                    .accessibilityAddTraits(showsInspector ? .isSelected : [])
                    .keyboardShortcut("i", modifiers: [.command, .option])
                    .help(showsInspector ? "Hide inspector (⌥⌘I)" : "Show inspector (⌥⌘I)")
            }
        }
        .sheet(item: $sheet) { item in
            switch item {
            case .passes: ModelPassEditor(state: state)
            case .library: ModelLibrarySheet(state: state)
            case .server: ServerPanel(server: state.server, state: state)
            }
        }
        .alert("Queue Error", isPresented: Binding(
            get: { state.planningError != nil },
            set: { if !$0 { state.planningError = nil } }
        )) {
            Button("OK") { state.planningError = nil }
        } message: {
            Text(state.planningError ?? "")
        }
        .confirmationDialog(
            "Remove “\(meetingToRemove?.label ?? "")” from the queue?",
            isPresented: Binding(get: { meetingToRemove != nil }, set: { if !$0 { meetingToRemove = nil } }),
            presenting: meetingToRemove
        ) { meeting in
            Button("Remove Meeting", role: .destructive) { state.removeMeeting(meeting) }
            Button("Cancel", role: .cancel) {}
        } message: { meeting in
            let files = state.items.filter { $0.zoomMeeting == meeting.id }.count
            Text("This removes the meeting and its \(files == 1 ? "file" : "\(files) files") from the queue. The audio and any saved transcripts stay on disk.")
        }
        .onChange(of: queue.visibleIDs, initial: true) { previous, visible in
            keepSelectionVisible(previous: previous, visible: visible)
        }
        // Emptying the queue also clears the selection, one update after the handler below
        // has shown Run Settings; switching panes for a nil selection would undo that.
        .onChange(of: selection) { _, selected in
            if selected != nil { inspectorPane = .recording }
        }
        // An empty queue has nothing to inspect; show what the next run will use.
        .onChange(of: queue.isEmpty, initial: true) { _, empty in
            if empty { inspectorPane = .runSettings }
        }
        .task(id: passes) { await validate(passes) }
        .task(id: request) { await loadPreview(request) }
    }

    private func runBar(_ queue: QueueSnapshot) -> some View {
        let progress = state.isRunning ? state.queueProgress : 0
        return HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                if state.isRunning {
                    HStack(spacing: 6) {
                        Text(state.isCancelling ? "Stopping…" : (state.activePassLabel ?? "Transcribing"))
                        if !state.isCancelling, !state.activePassUsesApple {
                            Text(progress.formatted(.percent.precision(.fractionLength(0))))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                                .accessibilityHidden(true)
                        }
                    }
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                    HStack(spacing: 8) {
                        Group {
                            if state.activePassUsesApple {
                                ProgressView().controlSize(.small)
                            } else {
                                ProgressView(value: progress)
                                    .frame(maxWidth: 180)
                            }
                        }
                        .accessibilityLabel("Overall transcription progress")
                        if let recording = state.activeRecordingLabel {
                            Text(recording).lineLimit(1).truncationMode(.middle)
                        }
                        TimelineView(.periodic(from: Date(timeIntervalSinceReferenceDate: 0), by: 1)) { _ in
                            Text(runTiming(progress: state.activePassUsesApple ? nil : progress))
                                .monospacedDigit()
                                .lineLimit(1)
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                } else if state.batchPausedForServer {
                    Label("Queue paused while the server is running", systemImage: "pause.circle")
                        .font(.subheadline.weight(.semibold))
                } else {
                    HStack(spacing: 5) {
                        if let outcome = lastRunOutcome {
                            Image(systemName: outcome.symbol)
                                .foregroundStyle(outcome.tint)
                                .accessibilityHidden(true)
                        }
                        Text(state.lastRunSummary ?? (queue.isEmpty ? "Add recordings to begin" : scopeSummary(queue)))
                    }
                    .font(.subheadline.weight(.semibold))
                    HStack(spacing: 8) {
                        if state.lastRunSummary != nil { Text(scopeSummary(queue)) }
                        let invalid = currentValidation?.error != nil
                        Button {
                            if invalid { sheet = .passes }
                            else {
                                inspectorPane = .runSettings
                                showsInspector = true
                            }
                        } label: {
                            if invalid {
                                Label("Review Model Passes", systemImage: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.orange)
                            } else {
                                Text(settingsSummary).underline()
                            }
                        }
                        .buttonStyle(.plain)
                        // VoiceOver reads the summary itself; "Open Run Settings" alone hid it.
                        .accessibilityLabel(invalid ? "Review Model Passes" : settingsSummary)
                        .accessibilityHint(invalid ? "" : "Opens Run Settings")
                        .help(currentValidation?.error ?? "Open run settings")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            Spacer()
            if !state.isRunning, state.lastRunSummary != nil, state.outputDestination == .folder {
                Button("Show Transcripts") { NSWorkspace.shared.open(state.outputFolder) }
                    .help(state.outputFolder.path)
                    .disabled(!FileManager.default.fileExists(atPath: state.outputFolder.path))
            }
            if state.showModelActivity || state.downloadingLibraryModel != nil {
                Button(state.modelActivityCanDismiss ? "Model Details…" : "Model Activity…") { sheet = .library }
                    .controlSize(.small)
            }
            if state.isRunning {
                Button(state.isCancelling ? "Stopping…" : "Cancel Run") { state.cancelTranscription() }
                    .controlSize(.large)
                    .disabled(state.isCancelling)
                    .help("Stop after saving finished work (⌘.)")
            } else {
                Button("Transcribe") { state.transcribe() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(!canStart)
                    .help(startHelp)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 9)
        .frame(minHeight: 55)
        .background(.bar)
    }

    private func scopeSummary(_ queue: QueueSnapshot) -> String {
        let visible = Set(queue.visibleIDs)
        var included: Set<QueueSelection> = []
        var includedDuration: [QueueSelection: TimeInterval] = [:]
        for item in state.selectedItems {
            let id = item.zoomMeeting.map(QueueSelection.meeting) ?? .recording(item.id)
            included.insert(id)
            if item.skipReason == nil {
                includedDuration[id] = max(includedDuration[id, default: 0], item.duration)
            }
        }
        let hidden = included.subtracting(visible).count
        var parts = ["\(included.count) included"]
        if hidden > 0 { parts.append("\(hidden) hidden by filters") }
        let duration = includedDuration.values.reduce(0, +)
        if duration > 0 { parts.append(formatElapsed(duration)) }
        return parts.joined(separator: " · ")
    }

    private var startHelp: String {
        if state.batchPausedForServer { return "Stop the realtime server to start batch transcription" }
        if state.selectedCount == 0 { return "Include at least one recording to transcribe" }
        if state.addOperations > 0 { return "Waiting for recordings to finish loading" }
        if state.downloadingLibraryModel != nil { return "Waiting for the model download to finish" }
        if let error = currentValidation?.error { return error }
        if currentValidation == nil { return "Checking model passes" }
        return "Transcribe included recordings (⌘↩)"
    }

    private var lastRunOutcome: (symbol: String, tint: Color)? {
        guard let summary = state.lastRunSummary else { return nil }
        if summary.hasPrefix("Run complete") { return ("checkmark.circle.fill", .green) }
        if summary.hasPrefix("Run canceled") { return ("stop.circle.fill", .orange) }
        return ("exclamationmark.triangle.fill", .red)
    }

    private var settingsSummary: String {
        let passes = state.configuredPasses.count
        let passLabel = passes == 1 ? "1 model pass" : "\(passes) model passes"
        let format = state.format == "txt" ? "Text" : state.format.uppercased()
        let destination = state.outputDestination == .folder
            ? "Saving to “\(state.outputFolder.lastPathComponent)”" : "Saving next to originals"
        return "\(passLabel) · \(format) · \(destination)"
    }

    private func runTiming(progress: Double?) -> String {
        let elapsed = state.queueElapsedSeconds
        let label = "\(formatElapsed(elapsed)) elapsed"
        guard let progress, progress > 0.01, elapsed > 0 else { return label }
        let remaining = max(0, elapsed * (1 / progress - 1))
        return "\(label) · ~\(formatElapsed(remaining)) remaining"
    }

    /// The inspector follows the visible list. When the selected row goes away (removed or
    /// filtered out), select the row that took its place rather than jumping to the top:
    /// a second Delete would otherwise remove an unrelated row.
    private func keepSelectionVisible(previous: [QueueSelection], visible: [QueueSelection]) {
        if let selection, visible.contains(selection) { return }
        guard let selection, let index = previous.firstIndex(of: selection) else {
            selection = visible.first
            return
        }
        let remaining = Set(visible)
        let next = previous[index...].first { remaining.contains($0) }
        let before = previous[..<index].last { remaining.contains($0) }
        self.selection = next ?? before ?? visible.first
    }

    /// One removal path for the Delete key, the context menu, and the inspector. Nothing is
    /// deleted from disk and a recording is easily added back, so only a meeting, which
    /// takes all of its tracks and results with it, asks first.
    private func remove(_ selection: QueueSelection) {
        guard !state.isRunning else { return }
        switch selection {
        case .recording(let id):
            if let item = state.items.first(where: { $0.id == id }) { state.remove(item) }
        case .meeting(let id):
            meetingToRemove = state.zoomMeetings.first { $0.id == id }
        case .merge(let id):
            state.removeMergedSession(id)
        }
    }

    private func retry(_ selection: QueueSelection) {
        guard let target = selection.runTarget, state.canStartBatch else { return }
        if currentValidation?.error != nil { sheet = .passes; return }
        state.retry(target)
    }

    private func validate(_ options: PassOptions) async {
        guard options == passOptions else { return }
        let passes = state.configuredPasses
        do {
            try await Task.sleep(for: .milliseconds(150))
            let tags = try await state.validateModelPasses(passes, appendSuffix: options.suffix)
            try Task.checkCancellation()
            validation = PassValidation(options: options, tags: tags, error: nil)
        } catch {
            guard !Task.isCancelled else { return }
            validation = PassValidation(options: options, tags: [], error: error.localizedDescription)
        }
    }

    private func loadPreview(_ request: PreviewRequest?) async {
        guard !Task.isCancelled else { return }
        completedPreview = nil
        guard let request, let validation = currentValidation, validation.options == request.passes else { return }
        let passes = state.configuredPasses
        do {
            try await Task.sleep(for: .milliseconds(200))
            let outputs = try await state.previewOutputs(for: request.selection, passes: passes, tags: validation.tags)
            try Task.checkCancellation()
            completedPreview = PreviewResult(request: request, outputs: outputs, error: nil)
        } catch {
            guard !Task.isCancelled else { return }
            completedPreview = PreviewResult(request: request, outputs: [], error: error.localizedDescription)
        }
    }
}

private extension QueueSelection {
    var runTarget: AppState.RunTarget? {
        switch self {
        case .recording(let id): .recording(id)
        case .meeting(let id): .meeting(id)
        case .merge: nil
        }
    }
}
