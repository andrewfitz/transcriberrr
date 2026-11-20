import SwiftUI

@MainActor
struct QueueWindow: View {
    @ObservedObject var state: AppState
    @State private var selection: QueueSelection?
    @State private var scope = QueueScope()
    @State private var inspectorPane: InspectorPane = .recording
    @State private var showsInspector = true
    @State private var sheet: WindowSheet?
    @State private var validation: PassValidation?
    @State private var plannedOutputs: [PlannedTranscript] = []
    @State private var completedPreview: PreviewRequest?
    @State private var planError: String?

    private enum WindowSheet: String, Identifiable {
        case passes, library
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
        state.selectedCount > 0 && state.addOperations == 0 && state.downloadingLibraryModel == nil
            && currentValidation?.error == nil && currentValidation != nil
    }

    var body: some View {
        let request = previewRequest
        let passes = passOptions
        VStack(spacing: 0) {
            QueueView(state: state, selection: $selection, scope: $scope)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .inspector(isPresented: $showsInspector) {
                    RecordingInspector(state: state, selection: selection, pane: $inspectorPane,
                                       editPasses: { sheet = .passes },
                                       plannedOutputs: completedPreview == request && request != nil ? plannedOutputs : [],
                                       planningOutputs: request != nil && completedPreview != request,
                                       planError: completedPreview == request && request != nil ? planError : nil,
                                       retry: retry)
                        .inspectorColumnWidth(min: 250, ideal: 325, max: 460)
                }
            Divider()
            runBar
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button("Model Library", systemImage: "square.stack.3d.up") { sheet = .library }
                Button("Add Recordings", systemImage: "plus") { state.chooseFiles() }
                    .disabled(state.isRunning)
            }
            ToolbarItem(placement: .automatic) {
                Button("Inspector", systemImage: "sidebar.right") { showsInspector.toggle() }
                    .accessibilityAddTraits(showsInspector ? .isSelected : [])
                    .keyboardShortcut("i", modifiers: [.command, .option])
            }
        }
        .sheet(item: $sheet) { item in
            switch item {
            case .passes: ModelPassEditor(state: state)
            case .library: ModelLibrarySheet(state: state)
            }
        }
        .alert("Couldn’t Plan the Queue", isPresented: Binding(
            get: { state.planningError != nil },
            set: { if !$0 { state.planningError = nil } }
        )) {
            Button("OK") { state.planningError = nil }
        } message: {
            Text(state.planningError ?? "")
        }
        .onChange(of: scope.visibleSelections(in: state), initial: true) { _, visible in
            if let selection, visible.contains(selection) { return }
            selection = visible.first
        }
        .onChange(of: selection) { _, _ in inspectorPane = .recording }
        .onChange(of: outputOptions) { _, _ in state.saveQueue() }
        .task(id: passes) { await validate(passes) }
        .task(id: request) { await loadPreview(request) }
    }

    private var runBar: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                if state.isRunning {
                    Text(state.isCancelling ? "Stopping…" : (state.activePassLabel ?? "Transcribing"))
                        .font(.subheadline.weight(.semibold))
                    HStack(spacing: 8) {
                        ProgressView(value: state.queueProgress)
                            .frame(maxWidth: 180)
                            .accessibilityLabel("Overall transcription progress")
                        if let recording = state.activeRecordingLabel {
                            Text(recording).lineLimit(1)
                        }
                        Text(runTiming).monospacedDigit().lineLimit(1)
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                } else {
                    Text(state.lastRunSummary ?? scopeSummary).font(.subheadline.weight(.semibold))
                    HStack(spacing: 8) {
                        if state.lastRunSummary != nil { Text(scopeSummary) }
                        Button(currentValidation?.error == nil ? settingsSummary : "Review Model Passes") {
                            if currentValidation?.error != nil { sheet = .passes }
                            else {
                                inspectorPane = .runSettings
                                showsInspector = true
                            }
                        }
                        .buttonStyle(.link)
                        .accessibilityLabel(currentValidation?.error == nil ? "Open Run Settings" : "Review Model Passes")
                        .help(currentValidation?.error ?? "Open run settings")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            Spacer()
            if state.showModelActivity || state.downloadingLibraryModel != nil {
                Button(state.modelActivityCanDismiss ? "Model Details…" : "Model Activity…") { sheet = .library }
                    .controlSize(.small)
            }
            if state.isRunning {
                Button(state.isCancelling ? "Stopping…" : "Cancel Run") { state.cancelTranscription() }
                    .disabled(state.isCancelling)
                    .keyboardShortcut(".", modifiers: .command)
            } else {
                Button("Transcribe") { state.transcribe() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!canStart)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 9)
        .frame(minHeight: 55)
        .background(.bar)
    }

    private var scopeSummary: String {
        let visible = Set(scope.visibleSelections(in: state))
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

    private var settingsSummary: String {
        let passes = state.configuredPasses.count
        let passLabel = passes == 1 ? "1 model pass" : "\(passes) model passes"
        let format = state.format == "txt" ? "Text" : state.format.uppercased()
        return "\(passLabel) · \(format) · \(state.outputDestination.title)"
    }

    private var runTiming: String {
        let elapsed = "\(formatElapsed(state.queueElapsedSeconds)) elapsed"
        guard let remaining = state.queueEstimatedRemainingSeconds else { return elapsed }
        return "\(elapsed) · ~\(formatElapsed(remaining)) remaining"
    }

    private func retry(_ selection: QueueSelection) {
        guard let target = selection.runTarget, !state.isRunning, state.downloadingLibraryModel == nil else { return }
        if currentValidation?.error != nil { sheet = .passes; return }
        state.retry(target)
    }

    private func validate(_ options: PassOptions) async {
        let passes = state.configuredPasses
        do {
            try await Task.sleep(for: .milliseconds(150))
            let tags = try await state.validateModelPasses(passes)
            try Task.checkCancellation()
            validation = PassValidation(options: options, tags: tags, error: nil)
        } catch {
            guard !Task.isCancelled else { return }
            validation = PassValidation(options: options, tags: [], error: error.localizedDescription)
        }
    }

    private func loadPreview(_ request: PreviewRequest?) async {
        guard let request, let validation = currentValidation, validation.options == request.passes else { return }
        let passes = state.configuredPasses
        plannedOutputs = []
        planError = nil
        do {
            try await Task.sleep(for: .milliseconds(200))
            let outputs = try await state.previewOutputs(for: request.selection, passes: passes, tags: validation.tags)
            try Task.checkCancellation()
            plannedOutputs = outputs
            completedPreview = request
        } catch {
            guard !Task.isCancelled else { return }
            planError = error.localizedDescription
            completedPreview = request
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
