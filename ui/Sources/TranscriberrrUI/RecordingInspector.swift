import AppKit
import SwiftUI

enum InspectorPane: String, CaseIterable {
    case recording = "Recording"
    case runSettings = "Run Settings"
}

@MainActor
struct RecordingInspector: View {
    @ObservedObject var state: AppState
    var selection: QueueSelection?
    /// The selected row's status exactly as the queue shows it.
    var status: QueueRowStatus?
    @Binding var pane: InspectorPane
    var editPasses: () -> Void
    var plannedOutputs: [PlannedTranscript]
    var planningOutputs: Bool
    var planError: String? = nil
    var retry: (QueueSelection) -> Void
    var remove: (QueueSelection) -> Void

    var body: some View {
        VStack(spacing: 0) {
            Picker("Inspector", selection: $pane) {
                ForEach(InspectorPane.allCases, id: \.self) { pane in
                    Text(pane.rawValue).tag(pane)
                }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            if pane == .recording {
                recording
                    .id(selection)
            } else {
                runSettings
            }
        }
    }

    @ViewBuilder
    private var recording: some View {
        switch selection {
        case .recording(let id):
            if let item = state.items.first(where: { $0.id == id }) {
                inspectorForm {
                    recordingOverview(item)
                    transcriptSections(resolvedOutputs(for: item))
                    if item.skipReason == nil { trackSettings(item) }
                    removeSection(.recording(id))
                }
            } else {
                noSelection
            }
        case .meeting(let id):
            if let meeting = state.zoomMeetings.first(where: { $0.id == id }) {
                let items = state.items.filter { $0.zoomMeeting == id }
                let tracks = items.filter { $0.skipReason == nil }
                let history = state.meetingResults(meeting, previous: state.mergedSessions.filter { $0.meetingID == id })
                let outcomes = state.passOutcomes(for: .meeting(id))
                let rows = resultRows(outcomes)
                inspectorForm {
                    Section {
                        statusRow
                        if let duration = tracks.map(\.duration).max() {
                            durationRow(duration)
                        }
                        if let source = tracks.first?.url.deletingLastPathComponent() {
                            let folder = source.lastPathComponent.caseInsensitiveCompare("Audio Record") == .orderedSame
                                ? source.deletingLastPathComponent() : source
                            LabeledContent("Location") { locationLabel(folder, reveal: folder) }
                        }
                        if let issue {
                            let trackFailure = tracks.lazy.compactMap { track -> String? in
                                if case .failed(let error) = track.state { return error }
                                return nil
                            }.first
                            let failure = outcomes.first { $0.phase == .failed }?.message
                                ?? trackFailure ?? history.last { $0.state == "failed" }?.message
                            issueCallout(issue == .failed ? failure : nil, failed: issue == .failed,
                                         retry: .meeting(id))
                        }
                    } header: {
                        Label(meeting.label, systemImage: "video")
                    }
                    speakerTracks(meeting, items: items)
                    transcriptSections(rows + mergeRows(history, excluding: availablePaths(rows)))
                    removeSection(.meeting(id))
                }
            } else {
                noSelection
            }
        case .merge(let id):
            if let session = state.mergedSessions.first(where: { $0.id == id }) {
                inspectorForm {
                    Section {
                        statusRow
                        LabeledContent("Location") {
                            locationLabel(URL(fileURLWithPath: session.id).deletingLastPathComponent(),
                                          reveal: URL(fileURLWithPath: session.id))
                        }
                    } header: {
                        Label(session.label, systemImage: "doc.on.doc")
                    }
                    transcriptSections(mergeRows([session], excluding: []), showPlans: false)
                    removeSection(.merge(id))
                }
            } else {
                noSelection
            }
        case nil:
            noSelection
        }
    }

    private var noSelection: some View {
        ContentUnavailableView(
            "No Recording Selected",
            systemImage: "waveform",
            description: Text("Select a recording in the queue to see its details.")
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func inspectorForm<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        Form(content: content)
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
    }

    private func recordingOverview(_ item: AudioItem) -> some View {
        Section {
            statusRow
            durationRow(item.duration)
            LabeledContent("Location") { locationLabel(item.url.deletingLastPathComponent(), reveal: item.url) }
            if item.skipReason == nil, let issue {
                issueCallout(failureMessage(item), failed: issue == .failed, retry: .recording(item.id))
            } else if let skipReason = item.skipReason {
                Text(skipReason)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else if item.state == .processing {
                if !item.resultSummary.isEmpty, item.resultSummary != itemStatus(item.state) {
                    Text(item.resultSummary).font(.callout).foregroundStyle(.secondary)
                }
            } else if let note = recordingNote(item), !note.isEmpty {
                Text(note)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        } header: {
            Label(item.url.lastPathComponent, systemImage: item.zoomRecording ? "video" : "waveform")
        }
    }

    @ViewBuilder
    private func durationRow(_ duration: TimeInterval) -> some View {
        if duration.isFinite, duration > 0 {
            let pattern: Duration.TimeFormatStyle.Pattern = duration >= 3600 ? .hourMinuteSecond : .minuteSecond
            LabeledContent("Duration", value: Duration.seconds(duration).formatted(.time(pattern: pattern)))
        }
    }

    private func failureMessage(_ item: AudioItem) -> String? {
        if case .failed(let error) = item.state { return error }
        return nil
    }

    private func recordingNote(_ item: AudioItem) -> String? {
        if case .failed(let error) = item.state {
            return state.passOutcomes(for: .recording(item.id)).isEmpty ? error : nil
        }
        let delimiter = ". Saved to "
        if let range = item.resultSummary.range(of: delimiter) {
            return String(item.resultSummary[..<range.lowerBound])
        }
        if item.resultSummary.hasPrefix("Saved to ") || item.resultSummary == itemStatus(item.state) {
            return nil
        }
        // "Waiting…" under a Waiting status says nothing new.
        if item.resultSummary.trimmingCharacters(in: CharacterSet(charactersIn: "…. ")) == status?.label {
            return nil
        }
        return item.resultSummary
    }

    private func speakerTracks(_ meeting: ZoomMeeting, items: [AudioItem]) -> some View {
        Section("Speaker Tracks") {
            LabeledContent("Participants", value: meeting.participants.formatted())
            LabeledContent("Bleed reduction", value: "Automatic")
            LabeledContent("Merge order", value: "By timestamp")
            let skipped = items.filter { $0.skipReason != nil }
            if !skipped.isEmpty {
                DisclosureGroup("Skipped Exports") {
                    ForEach(skipped) { item in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.url.lastPathComponent).font(.callout)
                            if let reason = item.skipReason {
                                Text(reason)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.vertical, 3)
                    }
                }
            }
        }
    }

    private func trackSettings(_ item: AudioItem) -> some View {
        Section("Track Settings") {
            Toggle("Speaker track", isOn: Binding(
                get: { item.isoTrack },
                set: { state.setIsoTrack(for: item, enabled: $0) }
            ))
            .help("Use for audio containing one speaker. Tracks from the same recording session are merged.")
            Toggle("Diarize track", isOn: Binding(
                get: { item.diarize },
                set: { state.setDiarization(for: item, enabled: $0) }
            ))
            .disabled(item.isoTrack)
            if item.diarize {
                Picker("Speakers", selection: Binding(
                    get: { item.speakerCount ?? 0 },
                    set: { state.setSpeakerCount(for: item, $0 == 0 ? nil : $0) }
                )) {
                    Text("Automatic").tag(0)
                    ForEach(1...8, id: \.self) { count in
                        Text(count.formatted()).tag(count)
                    }
                }
            }
        }
        .disabled(state.isRunning)
    }

    private func removeSection(_ target: QueueSelection) -> some View {
        Section {
            Button("Remove from Queue", role: .destructive) { remove(target) }
                .disabled(state.isRunning)
        }
    }

    /// Failed or canceled: the statuses that lead with a callout and offer a retry.
    private var issue: QueueStatusFilter? {
        guard let filter = status?.filter, filter == .failed || filter == .canceled else { return nil }
        return filter
    }

    /// Leads a failed or canceled recording or meeting with what went wrong and how to recover.
    private func issueCallout(_ message: String?, failed: Bool, retry target: QueueSelection) -> some View {
        let noun = if case .meeting = target { "meeting" } else { "recording" }
        return VStack(alignment: .leading, spacing: 8) {
            Label {
                Text(failed ? (message.flatMap { $0.isEmpty ? nil : $0 } ?? "Transcription failed.")
                     : "The run stopped before this \(noun) finished.")
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: failed ? "exclamationmark.triangle.fill" : "stop.circle.fill")
                    .foregroundStyle(failed ? .red : .orange)
            }
            .font(.callout)
            Button("Retry This \(noun.capitalized)") { retry(target) }
                .disabled(!state.canStartBatch)
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var statusRow: some View {
        if let status {
            LabeledContent("Status") {
                QueueStatusBadge(status: status.filter, label: status.label, progress: status.progress)
            }
        }
    }

    @ViewBuilder
    private func transcriptSections(_ results: [TranscriptRow], showPlans: Bool = true) -> some View {
        if !results.isEmpty {
            Section("Model Results") {
                ForEach(results) { result in
                    transcriptRow(result)
                }
            }
        }
        if showPlans {
            let visiblePlans = plansExcluding(results)
            if planningOutputs || !visiblePlans.isEmpty || planError != nil {
                Section("Planned Output") {
                    if planningOutputs && visiblePlans.isEmpty {
                        ProgressView("Planning outputs…")
                            .controlSize(.small)
                    }
                    if let planError {
                        Text(planError)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    ForEach(visiblePlans) { plan in
                        transcriptRow(plan)
                    }
                }
            }
        }
    }

    private struct TranscriptRow: Identifiable {
        var id: String
        var title: String
        var status: String?
        var message: String?
        var output: URL?
        var canReveal = false
        var modelIdentifier: String?
    }

    private func transcriptRow(_ row: TranscriptRow) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            if let status = row.status {
                LabeledContent {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } label: {
                    Text(row.title).fontWeight(.medium).help(row.modelIdentifier ?? row.title)
                }
            } else {
                Text(row.title).fontWeight(.medium).help(row.modelIdentifier ?? row.title)
            }
            if let output = row.output {
                if row.canReveal {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Button {
                            NSWorkspace.shared.open(output)
                        } label: {
                            Label(output.lastPathComponent, systemImage: "doc.text")
                                .lineLimit(2)
                                .truncationMode(.middle)
                        }
                        .buttonStyle(.link)
                        .accessibilityLabel("Open \(output.lastPathComponent)")
                        .help("Open \(output.path)")
                        Spacer(minLength: 4)
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([output])
                        } label: {
                            Label("Show in Finder", systemImage: "folder")
                                .labelStyle(.iconOnly)
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel("Show \(output.lastPathComponent) in Finder")
                        .help("Show in Finder")
                    }
                } else {
                    Text(output.lastPathComponent)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .help(output.path)
                }
            }
            if let message = row.message, !message.isEmpty {
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(.vertical, 3)
    }

    private func plansExcluding(_ results: [TranscriptRow]) -> [TranscriptRow] {
        let paths = Set(results.compactMap { $0.output?.standardizedFileURL.path })
        var seen: Set<String> = []
        return plannedOutputs.compactMap { plan in
            let output = URL(fileURLWithPath: plan.path)
            let path = output.standardizedFileURL.path
            guard !paths.contains(path), seen.insert(path).inserted else { return nil }
            let exists = plan.exists && isAvailable(output)
            return TranscriptRow(id: "plan-\(path)", title: modelTitle(plan.tag, identifier: plan.model),
                                 status: exists ? "Already Exists" : nil,
                                 output: output, canReveal: exists, modelIdentifier: plan.model)
        }
    }

    private func resolvedOutputs(for item: AudioItem) -> [TranscriptRow] {
        let outcomes = resultRows(state.passOutcomes(for: .recording(item.id)))
        let seen = availablePaths(outcomes)
        let history = state.modelOutputs(for: item).filter { !seen.contains(URL(fileURLWithPath: $0.path).standardizedFileURL.path) }
        var results = outcomes + history.map { saved in
            TranscriptRow(id: "saved-\(saved.id)", title: modelTitle(saved.tag), status: "Saved",
                          output: URL(fileURLWithPath: saved.path), canReveal: true)
        }
        if let output = item.outputURL, !results.contains(where: { $0.output?.standardizedFileURL == output.standardizedFileURL }),
           isAvailable(output) {
            results.append(TranscriptRow(id: "saved-\(output.path)", title: "Transcript", status: "Saved", output: output, canReveal: true))
        }
        return results
    }

    private func availablePaths(_ rows: [TranscriptRow]) -> Set<String> {
        Set(rows.lazy.filter(\.canReveal)
            .compactMap { $0.output?.standardizedFileURL.path })
    }

    private func resultRows(_ outcomes: [AppState.PassOutcome]) -> [TranscriptRow] {
        outcomes.map { outcome in
            let status = switch outcome.phase {
            case .waiting: "Waiting"
            case .running: "Transcribing"
            case .completed: "Complete"
            case .failed: "Failed"
            case .canceled: "Canceled"
            }
            var message = outcome.phase == .completed ? completionNote(outcome.message) : outcome.message
            if message == status { message = nil }
            return TranscriptRow(id: "pass-\(outcome.id)", title: modelTitle(outcome.modelTag, identifier: outcome.label), status: status,
                                 message: message, output: outcome.output,
                                 canReveal: outcome.output.map(isAvailable) == true, modelIdentifier: outcome.label)
        }
    }

    private func mergeRows(_ history: [MergedSession], excluding paths: Set<String>) -> [TranscriptRow] {
        history.compactMap { merge in
            let output = URL(fileURLWithPath: merge.id)
            guard !paths.contains(output.standardizedFileURL.path) else { return nil }
            return TranscriptRow(id: "merge-\(merge.id)", title: modelTitle(merge.modelTag),
                                 status: mergeStatus(merge.state), message: merge.message,
                                 output: output, canReveal: merge.state == "done" && isAvailable(output))
        }
    }

    private var runSettings: some View {
        inspectorForm {
            Section("Model Passes") {
                ForEach(Array(state.configuredPasses.enumerated()), id: \.offset) { index, pass in
                    VStack(alignment: .leading, spacing: 3) {
                        Text("\(index + 1). \(pass.displayName)")
                        if !pass.usesApple {
                            Text(pass.modelArgument)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                                .truncationMode(.middle)
                                .help(pass.modelArgument)
                        }
                    }
                }
                Button("Edit Passes…", action: editPasses)
                    .disabled(state.downloadingLibraryModel != nil)
            }
            Section("Output") {
                Picker("Format", selection: Binding(get: { state.format }, set: changeFormat)) {
                    Text("JSON").tag("json")
                    Text("Text").tag("txt")
                    Text("SRT").tag("srt")
                    Text("VTT").tag("vtt")
                }
                Picker("Destination", selection: saved(\.outputDestination)) {
                    ForEach(OutputDestination.allCases) { destination in
                        Text(destination.title).tag(destination)
                    }
                }
                if state.outputDestination == .folder {
                    LabeledContent("Folder") {
                        HStack(spacing: 8) {
                            locationLabel(state.outputFolder, reveal: state.outputFolder)
                            Button("Choose…") {
                                state.chooseFolder(state.outputFolder) {
                                    state.outputFolder = $0
                                    state.saveQueue()
                                }
                            }
                            .accessibilityLabel("Choose output folder")
                        }
                    }
                }
                Picker("Timestamps", selection: timestampSelection) {
                    if state.format == "json" || state.format == "txt" {
                        Text("Off").tag("none")
                    }
                    if state.format != "txt" {
                        Text("Sentences").tag("sentences")
                        Text("Words").tag("words")
                    }
                }
                .disabled(state.format == "txt")
                if state.format == "txt" {
                    Toggle("Speaker turns", isOn: saved(\.speakerTurns))
                        .help("Start a new labeled paragraph at each speaker change.")
                }
            }
            Section("Recognition") {
                if state.offersDeviceSelection {
                    Picker("Acceleration", selection: saved(\.device)) {
                        Text("Metal GPU").tag("metal")
                        Text("CPU").tag("cpu")
                    }
                }
                Toggle("Diarize new recordings", isOn: saved(\.diarize))
                if state.diarize || state.items.contains(where: \.diarize) {
                    LabeledContent("Diarization model") {
                        VStack(alignment: .trailing, spacing: 5) {
                            pathLabel(state.diarizationModelArgument)
                            Button("Choose…") {
                                state.chooseDiarizationModel(state.diarizationModel) {
                                    state.diarizationModel = $0
                                    state.saveQueue()
                                }
                            }
                            .accessibilityLabel("Choose diarization model")
                            if state.diarizationModelArgument != "sortformer" {
                                Button("Use Default") {
                                    state.diarizationModel = URL(fileURLWithPath: "/sortformer")
                                    state.saveQueue()
                                }
                            }
                        }
                    }
                }
            }
        }
        .disabled(state.isRunning)
    }

    private func saved<Value>(_ key: ReferenceWritableKeyPath<AppState, Value>) -> Binding<Value> {
        Binding(get: { state[keyPath: key] }, set: {
            state[keyPath: key] = $0
            state.saveQueue()
        })
    }

    private var timestampSelection: Binding<String> {
        Binding(get: {
            if state.format == "txt" { return "none" }
            if ["srt", "vtt"].contains(state.format), state.timestamps == "none" { return "sentences" }
            return state.timestamps
        }, set: {
            state.timestamps = $0
            state.saveQueue()
        })
    }

    private func changeFormat(_ format: String) {
        state.format = format
        if format == "txt" {
            state.timestamps = "none"
        } else if ["srt", "vtt"].contains(format), state.timestamps == "none" {
            state.timestamps = "sentences"
        }
        state.saveQueue()
    }

    private func pathLabel(_ path: String) -> some View {
        Text(path)
            .lineLimit(2)
            .truncationMode(.middle)
            .foregroundStyle(.secondary)
            .help(path)
    }

    private func locationLabel(_ url: URL, reveal: URL? = nil) -> some View {
        let name = url.lastPathComponent.isEmpty ? url.path : url.lastPathComponent
        return Group {
            if let reveal {
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([reveal])
                } label: {
                    Label(name, systemImage: "folder")
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .buttonStyle(.link)
                .accessibilityLabel("Show \(name) in Finder")
                .help("Show in Finder: \(url.path)")
            } else {
                pathLabel(name).help(url.path)
            }
        }
    }

    private func completionNote(_ message: String?) -> String? {
        guard let message, !message.hasPrefix("Saved to "), message != "Merged" else { return nil }
        if let range = message.range(of: ". Saved to ") { return String(message[..<range.lowerBound]) }
        return message
    }

    private func itemStatus(_ status: ItemState) -> String {
        switch status {
        case .queued: "Ready"
        case .processing: "Transcribing"
        case .done: "Complete"
        case .failed: "Failed"
        case .canceled: "Canceled"
        }
    }

    private func mergeStatus(_ status: String) -> String {
        switch status {
        case "done": "Complete"
        case "failed": "Failed"
        case "canceled": "Canceled"
        default: "Merging"
        }
    }

    private func modelTitle(_ tag: String?, identifier: String? = nil) -> String {
        switch tag {
        case "qw": "Qwen3 ASR 1.7B"
        case "qw06": "Qwen3 ASR 0.6B"
        case "pk": "Parakeet"
        case "pkt": "Parakeet TDT"
        case "ap": "Apple SpeechTranscriber"
        case let tag?: friendlyName(identifier) ?? identifier ?? "Model \(tag)"
        case nil: friendlyName(identifier) ?? identifier ?? "Transcript"
        }
    }

    /// A configured pass's engine name for a raw model identifier.
    private func friendlyName(_ identifier: String?) -> String? {
        guard let identifier else { return nil }
        return state.configuredPasses.first { $0.modelArgument == identifier || $0.displayName == identifier }?.displayName
    }

    private func isAvailable(_ output: URL) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: output.path, isDirectory: &isDirectory) && !isDirectory.boolValue
    }
}
