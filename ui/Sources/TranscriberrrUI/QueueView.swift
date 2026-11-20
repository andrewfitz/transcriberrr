import SwiftUI
import UniformTypeIdentifiers

enum QueueSelection: Hashable {
    case recording(UUID)
    case meeting(String)
    case merge(String)
}

enum QueueStatusFilter: String, CaseIterable, Identifiable {
    case all = "All Statuses"
    case ready = "Ready"
    case waiting = "Waiting"
    case running = "Transcribing"
    case complete = "Complete"
    case failed = "Failed"
    case canceled = "Canceled"
    case skipped = "Skipped"

    var id: Self { self }
}

struct QueueScope: Equatable {
    var query = ""
    var usesRegex = false
    var status: QueueStatusFilter = .all

    private var expression: Result<NSRegularExpression, Error>? {
        guard usesRegex, !query.isEmpty else { return nil }
        return Result { try NSRegularExpression(pattern: query, options: .caseInsensitive) }
    }

    var filterError: String? {
        guard case .failure(let error) = expression else { return nil }
        return error.localizedDescription
    }

    @MainActor
    fileprivate func allRows(in state: AppState) -> [QueueRow] {
        let meetings = Set(state.zoomMeetings.map(\.id))
        let groups = Dictionary(grouping: state.items, by: \.zoomMeeting)
        let meetingOutputs = Set(state.zoomMeetings.map(\.output))
        let grouped = state.zoomMeetings.compactMap { meeting -> QueueRow? in
            guard let tracks = groups[meeting.id] else { return nil }
            return QueueRow(meeting: meeting, tracks: tracks, state: state)
        }
        let recordings = state.items
            .filter { $0.zoomMeeting.map { !meetings.contains($0) } ?? true }
            .map { QueueRow(recording: $0, state: state) }
        let merges = state.mergedSessions.filter {
            !meetingOutputs.contains($0.id) && !($0.meetingID.map(meetings.contains) ?? false)
        }
        return grouped + recordings + merges.map(QueueRow.init(session:))
    }

    @MainActor
    fileprivate func visibleRows(in state: AppState) -> [QueueRow] {
        filter(allRows(in: state))
    }

    @MainActor
    fileprivate func filter(_ rows: [QueueRow]) -> [QueueRow] {
        let literal = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let regex: NSRegularExpression?
        switch expression {
        case .success(let pattern): regex = pattern
        case .failure: return []
        case nil: regex = nil
        }
        return rows.filter { row in
            guard status == .all || row.status == status else { return false }
            if let regex {
                return row.searchTerms.contains { term in
                    regex.firstMatch(in: term, range: NSRange(term.startIndex..., in: term)) != nil
                }
            }
            return literal.isEmpty || row.searchTerms.contains { $0.localizedCaseInsensitiveContains(literal) }
        }
    }

    @MainActor
    func visibleSelections(in state: AppState) -> [QueueSelection] {
        visibleRows(in: state).map(\.id)
    }
}

@MainActor
private struct QueueRow: Identifiable {
    let id: QueueSelection
    let title: String
    let detail: String
    let icon: String
    let status: QueueStatusFilter
    let statusLabel: String
    let statusDetail: String?
    let duration: TimeInterval
    let items: [AudioItem]
    let children: [QueueRow]

    var searchTerms: [String] {
        [title, detail] + items.flatMap { item in
            [item.url.path, item.speakerName, item.skipReason].compactMap { $0 }
        }
    }

    init(recording item: AudioItem, state: AppState, isChild: Bool = false) {
        id = .recording(item.id)
        title = isChild ? item.speakerName ?? item.url.lastPathComponent : item.url.lastPathComponent
        if isChild, item.speakerName != nil {
            detail = item.url.lastPathComponent
        } else if let skipped = item.skipReason {
            detail = skipped
        } else if item.isoTrack {
            detail = "Speaker track"
        } else {
            let extensionLabel = item.url.pathExtension.uppercased()
            detail = extensionLabel.isEmpty ? "Recording" : extensionLabel
        }
        icon = item.isoTrack ? "person" : item.skipReason != nil ? "minus" : "waveform"
        let fallback: QueueStatusFilter
        switch item.state {
        case .failed: fallback = .failed
        case .canceled: fallback = .canceled
        case .processing: fallback = .running
        case .done: fallback = item.skipReason == nil ? .complete : .skipped
        case .queued: fallback = item.skipReason == nil ? .ready : .skipped
        }
        let target: AppState.RunTarget = .recording(item.id)
        let current = !isChild && fallback != .skipped ? Self.passStatus(for: target, state: state) : nil
        status = current?.filter ?? fallback
        statusLabel = current?.label ?? fallback.rawValue
        statusDetail = item.skipReason ?? (item.resultSummary.isEmpty ? nil : item.resultSummary)
        duration = item.duration
        items = isChild ? [] : [item]
        children = []
    }

    init(meeting: ZoomMeeting, tracks: [AudioItem], state: AppState) {
        id = .meeting(meeting.id)
        title = meeting.label
        let speakers = tracks.filter { $0.skipReason == nil }
        detail = "Zoom meeting · \(speakers.count) \(speakers.count == 1 ? "speaker track" : "speaker tracks")"
        icon = "folder"
        let result = state.meetingResult(meeting)
        let failed = speakers.first { if case .failed = $0.state { return true }; return false }
        let fallback: QueueStatusFilter
        if result?.state == "running" || speakers.contains(where: { $0.state == .processing }) {
            fallback = .running
        } else if result?.state == "failed" || failed != nil {
            fallback = .failed
        } else if result?.state == "canceled" || speakers.contains(where: { $0.state == .canceled }) {
            fallback = .canceled
        } else {
            fallback = result?.state == "done" ? .complete : .ready
        }
        let current = Self.passStatus(for: .meeting(meeting.id), state: state)
        status = current?.filter ?? fallback
        statusLabel = current?.label ?? fallback.rawValue
        statusDetail = failed?.resultSummary ?? result?.message
        // Participant tracks commonly overlap for the full meeting.
        duration = speakers.map(\.duration).max() ?? 0
        items = tracks
        children = tracks.map { QueueRow(recording: $0, state: state, isChild: true) }
    }

    init(session: MergedSession) {
        id = .merge(session.id)
        title = session.label
        detail = URL(fileURLWithPath: session.id).lastPathComponent
        icon = "doc.text"
        switch session.state {
        case "done": status = .complete
        case "failed": status = .failed
        case "canceled": status = .canceled
        default: status = .running
        }
        statusLabel = status.rawValue
        statusDetail = session.message
        duration = 0
        items = []
        children = []
    }

    private static func passStatus(for target: AppState.RunTarget, state: AppState) -> (filter: QueueStatusFilter, label: String)? {
        guard state.isInCurrentRun(target) else { return nil }
        let passes = state.passOutcomes(for: target)
        guard !passes.isEmpty else { return nil }
        if passes.contains(where: { $0.phase == .running }) { return (.running, QueueStatusFilter.running.rawValue) }
        if passes.contains(where: { $0.phase == .failed }) { return (.failed, QueueStatusFilter.failed.rawValue) }
        if passes.contains(where: { $0.phase == .canceled }) { return (.canceled, QueueStatusFilter.canceled.rawValue) }
        let completed = passes.filter { $0.phase == .completed }.count
        if completed == passes.count { return (.complete, QueueStatusFilter.complete.rawValue) }
        if completed > 0 { return (.waiting, "\(completed)/\(passes.count) passes") }
        return (.waiting, QueueStatusFilter.waiting.rawValue)
    }
}

@MainActor
struct QueueView: View {
    @ObservedObject var state: AppState
    @Binding var selection: QueueSelection?
    @Binding var scope: QueueScope
    @State private var expandedMeetings: [String: Bool] = [:]
    @State private var dropTargeted = false
    @FocusState private var searchFocused: Bool

    var body: some View {
        let rows = scope.allRows(in: state)
        let visible = scope.filter(rows)
        return VStack(alignment: .leading, spacing: 12) {
            heading(rows)
            if rows.isEmpty {
                emptyState
            } else {
                filterBar(visible)
                if let filterError = scope.filterError {
                    Text("Invalid regular expression")
                        .font(.callout)
                        .foregroundStyle(.red)
                        .help(filterError)
                        .accessibilityHint(filterError)
                        .padding(.horizontal, 16)
                }
                if visible.isEmpty {
                    ContentUnavailableView {
                        Label("No Matching Recordings", systemImage: "magnifyingglass")
                    } actions: {
                        Button("Clear Filters") {
                            scope = QueueScope(usesRegex: scope.usesRegex)
                            searchFocused = true
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    table(visible)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted, perform: dropFiles)
    }

    private func heading(_ rows: [QueueRow]) -> some View {
        let meetings = rows.filter { if case .meeting = $0.id { return true }; return false }.count
        let recordings = rows.filter { if case .recording = $0.id { return true }; return false }.count
        let merges = rows.count - recordings - meetings
        let counts = [
            meetings > 0 ? "\(meetings) \(meetings == 1 ? "meeting" : "meetings")" : nil,
            recordings > 0 ? "\(recordings) \(recordings == 1 ? "recording" : "recordings")" : nil,
            merges > 0 ? "\(merges) \(merges == 1 ? "merged transcript" : "merged transcripts")" : nil,
        ].compactMap { $0 }
        return HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Queue").font(.title2.weight(.semibold)).accessibilityAddTraits(.isHeader)
                if !counts.isEmpty {
                    Text(counts.joined(separator: " · ")).foregroundStyle(.secondary)
                }
            }
            Spacer()
            if state.addOperations > 0 {
                HStack(spacing: 7) {
                    ProgressView().controlSize(.small)
                    Text("Finding recordings…").foregroundStyle(.secondary)
                }
                .accessibilityElement(children: .combine)
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label(dropTargeted && !state.isRunning ? "Release to Add Recordings" : "No Recordings",
                  systemImage: "waveform")
        } description: {
            Text("Drop recordings, folders, or a Zoom library here.")
        } actions: {
            Button("Add Recordings…", action: state.chooseFiles)
                .disabled(state.isRunning)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func filterBar(_ visible: [QueueRow]) -> some View {
        let visibleItems = visible.flatMap(\.items)
        let ordinary = visibleItems.filter { $0.zoomMeeting == nil }
        let allSpeakerTracks = !ordinary.isEmpty && ordinary.allSatisfy(\.isoTrack)
        return HStack(spacing: 8) {
            TextField("Find a recording", text: $scope.query)
                .textFieldStyle(.roundedBorder)
                .focused($searchFocused)
                .frame(idealWidth: 188, maxWidth: 245)
                .accessibilityLabel("Find a recording")
            Toggle(isOn: $scope.usesRegex) {
                Text(".*").monospaced()
            }
            .toggleStyle(.button)
            .help("Use a regular expression")
            .accessibilityLabel("Use a regular expression")
            Picker("Queue status", selection: $scope.status) {
                ForEach(QueueStatusFilter.allCases) { value in
                    Text(value.rawValue).tag(value)
                }
            }
            .labelsHidden()
            .fixedSize()
            .accessibilityLabel("Filter by status")
            Spacer(minLength: 4)
            Menu {
                Button("Include Visible") { state.selectAll(visibleItems, selected: true) }
                    .disabled(visibleItems.isEmpty)
                Button("Exclude Visible") { state.selectAll(visibleItems, selected: false) }
                    .disabled(visibleItems.isEmpty)
                Divider()
                Button(allSpeakerTracks ? "Unmark Visible Speaker Tracks" : "Mark Visible as Speaker Tracks") {
                    state.toggleIsoTracks(visibleItems)
                }
                .disabled(ordinary.isEmpty)
                Button("Clear Completed") { state.clearCompleted() }
                    .disabled(state.completedItemCount == 0 && !state.mergedSessions.contains { $0.state == "done" })
            } label: {
                Label("Queue Actions", systemImage: "ellipsis.circle")
                    .labelStyle(.iconOnly)
            }
            .menuIndicator(.hidden)
            .fixedSize()
            .disabled(state.isRunning)
        }
        .controlSize(.small)
        .padding(.horizontal, 16)
    }

    private func table(_ rows: [QueueRow]) -> some View {
        Table(of: QueueRow.self, selection: $selection) {
            TableColumn("Recording") { row in
                recordingCell(row)
            }
            .width(min: 180, ideal: 380)
            TableColumn("Duration") { row in
                if row.duration.isFinite, row.duration > 0 {
                    Text(Duration.seconds(row.duration).formatted(.time(pattern: row.duration >= 3600 ? .hourMinuteSecond : .minuteSecond)))
                        .monospacedDigit()
                }
            }
            .width(min: 60, ideal: 72, max: 100)
            .alignment(.trailing)
            TableColumn("Status") { row in
                Text(row.statusLabel)
                    .foregroundStyle(row.status == .failed ? Color.red : Color.secondary)
                    .help(row.statusDetail ?? row.statusLabel)
            }
            .width(min: 88, ideal: 96, max: 130)
            .alignment(.trailing)
        } rows: {
            ForEach(rows) { row in
                if case .meeting(let id) = row.id {
                    DisclosureTableRow(row, isExpanded: Binding(
                        get: { expandedMeetings[id, default: true] },
                        set: { expandedMeetings[id] = $0 }
                    )) {
                        ForEach(row.children) { child in
                            TableRow(child).selectionDisabled()
                        }
                    }
                } else {
                    TableRow(row)
                }
            }
        }
        .tableStyle(.inset(alternatesRowBackgrounds: false))
        .contextMenu(forSelectionType: QueueSelection.self) { ids in
            if let row = rows.first(where: { ids.contains($0.id) }) {
                rowActions(row)
            }
        }
    }

    private func recordingCell(_ row: QueueRow) -> some View {
        HStack(spacing: 8) {
            if !row.items.isEmpty {
                Toggle(isOn: Binding(
                    get: { row.items.contains(where: \.selected) },
                    set: { state.selectAll(row.items, selected: $0) }
                )) {
                    Text("Include \(row.title) in transcription")
                }
                .labelsHidden()
                .toggleStyle(.checkbox)
                .disabled(state.isRunning)
            }
            Image(systemName: row.icon)
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(row.title).lineLimit(1)
                if !row.detail.isEmpty {
                    Text(row.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            .help(row.detail.isEmpty ? row.title : "\(row.title)\n\(row.detail)")
        }
        .padding(.vertical, 3)
    }

    @ViewBuilder
    private func rowActions(_ row: QueueRow) -> some View {
        switch row.id {
        case .recording:
            if let item = row.items.first {
                Button(item.selected ? "Exclude from Run" : "Include in Run") {
                    state.setSelection(for: item, !item.selected)
                }
                .disabled(state.isRunning)
                Button("Remove from Queue", role: .destructive) { state.remove(item) }
                    .disabled(state.isRunning)
            }
        case .meeting(let id):
            let included = row.items.contains(where: \.selected)
            Button(included ? "Exclude from Run" : "Include in Run") {
                state.selectAll(row.items, selected: !included)
            }
            .disabled(state.isRunning)
            if let meeting = state.zoomMeetings.first(where: { $0.id == id }) {
                Button("Remove from Queue", role: .destructive) { state.removeMeeting(meeting) }
                    .disabled(state.isRunning)
            }
        case .merge(let id):
            if let session = state.mergedSessions.first(where: { $0.id == id && $0.state == "done" }) {
                Button("Show in Finder") { state.revealMerged(session) }
            }
        }
    }

    private func dropFiles(_ providers: [NSItemProvider]) -> Bool {
        guard !state.isRunning, !providers.isEmpty else { return false }
        let group = DispatchGroup()
        let collector = QueueDropURLs()
        for (index, provider) in providers.enumerated() {
            group.enter()
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                defer { group.leave() }
                if let url, url.isFileURL { collector.append(url, at: index) }
            }
        }
        group.notify(queue: .main) {
            Task { @MainActor in
                guard !state.isRunning else { return }
                state.add(urls: collector.values)
            }
        }
        return true
    }
}

/// NSItemProvider callbacks arrive on arbitrary queues. Preserve Finder's drop order.
private final class QueueDropURLs: @unchecked Sendable {
    private let lock = NSLock()
    private var urls: [Int: URL] = [:]

    var values: [URL] { lock.withLock { urls.sorted { $0.key < $1.key }.map(\.value) } }

    func append(_ url: URL, at index: Int) {
        lock.withLock { urls[index] = url }
    }
}
