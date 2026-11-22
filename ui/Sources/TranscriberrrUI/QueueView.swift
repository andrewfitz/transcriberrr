import AppKit
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

    /// Status symbol shared by the queue and the inspector. `nil` keeps calm states text-only.
    var symbol: String? {
        switch self {
        case .all, .ready: nil
        case .waiting: "clock"
        case .running: "waveform"
        case .complete: "checkmark.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        case .canceled: "stop.circle"
        case .skipped: "minus.circle"
        }
    }

    var tint: Color {
        switch self {
        case .complete: .green
        case .failed: .red
        case .canceled: .orange
        case .running: .accentColor
        default: .secondary
        }
    }
}

/// A status label with a symbol, or a compact progress bar while transcribing.
struct QueueStatusBadge: View {
    let status: QueueStatusFilter
    let label: String
    var progress: Double? = nil
    var indeterminate = false
    @Environment(\.backgroundProminence) private var backgroundProminence

    var body: some View {
        // Red, green, and the accent tint are unreadable on a focused row's accent-colored
        // selection, so follow the selection's own text colors there.
        let onSelection = backgroundProminence == .increased
        if status == .running {
            let value = indeterminate ? nil : progress.flatMap { $0.isFinite ? min(max($0, 0), 1) : nil }
            HStack(spacing: 6) {
                Group {
                    if let value {
                        ProgressView(value: value)
                            .tint(onSelection ? Color.white : nil)
                            .frame(width: 54)
                    } else {
                        ProgressView().controlSize(.mini)
                    }
                }
                Text(value?.formatted(.percent.precision(.fractionLength(0))) ?? label)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(label)
            .accessibilityValue(value?.formatted(.percent.precision(.fractionLength(0))) ?? "")
        } else {
            HStack(spacing: 4) {
                if let symbol = status.symbol {
                    Image(systemName: symbol)
                        .foregroundStyle(onSelection ? AnyShapeStyle(.primary) : AnyShapeStyle(status.tint))
                        .imageScale(.small)
                        .accessibilityHidden(true)
                }
                Text(label)
                    .foregroundStyle(status == .failed && !onSelection ? AnyShapeStyle(Color.red) : AnyShapeStyle(.secondary))
            }
        }
    }
}

struct QueueScope: Equatable {
    var query = ""
    var usesRegex = false
    var status: QueueStatusFilter = .all

    fileprivate var expression: Result<NSRegularExpression, Error>? {
        guard usesRegex, !query.isEmpty else { return nil }
        return Result { try NSRegularExpression(pattern: query, options: .caseInsensitive) }
    }

    @MainActor
    fileprivate func filter(_ rows: [QueueRow], expression: Result<NSRegularExpression, Error>?) -> [QueueRow] {
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
}

/// What a row's Status column shows. The inspector repeats it, so the two cannot disagree.
struct QueueRowStatus: Equatable {
    var filter: QueueStatusFilter
    var label: String
    var progress: Double?
}

/// The queue's rows for one render. Building a row reads every track (and, for meetings, the
/// disk), and the window re-renders every second during a run, so `QueueWindow` builds the
/// rows once and shares them with the list, the run bar, and the inspector.
@MainActor
struct QueueSnapshot {
    /// Every row, ignoring filters.
    fileprivate let rows: [QueueRow]
    /// The rows that pass the search and status filters.
    fileprivate let visible: [QueueRow]
    let visibleIDs: [QueueSelection]
    let filterError: String?

    init(state: AppState, scope: QueueScope) {
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
        rows = grouped + recordings + merges.map(QueueRow.init(session:))
        let expression = scope.expression
        if case .failure(let error) = expression {
            filterError = error.localizedDescription
        } else {
            filterError = nil
        }
        visible = scope.filter(rows, expression: expression)
        visibleIDs = visible.map(\.id)
    }

    var isEmpty: Bool { rows.isEmpty }

    /// "1 meeting · 3 recordings", for the window subtitle.
    var counts: String {
        var meetings = 0, recordings = 0, merges = 0
        for row in rows {
            switch row.id {
            case .meeting: meetings += 1
            case .recording: recordings += 1
            case .merge: merges += 1
            }
        }
        let parts = [
            meetings > 0 ? "\(meetings) \(meetings == 1 ? "meeting" : "meetings")" : nil,
            recordings > 0 ? "\(recordings) \(recordings == 1 ? "recording" : "recordings")" : nil,
            merges > 0 ? "\(merges) \(merges == 1 ? "merged transcript" : "merged transcripts")" : nil,
        ].compactMap { $0 }
        return parts.isEmpty ? "Queue is empty" : parts.joined(separator: " · ")
    }

    func status(of selection: QueueSelection?) -> QueueRowStatus? {
        guard let selection, let row = rows.first(where: { $0.id == selection }) else { return nil }
        return QueueRowStatus(filter: row.status, label: row.statusLabel, progress: row.progress)
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
    var detailIsError = false
    var isChild = false
    var progress: Double?
    /// Candidate transcripts, preferred first; existence is checked only when acted on.
    var transcripts: [URL] = []
    var transcript: URL? { transcripts.first { FileManager.default.fileExists(atPath: $0.path) } }
    /// The source file or folder to reveal in Finder.
    var source: URL?

    var searchTerms: [String] {
        [title, detail] + items.flatMap { item in
            [item.url.path, item.speakerName, item.skipReason].compactMap { $0 }
        }
    }

    init(recording item: AudioItem, state: AppState, isChild: Bool = false) {
        id = .recording(item.id)
        self.isChild = isChild
        title = isChild ? item.speakerName ?? item.url.lastPathComponent : item.url.lastPathComponent
        if isChild, item.speakerName != nil {
            detail = item.url.lastPathComponent
        } else if let skipped = item.skipReason {
            detail = skipped
        } else if case .failed(let error) = item.state, !isChild {
            detail = error.split(whereSeparator: \.isNewline).first.map(String.init) ?? error
            detailIsError = true
        } else {
            var parts = [item.isoTrack ? "Speaker track" : item.url.pathExtension.uppercased()]
            if item.diarize, !item.isoTrack { parts.append("Diarized") }
            let folder = item.url.deletingLastPathComponent().lastPathComponent
            if !folder.isEmpty, folder != "/" { parts.append(folder) }
            detail = parts.filter { !$0.isEmpty }.joined(separator: " · ")
        }
        icon = item.isoTrack ? "person" : item.skipReason != nil ? "minus" : "waveform"
        source = item.url
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
        if isChild, status == .ready || status == .waiting {
            statusLabel = item.isoTrack ? "Speaker track" : status.rawValue
        } else {
            statusLabel = current?.label ?? fallback.rawValue
        }
        statusDetail = item.skipReason ?? (item.resultSummary.isEmpty ? nil : item.resultSummary)
        duration = item.duration
        items = isChild ? [] : [item]
        children = []
        progress = item.state == .processing && !state.activePassUsesApple ? item.progress : nil
        if !isChild {
            transcripts = [item.outputURL].compactMap { $0 }
                + item.modelOutputs.sorted { ($0.tag, $0.path) > ($1.tag, $1.path) }.map { URL(fileURLWithPath: $0.path) }
        }
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
        if status == .running, !state.activePassUsesApple, !speakers.isEmpty {
            progress = speakers.reduce(0) { $0 + ($1.state == .done ? 1 : $1.progress) } / Double(speakers.count)
        }
        if result?.state == "done", let path = result?.id {
            transcripts = [URL(fileURLWithPath: path)]
        }
        if let folder = tracks.first?.url.deletingLastPathComponent() {
            source = folder.lastPathComponent.caseInsensitiveCompare("Audio Record") == .orderedSame
                ? folder.deletingLastPathComponent() : folder
        }
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
        statusLabel = status == .running ? "Merging" : status.rawValue
        statusDetail = session.message
        duration = 0
        items = []
        children = []
        if session.state == "done" {
            transcripts = [URL(fileURLWithPath: session.id)]
        }
    }

    var runTarget: AppState.RunTarget? {
        switch id {
        case .recording(let id) where !isChild: .recording(id)
        case .meeting(let id): .meeting(id)
        default: nil
        }
    }

    var canRetry: Bool { runTarget != nil && (status == .failed || status == .canceled) }

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
    let queue: QueueSnapshot
    @Binding var selection: QueueSelection?
    @Binding var scope: QueueScope
    var retry: (QueueSelection) -> Void
    var remove: (QueueSelection) -> Void
    @State private var expandedMeetings: [String: Bool] = [:]
    @State private var dropTargeted = false
    @FocusState private var searchFocused: Bool

    var body: some View {
        let rows = queue.rows
        let visible = queue.visible
        return VStack(alignment: .leading, spacing: 10) {
            if rows.isEmpty {
                emptyState
            } else {
                filterBar(visible)
                    .padding(.top, 10)
                if visible.isEmpty {
                    ContentUnavailableView {
                        if queue.filterError != nil {
                            Label("Invalid Regular Expression", systemImage: "exclamationmark.triangle")
                        } else {
                            Label("No Matching Recordings", systemImage: "magnifyingglass")
                        }
                    } description: {
                        if let error = queue.filterError { Text(error) }
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
        .overlay {
            if dropTargeted && !state.isRunning && !rows.isEmpty {
                dropHighlight
            }
        }
        .animation(.easeOut(duration: 0.15), value: dropTargeted)
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted, perform: dropFiles)
        .background {
            // ⌘F focuses the queue filter.
            Button("Find") { searchFocused = true }
                .keyboardShortcut("f")
                .opacity(0)
                .frame(width: 0, height: 0)
                .accessibilityHidden(true)
                .disabled(rows.isEmpty)
        }
    }

    private var dropHighlight: some View {
        RoundedRectangle(cornerRadius: 10)
            .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [7, 5]))
            .background(Color.accentColor.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
            .overlay(alignment: .bottom) {
                Label("Drop to Add Recordings", systemImage: "plus.circle.fill")
                    .font(.callout.weight(.medium))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.regularMaterial, in: Capsule())
                    .padding(.bottom, 18)
            }
            .padding(6)
            .allowsHitTesting(false)
            .transition(.opacity)
    }

    private var emptyState: some View {
        let targeted = dropTargeted && !state.isRunning
        return ContentUnavailableView {
            Label(targeted ? "Release to Add Recordings" : "No Recordings",
                  systemImage: targeted ? "plus.circle" : "waveform")
        } description: {
            if state.addOperations > 0 {
                Text("Finding recordings…")
            } else {
                Text("Drop audio or video files, folders, or a Zoom library here.")
            }
        } actions: {
            if state.addOperations > 0 {
                ProgressView().controlSize(.small)
                    .accessibilityLabel("Finding recordings")
            } else {
                Button("Add Recordings…", action: state.chooseFiles)
                    .disabled(state.isRunning)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(targeted ? Color.accentColor : Color.secondary.opacity(0.35),
                              style: StrokeStyle(lineWidth: targeted ? 2 : 1, dash: [7, 5]))
                .background(targeted ? Color.accentColor.opacity(0.06) : .clear,
                            in: RoundedRectangle(cornerRadius: 12))
                .padding(16)
        }
    }

    private func filterBar(_ visible: [QueueRow]) -> some View {
        let visibleItems = visible.flatMap(\.items)
        let ordinary = visibleItems.filter { $0.zoomMeeting == nil }
        let allSpeakerTracks = !ordinary.isEmpty && ordinary.allSatisfy(\.isoTrack)
        return HStack(spacing: 8) {
            HStack(spacing: 4) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
                TextField("Find a recording", text: $scope.query)
                    .textFieldStyle(.plain)
                    .focused($searchFocused)
                    .onExitCommand { scope.query = ""; searchFocused = false }
                    .accessibilityLabel("Find a recording")
                if !scope.query.isEmpty {
                    Button {
                        scope.query = ""
                        searchFocused = true
                    } label: {
                        Label("Clear Search", systemImage: "xmark.circle.fill")
                            .labelStyle(.iconOnly)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.borderless)
                }
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(.background, in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.separator))
            .frame(idealWidth: 210, maxWidth: 260)
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
            if state.addOperations > 0 {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("Finding recordings…").foregroundStyle(.secondary)
                }
                .accessibilityElement(children: .combine)
            }
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
            .help("Queue actions")
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
            .width(min: 140, ideal: 220)
            TableColumn("Duration") { row in
                if row.duration.isFinite, row.duration > 0 {
                    Text(Duration.seconds(row.duration).formatted(.time(pattern: row.duration >= 3600 ? .hourMinuteSecond : .minuteSecond)))
                        .monospacedDigit()
                        .foregroundStyle(row.isChild ? .secondary : .primary)
                }
            }
            .width(min: 52, ideal: 72, max: 100)
            .alignment(.trailing)
            TableColumn("Status") { row in
                QueueStatusBadge(status: row.status, label: row.statusLabel, progress: row.progress,
                                 indeterminate: state.activePassUsesApple)
                    .font(row.isChild ? .callout : nil)
                    .help(row.statusDetail ?? row.statusLabel)
            }
            .width(min: 84, ideal: 118, max: 150)
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
            if let row = target(of: ids, in: rows) {
                rowActions(row)
            }
        } primaryAction: { ids in
            guard let row = target(of: ids, in: rows) else { return }
            if let transcript = row.transcript {
                NSWorkspace.shared.open(transcript)
            } else if let source = row.source {
                NSWorkspace.shared.activateFileViewerSelecting([source])
            }
        }
        .onDeleteCommand {
            guard !state.isRunning, let selected = selection, rows.contains(where: { $0.id == selected }) else { return }
            remove(selected)
        }
        .onKeyPress(.space) {
            guard !state.isRunning, let selected = selection,
                  let row = rows.first(where: { $0.id == selected }), !row.items.isEmpty else { return .ignored }
            state.selectAll(row.items, selected: !row.items.contains(where: \.selected))
            return .handled
        }
    }

    /// The row a click landed on. A meeting's speaker tracks cannot be selected, but they
    /// can be double-clicked and have a (shorter) context menu.
    private func target(of ids: Set<QueueSelection>, in rows: [QueueRow]) -> QueueRow? {
        rows.first { ids.contains($0.id) }
            ?? rows.lazy.flatMap(\.children).first { ids.contains($0.id) }
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
                .help("Include in the next run")
            }
            Image(systemName: row.icon)
                .foregroundStyle(.secondary)
                .frame(width: 16)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(row.title)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(row.status == .skipped ? .secondary : .primary)
                if !row.detail.isEmpty {
                    QueueRowDetail(text: row.detail, isError: row.detailIsError)
                }
            }
            .help(row.detail.isEmpty ? row.title : "\(row.title)\n\(row.statusDetail ?? row.detail)")
        }
        .padding(.vertical, 3)
    }

    @ViewBuilder
    private func rowActions(_ row: QueueRow) -> some View {
        let transcript = row.transcript
        if let transcript {
            Button("Open Transcript") { NSWorkspace.shared.open(transcript) }
            Button("Show Transcript in Finder") { NSWorkspace.shared.activateFileViewerSelecting([transcript]) }
        }
        if let source = row.source {
            Button(row.children.isEmpty ? "Show Recording in Finder" : "Show Meeting Folder in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([source])
            }
        }
        if row.canRetry {
            Divider()
            Button("Retry") { retry(row.id) }
                .disabled(!state.canStartBatch)
        }
        if !row.items.isEmpty {
            Divider()
            let included = row.items.contains(where: \.selected)
            Button(included ? "Exclude from Run" : "Include in Run") {
                state.selectAll(row.items, selected: !included)
            }
            .disabled(state.isRunning)
        }
        if !row.isChild {
            // A merged transcript has no Include item above; separate only when something is.
            if row.items.isEmpty, transcript != nil { Divider() }
            Button("Remove from Queue", role: .destructive) { remove(row.id) }
                .disabled(state.isRunning)
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

/// A row's caption. An error is red, except on a focused row's accent-colored selection,
/// where red is unreadable and the selection's own text color takes over.
private struct QueueRowDetail: View {
    let text: String
    let isError: Bool
    @Environment(\.backgroundProminence) private var backgroundProminence

    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(isError && backgroundProminence != .increased
                             ? AnyShapeStyle(Color.red) : AnyShapeStyle(.secondary))
            .lineLimit(1)
            .truncationMode(.tail)
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
