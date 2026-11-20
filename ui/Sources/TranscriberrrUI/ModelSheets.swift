import AppKit
import SwiftUI

@MainActor
struct ModelPassEditor: View {
    @ObservedObject var state: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var passes: [TranscriptionPass]
    @State private var appendSuffix: Bool
    @State private var validation: Validation?

    private struct Draft: Equatable {
        let passes: [TranscriptionPass]
        let appendSuffix: Bool
    }

    private struct Validation {
        let draft: Draft
        let message: String?
    }

    init(state: AppState) {
        self.state = state
        _passes = State(initialValue: state.configuredPasses)
        _appendSuffix = State(initialValue: state.appendModelID)
    }

    private var draft: Draft { Draft(passes: passes, appendSuffix: appendSuffix) }
    private var isChecking: Bool { validation?.draft != draft }
    private var readyToSave: Bool {
        !isChecking && validation?.message == nil && !state.isRunning && state.downloadingLibraryModel == nil
    }

    var body: some View {
        let draft = self.draft
        VStack(spacing: 0) {
            Text("Model Passes")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding([.horizontal, .top])
            Form {
                ForEach($passes) { $pass in
                    let id = pass.id
                    if let index = passes.firstIndex(where: { $0.id == id }) {
                        Section {
                            Picker("Engine", selection: $pass.engine) {
                                Text("Qwen3 ASR").tag("qwen")
                                Text("Parakeet").tag("parakeet-v3")
                                if #available(macOS 26, *) {
                                    Text("Apple SpeechTranscriber").tag("apple-native")
                                } else {
                                    Text("Apple SpeechTranscriber (macOS 26+)").tag("apple-native").disabled(true)
                                }
                            }
                            .onChange(of: pass.engine) { _, _ in $pass.model.wrappedValue = "" }

                            if pass.usesApple {
                                LabeledContent("Model", value: "System model")
                            } else {
                                TextField("Model", text: $pass.model, prompt: Text(pass.defaultModel))
                                    .help("Model ID or local path")
                            }
                        } header: {
                            HStack {
                                Text("Pass \(index + 1)")
                                Spacer()
                                Button { move(id, by: -1) } label: {
                                    Label("Move Pass \(index + 1) Up", systemImage: "chevron.up")
                                }
                                .disabled(index == 0)
                                Button { move(id, by: 1) } label: {
                                    Label("Move Pass \(index + 1) Down", systemImage: "chevron.down")
                                }
                                .disabled(index == passes.count - 1)
                                Button {
                                    passes.removeAll { $0.id == id }
                                } label: {
                                    Label("Remove Pass \(index + 1)", systemImage: "minus")
                                }
                                .disabled(passes.count == 1)
                            }
                            .labelStyle(.iconOnly)
                            .buttonStyle(.borderless)
                        }
                    }
                }
                Section {
                    Toggle("Append model ID to filenames", isOn: Binding(
                        get: { passes.count > 1 || appendSuffix },
                        set: { appendSuffix = $0 }
                    ))
                    .disabled(passes.count > 1)
                    if !isChecking, let message = validation?.message {
                        Label(message, systemImage: "exclamationmark.circle")
                            .foregroundStyle(.red)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .formStyle(.grouped)
            .frame(minHeight: 245, idealHeight: min(520, 135 * CGFloat(passes.count) + 132))

            Divider()
            HStack {
                Button("Add Pass", systemImage: "plus", action: addPass)
                Spacer()
                if isChecking {
                    ProgressView().controlSize(.small)
                        .accessibilityLabel("Checking model passes")
                }
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Done") {
                    state.replaceModelPasses(passes, appendSuffix: appendSuffix)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!readyToSave)
            }
            .padding()
        }
        .frame(width: 520)
        .task(id: draft) {
            do {
                try await Task.sleep(for: .milliseconds(250))
                _ = try await state.validateModelPasses(draft.passes, appendSuffix: draft.appendSuffix)
                guard !Task.isCancelled else { return }
                validation = Validation(draft: draft, message: nil)
            } catch where !Task.isCancelled {
                validation = Validation(draft: draft, message: error.localizedDescription)
            } catch {}
        }
    }

    private func move(_ id: UUID, by step: Int) {
        guard let index = passes.firstIndex(where: { $0.id == id }), passes.indices.contains(index + step) else { return }
        passes.swapAt(index, index + step)
    }

    private func addPass() {
        let installedEngines = Set(passes.map(\.engine))
        var engines = ["qwen", "parakeet-v3"]
        if #available(macOS 26, *) { engines.append("apple-native") }
        if let engine = engines.first(where: { !installedEngines.contains($0) }) {
            passes.append(TranscriptionPass(engine: engine))
        } else {
            passes.append(TranscriptionPass(engine: "qwen", model: "Qwen/Qwen3-ASR-0.6B"))
        }
    }
}

@MainActor
struct ModelLibrarySheet: View {
    @ObservedObject var state: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var models: [LibraryModel]?
    @State private var cacheError: String?
    @State private var refresh = 0

    private struct RefreshKey: Equatable {
        let request: Int
        let activeDownload: String?
        let running: Bool
        let modelActivity: Bool
    }

    var body: some View {
        let key = RefreshKey(request: refresh, activeDownload: state.downloadingLibraryModel,
                             running: state.isRunning, modelActivity: state.showModelActivity)
        VStack(spacing: 0) {
            Text("Model Library")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding([.horizontal, .top])
            Form {
                if state.showModelActivity || state.downloadingLibraryModel != nil {
                    activity
                }
                if let cacheError {
                    Section {
                        Text(cacheError).foregroundStyle(.secondary)
                        Button("Refresh") { refresh += 1 }
                    }
                } else if let models {
                    modelSection("On This Mac", models: models.filter(\.cached))
                    modelSection("Available to Download", models: models.filter { !$0.cached && !$0.isLocal })
                    modelSection("Missing Local Models", models: models.filter { !$0.cached && $0.isLocal })
                } else {
                    ProgressView("Checking model cache…")
                }
                if #available(macOS 26, *) {
                    Section("System Model") {
                        LabeledContent("Apple SpeechTranscriber", value: "Managed by macOS")
                    }
                }
            }
            .formStyle(.grouped)
            .frame(minHeight: 300, idealHeight: 390, maxHeight: 470)

            Divider()
            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding()
        }
        .frame(width: 540)
        .task(id: key) {
            do {
                let models = try await state.modelLibrary()
                guard !Task.isCancelled else { return }
                self.models = models
                cacheError = nil
            } catch where !Task.isCancelled {
                cacheError = error.localizedDescription
            } catch {}
        }
    }

    private var activity: some View {
        Section(state.modelActivityTitle) {
            Text(state.modelActivityMessage)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            if !state.modelActivityCanDismiss {
                if let progress = state.modelActivityProgress {
                    ProgressView(value: progress)
                } else {
                    ProgressView()
                }
                if state.downloadingLibraryModel != nil {
                    Button(state.isCancelling ? "Stopping…" : "Cancel Download") {
                        state.cancelLibraryDownload()
                    }
                    .disabled(state.isCancelling)
                }
            }
            if !state.modelActivityLog.isEmpty {
                DisclosureGroup("Download Log") {
                    ScrollView {
                        Text(state.modelActivityLog.joined(separator: "\n"))
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(height: 130)
                }
            }
        }
    }

    @ViewBuilder
    private func modelSection(_ title: String, models: [LibraryModel]) -> some View {
        if !models.isEmpty {
            Section(title) {
                ForEach(models) { model in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.description)
                            if model.description != model.id {
                                Text(model.id)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                        }
                        Spacer()
                        if model.cached {
                            if let path = model.path {
                                Button("Show in Finder") {
                                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                                }
                                .accessibilityLabel("Show \(model.description) in Finder")
                            } else {
                                Text("Ready").foregroundStyle(.secondary)
                            }
                        } else if !model.isLocal {
                            Button("Download") {
                                Task { await state.fetchLibraryModel(model) }
                            }
                            .disabled(state.isRunning || state.downloadingLibraryModel != nil)
                            .accessibilityLabel("Download \(model.description)")
                        }
                    }
                }
            }
        }
    }
}
