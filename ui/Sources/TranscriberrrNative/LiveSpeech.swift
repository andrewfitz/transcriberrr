// One process serves independent SpeechAnalyzer streams over stdin/stdout.
import AVFoundation
import CoreMedia
import Dispatch
import Foundation
import Speech
import Synchronization

@available(macOS 26.0, *)
enum LiveSpeech {
    private enum StreamError: LocalizedError {
        case malformedPacket, audioConversion, inputOverloaded, streamLimit
        case parentMismatch(expected: pid_t, actual: pid_t)

        var errorDescription: String? {
            switch self {
            case .malformedPacket: "Invalid Apple Speech stream packet"
            case .audioConversion: "Could not convert audio for Apple Speech"
            case .inputOverloaded: "Apple Speech input queue is full"
            case .streamLimit: "Apple Speech helper stream limit reached"
            case let .parentMismatch(expected, actual):
                "Apple Speech helper expected parent process \(expected) from TRANSCRIBERRR_NATIVE_PARENT_PID, "
                    + "but its parent is \(actual). The launcher exited or the helper was started through another process"
            }
        }
    }

    private struct Event: Encodable {
        let kind: String
        var id: UInt64 = 0
        var text: String? = nil
        var words: [NativeWord]? = nil
        var error: String? = nil
    }

    // Results from multiple analyzer tasks share one pipe. Lock the whole line.
    private static let stdoutLock = NSLock()

    private static func emit(_ event: Event) {
        stdoutLock.withLock {
            do {
                var data = try JSONEncoder().encode(event)
                data.append(10)
                try FileHandle.standardOutput.write(contentsOf: data)
            } catch {
                exit(1)
            }
        }
    }

    static func emitStartupError(_ error: Error) {
        emit(Event(kind: "error", error: error.localizedDescription))
    }

    static func locale() async throws -> Locale {
        guard SpeechTranscriber.isAvailable else { throw NativeError.unavailable }
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "en-US")) else {
            throw NativeError.unsupportedLocale
        }
        return locale
    }

    private static func transcriber(locale: Locale, lowLatency: Bool) -> SpeechTranscriber {
        SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: lowLatency ? [.volatileResults, .fastResults] : [],
            attributeOptions: [.audioTimeRange]
        )
    }

    static func run(offline: Bool) async throws {
        let parentWatcher = try watchParent()
        defer { parentWatcher?.cancel() }
        let locale = try await locale()
        let module = transcriber(locale: locale, lowLatency: true)
        if offline {
            guard await AssetInventory.status(forModules: [module]) == .installed else {
                throw NativeError.assetsNotInstalled
            }
        } else if let request = try await AssetInventory.assetInstallationRequest(supporting: [module]) {
            try await request.downloadAndInstall()
        }
        guard let source = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: false),
              let preferred = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [module], considering: source)
        else { throw StreamError.audioConversion }
        let options = SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .processLifetime)
        let warmup = SpeechAnalyzer(modules: [module], options: options)
        try await warmup.prepareToAnalyze(in: preferred)
        await warmup.cancelAndFinishNow()

        // Reading stdin is blocking. Keep it off Swift's cooperative task pool.
        let reader = Thread {
            do {
                try readPackets(locale: locale, source: source, preferred: preferred, options: options)
                exit(0)
            } catch {
                FileHandle.standardError.write(Data("transcriberrr-native: \(error.localizedDescription)\n".utf8))
                exit(1)
            }
        }
        emit(Event(kind: "ready"))
        reader.start()
        // Process termination belongs to the reader once it closes the input pipe.
        for await _ in AsyncStream<Void>(unfolding: {
            try? await Task.sleep(for: .seconds(3600))
        }) {}
    }

    private static func watchParent() throws -> (any DispatchSourceProcess)? {
        guard let value = ProcessInfo.processInfo.environment["TRANSCRIBERRR_NATIVE_PARENT_PID"],
              let parent = pid_t(value), parent > 1
        else { return nil }
        // The reader starts after warmup. Observe parent exit during startup, too.
        let watcher = DispatchSource.makeProcessSource(identifier: parent, eventMask: .exit)
        watcher.setEventHandler { exit(0) }
        watcher.activate()
        // The launcher may have exited before the helper registered the watcher, or an
        // intermediate process started the helper. The caller reports the error on stdout
        // and stderr and exits non-zero, so the server's startup error names the cause.
        let actual = getppid()
        if actual != parent {
            watcher.cancel()
            throw StreamError.parentMismatch(expected: parent, actual: actual)
        }
        return watcher
    }

    private static func readPackets(
        locale: Locale, source: AVAudioFormat, preferred: AVAudioFormat, options: SpeechAnalyzer.Options
    ) throws {
        var streams: [UInt64: Stream] = [:]
        // Header: operation byte, little-endian stream ID, little-endian payload length.
        // Payload for append is mono 16 kHz signed little-endian PCM.
        while try autoreleasepool(invoking: {
            guard let packet = try readExact(13, allowEOF: true) else { return false }
            let operation = packet[0]
            let id = packet.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 1, as: UInt64.self)) }
            let length = packet.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 9, as: UInt32.self)) }
            guard length <= 64 * 1024, operation == 2 || length == 0,
                  let payload = try readExact(Int(length))
            else { throw StreamError.malformedPacket }

            do {
                switch operation {
                case 1, 5:
                    streams = streams.filter { !$0.value.completion.finished.withLock { $0 } }
                    guard streams[id] == nil else { throw StreamError.malformedPacket }
                    // The server admits 64 tracks. A bleed re-decode opens a committed
                    // stream beside them, so leave room for it.
                    guard streams.count < 72 else { throw StreamError.streamLimit }
                    let stream = try Stream(
                        id: id, locale: locale, lowLatency: operation == 1, source: source,
                        preferred: preferred, options: options
                    )
                    streams[id] = stream
                case 2:
                    try streams[id]?.append(payload)
                case 3:
                    try streams[id]?.finish()
                case 4:
                    streams.removeValue(forKey: id)?.cancel()
                default:
                    throw StreamError.malformedPacket
                }
            } catch {
                streams.removeValue(forKey: id)?.cancel()
                emit(Event(kind: "error", id: id, error: error.localizedDescription))
            }
            return true
        }) {}
    }

    private static func readExact(_ count: Int, allowEOF: Bool = false) throws -> Data? {
        guard count > 0 else { return Data() }
        let input = FileHandle.standardInput
        guard var bytes = try input.read(upToCount: count), !bytes.isEmpty else {
            if allowEOF { return nil }
            throw StreamError.malformedPacket
        }
        // Socket reads usually contain the complete header or payload. Keep that allocation.
        while bytes.count < count {
            guard let next = try input.read(upToCount: count - bytes.count), !next.isEmpty else {
                throw StreamError.malformedPacket
            }
            bytes.append(next)
        }
        return bytes
    }

    private final class Completion: Sendable {
        let finished = Mutex(false)
    }

    private final class Stream {
        let source: AVAudioFormat
        let converter: Converter?
        let input: AsyncStream<AnalyzerInput>.Continuation
        let analyzer: SpeechAnalyzer
        let task: Task<Void, Never>
        let completion = Completion()
        var hasAudio = false

        init(
            id: UInt64, locale: Locale, lowLatency: Bool, source: AVAudioFormat, preferred: AVAudioFormat,
            options: SpeechAnalyzer.Options
        ) throws {
            self.source = source
            if source == preferred {
                converter = nil
            } else {
                guard let audioConverter = AVAudioConverter(from: source, to: preferred) else {
                    throw StreamError.audioConversion
                }
                audioConverter.primeMethod = .none
                converter = Converter(audioConverter)
            }

            let module = transcriber(locale: locale, lowLatency: lowLatency)
            let analyzer = SpeechAnalyzer(modules: [module], options: options)
            self.analyzer = analyzer
            // Live audio arrives in 32 ms packets, so the longest turn (60 s) is 1,875 of
            // them. A replay after a reconnect can deliver all of it before the analyzer
            // reads any, and a full queue discards the turn.
            let (sequence, input) = AsyncStream<AnalyzerInput>.makeStream(bufferingPolicy: .bufferingOldest(2048))
            self.input = input
            let completion = completion
            task = Task {
                let terminal: Event
                do {
                    try Task.checkCancellation()
                    async let results = collect(id: id, module: module, reportPartials: lowLatency)
                    if let end = try await analyzer.analyzeSequence(sequence) {
                        try await analyzer.finalizeAndFinish(through: end)
                        let (text, words) = try await results
                        terminal = Event(kind: "final", id: id, text: text, words: words)
                    } else {
                        await analyzer.cancelAndFinishNow()
                        terminal = Event(kind: "final", id: id, text: "", words: [])
                    }
                } catch {
                    await analyzer.cancelAndFinishNow()
                    terminal = Event(kind: "error", id: id, error: error.localizedDescription)
                }
                completion.finished.withLock { $0 = true }
                if !Task.isCancelled { emit(terminal) }
            }
        }

        func append(_ pcm: Data) throws {
            guard pcm.count.isMultiple(of: 2), !pcm.isEmpty,
                  let buffer = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(pcm.count / 2)),
                  let samples = buffer.int16ChannelData?[0]
            else { throw StreamError.audioConversion }
            buffer.frameLength = buffer.frameCapacity
            pcm.copyBytes(to: UnsafeMutableRawBufferPointer(start: samples, count: pcm.count))
            hasAudio = true
            if let converter {
                if let converted = try converter.convert(buffer) {
                    try enqueue(converted)
                }
            } else {
                try enqueue(buffer)
            }
        }

        func finish() throws {
            if hasAudio, let converter {
                while let converted = try converter.convert(nil) {
                    try enqueue(converted)
                }
            }
            input.finish()
        }

        func cancel() {
            task.cancel()
            input.finish()
            let analyzer = analyzer
            Task { await analyzer.cancelAndFinishNow() }
        }

        private func enqueue(_ buffer: AVAudioPCMBuffer) throws {
            if case .dropped = input.yield(AnalyzerInput(buffer: buffer)) {
                throw StreamError.inputOverloaded
            }
        }
    }

    private final class Converter {
        private let converter: AVAudioConverter

        init(_ converter: AVAudioConverter) {
            self.converter = converter
        }

        func convert(_ buffer: AVAudioPCMBuffer?) throws -> AVAudioPCMBuffer? {
            let ratio = converter.outputFormat.sampleRate / converter.inputFormat.sampleRate
            let capacity = AVAudioFrameCount((Double(buffer?.frameLength ?? 0) * ratio).rounded(.up)) + 1024
            guard let output = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: capacity) else {
                throw StreamError.audioConversion
            }
            var input = buffer
            let status = converter.convert(to: output, error: nil) { _, status in
                guard let next = input else {
                    status.pointee = buffer == nil ? .endOfStream : .noDataNow
                    return nil
                }
                input = nil
                status.pointee = .haveData
                return next
            }
            guard status != .error else { throw StreamError.audioConversion }
            return output.frameLength > 0 ? output : nil
        }
    }

    private static func collect(id: UInt64, module: SpeechTranscriber, reportPartials: Bool) async throws -> (String, [NativeWord]) {
        var text = ""
        var words: [NativeWord] = []
        var pending = ""
        var hasFullTiming = true
        for try await result in module.results {
            try Task.checkCancellation()
            let fragment = String(result.text.characters)
            let prefix = separator(between: text, and: fragment)
            if result.isFinal {
                text += prefix + fragment
                if hasFullTiming {
                    hasFullTiming = appendTimedWords(result.text, prefix: prefix, words: &words, pending: &pending)
                }
                if reportPartials {
                    emit(Event(kind: "stable", id: id, text: text, words: hasFullTiming && pending.isEmpty ? words : []))
                }
            } else if reportPartials {
                emit(Event(kind: "draft", id: id, text: text + prefix + fragment))
            }
        }
        try Task.checkCancellation()
        return (text, hasFullTiming && pending.isEmpty ? words : [])
    }

    private static func separator(between previous: String, and next: String) -> String {
        guard previous.last?.isWhitespace == false, let first = next.first,
              !first.isWhitespace, !".,!?;:%…)]}’”".contains(first)
        else { return "" }
        return " "
    }

    private static func appendTimedWords(
        _ text: AttributedString, prefix: String, words: inout [NativeWord], pending: inout String
    ) -> Bool {
        var separator = prefix
        for run in text.runs {
            let fragment = separator + String(text[run.range].characters)
            separator = ""
            if let range = run.audioTimeRange, fragment.contains(where: { !$0.isWhitespace }) {
                let start = range.start.seconds
                let end = range.end.seconds
                guard start.isFinite, end.isFinite, start >= (words.last?.start ?? 0), end >= start else {
                    return false
                }
                words.append(NativeWord(text: pending + fragment, start: start, end: end))
                pending = ""
            } else {
                if fragment.contains(where: { $0.isLetter || $0.isNumber }) { return false }
                if words.isEmpty {
                    pending += fragment
                } else {
                    words[words.count - 1].text += fragment
                }
            }
        }
        return true
    }
}
