// Native Apple SpeechTranscriber adapter for macOS 26 and later.
//
// This intentionally stays as a small executable so the Rust CLI can invoke
// Apple's system-managed model without linking Speech.framework itself. The
// model is downloaded and updated by macOS through AssetInventory; no model
// weights are bundled with Transcriberrr.
import AVFoundation
import CoreMedia
import Foundation
import Speech

private struct NativeWord: Codable {
    var text: String
    let start: Double
    let end: Double
    var speaker: Int?

    private enum CodingKeys: String, CodingKey { case text, start, end }
}

private struct NativeSegment: Encodable {
    let text: String
    let start: Double
    let end: Double
    var speaker: Int?
    var words: [NativeWord]

    private enum CodingKeys: String, CodingKey { case text, start, end, words }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(start, forKey: .start)
        try container.encode(end, forKey: .end)
        if words.isEmpty {
            try container.encode(text, forKey: .text)
        } else {
            try container.encode(words, forKey: .words)
        }
    }
}

private struct NativeTranscript: Encodable {
    let source: String
    let model: String
    let device: String
    let language: String
    let duration_seconds: Double
    let text: String
    var segments: [NativeSegment]
    let metadata: NativeMetadata

    private enum CodingKeys: String, CodingKey {
        case source, model, device, language, duration_seconds, segments, metadata
    }
}

private struct NativeMetadata: Codable {
    let transcribedWith: String
    let version: String
    let sourceCreatedAt: String?
    let sourceModifiedAt: String?
    let transcribedAt: String
    let model: String
    let diarization: NativeDiarizationMetadata?

    private enum CodingKeys: String, CodingKey {
        case transcribedWith = "transcribed-with"
        case version
        case sourceCreatedAt = "source_created_at"
        case sourceModifiedAt = "source_modified_at"
        case transcribedAt = "transcribed_at"
        case model
        case diarization
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(transcribedWith, forKey: .transcribedWith)
        try container.encode(version, forKey: .version)
        try container.encodeIfPresent(sourceCreatedAt, forKey: .sourceCreatedAt)
        try container.encodeIfPresent(sourceModifiedAt, forKey: .sourceModifiedAt)
        try container.encode(transcribedAt, forKey: .transcribedAt)
        try container.encode(model, forKey: .model)
        try container.encodeIfPresent(diarization, forKey: .diarization)
    }
}

private struct NativeDiarizationMetadata: Codable {
    let enabled: Bool
    let speakers: Int?
}

private let transcriberrrVersion = "0.1.0"

private enum OutputFormat: String {
    case txt, json, srt, vtt
}

private enum TimestampMode: String {
    case none, words, sentences
}

private enum NativeError: LocalizedError {
    case usage
    case unavailable
    case unsupportedLocale
    case unsupportedDiarization
    case missingTimestamps

    var errorDescription: String? {
        switch self {
        case .usage:
            return "usage: transcriberrr-native INPUT [--output PATH]"
        case .unavailable:
            return "Apple SpeechTranscriber is unavailable on this Mac or requires macOS 26 or later"
        case .unsupportedLocale:
            return "Apple SpeechTranscriber does not support English (en-US) on this Mac"
        case .missingTimestamps:
            return "Apple SpeechTranscriber returned text without word timestamps"
        case .unsupportedDiarization:
            return "Apple SpeechTranscriber does not provide speaker diarization; use the Qwen engine with --diarize"
        }
    }
}

@main
struct TranscriberrrNative {
    static func main() async {
        do {
            let (input, output, format, timestamps) = try parseArguments()
            guard #available(macOS 26.0, *) else { throw NativeError.unavailable }
            let transcript = try await transcribe(input: input)
            let data = try render(transcript: transcript, format: format, timestamps: timestamps)
            if let output {
                try data.write(to: output, options: .atomic)
            } else {
                FileHandle.standardOutput.write(data)
                FileHandle.standardOutput.write(Data([0x0a]))
            }
        } catch {
            FileHandle.standardError.write(Data("transcriberrr-native: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    private static func parseArguments() throws -> (URL, URL?, OutputFormat, TimestampMode) {
        var arguments = Array(CommandLine.arguments.dropFirst())
        guard let inputPath = arguments.first else { throw NativeError.usage }
        arguments.removeFirst()
        var output: URL?
        var format: OutputFormat = .json
        var timestamps: TimestampMode = .words
        while !arguments.isEmpty {
            let option = arguments.removeFirst()
            switch option {
            case "--output":
                guard let path = arguments.first else { throw NativeError.usage }
                arguments.removeFirst()
                output = URL(fileURLWithPath: path)
            case "--format":
                guard let value = arguments.first, let parsed = OutputFormat(rawValue: value.lowercased()) else { throw NativeError.usage }
                arguments.removeFirst()
                format = parsed
            case "--timestamps":
                guard let value = arguments.first, let parsed = TimestampMode(rawValue: value.lowercased()) else { throw NativeError.usage }
                arguments.removeFirst()
                timestamps = parsed
            case "--diarize", "--speakers":
                throw NativeError.unsupportedDiarization
            default:
                throw NativeError.usage
            }
        }
        return (URL(fileURLWithPath: inputPath), output, format, timestamps)
    }

    private static func render(transcript: NativeTranscript, format: OutputFormat, timestamps: TimestampMode) throws -> Data {
        switch format {
        case .json:
            var document = transcript
            if timestamps != .words {
                document.segments = document.segments.map { segment in
                    var segment = segment
                    segment.words = []
                    return segment
                }
            }
            return try JSONEncoder().encode(document)
        case .txt:
            return Data((transcript.text + "\n").utf8)
        case .srt, .vtt:
            var output = format == .vtt ? "WEBVTT\n\n" : ""
            let cues: [(Double, Double, String)] = timestamps == .words
                ? transcript.segments.flatMap { $0.words.map { ($0.start, $0.end, $0.text.trimmingCharacters(in: .whitespacesAndNewlines)) } }
                : transcript.segments.map { ($0.start, $0.end, $0.text) }
            for (index, cue) in cues.enumerated() {
                if format == .srt { output += "\(index + 1)\n" }
                let cueText = cue.2.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.joined(separator: " ")
                output += "\(timestamp(cue.0, format: format)) --> \(timestamp(cue.1, format: format))\n\(cueText)\n\n"
            }
            return Data(output.utf8)
        }
    }

    private static func timestamp(_ seconds: Double, format: OutputFormat) -> String {
        let milliseconds = UInt64((max(0, seconds) * 1000.0).rounded())
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds % 3_600_000) / 60_000
        let secondsPart = (milliseconds % 60_000) / 1000
        let millis = milliseconds % 1000
        return format == .vtt
            ? String(format: "%02llu:%02llu:%02llu.%03llu", hours, minutes, secondsPart, millis)
            : String(format: "%02llu:%02llu:%02llu,%03llu", hours, minutes, secondsPart, millis)
    }

    @available(macOS 26.0, *)
    private static func transcribe(input: URL) async throws -> NativeTranscript {
        guard SpeechTranscriber.isAvailable else { throw NativeError.unavailable }
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "en-US")) else {
            throw NativeError.unsupportedLocale
        }

        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]
        )
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }

        let audioFile = try AVAudioFile(forReading: input)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let resultTask = Task { () throws -> [NativeWord] in
            var words: [NativeWord] = []
            var pendingPrefix = ""
            for try await result in transcriber.results {
                var firstRun = true
                for run in result.text.runs {
                    var text = String(result.text[run.range].characters)
                    guard !text.isEmpty else { continue }
                    if firstRun, let previous = words.last,
                       previous.text.last?.isWhitespace == false,
                       text.first?.isWhitespace == false {
                        text = " " + text
                    }
                    firstRun = false
                    guard let timeRange = run.audioTimeRange,
                          !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    else {
                        if text.contains(where: { $0.isLetter || $0.isNumber }) {
                            throw NativeError.missingTimestamps
                        }
                        // Untimed punctuation and spaces still belong to the transcript.
                        if !words.isEmpty {
                            words[words.count - 1].text += text
                        } else {
                            pendingPrefix += text
                        }
                        continue
                    }
                    text = pendingPrefix + text
                    pendingPrefix = ""
                    let start = CMTimeGetSeconds(timeRange.start)
                    let end = CMTimeGetSeconds(timeRange.end)
                    words.append(NativeWord(
                        text: text,
                        start: start.isFinite ? start : 0,
                        end: end.isFinite ? end : start,
                        speaker: nil
                    ))
                }
            }
            if !pendingPrefix.isEmpty { throw NativeError.missingTimestamps }
            return words
        }
        if let lastSample = try await analyzer.analyzeSequence(from: audioFile) {
            try await analyzer.finalizeAndFinish(through: lastSample)
        } else {
            await analyzer.cancelAndFinishNow()
        }
        let words = try await resultTask.value
        let text = words.map(\.text).joined()
        let segments = buildSegments(words)
        let duration = Double(audioFile.length) / audioFile.processingFormat.sampleRate
        let resourceValues = try? input.resourceValues(forKeys: [.creationDateKey, .contentModificationDateKey])
        return NativeTranscript(
            source: input.path,
            model: "Apple SpeechTranscriber",
            device: "apple-native",
            language: locale.identifier,
            duration_seconds: duration.isFinite ? duration : (words.last?.end ?? 0),
            text: text,
            segments: segments,
            metadata: NativeMetadata(
                transcribedWith: "transcriberrr",
                version: transcriberrrVersion,
                sourceCreatedAt: resourceValues?.creationDate.map(iso8601),
                sourceModifiedAt: resourceValues?.contentModificationDate.map(iso8601),
                transcribedAt: iso8601(Date()),
                model: "Apple SpeechTranscriber",
                diarization: nil
            )
        )
    }

    private static func iso8601(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }

    private static func buildSegments(_ words: [NativeWord]) -> [NativeSegment] {
        var segments: [NativeSegment] = []
        var current: [NativeWord] = []
        for word in words {
            let split = current.last.map { previous in
                let ending = previous.text.trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: "”’\"')]}」』")))
                return word.start - previous.end > 1.0 || ending.hasSuffix(".") || ending.hasSuffix("?") || ending.hasSuffix("!")
            } ?? false
            if split, !current.isEmpty {
                segments.append(segment(current))
                current.removeAll(keepingCapacity: true)
            }
            current.append(word)
        }
        if !current.isEmpty { segments.append(segment(current)) }
        return segments
    }

    private static func segment(_ words: [NativeWord]) -> NativeSegment {
        NativeSegment(
            text: words.map(\.text).joined(),
            start: words.first?.start ?? 0,
            end: words.map(\.end).max() ?? words.first?.start ?? 0,
            speaker: nil,
            words: words
        )
    }
}
