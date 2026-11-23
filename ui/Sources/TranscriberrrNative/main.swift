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

struct NativeWord: Encodable {
    var text: String
    let start: Double
    let end: Double
}

private struct NativeSegment: Encodable {
    let start: Double
    let end: Double
    let words: [NativeWord]

    var text: String { words.lazy.map(\.text).joined() }

    private enum CodingKeys: String, CodingKey { case text, start, end, words }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(start, forKey: .start)
        try container.encode(end, forKey: .end)
        if encoder.userInfo[.omitWords] as? Bool == true {
            try container.encode(text, forKey: .text)
        } else {
            try container.encode(words, forKey: .words)
        }
    }
}

private extension CodingUserInfoKey {
    static let omitWords = CodingUserInfoKey(rawValue: "omitWords")!
}

private struct NativeTranscript: Encodable {
    let source: String
    let model: String
    let device: String
    let language: String
    let duration_seconds: Double
    let segments: [NativeSegment]
    let metadata: NativeMetadata
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

private let transcriberrrVersion = "0.4.0"

private enum OutputFormat: String {
    case txt, json, srt, vtt
}

private enum TimestampMode: String {
    case none, words, sentences
}

enum NativeError: LocalizedError {
    case usage
    case unavailable
    case unsupportedLocale
    case assetsNotInstalled
    case unsupportedDiarization
    case missingTimestamps
    case outputIsInput

    var errorDescription: String? {
        switch self {
        case .usage:
            return "usage: transcriberrr-native INPUT [--output PATH] [--format txt|json|srt|vtt] [--timestamps none|words|sentences] [--offline]"
        case .unavailable:
            return "Apple SpeechTranscriber is unavailable on this Mac or requires macOS 26 or later"
        case .unsupportedLocale:
            return "Apple SpeechTranscriber does not support English (en-US) on this Mac"
        case .assetsNotInstalled:
            return "Apple SpeechTranscriber assets are not installed. Run once without --offline to install them."
        case .missingTimestamps:
            return "Apple SpeechTranscriber returned text without word timestamps"
        case .outputIsInput:
            return "Choose a transcript output path different from the input recording"
        case .unsupportedDiarization:
            return "Apple SpeechTranscriber does not provide speaker diarization; use the Qwen engine with --diarize"
        }
    }
}

@main
struct TranscriberrrNative {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            if arguments == ["--help"] || arguments == ["-h"] {
                print(NativeError.usage.localizedDescription)
                return
            }
            if arguments.first == "stream" || arguments == ["--stream-supported"] {
                guard #available(macOS 26.0, *) else { throw NativeError.unavailable }
                switch arguments {
                case ["--stream-supported"]:
                    _ = try await LiveSpeech.locale()
                case ["stream"], ["stream", "--offline"]:
                    do {
                        try await LiveSpeech.run(offline: arguments.contains("--offline"))
                    } catch {
                        LiveSpeech.emitStartupError(error)
                        throw error
                    }
                default:
                    throw NativeError.usage
                }
                return
            }
            let (input, output, format, timestamps, offline) = try parseArguments()
            if let output {
                var source = stat()
                var destination = stat()
                if stat(input.path, &source) == 0, stat(output.path, &destination) == 0,
                   source.st_dev == destination.st_dev, source.st_ino == destination.st_ino {
                    throw NativeError.outputIsInput
                }
            }
            guard #available(macOS 26.0, *) else { throw NativeError.unavailable }
            let transcript = try await transcribe(input: input, offline: offline)
            let data = try render(transcript: transcript, format: format, timestamps: timestamps)
            if let output {
                try data.write(to: output, options: .atomic)
            } else {
                try FileHandle.standardOutput.write(contentsOf: data)
                if data.last != 0x0a {
                    try FileHandle.standardOutput.write(contentsOf: Data([0x0a]))
                }
            }
        } catch {
            FileHandle.standardError.write(Data("transcriberrr-native: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    private static func parseArguments() throws -> (URL, URL?, OutputFormat, TimestampMode, Bool) {
        var arguments = Array(CommandLine.arguments.dropFirst())
        guard let inputPath = arguments.first else { throw NativeError.usage }
        arguments.removeFirst()
        var output: URL?
        var format: OutputFormat = .json
        var timestamps: TimestampMode = .words
        var offline = false
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
            case "--offline":
                offline = true
            case "--diarize", "--speakers":
                throw NativeError.unsupportedDiarization
            default:
                throw NativeError.usage
            }
        }
        return (URL(fileURLWithPath: inputPath), output, format, timestamps, offline)
    }

    private static func render(transcript: NativeTranscript, format: OutputFormat, timestamps: TimestampMode) throws -> Data {
        switch format {
        case .json:
            let encoder = JSONEncoder()
            encoder.userInfo[.omitWords] = timestamps != .words
            return try encoder.encode(transcript)
        case .txt:
            return Data((transcript.segments.lazy.map(\.text).joined() + "\n").utf8)
        case .srt, .vtt:
            var output = format == .vtt ? "WEBVTT\n\n" : ""
            var index = 0
            func appendCue(start: Double, end: Double, text: String) {
                var cueText = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
                guard !cueText.isEmpty else { return }
                cueText = cueText.replacingOccurrences(of: "-->", with: "->")
                if format == .vtt {
                    cueText = cueText.replacingOccurrences(of: "&", with: "&amp;")
                        .replacingOccurrences(of: "<", with: "&lt;")
                        .replacingOccurrences(of: ">", with: "&gt;")
                } else {
                    index += 1
                    output += "\(index)\n"
                }
                let first = UInt64((max(0, start) * 1000).rounded())
                let last = max(first + 1, UInt64((max(0, end) * 1000).rounded()))
                output += "\(timestamp(first, format: format)) --> \(timestamp(last, format: format))\n\(cueText)\n\n"
            }
            for segment in transcript.segments {
                if timestamps == .words {
                    for word in segment.words {
                        appendCue(start: word.start, end: word.end, text: word.text)
                    }
                } else {
                    appendCue(start: segment.start, end: segment.end, text: segment.text)
                }
            }
            return Data(output.utf8)
        }
    }

    private static func timestamp(_ milliseconds: UInt64, format: OutputFormat) -> String {
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds % 3_600_000) / 60_000
        let secondsPart = (milliseconds % 60_000) / 1000
        let millis = milliseconds % 1000
        return format == .vtt
            ? String(format: "%02llu:%02llu:%02llu.%03llu", hours, minutes, secondsPart, millis)
            : String(format: "%02llu:%02llu:%02llu,%03llu", hours, minutes, secondsPart, millis)
    }

    @available(macOS 26.0, *)
    private static func transcribe(input: URL, offline: Bool) async throws -> NativeTranscript {
        let audioFile = try AVAudioFile(forReading: input)
        let frames = audioFile.length
        let duration = Double(frames) / audioFile.processingFormat.sampleRate
        let words = frames > 0 ? try await transcribeAudio(audioFile, offline: offline) : []
        let resourceValues = try? input.resourceValues(forKeys: [.creationDateKey, .contentModificationDateKey])
        return NativeTranscript(
            source: input.path,
            model: "Apple SpeechTranscriber",
            device: "apple-native",
            // Same label as the other engines (the CLI is English-only).
            language: "English",
            duration_seconds: duration.isFinite ? duration : (words.last?.end ?? 0),
            segments: buildSegments(words),
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

    @available(macOS 26.0, *)
    private static func transcribeAudio(_ audioFile: AVAudioFile, offline: Bool) async throws -> [NativeWord] {
        guard SpeechTranscriber.isAvailable else { throw NativeError.unavailable }
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "en-US")) else {
            throw NativeError.unsupportedLocale
        }

        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange]
        )
        if offline {
            guard await AssetInventory.status(forModules: [transcriber]) == .installed else {
                throw NativeError.assetsNotInstalled
            }
        } else if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        do {
            try await analyzer.start(inputAudioFile: audioFile, finishAfterFile: true)
            return try await collectWords(from: transcriber)
        } catch {
            await analyzer.cancelAndFinishNow()
            throw error
        }
    }

    @available(macOS 26.0, *)
    private static func collectWords(from transcriber: SpeechTranscriber) async throws -> [NativeWord] {
        var words: [NativeWord] = []
        var pendingPrefix = ""
        for try await result in transcriber.results {
            var firstRun = true
            for run in result.text.runs {
                var text = String(result.text[run.range].characters)
                guard !text.isEmpty else { continue }
                if firstRun, let previous = words.last,
                   previous.text.last?.isWhitespace == false,
                   let first = text.first, !first.isWhitespace,
                   !".,!?;:%)]}’”…。？！".contains(first) {
                    text = " " + text
                }
                firstRun = false
                guard let timeRange = run.audioTimeRange,
                      text.contains(where: { !$0.isWhitespace })
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
                // Keep timings finite and ordered: the Rust side rejects a document with a
                // NaN, a backwards start, or end < start, losing the whole transcript.
                let rawStart = CMTimeGetSeconds(timeRange.start)
                let rawEnd = CMTimeGetSeconds(timeRange.end)
                let start = max(rawStart.isFinite ? rawStart : 0, words.last?.start ?? 0)
                let end = max(rawEnd.isFinite ? rawEnd : start, start)
                words.append(NativeWord(
                    text: text,
                    start: start,
                    end: end
                ))
            }
        }
        // Whitespace-only results have no timed words (for example, a silent file).
        if pendingPrefix.contains(where: { !$0.isWhitespace }) { throw NativeError.missingTimestamps }
        return words
    }

    private static func iso8601(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        // Whole seconds, like the other engines: merges order `transcribed_at` as text.
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }

    private static func buildSegments(_ words: [NativeWord]) -> [NativeSegment] {
        var segments: [NativeSegment] = []
        var current: [NativeWord] = []
        let trailingPunctuation = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "”’\"')]}」』"))
        for word in words {
            let split = current.last.map { previous in
                if word.start - previous.end > 1.0 { return true }
                let ending = previous.text.trimmingCharacters(in: trailingPunctuation)
                guard let finalCharacter = ending.last else { return false }
                return "。？！".contains(finalCharacter)
                    || (".?!".contains(finalCharacter)
                        && (previous.text.last?.isWhitespace == true || word.text.first?.isWhitespace == true))
            } ?? false
            if split {
                segments.append(segment(current))
                current = []
            }
            current.append(word)
        }
        if !current.isEmpty { segments.append(segment(current)) }
        return segments
    }

    private static func segment(_ words: [NativeWord]) -> NativeSegment {
        NativeSegment(
            start: words.first?.start ?? 0,
            end: words.lazy.map(\.end).max() ?? 0,
            words: words
        )
    }
}
