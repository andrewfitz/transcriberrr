# Transcriberrr

Offline English transcription for Apple Silicon Macs. Transcriberrr ships a native Rust CLI and a small SwiftUI drag-and-drop app. Audio stays on the Mac; model weights are downloaded to a user-owned cache on first use.

## Features

- Qwen3-ASR 1.7B (default), the latest English Parakeet Unified 0.6B export, or Apple SpeechTranscriber (macOS 26+).
- Metal GPU execution on Apple Silicon, with a CPU fallback for the Rust engines.
- Compact word-level JSON, sentence/segment or word timestamps in TXT, SRT, and WebVTT, and plain TXT.
- Convert saved transcript JSON between formats offline, without loading audio or models.
- Optional Sortformer speaker diarization, including a per-job speaker limit.
- Single files, recursive folders, JSON manifests, resumable JSONL progress, and output beside the source or in one destination folder.
- Native SwiftUI queue with folder drop, regex filtering, visible Check all/None, per-track diarization, saved queue results, elapsed time and ETA per track and selected queue, and a visible model setup/download activity window.

## Install a GitHub release

Download the arm64 `.dmg` from the repository Releases page, open it, and drag `Transcriberrr.app` to `/Applications`. The app bundle contains the UI, Rust CLI, and Apple-native helper; model weights are intentionally not bundled. The first run may download several gigabytes from Hugging Face or install Apple system models. macOS may require opening the app once in System Settings -> Privacy & Security for an unsigned development release.

Release archives should be built on an Apple Silicon Mac and distributed with a SHA-256 checksum. Do not commit `target/`, `ui/.build/`, the unpacked `dist/Transcriberrr.app/`, recordings, or model caches; attach only the final `.dmg` and `.dmg.sha256` files to a GitHub Release.

## Build from source

Requirements: Apple Silicon macOS, Xcode Command Line Tools, Swift 6, and Rust 1.99 or newer. `rust-toolchain.toml` selects the stable Rust toolchain.

```bash
# CLI with Parakeet Unified and Sortformer support
cargo build --release --features diarization

# Native app bundle; also embeds the matching CLI and SpeechTranscriber helper
./scripts/build-macos-app.sh
open dist/Transcriberrr.app

# Build the GitHub release installer and checksum
./scripts/build-release-dmg.sh
```

The build script refuses non-`Darwin arm64` hosts. For a CLI-only build, use `cargo build --release`; add `--features parakeet` for the Parakeet Unified engine without diarization. `cargo test`, `cargo fmt --check`, and `cargo clippy --all-targets --all-features -- -D warnings` are the required checks.

## CLI quick start

```bash
# JSON with word timestamps (the defaults)
./target/release/transcriberrr interview.m4a --output-dir transcripts

# Plain text
./target/release/transcriberrr interview.m4a --format txt --timestamps none

# Recursive bulk queue with resumable progress
./target/release/transcriberrr --input-dir recordings --recursive \
  --output-dir transcripts --progress-file transcripts.jsonl --resume

# Latest English Parakeet Unified model or Apple system model
./target/release/transcriberrr interview.wav --engine parakeet-v3
./target/release/transcriberrr interview.wav --engine apple-native
```

Use `--help` for all options. `--model-status` reports the local cache without network access. `--download-model qwen3-asr-1.7b` (repeat the flag for multiple models) prefetches model files. For a complete Rust-engine setup, use `--download-model qwen3-asr-1.7b --download-model qwen3-forced-aligner-0.6b --download-model parakeet-v3 --download-model sortformer-4spk`. Set `--model-cache PATH` or `TRANSCRIBERRR_MODEL_CACHE` to change the cache location. The default is `~/Library/Application Support/Transcriberrr/models/hub`.

## Convert a transcript

```bash
# Standard segment timestamps and speaker labels (the conversion defaults)
./target/release/transcriberrr convert interview.json -o interview.txt

# Sentence/segment subtitle cues
./target/release/transcriberrr convert interview.json --format srt -o interview.srt
./target/release/transcriberrr convert interview.json --format vtt -o interview.vtt

# Plain text or individual word cues
./target/release/transcriberrr convert interview.json --timestamps none -o plain.txt
./target/release/transcriberrr convert interview.json --format srt --timestamps words -o words.srt

# JSON with segment text instead of nested words
./target/release/transcriberrr convert interview.json --format json -o segments.json
```

Conversion defaults to TXT and `--timestamps sentences` (`segments` is an alias). It preserves the existing segment boundaries and speaker IDs. Without `-o/--output`, it writes `INPUT.converted.EXT` beside the input. Existing files require `--overwrite`. Subtitle exports flatten whitespace inside each cue so transcript blank lines cannot split cues. Word export requires word timing data; segment-only JSON cannot recreate word timestamps. No audio, model downloads, GPU initialization, or inference is needed.

Timestamped TXT looks like:

```text
[00:00:00.000 --> 00:00:02.400] [Speaker 1] Hello, world!
[00:00:02.600 --> 00:00:03.200] [Speaker 2] Hi!
```

The compact JSON schema stores each piece of transcript text **once**. There is no top-level `text`. In word mode, `segments[].words[].text` contains exact text fragments, including punctuation and preceding whitespace; segments omit `text`. Concatenate word texts in segment order **without inserting separators or trimming** to reconstruct the transcript. Speaker IDs and timing remain on the words and segments. For sentence/no-word JSON, each segment contains `text` instead of `words`; concatenate those strings without separators. This is a breaking schema change; the converter reads the compact schema, not the old duplicated-text JSON.

For example, the words for `Hello, world!` have texts `"Hello,"` and `" world!"`. Qwen aligner tokens are matched back to recognized text before writing, restoring punctuation and exact spacing. If the words cannot safely account for the recognized text, transcription reports an error instead of writing a lossy transcript.

Qwen word timestamps use `Qwen/Qwen3-ForcedAligner-0.6B`. JSON places words in `segments[].words[]` with `text`, `start`, and `end` seconds. JSON also includes `metadata` with the Transcriberrr version, source creation/modification times, transcription time, model name, and optional diarization settings. For diarized jobs, `metadata.diarization.detected_speakers` reports the distinct speaker IDs returned by Sortformer, while `metadata.diarization.speakers` remains the requested maximum. Word-level SRT/WebVTT emit one cue per word. Apple SpeechTranscriber installs its model through macOS `AssetInventory`; it is unavailable before macOS 26 and does not diarize.

## Diarization

Build with `--features diarization`, then provide a Sortformer ONNX export:

```bash
./target/release/transcriberrr meeting.wav --diarize \
  --diarization-model /models/nemotron3_diar_v3.onnx --speakers 2
```

The ONNX graph must be a Sortformer v3 export with `preds_diar` and `preds_hires` outputs. The original NVIDIA `.nemo` checkpoint is not directly loadable by this binary; use an ONNX export such as [altunenes/parakeet-rs](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-3-diarization), or pass a compatible local file. A manifest enables per-track settings:

```json
{"items":[
  {"input":"meeting.wav","diarize":true,"speakers":2},
  {"input":"solo.wav","diarize":false}
]}
```

## Model and audio notes

Supported inputs are AAC, AIFF/AIF, CAF, FLAC, M4A, MP3, OGG, WAV, and WebM. Qwen3-ASR and the forced aligner are downloaded from their Hugging Face repositories. The default Parakeet engine uses the newer English Unified model, [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b), through the Rust-compatible [eschmidbauer/parakeet-unified-en-0.6b-onnx](https://huggingface.co/eschmidbauer/parakeet-unified-en-0.6b-onnx) export. The previous TDT v3 export remains available for compatibility with `--model istupakov/parakeet-tdt-0.6b-v3-onnx`. Sortformer and Parakeet exports are third-party artifacts; review and accept each repository license before downloading or redistributing them. Transcriberrr does not bundle model weights or grant rights to redistribute them. Source code is MIT licensed; see [LICENSE](LICENSE).

The Rust Qwen and Parakeet backends use Metal; they do not use the Apple Neural Engine. `--device coreml` reports an explicit unsupported-device error for those engines. Apple-native mode uses Apple on-device Speech framework.

## Development and release checks

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
swift build -c release --package-path ui --product transcriberrr-ui
swift build -c release --package-path ui --product transcriberrr-native
./scripts/build-macos-app.sh
./scripts/build-release-dmg.sh
file dist/Transcriberrr.app/Contents/MacOS/Transcriberrr
shasum -a 256 dist/Transcriberrr-*-macos-arm64.dmg
```

Before publishing, test a short English recording through the CLI and the app, verify JSON word timing, and inspect the arm64 output with `file`. Keep credentials, recordings, downloaded weights, and generated release artifacts out of Git.
