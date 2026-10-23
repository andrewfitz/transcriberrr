# Transcriberrr

Offline English transcription for Apple Silicon Macs. Transcriberrr ships a native Rust CLI and a small SwiftUI drag-and-drop app. Audio stays on the Mac; model weights are downloaded to a user-owned cache on first use.

## Features

- Qwen3-ASR 1.7B (default), the latest English Parakeet Unified 0.6B export, or Apple SpeechTranscriber (macOS 26+).
- Metal GPU execution on Apple Silicon, with a CPU fallback for the Rust engines.
- Compact word-level JSON, sentence/segment or word timestamps in TXT, SRT, and WebVTT, and plain TXT.
- Convert saved transcript JSON between formats offline, without loading audio or models.
- Optional Sortformer speaker diarization, including a per-job speaker limit.
- Per-speaker tracks (Zoom participant files and other iso tracks): transcribed together with other speakers' bleed muted, then merged into one transcript with named speakers. Hand over a whole Zoom library folder: each meeting is planned like the wisdom ingest (mixed recording or tracks, restarted recordings joined, videos skipped), and reruns pick up only new meetings.
- Single files, recursive folders, JSON manifests, resumable JSONL progress, and output beside the source or in one destination folder.
- Native SwiftUI queue with folder drop, regex filtering, visible Check all/None, per-track diarization, saved queue results, elapsed time and ETA per track and selected queue, and a visible model setup/download activity window.

## Install a GitHub release

Download the arm64 `.dmg` from the repository Releases page, open it, and drag `Transcriberrr.app` to `/Applications`. The app bundle contains the UI, Rust CLI, and Apple-native helper; model weights are intentionally not bundled. The first run may download several gigabytes from Hugging Face or install Apple system models. A development release is only ad-hoc signed, so macOS may require opening it once from System Settings -> Privacy & Security; notarized releases open normally.

The build scripts ad-hoc sign the bundle by default. For a release that opens without a Gatekeeper warning, sign with a Developer ID and notarize: `CODESIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" NOTARY_PROFILE=PROFILE ./scripts/build-release-dmg.sh`, where `PROFILE` was created with `xcrun notarytool store-credentials`.

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

`--engine apple-native` runs the `transcriberrr-native` helper. The app bundle includes it; for a cargo build, run `swift build -c release --package-path ui` first (the CLI finds the helper under `ui/.build`) or set `TRANSCRIBERRR_NATIVE` to its path.

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
./target/release/transcriberrr convert interview.json --timestamps none --speaker-turns -o turns.txt
./target/release/transcriberrr convert interview.json --format srt --timestamps words -o words.srt

# JSON with segment text instead of nested words
./target/release/transcriberrr convert interview.json --format json -o segments.json
```

Conversion defaults to TXT and `--timestamps sentences` (`segments` is an alias). It preserves the existing segment boundaries and speaker IDs. Without `-o/--output`, it writes `INPUT.converted.EXT` beside the input. Existing files require `--overwrite`. Plain text is the exact transcript; add `--speaker-turns` (also accepted when transcribing, and the app's "Speaker turns" option for Text) to start a `[Speaker N]` paragraph (or `[Name]` for merged tracks) at each speaker change of a diarized transcript. Subtitle exports flatten whitespace inside each cue so transcript blank lines cannot split cues, and always carry cue timing: `--timestamps none` with SRT/VTT gives sentence cues. Word export requires word timing data; segment-only JSON cannot recreate word timestamps. No audio, model downloads, GPU initialization, or inference is needed.

Timestamped TXT looks like:

```text
[00:00:00.000 --> 00:00:02.400] [Speaker 0] Hello, world!
[00:00:02.600 --> 00:00:03.200] [Speaker 1] Hi!
```

The compact JSON schema stores each piece of transcript text **once**. There is no top-level `text`. In word mode, `segments[].words[].text` contains exact text fragments, including punctuation and preceding whitespace; segments omit `text`. Concatenate word texts in segment order **without inserting separators or trimming** to reconstruct the transcript. Speaker IDs and timing remain on the words and segments. For sentence/no-word JSON, each segment contains `text` instead of `words`; concatenate those strings without separators. This is a breaking schema change; the converter reads the compact schema, not the old duplicated-text JSON.

For example, the words for `Hello, world!` have texts `"Hello,"` and `" world!"`. Qwen aligner tokens are matched back to recognized text before writing, restoring punctuation and exact spacing. If the words cannot safely account for the recognized text, transcription reports an error instead of writing a lossy transcript.

Qwen3 skips 30 s chunks in which Silero VAD (compiled in, no download) finds no speech, since the model otherwise invents short phrases for silence, room tone, and music. Qwen word timestamps use `Qwen/Qwen3-ForcedAligner-0.6B`. JSON places words in `segments[].words[]` with `text`, `start`, and `end` seconds. JSON also includes `metadata` with the Transcriberrr version, source creation/modification times, transcription time, model name, and optional diarization settings. For diarized jobs, `metadata.diarization.detected_speakers` reports the distinct speaker IDs returned by Sortformer, while `metadata.diarization.speakers` remains the requested maximum. Word-level SRT/WebVTT emit one cue per word. Apple SpeechTranscriber installs its model through macOS `AssetInventory`; it is unavailable before macOS 26 and does not diarize.

## Per-speaker tracks

When each speaker was recorded on their own track (Zoom's "Record a separate audio file for each participant", a podcast tool, or one lav mic per person), each session's tracks are transcribed together, with the other speakers' bleed muted, and merged into one transcript. Each track is one speaker, so diarization is not needed (and is skipped for these tracks). Zoom's participant files are recognized by name, so a whole Zoom library can be handed over as is and run again as meetings are added:

```bash
./target/release/transcriberrr --input-dir ~/Documents/Zoom --recursive -o transcripts --skip-existing
./target/release/transcriberrr --iso-tracks podcast/host.wav podcast/guest.wav -o transcripts
```

A bare `--iso-tracks` makes every input a speaker track (for tracks from other tools, grouped by folder); `--iso-tracks=off` makes none; the default takes Zoom participant files. A JSON manifest marks tracks with `"iso_track": true` or `false`. A folder is planned meeting by meeting, following the wisdom library's ingest:

- Zoom participant files are grouped by meeting folder (the folder holding `Audio Record`) and the 9-digit meeting ID at the end of their names. Each session writes its tracks' transcripts as usual plus `SESSION.merged.EXT`, named after the meeting folder and placed beside the tracks' outputs, or one level up when those mirror `Audio Record`. Two sessions whose merged transcripts would share a folder and a name get the meeting ID (or a number) added; so does a session whose name is taken by another session's earlier JSON merge.
- A name ending in a digit (`audioGuest211067067067`: Guest2, file 1) is read as such unless that would leave a file numbered 0 or a number another file has, or the same speaker has files numbered below 10.
- With two or more speakers' tracks, the meeting's mixed recording (`audio1067067067.m4a`, or `audio_only.m4a` from 2018–2019 Zoom) is skipped. With only one participant's tracks, the mixed recording is transcribed as well and supplies the words that track does not cover, as "Other speakers" (speech over that participant is lost). Without tracks, it is transcribed on its own, diarized if asked.
- A recording stopped and started again (`audio1656027999.m4a`, `audio2656027999.m4a`) becomes one transcript, its parts joined in order (two files of one segment, such as `audio_only.m4a` and a copy `audio_only01.m4a`, stay separate). Participant files carry the same segment digit before the meeting ID; the segments are joined end to end, each starting where the previous one's longest track ended.
- The digits before the segment digit number the meeting's files in the order they started, across all participants (`audioAndrewDaniel11…`, `audioHana21…`, then `audioAndrewDaniel32…` after a restart). A participant who reconnected has a file for each connection; like every file of a segment, each covers the whole segment, silent while they were not connected, so their words are merged as they fall and no file is shifted. Silent stretches (before joining, after leaving, and while muted in Zoom) are left out when measuring other speakers' bleed.
- A video is skipped when its audio is there too: any video in a Zoom meeting folder (`video….mp4`, `zoom_0.mp4`, `shared_screen…`), as in wisdom's ingest, or a video (including WebM) beside an audio file of the same name, whatever the case. `.zoom` files Zoom has not converted yet are skipped with a warning; hidden files and folders and `#recycle`-style folders are skipped.
- A session's only track is transcribed on its own.

Every skipped file is noted (and recorded in `--progress-file`, which the app shows on its row); if the files it was skipped for cannot run (two of them would write the same transcript), it fails with them. `--skip-existing` skips files and sessions whose transcript is already there; a session counts as there once merged. With `--resume`, likewise, a session counts as finished once merged; otherwise all its tracks run again, since the merge needs each track's words. `--resume` replaces a transcript only for a file the progress file shows was started; another existing transcript stops that file (`--overwrite` replaces it). A file named twice (overlapping globs) is transcribed once.

Each mic also picks up the other speakers, quieter, and a recognizer transcribes that too, so a plain merge shows every sentence twice. Before transcribing, the tracks are compared every 20 ms: for each pair, the bleed is measured from moments when one speaker clearly leads (more than 10 dB louder than every other track), and a frame is muted, to exact silence, when another track is louder by about that much (within 6 dB, and never less than 8 dB). This absorbs mic gains, distances, and loud or soft talkers, and passes overlapping speech and interjections, which are loud on their own mic. Reverberation is handled too: muting continues while a muted stretch keeps fading after the other speaker stops, and short gaps in the bleed between their words are muted unless they get louder than it. Words timed wholly inside muted audio are dropped, and so are runs of three or more words the aligner put at a single instant: on the scraps of audio a muted track leaves, Qwen3 sometimes invents a sentence that matches nothing in the audio. With Qwen3 and Parakeet, muted stretches longer than 2 s (the other speakers' turns) are cut down to a half-second pause before transcription and the words put back in place; this makes Parakeet about 40% faster on a real three-person Zoom meeting, with no measurable change in accuracy, and makes no measurable difference to Qwen3. The Apple engine transcribes the whole muted track. Measured on synthetic two- and three-speaker sessions with all three engines, bleed at -12 to -18 dB, with or without room reverberation, came out once per sentence under the right speaker, and a listener's track carrying only bleed came out empty. On a real 2.6-hour, three-person Zoom meeting (with a reconnect and a restarted recording), 8 of about 19,900 words came out under two speakers; diarizing the meeting's mixed recording instead put 98.7–99.3% of words under the right speaker, while the speaker tracks also keep overlapping speech the mixed recording loses. Bleed within about 8 dB of a speaker's own level (mics that hear everyone almost equally) cannot be separated this way.

The app marks Zoom participant files as speaker tracks when they are added (a dropped folder is added recursively); use the row's "Speaker track" box or the queue's "Speaker tracks" button for other files. The run is planned as above: rows for skipped mixed recordings, videos, and later parts say why, and merged transcripts are listed under the queue. The CLI names them; an existing merged JSON of other recordings in the destination is kept, and the new one gets a number. A track still to do reruns its whole session, finished tracks included, since the merge needs every track's words from the same run.

### Merge saved transcripts

Transcripts made separately can be merged afterwards, without bleed muting:

```bash
./target/release/transcriberrr merge tracks -o talk.json
./target/release/transcriberrr merge tracks/audioHana31067067067.json --format txt --timestamps none --speaker-turns -o talk.txt
./target/release/transcriberrr merge alice.json bob.json --name "Alice Smith" --name Bob --format srt -o talk.srt
```

Tracks must share a timeline (all recorded from the same moment). Words from all tracks are ordered by time and segments are rebuilt exactly as transcription builds them, so the result has the shape of a diarized transcript: every format and `--timestamps` granularity of transcription (word timestamps by default), speaker IDs on every segment and word, and `metadata.diarization.detected_speakers`. As in a diarized transcript, overlapping speech interleaves word by word, so an interjection splits the sentence it interrupts. Inputs need word timings, the transcription default; sentence-only JSON is refused.

Speakers are named from the audio file names recorded in the transcripts. Zoom participant files (`audioAndrewDaniel21067067067.m4a`, `audio_jordan_k.41864297531.m4a`, `audio_only_13572468_Dana_Reyes.m4a`) give the participant's name; other names drop track numbers and words like `iso`, `mic`, and `track` (`1-alice.flac` is "Alice"). `--name` overrides the guesses in the order the files are listed (so it needs listed files, not a folder). A Zoom file name ends with a file number, a segment digit, and a 9-digit meeting ID; each connection starts a new file with the meeting's next number, covering the whole segment, so a speaker's files share the timeline. Zoom's mixed recording (`audio1067067067.m4a`) is not a speaker track and is refused when listed.

Listed files are merged as given, with a warning if they come from different Zoom sessions or differ in length. A folder, or a single Zoom track (or the mixed recording's transcript), merges that folder's transcripts from the same meeting, matched on the meeting ID whatever the file number and segment digit. A folder holding several meetings lists them; pick one with `--session ID` (the 9-digit meeting ID, or the 10 digits after the file number). When only one participant has tracks, the mixed recording's transcript supplies "Other speakers" as above. Earlier merges, other mixed recordings, other meetings, and unreadable JSON in the folder are skipped with a note. Without Zoom tracks, a folder merges all of its transcripts.

The merged JSON lists `speaker_names` by speaker ID and records each track's source, speaker, and offset in `metadata.merged_from`. TXT/SRT/VTT label speakers by name, and `convert` reads merged transcripts like any other. Mic bleed transcribed into these tracks stays; transcribe the audio as speaker tracks (above) to remove it.

## Diarization

Build with `--features diarization`. `--diarize` (or `--speakers N`) downloads the compatible Sortformer ONNX export on first use; `--diarization-model` selects a different model ID or a local file:

```bash
./target/release/transcriberrr meeting.wav --diarize --speakers 2
./target/release/transcriberrr meeting.wav --diarize \
  --diarization-model /models/nemotron3_diar_v3.onnx
```

The ONNX graph must be a Sortformer v3 export with `preds_diar` and `preds_hires` outputs. The original NVIDIA `.nemo` checkpoint is not directly loadable by this binary; use an ONNX export such as [altunenes/parakeet-rs](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-3-diarization), or pass a compatible local file. A manifest enables per-track settings:

```json
{"items":[
  {"input":"meeting.wav","diarize":true,"speakers":2},
  {"input":"solo.wav","diarize":false}
]}
```

## Model and audio notes

Supported inputs are AAC, AIFF/AIF, CAF, FLAC, M4A/M4B, MP3, MP4/M4V/MOV (audio track), OGG, Opus, WAV, and WebM (including video recordings; the audio track is used). Qwen3-ASR and the forced aligner are downloaded from their Hugging Face repositories. The default Parakeet engine uses the newer English Unified model, [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b), through the Rust-compatible [eschmidbauer/parakeet-unified-en-0.6b-onnx](https://huggingface.co/eschmidbauer/parakeet-unified-en-0.6b-onnx) export. The previous TDT v3 export remains available for compatibility with `--engine parakeet-v3 --model istupakov/parakeet-tdt-0.6b-v3-onnx`. Sortformer and Parakeet exports are third-party artifacts; review and accept each repository license before downloading or redistributing them. Transcriberrr does not bundle model weights or grant rights to redistribute them. Source code is MIT licensed; see [LICENSE](LICENSE).

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
