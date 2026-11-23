# Transcriberrr

Offline English transcription for Apple Silicon Macs. Transcriberrr ships a native Rust CLI and a small SwiftUI drag-and-drop app. Inference runs on the Mac. Model weights are downloaded to a user-owned cache on first use.

## Features

- Qwen3-ASR 1.7B (default), English Parakeet Unified 0.6B, or Apple SpeechTranscriber (macOS 26+). Run several models over the same inputs and save a separate transcript for each pass.
- Metal GPU execution on Apple Silicon, with a CPU fallback for the Rust engines.
- Compact word-level JSON, sentence/segment or word timestamps in TXT, SRT, and WebVTT, and plain TXT.
- Convert saved transcript JSON between formats offline, without loading audio or models.
- Optional Sortformer speaker diarization, including a per-job speaker limit.
- Per-speaker tracks (Zoom participant files and other iso tracks): transcribed together with other speakers' bleed muted, then merged into one transcript with named speakers. Hand over a whole Zoom library folder to get one transcript per meeting. Reruns pick up only missing transcripts.
- Single files, recursive folders, JSON manifests, resumable JSONL progress, and output beside the source or in one destination folder. The CLI can preview jobs without writes and stream machine-readable results.
- Native SwiftUI queue with folder drop, regex filtering, Include Visible and Exclude Visible actions, per-track diarization, saved queue results, elapsed time and ETA per track and selected queue, and a visible model setup/download activity window.
- Realtime WebSocket transcription for clients that capture their own audio, with headless and macOS app launch controls.
- [Recorded audio sessions](docs/session.md) for multicam editors that extract and place their own PCM tracks. Cache track-local results and reconcile them again when offsets change.

## Install a GitHub release

Download the arm64 `.dmg` from the repository Releases page, open it, and drag `Transcriberrr.app` to `/Applications`. The app bundle contains the UI, Rust CLI, and Apple-native helper; model weights are intentionally not bundled. The first run may download several gigabytes from Hugging Face or install Apple system models. A development release is only ad-hoc signed, so macOS may require opening it once from System Settings -> Privacy & Security; notarized releases open normally.

The build scripts ad-hoc sign the bundle by default. For a release that opens without a Gatekeeper warning, sign with a Developer ID and notarize: `CODESIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" NOTARY_PROFILE=PROFILE ./scripts/build-release-dmg.sh`, where `PROFILE` was created with `xcrun notarytool store-credentials`.

Release archives should be built on an Apple Silicon Mac and distributed with a SHA-256 checksum. Do not commit `target/`, `ui/.build/`, the unpacked `dist/Transcriberrr.app/`, recordings, or model caches; attach only the final `.dmg` and `.dmg.sha256` files to a GitHub Release.

## Build from source

Building requires Apple Silicon, Swift 6, a macOS 26 or newer SDK (for the Apple SpeechTranscriber helper), and Rust 1.99 or newer. The app runs on macOS 14 or newer, and the Apple SpeechTranscriber engine requires macOS 26 or newer. `rust-toolchain.toml` selects the stable Rust toolchain.

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

## CLI automation

Use `./target/release/transcriberrr` from a source build or the CLI embedded in the app bundle at `Contents/Helpers/transcriberrr`. The bundled executable finds the Apple SpeechTranscriber helper beside it. Invoke it with an argument list so paths with spaces and Unicode are passed intact.

```bash
TRANSCRIBERRR="$PWD/dist/Transcriberrr.app/Contents/Helpers/transcriberrr"

"$TRANSCRIBERRR" capabilities
"$TRANSCRIBERRR" --cached-model qwen3-asr-1.7b --cached-model forced-aligner --cached-model parakeet-v3

# Preview the exact inputs, output paths and models for each pass.
"$TRANSCRIBERRR" --input-dir "$HOME/Documents/Zoom" -o transcripts \
  --pass qwen3 --pass parakeet --offline --plan

# Run the same queue and save structured progress and results.
"$TRANSCRIBERRR" --input-dir "$HOME/Documents/Zoom" -o transcripts \
  --pass qwen3 --pass parakeet --offline --progress-file logs/zoom.jsonl --resume \
  --jsonl > events.jsonl

# Apply per-recording options from stdin.
printf '%s\n' '{"items":[{"input":"Voice Memo.qta","output":"notes.json","diarize":false}]}' \
  | "$TRANSCRIBERRR" --manifest - --plan
```

- `capabilities` reports the binary version, engine names and aliases, realtime server support, compiled features, model defaults, supported file extensions, output formats and automation flags. Use `--model-status` for the known model catalog and `--cached-model ID` for explicit Hub IDs, aliases or local paths. `--print-model-tag --engine ENGINE [--model MODEL]` prints the identifier used in filenames.
- `--plan` emits one JSON object on stdout and does not run inference, fetch assets, or write files. With `--pass`, the object contains `passes` in execution order. Check each plan’s `issues`, `summary.rejected_inputs` and `files[].error` before running. `files[].skip` gives the reason an input was left out, `done` marks reused outputs, and `meetings[].output` contains the single merged destination for a Zoom meeting with separate speaker tracks. `required_models` lists the Rust assets needed by pending jobs and whether each resolves locally. Apple checks its speech assets when the helper runs.
- For a Zoom mix, append its path to the arguments advertised in `automation.zoom_iso_inventory`. The mix row lists participant paths for its segment in `covered_by`. A renamed video matched against an audio-only export reports `zoom_audio_copy_clock_verified: true` after the provider verifies both the AAC stream and its presentation edits. Each participant row gives its `meeting`, `speaker`, and `zoom_segment`. When placing words on one recording’s timeline, use participants from its segment. A mix with no identifiable segment has a null `zoom_segment`.
- `--jsonl` streams `run_started`, `pass_started`, `progress`, `pass_finished` and `run_finished` events on stdout. Each line has `schema_version: 1`. A `progress` event has `input`, `status` and a `pass` with its one-based `index`, `engine`, `model` and `model_tag`. A `completed` or `skipped` progress event with an `output` names a saved or reused transcript. `kind: "merge"` identifies a merged result. Check `warning` for details such as no speech or partial participant coverage, and `error` for failures. Terminal events report `completed`, `failed` or `canceled`. Use the process exit status as well. Argument errors and abrupt termination can end a process before a terminal event, and nonzero exit can still leave usable outputs from successful jobs. Diagnostics go to stderr.
- `--offline` prevents automatic model downloads and cache repairs. Prefetch Rust assets with `--download-model` in a separate invocation. Apple SpeechTranscriber requires an installed speech asset when running offline. A pass that can reuse every destination needs no inference assets.
- For recurring scans, `--skip-existing` reuses existing ordinary transcript files. Zoom and `--pass` runs already reuse existing destinations by default. `--progress-file PATH --resume` also reuses completed outputs at the path recorded in the log when the file is still present. With `--pass`, `logs/zoom.jsonl` becomes `logs/zoom.qw.jsonl`, `logs/zoom.pk.jsonl`, and so on. Keep one writer per progress log. Use `--overwrite` to reprocess existing destinations.
- A manifest accepts `items` plus optional `discover_zoom` and `excluded_inputs`. Set `discover_zoom` to `false` to prevent adding unlisted Zoom companions. `excluded_inputs` blocks those paths from automatic companion discovery. An item accepts `input` and optional `diarize`, `speakers`, `iso_track`, `output`, `output_dir`, and `output_location` (`folder` or `next-to-source`). Relative paths in a manifest file resolve beside that file. Paths in a manifest from stdin resolve from the working directory. The same manifest can be sent with several model passes.
- `convert --json` and `merge --json` return a JSON object with the saved absolute `output` path. For a folder of per-track transcripts from multiple passes, add `merge --model-tag qw` to select only the `.qw.json` files.

## Realtime server

In the macOS app, click **Server** to open **Realtime Server**. Select an engine, configure the bind address and port, and click **Start Server**. The panel shows the actual endpoint and model readiness. The server stays running when you close the panel. Reopen it and click **Stop Server** to shut down. The app stores the bearer key in the macOS Keychain. Expand **Advanced** to allow specific browser origins, change admission limits, configure mixed-track speaker detection, or enable Qwen final word alignment.

Finish or stop a running batch before starting the server from the app. The app pauses new batch work and retries while its server is running, through the end of shutdown.

Run the CLI from a source build or use `Contents/Helpers/transcriberrr` inside the app bundle. Add `--offline` to use locally cached models and disable downloads during startup.

```bash
./target/release/transcriberrr serve --bind 127.0.0.1 --port 8080 --offline
curl -fsS http://127.0.0.1:8080/healthz
curl -fsS http://127.0.0.1:8080/v1/models
```

Use `serve --help` for model and device selection, capacity limits, and authentication options. The bundled CLI defaults to Parakeet Unified. For live Qwen, the server defaults to Qwen3-ASR-0.6B. A build without the `parakeet` feature starts with this Qwen model. Use `--model qwen3-asr-1.7b` to select the larger checkpoint. Port `0` selects an available port. Read the listening and readiness JSON events on stdout and connect when the model is ready. `GET /healthz` reports readiness, occupied session slots, and backlog without requiring a key. `GET /v1/models` lists the configured model, available compatibility aliases, and active backend capabilities. It requires the configured key when authentication is enabled. `SIGINT` and `SIGTERM` start graceful shutdown with a 10-second deadline. For a process you supervise with a pipe, pass `--exit-on-stdin-eof` to trigger the same shutdown by closing the write end.

The default admission limits are two sessions and two tracks per native session. A disconnected native session holds its slot while it remains eligible to resume. The listener allows up to twice `--max-sessions` plus two upgraded or pending sockets to leave room for reconnects. It returns HTTP `429` when those socket slots are full. The listener closes a connection that takes more than 5 seconds to send a request, including an idle keep-alive connection, and serves at most 64 connections that have not upgraded to a WebSocket at once.

Use `ws://127.0.0.1:8080/v1/realtime?intent=transcription` for the OpenAI-compatible transcription subset. Send JSON text messages with base64-encoded 24 kHz mono signed little-endian PCM16. Configure the session with `session.update`, append audio with `input_audio_buffer.append`, and send `input_audio_buffer.commit` to finalize each turn. `input_audio_buffer.clear` discards an uncommitted turn. Match transcript events to committed turns using `item_id` and treat the completed transcript as authoritative. A client owns audio capture and consumes captions from the socket.

```json
{"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"turn_detection":null,"transcription":{"model":"gpt-transcribe"}}}}}
```

| Client event | Server events and behavior |
| --- | --- |
| Connect, `session.update` | `session.created`, then `session.updated` after a valid update. Set `turn_detection` to `null` or omit it. `languages` accepts the active engine's supported language codes. Qwen accepts `prompt`. |
| `input_audio_buffer.append` | No acknowledgment. A streaming model can emit `conversation.item.input_audio_transcription.delta` before commit. |
| `input_audio_buffer.commit` | `input_audio_buffer.committed` identifies the item. The server sends `conversation.item.added` for a new committed item, and `conversation.item.input_audio_transcription.completed` or `.failed` carries its result. Committing an empty buffer returns an `error`. |
| `input_audio_buffer.clear` | `input_audio_buffer.cleared`. The discarded turn is not finalized. |

Use `gpt-transcribe` to request recognition after commit. The `gpt-live-transcribe` alias appears in `/v1/models` only when the active engine supports stable pre-commit text. You can also use the advertised local model ID. Each standard connection has a 60-second uncommitted-audio limit. The WebSocket accepts messages up to 256 KiB. One `input_audio_buffer.append` can carry at most 2 seconds of audio. The server pings each standard connection every 20 seconds and closes one that sends nothing, not even a pong, for 60 seconds. Slow readers can be disconnected with close code `4008`. `GET /v1/models` reports these limits under `transcriberrr.limits`, and an oversized or malformed frame ends the connection with an `error` event and close code `1009`, `1007`, or `1002`.

| Engine | OpenAI-compatible aliases | Native word timing | Languages and hints |
| --- | --- | --- | --- |
| `qwen3` | `gpt-transcribe` | Final words with `--forced-aligner` | Qwen languages and optional `prompt` |
| `parakeet-v3` with Parakeet Unified | `gpt-transcribe`, `gpt-live-transcribe` | Yes | English |
| `apple-native` on macOS 26+ | `gpt-transcribe` | Yes, from SpeechTranscriber audio time ranges | US English |

For a native two-microphone client on macOS, Apple Speech delivered pre-commit drafts for all 606 annotated utterances in a 30-minute capture with a 1.23 s first-text p95. For a standard OpenAI client that needs text before commit, use Parakeet with `gpt-live-transcribe`. Parakeet automatically adjusts its decoding window for native multitrack sessions. In a 30m18s two-mic replay it produced a first draft for all 606 utterances, matched all 5,500 scored words, and had zero unexpected gaps. Latency from sending the onset packet to first draft was 1.22 s median and 3.85 s at p95. The slowest draft took 34.08 s under load. [Measurements and test conditions](design/realtime-server-validation.md)

With a build that includes diarization, use `--diarization-model ID_OR_PATH` to load Sortformer for native mixed tracks and request `diarization: true` when registering the track. The optional integer `diarization_wait_ms` sets how long a final caption can wait for speaker labels after recognition finishes. The default is 2,000 ms. Use 0 to publish immediately with the labels already available, or up to 10,000 ms to allow more time for coverage. The server echoes the effective value in `transcriberrr.track.registered`. Qwen and Apple can send revisable native drafts. The native session announcement identifies the capabilities and limits of the running server.

A mixed-track final waits for Sortformer to cover the caption. If the deadline arrives with partial coverage, the final carries `diarization_partial: true` and `diarization_coverage`, a sorted list of half-open `[start_frame, end_frame)` intervals on the session clock. Its `speaker_spans` are clipped to those intervals. A timed word gets a speaker only when one speaker is in reach of it: every span the word overlaps, or in a pause every span within 240 ms, names the same speaker. A word that two speakers touch stays unlabeled. Words outside coverage and the caption-level `speaker` stay null. A final with no coverage reports `diarization_unavailable: true`. During server shutdown, a waiting final is published at once with the labels already decided. A final also stops waiting once Sortformer has moved past it. If Sortformer falls behind, captions wait for it for at most one second. After that the server stops sending it audio until it has caught up, and the captions it missed report partial or unavailable coverage. Speaker IDs start a new numbering after such a break, as they do after a capture gap.

Add `&transcriberrr=1` to opt into the native multi-track protocol. It uses one shared 24 kHz frame clock across tracks. Register each track with `transcriberrr.track.register` and a role of `isolated_mic` or `mixed_audio`, then send `transcriberrr.track.append` with `track_id`, `start_frame`, and base64 `audio`. Set `speaker` at registration to attach a name. Each append can carry up to one second of audio. The server can wait up to 250 ms for missing frames, with a pending limit of 48,000 frames or 128 packets per track. `transcriberrr.track.commit` flushes one track and `transcriberrr.session.end` flushes them all. Replace a draft with any higher revision for the same item. A final or retraction closes the item. A draft for the next item can arrive before the previous item's final, so key captions by `item_id`. An item is capped at 30 seconds. For continuous speech, the server looks for a confirmed word boundary in the final 2.4 seconds when the backend's drafts carry word timing (`draft_word_timing`, Parakeet). Otherwise, including with Apple Speech, it cuts at a quiet point in the final second. Parakeet's streaming decoder produces a word some time after it is spoken. The server moves each word's `start_frame` back by that measured delay. A word ends where the next one starts, or before a pause after a length estimated from its spelling. Punctuation sits at the end of the word before it. A word spoken just before a forced cut, whose text the decoder produced after the cut, starts at its item's first frame.

After a timed Parakeet cut, up to 1.44 seconds of preceding audio can seed the next decoder while the preceding caption retains its words.

When native recognition falls behind, the server sends `transcriberrr.track.backlog` with the affected track and queued frame count. It retains audio in order and pauses socket reads if capture buffers fill. A paused connection can delay every track multiplexed on that socket. Live clients should keep microphone capture independent of WebSocket writes and retain unsent audio for the duration of a stall. A client that loses captured frames can report the exact range with `transcriberrr.track.gap`.

For a reconnect, add `&resume=1` to the native WebSocket URL. This lets the server wait for the resume message when a client is slow to send it, including when all session slots are occupied. Send `transcriberrr.session.resume` as the first message with `session_id`, `resume_credential`, and `last_sequence`. Use the returned `next_expected_frames` to continue sending audio and process replayed events in sequence order. A disconnected session can resume for up to 30 seconds. The replay buffer holds up to 256 events or 4 MiB. If the server reports `replay_unavailable`, retry with `accept_loss: true` to take over and receive the exact `lost_range` followed by the retained events. A successful takeover closes the previous connection with code `4009`. The server pings each native connection every 20 seconds and closes one that sends nothing, not even a pong, for 60 seconds with code `1001`. Time the server spends not reading the socket does not count, and the session stays resumable.

The reference client streams 24 kHz mono PCM16 WAV files at capture speed. In native mode, stdout carries the wire events and stderr reports capture time, drain time, the longest send, backlog, replay loss and unexpected frame gaps. A reported unexpected audio gap makes the run exit unsuccessfully after draining. Once capture ends, the client waits up to 30 seconds between recognition progress events. It requires Python 3.11+ and `websockets==15.0.1`. [Reference client](scripts/realtime-client.py) · [Captured wire examples](design/realtime-traces/README.md) · [Measured performance and validation](design/realtime-server-validation.md)

```bash
python scripts/realtime-client.py --url ws://127.0.0.1:8080 --model gpt-transcribe alice.wav
python scripts/realtime-client.py --url ws://127.0.0.1:8080 --native --reconnect-at 2 alice.wav bob.wav
```

The default listener is on loopback. Set `TRANSCRIBERRR_API_KEY` or pass `--api-key-file PATH` to require a bearer key. A non-loopback bind requires a key. Send it with `Authorization: Bearer <key>` or the browser WebSocket subprotocols `realtime` and `openai-insecure-api-key.<key>`. To permit a browser origin, pass `--allow-origin` with the exact origin, including its scheme and port if applicable. Repeat the flag for multiple origins. Deploy a TLS-terminating reverse proxy for remote clients and connect over `wss://`.

## Model passes

```bash
# Write interview.qw.json and interview.pk.json, reusing any existing result
./target/release/transcriberrr interview.wav --pass qwen3 --pass parakeet

# Compare two Qwen models on a Zoom library
./target/release/transcriberrr ~/Documents/Zoom -o transcripts \
  --pass qwen3 --pass qwen3=qwen3-asr-0.6b

# Add the Parakeet identifier to a single-model run
./target/release/transcriberrr interview.wav --engine parakeet --model-suffix
```

Passes run in the order given. Each processes the whole selection and writes its own transcript, reusing existing results unless you request `--overwrite`. Zoom meetings with participant audio get one merged transcript per pass, such as `MEETING.merged.qw.json`. The default model identifiers are `qw` for Qwen3, `pk` for Parakeet, and `ap` for Apple SpeechTranscriber. Qwen 0.6B uses `qw06`, Parakeet TDT uses `pkt`, and custom models get an identifier derived from their model ID or local path.

Use `--pass ENGINE=MODEL` to override a model. With multiple passes, `--plan` returns a plan for each model and `--progress-file run.jsonl` writes a log such as `run.qw.jsonl` for each model.

On macOS, open **Run Settings** in the inspector and choose **Edit Passes…** to set the engine and model for each pass, add passes, or change their order. Select a recording to see its saved transcripts in the inspector and reveal them in Finder. Multi-model runs append the model identifiers automatically and reuse finished outputs. For a single pass, enable **Append model ID to filenames** in the pass editor.

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

Qwen3 skips 30 s chunks in which Silero VAD (compiled in, no download) finds no speech, since the model otherwise invents short phrases for silence, room tone, and music. Qwen word timestamps use `Qwen/Qwen3-ForcedAligner-0.6B`. JSON places words in `segments[].words[]` with `text`, `start`, and `end` seconds. JSON also includes `metadata` with the Transcriberrr version, source creation/modification times, transcription time, model name, and optional diarization settings. For diarized jobs, `metadata.diarization.detected_speakers` reports the distinct speaker IDs returned by Sortformer, while `metadata.diarization.speakers` remains the requested maximum. Word-level SRT/WebVTT emit one cue per word. Apple SpeechTranscriber supplies timed words on macOS 26 or later and installs its speech model through macOS `AssetInventory`. With a diarization-enabled Transcriberrr build, `--diarize` runs Sortformer on the same audio to label those words.

## Per-speaker tracks

Add a Zoom library, a meeting folder, or an individual Zoom export in the app or CLI. Transcriberrr matches the files to their meeting using the recording folder, Zoom's file names, and the meeting ID when present. It checks the meeting folder and its `Audio Record` subfolder for audio Zoom saved with “Record a separate audio file for each participant.”

```bash
./target/release/transcriberrr ~/Documents/Zoom -o transcripts
./target/release/transcriberrr --input-dir ~/Documents/Zoom -o transcripts --plan
./target/release/transcriberrr --iso-tracks podcast/host.wav podcast/guest.wav -o transcripts
```

- With participant audio, the CLI transcribes each track with bleed reduction and merges them by timestamp into one speaker-labeled transcript. It skips matching videos and mixed audio for each segment covered by participant tracks, including renamed videos whose AAC audio matches the mixed export. A restarted segment with no participant tracks gets a separate mixed-audio transcript on its own clock. An undated export stays separate when it could belong to several segments. If Zoom saved just one participant's audio, the transcript covers that participant and the app and CLI flag the limited coverage. Reconnected participants keep the same speaker name. When Zoom restarted a recording, its participant parts are joined by segment.
- For meetings with no participant tracks, the CLI picks one export per recording segment, using mixed audio when available and video to fill the other segments. The joined transcript can use diarization. If the folder contains multiple meeting IDs, an export without an ID is reported as ambiguous. A Zoom recording still being converted is deferred while its folder holds raw `.zoom` files.
- Zoom destinations are stable across bulk and individual runs. A shared output folder gets a subfolder for each meeting, named with the meeting name and a stable identifier, containing `MEETING.merged.EXT` for participant tracks. With `--output-location next-to-source`, the merged file is `MEETING [ID].merged.EXT`. The CLI skips Zoom work automatically when the destination file for the requested format exists. To redo a meeting, including one with newly added tracks, delete that output or use `--overwrite`.
- `--plan` lists what will run, which recordings will be skipped, and where transcripts will be written, then exits before loading models. `--input-dir` discovers nested Zoom meetings by default. Add `--recursive` to also include other audio in subfolders.
- For tracks from other recorders, use `--iso-tracks` to mark the inputs as speaker tracks, grouped by folder. Use `--iso-tracks=off` to turn off automatic Zoom discovery. A manifest can set `"iso_track": true` or `false` on individual files. Manually marked non-Zoom tracks keep their individual transcripts as well as the merged file. Use `--skip-existing` to also skip existing outputs for ordinary jobs.

Each mic also picks up the other speakers, quieter, and a recognizer transcribes that too, so a plain merge shows every sentence twice. Before transcribing, the tracks are compared every 20 ms: for each pair, the bleed is measured from moments when one speaker clearly leads (more than 10 dB louder than every other track), and a frame is muted, to exact silence, when another track is louder by about that much (within 6 dB, and never less than 8 dB). This absorbs mic gains, distances, and loud or soft talkers, and passes overlapping speech and interjections, which are loud on their own mic. Reverberation is handled too: muting continues while a muted stretch keeps fading after the other speaker stops, and short gaps in the bleed between their words are muted unless they get louder than it. Words timed wholly inside muted audio are dropped, and so are runs of three or more words the aligner put at a single instant: on the scraps of audio a muted track leaves, Qwen3 sometimes invents a sentence that matches nothing in the audio. With Qwen3 and Parakeet, muted stretches longer than 2 s (the other speakers' turns) are cut down to a half-second pause before transcription and the words put back in place; this makes Parakeet about 40% faster on a real three-person Zoom meeting, with no measurable change in accuracy, and makes no measurable difference to Qwen3. The Apple engine transcribes the whole muted track. Measured on synthetic two- and three-speaker sessions with all three engines, bleed at -12 to -18 dB, with or without room reverberation, came out once per sentence under the right speaker, and a listener's track carrying only bleed came out empty. On a real 2.6-hour, three-person Zoom meeting (with a reconnect and a restarted recording), 8 of about 19,900 words came out under two speakers; diarizing the meeting's mixed recording instead put 98.7–99.3% of words under the right speaker, while the speaker tracks also keep overlapping speech the mixed recording loses. Bleed within about 8 dB of a speaker's own level (mics that hear everyone almost equally) cannot be separated this way.

The macOS queue shows an expandable row for each Zoom meeting with participant tracks. Use the meeting’s checkbox to include it in a run. Expand the row to see the named tracks and why each mixed export was skipped. Select the meeting to see merge progress and reveal the saved transcripts in Finder. For recordings from other tools, select an individual file and enable **Speaker track** in the **Recording** inspector.

### Merge saved transcripts

Transcripts made separately can be merged afterwards, without bleed muting:

```bash
./target/release/transcriberrr merge tracks -o talk.json
./target/release/transcriberrr merge tracks/audioHana31067067067.json --format txt --timestamps none --speaker-turns -o talk.txt
./target/release/transcriberrr merge alice.json bob.json --name "Alice Smith" --name Bob --format srt -o talk.srt
```

Tracks must share a timeline (all recorded from the same moment). Words from all tracks are ordered by time and segments are rebuilt exactly as transcription builds them, so the result has the shape of a diarized transcript: every format and `--timestamps` granularity of transcription (word timestamps by default), speaker IDs on every segment and word, and `metadata.diarization.detected_speakers`. As in a diarized transcript, overlapping speech interleaves word by word, so an interjection splits the sentence it interrupts. Inputs need word timings, the transcription default; sentence-only JSON is refused.

Speakers are named from the audio file names recorded in the transcripts. Zoom participant files (`audioAndrewDaniel21067067067.m4a`, `audio_jordan_k.41864297531.m4a`, `audio_only_13572468_Dana_Reyes.m4a`) give the participant's name; other names drop track numbers and words like `iso`, `mic`, and `track` (`1-alice.flac` is "Alice"). `--name` overrides the guesses in the order the files are listed (so it needs listed files, not a folder). A Zoom file name ends with a file number, a segment digit, and a 9-digit meeting ID; each connection starts a new file with the meeting's next number, covering the whole segment, so a speaker's files share the timeline. Zoom's mixed recording (`audio1067067067.m4a`) is not a speaker track and is refused when listed.

Listed files are merged as given, with a warning if they come from different Zoom sessions or differ in length. A folder, or a single Zoom track (or the mixed recording's transcript), merges that folder's transcripts from the same meeting, matched on the meeting ID whatever the file number and segment digit. A folder holding several meetings lists them; pick one with `--session ID` (the 9-digit meeting ID, or the 10 digits after the file number). When only one participant has tracks, the merge can use a matching mixed transcript to label speech outside that participant’s turns as "Other speakers". Earlier merges, other mixed recordings, other meetings, and unreadable JSON in the folder are skipped with a note. Without Zoom tracks, a folder merges all of its transcripts.

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

Supported inputs are AAC, AIFF/AIF, CAF, FLAC, M4A/M4B, MP3, MP4/M4V/MOV (audio track), OGG, Opus, QTA (Apple Voice Memos), WAV, and WebM (audio track). QTA uses one playback-compatible track. To transcribe a mix of all layers and effects in an editable Voice Memo, use Voice Memos’ Rendered export. [Apple’s Voice Memos export guide](https://support.apple.com/en-gw/guide/iphone/iph3d6dc359/ios)

Qwen3-ASR and the forced aligner are downloaded from their Hugging Face repositories. The default Parakeet engine uses the newer English Unified model, [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b), through the [parakeet-unified-en-0.6b](https://huggingface.co/altunenes/parakeet-rs/tree/main/parakeet-unified-en-0.6b) export that parakeet-rs publishes. The multilingual TDT v3 model is available with `--engine parakeet-v3 --model parakeet-tdt-v3`. Unified, TDT v3 and Sortformer are downloaded from one pinned revision of [altunenes/parakeet-rs](https://huggingface.co/altunenes/parakeet-rs), the revision that matches the parakeet-rs release this build uses. Sortformer and Parakeet exports are third-party artifacts; review and accept each repository license before downloading or redistributing them. Transcriberrr does not bundle model weights or grant rights to redistribute them. Source code is MIT licensed; see [LICENSE](LICENSE).

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
