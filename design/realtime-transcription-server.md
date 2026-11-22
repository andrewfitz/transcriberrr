# Realtime transcription server v1

A client sends audio as it is captured and receives captions while people are speaking. The macOS app and the headless command launch the same Rust WebSocket server. Clients own capture and consume results. The deliverable is the listener, streaming recognition, protocol, and server controls.

## Scope

In v1: one OpenAI-compatible single stream per socket, an opt-in multi-track extension on the same endpoint, three candidate streaming engines (one loaded per server process), `transcriberrr serve`, and a Server panel in the app.

Not in v1:

- Server-side turn detection on the OpenAI path (`server_vad`, `semantic_vad`). Clients commit.
- Read-only subscribers. The socket that sends a session's audio is the only one that receives its results.
- Audio capture, WebRTC, and built-in TLS.
- Writing transcript files.
- Sessions that survive a server restart.

## Wire contract

### OpenAI-compatible single stream

Implement the transcription subset at `/v1/realtime?intent=transcription`. Accept a `session.update` with `session.type: "transcription"`, a configured local model or advertised compatibility alias, 24 kHz mono signed little-endian PCM16 (`audio/pcm`, rate 24000), and explicit client commits. Implement session creation and update acknowledgments, audio append/clear/commit and their documented acknowledgments, transcription deltas, completed and failed items, and structured errors. A commit finalizes a turn. The socket stays available for the next turn. Match events by item ID even when completions arrive out of order. Freeze a reference event trace and the client version used for compatibility testing. [OpenAI realtime transcription guide](https://developers.openai.com/api/docs/guides/realtime-transcription)

Deliver stable prefixes during ongoing speech. A delta contains only newly stable text. Completion carries the authoritative transcript for that item. Reject unsupported formats and options explicitly. Publish the supported event and model matrix with the server usage instructions.

Decisions the guide leaves to the implementation:

| Case | Behavior |
|---|---|
| Model alias | Two alias modes, mirroring the two OpenAI models. The streaming alias (`gpt-live-transcribe`) maps to an engine that emits deltas before commit. The committed-turn alias (`gpt-transcribe`) transcribes after commit and works with every engine, including ones with no streaming path. A server process loads one engine, and both aliases use it. Advertise the streaming alias only when that engine emits stable text before commit. `session.updated` echoes the alias the client sent. |
| `turn_detection` | Omitted or `null` is accepted. Any other value is rejected, as `gpt-live-transcribe` does. |
| `languages`, `prompt`, `keywords`, `delay` | Accept `languages` only when the engine supports every listed language (Parakeet Unified and the Apple helper are English-only today). Pass `prompt` to Qwen as stream context and reject it elsewhere. Reject `keywords` and `delay` until an engine implements them. |
| Item ID before commit | Deltas precede `input_audio_buffer.committed`, so allocate the item ID at the first appended audio after the previous commit and reuse it in `committed`. Confirm against the reference trace. |
| Commit with no audio, or `clear` | A commit with no appended audio is an error and creates no item. A commit of audio in which the speech gate found no speech is a normal item that completes with empty text. `clear` discards uncommitted audio, cancels the turn's recognition, and emits no completion for it. |
| Turn exceeds the uncommitted-audio limit | Handle it as the overload hard bound does: emit a terminal failure for the item naming the limit, then drain input through the next commit or clear. Never commit on the client's behalf. |
| Commit of an item that already failed | Acknowledge with `input_audio_buffer.committed` carrying the failed item's ID, and send no completion. The next append starts a new item. |
| Authentication | `Authorization: Bearer <key>`, plus the browser subprotocol form (`realtime`, `openai-insecure-api-key.<key>`) so the OpenAI browser example works once its origin is allowed with `--allow-origin`. Select `realtime` in the handshake response, never the key-bearing subprotocol. |

### Opt-in Transcriberrr streams

Use the same endpoint with an explicit extension version negotiated before audio starts: the query parameter `transcriberrr=1` on the upgrade request. Reject unknown versions with HTTP 400 before the upgrade. Proposed wire names follow. One capture client multiplexes the session's tracks on one socket. Audio is base64 PCM16 in JSON, as in the OpenAI events, so there is one codec. Two 24 kHz tracks cost about 130 kB/s.

| Client action | Server contract |
|---|---|
| Enable the extension | Return the version, session ID, resume credential, actual model and engine, supported timing and diarization features, capacity limits, and resume window. |
| Register a track | Accept a stable track ID, optional speaker label, and input role (isolated mic or mixed audio). A track may join after speech has started. Reject a track beyond the advertised limit without disturbing the others. |
| Append track audio | Accept PCM, track ID, and starting frame on a shared 24 kHz session clock. Frame 0 is the session start, positions are unsigned 64-bit, and the client may attach a wall-clock time for frame 0. The sender maps device clocks to that clock, including drift correction. One frame equals one sample per mono track. |
| Commit or end a track | Flush that track's pending speech independently. Ending a session flushes all tracks, returns outstanding final events, and acknowledges completion. |
| Acknowledge results | Optionally report the highest contiguous event sequence processed, so the server can trim replay history early. |
| Resume | Send the session ID and resume credential in the first message, never in the URL. Validate the client's last processed sequence before transferring ownership. Return the earliest replayable sequence (the next sequence to issue if history is empty). If the client's next required sequence is older, return `replay_unavailable` with that lower bound and leave any active socket in place. The client may then resume again with an explicit accept-loss flag: the server takes over and includes the exact lost range and next expected audio frame per track in an unsequenced resume acknowledgment. The client records acceptance through the last lost sequence, then advances its cursor only as it processes retained events in order, starting at the earliest replayable sequence. For a valid cursor, return the next expected audio frame per track and replay all subsequent events in order before live delivery. Return explicit session expiry when the resume window has elapsed. A successful takeover closes the old socket with a distinct close code. |

Items are server-detected utterances. The server ends an item at a Silero-detected pause or an explicit commit. Advertise a hard maximum native item duration in session frames. During continuous speech, reserve a bounded tail before the limit, finalize through the last stable word boundary within that tail when available, and carry the remaining audio into the next item. At the limit, roll over even if the engine has no stable word timing. How the boundary is made depends on the engine's capability record. For engines with word timing, split at a stable word boundary and retain a bounded PCM tail from that boundary to the latest received frame. Finalize only the prefix. Reset the decoder and replay the tail into the next item, preserving source frame positions and assigning each word to one item. An engine without word timing is cut at the quietest frame in the reserved tail and restarted there, with no overlap, so no audio is decoded twice. Each recognized word belongs to one item, with its original session timing. Measure the maximum item duration and reserved tail length in the engine benchmark.

Use namespaced events such as `transcriberrr.track.append` and `transcriberrr.transcript.draft`. Supply a short reference client and captured wire examples for registration, two interleaved tracks, and reconnect. Each native final carries the item ID, track ID, final text, session-relative start and end, and speaker identity when known. The session announcement identifies the actual model and engine. Include exact timed word fragments when available, preserving punctuation and leading spaces. Each draft carries a monotonically increasing revision. A higher revision replaces the entire draft for that item. An empty draft clears the provisional caption and leaves the item open. Close each native item with exactly one sequenced terminal event, either a final caption or `transcriberrr.transcript.retracted` carrying its item ID, track ID, and reason. Clients remove the draft on retraction. Terminal events are immutable and replayable, so a final waits until bleed for its interval is decided. Ordinary OpenAI clients receive the standard event set.

### Timing and failure behavior

Keep an independent resampler, turn state, and expected frame position per track. Use sample positions to align tracks and a bounded wait for late audio. When the wait expires, treat the late peer as silent for bleed decisions and mark the affected results as decided without that peer. Keep concurrent speech as overlapping intervals. Discard retransmitted frame prefixes already accepted, report gaps with their exact ranges, and advance past missing audio after the wait expires. Distinguish an explicitly signaled gap from a temporary disconnect. During the resume window, retain accepted audio and the current turn. Resume from the server's next expected frame, or report a gap if the sender cannot replay from there. Give every replayable result a stable event sequence so clients can deduplicate delivery. Keep finalized items stable across reconnect. Bound input queues, uncommitted audio, replay history, and outbound data for slow readers. Publish limits and disconnect behavior. A server restart expires in-memory sessions.

Overload policy, in order:

1. Refuse new sessions and tracks at admission once `--max-sessions` or `--max-tracks` is reached. Their defaults come from step 1 and describe one Mac, so the lag bounds below are the real safeguard on other hardware.
2. At the first lag threshold, report backlog in server health and send the affected track ID to extension clients.
3. At the hard bound on the standard socket, emit a terminal transcription failure for the affected item ID, including an item that has already emitted deltas. Reset its decoder and suppress queued results for that item. If it was still receiving audio, drain its remaining input through the next commit or clear before accepting a new turn. On native tracks, retain submitted PCM in capture order and retry when the worker has room. If a track's bounded capture queue fills, pause reads for its multiplexed session. Caption delivery and authenticated takeover continue. Commit, explicit gap and end markers follow the preceding audio. Clients keep capturing independently and can replay from the acknowledged frame if the connection breaks.
4. Close a reader whose outbound queue is full with a distinct close code. Native sessions remain eligible for resume within the advertised replay and time limits.

## Recognition

1. Introduce a long-lived recognition worker with operations for incoming frames, partial results, commit, reset, and cancellation. Keep model weights warm. Isolate per-session inference state and schedule tracks fairly. Keep socket reads independent of model execution. Models and streams live on dedicated OS threads that own them, and the async side talks to them over bounded channels. This is required for Qwen, whose stream borrows the model.
2. Use one streaming Silero detector per track. Accumulate 16 kHz samples across appends, evaluate complete 32 ms windows, and retain the partial window, recurrent state, filter state, and onset counters between packets. Use a causal level estimate and keep pre-roll. Reset at a discontinuity or track end. The file-oriented `has_speech` resets state on each call. Gate each track before recognition, and feed the recognizer only speech plus padding, recording the source frame interval of every retained span. Map both endpoints of every timed word and segment from recognizer samples through these spans onto the 24 kHz session clock, accounting for resampler delay. At a cut, an end maps to the end of the preceding span and a start to the beginning of the following span. In a meeting most tracks are silent most of the time, and a silent track must not take turns on the shared model.
3. Benchmark the existing streaming entry points below with audio paced by a capture clock. Use a separate stream per track. At least one engine must deliver useful deltas before commit with two active tracks.
4. For isolated mics, adapt the existing bleed logic to estimate levels using past frames and bounded lookahead. Test weak bleed, true overlap, and a missing or silent peer. For a mixed input, expose anonymous speaker IDs when the active backend supports live diarization. Feed the diarizer the track's ungated audio on the session clock, since removing silence would shift its timeline and starve its speaker cache. Speaker labels trail the audio by the diarizer's latency: drafts may carry no speaker, and a final waits for the labels covering its interval, up to a bounded wait, after which it is issued without a speaker and says so. Mixed-track finals carry `speaker_spans` with speaker IDs and session-relative intervals. Preserve overlapping spans. Assign timed words only when their speaker is unambiguous. Mark text attribution unknown when words lack reliable alignment, even when diarization intervals are available. Use the item-level speaker field only when it applies to the entire caption. Identify the source of a speaker label and reject unavailable diarization requests. Report timestamp granularity the engine actually provides. [Current bleed implementation](../src/bleed.rs) · [Parakeet engine](../src/parakeet_engine.rs) · [Apple Speech helper](../ui/Sources/TranscriberrrNative/main.swift)

Define one recognizer interface with a capability record: deltas before commit, revisable drafts, word timing and its granularity, languages, and maximum concurrent streams. Session code, both protocols, and the capabilities output read that record. The deterministic fake recognizer used in CI implements the same interface and takes its capability record as configuration (timed or untimed, revisable or append-only), with scripted hypotheses and delays for deterministic protocol and timing tests.

### Bleed in a live stream

The current implementation reads whole recordings. Three parts of it need a streaming form:

- Noise floors and pair thresholds are percentiles over the full track. Replace them with running estimates over a trailing window.
- A pair has no threshold until one speaker has led for a second (`MIN_LEAD_FRAMES`). Until then pass audio through and mark results as uncalibrated.
- Hole filling looks half a second either side of holes up to 400 ms. That lookahead is added to first-text latency for every isolated mic.

Gate causally with a short lookahead for drafts, and apply the full-context decision before the final. How a late bleed decision is applied depends on the engine's capability record:

| Engine capability | Drafts | Final |
|---|---|---|
| Word timing | Decoded from causally gated audio. | Drop the words whose intervals the full decision marks as bleed. If none remain, emit a terminal retraction with reason `bleed`. |
| No word timing | Decoded from causally gated audio. | Re-decode the item from audio muted by the full decision. The cost counts against commit-to-final latency. An all-bleed item gets a terminal retraction with reason `bleed`. |

Step 1 reports how often the causal and full decisions disagree. If drafts flicker, hold audio back by the full lookahead for isolated mics and accept the added first-text latency.

### Latency budget

The 2 s first-text target for two isolated mics has to cover every stage. Fill this in during step 1.

| Stage | Parakeet Unified defaults |
|---|---|
| Capture and send interval | client-defined, assume 100 ms |
| Resampler delay | measured |
| Speech gate onset detection (pre-roll keeps the onset audio, but feeding starts late) | measured, about 100 ms |
| Bleed lookahead | at most what the rows below leave |
| Chunk plus right context | 1.12 s |
| Inference, including waiting for the model mutex behind the other track | measured |
| Event delivery | measured |

About 0.7 s remains for bleed lookahead and inference together. If that is not enough, shorten chunk and right context with `from_shared_with_streaming_config` and report the accuracy cost.

### Packages to reuse

- Start the two-track prototype with the locked Parakeet Unified implementation. Load one `ParakeetUnifiedHandle`, create a decoder per track with `from_shared`, feed 16 kHz frames through `transcribe_chunk`, and use `flush` and `get_timed_transcript` at turn end. The shared model uses a mutex. Measure contention under simultaneous speech. The default chunk and right context are each 560 ms, which together consume 1.12 s of the first-result latency budget before inference. `transcribe_chunk` returns only newly emitted text and never revises it, so its output maps directly to OpenAI deltas and needs no draft revisions. The decoder trims its audio buffer to the 5.6 s left context but accumulates tokens and timings until `reset`, and `reset` also drops that context. Reset at pauses, commits, and forced rollovers. At rollover, extract timed words for the current bounded item and use the reserved-tail procedure above. Release finalized token history and frame mappings after emitting the result. Map returned timings through the retained audio spans. The shared model mutex covers both encoder and decoder work. Measure contention in both stages with simultaneous speech. The model is English-only. [Parakeet Unified 0.4.1 source](https://docs.rs/parakeet-rs/0.4.1/src/parakeet_rs/parakeet_unified.rs.html)
- Use the pinned Qwen library’s `start_stream`, `push_audio_chunk`, and `finish`. Benchmark explicit rolling audio and text windows and map revisable hypotheses to draft events. The defaults cannot meet the target: a 2 s chunk, the first two chunks entirely unfixed, and every step re-feeding all audio so far. Set `chunk_size_sec`, `audio_window_sec`, and `text_window_tokens` explicitly. The pinned public API returns a combined hypothesis and keeps the fixed-prefix boundary internal. Advertise Qwen for native revisable drafts and standard committed-turn output. Enable standard pre-commit deltas only after the adapter can obtain and verify a stable-prefix boundary. Streaming returns no word timings, so timed words need the forced aligner at finalization, which counts against commit-to-final latency. The `AsrStream` borrows the model and `finish` consumes it, so the stream stays on the thread that owns the model. The upstream CLI crate (not the library this project links) includes a WebSocket server at `/v1/ws/stream`; read it as an example for the engine integration. The public wire contract here is OpenAI Realtime. The commit is fixed only by `Cargo.lock`; add `rev` to `Cargo.toml`. [Pinned Qwen implementation and server routes](https://github.com/lumosimmo/qwen3-asr-rs/tree/7984967e8701cc90bb17d06453901036e36ca578)
- Extend the Apple helper with a persistent input stream feeding `AnalyzerInput` buffers to `SpeechAnalyzer`. Request volatile results for drafts and use final results for committed text. Benchmark independent streams on supported macOS versions. The helper today is one process per file, reads an audio file, requests no volatile results, and supports only en-US on macOS 26. Add a `stream` mode: one helper process per server, tracks multiplexed, length-prefixed PCM frames on stdin and JSON lines on stdout, converting to the analyzer's preferred format. Advertise the engine only when the helper is found and reports the OS as supported. [Apple streaming example](https://developer.apple.com/videos/play/wwdc2025/277/)
- Use Tokio and axum’s WebSocket support for the listener. Tokio is already in the lockfile as a transitive dependency; axum is new. Use rubato for stateful sample-rate conversion (0.16.2 is in the lockfile only through `qwen3_asr`, so add it as a direct dependency; the project's own resampler in `main.rs` is batch-only). Use the existing Silero library for speech detection and Sortformer’s `feed` and `flush` for mixed-stream diarization. Sortformer defaults to the offline profile with 30.4 s latency: select `low_latency` (1.04 s) with `set_profile`, confirm in step 1 that the shipped export runs at that window (`set_profile` checks only the numbers, and an export can be pinned to one window), and join segments that `feed` splits at window edges. It needs the `diarization` feature; without it `serve` rejects diarization requests. Account for resampler delay in sample timestamps. Keep the aligned multi-mic bleed logic, resume semantics, and wire event mapping in Transcriberrr. [axum WebSockets](https://docs.rs/axum/latest/axum/extract/ws/index.html) · [rubato](https://docs.rs/rubato/0.16.2/rubato/) · [Sortformer 0.4.1](https://docs.rs/parakeet-rs/0.4.1/parakeet_rs/sortformer/struct.Sortformer.html)

### Code layout

`src/main.rs` is 5,800 lines. Put the server in new modules and move nothing else:

| Module | Contents |
|---|---|
| `src/serve/mod.rs` | `serve` arguments, listener, authentication, health, shutdown |
| `src/serve/openai.rs` | OpenAI event types and mapping |
| `src/serve/tracks.rs` | Extension events, session clock, gaps, resume, replay |
| `src/serve/session.rs` | Session and item state, limits, overload policy |
| `src/serve/recognizer.rs` | Recognizer interface, worker threads, fake recognizer |
| `src/serve/live_bleed.rs` | Streaming bleed gate; shares `frame_levels` and constants with `bleed.rs` |

Build the server unconditionally. Engines stay behind their existing features. Add the streaming engines and `serve` to the `capabilities` output, and update `AGENTS.md` and the README.

## Launch and operation

Add `transcriberrr serve` with bind address, port, engine and model selection, session and track limits, and bearer-key configuration. Bind to loopback by default. Require authentication on non-loopback binds and document WSS deployment through TLS termination. Validate credentials before admitting audio. Keep keys, audio, and transcript contents out of routine logs. Expose model readiness, active session count, and backlog at `GET /healthz`, and list models and aliases at `GET /v1/models`.

- Read the key from `TRANSCRIBERRR_API_KEY` or `--api-key-file`, never from an argument, which `ps` shows. Compare in constant time.
- Reject any upgrade that carries an `Origin` header unless that origin is listed with `--allow-origin`. Browsers do not apply the same-origin policy to WebSockets, so without this any web page the operator visits can open the unauthenticated loopback endpoint and read captions.
- Resolve models through the existing model cache at startup and honor `--offline`. A session request never triggers a download.
- Accept port 0. Print one machine-readable line on stdout when listening, with the actual address and port, and one per readiness change.
- The app passes `--exit-on-stdin-eof` and holds the write end of a dedicated stdin pipe. EOF starts the bounded shutdown drain. By default headless `serve` ignores stdin and keeps serving when launched with stdin closed or redirected from `/dev/null`.
- The existing signal handler exits immediately before jobs start and otherwise sets a flag checked at chunk boundaries. `serve` installs its own handling: the first SIGTERM or SIGINT starts the drain, the second exits.

Add a Server panel to the existing SwiftUI app. Start the bundled CLI using its existing executable resolution, show the endpoint and readiness, and surface model-loading or port-binding errors. Track the child process the app owns. An occupied port is reported to the operator. Stopping the app-managed server and SIGTERM in headless mode use the same bounded drain, final-event delivery, and shutdown path. The drain bound must be shorter than the app's 20 s force-kill. Keep the key in the Keychain and pass it through the environment. The app is not sandboxed, so it needs no network entitlement; a non-loopback bind triggers the macOS firewall prompt for the helper binary.

The server and a batch job are separate processes that each load a model and share the GPU. Measure live latency while a batch job runs in step 1. Unless the result holds the targets, the app pauses the queue while a live session is active and says so in the panel.

## Delivery and release checks

| Step | Evidence required |
|---|---|
| 1. Measure inference and choose live defaults | On a named Apple Silicon Mac, report engine, model, memory, track count, and p50/p95 speech-onset-to-first-text and commit-to-final latency. Measure onset from the wall-clock time the annotated onset sample was sent. Report streaming word error rate beside the existing batch path on the same audio, the completed latency budget, the track count at which lag starts to grow, and latency with a batch job running. Start with proposed p95 targets of 2 s and 2 s respectively for two active isolated mics. Confirm or revise these targets from the prototype before publishing support claims. This step is a gate: if no engine delivers text before commit with two tracks, stop and revise the plan before building the protocol. |
| 2. Implement the standard socket | Run the published OpenAI WebSocket example against the local endpoint with the configured model. Verify setup, captions before commit, two successive turns, clearing buffered audio, and recoverable errors. Verify each row of the implementation-decisions table, a rejected `Origin`, and a rejected key. Run the OpenAI browser example with `--allow-origin` and confirm the handshake selects `realtime`. Use a deterministic fake recognizer for protocol CI and a real recognizer for acceptance. |
| 3. Implement native tracks and recovery | Stream two distinct voices at wall-clock speed for a 30-minute soak with repeated overlap. Include bleed-only intervals, a late join, a stalled track, a mid-turn reconnect, duplicate audio, and a slow reader. Verify both speakers’ reference phrases during overlap, one attributed instance of a bleed-only phrase, first text before either speaker finishes a long turn, accurate word-to-track timing against annotated cues, continued delivery on the healthy track during a peer stall, reported missing ranges, bounded memory and lag, and exactly one terminal outcome per item after replay deduplication. |
| 4. Verify both launch paths | Repeat single-stream and multi-track smoke tests with the packaged app and the headless binary. Verify unavailable models, key rejection, port conflicts, graceful stop during speech, exit when the app is force-quit, and isolation between simultaneous clients. Publish the measured hardware, model, latency, track limits, and supported options. |

### Regression cases

- Feed a turn containing two annotated phrases with suppressed silence between them. Commit explicitly and verify word and segment boundaries for both phrases on the original session clock.
- Force overload after a standard delta. Verify one terminal failure for that item, discard its remaining input through commit, then complete a fresh turn successfully. Repeat on a native track and verify backlog signaling, a continuous item through the final submitted sample, and correct completion of the other track.
- Evict a final from replay history while its session is live. Verify that a cursor one event behind the recoverable range receives `replay_unavailable`. A cursor immediately before the first retained event must replay successfully.
- Feed uninterrupted speech across at least two native item limits. Check bounded item duration, continuous frame coverage, and a reference phrase spanning the cutoff appearing exactly once. Exercise an engine without word timing too.
- Feed a recognizer hypotheses that revise an earlier word. Verify native draft replacement, standard committed-turn final text, and the advertised Qwen alias restrictions.
- Check native caption fields with and without word timing. Verify model and engine identity in the session announcement.
- After `replay_unavailable`, resume with the accept-loss flag. With retained events 40–100 and a cursor of 0, verify that the unsequenced takeover acknowledgment reports loss of 1–39. Disconnect after processing 55, resume from 55, and verify delivery of 56–100 before live events.
- Feed an isolated-mic item containing own speech followed by a peer's bleed. With a timed engine, verify the final keeps the own-speech words and drops the bleed words. With an untimed engine, verify the re-decoded final. Verify that a draft can follow an empty draft, while a terminal bleed or overload retraction closes the item permanently. Replay the terminal event and verify one terminal outcome after deduplication.
- On the standard socket, commit audio with no speech and verify a completed item with empty text. Commit with nothing appended and verify the error. Commit an item that already failed and verify the acknowledgment without a completion.
- On a mixed track, script a speaker handoff without silence and a brief overlap. Verify both speaker intervals, unknown word attribution during ambiguity, intervals with unknown text attribution for untimed engines, and explicitly unlabeled finals after the bounded wait.
- Feed identical PCM to the real Silero adapter in 20 ms, 100 ms, and irregular packets, including splits inside detector windows. Verify identical speech boundaries and retained audio, then use the deterministic recognizer to check captions.
- Run real Parakeet on two uninterrupted tracks across many forced rollovers. Record peak memory and p95 finalization time in successive intervals, plus encoder and decoder lock waits. Verify bounded retained history, no growth in finalization time with session age, and no duplicate or missing reference phrase at rollover.
- Launch headless with `</dev/null`, complete a turn and stop by signal. Launch with the app-owned pipe, close its writer, and verify bounded exit and release of the listening port.

Protocol, clock, gap, resume, and overload tests run in CI against the fake recognizer with synthetic audio and no model weights. The real-engine rollover benchmark, soak, and launch-path checks need recordings and models, so they are release checks run by hand, and their recordings stay out of the repository.

The first prototype proves inference latency. Release requires the standard socket, native multi-track behavior, and both launch paths.

## Open decisions

| Decision | Recommendation |
|---|---|
| Read-only subscribers (a caption overlay separate from the capture app) | Defer. The replay sequence and bounded outbound queue already fit a later subscribe role. |
| Binary audio frames for the extension | Defer until a measured need. Base64 costs a third more bytes, which is negligible on loopback. |
| Idle model unload | Defer. Warm weights hold several gigabytes; report memory in the panel first. |
