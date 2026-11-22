# Realtime server validation

The acceptance runs streamed synthetic voices over loopback at capture speed on an Apple M4 Max with 48 GiB of unified memory, running macOS 27.0.1 (26A434). The fixtures and their phrase cues can be regenerated locally. The packaged app launches the same Rust server as the headless CLI. Clients own audio capture.

## Astra audit and performance verification

The previous build reproduced overload on a 90-second, two-track replay at capture speed. It dropped 19.7 s from the first track and 19.1 s from the second, in addition to a deliberately omitted second of input. When the two-second worker queue filled, the session had already advanced its input clock. Subsequent packets were discarded until the queue fell below the lower backlog threshold.

Native sessions now keep the original PCM, commits, gaps and end markers in order until the recognition worker can accept them. When staging fills, the session pauses its WebSocket input. Replay, takeover, and outgoing captions remain available. The separate mixed-track diarization queue also retries complete packets. Deterministic blocked-worker tests exercise both tracks through session end and a takeover that resumes from the server's accepted frame.

| Measured operation | Before | After |
| --- | --- | --- |
| Apple helper memory growth over 100,000 synthetic IPC packets | 1,614,288 KiB | 144 KiB |
| 30 seconds of per-track resampling in 20 ms packets | 4,503 allocations, 6.13 ms median | Zero allocations, 5.62 ms median, bit-identical PCM |
| Bleed-level ingestion for 60 seconds and two microphones | 14.16 ms median | 5.35 ms median |
| Bleed-level ingestion for 60 seconds and sixteen microphones | 114.30 ms median | 48.16 ms median |
| CPU Sortformer inference-window p95 on the 19.158-second A/B/A fixture | 1,162 ms | 210 ms |

The IPC check uses a synthetic packet stream without model inference. The resampler and bleed measurements isolate their respective operations. The Sortformer test uses the real CPU model and preserved the speaker sequence. The revised CPU profile uses larger inference windows and a shorter speaker cache. The Metal profile retains its previous settings.

Other regressions cover partial and overlapping packets, multiple reorder deadlines, buffered audio inside a client-declared gap, clean terminal delivery to a temporarily full standard socket queue, and partial final PCM packets in the Apple helper. The Parakeet benchmark uses the production decoder profile and includes first text arriving during finalization.

The signed candidate completed the same 90-second replay with zero overload gaps and zero word errors across 220 scored words. Scoring excludes the first cycle, which contains the deliberate one-second input gap. First-text p95 rose from 5.07 s across 20 observed drafts to 12.54 s across 29 of 30 utterances. Six candidate drafts arrived before the end of speech. The 45.1-second continuous replay preserved all 155 words, including all ten at the forced boundary, with zero audio gaps. Both standard model aliases returned the expected text, and the live alias emitted three deltas before commit.

On the packaged Apple Speech and CPU Sortformer path, all three ASR finals arrived. Both audio consumers reported zero dropped frames. The first two speaker assignments were correct. The third was unavailable at the publication deadline, so integrated A/B/A speaker acceptance remains open. One inference window took 4.605 s in that run.

The first long replay exposed a client heartbeat failure after 390 s of capture. While the native server paused reads under pressure, its writer continued delivering new finals, but the reference client disconnected when a pong timed out behind the queued audio. The native reference client now disables automatic pings. A paused-transport regression reproduced the previous WebSocket 1011 and passed with all 6,000 frames after the fix.

The release passed 219 all-feature Rust tests, 205 minimal-build Rust tests, eight Python socket tests, strict Clippy, formatting and the Swift release build. The ad-hoc signed arm64 DMG passed image and signature verification. Its SHA-256 is `41961f46ad32653557bda62599e64748b8bcccf129aed74273724b82789ae4de`.

## Code audit and follow-up verification

The audit covered the macOS controller and helper, WebSocket listener, both protocols, session replay, audio queues, recognition and speaker attribution, model loading, reference client, benchmark and release scripts. The subsequent sections retain measurements from the earlier release. The 30-minute soak figures identify the builds used for those runs.

| Area | Confirmed defect and correction |
| --- | --- |
| WebSocket lifecycle | Both protocols could lose the close reply when a client disconnected. The handlers flush the client's close code and reason. Stalled native writers have a deadline. |
| Session ownership and replay | New-session negotiation could win the race against a delayed resume. The optional `resume=1` URL waits for credentials even at capacity. Completed-session replay now respects takeover ownership. The reference client restores missing registration and regenerates recoverable audio from its fixture after reconnect. |
| Audio and recognition | Repeated clears could consume admission slots, and a results reader could sleep through an overflow. Cleanup wakeups coalesce and a terminal overflow wakes the reader. Parakeet now pads a partial final encoder frame so the decoder processes the last real samples. |
| Multitrack attribution | Bleed analysis advanced one shared cursor, skipping slower microphones' audio. Progress is now per microphone, and final corrections use the audio actually sent for the draft. Speaker detection bounds queued PCM by duration, coalesces small packets, and discards abandoned work. |
| Apple and the GUI | The helper observes parent exit during warmup. A blocked helper write has a deadline, and each item keeps its first terminal result. Permanent helper failure now drains the server and reports the cause. A descendant holding stderr open can no longer block the GUI from reaping the CLI and releasing the batch queue. |
| Models and packaging | An interrupted model refresh could hide the prior usable revision from offline clients. Incomplete markers are now per snapshot. Release packaging stages the app, image, and checksum together and restores the previous release on a publication error. |

The audited source passed **206 all-feature Rust tests** and **192 minimal-build Rust tests**, including a real CLI subprocess that loses its helper after advertising readiness. Both scripted Python WebSocket tests, strict Clippy, Rust formatting, Ruff, and the Swift release build passed. In the GUI lifecycle fixture the queue interlock released in 213 ms while a descendant still held stderr.

The audited DMG SHA-256 is `e39ab97cfac8deaa66662bb39169378fe34e3a3f3277a1d8994a007a3840b32f`. The mounted app and both helpers matched the signed build byte for byte. From the app’s Server panel, Apple Speech delivered an exact standard turn and resumed two isolated tracks at frame 72,000. Both track finals had monotonic timestamps and the expected track-specific cue. The native run reported zero gaps and two lexical edits across 65 reference words. Stopping the server released the port and child.

On the audited Parakeet release, the standard socket delivered three deltas before commit and an exact normalized final. The 45.1 s continuous single-track replay retained all ten words around the forced boundary at 28.4 s. It also reported a separate 4,800-frame overload gap (200 ms) and five edits across 155 reference words. The zero-loss acceptance assertion failed for this run.

With Qwen 0.6B and forced alignment, the audited release delivered five drafts and an exact final with 16 timed fragments. First text arrived at 1.938 s and the aligned final at 1.320 s after the fixture ended. There was one backlog warning and zero audio gaps. Apple with CPU Sortformer delivered three timed ASR finals and no ASR gaps. The first final labeled 15 words; two finals reported unavailable labels. The separate diarizer queue dropped 45,600 frames, and the A/B/A speaker acceptance check failed. The audit did not repeat the 30-minute soaks.

## Original release validation

| Check | Local result |
| --- | --- |
| `cargo test --locked --all-features` | 184 passed |
| `cargo test --locked --no-default-features` | 174 passed |
| `cargo test --locked --all-features --bin transcriberrr serve::` | 50 server tests passed in the original focused suite |
| `cargo clippy --locked --all-targets --all-features -- -D warnings` | Passed |
| `cargo fmt --all -- --check` and `git diff --check` | Passed |
| `swift build -c release --package-path ui` | Passed |

The arm64 app and disk image were built with ad-hoc signing. The disk image verified and mounted read-only. The mounted app and both bundled helpers passed strict signature verification, and the mounted CLI displayed `serve --help`. The actual Server panel reported an occupied port and an unavailable offline model, and allowed another start attempt after each error.

The pre-audit smoke test launched the signed app directly from the read-only disk image. Its managed server completed an Apple standard turn and two native tracks after reconnect, with 18 and 13 timed fragments. Pressing Stop closed the child and released its port. The image SHA-256 is `b36888c9f7010b9e8d1a95a24de04f2a160f0adedb0843fcc595bde0053dab0a`.

In the lifecycle tests, the app's own child delivered stable single-stream captions, two native tracks, and a successful reconnect. Closing the panel left the child running. Stopping it during speech delivered the outstanding timed final and end events. Force-quitting the app released the child and port in 30 ms. The queue paused and resumed with the managed server.

## Protocol and audio handling

- The official OpenAI JavaScript SDK completed two successive manual turns. A paced 2.4-second Parakeet turn delivered text before commit and finalized using the same item ID. A longer turn reached the recognition queue limit after two deltas. The server emitted a terminal failure and acknowledged the subsequent commit. [Wire traces](realtime-traces/README.md)
- OpenAI Python SDK 3.57.0 streamed two 5.2-second recordings to the Apple-backed `gpt-transcribe` endpoint. Both finals read “The meeting starts on Monday morning. Please bring the red folder to the conference room.” They arrived 450 ms and 111 ms after their respective commits with distinct, matching item IDs. The tuned Parakeet server completed the same two turns with all words correct using `gpt-live-transcribe`. Its first deltas arrived 2.278 s and 1.912 s from capture start, and its finals arrived 150 ms and 117 ms after commit. [Reproduce the Python SDK check](../tests/realtime/README.md)
- Chrome 157 negotiated the `realtime` subprotocol using the configured origin and key. The listener rejected a wrong key and a disallowed origin. Concurrent clients received independent items and distinct transcripts. Closing app-owned stdin during speech drained the native session and emitted its final and session-ended events. Headless operation with stdin closed continued until SIGTERM.
- The native tests exercise per-track sample clocks, late joins, gaps, bleed corrections, forced rollover, resume and replay, ambiguous speaker attribution, and slow-reader recovery. The real slow-reader check received close code 4008 and resumed from its saved cursor. [Test and fixture commands](../tests/realtime/README.md)

## Two-microphone timing

The optimized Parakeet Unified server was tested with two isolated English voices, four turns per track, and 100 ms packets. First-text latency was measured from sending the packet containing the annotated onset. Finalization latency was measured from sending the commit. CPU contention was high and the machine was using approximately 48 GB of swap. The audio sender and inference server ran on the same machine.

**The tuned release server met both proposed two-second p95 targets in the repeated two-mic WebSocket test.** After a warm-up pair it delivered pre-commit drafts and post-commit finals on all eight turns. First-text p50/p95 were 1.777/1.839 s. Commit-to-final p50/p95 were 1.325/1.413 s. There were zero gaps and four word edits in 260 reference words (1.54% WER).

The server now uses **1.44 s chunks, 0.64 s of left context, and 0.16 s of right context**. In the optimized inference benchmark, two tracks completed two turns each with zero edits over 130 reference words. Across those four turns first-text p50/p95 were 2.04/2.28 s and commit-to-final p50/p95 were 0.28/1.11 s. This measurement includes the recognizer and its input queues. The original-profile table below includes the full WebSocket, gating, and bleed path.

| Measurement | Result with the original Parakeet profile |
| --- | --- |
| Turns with pre-commit text | 8 of 8 |
| Onset to first text, p50 / p95 | 5.54 s / 7.15 s |
| Turns with a post-commit final | 2 of 8 |
| Commit to final among those two, p50 / p95 | 5.08 s / 5.08 s |
| Overload gaps | 18 |
| Peak sampled server footprint | 2,889 MiB |
| Word edits across four captures of the reference pair | 215 of 260 words, including dropped audio |
| Batch reference for the same pair | 0 edits in 65 words |

The original profile missed both proposed two-second targets. The configured session and track limits control admission. Sustained capacity depends on the recognizer, model, and available compute.

With a real batch transcription overlapping the tuned two-mic test, first-text p95 rose to 2.792 s. The six finals delivered during batch overlap had a 2.614 s p95. The app therefore defers new batch work while its managed server is running. Separate one-, three-, and four-track capacity attempts showed variable queue loss as host load rose; those short runs do not establish a general maximum number of simultaneously speaking tracks.

In the single-track continuous test, Parakeet carried audio over a forced boundary at frame 677,760 (28.24 s). The adjacent items covered the full 36.9 s with exact continuity, including six reference words spanning the cut. The boundary final arrived 107 ms after the packet reaching the hard limit. An earlier 184 s two-track continuous test reported 50 overload gaps, accounting for roughly 15% of each track. No item in that run reached a clean forced rollover.

A supplemental 129.1 s paired run prohibited catch-up bursts. Each track crossed four forced boundaries. All seven cuts with a subsequent nonempty final had exact frame continuity; four of those seven also matched the six-word reference passage at the boundary. The remaining cut was at the speech tail with no later nonempty final to compare. Each track had two overload gaps, losing about 1.3% of audio. These tests establish bounded item duration and distinguish timestamp continuity from lexical accuracy across resets.

Paired replays of the original PCM reproduced three omitted words or word starts after decoder resets and recovered them when the decoder retained its context. A duplicated “store” was already present before the cut. Moving the cut to a quieter point failed to restore the missing words and made a previously correct crossing worse. Giving the fresh decoder 1.44 s of pre-roll recovered all three omitted words in the same local replays and preserved both control crossings. The implementation keeps the caption boundary fixed, excludes context from published words, and retries with owned audio alone if text and timing disagree.

On the final pre-roll build, the 129 s paired replay produced a clean forced boundary at 89.70 s on each track. Both ten-word passages spanning the cuts matched their references, and the next drafts started within their own audio. The same run reported 30 overload gaps, losing roughly 9–10% of the input under load. A separate single-track playback crossed the original 28.40 s seam and recovered “the river just after sunrise” exactly once. Its second cut at 56.24 s also preserved the reference passage. Across 217 words it had zero edits and zero gaps.

The 30-minute Parakeet soak used the tuned build immediately preceding the forced-rollover pre-roll change. The client completed 1,818 s of source audio over 1,957 s of wall time. The server returned 543 distinct finals, 46 retractions, and correctly deduplicated a replayed final. It also reported 143 overload gaps, losing 206.4 s and 204.2 s of the two tracks. Of 606 annotated utterances, 499 had a matching first draft and 290 received one before the voice ended. First-text p50/p95 on the 499 observed drafts were 2.177/4.157 s. Word error rates excluding the first deliberately interrupted cycle were 15.0% and 18.48%. Host one-minute load had a median of 56.8, with roughly 49 GiB of swap in use. This run demonstrates recovery and falls short of sustained two-track quality.

## Apple Speech

**The fast Apple native server passed the 30-minute two-microphone soak.** Two synthetic voices sent 1,818 s of audio in 1,846 s of wall time. Capture included overlap, −32 dB crosstalk, a late track, an intentional one-second gap, and a mid-turn reconnect. The optimized release CLI and final fast helper were pinned by digest for this run.

| Fast Apple measurement | Result |
| --- | --- |
| Utterances with a draft before the voice ended | 606 of 606 |
| Onset to first text across both tracks, p50 / p95 / max | 1.128 / 1.228 / 1.506 s |
| Annotated cue words on the correct track and within the expected time window | 606 of 606 |
| Overlapping caption pairs lasting more than 500 ms | 201 |
| Distinct finals | 607 |
| Replay | 16 events including six finals, all deduplicated |
| Audio gaps | Exactly the deliberate omission on track two, frames 124,800 through 148,799 |
| Word error rate after the first deliberately interrupted cycle | 0% on voice one and 3.45% on voice two |
| Peak sampled combined CLI and helper RSS | 70.1 MiB |
| End-of-session acknowledgment | 26 ms |

Enabling SpeechTranscriber's fast results for native live tracks reduced first-draft latency from 3.92 s to 1.04–1.06 s in the two-stream helper test. Through the optimized WebSocket server the two first drafts arrived at 1.105 s and 1.116 s from the start of paced playback. The finals arrived 314 ms and 546 ms after the client ended the session. Each had 39 monotonically timed fragments whose text reconstructed the caption. Fast reporting changed “1st” to “first” on one fixture. Standard Apple committed turns retain the previous recognition settings.

An earlier Apple soak streamed 1,808 s of two-track audio with late registration, a withheld frame range, overlapping voices, and mid-turn reconnect. It delivered 852 drafts, 217 finals, and 26 retractions. Every observed item closed exactly once, and all word fragments stayed inside their caption intervals. The server reported the injected gap and 51 additional overload gaps, totaling about 89 s per track. The sender sometimes delivered a burst after being descheduled. The two tracks' p95 first-text times were 4.23 s and 4.25 s with the original Apple reporting setting.

## Qwen and mixed-track diarization

The live Qwen default is **Qwen3-ASR-0.6B**. In the optimized native socket test it produced nine revisions and an exact transcript of 9.73 s of continuous speech across its eight-second rolling window. The first draft arrived 1.792 s after onset and the final arrived 441 ms after the fixture ended. There were no gaps or overload warnings.

The final packaged CLI also selected 0.6B with `--engine qwen3` and a cached forced aligner. At 1× playback it emitted five revisions and a final whose 16 timed fragments reconstructed the complete transcript. On the signed pre-roll release the first draft arrived 1.395 s after onset and the aligned final arrived 785 ms after the fixture ended. The server reported no gaps or queue warnings.

The optimized Qwen3-ASR-1.7B server completed two standard committed turns with the expected words, alias, and item IDs. On the loaded host, commit-to-final took 10.89 s and 3.74 s. Native streams produced revisable drafts with and without the forced aligner, then reached the queue limit at 1× playback speed. In a separate slowed test, the aligner returned 16 timed fragments reconstructing the final. These runs establish a working committed mode and alignment path, with a throughput limit on the tested 1.7B configuration.

The real low-latency Sortformer export returned speakers A/B/A on an annotated direct-inference fixture. The first integrated run produced a timed speaker interval and labeled words before the original Parakeet profile overloaded. A regression from that capture now checks that a timed overload final stops at its last confirmed word and that caption-wide speaker labels require timing coverage of the full caption.

The final packaged server produced all three timed Apple transcripts at capture speed. CPU Sortformer labeled the first and missed the next two deadlines. Metal missed all three deadlines. Diagnostics recorded a healthy worker with insufficient processed audio, including 96 dropped diarization input packets on the Metal run. Those finals explicitly announced unavailable speaker labels. Isolated microphones with assigned track names do not depend on Sortformer.

As a control, an optimized direct CPU run with a larger inference chunk and smaller history still returned the correct A/B/A identities. Its coverage arrived 4.7, 8.4, and 7.1 s after the respective caption ends. Reducing inference time did not satisfy the two-second label deadline on this host, so the alternative profile is excluded from the release.

## Model execution measurements

One worker per server owns the shared Parakeet model and schedules recognition across tracks. A separate instrumented parallel-decoder benchmark measured actual mutex acquisition under load. Across 16 encoder and 16 decoder measurements, encoder acquisition p50/p95 were 52/1,813 ms and decoder acquisition p50/p95 were 28/2,152 ms. Encoder hold p50/p95 were 1,442/2,152 ms and decoder hold p50/p95 were 23/54 ms. The raw parallel trial completed two of eight scheduled turns before its bounded input queue filled. The socket measurements above include the deployed worker's queueing.

A statically linked ONNX Runtime INT8 research build used 865 MiB of sampled process footprint in the two-track continuous socket test. It lost 12.3–12.6% of input frames under the observed load. It also attributed one of three Apple-backed Sortformer finals. The released app uses the tested Burn backend.
