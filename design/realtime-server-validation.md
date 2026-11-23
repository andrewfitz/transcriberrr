# Realtime server validation

The acceptance runs streamed synthetic voices over loopback at capture speed on an Apple M4 Max with 48 GiB of unified memory, running macOS 27.0.1 (26A434). The fixtures and their phrase cues can be regenerated locally. The packaged app launches the same Rust server as the headless CLI. Clients own audio capture.

## Parakeet caption latency

The native multitrack protocol decodes Parakeet in 0.96 s hops with 0.64 s of left context and 0.16 s of right context. An utterance decodes one hop at a time until it has its first text. After that, when more of the track's audio is already queued behind the recognizer, the utterance decodes up to three hops (2.88 s) in one encoder window, always ending on the 0.96 s grid. Audio within 4.84 s of a forced rollover decodes one hop at a time, which keeps word timing identical on both sides of the cut. Both standard OpenAI aliases keep their 1.44 s hop and never combine hops. A forced native rollover retains 1.44 s of preceding audio. The 1.92 s profile that 0.4.0 chose for a backlogged utterance is removed.

One encoder call costs 60–85 ms on Metal for any window from 0.3 s to 4 s, so the server keeps up easily until other GPU or CPU work delays encoder calls. Encoder time was 89% of the slow calls traced. With a fixed hop, a delayed call leaves the same number of calls still to run, and the first use of each window length in a process costs a further 0.15–1 s. Combining hops removes calls from a track that is behind. The listener now runs the encoder once at every window length a stream can produce before it reports ready. It became ready 8.5–9.7 s after launch on a lightly loaded host, against 3.6–4.7 s when only the full-hop lengths were prepared first and the rest followed on a background thread. That thread was removed: a background call still in flight when the first stream opened held the model for seconds on a busy host. In contended 4× runs that started at the ready line, three of eight runs with background warm-up had later-revision p95 between 11.3 s and 13.6 s, and none of four runs without it exceeded 6.6 s. That sample is small and the host was busy. In the seven later rounds reported below, the build without the background thread reached 7.08 s once on the same measure, at a one-minute load above 100, and its first drafts were consistently earlier than the 0.4.1 build's.

The streaming loop is `src/serve/parakeet_stream.rs`. It calls the public model API of parakeet-rs, which stays an unmodified crates.io dependency. `examples/parakeet_parity.rs` compares the loop token for token with parakeet-rs's own streaming and should be run after upgrading the crate. It matched on 480 of 480 decodes (160 Mini LibriSpeech utterances at three hops). On those utterances, three-hop windows scored 3.16% word error against 5.12% for single 0.96 s hops (3,166 words).

The build pins parakeet-rs to 0.4.4, the latest release, and downloads every Parakeet model from one pinned revision (`26684938ff242cb4f7a65fad712476c6015a4477`) of `altunenes/parakeet-rs`, the repository whose exports the crate generates its Burn graphs from: `parakeet-unified-en-0.6b`, `parakeet-tdt-0.6b-v3` and `nemotron-3-diarization`. A Burn graph loads only the export it was generated from. parakeet-rs 0.4.4 rejects the `eschmidbauer/parakeet-unified-en-0.6b-onnx`, `istupakov/parakeet-tdt-0.6b-v3-onnx` and earlier Sortformer files that 0.4.0 downloaded, so those are no longer used and can be deleted from the model cache. The crate version and the revision change together, followed by `examples/parakeet_parity.rs`. The upgrade needed one added argument at the encoder call (`run_encoder(&features, None)`). On 0.4.4 with the new English export the loop matched parakeet-rs's own streaming on 480 of 480 decodes, with the same word error rates as before (5.12% for single 0.96 s hops, 3.16% for three-hop windows). All three models downloaded through the app's downloader, resolved offline and transcribed. One encoder call costs the same on the new export (steady median 58–68 ms against 61–67 ms on the old one, flat from 0.3 s to 4 s windows).

### Final build measurements

These runs measure the packaged helper built with parakeet-rs 0.4.4, the pinned exports, the corrected word times and the warm-up before ready (SHA-256 `dee678704f7fde09cc0900fe050646ff392264e1996fe17bcf7821898018d29d`). It ran interleaved with the signed 0.4.0 helper (`e9a0ec77…`) and the 0.4.1 follow-up helper of the next section (`cbd55451…`) on the same M4 Max, with the same fixtures and contention. The worktree also holds unrelated changes from other work, so each comparison is between builds as a whole.

Published word times now end 0.3–0.4 s earlier (see Parakeet word timing). A latency keyed on the last timed word therefore reads higher for this build although captions arrive no later. Where builds are compared here, draft latency is the time from sending the packet that holds the last audio a draft covers (its `end_frame`) to receiving the draft. The two helpers sent the same drafts over the same audio: in the quiet natural-speech pair and both quiet 306 s pairs all 573 drafts matched one for one, each with the same revision number and the same `end_frame`. First-draft latency on the paced fixture is measured from the annotated onset and does not depend on word times.

Two voices each supplied 46.1 s of continuous speech at 4× capture speed with contention throughout. Values are per round. Another session's test run overlapped round 1.

| Measure | Signed 0.4.0 | 0.4.1 follow-up | Final |
| --- | --- | --- | --- |
| Later draft p95, from covered audio | 5.50 / 4.83 / 4.88 / 5.17 s | 3.14 / 2.08 / 2.21 / 2.63 s | 3.31 / 2.76 / 2.69 / 3.59 s |
| Later revision p95, from last timed word | 5.82 / 5.11 / 5.18 / 5.29 s | 3.31 / 2.28 / 2.47 / 2.84 s | 3.56 / 3.00 / 2.94 / 3.80 s |
| First draft median, from covered audio (four items) | 0.18 / 0.14 / 0.20 / 0.20 s | 1.05 / 0.45 / 0.55 / 0.65 s | 0.19 / 0.13 / 0.16 / 0.22 s |
| Wall time for 11.5 s of sends | 21.3 / 19.8 / 19.4 / 19.0 s | 15.9 / 15.2 / 15.5 / 15.7 s | 16.6 / 15.8 / 16.0 / 16.7 s |
| Maximum interval between packet groups | 0.93 / 0.72 / 0.75 / 0.66 s | 0.50 / 0.03 / 0.03 / 0.18 s | 0.51 / 0.17 / 0.28 / 0.92 s |
| Slower track’s commit acknowledgment | 3.90 / 3.79 / 3.68 / 4.67 s | 2.70 / 2.39 / 2.66 / 2.79 s | 3.15 / 2.84 / 2.86 / 2.80 s |
| Word edits across 325 reference words | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 |

All twelve sessions ended normally with exact rollover boundaries and zero gap frames. The final helper was ahead of the signed helper in every round and behind the 0.4.1 helper by 0.2–1.0 s at p95. Three further rounds with the recognition trace on did not reproduce that order: later-draft p95 was 2.49 / 2.19 / 6.74 s for the final helper against 2.86 / 2.72 / 4.20 s for the 0.4.1 helper, with the one-minute load rising from 50 to 122 across the rounds because of other work on the host. In those rounds the two builds made the same number of encoder calls (149 and 147) at the same cost (median 196 ms against 221 ms for a single hop and 313 ms against 293 ms for a three-hop window, under contention). The trace shows no step that is slower on 0.4.4, so the difference in the first four rounds is not attributed. The first draft of each track was earlier on the final helper in all seven rounds (median 0.12–0.46 s against 0.45–1.28 s). That is consistent with finishing the warm-up before ready: two builds of the same tree that differ only in that showed the same order in four of four earlier rounds (0.16–1.26 s against 0.71–1.40 s), and the first single-hop call of a traced run took 60–172 ms on the final helper against 436–937 ms on the 0.4.1 helper. Without added contention the three builds were equal at 4× (later-draft p95 0.11–0.13 s in two rounds each).

The paced two-microphone fixture ran for 126 s at capture speed with a late join, a deliberate one-second gap and a reconnect with replay. Contention ran for 8 s of every 20 s. Three rounds per build are pooled (126 annotated utterances each). One-minute load was 12–29.

| Measure | Signed 0.4.0 | 0.4.1 follow-up | Final |
| --- | --- | --- | --- |
| First draft from sending the onset packet, p50 / p95 / max | 1.09 / 1.22 / 1.93 s | 1.09 / 1.21 / 1.97 s | 1.09 / 1.25 / 1.88 s |
| First-draft p95 per round | 1.27 / 1.20 / 1.21 s | 1.21 / 1.21 / 1.20 s | 1.25 / 1.25 / 1.24 s |
| Later draft from covered audio, p50 / p95 / max | 0.18 / 0.40 / 1.07 s | 0.18 / 0.27 / 0.75 s | 0.18 / 0.34 / 0.60 s |
| Later revision from last timed word, p50 / p95 / max | 0.59 / 1.00 / 1.47 s | 0.59 / 0.99 / 1.85 s | 0.86 / 1.20 / 1.61 s |
| Drafts before the end-of-speech packet | 126 of 126 | 126 of 126 | 126 of 126 |
| Backlog events | 5 | 1 | 1 |
| Word edits across 990 scored words | 0 | 0 | 0 |

Every run had zero unexpected gaps, no duplicate or open items, and observed all replayed events. At this load the three builds delivered first drafts equally. The first-draft gain of the next section appeared under heavier host load (one-minute load up to 65) and was not repeated here.

Without added contention, 306 s runs in signed/final/final/signed order were equal: first-draft p95 1.20 s for both, later-draft p95 from covered audio 0.19 s against 0.21 s, 204 of 204 drafts before the end of speech for both, and zero edits across 880 scored words in every run. Measured from the last timed word, later-revision p95 was 1.18 s against 0.98 s.

Natural read speech (two Mini LibriSpeech speakers, 150 s, 460 reference words) gave the same 26 finals with identical text on both builds in all six runs, with 34 edits against the LibriSpeech transcripts and zero unexpected gaps.

| Run | Build | First draft p95 / max | Later draft p95 / max | Backlog events |
| --- | --- | --- | --- | --- |
| No added contention | Signed 0.4.0 | 0.26 / 0.26 s | 0.26 / 0.39 s | 0 |
| No added contention | Final | 0.26 / 0.26 s | 0.27 / 0.41 s | 0 |
| Scheduled contention, round 1 | Signed 0.4.0 | 0.53 / 0.53 s | 0.50 / 0.87 s | 0 |
| Scheduled contention, round 1 | Final | 0.54 / 0.63 s | 0.64 / 0.90 s | 0 |
| Scheduled contention, round 2 | Signed 0.4.0 | 0.45 / 0.49 s | 0.81 / 1.33 s | 4 |
| Scheduled contention, round 2 | Final | 0.40 / 0.51 s | 0.68 / 0.94 s | 2 |

These are measured from covered audio. Measured from the last timed word, the quiet runs read 1.47 s p95 for the final helper against 1.16 s for the signed helper.

The 30m18s soak on the final helper finished in 30m32s. All 606 annotated utterances had a draft before their end-of-speech packet. First-draft latency from the onset packet was 1.10 s median, 1.26 s p95 and 1.83 s maximum. The 705 later drafts took 0.18 s median, 0.38 s p95 and 0.72 s maximum from covered audio (0.88 s, 1.20 s and 1.54 s from the last timed word). There were zero edits across 5,500 scored words, zero unexpected gaps, zero backlog events and a recognition queue depth of at most ten. The longest packet-group send took 0.31 s and the drain took 0.01 s. Resident memory stayed between 134 and 275 MiB. One-minute host load ranged from 4 to 36, and other sessions ran release builds and test suites for 21 minutes of the run. The fixture sender trailed an uninterrupted capture clock by 13.5 s at the end (0.64 ms per 100 ms block plus 1.9 s from nineteen pauses). Measured against an uninterrupted clock as an estimate, first drafts were 8.76 s median and 13.78 s p95, almost all of it sender drift. No physical microphone latency was measured.

On the standard socket the final helper returned the reference sentence for both aliases, with no drafts before commit for `gpt-transcribe` and three stable deltas before commit for `gpt-live-transcribe`. Interleaved 30 s turns at capture speed completed 1.03–1.08 s after commit on the final helper and 1.01–1.02 s on the signed helper for `gpt-transcribe` (four turns each), and 0.06–0.09 s on both for `gpt-live-transcribe`.

As in the next section, the final helper retracts one mic-two item as bleed in the first cycle, which deliberately removes one second of mic-two input and is excluded from scoring. The signed helper does not.

The remaining limit is unchanged: encoder time under GPU contention. Under continuous 4× contention one encoder call took 0.2–0.3 s at the median and up to 0.9 s, against 60–85 ms on a free GPU, and work waited 0.5–0.9 s at the median for the call ahead of it.

### Follow-up measurements on parakeet-rs 0.4.1

These runs predate the upgrade to parakeet-rs 0.4.4, the warm-up before ready and the corrected word times. They compare the signed 0.4.0 helper (SHA-256 `e9a0ec77d0436d04edd1e6be5d1f464dada5c0a7607d9175890579710018872a`) with the packaged follow-up helper (`cbd5545168e994f35a0533e4be354f7a222313e62373c6dae374ecff5c42fba6`) on the same M4 Max, interleaved within each round. The follow-up helper was built from a worktree that also holds unrelated changes from other work, so the comparison is between the two builds as a whole. Added contention is four processes running the Parakeet encoder on Metal plus 32 CPU spinners.

Two voices each supplied 46.1 s of continuous speech at 4× capture speed, with the contention running throughout. Values are per round.

| Measure | Signed 0.4.0 | Follow-up |
| --- | --- | --- |
| Later caption revision p95 | 4.86 / 4.66 / 5.56 / 5.69 s | 2.34 / 2.22 / 2.33 / 3.30 s |
| Wall time for 11.5 s of sends | 19.1 / 18.4 / 19.4 / 20.3 s | 14.6 / 14.7 / 14.9 / 16.3 s |
| Maximum interval between packet groups | 0.61 / 0.71 / 1.00 / 1.32 s | 0.03 / 0.03 / 0.03 / 0.43 s |
| Slower track’s commit acknowledgment | 5.16 / 3.55 / 3.56 / 4.10 s | 2.21 / 2.23 / 2.37 / 3.49 s |
| Word edits across 325 reference words | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 |

All eight sessions ended normally with exact rollover boundaries and zero gap frames. Without added contention the two builds were equal at 4× (later-revision p95 0.32–0.33 s in two rounds each).

One measurement build with the hop policy as a switch isolates the cause from the other changes in the worktree. In four contended 4× rounds its later-revision p95 was 9.06 / 6.68 / 9.93 / 6.65 s with a fixed 0.96 s hop, 8.84 / 6.19 / 16.67 / 5.23 s with the 0.4.0 policy and 5.10 / 1.37 / 1.93 / 0.97 s when combining hops. The signed helper gave 7.92 / 5.83 / 5.09 / 5.74 s in the same rounds.

The paced two-microphone fixture ran for 126 s at capture speed with a late join, a deliberate one-second gap and a reconnect with replay. Contention ran for 8 s of every 20 s. Four rounds per build are pooled (168 annotated utterances each).

| Measure | Signed 0.4.0 | Follow-up |
| --- | --- | --- |
| First draft from sending the onset packet, p50 / p95 / max | 1.10 / 1.55 / 2.77 s | 1.09 / 1.22 / 2.09 s |
| First-draft p95 per round | 1.31 / 2.01 / 1.35 / 1.55 s | 1.19 / 1.22 / 1.24 / 1.22 s |
| Later revision, p50 / p95 / max | 0.60 / 1.24 / 2.52 s | 0.59 / 0.99 / 1.42 s |
| Drafts before the end-of-speech packet | 164 of 168 | 167 of 168 |
| Backlog events | 36 | 0 |
| Word edits across 1,320 scored words | 0 | 0 |

Every run had zero unexpected gaps, no duplicate or open items, and observed all replayed events. An earlier round of the same experiment under heavier host load (one-minute load 47–74) gave a signed first-draft p95 of 6.19 s and a later-revision maximum of 6.96 s, against 2.81 s and 1.86 s for an earlier measurement build that combined hops. A further round under light contention showed no first-draft difference between hop policies. The gain is in the tails under contention.

Without added contention, 306 s runs in signed/follow-up/follow-up/signed order were equal: first-draft p95 1.20 s against 1.18 s, later-revision p95 0.98 s against 0.97 s, 204 of 204 drafts before the end of speech for both, and zero edits across 880 scored words in every run.

Natural read speech went through the same socket as two Mini LibriSpeech speakers on separate tracks, 150 s at capture speed with 460 reference words. Captions were scored against the LibriSpeech transcripts after removing case and punctuation only. Caption latency is the time from sending the packet that holds a draft's last word to receiving the draft.

| Run | Build | Draft p95 / max | First text p95 | Backlog events | Edits |
| --- | --- | --- | --- | --- | --- |
| No added contention | Signed 0.4.0 | 1.16 / 2.47 s | 1.17 s | 0 | 34 |
| No added contention | Follow-up | 1.18 / 2.49 s | 1.18 s | 0 | 34 |
| Scheduled contention, round 1 | Signed 0.4.0 | 1.68 / 2.53 s | 1.70 s | 13 | 34 |
| Scheduled contention, round 1 | Follow-up | 1.17 / 2.47 s | 1.67 s | 0 | 34 |
| Scheduled contention, round 2 | Signed 0.4.0 | 2.40 / 4.66 s | 1.88 s | 21 | 34 |
| Scheduled contention, round 2 | Follow-up | 2.26 / 3.71 s | 1.97 s | 28 | 30 |

All six runs ended normally with zero unexpected gaps and 26 finals. The follow-up build was clearly ahead in one of the two contended rounds, and first-text latency did not improve. Its final text was identical to the signed helper's in every run except contended round 2, where one combined window corrected a phrase the single-hop decode got wrong.

The 30m18s soak on the packaged follow-up helper finished in 30m33s. All 606 annotated utterances had a draft before their end-of-speech packet. First-draft latency was 1.08 s median, 1.20 s p95 and 2.74 s maximum. The 705 later revisions took 0.58 s median, 0.97 s p95 and 1.43 s maximum. There were zero edits across 5,500 scored words, zero unexpected gaps, zero backlog events and a recognition queue depth of at most nine. The longest packet-group send took 0.50 s and the drain took 0.01 s. Resident memory stayed between 143 and 291 MiB. One-minute host load ranged from 7 to 35 during the run. The same soak on 0.4.0 took 31m08s with a first-draft p95 of 3.85 s, a maximum of 34.08 s and 540 early drafts, under different host load.

The fixture sender rebases its clock after every late wake, so it trails an uninterrupted capture clock even when the server never stalls it. In the soak it trailed by 15.2 s at the end: 12.7 s accumulated at 0.7 ms per 100 ms block and 2.5 s came from twelve pauses longer than 50 ms. Measured against an uninterrupted clock as an estimate, first drafts in the soak were therefore 10.3 s median and 15.2 s p95, almost all of it sender drift. In the 306 s runs the same estimate was 2.82 s p95 for the signed helper and 1.94 s for the follow-up. No physical microphone latency was measured.

On the standard socket, the follow-up helper returned the reference sentence for both aliases, with no drafts before commit for `gpt-transcribe` and three stable deltas before commit for `gpt-live-transcribe`. Commit-to-completed time showed no measurable difference. Five 30 s `gpt-transcribe` turns sent at capture speed completed in 1.28–1.78 s on the follow-up helper and 1.27–1.65 s on the signed helper, and a 10 s turn in 0.48 s and 0.97 s. The runs were not interleaved. `gpt-live-transcribe` completed 10 s and 30 s turns 0.06–0.07 s after commit on both builds.

In the first cycle, which deliberately removes one second of mic-two input and is excluded from scoring, the follow-up build retracts one mic-two item as bleed and the signed helper does not. Fixed-hop measurement builds of the same worktree do the same, so this comes from the worktree's live-bleed changes.

The remaining limit is encoder time under GPU contention. A track that is behind still waits for the encoder call in progress, and an utterance's first text still needs one full 0.96 s hop plus 0.16 s of right context.

### 0.4.0 measurements

The 0.4.0 release used a 0.96 s hop for a new utterance while its track had at most one second of admitted audio awaiting recognition, and a 1.92 s hop above that threshold. Each utterance kept its chosen hop through completion.

On an M4 Max, two different voices each supplied 46.1 s of continuous speech through the native WebSocket at 4× capture speed. The old and signed adaptive builds ran in old/new/new/old order. Revision latency measures elapsed time from sending the packet containing the last timed word to receiving the updated caption, excluding the first draft. Single runs are shown in order within each build.

| Measure | Previous build | Adaptive build |
| --- | --- | --- |
| Later caption revision p95 | 5.65 s / 6.76 s | 0.42 s / 0.39 s |
| Maximum interval between packet groups | 0.88 s / 1.73 s | 0.038 s / 0.038 s |
| Slower track’s commit acknowledgment | 2.00 s / 6.32 s | 0.35 s / 0.24 s |
| Word edits across 325 reference words | 3 / 3 | 1 / 1 |
| Unexpected gap frames | 0 / 0 | 0 / 0 |

These four tests used the signed candidate prior to the standard-alias compatibility correction. Both tracks reached their exact final frame and matched the reference words spanning each forced rollover in all four runs. Host load varied during the sequence. The 4× replay measures headroom when two speakers talk continuously.

The 30m18s native two-mic candidate run completed in 31m08s wall time. All 606 annotated utterances produced drafts. First-draft latency measured from sending the onset packet was 1.22 s median, 3.85 s p95 and 34.08 s maximum. For 540 utterances, the draft arrived before the end-of-speech packet. There were zero unexpected gaps, zero duplicate terminal items and zero edits across 5,500 scored words. Scoring excludes the first cycle, which deliberately removes one second of mic-two input. Both tracks ended and the reconnect recovered all replay events.

The earlier fixed-hop 30-minute replay reached 46.98 s p95 and 74.30 s maximum, with 205 early drafts. Measured one-minute host load was lower in the adaptive run (median 43.85 vs 80.14). In the adaptive run, the longest client write stalled for 21.13 s. The fixture sender rebases its clock after stalls. Using a continuously advancing source clock from the first packet as an estimate gives 49.55 s p95 from nominal capture time, compared with 325.79 s in the earlier run. No physical microphone latency was measured.

A separate standard live-alias check found that applying the 0.96 s native hop there omitted “The” and repeated “morning” on an existing SDK fixture. The compatibility build keeps adaptive decoding on the native protocol and restores the established standard window.

The final packaged CLI returned the expected text on the standard fixture for both model aliases, with three live deltas before commit. The subsequent 4× native stress repeat on this binary gave later-revision p95s of 11.10 s and 10.16 s versus 8.36 s and 8.58 s for the fixed-hop controls. This repeat reversed the latency result from the first sequence. All four sessions ended normally with zero unexpected gaps and exact rollover boundaries. The adaptive build had one word edit per 325 words and the controls had three. The 4× trials do not establish a consistent sustained-overload latency gain.

In the final 306 s capture-paced comparison, median first-draft latency improved from 1.74 s to 1.23 s. The p95 rose from 2.88 s to 4.04 s, with median host load rising from 39.67 to 48.90. Early drafts rose from 85 to 88 of 102 cues. The final build delivered drafts for all 102 cues and matched all 880 scored words with zero unexpected gaps. The 30-minute measurement above belongs to the earlier signed candidate. The final DMG SHA-256 is `67356c6e015e8fb9dede199874ac5899a8aec468212e2e29458758e1e7956bea`. The app signature, disk image, checksum, 288 all-feature tests, 261 minimal tests and strict Clippy passed.

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

On the original packaged Apple Speech and CPU Sortformer path, both consumers received every audio frame. One inference window took 4.605 s, and the third final lost all speaker labels despite having decisions for most of its words.

The revised release CLI and bundled Apple helper identified both speakers in A/B/A order in three capture-speed replays of the same 19.158 s mixed fixture. All nine ASR finals arrived, every session ended normally, and the server reported zero audio gaps. Counts below show unambiguously labeled words across the three captions, which contain 16, 12 and 16 words respectively.

| Per-track speaker wait | Words labeled in A/B/A | Finals with partial coverage | Third final after fixture speech ended |
| --- | --- | ---: | ---: |
| Default, 2 seconds | 15 / 10 / 13 | 2 | 2.310 s |
| 10 seconds | 15 / 10 / 15 | 0 | 3.050 s |
| Zero | 12 / 8 / 9 | 3 | 0.300 s |

The default run again reached frame 504,852 before the third caption's deadline, with the caption ending at frame 521,748. The revised final retained 13 confirmed labels and left the crossing and undecided words unlabeled. Coverage later reached the fixture's last frame, 549,920, in all three runs. These trials used the release CLI with SHA-256 `1be442d82939f5f5be7d54ace7517310fafea151f0f208ecb7c9eb3fd9f492f5` on a shared M4 Max.

The first long replay exposed a client heartbeat failure after 390 s of capture. While the native server paused reads under pressure, its writer continued delivering new finals, but the reference client disconnected when a pong timed out behind the queued audio. The native reference client now disables automatic pings. A paused-transport regression reproduced the previous WebSocket 1011 and passed with all 6,000 frames after the fix.

A follow-up run was interrupted by a host reboot after 24m36s of input. The completed post-reboot run replayed the entire 30m18s native Parakeet fixture in 35m46s wall time on the same M4 Max. Both tracks ended normally with zero unexpected gap frames, zero duplicate terminal items and zero open items. The only gap was the one second deliberately withheld on mic two. The session resumed with zero unobserved replay events. The 100 uninterrupted cycles scored 0 edits across 2,600 mic-one words and 2,900 mic-two words. All 607 final captions carried the correct track label.

On the 605 of 606 annotated utterances with a measured first draft, latency from sending the onset packet to receiving the first draft was 3.26 s median, 46.98 s p95 and 74.30 s maximum. Only 205 drafts arrived before the corresponding end-of-speech packet. A send stalled for up to 22.47 s, and final drain took 3.41 s. These timings start at the paced sender. A physical microphone capturing continuously would also need to buffer audio while writes stall.

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

That release used **1.44 s chunks, 0.64 s of left context, and 0.16 s of right context**. In the optimized inference benchmark, two tracks completed two turns each with zero edits over 130 reference words. Across those four turns first-text p50/p95 were 2.04/2.28 s and commit-to-final p50/p95 were 0.28/1.11 s. This measurement includes the recognizer and its input queues. The original-profile table below includes the full WebSocket, gating, and bleed path.

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

## Parakeet word timing

The streaming decoder stamps a word with the encoder frame that emitted its first token and gives it one or two 80 ms frames of length. That stamp trails the word. The listener keeps it for rollover cuts and bleed decisions, and sends clients and speaker labelling the time the word was spoken instead. The spoken time is the stamp less a latency that depends on how much audio the decoding window held before the token: 0.41 s with only the 0.64 s left context, 10 ms less for each further encoder frame, and never below 0.30 s. A stream's first window has no left context and trails by 0.50 s, falling 70 ms per frame after its ninth. A word ends at the next word's start, or before a pause after a length estimated from its spelling, as in batch transcription.

Each run sent two Mini LibriSpeech speakers through the native socket for 150 s at capture speed. The reference is the Qwen3 forced aligner on the same recordings, which reports times in 80 ms steps. Batch Parakeet word starts differ from it by −1 and −31 ms on the two fitted recordings (standard deviation 62 and 63 ms), so about 60 ms of the spread below belongs to the reference. Values are the mean and standard deviation of published time minus reference.

| Measure | Decoder stamps, 0.4.0 export | Spoken times, fitted pair | Spoken times, held-out pair |
| --- | --- | --- | --- |
| Word start | +364 ms, 84 ms | −6 ms, 65 ms | −4 ms, 74 ms |
| Word end | +207 ms, 148 ms | +2 ms, 73 ms | −4 ms, 85 ms |
| First word of an item | +516 ms, 62 ms | +23 ms, 68 ms | +70 ms, 138 ms |
| Words compared | 421 | 426 | 506 |

The latency constants were fitted on the first pair with the 0.4.4 English export. The held-out pair is two other speakers. On the synthetic two-microphone fixture, whose onsets are annotated from the audio, the first word of a phrase started 20 ms early (standard deviation 54 ms, 11 phrases) and the last word ended 30 ms late (55 ms, 12 phrases). A run of the fitted pair while four processes ran the Parakeet encoder on Metal, which delayed finals by up to 43 s and made windows span several hops, gave −7 ms for starts (63 ms) and +2 ms for ends (74 ms) over 433 words.

Bleed decisions stay on the decoder's stamps. With spoken times, the fixture's first cycle published two words of the other speaker on microphone two that the stamps mute: "The" in the plain run and "Apples" in the run with a late join and a gap. A short word at the start of a phrase can end before the bleed is measurable, and the stamp falls a few hundred milliseconds into the phrase. Which clock mutes better on natural overlapping speech has not been measured.

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
