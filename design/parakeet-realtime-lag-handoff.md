# Handoff prompt — fix Parakeet live caption lag

You own the Parakeet caption-lag fix in Transcriberrr’s realtime transcription server. A listener following two live microphones should see words while people are still speaking, including during overlap. Improve both time to first draft and freshness of subsequent captions on Apple Silicon. The macOS app and headless CLI run the same Rust server. Work through the WebSocket entry point and verify the packaged CLI. Preserve current worktree edits and record the starting binary hash, source diff, hardware, and host load so someone else can reproduce your comparison.

## What remains broken

The current signed build still has multi-second caption stalls. Three comparisons explain why another pass is needed. All onset figures below measure from the actual send of the onset packet to receipt of the first draft.

| Workload | Observed result |
| --- | --- |
| Paced 306 seconds, older fixed-hop build → final signed build | Median improved from 1.74 to 1.23 s. P95 worsened from 2.88 to 4.04 s and worst case from 3.39 to 7.15 s. Drafts arrived before speech ended for 85 → 88 of 102 utterances. The final build drafted all 102, with 0/880 word edits and only the deliberately injected gap. |
| Two continuous voices at 4×, old/final/final/old | Later-caption revision p95 was 8.36 / 11.10 / 10.16 / 8.58 s. This measures receipt relative to the send of the packet containing the revision’s last timed word, excluding first drafts. The final build had 1/325 word edits, with exact forced-cut boundaries and no gaps. An earlier candidate’s subsecond result failed to reproduce. |
| Paced 30m18s, older fixed hop → earlier adaptive candidate | First-draft p95 fell from 46.98 to 3.85 s, worst case from 74.30 to 34.08 s, and drafts before speech ended rose from 205 to 540 of 606. Host loads differed substantially. The candidate took 31m08s wall time and a single send stalled for 21.13 s. Its estimated uninterrupted-capture p95 was 49.55 s. The final signed compatibility build still needs this long run. |

The paced client rebases its clock after a blocked send. Measure both actual-send latency and lag against uninterrupted capture. Label the latter as an estimate when using fixtures. An earlier overload bug lost roughly 205 seconds of audio **per microphone**. Bounded backpressure and replay from the accepted-frame cursor fixed that loss. Keep those guarantees.

## Implementation and prior attempts

Read `src/serve/parakeet_profile.rs`, `recognizer.rs`, and `pipeline.rs` under the same directory. Follow `run_worker`, `start_native`, and `Item::decode_pending`. See `src/serve/tracks.rs` for staging and reconnect, and `scripts/realtime-client.py` for pacing. Native sessions choose a 0.96 s hop at item creation, or 1.92 s when that track has over one second of queued audio. The choice lasts through the utterance. Left/right context is 0.64/0.16 s. Standard aliases use a 1.44 s hop. Applying native tuning to standard sessions previously dropped “The” and duplicated “morning” in the compatibility fixture.

- Shortening left context to 0.32 s introduced 5/65 word edits. An unconditional 0.96 s hop increased encoder launches and reached 22.22 s p95 in one stress run. A 2.4 s hop drafted only 12/18 short utterances, versus 17/18 at 1.92 s.
- QoS and GPU submission tuning produced no repeatable same-hop gain. Parallel workers shared an encoder mutex. In one benchmark the median encoder hold was 1.442 s versus 0.023 s for the decoder. Suppressing outgoing drafts still leaves stateful encoder work.
- An unshipped FP32 ONNX Runtime six-thread CPU probe retained 0/65 and 0/110 fixture edits and warrants a matched comparison. An older INT8 probe degraded accuracy. Include natural speech in quality checks. Two AMI excerpts produced 35/92 edits on both the original and short-hop profiles.

## Reproduce and investigate

Run from the repository root. The signed 0.4.0 DMG used for the final comparison had SHA-256 `67356c6e015e8fb9dede199874ac5899a8aec468212e2e29458758e1e7956bea`. Verify the helper against `final-306s/soak.run.json` in the evidence directory before treating it as the final baseline.

```sh
E=/Volumes/8TB/cinesomatic-build/transcriberrr-realtime/evidence
D="$E/parakeet-lag-jmIBjD"
PY="$E/../sdk/bin/python"
RUN=$(mktemp -d "$E/parakeet-followup.XXXXXX")
CLI="$PWD/dist/Transcriberrr.app/Contents/Helpers/transcriberrr"
SOAK_CLIENT_PATH="$PWD/scripts/realtime-client.py" "$PY" "$E/astra-drop-audit/soak.py" "$CLI" \
  --engine parakeet-v3 --repeat 17 --late-join 3.6 --gap-at 5.2 \
  --reconnect-at 23.1 --rewind-sequence 16 --steady-pacing --timed \
  --fixtures "$E/soak/annotated-fixtures" --output "$RUN/normal"
"$PY" "$E/astra-drop-audit/analyze.py" "$RUN/normal" > "$RUN/normal/summary.json"
```

Use `--repeat 101` for the long soak. Start with `$D/README.md`. Final evidence is in `final-306s/summary.json`, `final-matched-306s.json`, and `compat-dense-acceptance.json`. Earlier candidate evidence is in `adaptive-30m/summary.json` and `production-dense-acceptance.json`. Adapt `compat-dense-acceptance.py` and `final-alias-check.py` to fresh outputs and explicit binary paths. Some drivers hardcode historical paths.

Trace a stalled utterance from capture through send, admission, queueing, encoder start/end, and client receipt. Check whether per-track pressure at item creation misses aggregate contention or pressure changes during continuous speech. Use those measurements to pick the fix.

## Evidence required to close

Compare the signed baseline and candidate in interleaved runs under comparable load. Prioritize paced 1× multitrack speech. Use 4× continuous overlap to expose saturation. Report first-draft and later-revision p50/p95/max, event counts, drafts before speech ends, send stalls, wall time, drain time, and queue growth. Retain failed runs. Demonstrate a repeatable tail-latency improvement, then run the 30m18s soak on the exact packaged candidate.

Verify word accuracy, per-track timestamps, forced rollovers, late join, deliberate gap, and reconnect/replay. Require zero unexpected gaps. Check both standard aliases against the existing reference sentence, with stable deltas before commit for `gpt-live-transcribe` and zero drafts for `gpt-transcribe`. Run the Rust suites with all features and no default features, strict Clippy, and package verification. Return the patch, binary hashes, commands, and raw evidence paths. If the tails still regress, identify the measured bottleneck and report the remaining limitation.
