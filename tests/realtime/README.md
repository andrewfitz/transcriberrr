# Realtime acceptance tests

Run the deterministic WebSocket tests with a fake recognizer and the production listener. The tests need no downloaded model weights.

```sh
cargo test --all-features serve::ws_tests --bin transcriberrr
```

The reference client also has a scripted WebSocket peer that checks delayed reconnects, replay, and exact audio ranges. With Python 3.11+ and `websockets==15.0.1` installed, run `python tests/realtime/reference_client.py -v`.

To check an actual client SDK, start a release server and run two paced turns through the official Python client. On macOS, generate a short recording with `say` and ffmpeg. Use `--stable` with Parakeet to require text before commit. Omit it to exercise committed turns on any engine.

```sh
sdk_test="$(mktemp -d)"
python3 -m venv "$sdk_test/client"
"$sdk_test/client/bin/python" -m pip install 'openai[realtime]==3.57.0'
say -v Samantha -o "$sdk_test/speech.aiff" 'The meeting starts on Monday morning. Please bring the red folder to the conference room.'
ffmpeg -loglevel error -i "$sdk_test/speech.aiff" -ac 1 -ar 24000 -c:a pcm_s16le "$sdk_test/speech.wav"
"$sdk_test/client/bin/python" tests/realtime/openai_sdk.py "$sdk_test/speech.wav" \
  --base-url http://127.0.0.1:8080/v1 --contains monday --contains folder --stable
```

For the real two-track soak, build the release CLI, warm the Parakeet Unified model, and start the server. Generate synthetic voices with macOS `say` and ffmpeg. The fixture directory contains PCM WAVs and independently placed phrase cues in `fixtures.json`.

```sh
cargo build --release --features diarization
target/release/transcriberrr serve --engine parakeet-v3 --port 8080
```

In a second terminal, generate and stream the fixtures.

```sh
fixtures="$(mktemp -d)"
python3 -m venv "$fixtures/client"
"$fixtures/client/bin/python" -m pip install websockets==15.0.1
python3 tests/realtime/fixtures.py "$fixtures"
"$fixtures/client/bin/python" scripts/realtime-client.py --url ws://127.0.0.1:8080 \
  --native --repeat 101 --late-join 3.6 --gap-at 5.2 --gap-seconds 1 \
  --reconnect-at 23.1 "$fixtures/alice-bleed.wav" "$fixtures/bob-bleed.wav"
```

Each clip is 18 seconds and includes distinct phrases, overlap, silence, and 2.5% crosstalk. The 101 repetitions contain 30 minutes and 18 seconds of audio. The client checks sequence continuity, monotonically increasing draft revisions, and one terminal event per observed item. The capture should include a gap for the second track spanning frames 124,800 through 148,799. Inspect the caption intervals and word fragments against the cue ranges and read `GET /healthz` during the run for recognition backlog.

[Reference client](../../scripts/realtime-client.py) · [Wire examples](../../design/realtime-traces/README.md)
