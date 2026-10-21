# Repository Guidelines

## Project Structure & Module Organization

- `src/main.rs` contains the Rust CLI: job collection, audio decoding/resampling, chunking, Qwen3 inference, alignment, diarization, and progress reporting.
- `src/transcript.rs` holds the transcript schema, validation, and TXT/JSON/SRT/VTT writers; `src/model_cache.rs` resolves, downloads, and repairs Hugging Face models; `src/audio_convert.rs` holds the Core Audio and WebM→Ogg Opus fallbacks; `src/parakeet_engine.rs` wraps Parakeet (feature `parakeet`).
- `ui/Sources/TranscriberrrUI/main.swift` contains the SwiftUI drag-and-drop macOS front end; `ui/Sources/TranscriberrrNative/main.swift` is the Apple SpeechTranscriber helper the CLI runs for `--engine apple-native`; `ui/Package.swift` defines both.
- `scripts/build-macos-app.sh` assembles and signs `dist/Transcriberrr.app` (CLI and helper in `Contents/Helpers`); `scripts/build-release-dmg.sh` builds the DMG and optionally notarizes it.
- `tests/` holds Rust integration tests; add tests there as behavior grows. Do not commit `target/`, `ui/.build/`, model weights, recordings, or `dist/` artifacts.

## Build, Test, and Development Commands

Use Rust 1.99+ on Apple Silicon macOS. From the repository root:

```bash
cargo build --release --features diarization   # target/release/transcriberrr with Parakeet + Sortformer
cargo run --release --features diarization -- audio.m4a
cargo fmt --check                     # Check Rust formatting
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features             # Run Rust tests
swift build -c release --package-path ui        # UI and Apple helper
./scripts/build-macos-app.sh          # Build and sign the app bundle
./scripts/build-release-dmg.sh        # Build arm64 DMG and SHA-256 checksum
```

A plain `cargo build --release` omits Parakeet and diarization, and those engines then fail with "not included in this build". The app does not use `PATH`: it runs `TRANSCRIBERRR_CLI` / `TRANSCRIBERRR_NATIVE` when set, then the binaries in `Contents/Helpers`, then (for `swift run`) `target/release/transcriberrr` and the helper under `ui/.build`. A cargo-built CLI finds the helper under `ui/.build` too, so build the Swift package before using `--engine apple-native`.

## Coding Style & Naming Conventions

Run `cargo fmt` before committing Rust changes and keep `clippy` warning-free. Follow idiomatic Rust: `snake_case` for functions, variables, and modules; `PascalCase` for types; and uppercase `SCREAMING_SNAKE_CASE` for constants. Swift code follows Swift naming conventions (`camelCase` members and `PascalCase` types). Keep model loading, audio handling, output formatting, and UI concerns isolated.

## Testing Guidelines

Add focused Rust tests for parsing, resampling, timestamp formatting, and failure paths in `tests/` or module-local `#[cfg(test)]` blocks. Keep tests deterministic and avoid requiring downloaded model weights. Run `cargo test` for the full suite; use `cargo test test_name` while iterating. Build the SwiftUI target with `swift build -c release --package-path ui` when changing the UI.

## Commit & Pull Request Guidelines

There is no existing Git history to establish a project-specific convention. Use short, imperative commit subjects (for example, `Add transcript parser`) and keep unrelated changes separate. Pull requests should explain the user-visible change, identify important implementation choices, link a relevant issue when one exists, and include test results. Add screenshots or sample input/output when changing a user interface or generated artifacts.

## Security & Configuration Tips

Do not commit API keys, credentials, local recordings, or generated secrets. Keep developer-specific values in ignored environment files and provide a checked-in example configuration with placeholder values. Review new dependencies and generated files before committing.
