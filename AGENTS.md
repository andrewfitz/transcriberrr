# Repository Guidelines

## Project Structure & Module Organization

- `src/main.rs` contains the Rust CLI, audio decoding/resampling, Qwen3 inference, alignment, diarization, and output writers.
- `ui/Sources/TranscriberrrUI/main.swift` contains the SwiftUI drag-and-drop macOS front end; `ui/Package.swift` defines its Swift package.
- `scripts/build-macos-app.sh` assembles `dist/Transcriberrr.app` after the Swift build.
- `tests/` is reserved for Rust integration tests; add tests there as behavior grows. Do not commit `target/`, `ui/.build/`, model weights, recordings, or `dist/` artifacts.

## Build, Test, and Development Commands

Use Rust 1.99+ on Apple Silicon macOS. From the repository root:

```bash
cargo build --release                 # Build target/release/transcriberrr
cargo run --release -- audio.m4a      # Run the CLI locally
cargo fmt --check                     # Check Rust formatting
cargo clippy --all-targets --all-features -- -D warnings
cargo test                            # Run Rust tests
./scripts/build-macos-app.sh          # Build the SwiftUI app bundle
./scripts/build-release-dmg.sh        # Build arm64 DMG and SHA-256 checksum
```

The UI launches the CLI by name, so put `target/release` on `PATH` before opening the app.

## Coding Style & Naming Conventions

Run `cargo fmt` before committing Rust changes and keep `clippy` warning-free. Follow idiomatic Rust: `snake_case` for functions, variables, and modules; `PascalCase` for types; and uppercase `SCREAMING_SNAKE_CASE` for constants. Swift code follows Swift naming conventions (`camelCase` members and `PascalCase` types). Keep model loading, audio handling, output formatting, and UI concerns isolated.

## Testing Guidelines

Add focused Rust tests for parsing, resampling, timestamp formatting, and failure paths in `tests/` or module-local `#[cfg(test)]` blocks. Keep tests deterministic and avoid requiring downloaded model weights. Run `cargo test` for the full suite; use `cargo test test_name` while iterating. Build the SwiftUI target with `swift build -c release --package-path ui` when changing the UI.

## Commit & Pull Request Guidelines

There is no existing Git history to establish a project-specific convention. Use short, imperative commit subjects (for example, `Add transcript parser`) and keep unrelated changes separate. Pull requests should explain the user-visible change, identify important implementation choices, link a relevant issue when one exists, and include test results. Add screenshots or sample input/output when changing a user interface or generated artifacts.

## Security & Configuration Tips

Do not commit API keys, credentials, local recordings, or generated secrets. Keep developer-specific values in ignored environment files and provide a checked-in example configuration with placeholder values. Review new dependencies and generated files before committing.
