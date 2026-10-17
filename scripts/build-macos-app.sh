#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UI_DIR="$ROOT_DIR/ui"
APP_DIR="$ROOT_DIR/dist/Transcriberrr.app"
VERSION="$(awk -F '"' '/^version = / { print $2; exit }' "$ROOT_DIR/Cargo.toml")"

if [[ -z "$VERSION" ]]; then
    echo "Unable to determine the package version from Cargo.toml" >&2
    exit 1
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "Transcriberrr requires an Apple Silicon Mac (Darwin arm64)." >&2
    exit 1
fi

command -v swift >/dev/null 2>&1 || { echo "swift is required to build the app" >&2; exit 1; }
if command -v cargo >/dev/null 2>&1; then
    CARGO_CMD=(cargo)
elif command -v rustup >/dev/null 2>&1; then
    # rustup installations commonly expose cargo only through a toolchain.
    TOOLCHAIN_BIN="$(dirname "$(rustup which cargo)")"
    export PATH="$TOOLCHAIN_BIN:$PATH"
    CARGO_CMD=("$TOOLCHAIN_BIN/cargo")
else
    echo "cargo (or rustup with a stable toolchain) is required to build the bundled CLI" >&2
    exit 1
fi

# Build both executables before assembling the app. The CLI lives in Resources so
# a Finder-launched app never depends on the user's interactive shell PATH.
"${CARGO_CMD[@]}" build --release --features diarization --manifest-path "$ROOT_DIR/Cargo.toml"
swift build -c release --package-path "$UI_DIR" --product transcriberrr-ui
swift build -c release --package-path "$UI_DIR" --product transcriberrr-native
UI_BIN="$(swift build -c release --package-path "$UI_DIR" --show-bin-path)/transcriberrr-ui"
NATIVE_BIN="$(swift build -c release --package-path "$UI_DIR" --show-bin-path)/transcriberrr-native"
CLI_BIN="$ROOT_DIR/target/release/transcriberrr"
[[ -x "$UI_BIN" ]] || { echo "Swift UI binary not found: $UI_BIN" >&2; exit 1; }
[[ -x "$NATIVE_BIN" ]] || { echo "Native Speech helper not found: $NATIVE_BIN" >&2; exit 1; }
[[ -x "$CLI_BIN" ]] || { echo "Rust CLI binary not found: $CLI_BIN" >&2; exit 1; }

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "$UI_BIN" "$APP_DIR/Contents/MacOS/Transcriberrr"
cp "$CLI_BIN" "$APP_DIR/Contents/Resources/transcriberrr"
cp "$NATIVE_BIN" "$APP_DIR/Contents/Resources/transcriberrr-native"
chmod 755 "$APP_DIR/Contents/MacOS/Transcriberrr" "$APP_DIR/Contents/Resources/transcriberrr" "$APP_DIR/Contents/Resources/transcriberrr-native"
cp "$UI_DIR/Info.plist" "$APP_DIR/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$APP_DIR/Contents/Info.plist"
echo "Built $APP_DIR"
