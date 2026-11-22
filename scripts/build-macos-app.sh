#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UI_DIR="$ROOT_DIR/ui"
DIST_DIR="$ROOT_DIR/dist"
APP_DIR="$DIST_DIR/Transcriberrr.app"
VERSION="$(awk -F '"' '/^version = / { print $2; exit }' "$ROOT_DIR/Cargo.toml")"
cd "$ROOT_DIR"

if [[ -z "$VERSION" ]]; then
    echo "Unable to determine the package version from Cargo.toml" >&2
    exit 1
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "Transcriberrr requires an Apple Silicon Mac (Darwin arm64)." >&2
    exit 1
fi

# The Speech helper reports its own version when run standalone; keep it in step with Cargo.
NATIVE_VERSION="$(awk -F '"' '/^private let transcriberrrVersion = / { print $2; exit }' "$UI_DIR/Sources/TranscriberrrNative/main.swift")"
if [[ "$NATIVE_VERSION" != "$VERSION" ]]; then
    echo "Version mismatch: Cargo.toml is $VERSION but TranscriberrrNative reports $NATIVE_VERSION" >&2
    exit 1
fi

# Ad-hoc signing ("-") by default. Use a Developer ID Application identity for notarization.
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"

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

# A release build supplies its staging path and inherits the distribution directory lock.
mkdir -p "$DIST_DIR"
if [[ "${TRANSCRIBERRR_LOCKED_DIST:-}" == "$DIST_DIR" && -e /dev/fd/9 ]]; then
    APP_DIR="${TRANSCRIBERRR_STAGED_APP:-$APP_DIR}"
else
    exec 9>>"$DIST_DIR/.build.lock"
fi
/usr/bin/lockf 9

# Build both executables before assembling the app. The CLI lives in Contents/Helpers so
# a Finder-launched app never depends on the user's interactive shell PATH.
if ! CLI_BIN="$("${CARGO_CMD[@]}" build --locked --release --features diarization \
    --message-format=json-render-diagnostics |
    awk '/"reason":"compiler-artifact"/ && /"kind":\["bin"\]/ && /"name":"transcriberrr"/ { print }' |
    plutil -extract executable raw -)"; then
    echo "Cargo did not report a release executable${CLI_BIN:+: $CLI_BIN}" >&2
    exit 1
fi
swift build -c release --package-path "$UI_DIR"
SWIFT_BIN_PATH="$(swift build -c release --package-path "$UI_DIR" --show-bin-path)"
UI_BIN="$SWIFT_BIN_PATH/transcriberrr-ui"
NATIVE_BIN="$SWIFT_BIN_PATH/transcriberrr-native"
[[ -x "$UI_BIN" ]] || { echo "Swift UI binary not found: $UI_BIN" >&2; exit 1; }
[[ -x "$NATIVE_BIN" ]] || { echo "Native Speech helper not found: $NATIVE_BIN" >&2; exit 1; }
[[ -x "$CLI_BIN" ]] || { echo "Rust CLI binary not found: $CLI_BIN" >&2; exit 1; }
for binary in "$UI_BIN" "$NATIVE_BIN" "$CLI_BIN"; do
    lipo -verify_arch arm64 "$binary"
done

WORK_DIR="$(mktemp -d "$DIST_DIR/.app-build.XXXXXX")"
STAGED_APP="$WORK_DIR/Transcriberrr.app"
PREVIOUS_APP="$WORK_DIR/previous.app"
cleanup() {
    if [[ ( -e "$PREVIOUS_APP" || -L "$PREVIOUS_APP" ) && ! -e "$APP_DIR" && ! -L "$APP_DIR" ]] &&
        ! mv "$PREVIOUS_APP" "$APP_DIR"; then
        echo "Previous app left at $PREVIOUS_APP" >&2
        return
    fi
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ICON_SET="$WORK_DIR/AppIcon.iconset"
mkdir "$ICON_SET"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$UI_DIR/Resources/AppIcon.png" \
        --out "$ICON_SET/icon_${size}x${size}.png" >/dev/null
    retina=$((size * 2))
    sips -z "$retina" "$retina" "$UI_DIR/Resources/AppIcon.png" \
        --out "$ICON_SET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil --convert icns --output "$WORK_DIR/AppIcon.icns" "$ICON_SET"

mkdir -p "$STAGED_APP/Contents/MacOS" "$STAGED_APP/Contents/Helpers" "$STAGED_APP/Contents/Resources"
install -m 755 "$UI_BIN" "$STAGED_APP/Contents/MacOS/Transcriberrr"
install -m 755 "$CLI_BIN" "$STAGED_APP/Contents/Helpers/transcriberrr"
install -m 755 "$NATIVE_BIN" "$STAGED_APP/Contents/Helpers/transcriberrr-native"
cp "$WORK_DIR/AppIcon.icns" "$STAGED_APP/Contents/Resources/AppIcon.icns"
cp "$UI_DIR/Info.plist" "$STAGED_APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$STAGED_APP/Contents/Info.plist"

# Sign nested executables before the bundle. Editing Info.plist after the linker's ad-hoc
# signature leaves an invalid bundle seal, which Gatekeeper reports as "damaged".
SIGN_ARGS=(--force --sign "$CODESIGN_IDENTITY")
if [[ "$CODESIGN_IDENTITY" != "-" ]]; then
    # Notarization requires the hardened runtime and a secure timestamp.
    SIGN_ARGS+=(--options runtime --timestamp)
fi
codesign "${SIGN_ARGS[@]}" "$STAGED_APP/Contents/Helpers/transcriberrr"
codesign "${SIGN_ARGS[@]}" "$STAGED_APP/Contents/Helpers/transcriberrr-native"
codesign "${SIGN_ARGS[@]}" "$STAGED_APP"
codesign --verify --strict --deep "$STAGED_APP"
if [[ -e "$APP_DIR" || -L "$APP_DIR" ]]; then
    mv "$APP_DIR" "$PREVIOUS_APP"
fi
mv "$STAGED_APP" "$APP_DIR"
if [[ "$APP_DIR" == "$DIST_DIR/Transcriberrr.app" ]]; then
    echo "Built $APP_DIR (signed: $CODESIGN_IDENTITY)"
fi
