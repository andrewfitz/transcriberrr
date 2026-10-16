#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$ROOT_DIR/Cargo.toml" | head -1)"
ARCH="$(uname -m)"
DIST_DIR="$ROOT_DIR/dist"
APP_DIR="$DIST_DIR/Transcriberrr.app"
STAGING_DIR="$DIST_DIR/.dmg-staging"
DMG_PATH="$DIST_DIR/Transcriberrr-${VERSION}-macos-${ARCH}.dmg"
SHA_PATH="$DMG_PATH.sha256"

if [[ "$(uname -s)" != "Darwin" || "$ARCH" != "arm64" ]]; then
    echo "Transcriberrr releases require an Apple Silicon Mac (Darwin arm64)." >&2
    exit 1
fi
command -v hdiutil >/dev/null 2>&1 || { echo "hdiutil is required" >&2; exit 1; }

"$ROOT_DIR/scripts/build-macos-app.sh"

rm -rf "$STAGING_DIR" "$DMG_PATH" "$SHA_PATH"
mkdir -p "$STAGING_DIR"
cp -R "$APP_DIR" "$STAGING_DIR/Transcriberrr.app"
ln -s /Applications "$STAGING_DIR/Applications"

hdiutil create \
    -volname "Transcriberrr ${VERSION}" \
    -srcfolder "$STAGING_DIR" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$DMG_PATH"
shasum -a 256 "$DMG_PATH" > "$SHA_PATH"
rm -rf "$STAGING_DIR"

echo "Built $DMG_PATH"
echo "Wrote $SHA_PATH"
