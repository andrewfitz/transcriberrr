#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(awk -F '"' '/^version = / { print $2; exit }' "$ROOT_DIR/Cargo.toml")"
ARCH="$(uname -m)"
DIST_DIR="$ROOT_DIR/dist"
APP_DIR="$DIST_DIR/Transcriberrr.app"
STAGING_DIR="$DIST_DIR/.dmg-staging"
DMG_PATH="$DIST_DIR/Transcriberrr-${VERSION}-macos-${ARCH}.dmg"
SHA_PATH="$DMG_PATH.sha256"

if [[ -z "$VERSION" ]]; then
    echo "Unable to determine the package version from Cargo.toml" >&2
    exit 1
fi

if [[ "$(uname -s)" != "Darwin" || "$ARCH" != "arm64" ]]; then
    echo "Transcriberrr releases require an Apple Silicon Mac (Darwin arm64)." >&2
    exit 1
fi
command -v hdiutil >/dev/null 2>&1 || { echo "hdiutil is required" >&2; exit 1; }
command -v shasum >/dev/null 2>&1 || { echo "shasum is required" >&2; exit 1; }
if [[ -n "${NOTARY_PROFILE:-}" && "${CODESIGN_IDENTITY:--}" == "-" ]]; then
    echo "NOTARY_PROFILE requires CODESIGN_IDENTITY to be a Developer ID identity" >&2
    exit 1
fi

# Submit to Apple's notary service (NOTARY_PROFILE is a keychain profile created with
# `xcrun notarytool store-credentials`) and stop with the notary log unless the submission is
# "Accepted": `notarytool submit --wait` exits 0 even when the result is "Invalid".
notarize() {
    local result id status
    result="$(xcrun notarytool submit "$1" --keychain-profile "$NOTARY_PROFILE" --wait --output-format json)"
    id="$(printf '%s' "$result" | plutil -extract id raw -)"
    status="$(printf '%s' "$result" | plutil -extract status raw -)"
    if [[ "$status" != "Accepted" ]]; then
        echo "Notarization of $1 finished with status: $status" >&2
        xcrun notarytool log "$id" --keychain-profile "$NOTARY_PROFILE" >&2 || true
        exit 1
    fi
}

"$ROOT_DIR/scripts/build-macos-app.sh"

rm -rf "$STAGING_DIR" "$DMG_PATH" "$SHA_PATH"
trap 'rm -rf "$STAGING_DIR"' EXIT
mkdir -p "$STAGING_DIR"

# Notarize and staple the app itself, not only the disk image: once copied to /Applications the
# app no longer carries the DMG's ticket, and an offline first launch would need its own.
if [[ -n "${NOTARY_PROFILE:-}" ]]; then
    ditto -c -k --keepParent "$APP_DIR" "$STAGING_DIR/Transcriberrr.zip"
    notarize "$STAGING_DIR/Transcriberrr.zip"
    rm "$STAGING_DIR/Transcriberrr.zip"
    xcrun stapler staple "$APP_DIR"
fi

# ditto preserves the signature and stapled ticket exactly.
ditto "$APP_DIR" "$STAGING_DIR/Transcriberrr.app"
ln -s /Applications "$STAGING_DIR/Applications"

hdiutil create \
    -volname "Transcriberrr ${VERSION}" \
    -srcfolder "$STAGING_DIR" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$DMG_PATH"

# With a Developer ID identity, sign the disk image too, and notarize and staple it when
# NOTARY_PROFILE is set.
if [[ "${CODESIGN_IDENTITY:--}" != "-" ]]; then
    codesign --force --sign "$CODESIGN_IDENTITY" --timestamp "$DMG_PATH"
    if [[ -n "${NOTARY_PROFILE:-}" ]]; then
        notarize "$DMG_PATH"
        xcrun stapler staple "$DMG_PATH"
    fi
fi
(
    cd "$DIST_DIR"
    shasum -a 256 "$(basename "$DMG_PATH")" > "$(basename "$SHA_PATH")"
)

echo "Built $DMG_PATH"
echo "Wrote $SHA_PATH"
