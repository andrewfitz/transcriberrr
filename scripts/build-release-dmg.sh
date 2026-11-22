#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(awk -F '"' '/^version = / { print $2; exit }' "$ROOT_DIR/Cargo.toml")"
ARCH="$(uname -m)"
DIST_DIR="$ROOT_DIR/dist"
APP_DIR="$DIST_DIR/Transcriberrr.app"
DMG_NAME="Transcriberrr-${VERSION}-macos-${ARCH}.dmg"
DMG_PATH="$DIST_DIR/$DMG_NAME"
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

# Hold the same lock used by standalone app builds until the DMG and checksum are published.
mkdir -p "$DIST_DIR"
exec 9>>"$DIST_DIR/.build.lock"
/usr/bin/lockf 9

WORK_DIR="$(mktemp -d "$DIST_DIR/.dmg-build.XXXXXX")"
STAGING_DIR="$WORK_DIR/volume"
STAGED_APP="$STAGING_DIR/Transcriberrr.app"
STAGED_DMG="$WORK_DIR/$DMG_NAME"
OUTPUTS=("$APP_DIR" "$DMG_PATH" "$SHA_PATH")
STAGED_OUTPUTS=("$STAGED_APP" "$STAGED_DMG" "$STAGED_DMG.sha256")
publishing=false
cleanup() {
    local index output backup restored=true
    if [[ "$publishing" == true ]]; then
        for index in "${!OUTPUTS[@]}"; do
            output="${OUTPUTS[$index]}"
            backup="$WORK_DIR/previous-${output##*/}"
            # A staged path disappears when it has been moved to its final destination.
            if [[ ! -e "${STAGED_OUTPUTS[$index]}" ]] && ! rm -rf "$output"; then
                restored=false
                continue
            fi
            if [[ -e "$backup" || -L "$backup" ]] && ! mv "$backup" "$output"; then
                restored=false
            fi
        done
    fi
    if [[ "$restored" == false ]]; then
        echo "Previous release files left in $WORK_DIR" >&2
        return
    fi
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$STAGING_DIR"
TRANSCRIBERRR_LOCKED_DIST="$DIST_DIR" TRANSCRIBERRR_STAGED_APP="$STAGED_APP" \
    "$ROOT_DIR/scripts/build-macos-app.sh"

# Notarize and staple the app itself, not only the disk image: once copied to /Applications the
# app no longer carries the DMG's ticket, and an offline first launch would need its own.
if [[ -n "${NOTARY_PROFILE:-}" ]]; then
    ditto -c -k --keepParent "$STAGED_APP" "$WORK_DIR/Transcriberrr.zip"
    notarize "$WORK_DIR/Transcriberrr.zip"
    xcrun stapler staple "$STAGED_APP"
fi

ln -s /Applications "$STAGING_DIR/Applications"

hdiutil create \
    -volname "Transcriberrr ${VERSION}" \
    -srcfolder "$STAGING_DIR" \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$STAGED_DMG"

# With a Developer ID identity, sign the disk image too, and notarize and staple it when
# NOTARY_PROFILE is set.
if [[ "${CODESIGN_IDENTITY:--}" != "-" ]]; then
    codesign --force --sign "$CODESIGN_IDENTITY" --timestamp "$STAGED_DMG"
    if [[ -n "${NOTARY_PROFILE:-}" ]]; then
        notarize "$STAGED_DMG"
        xcrun stapler staple "$STAGED_DMG"
    fi
fi
(
    cd "$WORK_DIR"
    shasum -a 256 "$DMG_NAME" > "$DMG_NAME.sha256"
)

# Keep the previous release until the new app, disk image and checksum are ready to publish.
publishing=true
for output in "${OUTPUTS[@]}"; do
    if [[ -e "$output" || -L "$output" ]]; then
        mv "$output" "$WORK_DIR/previous-${output##*/}"
    fi
done
for index in "${!OUTPUTS[@]}"; do
    mv "${STAGED_OUTPUTS[$index]}" "${OUTPUTS[$index]}"
done
publishing=false

echo "Built $APP_DIR (signed: ${CODESIGN_IDENTITY:--})"
echo "Built $DMG_PATH"
echo "Wrote $SHA_PATH"
