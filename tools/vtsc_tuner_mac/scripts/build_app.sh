#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INFO_PLIST="$PACKAGE_DIR/Support/Info.plist"
OUTPUT_DIR="${VTSC_APP_OUTPUT_DIR:-$PACKAGE_DIR/dist}"
APP_NAME="VTSC Tuner.app"
APP_BUNDLE="$OUTPUT_DIR/$APP_NAME"
EXECUTABLE_NAME="VTSCTuner"
TILE_DECODER_NAME="vtsc-tile-decoder"
TILE_DECODER_PATH="$PACKAGE_DIR/.build-tools/bin/$TILE_DECODER_NAME"
INSTALL_REQUESTED=0
OPEN_REQUESTED=0

usage() {
  /bin/cat <<'EOF'
Usage: bash scripts/build_app.sh [--install] [--open]

Builds the SwiftPM VTSCTuner release product and packages an ad-hoc-signed,
unsandboxed macOS application at dist/VTSC Tuner.app.

Options:
  --install  Copy the finished app to ~/Applications, or to
             VTSC_APP_INSTALL_DIR when that variable is set.
  --open     Open the packaged or installed app after verification.
  -h, --help Show this help.

Environment:
  VTSC_APP_OUTPUT_DIR  App-bundle output directory (default: <package>/dist)
  VTSC_APP_INSTALL_DIR Install destination (default: ~/Applications)
  VTSC_GO              Trusted Go binary override for the tile decoder build
EOF
}

while (($# > 0)); do
  case "$1" in
    --install)
      INSTALL_REQUESTED=1
      ;;
    --open)
      OPEN_REQUESTED=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      /bin/echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -f "$INFO_PLIST" ]]; then
  /bin/echo "Missing Info.plist template: $INFO_PLIST" >&2
  exit 1
fi

/usr/bin/plutil -lint "$INFO_PLIST" >/dev/null

/bin/echo "Building native map tile decoder…"
VTSC_TILE_DECODER_OUTPUT="$TILE_DECODER_PATH" \
  /bin/bash "$SCRIPT_DIR/build_tile_decoder.sh"

/bin/echo "Building VTSCTuner (release)…"
/usr/bin/swift build \
  --package-path "$PACKAGE_DIR" \
  --configuration release \
  --product "$EXECUTABLE_NAME"

BIN_DIR="$(
  /usr/bin/swift build \
    --package-path "$PACKAGE_DIR" \
    --configuration release \
    --show-bin-path
)"
BUILT_EXECUTABLE="$BIN_DIR/$EXECUTABLE_NAME"

if [[ ! -x "$BUILT_EXECUTABLE" ]]; then
  /bin/echo "SwiftPM did not produce an executable at: $BUILT_EXECUTABLE" >&2
  exit 1
fi

/bin/echo "Packaging ${APP_BUNDLE}…"
/bin/rm -rf "$APP_BUNDLE"
/bin/mkdir -p \
  "$APP_BUNDLE/Contents/MacOS" \
  "$APP_BUNDLE/Contents/Helpers" \
  "$APP_BUNDLE/Contents/Resources"
/usr/bin/ditto "$BUILT_EXECUTABLE" "$APP_BUNDLE/Contents/MacOS/$EXECUTABLE_NAME"
/usr/bin/ditto "$TILE_DECODER_PATH" "$APP_BUNDLE/Contents/Helpers/$TILE_DECODER_NAME"
/usr/bin/install -m 0644 "$INFO_PLIST" "$APP_BUNDLE/Contents/Info.plist"
/bin/chmod 0755 "$APP_BUNDLE/Contents/MacOS/$EXECUTABLE_NAME"
/bin/chmod 0755 "$APP_BUNDLE/Contents/Helpers/$TILE_DECODER_NAME"

# A local ad-hoc signature is sufficient for a bundle built and run on this Mac.
# No sandbox entitlements are supplied: repository and subprocess access are
# deliberate requirements of the tuner.
/usr/bin/codesign \
  --force \
  --sign - \
  --timestamp=none \
  "$APP_BUNDLE/Contents/Helpers/$TILE_DECODER_NAME"
/usr/bin/codesign \
  --force \
  --sign - \
  --timestamp=none \
  "$APP_BUNDLE"

/usr/bin/plutil -lint "$APP_BUNDLE/Contents/Info.plist" >/dev/null
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"

APP_TO_OPEN="$APP_BUNDLE"
if ((INSTALL_REQUESTED)); then
  INSTALL_DIR="${VTSC_APP_INSTALL_DIR:-$HOME/Applications}"
  INSTALLED_APP="$INSTALL_DIR/$APP_NAME"
  /bin/echo "Installing ${INSTALLED_APP}…"
  /bin/mkdir -p "$INSTALL_DIR"
  /bin/rm -rf "$INSTALLED_APP"
  /usr/bin/ditto "$APP_BUNDLE" "$INSTALLED_APP"
  /usr/bin/codesign --verify --deep --strict --verbose=2 "$INSTALLED_APP"
  APP_TO_OPEN="$INSTALLED_APP"
fi

/bin/echo "VTSC Tuner app ready: $APP_TO_OPEN"

if ((OPEN_REQUESTED)); then
  /usr/bin/open "$APP_TO_OPEN"
fi
