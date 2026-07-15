#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HELPER_SOURCE="$PACKAGE_DIR/Helpers/VTSCTileTransaction"
BUILD_TOOLS_DIR="$PACKAGE_DIR/.build-tools"
GO_VERSION="1.26.5"
GO_ARCHIVE="go${GO_VERSION}.darwin-arm64.tar.gz"
GO_SHA256="efb87ff28af9a188d0536ef5d42e63dd52ba8263cd7344a993cc48dd11dedb6a"
GO_URL="https://go.dev/dl/$GO_ARCHIVE"
TOOLCHAIN_DIR="$BUILD_TOOLS_DIR/go-$GO_VERSION-darwin-arm64"
OUTPUT_PATH="${VTSC_TILE_TRANSACTION_OUTPUT:-$BUILD_TOOLS_DIR/bin/vtsc-tile-transaction}"
RUN_TESTS=0

usage() {
  /bin/cat <<'EOF'
Usage: bash scripts/build_tile_transaction_helper.sh [--test]

Builds the static Linux ARM64 tile-transaction helper for a tici. The default
Go toolchain is pinned and SHA-256 verified. Set VTSC_GO to an explicit Go
binary to use an already installed trusted toolchain instead.

Options:
  --test      Run the helper's host Go tests before cross-compiling.
  -h, --help  Show this help.

Environment:
  VTSC_GO                         Explicit Go binary override.
  VTSC_TILE_TRANSACTION_OUTPUT    Output helper binary path.
EOF
}

while (($# > 0)); do
  case "$1" in
    --test)
      RUN_TESTS=1
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

if [[ ! -f "$HELPER_SOURCE/main.go" || ! -f "$HELPER_SOURCE/go.mod" ]]; then
  /bin/echo "Missing VTSCTileTransaction helper sources: $HELPER_SOURCE" >&2
  exit 1
fi

resolve_go() {
  if [[ -n "${VTSC_GO:-}" ]]; then
    if [[ ! -x "$VTSC_GO" ]]; then
      /bin/echo "VTSC_GO is not executable: $VTSC_GO" >&2
      exit 1
    fi
    /bin/echo "$VTSC_GO"
    return
  fi

  local go_binary="$TOOLCHAIN_DIR/bin/go"
  if [[ ! -x "$go_binary" ]]; then
    /bin/mkdir -p "$BUILD_TOOLS_DIR/downloads"
    local archive="$BUILD_TOOLS_DIR/downloads/$GO_ARCHIVE"
    if [[ ! -f "$archive" ]]; then
      local partial="$archive.partial"
      /bin/rm -f "$partial"
      /usr/bin/curl --fail --location --retry 3 --output "$partial" "$GO_URL"
      /bin/mv "$partial" "$archive"
    fi
    local actual_sha
    actual_sha="$(/usr/bin/shasum -a 256 "$archive" | /usr/bin/awk '{print $1}')"
    if [[ "$actual_sha" != "$GO_SHA256" ]]; then
      /bin/echo "Go archive checksum mismatch: got $actual_sha, want $GO_SHA256" >&2
      /bin/rm -f "$archive"
      exit 1
    fi
    local extract_dir
    extract_dir="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/vtsc-go.XXXXXX")"
    trap '/bin/rm -rf "$extract_dir"' RETURN
    /usr/bin/tar -xzf "$archive" -C "$extract_dir"
    /bin/rm -rf "$TOOLCHAIN_DIR"
    /bin/mv "$extract_dir/go" "$TOOLCHAIN_DIR"
    /bin/rm -rf "$extract_dir"
    trap - RETURN
  fi
  /bin/echo "$go_binary"
}

GO_BINARY="$(resolve_go)"
export GOTOOLCHAIN=local
export GOMODCACHE="$BUILD_TOOLS_DIR/go-mod-cache"
export GOCACHE="$BUILD_TOOLS_DIR/go-build-cache"

if ((RUN_TESTS)); then
  /bin/echo "Testing VTSCTileTransaction with $($GO_BINARY version)…"
  (cd "$HELPER_SOURCE" && CGO_ENABLED=0 "$GO_BINARY" test ./...)
fi

/bin/echo "Building static Linux ARM64 VTSCTileTransaction with $($GO_BINARY version)…"
/bin/mkdir -p "$(/usr/bin/dirname "$OUTPUT_PATH")"
(
  cd "$HELPER_SOURCE"
  GOOS=linux GOARCH=arm64 CGO_ENABLED=0 "$GO_BINARY" build -trimpath -ldflags="-s -w" -o "$OUTPUT_PATH" .
)
/bin/chmod 0755 "$OUTPUT_PATH"

FILE_INFO="$(/usr/bin/file "$OUTPUT_PATH")"
if [[ "$FILE_INFO" != *"ELF 64-bit LSB executable, ARM aarch64"* ]] || [[ "$FILE_INFO" != *"statically linked"* ]]; then
  /bin/echo "Unexpected helper binary format: $FILE_INFO" >&2
  exit 1
fi

/bin/echo "VTSC tile transaction helper ready: $OUTPUT_PATH"
