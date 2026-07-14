#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_ROOT="$(cd "$PACKAGE_DIR/../.." && pwd)"
HELPER_SOURCE="$PACKAGE_DIR/Helpers/VTSCTileDecoder"
GENERATED_BINDING="$REPOSITORY_ROOT/mapd_repo/openpilot-mapd/offline.capnp.go"
BUILD_TOOLS_DIR="$PACKAGE_DIR/.build-tools"
GO_VERSION="1.26.5"
GO_ARCHIVE="go${GO_VERSION}.darwin-arm64.tar.gz"
GO_SHA256="efb87ff28af9a188d0536ef5d42e63dd52ba8263cd7344a993cc48dd11dedb6a"
GO_URL="https://go.dev/dl/$GO_ARCHIVE"
TOOLCHAIN_DIR="$BUILD_TOOLS_DIR/go-$GO_VERSION-darwin-arm64"
OUTPUT_PATH="${VTSC_TILE_DECODER_OUTPUT:-$BUILD_TOOLS_DIR/bin/vtsc-tile-decoder}"
RUN_TESTS=0

usage() {
  /bin/cat <<'EOF'
Usage: bash scripts/build_tile_decoder.sh [--test]

Builds the native macOS map tile decoder used by VTSC Tuner. The default
toolchain is pinned and SHA-256 verified. Set VTSC_GO to an explicit Go binary
to use an already installed trusted toolchain instead.

Options:
  --test      Run the helper's Go tests before building.
  -h, --help  Show this help.

Environment:
  VTSC_GO                  Explicit Go binary override.
  VTSC_TILE_DECODER_OUTPUT Output binary path.
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

if [[ ! -f "$GENERATED_BINDING" ]]; then
  /bin/echo "Missing generated mapd binding: $GENERATED_BINDING" >&2
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
BUILD_DIR="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/vtsc-tile-decoder.XXXXXX")"
trap '/bin/rm -rf "$BUILD_DIR"' EXIT

/bin/cp "$HELPER_SOURCE/main.go" "$BUILD_DIR/main.go"
/bin/cp "$HELPER_SOURCE/main_test.go" "$BUILD_DIR/main_test.go"
/bin/cp "$HELPER_SOURCE/go.mod" "$BUILD_DIR/go.mod"
/bin/cp "$HELPER_SOURCE/go.sum" "$BUILD_DIR/go.sum"
/bin/cp "$GENERATED_BINDING" "$BUILD_DIR/offline.capnp.go"

export GOTOOLCHAIN=local
export GOMODCACHE="$BUILD_TOOLS_DIR/go-mod-cache"
export GOCACHE="$BUILD_TOOLS_DIR/go-build-cache"

if ((RUN_TESTS)); then
  /bin/echo "Testing VTSCTileDecoder with $($GO_BINARY version)…"
  (cd "$BUILD_DIR" && "$GO_BINARY" test ./...)
fi

/bin/echo "Building VTSCTileDecoder with $($GO_BINARY version)…"
/bin/mkdir -p "$(/usr/bin/dirname "$OUTPUT_PATH")"
(cd "$BUILD_DIR" && "$GO_BINARY" build -trimpath -ldflags="-s -w" -o "$OUTPUT_PATH" .)
/bin/chmod 0755 "$OUTPUT_PATH"
/bin/echo "VTSC tile decoder ready: $OUTPUT_PATH"
