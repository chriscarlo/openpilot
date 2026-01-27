#!/usr/bin/env bash
set -euo pipefail

DUR=${1:-30}
OUT=${2:-/data/local/tmp/btmon.cap}

echo "[btmon] capturing for ${DUR}s to ${OUT}"
timeout "${DUR}s" btmon -w "$OUT" || true
echo "[btmon] done"

