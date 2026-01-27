#!/usr/bin/env bash
set -euo pipefail

# Kill known noisy watchers that can interfere with strict attach timing.

killp() {
  local pat="$1"
  pids=$(pgrep -f "$pat" || true)
  if [[ -n "$pids" ]]; then
    echo "[kill_watchers] killing $pat: $pids" >&2
    kill -9 $pids 2>/dev/null || true
  fi
}

killp "dmesg -w"
killp "tail -F /data/log/swaglog"
killp "btmon -w"

echo "[kill_watchers] done" >&2

