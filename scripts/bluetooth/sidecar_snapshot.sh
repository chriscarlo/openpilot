#!/usr/bin/env bash
set -euo pipefail

# Sidecar snapshotter: while device is present, every SNAP_SECS seconds
# capture a compact system snapshot to RUN_DIR. Each adb call ≤30s.

ADB=${ADB:-adb}
SNAP_SECS=${SNAP_SECS:-10}
TOTAL_SECS=${TOTAL_SECS:-480}
RUN_DIR=${RUN_DIR:?RUN_DIR not set}

ts() { date +%Y-%m-%dT%H:%M:%S%z; }
req() { timeout 30s $ADB "$@"; }
shc() { timeout 30s $ADB shell "$@"; }

start=$(date +%s)
while :; do
  now=$(date +%s)
  (( now - start >= TOTAL_SECS )) && break
  if req get-state 2>/dev/null | grep -q '^device$'; then
    stamp=$(date +%H%M%S)
    shc 'date' > "$RUN_DIR/snap_${stamp}_date.txt" 2>&1 || true
    shc 'uptime' > "$RUN_DIR/snap_${stamp}_uptime.txt" 2>&1 || true
    shc 'cat /proc/meminfo | head -n 20' > "$RUN_DIR/snap_${stamp}_meminfo.txt" 2>&1 || true
    shc 'ps -A -o PID,STAT,PCPU,PMEM,COMM --sort=-PCPU | head -n 30' > "$RUN_DIR/snap_${stamp}_ps.txt" 2>&1 || true
    shc 'tail -n 80 /data/log/swaglog.* 2>/dev/null' > "$RUN_DIR/snap_${stamp}_swag_tail.txt" 2>&1 || true
    shc 'dmesg | tail -n 120' > "$RUN_DIR/snap_${stamp}_dmesg_tail.txt" 2>&1 || true
  fi
  sleep "$SNAP_SECS" || true
done

exit 0

