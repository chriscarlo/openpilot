#!/usr/bin/env bash
set -euo pipefail

# Bounded ADB connection monitor and snapshotter.
# - Loops for a fixed duration (default: 300s)
# - Detects device attach events
# - On each attach, collects a fast snapshot of kernel/user logs and key files
# - Every adb call is individually wrapped by `timeout` (≤30s) to avoid hangs

DUR_SECS=${1:-300}
SNAP_BASE=${SNAP_BASE:-docs/chauffeur/bluetooth3x/diagnostics}
ADB=${ADB:-adb}

mkdir -p "$SNAP_BASE"

ts() { date +%Y-%m-%dT%H:%M:%S%z; }

log() { echo "[$(ts)] $*"; }

state_device() {
  timeout 3s $ADB get-state 2>/dev/null | grep -q '^device$'
}

snapshot() {
  local snap_dir="$1"
  mkdir -p "$snap_dir"
  log "Snapshot → $snap_dir"

  # Quick facts
  timeout 5s $ADB shell 'date' > "$snap_dir/date.txt" 2>&1 || true
  timeout 5s $ADB shell 'cat /proc/uptime' > "$snap_dir/uptime.txt" 2>&1 || true
  timeout 5s $ADB shell 'uname -a' > "$snap_dir/uname.txt" 2>&1 || true
  timeout 5s $ADB shell 'getprop 2>/dev/null | head -n 200' > "$snap_dir/getprop_head.txt" 2>&1 || true
  timeout 5s $ADB shell 'logcat -b events -d -v threadtime 2>/dev/null | tail -n 500' > "$snap_dir/logcat_events_tail.txt" 2>&1 || true

  # Kernel logs (tail only to be fast)
  timeout 5s $ADB shell 'dmesg | tail -n 1200' > "$snap_dir/dmesg_tail.txt" 2>&1 || true

  # Pstore / last-kmsg if present
  timeout 5s $ADB shell 'ls -l /sys/fs/pstore 2>/dev/null' > "$snap_dir/pstore_ls.txt" 2>&1 || true
  timeout 7s $ADB shell 'for f in /sys/fs/pstore/* 2>/dev/null; do echo ==== $f ====; head -c 131072 "$f"; echo; done' > "$snap_dir/pstore_dump.txt" 2>&1 || true
  timeout 5s $ADB shell 'cat /proc/last_kmsg 2>/dev/null | tail -n 1200' > "$snap_dir/last_kmsg_tail.txt" 2>&1 || true

  # Bluetooth quick state (best effort)
  timeout 3s $ADB shell 'ls -l /sys/class/bluetooth 2>/dev/null' > "$snap_dir/sys_class_bluetooth.txt" 2>&1 || true
  timeout 3s $ADB shell 'hciconfig -a 2>/dev/null' > "$snap_dir/hciconfig.txt" 2>&1 || true
  timeout 3s $ADB shell 'rfkill list 2>/dev/null' > "$snap_dir/rfkill.txt" 2>&1 || true

  # Firmware presence
  timeout 3s $ADB shell 'ls -l /data/firmware/qca 2>/dev/null || ls -l /lib/firmware/qca 2>/dev/null' > "$snap_dir/fw_ls.txt" 2>&1 || true

  # Process + mounts snapshot (head)
  timeout 5s $ADB shell 'ps -A 2>/dev/null | head -n 200' > "$snap_dir/ps_head.txt" 2>&1 || true
  timeout 5s $ADB shell 'cat /proc/mounts 2>/dev/null | head -n 200' > "$snap_dir/mounts_head.txt" 2>&1 || true
}

main() {
  local start_ts=$(date +%s)
  local last_state="unknown"
  log "Starting ADB bounded monitor for ${DUR_SECS}s"
  while :; do
    local now=$(date +%s)
    if (( now - start_ts >= DUR_SECS )); then
      log "Monitor duration reached (${DUR_SECS}s). Exiting."
      break
    fi

    if state_device; then
      if [[ "$last_state" != device ]]; then
        last_state=device
        local snap_ts=$(date +%Y%m%d-%H%M%S)
        local snap_dir="$SNAP_BASE/snap_${snap_ts}"
        snapshot "$snap_dir"
      fi
    else
      last_state=offline
    fi

    sleep 1
  done
}

main "$@"
