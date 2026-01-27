#!/usr/bin/env bash
set -euo pipefail

ADB_BIN=${ADB_BIN:-adb}
OUT_BASE=${OUT_BASE:-docs/chauffeur/bluetooth3x/bootloops}
mkdir -p "$OUT_BASE"

log() { echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] $*"; }

pull_safe() {
  local src="$1" dst="$2"
  $ADB_BIN pull "$src" "$dst" >/dev/null 2>&1 || true
}

cmd_out() {
  local cmd="$1" dst="$2"
  $ADB_BIN shell "$cmd" > "$dst" 2>&1 || true
}

while true; do
  log "Waiting for device attach…"
  $ADB_BIN wait-for-device || true
  # best-effort root
  $ADB_BIN root >/dev/null 2>&1 || true
  sleep 1
  stamp=$(date +%Y%m%d-%H%M%S)
  outdir="$OUT_BASE/BL_$stamp"
  mkdir -p "$outdir"
  log "Attached; collecting snapshot into $outdir"

  # System context
  cmd_out 'date; uptime; id; getprop ro.build.fingerprint; getprop ro.boot.serialno; cat /proc/uptime; cat /proc/loadavg; df -h' "$outdir/sys_context.txt"
  cmd_out 'cat /proc/cmdline' "$outdir/proc_cmdline.txt"

  # Kernel logs
  cmd_out 'dmesg -T | tail -n 4000' "$outdir/dmesg_tail.txt"

  # pstore / last_kmsg
  cmd_out 'ls -l /sys/fs/pstore 2>/dev/null' "$outdir/pstore_ls.txt"
  cmd_out 'for f in /sys/fs/pstore/* 2>/dev/null; do echo ==== $f ====; head -c 524288 "$f"; echo; done' "$outdir/pstore_dump.txt"
  cmd_out 'cat /proc/last_kmsg 2>/dev/null | tail -n 2000' "$outdir/last_kmsg_tail.txt"

  # BlueZ/BT quick state
  cmd_out 'ls -l /sys/class/bluetooth 2>/dev/null' "$outdir/sys_class_bluetooth.txt"
  cmd_out 'rfkill list 2>/dev/null' "$outdir/rfkill.txt"
  cmd_out 'hciconfig -a 2>/dev/null' "$outdir/hciconfig.txt"

  # Comma logs
  cmd_out 'ls -lt /data/log 2>/dev/null | head -n 200' "$outdir/data_log_ls.txt"
  # Pull recent small logs first; then full dir as last step (may be large)
  pull_safe '/data/log' "$outdir/data_log" &

  # Swaglog tail and manager/launcher status
  cmd_out 'tail -n 120 /data/log/swaglog.* 2>/dev/null' "$outdir/swag_tail.txt"
  cmd_out 'ps -A -o PID,PPID,STAT,PCPU,PMEM,COMM --sort=-PCPU | head -n 60' "$outdir/ps_top.txt"
  cmd_out 'cat /proc/meminfo | head -n 30' "$outdir/meminfo.txt"

  # comma user tmux capture if present
  cmd_out 'su - comma -c "tmux ls 2>/dev/null || true"' "$outdir/tmux_ls.txt"
  cmd_out 'su - comma -c "tmux capture-pane -p -S -5000 -t comma-bt:0 2>/dev/null || true"' "$outdir/tmux_comma_bt_capture.txt"

  # Android logs (if logcat present)
  cmd_out 'type logcat >/dev/null 2>&1 && logcat -b all -d -v time | tail -n 5000 || true' "$outdir/logcat_tail.txt"

  # Sync and brief pause to avoid races
  cmd_out 'sync' "$outdir/sync.txt"
  log "Snapshot complete: $outdir"
  # Small delay before waiting for next attach
  sleep 3
done

