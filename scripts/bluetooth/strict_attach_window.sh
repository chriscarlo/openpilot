#!/usr/bin/env bash
set -euo pipefail

# Strict attach window orchestrator for WCN3990 (UART/H4+IBS) on comma 3/3X.
# - Enforces rfkill pulse + firmware bind
# - Sends OFF/ON power pulses and starts btattach at 115200
# - Waits quietly for the window (no watchers/log tails)
# - Scrapes minimal state after the window using bounded adb calls
#
# Usage (host):
#   ADB=scripts/dev/adb_safe.sh \
#   scripts/bluetooth/strict_attach_window.sh [--tty /dev/ttyHS0] [--baud 115200] [--window 75]
#
# Behavior:
#   - Every adb call is wrapped with a timeout via $ADB
#   - Does not run adb root/kill-server during the window
#   - Does not start watchers (dmesg -w, logcat -d, etc.) during the window

TTY=/dev/ttyHS0
INIT_BAUD=115200
WINDOW_SEC=75
ADB=${ADB:-adb}
PATCH115200=${PATCH115200:-Y}
WAKE_BEFORE_RESET=${WAKE_BEFORE_RESET:-N}

usage() {
  echo "Usage: $0 [--tty DEV] [--baud BPS] [--window SEC]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tty) TTY="$2"; shift 2 ;;
    --baud) INIT_BAUD="$2"; shift 2 ;;
    --window) WINDOW_SEC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

ts() { date +%Y-%m-%dT%H:%M:%S%z; }
log() { echo "[$(ts)] $*"; }
req() { timeout 30s $ADB "$@"; }
shc() { timeout 30s $ADB shell "$@"; }

SESSION_DIR=${OUTBASE:-docs/chauffeur/bluetooth3x/diagnostics}/strict_$(date +%Y%m%d-%H%M%S)
mkdir -p "$SESSION_DIR"

log "Strict attach window: TTY=$TTY, INIT_BAUD=$INIT_BAUD, WINDOW=${WINDOW_SEC}s"

log "Pre-flight: ensure device and prep firmware bind + rfkill pulse"
req wait-for-device || true
shc 'mount --bind /data/firmware /lib/firmware || true'
shc 'rfkill block bluetooth 2>/dev/null || true; sleep 0.1; rfkill unblock bluetooth 2>/dev/null || true'
shc 'pkill -f btattach || true; rm -f /data/local/tmp/btattach.out /data/local/tmp/btattach.pid || true'
shc "echo ${PATCH115200} > /sys/module/hci_uart/parameters/patch115200 || true"
shc "test -e /sys/module/hci_uart/parameters/wake_before_reset && echo ${WAKE_BEFORE_RESET} > /sys/module/hci_uart/parameters/wake_before_reset || true"

# Optional TLV pacing/ack tunables (apply only if exported and present)
for p in tlv_seg0_len tlv_preseg0_us tlv_pace_us tlv_full_skipvse tlv_force_sync; do
  val="${!p-}"
  if [[ -n "$val" ]]; then
    shc "test -e /sys/module/btqca/parameters/${p} && echo ${val} > /sys/module/btqca/parameters/${p} || true"
  fi
done

log "Start: power pulses + btattach (bounded)"
# Use the on-device script if present; otherwise inline the pulses
if shc 'test -x /system/bin/sh' >/dev/null 2>&1; then
  # Inline pulses to avoid pushing files during window
  shc "pkill -f dmesg\ -w || true; pkill -f 'tail -F /data/log/swaglog' || true; \
       stty -F ${TTY} 2400 -echo -crtscts cs8 -parenb -cstopb 2>/dev/null || stty 2400 -echo -crtscts cs8 -parenb -cstopb < ${TTY}; \
       printf '\\xC0' > ${TTY} || true; \
       usleep 10000 2>/dev/null || sleep 0.02; \
       stty -F ${TTY} ${INIT_BAUD} -echo -crtscts cs8 -parenb -cstopb 2>/dev/null || stty ${INIT_BAUD} -echo -crtscts cs8 -parenb -cstopb < ${TTY}; \
       printf '\\xFC' > ${TTY} || true; \
       sleep 0.10; \
       stty -F ${TTY} ${INIT_BAUD} -echo crtscts cs8 -parenb -cstopb 2>/dev/null || stty ${INIT_BAUD} -echo crtscts cs8 -parenb -cstopb < ${TTY}; \
       nohup sh -c 'timeout ${WINDOW_SEC}s btattach -B ${TTY} -P qca -S ${INIT_BAUD} > /data/local/tmp/btattach.out 2>&1 & echo $! > /data/local/tmp/btattach.pid' >/dev/null 2>&1 || true"
else
  # Fallback: start btattach without pulse (unlikely on AGNOS)
  shc "nohup sh -c 'timeout ${WINDOW_SEC}s btattach -B ${TTY} -P qca -S ${INIT_BAUD} > /data/local/tmp/btattach.out 2>&1 & echo $! > /data/local/tmp/btattach.pid' >/dev/null 2>&1 || true"
fi

log "Quiet window: sleeping ${WINDOW_SEC}s (no watchers)"
sleep "$WINDOW_SEC"

log "Post-window scrape (bounded) → $SESSION_DIR"
timeout 15s $ADB shell 'tail -n 200 /data/local/tmp/btattach.out 2>/dev/null || true' > "$SESSION_DIR/btattach_out.txt" 2>&1 || true
timeout 15s $ADB shell 'dmesg | rg -n "Bluetooth|hci0|qca|ROME|baud|hci_qca|hci_uart" | tail -n 300 || dmesg | tail -n 300' > "$SESSION_DIR/dmesg_tail.txt" 2>&1 || true
timeout 10s $ADB shell 'ls -l /sys/class/bluetooth 2>/dev/null' > "$SESSION_DIR/sys_class_bluetooth.txt" 2>&1 || true
timeout 10s $ADB shell 'hciconfig -a 2>/dev/null' > "$SESSION_DIR/hciconfig.txt" 2>&1 || true
timeout 10s $ADB shell 'rfkill list 2>/dev/null' > "$SESSION_DIR/rfkill.txt" 2>&1 || true

log "Done. Artifacts in $SESSION_DIR"

exit 0
