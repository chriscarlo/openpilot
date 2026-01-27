#!/usr/bin/env bash
set -euo pipefail

# Attach-and-capture runner for comma3x (WCN3990 over UART)
# - Caps every adb call with `timeout` (≤30s) but loops internally
# - Performs a single attach attempt within a controlled window
# - Captures pre/post dmesg slices, btattach stdout, sysfs bluetooth, rfkill, hciconfig

ADB=${ADB:-adb}
TTY=${TTY:-/dev/ttyHS0}
SPEED=${SPEED:-115200}
PATCH115200=${PATCH115200:-Y}   # Y keeps TLV/NVM at 115200
DUR=${DUR:-90}                   # total session duration in seconds
OUTBASE=${OUTBASE:-docs/chauffeur/bluetooth3x/diagnostics}

ts() { date +%Y-%m-%dT%H:%M:%S%z; }
log() { echo "[$(ts)] $*"; }

die() { echo "ERR: $*" >&2; exit 1; }

mkdir -p "$OUTBASE"
SESSION_DIR="$OUTBASE/session_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SESSION_DIR"

cap() { local cmd="$1" out="$2"; timeout 30s $ADB shell "$cmd" > "$SESSION_DIR/$out" 2>&1 || true; }

quick_pre() {
  log "Pre-capture"
  cap 'uname -a' pre_uname.txt
  cap 'date' pre_date.txt
  cap 'cat /proc/uptime' pre_uptime.txt
  cap 'dmesg | tail -n 500' pre_dmesg_tail.txt
  cap 'ls -l /data/firmware/qca 2>/dev/null || ls -l /lib/firmware/qca 2>/dev/null' pre_fw_ls.txt
}

quick_post() {
  log "Post-capture"
  cap 'dmesg | tail -n 800' post_dmesg_tail.txt
  cap 'ls -l /sys/class/bluetooth 2>/dev/null' post_sys_class_bluetooth.txt
  cap 'hciconfig -a 2>/dev/null' post_hciconfig.txt
  cap 'rfkill list 2>/dev/null' post_rfkill.txt
}

run_attach() {
  log "Setting hci_uart.patch115200=$PATCH115200"
  timeout 30s $ADB shell "echo $PATCH115200 > /sys/module/hci_uart/parameters/patch115200" || true

  log "Mounting firmware bind and prepping"
  timeout 30s $ADB shell 'mount --bind /data/firmware /lib/firmware || true' || true
  timeout 30s $ADB shell 'pkill -f btattach || true' || true
  timeout 30s $ADB shell 'rm -f /data/local/tmp/btattach.out /data/local/tmp/btattach.pid || true' || true

  log "Starting bounded btattach ($TTY @ $SPEED)"
  timeout 30s $ADB shell "timeout 15s btattach -B $TTY -P qca -S $SPEED > /data/local/tmp/btattach.out 2>&1 || true" || true
  timeout 30s $ADB shell 'tail -n 200 /data/local/tmp/btattach.out 2>/dev/null || true' > "$SESSION_DIR/btattach_out.txt" || true

  # Poll dmesg slices 3x spaced ~3s, each ≤30s
  for i in 1 2 3; do
    cap 'dmesg | rg -n "Bluetooth|hci0|qca|ROME|baud|hci_qca|hci_uart" | tail -n 200 || dmesg | tail -n 200' "dmesg_step_${i}.txt"
    sleep 2
  done
}

main() {
  local start=$(date +%s)
  log "Attach-and-capture start (DUR=${DUR}s, TTY=${TTY}, PATCH115200=${PATCH115200})"

  # Wait up to DUR seconds, but every adb call is bounded
  while :; do
    local now=$(date +%s)
    (( now - start > DUR )) && { log "Session timeout"; break; }
    if timeout 5s $ADB get-state 2>/dev/null | grep -q '^device$'; then
      quick_pre
      run_attach
      quick_post
      break
    fi
    sleep 1
  done

  log "Artifacts in $SESSION_DIR"
}

main "$@"

