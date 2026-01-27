#!/usr/bin/env bash
set -euo pipefail

# 10-minute (default) conservative session cycle:
# - Ensures on-device watchers are running
# - Waits for device attach (adb get-state)
# - After ATTACH_OFFSET seconds, performs one bounded attach
# - At the end pulls watcher logs, dmesg tails, pstore listing
# - Each adb call is wrapped in timeout ≤30s

TOTAL_SECS=${TOTAL_SECS:-600}
ATTACH_OFFSET=${ATTACH_OFFSET:-60}
ADB=${ADB:-adb}
OUTBASE=${OUTBASE:-docs/chauffeur/bluetooth3x/diagnostics}
PATCH115200=${PATCH115200:-Y}
TTY=${TTY:-/dev/ttyHS0}
SPEED=${SPEED:-115200}

ts() { date +%Y-%m-%dT%H:%M:%S%z; }
log() { echo "[$(ts)] $*"; }

RUN_DIR="$OUTBASE/run_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

req() { timeout 30s $ADB "$@"; }
shc() { timeout 30s $ADB shell "$@"; }

log "Waiting for device..."
req wait-for-device || true

log "Starting on-device watchers"
req push scripts/bluetooth/on_device_watchers.sh /data/local/tmp/on_device_watchers.sh >/dev/null 2>&1 || true
shc 'sh /data/local/tmp/on_device_watchers.sh || /system/bin/sh /data/local/tmp/on_device_watchers.sh || true' >/dev/null 2>&1 || true

log "Session start (TOTAL_SECS=${TOTAL_SECS}, ATTACH_OFFSET=${ATTACH_OFFSET})"
start=$(date +%s)
attached_once=false

while :; do
  now=$(date +%s)
  elapsed=$(( now - start ))
  if (( elapsed >= TOTAL_SECS )); then
    log "Session reached ${TOTAL_SECS}s; collecting and exiting"
    break
  fi

  # After offset, perform exactly one bounded attach if not done
  if ! $attached_once && (( elapsed >= ATTACH_OFFSET )); then
    log "Performing bounded attach (pulses + btattach; patch115200=${PATCH115200}, ${TTY}@${SPEED})"
    shc "mount --bind /data/firmware /lib/firmware || true; echo ${PATCH115200} > /sys/module/hci_uart/parameters/patch115200 || true; pkill -f btattach || true; sh -c 'stty -F ${TTY} 2400 -echo -crtscts cs8 -parenb -cstopb 2>/dev/null || stty 2400 -echo -crtscts cs8 -parenb -cstopb < ${TTY}; printf \\\xC0 > ${TTY} || true; usleep 10000 2>/dev/null || sleep 0.02; stty -F ${TTY} ${SPEED} -echo -crtscts cs8 -parenb -cstopb 2>/dev/null || stty ${SPEED} -echo -crtscts cs8 -parenb -cstopb < ${TTY}; printf \\\xFC > ${TTY} || true; sleep 0.10; stty -F ${TTY} ${SPEED} -echo crtscts cs8 -parenb -cstopb 2>/dev/null || stty ${SPEED} -echo crtscts cs8 -parenb -cstopb < ${TTY}; timeout 20s btattach -B ${TTY} -P qca -S ${SPEED} > /data/local/tmp/btattach.out 2>&1 || true'" || true
    attached_once=true
  fi

  sleep 1
done

log "Pulling logs to $RUN_DIR"
# Ensure device is present for pulls (best-effort, up to ~2 min)
tries=40
while ! timeout 5s $ADB get-state 2>/dev/null | grep -q '^device$'; do
  tries=$((tries-1)) || true
  if [ "$tries" -le 0 ]; then
    log "Device not present for pulls; saving placeholders"
    break
  fi
  sleep 3
done
# btattach out
shc 'tail -n 200 /data/local/tmp/btattach.out 2>/dev/null' > "$RUN_DIR/btattach_out.txt" 2>&1 || true
# dmesg tails
shc 'dmesg | tail -n 600' > "$RUN_DIR/dmesg_tail.txt" 2>&1 || true
# watcher tails
shc 'tail -n 300 /data/local/tmp/mon/dmesg_w.txt 2>/dev/null' > "$RUN_DIR/dmesg_w_tail.txt" 2>&1 || true
shc 'tail -n 200 /data/local/tmp/mon/swag_tail.txt 2>/dev/null' > "$RUN_DIR/swag_tail.txt" 2>&1 || true
# pstore listing
shc 'ls -l /sys/fs/pstore 2>/dev/null' > "$RUN_DIR/pstore_ls.txt" 2>&1 || true
# bluetooth sysfs + rfkill + hciconfig
shc 'ls -l /sys/class/bluetooth 2>/dev/null' > "$RUN_DIR/sys_class_bluetooth.txt" 2>&1 || true
shc 'rfkill list 2>/dev/null' > "$RUN_DIR/rfkill.txt" 2>&1 || true
shc 'hciconfig -a 2>/dev/null' > "$RUN_DIR/hciconfig.txt" 2>&1 || true

log "Done. Artifacts in $RUN_DIR"

exit 0
