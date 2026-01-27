#!/usr/bin/env bash
set -euo pipefail

# One-command quick run: push firmware (if provided), strict attach, parse summary.

ADB=${ADB:-scripts/dev/adb_safe.sh}
FW_DIR=${FW_DIR:-}
TTY=${TTY:-/dev/ttyHS0}
INIT_BAUD=${INIT_BAUD:-115200}
WINDOW=${WINDOW:-75}

if [[ -n "$FW_DIR" ]]; then
  echo "[quickrun] pushing firmware from $FW_DIR"
  ADB="$ADB" scripts/bluetooth/push_qca_fw.sh "$FW_DIR"
else
  echo "[quickrun] FW_DIR not set; skipping firmware push"
fi

echo "[quickrun] strict attach window"
ADB="$ADB" scripts/bluetooth/strict_attach_window.sh --tty "$TTY" --baud "$INIT_BAUD" --window "$WINDOW"

echo "[quickrun] parse bring-up summary"
$ADB shell 'sh -c "dmesg | tail -n 2000"' > .cache/adb/dmesg_tail_bt.txt 2>/dev/null || true
scripts/bluetooth/parse_bt_dmesg.sh .cache/adb/dmesg_tail_bt.txt || true

echo "[quickrun] done"

