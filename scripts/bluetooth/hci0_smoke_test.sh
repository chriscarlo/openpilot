#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] rfkill:" >&2
rfkill list 2>/dev/null || true
rfkill unblock all 2>/dev/null || true

echo "[smoke] hciconfig (before):" >&2
hciconfig -a || true

if hciconfig hci0 up; then
  echo "[smoke] hci0 up OK" >&2
else
  echo "[smoke] hci0 up failed" >&2
fi

echo "[smoke] Start short LE scan (8s)" >&2
timeout 8s hcitool lescan 2>/dev/null | head -n 20 || true

echo "[smoke] hciconfig (after):" >&2
hciconfig -a || true

