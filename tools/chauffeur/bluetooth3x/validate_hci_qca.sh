#!/usr/bin/env bash
set -euo pipefail

echo "[validate_hci_qca] rfkill state:" >&2
rfkill list 2>/dev/null || true

echo "[validate_hci_qca] dmesg (power/serial/hci_qca):" >&2
dmesg | grep -i -E "bt_power|hci_qca|serdev|msm|geni|qup" | tail -n 200 || true

echo "[validate_hci_qca] TTYs:" >&2
ls -l /sys/class/tty | grep -i -E "tty(MSM|GENI|HS|HSL)" || true

echo "[validate_hci_qca] serdev devices:" >&2
ls -l /sys/bus/serdev/devices 2>/dev/null || echo "(no serdev)"

echo "[validate_hci_qca] bluetoothctl show:" >&2
bluetoothctl show || true

echo "[validate_hci_qca] hciconfig (before):" >&2
hciconfig -a || true

echo "[validate_hci_qca] Trying to power up hci0..." >&2
if hciconfig hci0 up; then
  echo "[validate_hci_qca] hci0 up OK" >&2
else
  echo "[validate_hci_qca] hci0 up failed (will show recent dmesg)" >&2
  dmesg | tail -n 120
fi

echo "[validate_hci_qca] hciconfig (after):" >&2
hciconfig -a || true

echo "[validate_hci_qca] Done." >&2

