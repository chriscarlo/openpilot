#!/usr/bin/env bash
set -euo pipefail

echo "[serdev_check] Kernel and DT quick probe" >&2

echo "[serdev_check] CONFIG (grep):" >&2
if zcat /proc/config.gz 2>/dev/null | rg -n "CONFIG_SERIAL_DEV_BUS|CONFIG_BT_HCIUART|CONFIG_BT_HCIUART_QCA|CONFIG_BT_HCIUART_SERDEV"; then
  :
else
  echo "(no /proc/config.gz; skip)" >&2
fi

echo "[serdev_check] serdev devices:" >&2
ls -l /sys/bus/serdev/devices 2>/dev/null || echo "(none)"

echo "[serdev_check] UART nodes (tty classes):" >&2
ls -l /sys/class/tty | rg -i "tty(HS|HSL|MSM|GENI)" || true

echo "[serdev_check] bluetooth class:" >&2
ls -l /sys/class/bluetooth 2>/dev/null || echo "(none)"

echo "[serdev_check] dmesg (qca/serdev excerpt):" >&2
dmesg | rg -n "qca|serdev|bluetooth|hci_qca|rome" | tail -n 300 || true

