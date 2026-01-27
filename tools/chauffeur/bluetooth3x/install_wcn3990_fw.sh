#!/usr/bin/env bash
set -euo pipefail

# Installs WCN3990 BT firmware into a writable firmware search path on device.
# Usage: adb push this script to /data and run as root on the device, or run via SSH.

FW_DST_BASE="/data/firmware"
FW_DST_QCA="${FW_DST_BASE}/qca"
KREL="$(uname -r 2>/dev/null || true)"

echo "[wcn3990-fw] Preparing firmware path ${FW_DST_QCA}"
mkdir -p "${FW_DST_QCA}"

have21=0
if [ -f /lib/firmware/qca/crbtfw21.tlv.zst ]; then
  echo "[wcn3990-fw] Decompressing crbtfw21.tlv.zst -> ${FW_DST_QCA}/crbtfw21.tlv"
  zstd -d -f -c /lib/firmware/qca/crbtfw21.tlv.zst > "${FW_DST_QCA}/crbtfw21.tlv"
  have21=1
fi
if [ -f /lib/firmware/qca/crnv21.bin.zst ]; then
  echo "[wcn3990-fw] Decompressing crnv21.bin.zst -> ${FW_DST_QCA}/crnv21.bin"
  zstd -d -f -c /lib/firmware/qca/crnv21.bin.zst > "${FW_DST_QCA}/crnv21.bin"
  have21=1
fi

if [ $have21 -eq 0 ]; then
  echo "[wcn3990-fw] Could not find crbtfw21.tlv.zst/crnv21.bin.zst under /lib/firmware/qca" >&2
  exit 1
fi

echo "[wcn3990-fw] Bind-mounting ${FW_DST_BASE} over /lib/firmware (override)"
mountpoint -q /lib/firmware || mount -o bind "${FW_DST_BASE}" /lib/firmware

echo "[wcn3990-fw] Firmware installed:"
ls -l "${FW_DST_QCA}"

echo "[wcn3990-fw] Done. Next: rfkill unblock bluetooth; btattach -B /dev/ttyHS0 -S 115200 -P qca; hciconfig hci0 up"

