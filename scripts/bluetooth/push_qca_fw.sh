#!/usr/bin/env bash
set -euo pipefail

# Push required QCA WCN3990 firmware files to device under /data/firmware/qca
# Usage: ADB=scripts/dev/adb_safe.sh scripts/bluetooth/push_qca_fw.sh [/path/to/firmware/qca]

ADB=${ADB:-adb}
SRC_DIR=${1:-firmware/qca}

need=(crbtfw21.tlv crnv21.bin)

for f in "${need[@]}"; do
  if [[ ! -f "${SRC_DIR}/${f}" ]]; then
    echo "[push_qca_fw] Missing ${SRC_DIR}/${f}" >&2
    exit 2
  fi
done

echo "[push_qca_fw] Creating target directories"
timeout 10s $ADB shell 'mkdir -p /data/firmware/qca && chmod 755 /data/firmware /data/firmware/qca' || true

echo "[push_qca_fw] Pushing firmware: ${need[*]}"
for f in "${need[@]}"; do
  timeout 60s $ADB push "${SRC_DIR}/${f}" "/data/firmware/qca/${f}" >/dev/null
done

echo "[push_qca_fw] Binding /data/firmware to /lib/firmware (on-device)"
timeout 10s $ADB shell 'mount --bind /data/firmware /lib/firmware || true'

echo "[push_qca_fw] Listing /data/firmware/qca on device"
timeout 10s $ADB shell 'ls -l /data/firmware/qca'

echo "[push_qca_fw] Done"

