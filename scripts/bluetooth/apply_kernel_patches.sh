#!/usr/bin/env bash
set -euo pipefail

# Apply the QCA/WCN3990 patchset to a kernel tree and optionally build.
# Usage:
#   scripts/bluetooth/apply_kernel_patches.sh /path/to/agnos-kernel-sdm845 [--build]

KDIR=${1:-}
DO_BUILD=false
[[ $# -ge 2 && "$2" == "--build" ]] && DO_BUILD=true

if [[ -z "$KDIR" || ! -d "$KDIR" ]]; then
  echo "Usage: $0 /path/to/kernel [--build]" >&2
  exit 2
fi

PATCHDIR=docs/chauffeur/bluetooth3x/patches

for p in 0001-hci_qca-wcn3990-power-pulses-and-port-reopen.patch \
         0002-hci_qca-drop-baudrate-change-vendor-event-and-add-3.2M.patch \
         0003-btqca-wcn3990-firmware-tlv-nvm-download-harder-retries.patch \
         0004-firmware-class-add-zstd-compressed-firmware-support.patch \
         0005-lib-add-decompress_unzstd-stub.patch \
         0006-btqca-fallback-to-request_firmware-on-ENOENT.patch \
         kernel_defconfig_fwloader_zstd_enable.patch; do
  echo "[apply] applying $p"
  git -C "$KDIR" am "$PATCHDIR/$p"
done

if $DO_BUILD; then
  echo "[apply] building kernel (make olddefconfig && make -j$(nproc))"
  make -C "$KDIR" olddefconfig
  make -C "$KDIR" -j"$(nproc)"
fi

echo "[apply] done"
