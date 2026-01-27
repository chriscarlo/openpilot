#!/usr/bin/env bash
set -euo pipefail

# Push userspace units and firmware helper to the device via adb wrapper
# and enable persistent Bluetooth attach services.

ADB=${ADB:-scripts/dev/adb_safe.sh}

die(){ echo "[install_bt] $*" >&2; exit 2; }

[[ -x "${ADB}" ]] || die "ADB wrapper not found: ${ADB}"

src_root="$(cd "$(dirname "$0")" && pwd)"

push_file() {
  local src="$1" dst="$2"
  "${ADB}" -t 30 push "$src" "$dst" >/dev/null
}

shell() {
  "${ADB}" -t 20 shell "$@"
}

echo "[*] Installing firmware helper and udev rule"
shell "mkdir -p /usr/libexec /etc/udev/rules.d /usr/lib/systemd/system /usr/lib/systemd/system/bluetooth.service.d"
push_file "${src_root}/firmware_zstd_helper.sh" "/usr/libexec/firmware_zstd_helper.sh"
shell "chmod 0755 /usr/libexec/firmware_zstd_helper.sh"
push_file "${src_root}/99-firmware-zstd.rules" "/etc/udev/rules.d/99-firmware-zstd.rules"

echo "[*] Installing systemd units"
push_file "${src_root}/bt-unblock.service" "/usr/lib/systemd/system/bt-unblock.service"
push_file "${src_root}/qca-bt-attach.service" "/usr/lib/systemd/system/qca-bt-attach.service"
push_file "${src_root}/../userspace/bluetooth.service.d/override.conf" "/usr/lib/systemd/system/bluetooth.service.d/override.conf"

echo "[*] Enabling units"
shell "systemctl daemon-reload && systemctl enable bt-unblock.service qca-bt-attach.service || true"
shell "udevadm control --reload-rules && udevadm trigger --subsystem-match=firmware || true"

echo "[*] Done. Reboot to validate persistent attach."

