Title: Enable Bluetooth on AGNOS (comma 3/3X) — On-Device Playbook
Author: Codex Agent (this device)
Created: $(date -u)

Summary
- Goal: Enable Bluetooth on AGNOS for comma 3/3X without wiping user data.
- Strategy: Kernel and DT already include WCN3990 support; add userspace (BlueZ) and firmware. Do it entirely on-device by installing into the inactive A/B system slot, then switch slots.
- Reboot-safe: This file is stored under /persist so you can follow it after reboot.

Environment Snapshot (from this device)
- Kernel: uname -a -> Linux $(uname -n) $(uname -r) $(uname -v) $(uname -m)
- Selected kernel config flags (zcat /proc/config.gz | rg ...):
  - CONFIG_MODULES=y, CONFIG_MODVERSIONS=y, CONFIG_MODULE_SIG=y (SIG_FORCE not set)
  - CONFIG_BT=y, CONFIG_BT_BREDR=y, CONFIG_BT_LE=y
  - CONFIG_BT_HCIUART is NOT set (not needed for WCN3990 SLIM path)
  - CONFIG_BTFM_SLIM=y, CONFIG_BTFM_SLIM_WCN3990=y
  - CONFIG_RFKILL=y
  - SLIMBus + MSM GENI UART present (not required for UART HCI)
- Device Tree nodes (grep /proc/device-tree):
  - /proc/device-tree/soc/slim@17240000/wcn3990 (compatible: qcom,btfmslim_slave)
  - /proc/device-tree/vendor/bt_wcn3990 (compatible: qca,wcn3990)
- Userspace present before change:
  - BlueZ: absent (no bluetoothd, no bluetooth.service)
  - Firmware: /lib/firmware/qca missing
- Modules dir: /lib/modules/$(uname -r) missing; kernel is built monolithic for BT path.
- Active boot slot: abctl --boot_slot -> _a

Conclusion
- “Modules only” is unnecessary and impractical here. Kernel already contains BT core + WCN3990 driver and DT describes the device. Missing pieces are BlueZ and firmware files.
- We will install BlueZ + firmware into the INACTIVE system slot, then switch slots and verify.
- Data safety: This does not touch /data or /persist. Only the inactive system partition is modified; then we switch slots.

Plan (High-Level)
1) Determine inactive slot and mount it at /mnt/next
2) Chroot into /mnt/next and install packages: bluez, rfkill, linux-firmware
3) Enable bluetooth service in the chroot
4) Unmount and switch active slot with abctl; reboot
5) Verify controller bring-up and logs; mark slot successful

Detailed Step-by-Step (Copy/Paste Friendly)
Note: This assumes current active is “_a” (verified above). If your active differs, adapt TARGET accordingly.

Pre-flight
- Ensure the device has network connectivity for apt (Ethernet or USB networking).
- Optional: Save this doc and your current params to be extra safe.

Commands

1) Identify active slot and set target variable:
   current=$(abctl --boot_slot)   # shows _a or _b
   if [ "$current" = "_a" ]; then TARGET=/dev/disk/by-partlabel/system_b; else TARGET=/dev/disk/by-partlabel/system_a; fi
   echo "Active: $current, Target: $TARGET"

2) Mount target rootfs to /mnt/next:
   mkdir -p /mnt/next
   mount "$TARGET" /mnt/next

   # Prepare minimal chroot mounts
   mount --bind /dev /mnt/next/dev
   mount -t proc proc /mnt/next/proc
   mount -t sysfs sys /mnt/next/sys
   cp -L /etc/resolv.conf /mnt/next/etc/resolv.conf

3) Install BlueZ + firmware inside target rootfs:
   chroot /mnt/next bash -lc 'set -e; apt-get update && apt-get install -y --no-install-recommends bluez rfkill linux-firmware'

   Notes:
   - Package bluez provides: bluetoothd, bluetoothctl, btmgmt, udev rules, and systemd unit.
   - Package linux-firmware provides QCA firmware under /lib/firmware/qca.
   - The kernel cmdline includes firmware_class.path=/lib/firmware/updates which takes precedence. If you later need to override firmware, place files under /lib/firmware/updates/.

4) Enable bluetooth service for the target system:
   chroot /mnt/next bash -lc 'systemctl enable bluetooth || true'

   Optional auto-enable Bluetooth controller at boot:
   echo -e '\n[Policy]\nAutoEnable=true\n' >> /mnt/next/etc/bluetooth/main.conf

5) Clean up mounts:
   umount /mnt/next/sys /mnt/next/proc /mnt/next/dev || true
   umount /mnt/next

6) Switch slots and reboot:
   # _a = slot 0, _b = slot 1 (abctl uses numbers)
   if [ "$current" = "_a" ]; then abctl --set_active 1; else abctl --set_active 0; fi
   sync; reboot

Post-Reboot Verification
- Check that we booted into the new slot:
  abctl --boot_slot    # should now be the opposite of before

- Confirm BlueZ service:
  systemctl status bluetooth --no-pager

- Check firmware presence:
  ls /lib/firmware/qca | head

- Unblock RFKill and bring up controller:
  rfkill list || true
  rfkill unblock all || true

  # Show controller info (pick 0 or per output of btmgmt)
  btmgmt --index 0 info || hciconfig -a

  # Power on controller and inspect
  btmgmt --index 0 power on || bluetoothctl show

- Logs (kernel/user):
  dmesg | rg -i 'bluetooth|qca|wcn|btfm'
  journalctl -u bluetooth -n 100 --no-pager

Rollback / Recovery
- If anything fails to boot or Bluetooth still misbehaves:
  - At bootloader/A/B level: set the other slot active
    - abctl --set_active 0   # slot a
    - abctl --set_active 1   # slot b
    - reboot
  - Flash back a known-good AGNOS using https://flash.comma.ai or agnos-builder’s scripts.

Alternative: External Host Build (agnos-builder)
- If you prefer baking this into a reproducible system image (no on-device apt):
  - Repo: https://github.com/commaai/agnos-builder
  - Minimal change: install bluez rfkill linux-firmware in Dockerfile.agnos and enable bluetooth in userspace/services.sh
  - I prepared a sample patch earlier (see docs/agnos/enable_bluetooth_on_agnos.md in the openpilot workspace where I’m running) and instructions to build & flash. Flashing only system_a/b will not wipe /data or /persist.

Why “modules only” is not the path
- Kernel config on this device:
  - CONFIG_BT=y (built-in), CONFIG_BTFM_SLIM[_WCN3990]=y (built-in)
  - CONFIG_BT_HCIUART is not set (not required for WCN3990 SLIM transport)
- DT declares WCN3990 nodes; modules cannot add missing DT bindings.
- /lib/modules/$(uname -r) is absent; ABI/module signing would require rebuilding exact kernel headers and Module.symvers anyway.
- Therefore, userspace+firmware is the correct minimal change.

Reference Links
- AGNOS builder repo: https://github.com/commaai/agnos-builder
- AGNOS kernel (sdm845, 4.9.103): https://github.com/commaai/agnos-kernel-sdm845
- Kernel Makefile (4.9.103): https://github.com/commaai/agnos-kernel-sdm845/blob/master/Makefile
- BlueZ upstream: https://git.kernel.org/pub/scm/bluetooth/bluez.git/about/
- Linux firmware repository: https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/about/

Caveats & Notes
- Ensure network connectivity before apt operations in the chroot.
- linux-firmware generally contains QCA BT blobs; if the specific WCN3990 files are missing, you can place firmware overrides under /lib/firmware/updates/qca in the target root.
- After confirming stability, mark the slot as successful:
  abctl --set_success
- This document lives in /persist and survives reboots.

Appendix: Raw outputs captured earlier (for context)
- Kernel config grep (selected flags):
  CONFIG_MODULES=y
  CONFIG_MODVERSIONS=y
  CONFIG_MODULE_SIG=y
  CONFIG_MODULE_SIG_ALL=y
  CONFIG_BT=y
  CONFIG_BT_BREDR=y
  CONFIG_BT_LE=y
  CONFIG_BT_HCIUART is not set
  CONFIG_BTFM_SLIM=y
  CONFIG_BTFM_SLIM_WCN3990=y
  CONFIG_RFKILL=y

- DT nodes observed:
  /proc/device-tree/soc/slim@17240000/wcn3990 (qcom,btfmslim_slave)
  /proc/device-tree/vendor/bt_wcn3990 (qca,wcn3990)

- Absent before change:
  - bluetoothd not found
  - bluetooth.service not present
  - /lib/firmware/qca missing

Offroad UI usage (optional)
- Navigate to Network → Bluetooth to view adapter status.
- Use the Enable and Discoverable toggles to control power and visibility.
- Tap “Scan (once)” to discover nearby devices. The UI shows progress and then lists devices with actions (Pair, Connect/Disconnect, Trust/Untrust, Remove).
- If a device requires a PIN or passkey confirmation, on-screen prompts will guide you through pairing.

End of document.
