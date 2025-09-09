# Comma 3X — System-Only Rescue + Bluetooth Enablement

Goal
- Recover a comma 3X/comma three devkit that no longer boots due to Bluetooth/BlueZ experiments by reflashing only the system partition (not the full AGNOS).
- End with a working Bluetooth stack on AGNOS 24.04, with a controlled, reversible setup.

What this does NOT do
- It does not wipe `userdata` (drives, params) or other partitions.
- It does not change bootloader/firmware partitions unless you choose the full-flash fallback.

References
- agnos-builder (master): https://github.com/commaai/agnos-builder
- Official QDL/EDL flashing entry and device-specific notes: https://flash.comma.ai/

## Overview

1) Build or download a fresh `system.img` via agnos-builder.
2) Put the comma 3/3X into QDL/EDL mode and flash only the active slot’s system partition.
3) Boot and verify basic OS health.
4) Enable Bluetooth in a low‑risk way (on‑device apt + a small attach service), then optionally bake it into a forked agnos-builder for repeatability.

If anything goes wrong, you can always re-enter QDL and re-flash `system.img` again, or use the official web flasher to restore the device to a clean factory state.

---

## Prerequisites

- Host machine with Docker and libusb installed (Linux or macOS works; see README notes for macOS M‑series libusb symlink if needed).
- USB‑C data cable.
- Ability to put the device into QDL/EDL mode (see https://flash.comma.ai for instructions; device should enumerate as "Qualcomm HS-USB QDLoader 9008").

Optional but helpful
- SSH access or local shell on the comma 3X after it boots.

---

## Part 1 — Produce a Clean system.img

Fastest path (download from CI)
- agnos-builder ships a helper to fetch the latest release artifacts:
  - `git clone https://github.com/commaai/agnos-builder.git`
  - `cd agnos-builder && git submodule update --init agnos-kernel-sdm845`
  - `./tools/extract_tools.sh`
  - `python3 scripts/download-from-manifest.py` (pulls latest images) 
  - Result: `output/system.img`

Build locally (reproducible and editable)
- From a clean clone of the repo:
  - `./tools/extract_tools.sh`
  - `./build_system.sh`
  - Result: `output/system.img`

Note: Building only the system image is sufficient; no kernel/boot changes are required to recover from a BlueZ config mishap.

---

## Part 2 — Flash Only the System Partition

1) Put the comma 3/3X into QDL/EDL mode per https://flash.comma.ai/.
2) From inside the `agnos-builder` directory:
   - Check active A/B slot: `tools/edl getactiveslot`
   - Flash only system on the active slot: `./flash_system.sh`
     - This writes `system_a` or `system_b` automatically based on the active slot and then resets the device.

This is the minimal corrective action to restore boot without touching userdata or other partitions.

---

## Part 3 — Enable Bluetooth (Low‑Risk On‑Device Steps)

Rationale
- AGNOS 24.04 base does not include BlueZ by default. The kernel and firmware support are present, but the userspace stack and service wiring are not enabled. We add them in a controlled way using apt and a small attach service. If anything breaks, simply re‑flash `system.img` again as above.

Steps (on the device after it boots)

1) Install BlueZ and helpers
   - `sudo apt update`
   - `sudo apt install -y --no-install-recommends bluez bluez-tools rfkill`

2) Create a small attach helper (picks the right UART and attaches QCA BT)
   - `sudo tee /usr/local/bin/qca_bt_attach.sh >/dev/null <<'SH'
#!/usr/bin/env bash
set -euo pipefail

rfkill unblock bluetooth || true

# Try likely UARTs used on SDM845 boards; adjust if dmesg shows a specific one.
for dev in /dev/ttyHS0 /dev/ttyMSM1 /dev/ttyMSM0; do
  if [ -e "$dev" ]; then
    echo "[qca-bt] Attaching on $dev"
    # Prefer btattach if present; it cleanly daemonizes.
    if command -v btattach >/dev/null 2>&1; then
      exec btattach -B "$dev" -S 3000000 -P qca
    else
      # Fallback to hciattach; -n stays in foreground for systemd.
      exec hciattach -n -s 3000000 "$dev" qca
    fi
  fi
done

echo "[qca-bt] No known UART found; check dmesg for hci_qca/serdev logs" >&2
exit 1
SH`
   - `sudo chmod +x /usr/local/bin/qca_bt_attach.sh`

3) Add a systemd unit to run the attach before bluetoothd
   - `sudo tee /etc/systemd/system/qca-bt-attach.service >/dev/null <<'UNIT'
[Unit]
Description=Attach Qualcomm Bluetooth controller (QCA)
After=systemd-udevd.service
Before=bluetooth.service
# Only start if one of the likely UARTs exists
ConditionPathExistsGlob=/dev/ttyMSM* /dev/ttyHS*

[Service]
Type=simple
ExecStart=/usr/local/bin/qca_bt_attach.sh
Restart=on-failure
RestartSec=2s

[Install]
WantedBy=multi-user.target
UNIT`

4) Enable and start services
   - `sudo systemctl daemon-reload`
   - `sudo systemctl enable qca-bt-attach.service bluetooth.service`
   - `sudo systemctl start qca-bt-attach.service`
   - `sudo systemctl start bluetooth.service`

5) Verify
   - `dmesg | rg -i 'bluetooth|hci|qca'`
   - `rfkill list`
   - `hciconfig -a` (expect `hci0: UP RUNNING`)
   - `bluetoothctl show` (then `power on`, `scan on` to confirm)

If `hci0` never appears, check `dmesg` to see which UART was claimed by `hci_qca`/serdev and update the script to match that device node. You can also try a lower speed (e.g. `-S 115200`) if the controller requires it.

Persistence note
- AGNOS uses a mostly read‑only root with controlled apt hooks. The steps above persist across reboots. If you later want to undo them, disable the services and remove the files.

---

## Part 4 — Bake It In (Optional, for Reproducibility)

Once validated, you can upstream the change into your own branch of `agnos-builder` so new `system.img` builds have Bluetooth enabled by default.

Minimal diff outline
- Add BlueZ install to the image build (no recommends):
  - Append to `userspace/install_extras.sh` or create `userspace/install_bluetooth.sh` and `RUN` it from `Dockerfile.agnos`:
    - `apt-fast update && apt-fast install -y --no-install-recommends bluez bluez-tools rfkill`
- Add the attach unit to `userspace/files/qca-bt-attach.service` and make sure `userspace/services.sh` enables both `qca-bt-attach.service` and `bluetooth.service`.
- Rebuild and flash system only:
  - `./build_system.sh && ./flash_system.sh`

This avoids ad‑hoc on‑device changes and keeps your Bluetooth setup reproducible.

---

## Recovery and Fallbacks

- If the device fails to boot after any Bluetooth changes, re‑enter QDL/EDL and run `./flash_system.sh` with a known‑good `system.img` (either fresh CI download or a local build without your changes).
- If the boot chain or other partitions are suspected, use the official web flasher (full device restore): https://flash.comma.ai/.

---

## Quick Troubleshooting

- `tools/edl` says "No backend available" on macOS: install libusb and (on M‑series) symlink as noted in agnos-builder README.
- `qca-bt-attach.service` fails: check `journalctl -u qca-bt-attach -b`, confirm the correct `/dev/tty*` by reviewing `dmesg | rg -i 'tty|bluetooth|qca'`.
- Firmware missing: ensure `linux-firmware` and/or device‑specific WLAN/BT firmware packages are present. agnos-builder’s `agnos-wlan_*.deb` typically supplies QCA firmware; if not, install `linux-firmware`.

---

## Recommended Path (Summary)

- Do a system‑only flash using agnos-builder’s `flash_system.sh` to fix boot without wiping data.
- Start with the on‑device Bluetooth steps to validate hardware and wiring quickly.
- Once confirmed, bake the packages and service into a small agnos-builder branch to make future `system.img` builds Bluetooth‑ready by default.

This approach keeps risk low, preserves your data, and gets you to a working BlueZ stack on 3X without a full AGNOS reflash.

