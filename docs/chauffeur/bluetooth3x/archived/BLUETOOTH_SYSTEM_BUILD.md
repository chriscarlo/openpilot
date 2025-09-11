Bluetooth-Ready AGNOS System (Patch + Build)

This guide applies a minimal patch to agnos-builder so `system.img` ships with BlueZ and a small QCA attach service. You can then flash only the system partition on the comma 3/3X.

Files in this folder
- agnos-builder-bt.patch — Unified diff against agnos-builder master.
- See also: comma3x-bluetooth-rescue.md for the broader rescue plan.

Steps
1) Clone agnos-builder and prep tools
   - git clone https://github.com/commaai/agnos-builder.git
   - cd agnos-builder
   - git submodule update --init agnos-kernel-sdm845
   - ./tools/extract_tools.sh

2) Apply the Bluetooth patch
   - git checkout -b bluetooth-ready-system
   - git apply --whitespace=fix ../path/to/agnos-builder-bt.patch
     - Adjust path to this repo’s docs/chauffeur/bluetooth/agnos-builder-bt.patch
   - git commit -am "userspace: add bluez + qca-bt attach service"

3) Build system image
   - ./build_system.sh
   - Result: output/system.img

4) Flash only system (QDL/EDL mode required)
   - Put device into QDL mode (see https://flash.comma.ai)
   - ./flash_system.sh

5) Verify Bluetooth
   - After boot: dmesg | rg -i "bluetooth|hci|qca"
   - hciconfig -a (expect hci0 UP RUNNING)
   - bluetoothctl show; then power on; scan on

Notes
- The patch installs packages via apt-fast (used elsewhere in AGNOS image build), enables bluetooth.service, and adds qca-bt-attach.service (ExecStart=/usr/comma/bin/qca_bt_attach.sh).
- The attach script tries common SDM845 UARTs (/dev/ttyHS0, /dev/ttyMSM1, /dev/ttyMSM0) at 3Mbaud; adjust if dmesg shows another serdev.
- If you prefer to keep a clean upstream clone, fork on GitHub and apply the patch in the fork.

Rollback
- If anything regresses, re-enter QDL and re-run ./flash_system.sh with a known-good system.img (downloaded via scripts/download-from-manifest.py or built before this patch).

