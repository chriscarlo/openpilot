Bluetooth Bring‑Up Investigation (AGNOS / comma three)

Date: 2025-09-06
Host: /data/openpilot (AGNOS, aarch64)

Summary
- Device tree reports platform "comma tizi" compatible with qcom,sda845 (SDM845 family). This platform typically pairs with a Qualcomm WCN39xx (WCN3990/3988) Wi‑Fi/BT combo.
- Firmware partition contains Qualcomm BT/WLAN blobs. ver_info shows: btfm=BTFM.CHE.2.1.3-00289-QCACHROMZ-1, wlan=WLAN.HL.2.0.c10-00459-QCAHLSWMTPLZ-1.
- Kernel currently exposes no Bluetooth stack, drivers, or devices. No HCI devices present; no BlueZ utilities installed.
- Conclusion: Native BT bring‑up is not possible without adding/booting a kernel with Bluetooth support (plus userspace tools). A kernel/module update is required.

Environment Observations
- Kernel: Linux 4.9.103 #32 SMP PREEMPT (aarch64)
- Memory: 3.5 GiB total; ~1.5 GiB available at time of check; no swap configured.
- Disk: / at 93% (346 MiB free), /data at 90% (8.9 GiB free). Root is tight; /data has adequate headroom.
 - Kernel config (from /proc/config.gz): CONFIG_BT=y (built-in), but CONFIG_BT_HCIUART is not set; CONFIG_BTFM_SLIM=y and CONFIG_BTFM_SLIM_WCN3990=y are present. This confirms BT core exists but the UART/HCI transport for QCA is disabled.

Evidence Collected
- /proc/device-tree:
  - model: "comma tizi"
  - compatible: qcom,sda845-mtp; qcom,sda845; qcom,mtp
- /firmware (mounted vfat from /dev/sde4):
  - verinfo/ver_info.txt contains btfm and wlan versions (Qualcomm stack)
  - numerous WLAN/ADSP/SLPI images present
- Kernel/userspace state:
  - No /sys/class/bluetooth entries
  - No /dev/hci* nodes
  - lsmod shows no bluetooth/hci_uart/hci_qca/btqca modules
  - /lib/modules/<4.9.103> contains no bluetooth or qca UART modules
  - BlueZ tools not installed (no btattach/hciattach/hciconfig/bluetoothctl)

Likely Chipset
- Based on SoC (sda845) and firmware strings, the on-board BT is a Qualcomm WCN39xx (commonly WCN3990 on SDM845 designs).
- Linux driver stack: CONFIG_BT + CONFIG_BT_HCIUART + CONFIG_BT_HCIUART_QCA (+ btqca helper). On modern kernels, serdev brings this up via DT.

What’s Missing
1) Kernel support:
   - Bluetooth core and HCI UART
   - Qualcomm HCI UART (QCA) glue (btqca)
   - Device tree nodes for the BT UART + power/reset GPIO/regulators
2) Userspace:
   - BlueZ utilities for control (`btattach`/`hciattach`, `btmgmt`, `bluetoothd`/`bluetoothctl`)
3) Integration:
   - Boot-time power sequence for the QCA BT, correct UART speed, and firmware rampatch/nv download (handled by kernel+btqca on serdev, or by userspace tools for legacy paths)

Bring‑Up Paths
Option A — Proper kernel enablement (recommended if you proceed)
- Rebuild AGNOS kernel with:
  - CONFIG_BT=y/m, CONFIG_BT_HCIUART=y/m, CONFIG_BT_HCIUART_QCA=y/m, CONFIG_BT_DEBUGFS=n (optional)
  - Ensure UART (likely /dev/ttyMSM0 on SDM845) is bound in DT to the Bluetooth node with `qcom,wcn3990-bt`/`qcom,qca` compatible and proper vregs/enable GPIO.
  - Include btqca helper if not selected automatically.
  - Confirm firmware search paths for rampatch/nv (often under /lib/firmware/qca or vendor firmware partition; AGNOS has /firmware mounted and may need a symlink or kernel path config).
- Userspace:
  - Install BlueZ tools (`btmgmt`, `bluetoothctl`) and, if not using serdev auto‑probe, `btattach` with `-P qca`.
  - Create a systemd unit (if using AGNOS services) to power up and attach the BT UART on boot.
- Validate:
  - dmesg should show: "Bluetooth: HCI UART driver ver …", "Bluetooth: qca" lines, and `hci0` creation.
  - `btmgmt info`, `hciconfig -a`, and `bluetoothctl power on` should work.

Option B — Out‑of‑tree modules only (lowers risk slightly vs full kernel)
- Build modules against the running kernel (4.9.103) with the exact toolchain and configs. This requires matching kernel headers/configs from AGNOS. If headers/config aren’t available, this path is impractical.
- Load modules: `modprobe hci_uart`, `modprobe btqca` and attach via `btattach -B /dev/ttyMSM0 -P qca -S 3000000`.
- Caveats: Mismatch risks, missing DT power/clock support, and the need for correct firmware paths make this fragile.

Option C — Avoid modifying kernel (not feasible here)
- If drivers were built-in already, we would expect dmesg and /sys/class/bluetooth/hci* to exist. They do not. Userspace alone (BlueZ) cannot create an HCI device without kernel support.

Risks & Notes
- Kernel flash risk: device unbootable if images are wrong. Ensure you can recover via AGNOS updater and that /persist is backed up.
- Root FS is tight (~346 MiB free). Prefer placing tools and any large artifacts under /data.
- No swap means memory spikes can OOM; be cautious when building on‑device.
- The AGNOS partition recipe references a "bluetooth" image in system/hardware/tici/all-partitions.json, but on this unit no separate partition is labeled as such. Firmware blobs under /firmware do include BT though.

Minimal Viability Checklist (before doing any flash)
- [ ] Obtain/confirm AGNOS kernel sources/config for 4.9.103 build used here.
- [ ] Identify UART+GPIO wiring for BT in DT (SDM845 typically uses a dedicated BLSP UART).
- [ ] Confirm firmware filenames expected by hci_qca on this platform (rampatch/nv names vary by WCN HW revision).
- [ ] Decide module vs built‑in; module approach requires matching headers and build system.
- [ ] Plan recovery: ensure you can revert to stable AGNOS if BT bring‑up fails.

Quick Commands (post-enable validation)
- Kernel logs: `dmesg | rg -i "bluetooth|hci|qca|wcn"`
- Devices: `ls -l /sys/class/bluetooth`, `hciconfig -a`, `btmgmt info`
- BlueZ bring‑up (legacy): `btattach -B /dev/ttyMSM0 -P qca -S 3000000` then `hciconfig hci0 up`

Bottom Line
- This device includes Qualcomm BT firmware, and the SoC suggests a QCA WCN39xx is present. However, the running AGNOS kernel lacks the BT driver stack. To make the BT chip "come alive", you will need a kernel that enables Bluetooth (HCI UART + QCA) and the corresponding userspace tools. That implies a kernel/module build and likely a flash. If you prefer to avoid that risk, BT bring‑up is not realistically achievable on this image.
