Title: Comma 3/3X Bluetooth Bring‑Up — Current Status (Live Document)

Read me first (agent + human)
- This document is the always‑up‑to‑date source of truth for the Bluetooth bring‑up effort on Comma 3/3X.
- Maintain it in real time: keep “Current Status” and “Plan of Record” accurate; remove stale text promptly. Do NOT grow a history here — use the separate CHANGELOG for brief audit notes.
- Any new chat/session should start by reading this file and then proceeding with “Plan of Record”.

Canonical paths, access, and environment
- Authoritative AGNOS builder: `/projects/agnos/agnos-builder`
  - Do not use shadow copies. If a reference copy is kept, archive it under `docs/chauffeur/bluetooth3x/archived/` and call it out here.
- Working repo (this tree): `/projects/chauffeur/data/openpilot`
- Device: Comma 3/3X (SDM845 + WCN3990; UART HCI via `hci_qca` on `/dev/ttyHS0`)
- Host dev: WSL Ubuntu on PC; ADB via USB or TCP/IP
  - Host sudo password (WSL): `Ne!sonB00ger`
  - On device: `adb root` allowed; `su - comma` has no password.

ADB connectivity (pick one)
- USB via usbipd-win 4.x (WSL):
  - Windows (PowerShell):
    - List devices: `usbipd list`
    - Share once (persistent): `usbipd bind --busid <BUSID>`
    - Attach to WSL: `usbipd attach --wsl --busid <BUSID>` (optionally add `--auto-attach`)
    - Detach when done: `usbipd detach --busid <BUSID>`
  - Example (user’s device): `usbipd attach --auto-attach --wsl --busid 2-1`
  - WSL (Ubuntu):
    - `adb kill-server && sudo adb start-server && adb devices -l`
    - If permissions persist, ensure udev rules cover the device VID (some ADB interfaces enumerate as `04d8:1234`, otherwise `18d1:*` or `05c6:*`).
- TCP/IP (fallback during attach if USB unstable):
  - `adb wait-for-device && adb tcpip 5555`
  - `DEVICE_IP=$(adb shell 'ip -o -4 route get 1.1.1.1 | awk '{print $7}'')`
  - `adb connect $DEVICE_IP:5555` and run tools with `ADB="adb -s $DEVICE_IP:5555"`

Windows USB enumeration note (Sep 11, 2025)
- After a device reboot, Windows temporarily did not enumerate the Comma 3/3X; `usbipd list` showed only other devices under Connected and the Comma device under Persisted as `UsbNcm Host Device, ADB Interface` (GUID present).
- Recovery path: ensure a known‑good data cable and a direct motherboard USB port. Once Windows lists the device in `usbipd list` under Connected with a BUSID, run `usbipd bind --busid <BUSID>` (once) and `usbipd attach --wsl --busid <BUSID>` (add `--auto-attach` if desired). Then continue with ADB steps.
- If Windows shows the device but ADB in WSL says `no permissions`, reload udev (`sudo udevadm control --reload-rules && sudo udevadm trigger`), and ensure a permissive rule for VID `04d8` in addition to `18d1` and `05c6`.

Stability retrospective (what broke, what fixed it) — Sep 11, 2025
- Symptoms observed:
  - WSL: intermittent `adb: no devices/emulators found` during long attach attempts; device vanished mid‑run.
  - Windows PowerShell from usbipd: “Device 2‑1 is now detached” followed by “BUSID 2‑1 is not available on host 172.24.32.1”. The device then rebooted and reattached.
- Root causes:
  1) Overlong attach windows (≥60–90 s) while pounding UART caused the USB gadget to drop and trigger a device reboot.
  2) ADB/usbipd churn: earlier sessions occasionally restarted `adbd` (`adb root`, `adb kill-server`) or detached/reattached via old `usbipd wsl attach ...`, which can briefly sever the transport during attach.
  3) Udev permissions mismatch for VID `04d8:1234` before the new rule was added, leading to initial “no permissions”.
- Mitigations now in place (keeping it steady):
  - Use current usbipd syntax and auto‑attach: `usbipd attach --wsl --auto-attach --busid <BUSID>`; avoid detach/attach churn once attached.
  - Do not restart ADB during attach cycles; prefer `adb shell su -c ...` to avoid `adbd` restarts.
  - Cap attach windows to ≤40 s per attempt; avoid 60–90 s brute‑force retries that destabilize USB.
- Maintain permissive udev rules for `04d8`, `18d1`, `05c6`; avoid mixing multiple ADB servers.
  - Avoid `adb root` and `adb kill-server` during runs; these correlate with usbipd detachments. Use `su -c` on‑device instead.

Session log (USB stability)
- 12:41:57: usbipd: Device 2‑1 attached (stable).
- 12:45:25: usbipd: Device 2‑1 detached; BUSID not available; device rebooted. This aligned with an overlong (90 s) attach attempt. Since capping to ≤40 s and avoiding adb resets, usbipd has remained steady.

Firmware assets
- Bind firmware: `mount --bind /data/firmware /lib/firmware`
- WCN3990 files present under `/data/firmware/qca/`: `crbtfw21.tlv`, `crnv21.bin` (uncompressed).

Current kernel and state (Sep 11, 2025)
- Kernel: `4.9.103 #63 SMP PREEMPT Thu Sep 11 20:29:47 UTC 2025` (BT timing/rate + optional wake‑before‑reset)
- Driver changes in #56:
  - btqca: Wait for `HCI_EV_VENDOR (0xFF)` in ROM read (fixes earlier 0xFC00 vendor wait bug).
  - btqca: Upstream skip‑VSE TLV semantics (early segments async with `__hci_cmd_send`, last segment sync; inject one synthetic CC to satisfy HCI credits).
  - hci_qca: WCN3990 baud‑switch “drop vendor event 0x92 + wait ~100 ms”, flip host UART immediately after controller command; IBS disabled during download.
  - btqca: `request_firmware_direct()` to avoid fallback races.
  - Added minimal BT_INFO breadcrumbs in btqca/hci_qca.

Boot loop note (resolved)
- A ~60 s cycle occurred while tethered over USB; unplugging once allowed a stable boot. Cause was host ADB/USB role flipping, not a kernel 60 s timeout. Proceed with TCP/IP or a sudo‑started adb server for stability.

Current BT status (fresh baseline)
- Device is up and stable; GUI responsive.
- Firmware present and bind‑mounted.
- Attach path is UART on `/dev/ttyHS0` with `hci_qca`.
- Cycle 1 (conservative) executed on Sep 11, 2025; TLV download failed with repeated `0xfc00 tx timeout` despite ROM read succeeding.
- Cycle 2 (pre‑bump) executed on Sep 11, 2025; early UART bump to 3,000,000 results in `HCI Reset (0x0C03) tx timeout` followed by repeated `0xfc00 tx timeout`.

Cycle 1 findings (Sep 11, 2025 — conservative)
- Key dmesg breadcrumbs:
  - `Bluetooth: hci0: ROME setup`
  - `Bluetooth: hci0: Frame reassembly failed (-84)`
  - `Bluetooth: hci0: ROME controller version 0x02140201`
  - `Bluetooth: hci0: ROME Downloading file: qca/crbtfw21.tlv`
  - `Bluetooth: hci0 command 0xfc00 tx timeout` (multiple occurrences)
  - `Bluetooth: hci0: Failed to send TLV segment (-110)` → `Failed to download FW/patch (-5)`
- btattach.out (tail):
  - `Attaching Primary controller to /dev/ttyHS0`
  - `Switched line discipline from 0 to 15`
  - `Device index 0 attached`
- Interpretation: ROM probe and start of TLV download succeeded; transfer stalls on vendor 0xFC00 transactions. Next step is pre‑bump attach to raise UART before TLV/NVM.

Cycle 2 findings (Sep 11, 2025 — pre‑bump)
- Key dmesg breadcrumbs:
  - `Bluetooth: hci0: ROME setup`
  - `Bluetooth: hci0: ROME controller version 0x02140201`
  - `Bluetooth: hci0: Pre-bump UART to 3000000 before TLV/NVM`
  - `Bluetooth: hci0: Reset failed (-110)` / `hci0 command 0x0c03 tx timeout`
  - Subsequent `hci0 command 0xfc00 tx timeout` (multiple) and repeat ROM version read retries.
- btattach.out (tail for pre‑bump runs):
  - `Attaching Primary controller to /dev/ttyHS0`
  - `Switched line discipline from 0 to 15`
  - `Device index 0 attached`
- Interpretation: After host pre‑bump, controller appears not ready at the new rate; first `HCI Reset` at high speed times out, leading to downstream vendor timeouts.

Cycle 1 (extended, instrumented) findings — 40 s @115200 with dynamic debug
- Dynamic debug enabled for `hci_qca`/`btqca` (IBS traces visible).
- Key dmesg breadcrumbs:
  - `hci0: ROME Patch Version Request` → `ROME controller version 0x02140201` → `ROME HCI_RESET`.
  - `Bluetooth: hci0: ROME Downloading file: qca/crbtfw21.tlv` → initially one run hit `Failed to request file: err = (-2)` (transient path resolution), subsequent run succeeded.
  - During patch download: `Bluetooth: hci0 command 0xfc00 tx timeout` (multiple).
  - NVM stage seen: `Bluetooth: hci0: ROME Downloading file: qca/crnv21.bin`.
  - IBS activity observed: `Received HCI_IBS_WAKE_ACK in tx state 0`.
  - Final failure: `hci0: Reset failed (-110)` / `Failed to run HCI_RESET (-110)` followed by `command tx timeout`.
- Interpretation: With conservative timing the ROM probe and TLV/NVM fetches proceed, but firmware download transactions over 0xFC00 intermittently stall; subsequent reset also times out. IBS shows wake/ack events, so the stall is likely around the vendor command credit/ack timing rather than the link being asleep.

Stability note (Sep 11, 2025)
- A single extended attach attempt (90 s window) caused the device’s USB gadget to drop and the device to reboot. To minimize impact on development, keep attach windows ≤40 s while iterating. We will focus next on a kernel‑side timing patch instead of longer brute‑force retries.

Plan of Record (do this now)
1) Establish ADB link over USB via usbipd [done]
  - Windows (PowerShell): ensure device is Shared then Attached to WSL using new syntax: `usbipd list`; `usbipd bind --busid 2-1` (once); `usbipd attach --wsl --busid 2-1` (or `--auto-attach`).
  - WSL: `adb kill-server && sudo adb start-server && adb devices -l` should list the device without the permissions error. If vendor shows as `04d8:1234`, ensure udev has a permissive rule for `04d8`.
2) Conservative attach + capture (Cycle 1) [done]
  - `mount --bind /data/firmware /lib/firmware`
  - `echo Y > /sys/module/hci_uart/parameters/patch115200` (keep TLV/NVM at 115200)
  - `timeout 20-28s btattach -B /dev/ttyHS0 -P qca -S 115200 > /data/local/tmp/btattach.out 2>&1`
  - `dmesg | tail -n 1200 | rg -n 'Bluetooth|qca/(btqca|hci_qca)|ROME|0xfc00|Downloading'`
  - Pull `/data/local/tmp/btattach.out` and the dmesg slice; paste the key lines below.
3) If Cycle 1 shows no “ROM version …” [n/a; ROM read seen]
  - Cycle 2: pre‑bump before TLV/NVM — `echo N > /sys/module/hci_uart/parameters/patch115200` [done]; result: `HCI Reset` then `0xfc00` tx timeouts.
  - If still stalled, extend window and/or vary host speed. We also tried host‑start at 3,000,000 (`btattach -S 3000000`) with same outcome.
4) Next concrete steps (Cycle 3 — instrumentation) [done]
  - Enabled dynamic debug for `btqca`/`hci_qca`; reran Cycle 1@115200 with 40 s window and captured detailed breadcrumbs (see above).
  - Sanity long run (90 s) showed device instability; cap attach windows to ≤40 s.
5) Cycle 4 — kernel timing patch [built, flashed]
  - Changes:
    - hci_qca: increased baud-change vendor event (0x92) wait from ~100 ms to 300 ms and added small post‑flip settle.
    - btqca: added module param `btqca.tlv_force_sync=Y/N` (default Y) to force synchronous TLV segments; reduced TLV segment size to 96 B with 15 ms pacing.
    - Early ROM‑read attempt in `hci_qca` removed; rely on `qca_uart_setup_rome()` for reset+ROM read.
    - hci_qca: added module param `hci_qca.wake_before_reset=Y/N` (built‑in default Y in #63) to send a single IBS WAKE/ACK nudge right before `rome_reset()`.
    - btqca: extended HCI_RESET completion timeout from 1 s to 2 s and added a small pre‑reset settle in `qca_uart_setup_rome()`.
  - Results (30–40 s windows, su -c only):
    - With `tlv_force_sync=N`, `patch115200=Y`, and `wake_before_reset=Y` (built‑in), attach still fails early: `HCI Reset (0x0C03) tx timeout`, followed by repeated `0xfc00 tx timeout` when attempting ROM version.
    - Unblocking rfkill (`rfkill unblock all`) prior to attach did not change early failure (confirmed rfkill now shows Soft blocked: no).
6) Next steps (Cycle 5 — early handshake focus)
  - Verify controller readiness before first HCI_RESET by adding a short synchronous TX/echo probe at 115200 (H4 idle write), or by waiting for an IBS WAKE_ACK; instrument with dynamic debug to confirm.
  - Increase the initial RESET completion timeout further (up to 3–4 s) and add an extra 40–60 ms pre‑RESET settle; A/B this change alone.
  - If RESET continues to time out, inject a single vendor wake‑up sequence known for ROME parts (if available) before ROM version request.
  - Continue to cap attach windows at ≤40 s and avoid adb/usbipd churn; capture `btattach.out` and a 1600‑line `dmesg` tail for each attempt.

ADB connection outcome (Sep 11, 2025 — this session)
- Commands run:
  - `adb kill-server && sudo adb start-server && adb devices -l`
  - Installed udev rules for VID 18d1 and 05c6; reloaded rules.
  - Retried with root adb server only.
- Observed output (before correcting usbipd usage):
  - `List of devices attached`
  - `e521630c               no permissions (missing udev rules? user is in the plugdev group); ... usb:1-1 transport_id:1`
- Updated procedure (confirmed):
  - Use `usbipd attach --wsl --busid 2-1` (optionally `--auto-attach`) instead of deprecated `usbipd wsl attach ...`. If needed, run `usbipd bind --busid 2-1` once to persist sharing.
  - After attach, continue at Step 2 (USB path). TCP/IP fallback not required once usbipd is attached.

Blockers to watch for
- Host ADB flapping on USB role changes (fixed by sudo server or TCP/IP).
- If ROM read still times out with #56: re‑examine early 0xFC00 timing and IBS wake/settle; breadcrumbs should show where we are.

Procedures (reference)
- Build kernel: `cd /projects/agnos/agnos-builder && ./build_kernel.sh` → `output/boot.img`
- Flash (both slots):
  - `adb root; adb remount; adb push output/boot.img /data/local/tmp/boot.img`
  - `adb shell 'dd if=/data/local/tmp/boot.img of=/dev/block/bootdevice/by-name/boot_a bs=4M && sync'`
  - `adb shell 'dd if=/data/local/tmp/boot.img of=/dev/block/bootdevice/by-name/boot_b bs=4M && sync'`
  - `adb reboot && adb wait-for-device`
- Quick attach commands listed in “Plan of Record”.
- Optional live watchers: run `sh /data/local/tmp/on_device_watchers.sh` to record dmesg/swag tails into `/data/local/tmp/mon/` for later pull.

tmux (comma user)
- Attach: `adb shell su - comma -c 'tmux a -t comma-bt'`
- If session missing, re‑create:
  - `su - comma -c "tmux new-session -d -s comma-bt 'tail -F /data/log/swaglog.*'"`
  - `su - comma -c "tmux split-window -h -t comma-bt:0 'dmesg -w'"`
  - `su - comma -c "tmux split-window -v -t comma-bt:0.1 'logcat -b events -v brief'"`

Maintenance policy (agents)
- Keep this file live. Update “Current kernel and state”, “Current BT status”, and “Plan of Record” with each change.
- Keep it concise — all historical notes belong in `docs/chauffeur/bluetooth3x/CHANGELOG.md`.

Appendix A — Commands quick sheet
- ADB TCP/IP: `adb tcpip 5555; adb connect <IP>:5555`
- Bind firmware: `mount --bind /data/firmware /lib/firmware`
- Attach conservative: `echo Y > /sys/module/hci_uart/parameters/patch115200; btattach -B /dev/ttyHS0 -P qca -S 115200`
- Attach pre‑bump: `echo N > /sys/module/hci_uart/parameters/patch115200; btattach -B /dev/ttyHS0 -P qca -S 115200`
- Inspect BT: `ls -l /sys/class/bluetooth; rfkill list; hciconfig -a`
