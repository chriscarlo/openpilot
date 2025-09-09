Title: Bluetooth (comma 3/3X on AGNOS) — Continuation Roadmap

Scope
- Continue and finalize Bluetooth support on comma 3/3X (AGNOS 24.04, kernel 4.9.103, WCN3990) with: robust on‑device enablement, Offroad UI controls, DBus backend for scan/pair/connect, and optional setup/launch hooks.
- This file is a self‑contained plan. Use it if prior discussion context is unavailable.

Ground Truth (validated on device)
- Kernel: 4.9.103 aarch64 with built‑in BT core and WCN3990 SLIM driver
  - CONFIG_BT=y, CONFIG_BT_LE=y, CONFIG_BTFM_SLIM=y, CONFIG_BTFM_SLIM_WCN3990=y
  - CONFIG_BT_HCIUART is not set (not required for WCN3990)
- DT nodes present for WCN3990 (SLIM) → driver should expose an HCI adapter without userland attach.
- Userspace requirements: BlueZ (bluetoothd, bluetoothctl, btmgmt) and QCA firmware (from linux‑firmware).
- rfkill device present (“bt_power”); ensure not blocked.

Artifacts (already in repo)
- Docs
  - docs/chauffeur/bluetooth3x/AGNOS_BLUETOOTH_ENABLE_GUIDE.md (baseline on‑device guide)
  - docs/chauffeur/bluetooth3x/AGNOS_BLUETOOTH_ENABLE_GUIDE_expanded.md (device‑specific run log + guidance)
  - docs/chauffeur/bluetooth3x/README.md (overview + UI plan)
  - docs/chauffeur/bluetooth3x/SCRIPTS.md (script usage)
- Scripts
  - tools/chauffeur/bluetooth3x/check_bt_only.sh (probe readiness)
  - tools/chauffeur/bluetooth3x/check_and_enable_bt.sh (install/enable with strict timeouts)
- Params (for UI)
  - BluetoothEnabled (PERSISTENT|BACKUP, BOOL, default "0")
  - BluetoothDiscoverable (PERSISTENT|BACKUP, BOOL, default "0")
- UI (scaffold)
  - selfdrive/ui/qt/network/bluetooth.{h,cc}: minimal Bluetooth panel
  - selfdrive/ui/qt/network/networking.{h,cc}: adds a button to open Bluetooth panel

Phase 1 — Stabilize Bring‑Up and Service Behavior
1. Confirm HCI adapter exposure on boot when bluetoothd is running:
   - bluetoothctl show (adapter present?)
   - btmgmt --index 0 info; rfkill list; dmesg | rg -i 'btfm|wcn3990|hci|bluetooth'
2. If adapter not present:
   - Verify firmware under /lib/firmware/qca (linux-firmware was installed).
   - Check for vendor‑specific power sequencing needs (rfkill “bt_power” should be unblocked; try toggling).
   - Restart bluetoothd: systemctl restart bluetooth
   - If still absent, capture a longer dmesg (bounded) for btfm_slim logs.
3. Service policy
   - Create a small systemd drop‑in or watcher (deferred) to ensure bluetoothd is only active Offroad unless user explicitly enables onroad use (security choice).
   - Honor Param BluetoothEnabled: when false, stop bluetoothd and disable discoverability.

Phase 2 — Backend Helper (scan/pair/connect) — DONE
Goal: Provide a bounded helper callable from the UI (QProcess) to avoid heavy deps in C++.
1. Location: tools/chauffeur/bluetooth3x/bt_helper.py
2. Stack: bluetoothctl with bounded timeouts. Optional later: bluez DBus via dbus-next.
3. API (CLI subcommands):
   - status: print adapter props (Powered, Discoverable, Address, Name)
   - set-powered on|off
   - set-discoverable on|off [timeout N]
   - scan start|stop|once --timeout N → prints JSON list of devices {addr,name,rssi,paired,trusted,connected}
   - pair <addr> [--timeout N] → best-effort JustWorks
   - pair-interactive <addr> [--timeout N] [--json] → detect need for confirm/PIN/passkey and exit with clear codes + JSON
   - pair-complete <addr> (--confirm yes|no | --pin PIN | --passkey N)
   - trust <addr> on|off
   - connect <addr> / disconnect <addr>
   - remove <addr>
4. Behavior:
   - All operations bounded with timeouts (5–15s typical; pairing up to 60s).
   - Map DBus errors to concise exit codes/messages.
   - If dbus-next not available, provide a best‑effort bluetoothctl fallback (echo commands; parse minimal output).
5. Output format: human‑readable + optional --json for UI parsing.

Phase 3 — Complete the Offroad UI — MOSTLY DONE
1. Add controls to selfdrive/ui/qt/network/bluetooth.cc:
   - Status line: Adapter address, Powered, Discoverable, Connected count.
   - Buttons: Power On/Off (ties to BluetoothEnabled Param and bt_helper set-powered), Discoverable On/Off (BluetoothDiscoverable Param).
   - Scan button: invokes bt_helper scan once (8–10s) asynchronously with progress label; shows list with name/RSSI.
   - Device list: for each discovered/paired device:
     - Buttons: Pair, Connect/Disconnect, Trust/Untrust, Remove
     - Pairing completes JustWorks or prompts for PIN/passkey; passkey confirmation via modal dialog.
2. Keep all QProcess calls bounded with timeout (e.g., wrap via a small helper that enforces timeouts and kills child on expiry).
3. Visuals: implemented per Offroad UI Brand Style Guide (titles, toggles, list styling).

Phase 4 — Optional Setup/Launch Hooks (deferred until approved)
1. Hook for install/launch (not committed yet):
   - In /data/openpilot/launch_openpilot.sh (or an earlier setup step), call tools/chauffeur/bluetooth3x/check_bt_only.sh; if non‑zero, call check_and_enable_bt.sh.
   - Gate by a Param (e.g., BluetoothAutoSetup) to avoid unexpected installs during critical moments.
2. Systemd alternative: ship a oneshot that runs at boot (offroad) to ensure bluetoothd is configured per Params. Not implemented yet; decide policy first.

Phase 5 — Testing & Validation
1. Manual tests (Offroad):
   - Enable Now → verify bluetooth.service active; bluetoothctl show reveals adapter.
   - Toggle Discoverable → verify bluetoothctl show Discoverable=yes; another device can see “comma‑…”
   - Scan → device list populates with expected nearby devices.
   - Pair flows: JustWorks, Passkey confirm, PIN entry; verify trust and connect; removal.
2. Logs:
   - journalctl -u bluetooth (bounded) for daemon output
   - dmesg (bounded) for driver messages (btfm, wcn3990, hci)
3. Edge cases:
   - rfkill blocked → unblock automatically and notify
   - Controller missing after service start → show actionable error; offer to collect short dmesg
   - Timeouts → surface short error toasts; no UI freeze

Phase 6 — Security & UX Policy
1. Offroad only by default: keep Discoverable off by default; require explicit toggle.
2. Onroad implications: optionally auto‑stop scans and disallow pairing onroad; show an Offroad‑only warning if attempted.
3. Persisted devices: list paired devices with ability to remove trust/bond.

Phase 7 — Packaging & CI
1. Ensure selfdrive/ui/SConscript includes bluetooth.cc (already added).
2. No new Python deps are required if using bluetoothctl fallback; if dbus-next is used, add to AGNOS userspace image or vendor a minimal single‑file DBus helper.
3. Lint and mypy: scripts under tools/chauffeur/bluetooth3x should pass basic shellcheck (optional) and have bounded runtime.

Phase 8 — Fallback & Rollback
1. If controller remains absent despite BlueZ+firmware:
   - Capture bounded dmesg; validate btfm_slim_wcn3990 probe; check /sys/class/bluetooth is empty
   - Consider re‑installing linux‑firmware or placing required blobs under /lib/firmware/updates/qca
   - If still failing, document symptoms in AGNOS_BLUETOOTH_ENABLE_GUIDE_expanded.md and proceed with external debugging
2. Rollback: disable bluetooth.service, set Params to 0, and remove cached firmware package from /data/pkgcache if space is needed.

Milestones & Checkpoints
M1. Backend ready: bt_helper.py provides status, power/discoverable toggles, scan once, pair (interactive)/trust/connect/remove — COMPLETE.
M2. UI complete: Offroad panel shows status, toggles respect Params, can scan/pair/connect/remove, async with progress/errors — COMPLETE.
M3. Optional integration: scripts invoked from setup/launch behind a guard Param (no default integration without approval).
M4. Validation: documented test pass on a real device; docs updated with examples/screenshots.

File Map (quick reference)
- Scripts: tools/chauffeur/bluetooth3x/*.sh (probe + enable)
- Docs: docs/chauffeur/bluetooth3x/* (guides, this roadmap)
- Params: common/params_keys.h (BluetoothEnabled, BluetoothDiscoverable)
- UI: selfdrive/ui/qt/network/bluetooth.{h,cc}, networking.{h,cc}, SConscript

References
- BlueZ DBus API: https://git.kernel.org/pub/scm/bluetooth/bluez.git/tree/doc
- linux-firmware (QCA): https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/about/
- AGNOS builder: https://github.com/commaai/agnos-builder
- AGNOS kernel (sdm845, 4.9.103): https://github.com/commaai/agnos-kernel-sdm845
