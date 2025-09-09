Bluetooth on AGNOS (comma 3/3X): Docs, Scripts, and UI Plan

Overview
- Goal: provide a robust, repeatable way to enable Bluetooth on AGNOS devices and add Offroad UI controls (enable/disable, discoverability, scan/pair basics).
- Current device state: Kernel and DT already include WCN3990 (SLIM) support; userspace (BlueZ) and firmware are required. We validated this on-device and documented the full bring-up.

What’s here
- AGNOS_BLUETOOTH_ENABLE_GUIDE.md: original on-device enablement guide.
- AGNOS_BLUETOOTH_ENABLE_GUIDE_expanded.md: live, device-specific run log and guidance.
- Scripts (see tools/chauffeur/bluetooth3x):
  - check_and_enable_bt.sh: probes system and, if needed, installs bluez, rfkill, and linux-firmware noninteractively on the active slot, enables bluetooth.service, and verifies controller availability. Each step is time-bounded.
  - check_bt_only.sh: read-only probe that reports readiness (service, rfkill, HCI, firmware sample).
  - bt_helper.py: bounded backend helper (bluetoothctl-based) with subcommands for status, power/discoverable toggles, scan once, pair/trust/connect/disconnect/remove.
    - Interactive pairing support:
      - pair-interactive <ADDR> [--json]: exits with codes for prompts (10 confirm, 11 pin, 12 passkey) and emits JSON payloads
      - pair-complete <ADDR> --confirm yes|no | --pin PIN | --passkey N
    - Intended for UI use via QProcess; all operations have kill-guards.

Intended integration (not wired yet)
- openpilot setup/launch could call check_and_enable_bt.sh early to ensure BT is enabled after an AGNOS reflash. For now, scripts are standalone; integrate later by sourcing from /data/openpilot/launch_openpilot.sh or a setup phase.

UI plan (Offroad)
- Network → Bluetooth submenu includes:
  - Toggles: BluetoothEnabled (master enable/disable) and BluetoothDiscoverable; both persist via Params and call the helper asynchronously.
  - Adapter status: shows controller address, powered, and discoverable using helper status --json.
  - Scan (once): runs asynchronously with an on-screen “Scanning…” indicator; results sorted by connected → paired → RSSI.
  - Device list: shows name, address, RSSI, and flags (paired/trusted/connected) with actions per row: Pair, Connect/Disconnect, Trust/Untrust, Remove.
  - Pairing flows: JustWorks auto-completes; passkey confirmation shows a large confirm dialog; PIN/passkey entry uses input dialogs and completes pairing via helper.
  - Concise on-screen messages show progress/errors in-UI (grey for info, amber for attention), per the brand style guide.
- Params (under /common/params): BluetoothEnabled, BluetoothDiscoverable
- Backend: reuse system bluetoothd; for scanning/pairing, a small helper using bluetoothctl/btmgmt or DBus (bluez) will service UI requests.
  - Status and scan/pair flows should go through bt_helper.py. See SCRIPTS.md for supported subcommands and JSON outputs.

Notes
- SLIM (WCN3990) path doesn’t require HCI UART; do not enable BT_HCIUART.
- linux-firmware is large (~500 MB). The script allows pre-downloading to /data/pkgcache to reduce latency.
- A/B slot staging is supported in the guide; however, on this device abctl slot switching failed without external tools.

Params
- BluetoothEnabled: PERSISTENT|BACKUP, BOOL, default "0"
- BluetoothDiscoverable: PERSISTENT|BACKUP, BOOL, default "0"
