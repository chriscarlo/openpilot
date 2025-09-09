Scripts for Enabling Bluetooth on AGNOS (comma 3/3X)

Location
- tools/chauffeur/bluetooth3x/check_bt_only.sh
- tools/chauffeur/bluetooth3x/check_and_enable_bt.sh
- tools/chauffeur/bluetooth3x/bt_helper.py

Usage
- Probe only (read-only):
  - bash tools/chauffeur/bluetooth3x/check_bt_only.sh
  - Exits 0 if Bluetooth is ready (service enabled/active and controller present), nonzero otherwise.

- Enable (active slot, noninteractive, time-bounded):
  - sudo bash tools/chauffeur/bluetooth3x/check_and_enable_bt.sh
  - Installs: bluez, rfkill, linux-firmware (from /data/pkgcache if available)
  - Enables + starts bluetooth.service, unblocks rfkill, attempts to power on HCI
  - Prints a concise status summary at the end.

- Backend helper (bounded, UI-friendly):
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py status [--json]
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py set-powered on|off
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py set-discoverable on|off [--timeout-sec N]
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py scan once [--timeout N] [--with-info] [--json]
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py pair <ADDR> [--timeout N]
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py pair-interactive <ADDR> [--timeout N] [--json]
    - Exits 0 on success; exits 10 with event=confirm and passkey; 11 with event=pin; 12 with event=passkey; 2 no controller; 3 timeout.
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py pair-complete <ADDR> [--confirm yes|no | --pin PIN | --passkey N] [--timeout N]
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py trust <ADDR> on|off
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py connect <ADDR>
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py disconnect <ADDR>
  - python3 tools/chauffeur/bluetooth3x/bt_helper.py remove <ADDR>
  - Notes: uses bluetoothctl under the hood; all operations time-bounded and guarded against hangs.

Design
- Each step is wrapped in a short timeout to avoid hanging on interactive prompts.
- dpkg/apt operations use noninteractive flags to keep config files.
- A pre-download cache directory (/data/pkgcache) avoids long downloads on the hot path.

Integration (deferred)
- openpilot setup/launch can call the enable script early to ensure BT is ready after an AGNOS flash.
- Do not integrate now; invoke manually for testing.

UI backend
- The Offroad Bluetooth panel calls bt_helper.py via QProcess for status, toggles, scan, pairing, and device actions.
- All operations are asynchronous with kill-guards to keep the UI responsive.
- JSON outputs are used for parsing status and scan results; pairing flows use pair-interactive and pair-complete.
