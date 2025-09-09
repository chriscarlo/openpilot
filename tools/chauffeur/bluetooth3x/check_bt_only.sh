#!/usr/bin/env bash
set -euo pipefail

# Read-only probe. Prints a concise status and exits nonzero if BT is not ready.

timeout_cmd() {
  local secs="$1"; shift
  command -v timeout >/dev/null 2>&1 && timeout -k 5s "$secs" "$@" || "$@"
}

log() { echo "[bt-check] $*"; }

is_enabled=$(systemctl is-enabled bluetooth 2>/dev/null || true)
is_active=$(systemctl is-active bluetooth 2>/dev/null || true)

rfk=$(timeout_cmd 5s rfkill list 2>/dev/null || true)

fw_sample=$(ls /lib/firmware/qca 2>/dev/null | head -n 5 || true)

bt_info=""; bt_power_state=""
if command -v btmgmt >/dev/null 2>&1; then
  bt_info=$(timeout_cmd 5s btmgmt --index 0 info 2>/dev/null || true)
fi

ctl_show=""; if command -v bluetoothctl >/dev/null 2>&1; then
  ctl_show=$(timeout_cmd 5s bluetoothctl show 2>/dev/null || true)
fi

echo "bluetooth.service enabled=${is_enabled:-unknown} active=${is_active:-unknown}"
echo "rfkill:"; echo "$rfk" | sed 's/^/  /'
echo "firmware sample (/lib/firmware/qca):"; echo "$fw_sample" | sed 's/^/  - /'
echo "btmgmt info:"; echo "$bt_info" | sed 's/^/  /'
echo "bluetoothctl show:"; echo "$ctl_show" | sed 's/^/  /'

# Determine readiness
ready=0
[[ "$is_enabled" == "enabled" ]] || ready=1
[[ "$is_active" == "active" ]] || ready=1

# HCI available if bluetoothctl can show a controller
echo "$ctl_show" | rg -q "Controller" || ready=1

exit $ready

