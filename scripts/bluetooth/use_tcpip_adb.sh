#!/usr/bin/env bash
set -euo pipefail

# Switch adb to TCP/IP mode and print the serial to use (IP:port)
# Usage: scripts/bluetooth/use_tcpip_adb.sh [PORT]

PORT=${1:-5555}

timeout 30s adb wait-for-device >/dev/null

# Try to get device IPv4 address (prefer wlan0)
IP=$(adb shell 'ip -o -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' || getprop dhcp.wlan0.ipaddress || ip -o -4 addr show wlan0 2>/dev/null | awk '{print $4}' | cut -d/ -f1' | tr -d '\r\n' || true)
if [ -z "$IP" ]; then
  # fallback: try rmnet/wwan
  IP=$(adb shell 'ip -o -4 addr show | awk "/wlan|rmnet|wwan/ {print \$4}" | head -n1 | cut -d/ -f1' | tr -d '\r\n' || true)
fi

if [ -z "$IP" ]; then
  echo "ERR: could not determine device IP" >&2
  exit 1
fi

adb tcpip "$PORT" >/dev/null
sleep 1
adb connect "$IP:$PORT" || true

echo "$IP:$PORT"

