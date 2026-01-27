#!/bin/sh
set -ex
echo "[bt_raw_test] Starting on $(date)"

# Step A: power cycle via rfkill
rfkill block bluetooth || true
sleep 1
rfkill unblock bluetooth || true
sleep 1

# Step B/C: wake pulse then HCI Reset (01 03 0C 00)
printf "\xFD" > /dev/ttyHS0 || true
sleep 1
printf "\x01\x03\x0C\x00" > /dev/ttyHS0 || true

# Step D: read a little back to see any event bytes
echo "[bt_raw_test] Reading RX for 2s..."
timeout 2s dd if=/dev/ttyHS0 bs=1 count=64 2>/dev/null | od -An -tx1 -v || true

# Step E: try btattach for 12s and then dump dmesg indicators
echo "[bt_raw_test] Running btattach for 12s..."
timeout 12s btattach -B /dev/ttyHS0 -S 115200 -P qca || true
echo "[bt_raw_test] Dmesg tail (filtered):"
dmesg | grep -E "ttyHS0|Bluetooth: hci0|QCA|ROME|Downloading|crbtfw|crnv|Failed|tx timeout|Frame reassembly|bt_power" | tail -n 300 || true

echo "[bt_raw_test] Done"
