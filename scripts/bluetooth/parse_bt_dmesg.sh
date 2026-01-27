#!/usr/bin/env bash
set -euo pipefail

# Parse dmesg lines for QCA bring-up milestones and print a compact summary.

IN=${1:-}
if [[ -n "$IN" && -f "$IN" ]]; then
  SRC=(cat "$IN")
else
  SRC=(dmesg)
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

"${SRC[@]}" | rg -n "Bluetooth|hci_qca|btqca|qca|ROME|hci0|baud|firmware" | tail -n 2000 > "$tmp" || true

passages=(
  "qca.*Power pulse" \
  "qca.*PATCH_VER" \
  "qca.*Firmware file" \
  "qca.*Downloading" \
  "qca.*NVM" \
  "hci0: command tx timeout" \
  "hci0: Reset failed" \
  "HCI reset" \
  "qca.*baud.*3.2|qca.*3200000|qca.*3000000" \
  "Bluetooth: hci0: QCA controller" \
)

echo "== QCA Bring-up Summary =="
for pat in "${passages[@]}"; do
  rg -n "$pat" "$tmp" | tail -n 3 || true
done

echo
echo "== hci0 existence =="
if ls /sys/class/bluetooth/hci0 >/dev/null 2>&1; then
  echo "hci0 present"
else
  echo "hci0 missing"
fi

echo
echo "== hciconfig -a (head) =="
hciconfig -a 2>/dev/null | head -n 40 || true

