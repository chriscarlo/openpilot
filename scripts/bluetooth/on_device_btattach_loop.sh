#!/system/bin/sh
# Lightweight attach loop on device: tries bounded btattach attempts and logs dmesg.

MON=/data/local/tmp/mon
mkdir -p "$MON"

echo "BTATTACH LOOP START $(date)" >> "$MON/attach_loop.log"

while true; do
  # If hci0 is present, wait a bit and continue
  if [ -e /sys/class/bluetooth/hci0 ]; then
    sleep 10
    continue
  fi

  # Ensure firmware bind and patch mode flag (keep TLV/NVM at 115200 by default)
  mount --bind /data/firmware /lib/firmware 2>/dev/null
  echo Y > /sys/module/hci_uart/parameters/patch115200 2>/dev/null

  echo "$(date) btattach attempt (with OFF/ON pulses)" >> "$MON/attach_loop.log"

  # rfkill pulse to ensure a clean state
  rfkill block bluetooth 2>/dev/null || true
  sleep 0.1
  rfkill unblock bluetooth 2>/dev/null || true

  # OFF pulse: 0xC0 @ 2400, flow control disabled
  stty -F /dev/ttyHS0 2400 -echo -crtscts cs8 -parenb -cstopb 2>/dev/null || stty 2400 -echo -crtscts cs8 -parenb -cstopb < /dev/ttyHS0
  printf "\xC0" > /dev/ttyHS0 2>/dev/null || true
  usleep 10000 2>/dev/null || sleep 0.02

  # ON pulse: 0xFC @ 115200, flow control disabled then re-enable
  stty -F /dev/ttyHS0 115200 -echo -crtscts cs8 -parenb -cstopb 2>/dev/null || stty 115200 -echo -crtscts cs8 -parenb -cstopb < /dev/ttyHS0
  printf "\xFC" > /dev/ttyHS0 2>/dev/null || true
  sleep 0.10
  stty -F /dev/ttyHS0 115200 -echo crtscts cs8 -parenb -cstopb 2>/dev/null || stty 115200 -echo crtscts cs8 -parenb -cstopb < /dev/ttyHS0

  # Run bounded attach (15s) and capture output
  timeout 15s btattach -B /dev/ttyHS0 -P qca -S 115200 >> "$MON/btattach_loop.out" 2>&1
  # Short settle
  sleep 5
done
