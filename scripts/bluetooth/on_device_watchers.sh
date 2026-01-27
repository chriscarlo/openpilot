#!/system/bin/sh
# Start lightweight watchers that persist across adb disconnects.
# Writes into /data/local/tmp/mon; safe to call multiple times.

set -e
MON=/data/local/tmp/mon
mkdir -p "$MON"
chmod 777 "$MON"

# Start a background dmesg follower if not running
if ! ps | grep -v grep | grep -q "dmesg -w"; then
  /system/bin/sh -c "setsid dmesg -w >> $MON/dmesg_w.txt 2>&1 & echo \$! > $MON/dmesg_w.pid" >/dev/null 2>&1 || true
fi

# Start swaglog tail if not running
if ! ps | grep -v grep | grep -q "tail -F /data/log/swaglog"; then
  /system/bin/sh -c "setsid tail -F /data/log/swaglog.* >> $MON/swag_tail.txt 2>&1 & echo \$! > $MON/swag_tail.pid" >/dev/null 2>&1 || true
fi

# Stamp basic context
date > $MON/last_start.txt 2>&1
uname -a > $MON/uname.txt 2>&1
cat /proc/uptime > $MON/uptime_start.txt 2>&1

exit 0

