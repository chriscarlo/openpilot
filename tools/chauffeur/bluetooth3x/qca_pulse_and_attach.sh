#!/bin/sh
set -e

# WCN3990 UART power-pulse prelude + btattach.
# OFF pulse: 0xC0 @ 2400 (no flow), wait ~10ms
# ON  pulse: 0xFC @ INIT_BAUD (no flow), wait ~100ms, then re-enable flow control
# References: upstream qca_wcn3990_init() power-pulse sequence and timings.

DEV=${1:-/dev/ttyHS0}
INIT_BAUD=${2:-115200}
OFF_DELAY_MS=${OFF_DELAY_MS:-10}
ON_DELAY_MS=${ON_DELAY_MS:-100}

rfkill unblock bluetooth 2>/dev/null || true

# OFF pulse
stty -F "$DEV" 2400 -echo -crtscts cs8 -parenb -cstopb 2>/dev/null || stty 2400 -echo -crtscts cs8 -parenb -cstopb < "$DEV"
printf "\xC0" > "$DEV" || true
if command -v usleep >/dev/null 2>&1; then
  usleep $(( OFF_DELAY_MS * 1000 )) || true
else
  # fallback: coarse sleep
  sleep $(awk "BEGIN{printf \"%.3f\", ${OFF_DELAY_MS}/1000}") || true
fi

# ON pulse
stty -F "$DEV" "$INIT_BAUD" -echo -crtscts cs8 -parenb -cstopb 2>/dev/null || stty "$INIT_BAUD" -echo -crtscts cs8 -parenb -cstopb < "$DEV"
printf "\xFC" > "$DEV" || true
sleep $(awk "BEGIN{printf \"%.3f\", ${ON_DELAY_MS}/1000}")

# Re-enable flow control at 115200
stty -F "$DEV" "$INIT_BAUD" -echo crtscts cs8 -parenb -cstopb 2>/dev/null || stty "$INIT_BAUD" -echo crtscts cs8 -parenb -cstopb < "$DEV"

# Attach
exec timeout 45s btattach -B "$DEV" -S "$INIT_BAUD" -P qca
