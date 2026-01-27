#!/usr/bin/env bash
set -euo pipefail

# Run a small matrix of attach windows varying TLV pacing knobs.
# Stops early if hci0 appears.

ADB=${ADB:-scripts/dev/adb_safe.sh}
TTY=${TTY:-/dev/ttyHS0}
INIT_BAUD=${INIT_BAUD:-115200}
WINDOW=${WINDOW:-60}

OUTBASE=${OUTBASE:-docs/chauffeur/bluetooth3x/diagnostics}
run_ts=$(date +%Y%m%d-%H%M%S)
MROOT="$OUTBASE/matrix_${run_ts}"
mkdir -p "$MROOT"

seg0=(20 24 32)
pre0=(20000 30000 35000)
pace=(2000 6000 12000)
skipv=(0 1)

try=0
for s in "${seg0[@]}"; do
  for p0 in "${pre0[@]}"; do
    for pc in "${pace[@]}"; do
      for sv in "${skipv[@]}"; do
        try=$((try+1))
        tag="s${s}_p${p0}_pc${pc}_sv${sv}"
        out="$MROOT/$tag"; mkdir -p "$out"
        echo "[matrix] Try #${try} ${tag}" | tee "$out/run.txt"
        TLV_ENV=("tlv_seg0_len=${s}" "tlv_preseg0_us=${p0}" "tlv_pace_us=${pc}" "tlv_full_skipvse=${sv}" "tlv_force_sync=1")
        env ${TLV_ENV[@]} ADB="$ADB" TTY="$TTY" INIT_BAUD="$INIT_BAUD" OUTBASE="$out" \
          scripts/bluetooth/strict_attach_window.sh --tty "$TTY" --baud "$INIT_BAUD" --window "$WINDOW" || true
        # Quick check for hci0
        if $ADB shell 'test -e /sys/class/bluetooth/hci0' >/dev/null 2>&1; then
          echo "[matrix] hci0 present; stopping at ${tag}" | tee -a "$out/run.txt"
          exit 0
        fi
      done
    done
  done
done

echo "[matrix] Completed all tries without hci0"
exit 1

