#!/usr/bin/env bash
# Lightweight wrapper to ensure every adb call has a host-side timeout.
# Usage examples:
#   scripts/dev/adb_safe.sh shell su -c "dmesg | tail -n 3000"
#   scripts/dev/adb_safe.sh -t 90 shell su -c "dmesg | tail -n 30000"
#   scripts/dev/adb_safe.sh pull /data/local/tmp/btattach.out .cache/adb/btattach.out
#   scripts/dev/adb_safe.sh -s e521630c shell su -c "rfkill list"

set -euo pipefail

DEFAULT_PUSH_PULL=30   # seconds
DEFAULT_SHELL=20       # seconds
DEFAULT_OTHER=20       # seconds
DEFAULT_LONG=90        # for heavy shell reads like large dmesg/logcat tails

TIMEOUT_S=""
SERIAL_ARG=""

while getopts ":t:s:" opt; do
  case "$opt" in
    t) TIMEOUT_S="$OPTARG" ;;
    s) SERIAL_ARG="$OPTARG" ;;
    *) ;;
  esac
done
shift $((OPTIND - 1))

if ! command -v timeout >/dev/null 2>&1; then
  echo "[adb_safe] ERROR: 'timeout' not found on host PATH" >&2
  exit 127
fi
if ! command -v adb >/dev/null 2>&1; then
  echo "[adb_safe] ERROR: 'adb' not found on host PATH" >&2
  exit 127
fi

if [[ $# -lt 1 ]]; then
  echo "[adb_safe] ERROR: no adb subcommand provided" >&2
  exit 2
fi

subcmd="$1"; shift || true

# Decide default timeout based on subcommand and content
case "$subcmd" in
  pull|push)
    T=${TIMEOUT_S:-$DEFAULT_PUSH_PULL}
    ;;
  shell)
    # Inspect rest of args to detect long tails
    rest_str="$(printf '%s ' "$@")"
    if echo "$rest_str" | grep -E -q "dmesg.*tail|logcat.*-d|cat .*\/proc\/kmsg"; then
      T=${TIMEOUT_S:-$DEFAULT_LONG}
    else
      T=${TIMEOUT_S:-$DEFAULT_SHELL}
    fi
    ;;
  *)
    T=${TIMEOUT_S:-$DEFAULT_OTHER}
    ;;
esac

ADB_ARGS=()
if [[ -n "${SERIAL_ARG}" ]]; then
  ADB_ARGS+=("-s" "${SERIAL_ARG}")
fi
ADB_ARGS+=("${subcmd}")
if [[ $# -gt 0 ]]; then
  ADB_ARGS+=("$@")
fi

# Use --foreground so Ctrl-C propagates if run interactively.
exec timeout --foreground "${T}s" adb "${ADB_ARGS[@]}"

