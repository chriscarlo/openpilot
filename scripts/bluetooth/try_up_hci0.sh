#!/usr/bin/env bash
set -euo pipefail

TRIES=${TRIES:-10}
SLEEP=${SLEEP:-1}

echo "[hci_up] trying to bring up hci0 (${TRIES} tries)" >&2
for i in $(seq 1 "$TRIES"); do
  if hciconfig hci0 up 2>/dev/null; then
    echo "[hci_up] hci0 up (try $i)" >&2
    exit 0
  fi
  sleep "$SLEEP"
done
echo "[hci_up] failed to bring up hci0 after ${TRIES} tries" >&2
exit 1

