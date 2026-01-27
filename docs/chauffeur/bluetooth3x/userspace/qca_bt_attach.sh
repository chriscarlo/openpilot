#!/usr/bin/env bash
set -euo pipefail

# QCA WCN3990 attach helper for comma3x (SDM845)
# - Initializes at 115200; kernel quirk bumps to 3M after PATCH_VER
# - Accepts overrides via env: TTY, INIT_BAUD, PROTO

TTY=${TTY:-/dev/ttyHS0}
INIT_BAUD=${INIT_BAUD:-115200}
PROTO=${PROTO:-qca}

if ! command -v btattach >/dev/null 2>&1; then
  echo "btattach not found" >&2
  exit 127
fi

exec btattach -B "${TTY}" -S "${INIT_BAUD}" -P "${PROTO}"

