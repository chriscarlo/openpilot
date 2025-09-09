#!/usr/bin/env bash
set -euo pipefail

# On-device, active-slot enable script for AGNOS (comma 3/3X)
# - Installs bluez, rfkill, linux-firmware noninteractively
# - Enables and starts bluetooth.service
# - Unblocks rfkill and attempts to power on controller
# - Each step is time-bounded to avoid stalls

timeout_cmd() {
  local secs="$1"; shift
  command -v timeout >/dev/null 2>&1 && timeout -k 5s "$secs" "$@" || "$@"
}

log() { echo "[bt-setup] $*"; }

require_root() {
  if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      exec sudo -E "$0" "$@"
    else
      echo "This script must run as root (sudo)" >&2
      exit 2
    fi
  fi
}

main() {
  require_root "$@"

  # Fast path: check-only
  if tools/chauffeur/bluetooth3x/check_bt_only.sh >/dev/null 2>&1; then
    log "Bluetooth already ready; exiting"
    tools/chauffeur/bluetooth3x/check_bt_only.sh
    exit 0
  fi

  # Pre-download linux-firmware if cached path provided or default cache exists
  PKGDIR=${PKGDIR:-/data/pkgcache}
  mkdir -p "$PKGDIR" || true

  # Complete any partial dpkg config
  timeout_cmd 30s dpkg --force-confdef --force-confold --configure -a || true

  # Update and install bluez/rfkill first (small)
  timeout_cmd 120s apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=20 update
  timeout_cmd 180s env DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
      bluez rfkill || true

  # Install linux-firmware from cache if present; else via apt (may be large)
  lf_deb=$(ls -1 "$PKGDIR"/linux-firmware_*.deb 2>/dev/null | head -n 1 || true)
  if [[ -n "${lf_deb}" && -f "${lf_deb}" ]]; then
    log "Installing linux-firmware from cache: ${lf_deb}"
    timeout_cmd 300s env DEBIAN_FRONTEND=noninteractive \
      apt-get install -y --no-install-recommends \
      -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
      "${lf_deb}" || true
  else
    log "Downloading linux-firmware into cache and installing"
    pushd "$PKGDIR" >/dev/null
    timeout_cmd 120s apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=20 update || true
    timeout_cmd 600s apt-get download linux-firmware || true
    lf_deb=$(ls -1 linux-firmware_*.deb 2>/dev/null | head -n 1 || true)
    if [[ -n "${lf_deb}" ]]; then
      timeout_cmd 300s env DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends \
        -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
        "${lf_deb}" || true
    fi
    popd >/dev/null
  fi

  # Enable and start service
  timeout_cmd 20s systemctl enable bluetooth || true
  timeout_cmd 20s systemctl restart bluetooth || timeout_cmd 20s systemctl start bluetooth || true

  # Unblock rfkill and attempt to expose controller
  timeout_cmd 5s rfkill unblock all || true
  timeout_cmd 10s btmgmt --index 0 info 2>/dev/null || true
  timeout_cmd 10s btmgmt --index 0 power on 2>/dev/null || true

  tools/chauffeur/bluetooth3x/check_bt_only.sh || true
}

main "$@"

