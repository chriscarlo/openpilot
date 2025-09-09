#!/usr/bin/env bash
# Simple SSH-over-ADB launcher that reuses your existing commaHome/commaCar SSH identity
# Usage:
#   ./sshCommaAdb.sh [extra ssh args]
#
# Behavior:
# - Ensures ADB server is running and forwards tcp:2222 -> device:22
# - Reuses IdentityFile/CertificateFile from your existing ssh profiles (commaHome or commaCar)
# - Connects to comma@127.0.0.1:2222

set -euo pipefail

log() { printf "[%s] %s\n" "sshCommaAdb" "$*"; }
err() { printf "[%s][ERROR] %s\n" "sshCommaAdb" "$*" 1>&2; }

# 1) Pick a base SSH profile to copy identity settings from
choose_profile() {
  local candidates
  if [[ -n "${COMMA_SSH_PROFILE:-}" ]]; then
    IFS=',' read -r -a candidates <<<"${COMMA_SSH_PROFILE}"
  else
    candidates=(commaHome commaCar)
  fi
  for p in "${candidates[@]}"; do
    if ssh -G "$p" >/dev/null 2>&1; then
      echo "$p"
      return 0
    fi
  done
  echo ""  # none
}

PROFILE=$(choose_profile)
if [[ -z "$PROFILE" ]]; then
  log "No ssh profile 'commaHome' or 'commaCar' found. Falling back to default keys."
fi

# 2) Resolve IdentityFile / CertificateFile
resolve_from_profile() {
  local profile="$1" key cert
  key=$(ssh -G "$profile" 2>/dev/null | awk '/^identityfile /{print $2; exit}') || true
  cert=$(ssh -G "$profile" 2>/dev/null | awk '/^certificatefile /{print $2; exit}') || true
  [[ -n "$key" && -f "$key" ]] || key=""
  [[ -n "$cert" && -f "$cert" ]] || cert=""
  printf "%s\n%s\n" "$key" "$cert"
}

IDENTITY="${COMMA_SSH_IDENTITY:-}"; CERTFILE="${COMMA_SSH_CERT:-}"
if [[ -n "$PROFILE" ]]; then
  mapfile -t resolved < <(resolve_from_profile "$PROFILE")
  [[ -z "$IDENTITY" ]] && IDENTITY="${resolved[0]:-}"
  [[ -z "$CERTFILE" ]] && CERTFILE="${resolved[1]:-}"
fi

# If neither profile nor explicit env provided an identity, allow agent/defaults to decide.

# 3) Ensure ADB forward is in place (unless skipped)
if [[ "${SKIP_ADB_FORWARD:-}" != "1" ]]; then
  if ! command -v adb >/dev/null 2>&1; then
    err "adb not found. Install android-tools-adb and ensure the device is attached to WSL via usbipd."
    exit 3
  fi
  # Start server, ensure we see at least one device (may show 'device' or 'unauthorized' briefly)
  adb start-server >/dev/null || true
  if ! adb devices | awk 'NR>1 && $2 ~ /device|unauthorized|unknown/ {found=1} END{exit !found}'; then
    log "No ADB device detected yet. Continuing; forward may fail until it appears."
  fi
  adb forward tcp:2222 tcp:22 >/dev/null || true
fi

# 4) Build ssh args and connect
SSH_ARGS=( -p 2222 -o StrictHostKeyChecking=accept-new )
if [[ -n "$IDENTITY" ]]; then
  SSH_ARGS+=( -o IdentitiesOnly=yes -i "$IDENTITY" )
  [[ -n "$CERTFILE" ]] && SSH_ARGS+=( -o "CertificateFile=$CERTFILE" )
else
  # No explicit identity; prefer agent
  SSH_ARGS+=( -A )
fi

# Allow user to pass through extra ssh options/command

# If no identity set, try agent first with a quick test
if [[ -z "${IDENTITY:-}" ]]; then
  if ssh -o BatchMode=yes -o ConnectTimeout=3 "${SSH_ARGS[@]}" -q comma@127.0.0.1 exit 2>/dev/null; then
    exec ssh "${SSH_ARGS[@]}" comma@127.0.0.1 "$@"
  fi
  # Try candidate private keys under ~/.ssh (skip .pub/.ppk)
  for k in "$HOME"/.ssh/id_*; do
    [[ -f "$k" ]] || continue
    [[ "$k" == *.pub || "$k" == *.ppk ]] && continue
    if ssh -o BatchMode=yes -o ConnectTimeout=3 -o IdentitiesOnly=yes -i "$k" "${SSH_ARGS[@]}" -q comma@127.0.0.1 exit 2>/dev/null; then
      SSH_ARGS+=( -o IdentitiesOnly=yes -i "$k" )
      exec ssh "${SSH_ARGS[@]}" comma@127.0.0.1 "$@"
    fi
  done
  # Fall back to plain ssh (may prompt if a passphrase is needed)
  exec ssh "${SSH_ARGS[@]}" comma@127.0.0.1 "$@"
else
  exec ssh "${SSH_ARGS[@]}" comma@127.0.0.1 "$@"
fi
