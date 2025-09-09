#!/usr/bin/env bash
# Install or update a Host commaAdb stanza in your SSH config by mirroring
# IdentityFile/CertificateFile from an existing profile (commaHome/commaCar).
#
# Usage:
#   ./installCommaAdbProfile.sh [-p commaHome|commaCar] [-t /path/to/ssh_config] [-f]
#
# Defaults:
#   -p auto (prefer commaHome, then commaCar)
#   -t ~/.ssh/config
#   -f (force replace existing COMMA-ADB block)

set -euo pipefail

log() { printf "[%s] %s\n" "commaAdbInstaller" "$*"; }
err() { printf "[%s][ERROR] %s\n" "commaAdbInstaller" "$*" 1>&2; }

PROFILE=""
TARGET="${HOME}/.ssh/config"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--profile) PROFILE="$2"; shift 2;;
    -t|--target) TARGET="$2"; shift 2;;
    -f|--force) FORCE=1; shift;;
    -h|--help)
      cat <<EOF
Install or update Host commaAdb in your SSH config using identity from commaHome/commaCar.

Usage: $0 [-p commaHome|commaCar] [-t /path/to/ssh_config] [-f]
  -p, --profile   Source profile to mirror (default: auto-detect commaHome, then commaCar)
  -t, --target    SSH config path (default: ~/.ssh/config)
  -f, --force     Replace any existing COMMA-ADB block if present
EOF
      exit 0;;
    *) err "Unknown argument: $1"; exit 2;;
  esac
done

choose_profile() {
  local candidates
  if [[ -n "$PROFILE" ]]; then
    candidates=("$PROFILE")
  else
    candidates=(commaHome commaCar)
  fi
  for p in "${candidates[@]}"; do
    if ssh -G "$p" >/dev/null 2>&1; then
      echo "$p"; return 0
    fi
  done
  echo ""; return 1
}

SRC_PROFILE=$(choose_profile || true)
if [[ -z "$SRC_PROFILE" ]]; then
  err "No source profile found (commaHome/commaCar). Provide -p or create one in ~/.ssh/config."
  exit 3
fi

log "Using source profile: $SRC_PROFILE"

# Resolve identity and certificate from ssh -G (first entries)
mapfile -t ALL_IDS < <(ssh -G "$SRC_PROFILE" | awk '/^identityfile /{print $2}') || true
SRC_IDENTITY=""
for cand in "${ALL_IDS[@]}"; do
  [[ -f "$cand" ]] && SRC_IDENTITY="$cand" && break
done
# Do NOT fallback to generic keys here; only use identity if sourced from the profile.

# Try to obtain certificate from ssh -G; if not present, check adjacent -cert.pub next to the chosen key
SRC_CERT=$(ssh -G "$SRC_PROFILE" | awk '/^certificatefile /{print $2; exit}') || true
if [[ -z "$SRC_CERT" ]]; then
  if [[ -f "${SRC_IDENTITY}-cert.pub" ]]; then
    SRC_CERT="${SRC_IDENTITY}-cert.pub"
  fi
fi
if [[ -n "$SRC_CERT" ]] && [[ ! -f "$SRC_CERT" ]]; then
  log "CertificateFile reported but not found; ignoring: $SRC_CERT"
  SRC_CERT=""
fi

mkdir -p "$(dirname "$TARGET")"
touch "$TARGET"

# Prepare the block content
BLOCK_BEGIN="# >>> COMMA-ADB START (managed by installCommaAdbProfile.sh)"
BLOCK_END="# <<< COMMA-ADB END"
{
  printf "%s\n" "$BLOCK_BEGIN"
  printf "Host commaAdb\n"
  printf "  HostName 127.0.0.1\n"
  printf "  Port 2222\n"
  printf "  User comma\n"
  printf "  StrictHostKeyChecking accept-new\n"
  if [[ -n "$SRC_IDENTITY" ]]; then
    printf "  IdentitiesOnly yes\n"
    printf "  IdentityFile \"%s\"\n" "$SRC_IDENTITY"
  fi
  if [[ -n "$SRC_CERT" ]]; then
    printf "  CertificateFile \"%s\"\n" "$SRC_CERT"
  fi
  printf "%s\n" "$BLOCK_END"
} >"${BLOCK_TMP:=/tmp/comma-adb-block}"
NEW_BLOCK=$(cat "$BLOCK_TMP")

# If a managed block exists, replace it; else append unless an unmanaged Host commaAdb exists
TMP=$(mktemp)
MANAGED=0
if rg -n "^$(printf %q "$BLOCK_BEGIN")$" "$TARGET" >/dev/null 2>&1; then
  MANAGED=1
fi

if [[ $MANAGED -eq 1 ]]; then
  if [[ $FORCE -ne 1 ]]; then
    log "Updating existing managed COMMA-ADB block in $TARGET"
  else
    log "Forcing replacement of managed COMMA-ADB block in $TARGET"
  fi
  awk -v start="$BLOCK_BEGIN" -v end="$BLOCK_END" -v repl="$NEW_BLOCK" '
    BEGIN{printed=0}
    { if (!printed) {
        if ($0==start) { inblock=1; print repl; printed=1; next }
      }
      if (inblock && $0==end) { inblock=0; next }
      if (!inblock) print $0
  }' "$TARGET" > "$TMP"
  mv "$TMP" "$TARGET"
else
  # Detect any Host commaAdb stanza (unmanaged)
  if rg -n "^Host[[:space:]]+commaAdb(\s|$)" "$TARGET" >/dev/null 2>&1; then
    if [[ $FORCE -eq 1 ]]; then
      log "Found existing Host commaAdb (unmanaged). Replacing due to --force."
      # Remove existing Host commaAdb block (until next Host or EOF), then append managed block
      awk '/^Host[[:space:]]+commaAdb(\s|$)/{skip=1} /^Host[[:space:]]+/{if(skip){skip=0}} {if(!skip)print}' "$TARGET" > "$TMP"
      printf "\n%s\n" "$NEW_BLOCK" >> "$TMP"
      mv "$TMP" "$TARGET"
    else
      err "Host commaAdb already exists in $TARGET. Re-run with -f to replace with managed block."
      rm -f "$TMP"
      exit 5
    fi
  else
    log "Appending managed COMMA-ADB block to $TARGET"
    printf "\n%s\n" "$NEW_BLOCK" >> "$TARGET"
  fi
fi

chmod 600 "$TARGET" || true

log "Installed/updated Host commaAdb. Preview:" 
ssh -G commaAdb | rg -n '^(hostname|port|user|identityfile|certificatefile|identitiesonly|stricthostkeychecking) '

log "Done. Connect with: ssh commaAdb"
