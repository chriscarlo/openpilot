#!/usr/bin/env bash
set -euo pipefail

# Minimal firmware hotplug/udev helper that can serve .zst-compressed blobs
# to the kernel firmware loader via sysfs. Safe on systems without Zstd.

log() { echo "[fw-zstd-helper] $*" >&2; }

REQ_FIRMWARE=${FIRMWARE:-}
DEVPATH_ENV=${DEVPATH:-}
TIMEOUT=${FW_TIMEOUT_SEC:-120}

sysfs_path="/sys${DEVPATH_ENV:-}"  # e.g., /sys/devices/.../firmware/xyz
loading_file="${sysfs_path}/loading"
data_file="${sysfs_path}/data"

# Resolve firmware path using standard search roots. Prefer updates overlay.
roots=(
  "/lib/firmware/updates" 
  "/usr/lib/firmware/updates" 
  "/lib/firmware" 
  "/usr/lib/firmware" 
  "/firmware/image"
)

find_fw() {
  local name="$1"
  for r in "${roots[@]}"; do
    if [[ -f "${r}/${name}" ]]; then
      printf '%s\n' "${r}/${name}"
      return 0
    fi
  done
  return 1
}

# Prefer the requested name; if it is not present and lacks .zst, try adding it.
fw_path=""
if [[ -n "${REQ_FIRMWARE}" ]]; then
  fw_path=$(find_fw "${REQ_FIRMWARE}") || true
  if [[ -z "${fw_path}" && "${REQ_FIRMWARE}" != *.zst ]]; then
    fw_path=$(find_fw "${REQ_FIRMWARE}.zst") || true
  fi
fi

if [[ -z "${fw_path}" ]]; then
  log "firmware not found: ${REQ_FIRMWARE}"
  exit 1
fi

# Ensure sysfs endpoints exist
if [[ ! -w "${loading_file}" || ! -w "${data_file}" ]]; then
  log "sysfs endpoints not writable: ${loading_file} ${data_file}"
  exit 2
fi

# Serve the firmware, decompressing if .zst
serve() {
  local src="$1"
  local is_zst=0
  [[ "${src}" == *.zst ]] && is_zst=1

  # Kick off loading
  printf '1' > "${loading_file}"

  if [[ ${is_zst} -eq 1 ]]; then
    if ! command -v zstd >/dev/null 2>&1; then
      log "zstd not available to decompress ${src}"
      printf '-1' > "${loading_file}"
      return 3
    fi
    if ! zstd -dc --no-progress -- "${src}" > "${data_file}" 2>/dev/null; then
      log "zstd decompression failed for ${src}"
      printf '-1' > "${loading_file}"
      return 4
    fi
  } else
    if ! cat -- "${src}" > "${data_file}"; then
      log "copy failed for ${src}"
      printf '-1' > "${loading_file}"
      return 5
    fi
  fi

  # Signal success
  printf '0' > "${loading_file}"
  log "served ${REQ_FIRMWARE} from ${src} (zst=${is_zst})"
  return 0
}

# Apply a simple timeout to avoid hanging udev workers
(
  serve "${fw_path}"
) &
helper_pid=$!

SECONDS=0
while kill -0 "$helper_pid" 2>/dev/null; do
  if (( SECONDS > TIMEOUT )); then
    log "timeout after ${TIMEOUT}s serving ${REQ_FIRMWARE}"
    kill -TERM "$helper_pid" 2>/dev/null || true
    printf '-1' > "${loading_file}"
    exit 124
  fi
  sleep 0.2
done
wait "$helper_pid"

