#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import hashlib
import json
import logging
import os
import stat
import struct
import subprocess
import time
import traceback
import requests
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol
from urllib.request import urlopen

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.hw import Paths
from openpilot.common.spinner import Spinner
from openpilot.system.version import is_prebuilt
from openpilot.sunnypilot.mapd import MAPD_PATH, MAPD_BIN_DIR
import openpilot.system.sentry as sentry


class _StatusReporter(Protocol):
  def update(self, msg: str) -> None: ...
  def close(self) -> None: ...


class _CloudlogStatusReporter:
  """Daemon-friendly stand-in for Spinner. Writes status to cloudlog so the
  install flow can run from mapd_manager without a UI spinner attached."""
  def update(self, msg: str) -> None:
    cloudlog.info(f"mapd_installer: {msg}")
  def close(self) -> None:
    pass

@dataclass(frozen=True)
class MapdReleaseSpec:
  version: str
  release_id: str
  build_id: str
  sha256: str
  capability: str
  binary_url: str


# Immutable Linux ARM64 release identity. A new artifact must use a new version
# and source build ID; never replace the asset behind this tuple.
DEFAULT_VERSION = 'chauffeur-whole-curve-v3'
DEFAULT_BINARY_URL_TEMPLATE = "https://github.com/chriscarlo/mapd/releases/download/{version}/mapd"
VERSION = DEFAULT_VERSION
MAP_WHOLE_CURVE_CAPABILITY = "MapWholeCurveProfile:whole-curve-v3"
DEFAULT_RELEASE = MapdReleaseSpec(
  version=DEFAULT_VERSION,
  release_id=DEFAULT_VERSION,
  build_id="tree-fc148f05d6574ff4c114",
  sha256="6c6911a90722a0defe28263798272b515c9a027ccd23775095d895e2969532f8",
  capability=MAP_WHOLE_CURVE_CAPABILITY,
  binary_url=DEFAULT_BINARY_URL_TEMPLATE.format(version=DEFAULT_VERSION),
)
ELFCLASS64 = 2
ELFDATA2LSB = 1
EM_AARCH64 = 183
_PERSISTENT_BINARY_CACHE_DIR = "binaries"


def _clean_override(raw_value: str | bytes | None) -> str:
  if raw_value is None:
    return ""
  if isinstance(raw_value, bytes):
    raw_value = raw_value.decode('utf-8', errors='ignore')
  return str(raw_value).strip()


def get_target_version(params: Params | None = None) -> str:
  params = params or Params()
  return _clean_override(os.getenv("SP_MAPD_RELEASE_VERSION")) or \
    _clean_override(params.get("MapdReleaseVersion")) or DEFAULT_VERSION


def get_target_binary_url(version: str, params: Params | None = None) -> str:
  params = params or Params()
  override = _clean_override(os.getenv("SP_MAPD_BINARY_URL")) or _clean_override(params.get("MapdBinaryUrl"))
  if override:
    return override.format(version=version)
  return DEFAULT_BINARY_URL_TEMPLATE.format(version=version)


def _validate_release_spec(release: MapdReleaseSpec) -> None:
  if not release.version or not release.release_id or not release.build_id or not release.capability:
    raise ValueError("mapd release identity is incomplete")
  if (len(release.sha256) != 64 or release.sha256 != release.sha256.lower() or
      any(ch not in "0123456789abcdef" for ch in release.sha256)):
    raise ValueError(f"mapd release {release.version!r} has an invalid SHA-256")
  if not release.binary_url:
    raise ValueError(f"mapd release {release.version!r} has no binary URL")
  if release.sha256 == "0" * 64 or release.build_id.startswith("REPLACE_WITH_"):
    raise ValueError(f"mapd release {release.version!r} has not been bound to a production artifact")


def get_release_spec(version: str) -> MapdReleaseSpec:
  if version != DEFAULT_RELEASE.version:
    raise ValueError(f"untrusted mapd release version {version!r}; no immutable release spec is registered")
  _validate_release_spec(DEFAULT_RELEASE)
  return DEFAULT_RELEASE


def get_target_release(params: Params | None = None) -> MapdReleaseSpec:
  params = params or Params()
  version = get_target_version(params)
  release = get_release_spec(version)
  binary_url = get_target_binary_url(version, params)
  return replace(release, binary_url=binary_url)


def update_installed_version(version: str, params: Params = None) -> None:
  if params is None:
    params = Params()

  params.put("MapdVersion", version)


def get_persistent_binary_cache_path(release: MapdReleaseSpec | str) -> str:
  """Return an identity-keyed path outside the update-swapped checkout."""
  release_spec = get_release_spec(release) if isinstance(release, str) else release
  _validate_release_spec(release_spec)
  version_key = hashlib.sha256(release_spec.version.encode("utf-8")).hexdigest()[:16]
  return os.path.join(
    Paths.mapd_root(),
    _PERSISTENT_BINARY_CACHE_DIR,
    f"mapd-{version_key}-{release_spec.sha256[:16]}",
  )


class MapdInstallManager:
  def __init__(self, spinner_ref: _StatusReporter, params: Params | None = None):
    self._spinner = spinner_ref
    self._params = params or Params()

  def download(self) -> None:
    self.ensure_directories_exist()
    release = get_target_release(self._params)
    self._download_file(release.binary_url, release)
    # Only commit MapdVersion AFTER the download has fully succeeded and the
    # binary on disk passes integrity checks. Otherwise a silent download
    # failure (network, 404, truncated tarball) would leave
    # MapdVersion=<target> with no binary, and download_needed() would
    # return False forever — the exact state that silently kills MapCurvatures
    # and takes the VTSC HUD with it.
    self._verify_installed_binary(MAPD_PATH, release)
    try:
      self._cache_installed_binary(release)
    except OSError as exc:
      # The active binary is already exact and atomically installed. Persistent
      # cache refresh is important for offline update recovery, but a storage
      # failure must not leave MapdVersion falsely stale and force a download
      # loop on every boot.
      self._spinner.update(f"Could not refresh persistent mapd cache: {exc}")
    update_installed_version(release.version, self._params)

  @staticmethod
  def _required_binary_markers(release: MapdReleaseSpec) -> tuple[bytes, ...]:
    return (
      f"MapdReleaseID:{release.release_id}".encode(),
      f"MapdBuildID:{release.build_id}".encode(),
      release.capability.encode("utf-8"),
    )

  @staticmethod
  def _verify_installed_binary(path: str, release: MapdReleaseSpec | None = None) -> None:
    """Verify the exact immutable Linux ARM64 mapd release without running it."""
    release = release or get_release_spec(DEFAULT_VERSION)
    _validate_release_spec(release)
    if not os.path.exists(path):
      raise FileNotFoundError(f"mapd binary not found at {path} after download")
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
      raise OSError(f"mapd binary at {path} is not a regular file")
    # Static arm64 mapd is ~9 MB; anything under 1 MB is a truncated or
    # HTML error page written in place of the binary.
    if st.st_size < 1_000_000:
      raise OSError(f"mapd binary at {path} is too small ({st.st_size} bytes); likely a truncated or HTML error response")
    if not (st.st_mode & stat.S_IEXEC):
      raise OSError(f"mapd binary at {path} is not executable")
    with open(path, 'rb') as fp:
      content = fp.read()
    if content[:4] != b'\x7fELF':
      raise OSError(f"mapd binary at {path} is not an ELF executable (magic={content[:4]!r})")
    if len(content) < 20 or content[4] != ELFCLASS64 or content[5] != ELFDATA2LSB:
      raise OSError(f"mapd binary at {path} is not a little-endian ELF64 executable")
    machine = struct.unpack_from("<H", content, 18)[0]
    if machine != EM_AARCH64:
      raise OSError(f"mapd binary at {path} has ELF machine {machine}, expected Linux ARM64 ({EM_AARCH64})")
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != release.sha256:
      raise OSError(
        f"mapd binary at {path} has SHA-256 {actual_sha256}, expected {release.sha256} for {release.version}"
      )
    missing_markers = [
      marker.decode('utf-8')
      for marker in MapdInstallManager._required_binary_markers(release)
      if marker not in content
    ]
    if missing_markers:
      raise OSError(f"mapd binary at {path} is missing immutable identity/capability markers: {', '.join(missing_markers)}")

  @staticmethod
  def _verify_runtime_build_info(path: str, release: MapdReleaseSpec | None = None) -> dict:
    """Execute the already hash-verified ARM64 binary and verify its identity JSON."""
    release = release or get_release_spec(DEFAULT_VERSION)
    _validate_release_spec(release)
    try:
      result = subprocess.run(
        [path, "--build-info"],
        check=False,
        capture_output=True,
        text=True,
        timeout=3.0,
      )
    except (OSError, subprocess.SubprocessError) as exc:
      raise OSError(f"could not execute mapd --build-info: {exc}") from exc
    if result.returncode != 0:
      raise OSError(f"mapd --build-info failed with exit {result.returncode}: {result.stderr.strip()}")
    try:
      info = json.loads(result.stdout)
    except Exception as exc:
      raise OSError("mapd --build-info did not return valid JSON") from exc
    if not isinstance(info, dict):
      raise OSError("mapd --build-info did not return an object")
    if info.get("releaseID") != release.release_id:
      raise OSError(f"mapd release ID {info.get('releaseID')!r} != expected {release.release_id!r}")
    if info.get("buildID") != release.build_id:
      raise OSError(f"mapd build ID {info.get('buildID')!r} != expected {release.build_id!r}")
    capabilities = info.get("capabilities")
    if not isinstance(capabilities, list) or not all(isinstance(capability, str) for capability in capabilities):
      raise OSError("mapd capabilities are malformed")
    if release.capability not in capabilities:
      raise OSError(f"mapd is missing required capability {release.capability!r}")
    return info

  def check_and_download(self) -> None:
    if self.download_needed():
      self.download()

  def download_needed(self) -> bool:
    release = get_target_release(self._params)
    try:
      self._verify_installed_binary(MAPD_PATH, release)
    except (FileNotFoundError, OSError):
      return True
    return self.get_installed_version() != release.version

  def _copy_verified_binary(self, source: str, destination: str, release: MapdReleaseSpec) -> None:
    self._verify_installed_binary(source, release)
    destination_path = Path(destination)
    temp_path = destination_path.with_name(destination_path.name + ".tmp")
    if temp_path.exists():
      temp_path.unlink()
    self._safe_write_and_set_executable(temp_path, Path(source).read_bytes())
    self._verify_installed_binary(str(temp_path), release)
    temp_path.replace(destination_path)
    self._fsync_directory(destination_path.parent)

  def _cache_installed_binary(self, release: MapdReleaseSpec) -> None:
    cache_path = get_persistent_binary_cache_path(release)
    try:
      self._verify_installed_binary(cache_path, release)
      return
    except (FileNotFoundError, OSError):
      pass
    self._copy_verified_binary(MAPD_PATH, cache_path, release)

  def restore_cached_binary(self, release: MapdReleaseSpec) -> bool:
    cache_path = get_persistent_binary_cache_path(release)
    try:
      self._copy_verified_binary(cache_path, MAPD_PATH, release)
      self._spinner.update(f"Restored mapd [{release.version}] from persistent cache.")
      return True
    except (FileNotFoundError, OSError):
      return False

  @staticmethod
  def ensure_directories_exist() -> None:
    if not os.path.exists(Paths.mapd_root()):
      os.makedirs(Paths.mapd_root())
    if not os.path.exists(MAPD_BIN_DIR):
      os.makedirs(MAPD_BIN_DIR)
    cache_dir = os.path.join(Paths.mapd_root(), _PERSISTENT_BINARY_CACHE_DIR)
    if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)

  @staticmethod
  def _safe_write_and_set_executable(file_path: Path, content: bytes) -> None:
    with open(file_path, 'wb') as output:
      output.write(content)
      output.flush()
      os.fsync(output.fileno())
    current_permissions = stat.S_IMODE(os.lstat(file_path).st_mode)
    os.chmod(file_path, current_permissions | stat.S_IEXEC)

  @staticmethod
  def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)

  def _download_file(self, url: str, release: MapdReleaseSpec, num_retries=5) -> None:
    """Download, fully verify, then atomically activate an immutable release."""
    _validate_release_spec(release)
    temp_file = Path(MAPD_PATH + ".tmp")
    download_timeout = 60
    last_exception: Exception | None = None
    for cnt in range(num_retries):
      try:
        if temp_file.exists():
          temp_file.unlink()
        response = requests.get(url, stream=True, timeout=download_timeout)
        response.raise_for_status()
        self._safe_write_and_set_executable(temp_file, response.content)
        # The active binary remains untouched until every static identity check
        # passes. This preserves a known-good release on a corrupt/HTML/foreign
        # architecture response.
        self._verify_installed_binary(str(temp_file), release)
        temp_file.replace(MAPD_PATH)
        self._fsync_directory(temp_file.parent)
        return
      except requests.exceptions.ReadTimeout as e:
        last_exception = e
        self._spinner.update(f"ReadTimeout caught. Timeout is [{download_timeout}]. Retrying download... [{cnt}]")
        time.sleep(0.5)
      except requests.exceptions.RequestException as e:
        last_exception = e
        self._spinner.update(f"RequestException caught: {e}. Retrying download... [{cnt}]")
        time.sleep(0.5)
      except (OSError, ValueError) as e:
        last_exception = e
        self._spinner.update(f"Downloaded mapd failed immutable release verification: {e}. Retrying... [{cnt}]")
        time.sleep(0.5)

    # Delete temp file if the process was not successful.
    if temp_file.exists():
      temp_file.unlink()
    logging.error("Failed to download mapd binary from %s after %d retries", url, num_retries)
    raise RuntimeError(f"mapd binary download from {url} failed after {num_retries} retries") from last_exception

  def get_installed_version(self) -> str:
    return _clean_override(self._params.get("MapdVersion"))

  def wait_for_internet_connection(self, return_on_failure: bool = False) -> bool:
    max_retries = 10
    for retries in range(max_retries + 1):
      self._spinner.update(f"Waiting for internet connection... [{retries}/{max_retries}]")
      time.sleep(2)
      try:
        _ = urlopen('https://sentry.io', timeout=10)
        return True
      except Exception as e:
        print(f'Wait for internet failed: {e}')
        if return_on_failure and retries == max_retries:
          return False

    return False

  def non_prebuilt_install(self) -> None:
    sm = messaging.SubMaster(['deviceState'])
    metered = sm['deviceState'].networkMetered

    if metered:
      self._spinner.update("Can't proceed with mapd install since network is metered!")
      time.sleep(5)
      return

    try:
      self.ensure_directories_exist()
      if not self.download_needed():
        self._spinner.update("Mapd is good!")
        time.sleep(0.1)
        return

      if self.wait_for_internet_connection(return_on_failure=True):
        release = get_target_release(self._params)
        self._spinner.update(
          f"Downloading mapd [{self.get_installed_version()}] => [{release.version}] from [{release.binary_url}]."
        )
        time.sleep(0.1)
        self.check_and_download()
      self._spinner.close()

    except Exception:
      for i in range(6):
        self._spinner.update("Failed to download OSM maps won't work until properly downloaded!" +
                             "Try again manually rebooting. " +
                             f"Boot will continue in {5 - i}s...")
        time.sleep(1)

      sentry.init(sentry.SentryProject.SELFDRIVE)
      traceback.print_exc()
      sentry.capture_exception()


def ensure_mapd_installed(params: Params | None = None,
                          reporter: _StatusReporter | None = None) -> bool:
  """Idempotent mapd-binary install with integrity verification, safe to call
  from a long-running daemon. Returns True if the binary is present and valid
  after this call, False otherwise. Does NOT block forever on network failure
  — single best-effort pass, retries happen on subsequent invocations.

  Contract (why this exists as a standalone entry point):
    - The __main__ block of this file is not wired into the normal boot flow;
      `mapd_manager` is the daemon that actually runs every boot. If it used
      to stamp MapdVersion without downloading (historical sunnypilot
      inheritance), the device could end up with version-set-but-no-binary
      and the native mapd process would crash-loop silently.
    - This function is the single source of truth for "get mapd onto disk";
      it is safe to call unconditionally on every boot.
  """
  params = params or Params()
  reporter = reporter or _CloudlogStatusReporter()
  manager = MapdInstallManager(reporter, params)
  manager.ensure_directories_exist()
  try:
    release = get_target_release(params)
  except (ValueError, KeyError) as exc:
    reporter.update(f"mapd release configuration rejected: {exc}")
    return False
  installed_version = _clean_override(params.get("MapdVersion"))

  try:
    manager._verify_installed_binary(MAPD_PATH, release)
    try:
      manager._cache_installed_binary(release)
    except OSError as e:
      reporter.update(f"Could not refresh persistent mapd cache: {e}")
    if installed_version != release.version:
      reporter.update(
        f"Exact mapd release is installed; repairing MapdVersion [{installed_version or 'unset'}] => [{release.version}]."
      )
      update_installed_version(release.version, params)
    return True
  except (FileNotFoundError, OSError):
    pass  # Fall through to download path.

  # Updates deliberately clean ignored files from the checkout, including the
  # release binary. Restore the last verified copy from persistent OSM storage
  # before consulting network state so an offline boot still starts mapd.
  if manager.restore_cached_binary(release):
    update_installed_version(release.version, params)
    return True

  # deviceState may not be available super-early in boot; tolerate the SubMaster
  # attempt and fall back to "not metered".
  try:
    sm = messaging.SubMaster(['deviceState'])
    sm.update(0)
    metered = bool(sm['deviceState'].networkMetered)
  except Exception:
    metered = False
  if metered:
    reporter.update("Skipping mapd install: network is metered.")
    return False

  try:
    reporter.update(
      f"Downloading mapd [{manager.get_installed_version()}] => [{release.version}] from [{release.binary_url}]."
    )
    manager.download()
    return True
  except Exception as e:
    reporter.update(f"mapd install failed: {e}. Will retry on next boot.")
    try:
      sentry.init(sentry.SentryProject.SELFDRIVE)
      sentry.capture_exception()
    except Exception:
      pass
    return False


if __name__ == "__main__":
  spinner = Spinner()
  spinner.update(f"Checking if mapd is installed and valid. Prebuilt [{is_prebuilt()}]")
  ensure_mapd_installed(reporter=spinner)
