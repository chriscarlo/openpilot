#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import logging
import os
import stat
import time
import traceback
import requests
from pathlib import Path
from urllib.request import urlopen

from cereal import messaging
from openpilot.common.params import Params
from openpilot.system.hardware.hw import Paths
from openpilot.common.spinner import Spinner
from openpilot.system.version import is_prebuilt
from openpilot.sunnypilot.mapd import MAPD_PATH, MAPD_BIN_DIR
import openpilot.system.sentry as sentry

DEFAULT_VERSION = 'v1.10.0'
DEFAULT_BINARY_URL_TEMPLATE = "https://github.com/pfeiferj/openpilot-mapd/releases/download/{version}/mapd"
VERSION = DEFAULT_VERSION


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


def update_installed_version(version: str, params: Params = None) -> None:
  if params is None:
    params = Params()

  params.put("MapdVersion", version)


class MapdInstallManager:
  def __init__(self, spinner_ref: Spinner):
    self._spinner = spinner_ref
    self._params = Params()

  def download(self) -> None:
    self.ensure_directories_exist()
    target_version = get_target_version(self._params)
    self._download_file(get_target_binary_url(target_version, self._params))
    update_installed_version(target_version, self._params)

  def check_and_download(self) -> None:
    if self.download_needed():
      self.download()

  def download_needed(self) -> bool:
    return not os.path.exists(MAPD_PATH) or self.get_installed_version() != get_target_version(self._params)

  @staticmethod
  def ensure_directories_exist() -> None:
    if not os.path.exists(Paths.mapd_root()):
      os.makedirs(Paths.mapd_root())
    if not os.path.exists(MAPD_BIN_DIR):
      os.makedirs(MAPD_BIN_DIR)

  @staticmethod
  def _safe_write_and_set_executable(file_path: Path, content: bytes) -> None:
    with open(file_path, 'wb') as output:
      output.write(content)
      output.flush()
      os.fsync(output.fileno())
    current_permissions = stat.S_IMODE(os.lstat(file_path).st_mode)
    os.chmod(file_path, current_permissions | stat.S_IEXEC)

  def _download_file(self, url: str, num_retries=5) -> None:
    temp_file = Path(MAPD_PATH + ".tmp")
    download_timeout = 60
    for cnt in range(num_retries):
      try:
        response = requests.get(url, stream=True, timeout=download_timeout)
        response.raise_for_status()
        self._safe_write_and_set_executable(temp_file, response.content)
        # No exceptions encountered. Safe to replace original file.
        temp_file.replace(MAPD_PATH)
        return
      except requests.exceptions.ReadTimeout:
        self._spinner.update(f"ReadTimeout caught. Timeout is [{download_timeout}]. Retrying download... [{cnt}]")
        time.sleep(0.5)
      except requests.exceptions.RequestException as e:
        self._spinner.update(f"RequestException caught: {e}. Retrying download... [{cnt}]")
        time.sleep(0.5)

    # Delete temp file if the process was not successful.
    if temp_file.exists():
      temp_file.unlink()
    logging.error("Failed to download file after all retries")

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
        target_version = get_target_version(self._params)
        binary_url = get_target_binary_url(target_version, self._params)
        self._spinner.update(f"Downloading mapd [{self.get_installed_version()}] => [{target_version}] from [{binary_url}].")
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


if __name__ == "__main__":
  spinner = Spinner()
  install_manager = MapdInstallManager(spinner)
  install_manager.ensure_directories_exist()
  if is_prebuilt():
    target_version = get_target_version()
    debug_msg = f"[DEBUG] This is prebuilt, no mapd install required. VERSION: [{target_version}], Param [{install_manager.get_installed_version()}]"
    spinner.update(debug_msg)
    update_installed_version(target_version)
  else:
    spinner.update(f"Checking if mapd is installed and valid. Prebuilt [{is_prebuilt()}]")
    install_manager.non_prebuilt_install()
