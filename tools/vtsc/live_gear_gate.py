#!/usr/bin/env python3
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any


FORWARD_GEARS = {
  "drive",
  "eco",
  "sport",
  "low",
  "brake",
  "manumatic",
}


def normalize_gear_shifter(raw: Any) -> str:
  try:
    text = str(raw or "").strip().lower()
  except Exception:
    return "unknown"
  return text or "unknown"


def monitoring_enabled(*, started: bool, gear: Any) -> bool:
  return bool(started) and normalize_gear_shifter(gear) in FORWARD_GEARS


def _load_messaging():
  try:
    from cereal import messaging  # pylint: disable=import-outside-toplevel
    return messaging
  except ModuleNotFoundError:
    sys.path.append("/data/openpilot")
    try:
      sys.path.append(str(Path(__file__).resolve().parents[2]))
    except Exception:
      pass
    from cereal import messaging  # pylint: disable=import-outside-toplevel
    return messaging


class LiveForwardGearGate:
  def __init__(self, *, enabled: bool, poll_hz: float = 2.0):
    self.enabled = bool(enabled)
    self._lock = threading.Lock()
    self._active = not self.enabled
    self._started = not self.enabled
    self._gear = "all"
    self._available = True

    if not self.enabled:
      return

    try:
      self._messaging = _load_messaging()
    except Exception:
      self._available = False
      self._active = True
      self._started = True
      self._gear = "unknown"
      return

    timeout_ms = max(100, int(1000.0 / max(0.1, float(poll_hz))))
    thread = threading.Thread(target=self._run, args=(timeout_ms,), daemon=True)
    thread.start()

  def _run(self, timeout_ms: int) -> None:
    sm = self._messaging.SubMaster(["carState", "deviceState"], poll="carState")
    while True:
      sm.update(timeout_ms)
      try:
        gear = normalize_gear_shifter(getattr(sm["carState"], "gearShifter", "unknown"))
      except Exception:
        gear = "unknown"
      try:
        started = bool(getattr(sm["deviceState"], "started", False))
      except Exception:
        started = False
      active = monitoring_enabled(started=started, gear=gear)
      with self._lock:
        self._gear = gear
        self._started = started
        self._active = active

  def is_active(self) -> bool:
    with self._lock:
      return bool(self._active)

  def snapshot(self) -> tuple[bool, bool, str]:
    with self._lock:
      return bool(self._active), bool(self._started), str(self._gear)

  @property
  def available(self) -> bool:
    return bool(self._available)
