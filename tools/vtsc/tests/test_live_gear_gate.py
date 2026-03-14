#!/usr/bin/env python3
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtsc.live_gear_gate import FORWARD_GEARS, monitoring_enabled, normalize_gear_shifter


def test_normalize_gear_shifter_handles_empty_values():
  assert normalize_gear_shifter(None) == "unknown"
  assert normalize_gear_shifter("") == "unknown"
  assert normalize_gear_shifter(" Drive ") == "drive"


def test_monitoring_enabled_requires_started_and_forward_gear():
  for gear in FORWARD_GEARS:
    assert monitoring_enabled(started=True, gear=gear) is True

  for gear in ("unknown", "park", "neutral", "reverse", None, ""):
    assert monitoring_enabled(started=True, gear=gear) is False

  assert monitoring_enabled(started=False, gear="drive") is False
