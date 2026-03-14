#!/usr/bin/env python3
"""Tests for gas-event window labeling helpers."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtsc.vtsc_gas_event_window import summarize_gas_event_window


def _row(dt: float, **overrides):
  base = {
    "dt": dt,
    "vEgo": 12.0,
    "vtscVelMps": 10.0,
    "lowSpeedCalActive": True,
    "lowSpeedCalScale": 0.96,
    "lowSpeedCalReason": "tighten_tracking",
    "activeCap": "visible",
  }
  base.update(overrides)
  return base


def test_summarize_gas_event_window_marks_press_through_cap_with_tighten_bias():
  rows = [
    _row(-0.5),
    _row(0.0, vEgo=11.8, vtscVelMps=9.5, lowSpeedCalScale=0.965),
    _row(0.5, vEgo=11.7, vtscVelMps=9.6, lowSpeedCalScale=0.97),
  ]

  summary = summarize_gas_event_window(rows)

  assert summary["samples"] == 3
  assert summary["constraint_label"] == "pressing_through_cap"
  assert summary["calibration_label"] == "tighten_bias"
  assert summary["gap_median"] == pytest.approx(2.1, abs=1e-9)
  assert summary["scale_min"] == pytest.approx(0.96, abs=1e-9)
  assert summary["reason_mode"] == "tighten_tracking"


def test_summarize_gas_event_window_marks_non_constraining_case():
  rows = [
    _row(-0.3, vEgo=7.2, vtscVelMps=9.8, lowSpeedCalActive=False, lowSpeedCalScale=1.0004),
    _row(0.0, vEgo=7.1, vtscVelMps=9.7, lowSpeedCalActive=False, lowSpeedCalScale=1.0008),
    _row(0.3, vEgo=7.0, vtscVelMps=9.6, lowSpeedCalActive=True, lowSpeedCalScale=1.0011),
  ]

  summary = summarize_gas_event_window(rows)

  assert summary["constraint_label"] == "not_constraining"
  assert summary["calibration_label"] == "not_constraining"
  assert summary["gap_median"] == pytest.approx(-2.6, abs=1e-9)


def test_summarize_gas_event_window_marks_relax_candidate_when_cap_still_below_ego():
  rows = [
    _row(-0.4, vEgo=12.0, vtscVelMps=10.8, lowSpeedCalScale=1.007, lowSpeedCalReason="relax_clean"),
    _row(0.0, vEgo=11.9, vtscVelMps=10.9, lowSpeedCalScale=1.008, lowSpeedCalReason="relax_clean"),
    _row(0.4, vEgo=11.8, vtscVelMps=10.7, lowSpeedCalScale=1.006, lowSpeedCalReason="relax_clean"),
  ]

  summary = summarize_gas_event_window(rows)

  assert summary["constraint_label"] == "pressing_through_cap"
  assert summary["calibration_label"] == "relax_candidate"
  assert summary["scale_max"] == pytest.approx(1.008, abs=1e-9)
