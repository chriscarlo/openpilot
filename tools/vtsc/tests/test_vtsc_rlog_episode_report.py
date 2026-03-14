#!/usr/bin/env python3
"""Tests for VTSC replay-report calibration summaries."""
from pathlib import Path
import sys

import pytest

# Ensure the tools dir is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtsc.vtsc_rlog_episode_report import (
  Sample,
  _build_sample_rows,
  _extract_route_and_segment,
  _summarize_low_speed_calibration,
)


def _sample(t: float, **overrides) -> Sample:
  base = dict(
    t=t,
    v_ego=10.0,
    a_ego=0.0,
    v_turn=9.0,
    a_target=-0.2,
    vtsc_cmd=9.0,
    active_cap="visible",
    low_speed_calibration_active=False,
    low_speed_calibration_reason="decay_ambiguous",
    low_speed_calibration_state=0.0,
    low_speed_calibration_scale=1.0,
    low_speed_calibration_headroom=0.0,
    low_speed_calibration_headroom_ema=0.0,
    low_speed_calibration_gap=0.0,
    low_speed_calibration_gap_ratio=0.0,
    low_speed_calibration_output=0.0,
    low_speed_calibration_curve_mph=18.0,
    low_speed_calibration_saturated=False,
  )
  base.update(overrides)
  return Sample(**base)


def test_summarize_low_speed_calibration_tracks_duration_reason_and_extrema():
  samples = [
    _sample(0.0),
    _sample(
      0.1,
      low_speed_calibration_active=True,
      low_speed_calibration_reason="tighten_effort",
      low_speed_calibration_state=-0.03,
      low_speed_calibration_scale=0.97,
      low_speed_calibration_headroom=-0.22,
      low_speed_calibration_gap_ratio=0.35,
      low_speed_calibration_output=0.40,
    ),
    _sample(
      0.2,
      low_speed_calibration_active=True,
      low_speed_calibration_reason="tighten_effort",
      low_speed_calibration_state=-0.06,
      low_speed_calibration_scale=0.94,
      low_speed_calibration_headroom=-0.31,
      low_speed_calibration_gap_ratio=0.62,
      low_speed_calibration_output=0.52,
      low_speed_calibration_saturated=True,
    ),
    _sample(0.3),
  ]

  summary = _summarize_low_speed_calibration(samples)

  assert summary["active_s"] == pytest.approx(0.2, abs=1e-9)
  assert summary["first_active_s"] == pytest.approx(0.1, abs=1e-9)
  assert summary["scale_min"] == pytest.approx(0.94, abs=1e-9)
  assert summary["scale_max"] == pytest.approx(1.0, abs=1e-9)
  assert summary["state_min"] == pytest.approx(-0.06, abs=1e-9)
  assert summary["state_max"] == pytest.approx(0.0, abs=1e-9)
  assert summary["headroom_min"] == pytest.approx(-0.31, abs=1e-9)
  assert summary["headroom_max"] == pytest.approx(0.0, abs=1e-9)
  assert summary["gap_ratio_max"] == pytest.approx(0.62, abs=1e-9)
  assert summary["output_abs_max"] == pytest.approx(0.52, abs=1e-9)
  assert summary["saturated_s"] == pytest.approx(0.1, abs=1e-9)
  assert summary["reason_mode"] == "tighten_effort"


def test_build_sample_rows_emits_before_after_calibration_columns():
  before = [_sample(1.0, vtsc_cmd=8.8, low_speed_calibration_reason="disabled_for_replay")]
  after = [_sample(
    1.0,
    vtsc_cmd=8.5,
    active_cap="map",
    low_speed_calibration_active=True,
    low_speed_calibration_reason="tighten_tracking",
    low_speed_calibration_scale=0.95,
    low_speed_calibration_state=-0.05,
    low_speed_calibration_headroom=-0.18,
    low_speed_calibration_headroom_ema=-0.12,
    low_speed_calibration_gap=0.002,
    low_speed_calibration_gap_ratio=0.48,
    low_speed_calibration_output=0.61,
    low_speed_calibration_curve_mph=23.0,
    low_speed_calibration_saturated=True,
  )]

  rows = _build_sample_rows(
    route="000000b5--c27d51740b",
    seg="61",
    rlog=Path("/tmp/fake/rlog.zst"),
    before_samples=before,
    after_samples=after,
  )

  assert len(rows) == 1
  row = rows[0]
  assert row["delta_vtsc_cmd"] == pytest.approx(-0.3, abs=1e-9)
  assert row["before_low_speed_calibration_reason"] == "disabled_for_replay"
  assert row["after_active_cap"] == "map"
  assert row["after_low_speed_calibration_active"] is True
  assert row["after_low_speed_calibration_scale"] == pytest.approx(0.95, abs=1e-9)
  assert row["after_low_speed_calibration_gap_ratio"] == pytest.approx(0.48, abs=1e-9)
  assert row["after_low_speed_calibration_saturated"] is True


def test_extract_route_and_segment_handles_segment_dir_layout():
  route, seg = _extract_route_and_segment(Path("/tmp/000000b5--c27d51740b--61/rlog.zst"))
  assert route == "000000b5--c27d51740b"
  assert seg == "61"
