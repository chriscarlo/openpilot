#!/usr/bin/env python3
"""Tests for turn_desire_capture low-speed calibration replay."""
from pathlib import Path
import sys

import pytest

# Ensure the tools dir is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtsc.vtsc_turn_desire_capture_report import (
  analyze_turn_desire_capture,
  detect_calibration_episodes,
  summarize_calibration_trace,
)


def _capture_sample(t_s: float, **overrides):
  row = {
    "t_s": float(t_s),
    "frame_id": int(round(t_s * 20.0)),
    "speed_mps": 12.0,
    "speed_mph": 12.0 * 2.2369362920544,
    "left_blinker": False,
    "right_blinker": False,
    "one_blinker": False,
    "steering_angle_deg": 2.0,
    "steering_torque": 0.0,
    "steering_pressed": False,
    "top_desire": "none",
    "top_desire_prob": 0.99,
    "turn_left_prob": 0.0,
    "turn_right_prob": 0.0,
    "model_desired_curvature": 0.02,
    "controls_desired_curvature": 0.02,
    "controls_curvature": 0.0195,
    "turn_context_active": False,
    "plan_y_1": 0.0,
    "plan_y_2": 0.0,
    "plan_y_4": 0.0,
    "plan_y_end": 0.0,
    "yaw_1": 0.0,
    "yaw_2": 0.0,
    "yaw_rate_0": 0.0,
    "lat_state": "torqueState",
    "lat_active": True,
    "lat_saturated": False,
    "output": 0.15,
    "desiredLateralAccel": 0.5,
    "actualLateralAccel": 0.48,
  }
  row.update(overrides)
  return row


def test_analyze_turn_desire_capture_relaxes_clean_trace():
  samples = [_capture_sample(i * 0.1) for i in range(200)]

  trace = analyze_turn_desire_capture(
    samples,
    assumed_cruise_floor_mps=20.0,
    assumed_cruise_margin_mps=5.0,
  )

  assert len(trace) == len(samples)
  assert max(float(row["low_speed_calibration_scale"]) for row in trace) > 1.0
  assert max(float(row["delta_curve_speed_mps"]) for row in trace) > 0.0
  assert "relax_clean" in {str(row["low_speed_calibration_reason"]) for row in trace if bool(row["low_speed_calibration_active"])}


def test_analyze_turn_desire_capture_tightens_only_when_saturated():
  samples = [
    _capture_sample(
      i * 0.1,
      output=0.95,
      controls_desired_curvature=0.02,
      controls_curvature=0.014,
      lat_saturated=True,
      actualLateralAccel=0.30,
      top_desire="turnRight",
    )
    for i in range(200)
  ]

  trace = analyze_turn_desire_capture(
    samples,
    assumed_cruise_floor_mps=20.0,
    assumed_cruise_margin_mps=5.0,
  )

  assert min(float(row["low_speed_calibration_scale"]) for row in trace) < 1.0
  assert min(float(row["delta_curve_speed_mps"]) for row in trace) < 0.0
  assert "tighten_saturated" in {str(row["low_speed_calibration_reason"]) for row in trace if bool(row["low_speed_calibration_active"])}


def test_detect_episodes_and_summary_capture_mode_counts():
  tighten = [
    _capture_sample(
      i * 0.1,
      output=0.95,
      controls_desired_curvature=0.02,
      controls_curvature=0.014,
      lat_saturated=True,
      top_desire="turnRight",
    )
    for i in range(80)
  ]
  gap = [_capture_sample(8.0 + i * 0.1, speed_mps=0.0, speed_mph=0.0, lat_active=False, output=0.0) for i in range(200)]
  relax = [_capture_sample(28.0 + i * 0.1, output=0.12, controls_desired_curvature=0.02, controls_curvature=0.0198, top_desire="none") for i in range(160)]
  trace = analyze_turn_desire_capture(
    tighten + gap + relax,
    assumed_cruise_floor_mps=20.0,
    assumed_cruise_margin_mps=5.0,
  )

  episodes = detect_calibration_episodes(trace, gap_s=0.25)
  summary = summarize_calibration_trace(
    trace,
    episodes,
    source_path=Path("/tmp/turn_capture.jsonl"),
    assumed_cruise_floor_mps=20.0,
    assumed_cruise_margin_mps=5.0,
  )

  assert len(episodes) >= 2
  assert summary["episode_count"] == len(episodes)
  assert summary["tighten_episode_count"] >= 1
  assert summary["relax_episode_count"] >= 1
  assert float(summary["tighten_duration_s"]) <= float(summary["active_duration_s"]) + 1e-9
  assert float(summary["relax_duration_s"]) <= float(summary["active_duration_s"]) + 1e-9
  assert float(summary["peak_abs_delta_curve_speed_mph"]) > 0.1
