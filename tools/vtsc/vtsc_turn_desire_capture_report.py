#!/usr/bin/env python3
from __future__ import annotations

"""
Replay the VTSC low-speed calibration helper against turn_desire_capture JSONL traces.

This is not a full VTSC/rlog replay. It replays the adaptive low-speed calibration state
machine directly from recorded speed, curvature, and lateral-control feedback so captures
with broken rlog carState speed can still be mined for tighten/relax evidence.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

TOOLS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (TOOLS_ROOT, REPO_ROOT):
  if str(p) not in sys.path:
    sys.path.insert(0, str(p))

import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vtc
from vtsc.vtsc_rlog_episode_report import _fmt, _mk_controller_deterministic


ACTIVE_SCALE_EPSILON = 1e-3


def _safe_float(value: Any, default: float = 0.0) -> float:
  try:
    out = float(value)
  except Exception:
    return float(default)
  return out if math.isfinite(out) else float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
  try:
    return bool(value)
  except Exception:
    return bool(default)


def _mode_from_delta(delta_mps: float) -> str:
  if float(delta_mps) > 1e-6:
    return "relax"
  if float(delta_mps) < -1e-6:
    return "tighten"
  return "neutral"


def _median_dt(rows: List[Dict[str, Any]]) -> float:
  if len(rows) < 2:
    return 0.05
  dts = [max(0.0, float(b["t_s"] - a["t_s"])) for a, b in zip(rows, rows[1:], strict=False)]
  if not dts:
    return 0.05
  return float(sorted(dts)[len(dts) // 2])


def _active_duration(rows: List[Dict[str, Any]], pred) -> float:
  if not rows:
    return 0.0
  dt_tail = max(0.01, min(0.20, _median_dt(rows)))
  total = 0.0
  for i, row in enumerate(rows):
    if not pred(row):
      continue
    if i + 1 < len(rows):
      dt_i = max(0.0, float(rows[i + 1]["t_s"] - row["t_s"]))
    else:
      dt_i = dt_tail
    total += max(0.0, float(dt_i))
  return float(total)


def _mode_value(values: List[str]) -> str:
  counts = Counter(v for v in values if v)
  return counts.most_common(1)[0][0] if counts else ""


def _write_tsv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
  out_path = Path(path).expanduser()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys()) if rows else []
  with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    if fieldnames:
      writer.writeheader()
      writer.writerows(rows)


def _load_turn_desire_capture(path: Path) -> List[Dict[str, Any]]:
  rows: List[Dict[str, Any]] = []
  first_ns: int | None = None
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      raw = json.loads(line)
      if raw.get("type") != "sample":
        continue
      t_mono_ns = int(raw.get("t_mono_ns") or 0)
      if first_ns is None:
        first_ns = t_mono_ns
      speed_mps = max(0.0, _safe_float(raw.get("speed_mps")))
      speed_mph = max(0.0, _safe_float(raw.get("speed_mph")))
      rows.append({
        "t_s": float(t_mono_ns - int(first_ns)) * 1e-9,
        "frame_id": int(raw.get("frame_id") or 0),
        "speed_mps": speed_mps,
        "speed_mph": speed_mph,
        "left_blinker": _safe_bool(raw.get("left_blinker")),
        "right_blinker": _safe_bool(raw.get("right_blinker")),
        "one_blinker": _safe_bool(raw.get("one_blinker")),
        "steering_angle_deg": _safe_float(raw.get("steering_angle_deg")),
        "steering_torque": _safe_float(raw.get("steering_torque")),
        "steering_pressed": _safe_bool(raw.get("steering_pressed")),
        "top_desire": str(raw.get("top_desire") or ""),
        "top_desire_prob": _safe_float(raw.get("top_desire_prob")),
        "turn_left_prob": _safe_float(raw.get("turn_left_prob")),
        "turn_right_prob": _safe_float(raw.get("turn_right_prob")),
        "model_desired_curvature": _safe_float(raw.get("model_desired_curvature")),
        "controls_desired_curvature": _safe_float(raw.get("controls_desired_curvature")),
        "controls_curvature": _safe_float(raw.get("controls_curvature")),
        "turn_context_active": _safe_bool(raw.get("turn_context_active")),
        "plan_y_1": _safe_float(raw.get("plan_y_1")),
        "plan_y_2": _safe_float(raw.get("plan_y_2")),
        "plan_y_4": _safe_float(raw.get("plan_y_4")),
        "plan_y_end": _safe_float(raw.get("plan_y_end")),
        "yaw_1": _safe_float(raw.get("yaw_1")),
        "yaw_2": _safe_float(raw.get("yaw_2")),
        "yaw_rate_0": _safe_float(raw.get("yaw_rate_0")),
        "lat_state": str(raw.get("lat_state") or "torqueState"),
        "lat_active": _safe_bool(raw.get("lat_active"), default=True),
        "lat_saturated": _safe_bool(raw.get("lat_saturated")),
        "output": abs(_safe_float(raw.get("output"))),
        "desiredLateralAccel": _safe_float(raw.get("desiredLateralAccel")),
        "actualLateralAccel": _safe_float(raw.get("actualLateralAccel")),
      })
  if not rows:
    raise SystemExit(f"No sample rows found in {path}")
  return rows


class _SM:
  def __init__(self) -> None:
    self.valid = {"controlsState": True}
    self._data: Dict[str, Any] = {}

  def __getitem__(self, key: str):
    return self._data.get(key)


def analyze_turn_desire_capture(
  samples: List[Dict[str, Any]],
  *,
  assumed_cruise_floor_mps: float,
  assumed_cruise_margin_mps: float,
) -> List[Dict[str, Any]]:
  ctrl = _mk_controller_deterministic()
  ctrl._is_enabled = True
  ctrl._op_enabled = True

  sm = _SM()
  trace: List[Dict[str, Any]] = []
  for sample in samples:
    t_s = float(sample["t_s"])
    assumed_cruise_mps = float(max(
      float(sample["speed_mps"]) + float(assumed_cruise_margin_mps),
      float(assumed_cruise_floor_mps),
    ))
    ctrl._v_cruise_setpoint = assumed_cruise_mps
    ctrl._v_ego = float(sample["speed_mps"])

    lat_state_name = str(sample["lat_state"] or "torqueState")
    lat_state = SimpleNamespace(
      active=bool(sample["lat_active"]),
      saturated=bool(sample["lat_saturated"]),
      output=float(sample["output"]),
      desiredLateralAccel=float(sample["desiredLateralAccel"]),
      actualLateralAccel=float(sample["actualLateralAccel"]),
    )
    controls_state = SimpleNamespace(
      desiredCurvature=float(sample["controls_desired_curvature"]),
      curvature=float(sample["controls_curvature"]),
      lateralControlState=SimpleNamespace(which=lambda: lat_state_name, **{lat_state_name: lat_state}),
    )
    sm._data["controlsState"] = controls_state

    reference_curvature = abs(float(sample["controls_curvature"]))
    curve_basis_curvature = max(
      abs(float(sample["controls_desired_curvature"])),
      abs(float(sample["controls_curvature"])),
    )

    with patch.object(vtc.time, "monotonic", lambda: t_s), patch.object(vtc.time, "time", lambda: t_s):
      ctrl._update_low_speed_calibration(sm, reference_curvature=reference_curvature)

    basis = max(1e-8, float(curve_basis_curvature))
    baseline_curve_speed_mps = float(vtc.curvature_to_speed(basis))
    calibrated_curve_speed_mps = float(ctrl._curve_speed(basis))
    delta_curve_speed_mps = float(calibrated_curve_speed_mps - baseline_curve_speed_mps)
    reason = str(getattr(ctrl, "_dbg_low_speed_calibration_reason", "") or "")
    scale = float(getattr(ctrl, "_dbg_low_speed_calibration_scale", 1.0) or 1.0)
    active = bool(abs(scale - 1.0) > ACTIVE_SCALE_EPSILON)

    trace.append({
      **sample,
      "assumed_cruise_mps": assumed_cruise_mps,
      "reference_curvature": float(reference_curvature),
      "curve_basis_curvature": float(curve_basis_curvature),
      "baseline_curve_speed_mps": baseline_curve_speed_mps,
      "baseline_curve_speed_mph": float(baseline_curve_speed_mps * vtc.CV.MS_TO_MPH),
      "calibrated_curve_speed_mps": calibrated_curve_speed_mps,
      "calibrated_curve_speed_mph": float(calibrated_curve_speed_mps * vtc.CV.MS_TO_MPH),
      "delta_curve_speed_mps": delta_curve_speed_mps,
      "delta_curve_speed_mph": float(delta_curve_speed_mps * vtc.CV.MS_TO_MPH),
      "calibration_mode": _mode_from_delta(delta_curve_speed_mps),
      "low_speed_calibration_active": active,
      "low_speed_calibration_reason": reason,
      "low_speed_calibration_state": float(getattr(ctrl, "_low_speed_calibration_state", 0.0) or 0.0),
      "low_speed_calibration_scale": scale,
      "low_speed_calibration_headroom": float(getattr(ctrl, "_dbg_low_speed_calibration_headroom", 0.0) or 0.0),
      "low_speed_calibration_headroom_ema": float(getattr(ctrl, "_dbg_low_speed_calibration_headroom_ema", 0.0) or 0.0),
      "low_speed_calibration_gap": float(getattr(ctrl, "_dbg_low_speed_calibration_gap", 0.0) or 0.0),
      "low_speed_calibration_gap_ratio": float(getattr(ctrl, "_dbg_low_speed_calibration_gap_ratio", 0.0) or 0.0),
      "low_speed_calibration_output": float(getattr(ctrl, "_dbg_low_speed_calibration_output", 0.0) or 0.0),
      "low_speed_calibration_curve_mph": float(getattr(ctrl, "_dbg_low_speed_calibration_curve_mph", 0.0) or 0.0),
      "low_speed_calibration_saturated": bool(getattr(ctrl, "_dbg_low_speed_calibration_saturated", False)),
    })
  return trace


def detect_calibration_episodes(
  trace: List[Dict[str, Any]],
  *,
  gap_s: float = 0.25,
) -> List[Dict[str, Any]]:
  if not trace:
    return []

  active = [bool(row["low_speed_calibration_active"]) for row in trace]
  episodes: List[Dict[str, Any]] = []
  ep_start_idx: int | None = None
  last_active_t: float | None = None

  def close_episode(ep_end_idx: int) -> None:
    nonlocal ep_start_idx
    if ep_start_idx is None or ep_end_idx < ep_start_idx:
      ep_start_idx = None
      return
    rows = trace[ep_start_idx:ep_end_idx + 1]
    active_rows = [row for row in rows if bool(row["low_speed_calibration_active"])]
    if not active_rows:
      ep_start_idx = None
      return
    delta_vals = [float(row["delta_curve_speed_mph"]) for row in active_rows]
    min_delta = min(delta_vals) if delta_vals else 0.0
    max_delta = max(delta_vals) if delta_vals else 0.0
    peak_abs_delta = max(abs(min_delta), abs(max_delta))
    mode = "mixed"
    if abs(min_delta) > max_delta:
      mode = "tighten"
    elif max_delta > abs(min_delta):
      mode = "relax"
    elif peak_abs_delta <= 1e-6:
      mode = "neutral"
    episodes.append({
      "ep_idx": len(episodes),
      "start_t_s": float(rows[0]["t_s"]),
      "end_t_s": float(rows[-1]["t_s"]),
      "duration_s": _active_duration(rows, lambda row: bool(row["low_speed_calibration_active"])),
      "active_samples": len(active_rows),
      "mode": mode,
      "delta_curve_speed_mph_min": min_delta,
      "delta_curve_speed_mph_max": max_delta,
      "peak_abs_delta_curve_speed_mph": peak_abs_delta,
      "scale_min": min(float(row["low_speed_calibration_scale"]) for row in active_rows),
      "scale_max": max(float(row["low_speed_calibration_scale"]) for row in active_rows),
      "state_min": min(float(row["low_speed_calibration_state"]) for row in active_rows),
      "state_max": max(float(row["low_speed_calibration_state"]) for row in active_rows),
      "reason_mode": _mode_value([str(row["low_speed_calibration_reason"]) for row in active_rows]),
      "top_desire_mode": _mode_value([str(row["top_desire"]) for row in active_rows]),
      "speed_mph_min": min(float(row["speed_mph"]) for row in active_rows),
      "speed_mph_max": max(float(row["speed_mph"]) for row in active_rows),
      "output_abs_max": max(float(row["output"]) for row in active_rows),
      "gap_ratio_max": max(float(row["low_speed_calibration_gap_ratio"]) for row in active_rows),
      "curve_mph_min": min(float(row["low_speed_calibration_curve_mph"]) for row in active_rows),
      "curve_mph_max": max(float(row["low_speed_calibration_curve_mph"]) for row in active_rows),
      "turn_context_share": (sum(1 for row in active_rows if bool(row["turn_context_active"])) / len(active_rows)),
      "one_blinker_share": (sum(1 for row in active_rows if bool(row["one_blinker"])) / len(active_rows)),
      "saturated_share": (sum(1 for row in active_rows if bool(row["low_speed_calibration_saturated"])) / len(active_rows)),
    })
    ep_start_idx = None

  for i, (row, is_active) in enumerate(zip(trace, active, strict=False)):
    if is_active:
      if ep_start_idx is None:
        ep_start_idx = i
      last_active_t = float(row["t_s"])
      continue
    if ep_start_idx is not None and last_active_t is not None:
      if (float(row["t_s"]) - float(last_active_t)) >= float(gap_s):
        close_episode(i - 1)
        last_active_t = None

  if ep_start_idx is not None:
    close_episode(len(trace) - 1)

  return episodes


def summarize_calibration_trace(
  trace: List[Dict[str, Any]],
  episodes: List[Dict[str, Any]],
  *,
  source_path: Path,
  assumed_cruise_floor_mps: float,
  assumed_cruise_margin_mps: float,
) -> Dict[str, Any]:
  active_rows = [row for row in trace if bool(row["low_speed_calibration_active"])]
  tighten_rows = [row for row in active_rows if str(row["calibration_mode"]) == "tighten"]
  relax_rows = [row for row in active_rows if str(row["calibration_mode"]) == "relax"]
  moving_rows = [row for row in trace if float(row["speed_mps"]) > 0.1]
  delta_vals = [float(row["delta_curve_speed_mph"]) for row in active_rows] or [0.0]

  return {
    "file": str(source_path),
    "sample_count": len(trace),
    "moving_sample_count": len(moving_rows),
    "active_sample_count": len(active_rows),
    "active_duration_s": _active_duration(trace, lambda row: bool(row["low_speed_calibration_active"])),
    "tighten_duration_s": _active_duration(trace, lambda row: bool(row["low_speed_calibration_active"]) and str(row["calibration_mode"]) == "tighten"),
    "relax_duration_s": _active_duration(trace, lambda row: bool(row["low_speed_calibration_active"]) and str(row["calibration_mode"]) == "relax"),
    "episode_count": len(episodes),
    "tighten_episode_count": sum(1 for ep in episodes if str(ep["mode"]) == "tighten"),
    "relax_episode_count": sum(1 for ep in episodes if str(ep["mode"]) == "relax"),
    "delta_curve_speed_mph_min": min(delta_vals),
    "delta_curve_speed_mph_max": max(delta_vals),
    "peak_abs_delta_curve_speed_mph": max(abs(min(delta_vals)), abs(max(delta_vals))),
    "scale_min": min((float(row["low_speed_calibration_scale"]) for row in active_rows), default=1.0),
    "scale_max": max((float(row["low_speed_calibration_scale"]) for row in active_rows), default=1.0),
    "state_min": min((float(row["low_speed_calibration_state"]) for row in active_rows), default=0.0),
    "state_max": max((float(row["low_speed_calibration_state"]) for row in active_rows), default=0.0),
    "reason_mode_active": _mode_value([str(row["low_speed_calibration_reason"]) for row in active_rows]),
    "top_desire_mode_active": _mode_value([str(row["top_desire"]) for row in active_rows]),
    "turn_context_share_active": (
      sum(1 for row in active_rows if bool(row["turn_context_active"])) / len(active_rows)
      if active_rows else 0.0
    ),
    "one_blinker_share_active": (
      sum(1 for row in active_rows if bool(row["one_blinker"])) / len(active_rows)
      if active_rows else 0.0
    ),
    "saturated_share_active": (
      sum(1 for row in active_rows if bool(row["low_speed_calibration_saturated"])) / len(active_rows)
      if active_rows else 0.0
    ),
    "speed_mph_min_active": min((float(row["speed_mph"]) for row in active_rows), default=0.0),
    "speed_mph_max_active": max((float(row["speed_mph"]) for row in active_rows), default=0.0),
    "curve_mph_min_active": min((float(row["low_speed_calibration_curve_mph"]) for row in active_rows), default=0.0),
    "curve_mph_max_active": max((float(row["low_speed_calibration_curve_mph"]) for row in active_rows), default=0.0),
    "assumed_cruise_floor_mps": float(assumed_cruise_floor_mps),
    "assumed_cruise_margin_mps": float(assumed_cruise_margin_mps),
  }


def _print_summary(summary: Dict[str, Any], episodes: List[Dict[str, Any]]) -> None:
  print(
    "file={file} samples={sample_count} active_samples={active_sample_count} "
    "active_s={active_duration_s:.3f} episodes={episode_count} tighten_eps={tighten_episode_count} "
    "relax_eps={relax_episode_count} delta_mph=[{delta_curve_speed_mph_min:.3f},{delta_curve_speed_mph_max:.3f}] "
    "scale=[{scale_min:.4f},{scale_max:.4f}] reason={reason_mode_active} top_desire={top_desire_mode_active} "
    "turn_ctx_share={turn_context_share_active:.3f}".format(**summary)
  )
  for ep in sorted(episodes, key=lambda row: float(row["peak_abs_delta_curve_speed_mph"]), reverse=True)[:10]:
    print(
      "ep={ep_idx} mode={mode} start={start_t_s:.3f}s end={end_t_s:.3f}s dur={duration_s:.3f}s "
      "peak_abs_delta_mph={peak_abs_delta_curve_speed_mph:.3f} scale=[{scale_min:.4f},{scale_max:.4f}] "
      "reason={reason_mode} top_desire={top_desire_mode} turn_ctx_share={turn_context_share:.3f}".format(**ep)
    )


def main() -> int:
  ap = argparse.ArgumentParser(description="VTSC low-speed calibration report for turn_desire_capture JSONL.")
  ap.add_argument("input", help="turn_desire_capture_*.jsonl path")
  ap.add_argument("--assumed-cruise-floor-mps", type=float, default=20.0,
                  help="Minimum assumed cruise setpoint used to evaluate relevance gating")
  ap.add_argument("--assumed-cruise-margin-mps", type=float, default=5.0,
                  help="Added margin over recorded ego speed for the assumed cruise setpoint")
  ap.add_argument("--episode-gap-s", type=float, default=0.25,
                  help="Gap used to close calibration-active episodes")
  ap.add_argument("--summary-out", type=str, default=None, help="Optional TSV path for one-row summary output")
  ap.add_argument("--episodes-out", type=str, default=None, help="Optional TSV path for episode output")
  ap.add_argument("--samples-out", type=str, default=None, help="Optional TSV path for per-sample trace output")
  args = ap.parse_args()

  source_path = Path(args.input).expanduser().resolve()
  samples = _load_turn_desire_capture(source_path)
  trace = analyze_turn_desire_capture(
    samples,
    assumed_cruise_floor_mps=float(args.assumed_cruise_floor_mps),
    assumed_cruise_margin_mps=float(args.assumed_cruise_margin_mps),
  )
  episodes = detect_calibration_episodes(trace, gap_s=float(args.episode_gap_s))
  summary = summarize_calibration_trace(
    trace,
    episodes,
    source_path=source_path,
    assumed_cruise_floor_mps=float(args.assumed_cruise_floor_mps),
    assumed_cruise_margin_mps=float(args.assumed_cruise_margin_mps),
  )

  _print_summary(summary, episodes)
  if args.summary_out:
    _write_tsv(args.summary_out, [summary])
  if args.episodes_out:
    _write_tsv(args.episodes_out, episodes)
  if args.samples_out:
    _write_tsv(args.samples_out, trace)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
