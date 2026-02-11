#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running without `pip install -e .` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from openpilot.tools.lib.logreader import LogReader


def _mean(vals: list[float] | None) -> float | None:
  if not vals:
    return None
  try:
    return float(sum(float(x) for x in vals) / len(vals))
  except Exception:
    return None


def _safe_float(v: Any) -> float | None:
  try:
    if v is None:
      return None
    return float(v)
  except Exception:
    return None


def _safe_bool(v: Any) -> bool | None:
  try:
    if v is None:
      return None
    return bool(v)
  except Exception:
    return None


def _curvature_now_and_ahead(model: Any, *, ahead_points: int) -> tuple[float | None, float | None, float | None]:
  """Return (k_now, k_ahead, confidence) from modelV2-like msg.

  - k_now uses orientationRate.z[0] / velocity.x[0]
  - k_ahead uses max(abs(orientationRate.z[1:N])) / velocity.x[0]
  - confidence uses mean(laneLineProbs)
  """
  try:
    v_list = list(getattr(getattr(model, "velocity", None), "x", []) or [])
    v_pred = float(v_list[0]) if v_list else 0.0
  except Exception:
    v_pred = 0.0
  v_pred = float(max(0.1, v_pred))

  try:
    z_list = list(getattr(getattr(model, "orientationRate", None), "z", []) or [])
  except Exception:
    z_list = []

  k_now = None
  k_ahead = None
  if z_list:
    try:
      k_now = abs(float(z_list[0])) / v_pred
    except Exception:
      k_now = None
    try:
      ahead = [abs(float(z)) / v_pred for z in z_list[1:1 + int(max(0, ahead_points))]]
      k_ahead = max(ahead) if ahead else None
    except Exception:
      k_ahead = None

  try:
    lane_probs = list(getattr(model, "laneLineProbs", []) or [])
    conf = _mean([float(x) for x in lane_probs]) if lane_probs else None
  except Exception:
    conf = None
  return k_now, k_ahead, conf


@dataclass
class _Latest:
  # Messages we need at each sample time
  v_ego: float | None = None
  a_ego: float | None = None
  v_cruise_kph: float | None = None
  steering_angle_deg: float | None = None
  long_active: bool | None = None
  lead_d_rel_m: float | None = None
  k_now: float | None = None
  k_ahead: float | None = None
  confidence: float | None = None


def _build_trace(*,
                 rlog_path: str,
                 t0: float,
                 pre_s: float,
                 post_s: float,
                 dt_s: float,
                 ahead_points: int) -> list[dict[str, Any]]:
  # Target sample timestamps in monotonic seconds.
  n = int(round((float(pre_s) + float(post_s)) / float(dt_s))) + 1
  targets = [float(t0) - float(pre_s) + i * float(dt_s) for i in range(n)]
  t_min = targets[0]
  t_max = targets[-1]

  latest = _Latest()
  out: list[dict[str, Any]] = []
  next_i = 0

  for msg in LogReader(rlog_path):
    t = float(msg.logMonoTime) * 1e-9
    if t < t_min - 1.0:
      continue
    if t > t_max + 1.0 and next_i >= len(targets):
      break

    which = None
    try:
      which = msg.which()
    except Exception:
      which = None

    if which == "carState":
      cs = msg.carState
      latest.v_ego = _safe_float(getattr(cs, "vEgo", None))
      latest.a_ego = _safe_float(getattr(cs, "aEgo", None))
      latest.v_cruise_kph = _safe_float(getattr(cs, "vCruise", None))
      latest.steering_angle_deg = _safe_float(getattr(cs, "steeringAngleDeg", None))
    elif which == "carControl":
      cc = msg.carControl
      latest.long_active = _safe_bool(getattr(cc, "longActive", None))
    elif which == "radarState":
      rs = msg.radarState
      lead = getattr(rs, "leadOne", None)
      status = bool(getattr(lead, "status", False)) if lead is not None else False
      latest.lead_d_rel_m = _safe_float(getattr(lead, "dRel", None)) if status else None
    elif which == "modelV2":
      model = msg.modelV2
      k_now, k_ahead, conf = _curvature_now_and_ahead(model, ahead_points=ahead_points)
      latest.k_now = k_now
      latest.k_ahead = k_ahead
      latest.confidence = conf

    # Flush any pending targets up to current time using last-known values.
    while next_i < len(targets) and t >= targets[next_i]:
      target_t = targets[next_i]
      v_cruise_mps = None
      if latest.v_cruise_kph is not None:
        v_cruise_mps = float(latest.v_cruise_kph) * (1000.0 / 3600.0)
      out.append({
        "t": float(f"{(target_t - t0):.2f}"),
        "v_ego": latest.v_ego,
        "a_ego": latest.a_ego,
        "v_cruise": v_cruise_mps,
        "long_active": latest.long_active,
        "lead_d_rel_m": latest.lead_d_rel_m,
        "steering_angle_deg": latest.steering_angle_deg,
        "curvature": latest.k_now,
        "curvature_ahead": latest.k_ahead,
        "confidence": latest.confidence,
      })
      next_i += 1

  # Pad any remaining targets (e.g. log ended) with last-known values.
  while next_i < len(targets):
    target_t = targets[next_i]
    v_cruise_mps = None
    if latest.v_cruise_kph is not None:
      v_cruise_mps = float(latest.v_cruise_kph) * (1000.0 / 3600.0)
    out.append({
      "t": float(f"{(target_t - t0):.2f}"),
      "v_ego": latest.v_ego,
      "a_ego": latest.a_ego,
      "v_cruise": v_cruise_mps,
      "long_active": latest.long_active,
      "lead_d_rel_m": latest.lead_d_rel_m,
      "steering_angle_deg": latest.steering_angle_deg,
      "curvature": latest.k_now,
      "curvature_ahead": latest.k_ahead,
      "confidence": latest.confidence,
    })
    next_i += 1

  return out


def _write_json(path: Path, obj: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
  tmp.replace(path)


def main() -> int:
  p = argparse.ArgumentParser(description="Extract anonymized VTSC RCA fixtures from an rlog window.")
  p.add_argument("--rlog", required=True, help="Path to rlog.zst (or rlog.bz2)")
  p.add_argument("--t0", required=True, type=float, help="Monotonic timestamp (s) of intervention")
  p.add_argument("--out", required=True, help="Output JSON fixture path")
  p.add_argument("--label", default=None, help="Short label for this fixture (no route/dongle IDs)")
  p.add_argument("--pre", type=float, default=10.0, help="Seconds before t0 to include")
  p.add_argument("--post", type=float, default=10.0, help="Seconds after t0 to include")
  p.add_argument("--dt", type=float, default=0.05, help="Sampling interval (s)")
  p.add_argument("--ahead-points", type=int, default=10, help="How many horizon points to use for k_ahead")
  args = p.parse_args()

  steps = _build_trace(
    rlog_path=str(args.rlog),
    t0=float(args.t0),
    pre_s=float(args.pre),
    post_s=float(args.post),
    dt_s=float(args.dt),
    ahead_points=int(args.ahead_points),
  )

  fixture = {
    "version": 1,
    "label": str(args.label) if args.label else Path(str(args.out)).stem,
    "dt_s": float(args.dt),
    "pre_s": float(args.pre),
    "post_s": float(args.post),
    "ahead_points": int(args.ahead_points),
    "steps": steps,
  }
  _write_json(Path(str(args.out)), fixture)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
