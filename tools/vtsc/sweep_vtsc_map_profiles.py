#!/usr/bin/env python3
from __future__ import annotations

"""
Sweep synthetic blind-curve map profiles and compare advisory vs strategic VTSC.

This is a lightweight local exploration tool for the synthetic map-profile harness:
- Vision is kept effectively blind in the sampled frame (`k_now = k_ahead = 0`)
- Map injects a shallow entry and a tighter hidden apex
- Advisory vs strategic are replayed side-by-side

Typical usage:
  .venv/bin/python tools/vtsc/sweep_vtsc_map_profiles.py
  .venv/bin/python tools/vtsc/sweep_vtsc_map_profiles.py --show-all
  .venv/bin/python tools/vtsc/sweep_vtsc_map_profiles.py --v0 24 --v-cruise 29
"""

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from openpilot.selfdrive.controls.lib.longitudinal_response_model import build_cruise_response_model
from sunnypilot.selfdrive.controls.lib.tests.vtsc.harness import Step, mk_vtsc_with_params, simulate_sequence
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed


def _build_map_profile_polyline(
  lat0: float,
  lon0: float,
  curvature_profile: list[float],
  *,
  profile_start_m: float = 0.0,
  total_m: float = 1500.0,
  step_m: float = 10.0,
) -> list[tuple[float, float, float]]:
  pts: list[tuple[float, float, float]] = []
  n = int(total_m // step_m)
  profile_start_idx = int(round(float(profile_start_m) / float(step_m)))
  for i in range(n + 1):
    profile_idx = i - profile_start_idx
    if 0 <= profile_idx < len(curvature_profile):
      k = float(max(0.0, curvature_profile[profile_idx]))
    else:
      k = 0.0
    pts.append((lat0 + (i * step_m) / 111000.0, lon0, k))
  return pts


def _parse_list(values: str) -> list[float]:
  out: list[float] = []
  for raw in values.split(","):
    s = raw.strip()
    if not s:
      continue
    out.append(float(s))
  if not out:
    raise ValueError("expected at least one numeric value")
  return out


def _enable_map_lookahead(vtsc) -> None:
  orig_get_bool = vtsc._get_bool_param

  def _get_bool(key: str, default: bool = False) -> bool:
    if key == "MTSCLookaheadEnabled":
      return True
    return bool(orig_get_bool(key, default))

  vtsc._get_bool_param = _get_bool


def _set_map_strategy(vtsc, mode: str) -> None:
  orig_get_string = vtsc._get_string_param

  def _get_string(key: str, default: str = "") -> str:
    if key == "VTSCMapStrategy":
      return mode
    return str(orig_get_string(key, default))

  vtsc._map_strategy_mode = mode
  vtsc._get_string_param = _get_string


def _set_response_model(vtsc, *, min_accel: float = -6.0, max_accel: float = 5.0, delay_s: float = 0.35) -> None:
  vtsc.set_longitudinal_response_model(build_cruise_response_model(
    min_accel_mps2=min_accel,
    max_accel_mps2=max_accel,
    actuation_delay_s=delay_s,
  ))


def _patch_map_tail_inputs(vtsc, lat0: float, lon0: float, pts: list[tuple[float, float, float]]) -> None:
  vtsc._get_last_gps_pose = lambda: (lat0, lon0, None)
  vtsc._load_map_curvatures = lambda: pts


def _make_blind_apex_profile(entry_k: float, apex_k: float) -> list[float]:
  unwind_k = max(entry_k, apex_k * 0.40)
  return [
    0.0,
    max(0.0, entry_k * 0.50),
    max(0.0, entry_k),
    max(0.0, apex_k),
    max(0.0, unwind_k),
    max(0.0, entry_k),
    0.0,
  ]


def _run_profile_case(*, mode: str, map_profile: list[float], v0: float, v_cruise: float, planner_delay_s: float) -> dict:
  lat0, lon0 = 37.0, -122.0
  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc)
  _set_map_strategy(vtsc, mode)
  _set_response_model(vtsc, delay_s=planner_delay_s)

  pts = _build_map_profile_polyline(lat0, lon0, map_profile)
  _patch_map_tail_inputs(vtsc, lat0, lon0, pts)

  snap = simulate_sequence(
    steps=[Step(curvature=0.0, curvature_ahead=0.0, confidence=0.95) for _ in range(10)],
    vtsc=vtsc,
    v0_mps=float(v0),
    v_cruise_mps=float(v_cruise),
    dt=0.05,
  )
  return snap


def _fmt(value: float | int | None, *, width: int = 0, prec: int = 2) -> str:
  if value is None:
    s = "-"
  elif isinstance(value, int):
    s = str(value)
  else:
    s = f"{float(value):.{prec}f}"
  return s.rjust(width) if width > 0 else s


def main() -> None:
  ap = argparse.ArgumentParser(description="Sweep synthetic blind-apex map profiles for advisory vs strategic VTSC.")
  ap.add_argument("--entry-curvatures", default="0.0005,0.0015,0.0030,0.0045", help="comma-separated entry curvatures (1/m)")
  ap.add_argument("--apex-curvatures", default="0.0080,0.0120,0.0160,0.0200,0.0240", help="comma-separated apex curvatures (1/m)")
  ap.add_argument("--v0", type=float, default=22.0, help="ego speed in m/s")
  ap.add_argument("--v-cruise", type=float, default=27.0, help="cruise speed in m/s")
  ap.add_argument("--planner-delay-s", type=float, default=0.35, help="shared response-model delay in seconds")
  ap.add_argument("--min-gap-mps", type=float, default=0.5, help="only print rows where advisory - strategic >= this")
  ap.add_argument("--show-all", action="store_true", help="print all swept cases, not just meaningful divergences")
  args = ap.parse_args()

  entry_curvatures = _parse_list(args.entry_curvatures)
  apex_curvatures = _parse_list(args.apex_curvatures)

  rows: list[dict] = []
  for entry_k in entry_curvatures:
    for apex_k in apex_curvatures:
      if apex_k <= entry_k + 1e-9:
        continue
      profile = _make_blind_apex_profile(entry_k, apex_k)
      advisory = _run_profile_case(mode="advisory", map_profile=profile, v0=args.v0, v_cruise=args.v_cruise, planner_delay_s=args.planner_delay_s)
      strategic = _run_profile_case(mode="strategic", map_profile=profile, v0=args.v0, v_cruise=args.v_cruise, planner_delay_s=args.planner_delay_s)

      adv_cmd = float(advisory.get("vtsc_cmd", math.nan))
      strat_cmd = float(strategic.get("vtsc_cmd", math.nan))
      gap = adv_cmd - strat_cmd
      row = {
        "entry_k": float(entry_k),
        "apex_k": float(apex_k),
        "entry_v": float(curvature_to_speed(max(entry_k, 1e-6))),
        "apex_v": float(curvature_to_speed(max(apex_k, 1e-6))),
        "advisory_cmd": adv_cmd,
        "strategic_cmd": strat_cmd,
        "gap_mps": gap,
        "adv_map_cap": float(advisory.get("map_advisory_cap", math.nan)),
        "strat_map_cap": float(strategic.get("map_strategic_cap", math.nan)),
        "anchor_dist_m": float(strategic.get("map_floor_anchor_dist_m", math.nan)),
        "anchor_k": float(strategic.get("map_floor_anchor_k", math.nan)),
      }
      rows.append(row)

  rows.sort(key=lambda r: (-r["gap_mps"], r["entry_k"], r["apex_k"]))

  header = (
    " entry_k  apex_k  entry_v  apex_v  advisory  strategic   gap  anchor_m  anchor_k"
  )
  print(header)
  print("-" * len(header))
  shown = 0
  for row in rows:
    if not args.show_all and row["gap_mps"] < float(args.min_gap_mps):
      continue
    shown += 1
    print(
      f" {_fmt(row['entry_k'], width=7, prec=4)}"
      f" {_fmt(row['apex_k'], width=7, prec=4)}"
      f" {_fmt(row['entry_v'], width=8, prec=2)}"
      f" {_fmt(row['apex_v'], width=7, prec=2)}"
      f" {_fmt(row['advisory_cmd'], width=9, prec=2)}"
      f" {_fmt(row['strategic_cmd'], width=10, prec=2)}"
      f" {_fmt(row['gap_mps'], width=6, prec=2)}"
      f" {_fmt(row['anchor_dist_m'], width=9, prec=2)}"
      f" {_fmt(row['anchor_k'], width=9, prec=4)}"
    )

  if shown == 0:
    print("No cases exceeded the requested advisory-vs-strategic gap threshold.")
  else:
    print(f"\nShown {shown} / {len(rows)} swept cases with gap >= {args.min_gap_mps:.2f} m/s.")


if __name__ == "__main__":
  main()
