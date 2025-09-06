#!/usr/bin/env python3
"""
Full-trace VTSC replay: run VisionTurnController over a recorded rlog and emit
per-frame JSON lines capturing method-level pre/post state and the final VTSC
snapshot. Designed for off-road, deterministic triage.

Usage:
  python docs/chauffeur/vtsc/testing/full_trace_replay.py \
    /path/to/rlog.zst --out /path/to/out.jsonl [--cruise MPS] [--max-frames N] [--disable-failopen]

Notes:
  - Wraps key VTSC methods to capture pre/post state without modifying controller.
  - Extracts a curated attribute set; extend ATTRS_* lists as needed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

try:
  from openpilot.tools.lib.logreader import LogReader
except Exception:
  from tools.lib.logreader import LogReader  # type: ignore

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController


# Attribute groups to extract from the controller for traces
ATTRS_CORE = [
  "_v_ego", "_a_ego", "_v_cruise_setpoint",
  "_prev_target_speed", "_current_accel", "_a_target",
]
ATTRS_CURV = [
  "_filtered_curvature", "_current_lat_acc", "_max_pred_lat_acc",
]
ATTRS_GATING = [
  "_fov_occluded", "_dbg_psi_vis", "_dbg_psi_thresh",
  "_psi_fov_rad", "_psi_margin_rad", "_freeway_failopen_active",
]
ATTRS_CAPS = [
  "_dbg_active_cap", "_dbg_cap_visible_vmin", "_dbg_cap_occl_vmin",
  "_dbg_cap_map_vmin", "_dbg_vtsc_cmd", "_dbg_tail_frac", "_dbg_s_tail",
  "_pre_cap_target_speed", "_dbg_pre_cap_target",
]
ATTRS_TUNABLES = [
  "_vis_horizon_s", "_vis_margin_m", "_lat_jerk_cap",
  "_anticipation_target_reduction", "_anticipation_max_reduction_mps",
  "_onset_no_raise_active", "_occlusion_onset_timer_s", "_v_cap_active_at_onset_mps",
]


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
  try:
    return getattr(obj, name)
  except Exception:
    return default


def _extract_occl_state(vtsc: VisionTurnController) -> Dict[str, Any]:
  occl = _safe_get(vtsc, "_occlusion_state")
  if not occl:
    return {}
  fields = [
    "vision_good", "vision_status", "smoothed_confidence",
    "last_valid_curvature", "est_curvature", "distance_since_m",
    "occluded_since_time", "reacquired_at",
    "good_threshold", "bad_threshold", "tail_started", "tail_start_time",
  ]
  out = {}
  for f in fields:
    out[f] = _safe_get(occl, f)
  return out


def _extract_state(vtsc: VisionTurnController) -> Dict[str, Any]:
  state: Dict[str, Any] = {"occlusion_state": _extract_occl_state(vtsc)}
  for name in ATTRS_CORE + ATTRS_CURV + ATTRS_GATING + ATTRS_CAPS + ATTRS_TUNABLES:
    state[name] = _safe_get(vtsc, name)
  return state


def _wrap_method(vtsc: VisionTurnController, name: str, trace_list: List[Dict[str, Any]]) -> None:
  orig = getattr(vtsc, name)

  def wrapped(*args, **kwargs):  # type: ignore
    # pre
    try:
      trace_list.append({"phase": "pre", "method": name, "state": _extract_state(vtsc)})
    except Exception:
      pass
    # call original
    ret = orig(*args, **kwargs)
    # post
    try:
      trace_list.append({"phase": "post", "method": name, "state": _extract_state(vtsc)})
    except Exception:
      pass
    return ret

  setattr(vtsc, name, wrapped)


def _mk_sm_from(ev_model: Any, gas_pressed: bool = False) -> Any:
  class SM:
    def __init__(self, model, gas):
      self.valid = {'modelV2': model is not None}
      self._data = {
        'modelV2': model,
        'carState': SimpleNamespace(gasPressed=bool(gas)),
      }
    def __getitem__(self, k):
      return self._data.get(k)
  return SM(ev_model, gas_pressed)


def replay_full_trace(rlog_path: str, out_path: str, cruise_mps: Optional[float], max_frames: Optional[int], disable_failopen: bool) -> None:
  # Minimal CP
  class CP: ...
  vtsc = VisionTurnController(CP())
  vtsc._is_enabled = True
  vtsc._op_enabled = True
  vtsc._gas_pressed = False
  if disable_failopen:
    vtsc._freeway_failopen_active = False

  # Attach a per-update trace buffer on the instance
  vtsc._full_trace: List[Dict[str, Any]] = []  # type: ignore
  # Wrap key methods to get pre/post snapshots
  for mname in ["_update_params", "_update_calculations", "_state_transition", "_update_solution", "_plan_advanced_speed_trajectory"]:
    if hasattr(vtsc, mname):
      _wrap_method(vtsc, mname, vtsc._full_trace)  # type: ignore

  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  v_ego = 0.0
  a_ego = 0.0
  idx = 0
  with open(out_path, 'w') as f:
    for ev in LogReader(rlog_path):
      w = ev.which()
      if w == 'carState':
        try:
          v_ego = float(ev.carState.vEgo)
          a_ego = float(getattr(ev.carState, 'aEgo', 0.0))
        except Exception:
          pass
      elif w == 'modelV2':
        # Build minimum model struct
        try:
          model = SimpleNamespace(
            orientationRate=SimpleNamespace(z=list(ev.modelV2.orientationRate.z)),
            velocity=SimpleNamespace(x=list(ev.modelV2.velocity.x)),
            laneLineProbs=list(ev.modelV2.laneLineProbs),
          )
        except Exception:
          model = None
        sm = _mk_sm_from(model, gas_pressed=False)
        v_cruise = float(cruise_mps) if cruise_mps is not None else max(v_ego, 0.0)

        # Clear trace buffer for this frame
        try:
          vtsc._full_trace.clear()  # type: ignore
        except Exception:
          vtsc._full_trace = []  # type: ignore

        # Optional: force-disable freeway fail-open
        if disable_failopen:
          vtsc._freeway_failopen_active = False

        try:
          vtsc.update(sm, True, v_ego, a_ego, v_cruise)
        except Exception:
          continue

        # Build one output record
        rec: Dict[str, Any] = {
          "idx": idx,
          "inputs": {"v_ego": v_ego, "a_ego": a_ego, "v_cruise": v_cruise},
          "trace": list(getattr(vtsc, "_full_trace", [])),
          "snapshot": vtsc.snapshot_debug_state() or {},
        }
        try:
          rec["ts"] = float(getattr(ev, 'logMonoTime', 0)) * 1e-9
        except Exception:
          pass
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        idx += 1
        if max_frames and idx >= max_frames:
          break

  print(f"Wrote {idx} full-trace VTSC frames to {out_path}")


def main():
  ap = argparse.ArgumentParser(description='Full-trace VTSC replay over a recorded rlog, outputs JSONL')
  ap.add_argument('rlog', help='Path to rlog.zst or rlog.bz2')
  ap.add_argument('--out', required=True, help='Output JSONL path')
  ap.add_argument('--cruise', type=float, default=None, help='Constant cruise setpoint (m/s), default uses v_ego')
  ap.add_argument('--max-frames', type=int, default=None, help='Limit number of model frames to process')
  ap.add_argument('--disable-failopen', action='store_true', help='Force-disable freeway fail-open gating during replay')
  args = ap.parse_args()

  replay_full_trace(args.rlog, args.out, args.cruise, args.max_frames, args.disable_failopen)


if __name__ == '__main__':
  main()

