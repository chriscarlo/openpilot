#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-trace VTSC harness: replays recorded rlogs through VisionTurnController,
captures method-level pre/post state and final snapshot per model frame, and
writes JSONL plus a small metrics TSV next to the output.

This file is self-contained (read-only to VTSC logic). It mirrors the baseline
approach in docs/chauffeur/vtsc/fullTrace/full_trace_replay.py, with extras:
- optional "only_keys" filtering of recorded state
- mph unit echoes for sanity checks
- simple metrics aggregation
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

try:
  from openpilot.tools.lib.logreader import LogReader
except Exception:
  # Fallback for non-editable installs
  from tools.lib.logreader import LogReader  # type: ignore

from opendbc.car.common.conversions import Conversions as CV

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController


ATTRS_CORE = ["_v_ego", "_a_ego", "_v_cruise_setpoint", "_prev_target_speed", "_current_accel", "_a_target"]
ATTRS_CURV = ["_filtered_curvature", "_current_lat_acc", "_max_pred_lat_acc", "_max_v_for_current_curvature"]
ATTRS_GATING = [
  "_fov_occluded", "_dbg_psi_vis", "_dbg_psi_thresh", "_psi_fov_rad", "_psi_margin_rad", "_freeway_failopen_active",
]
ATTRS_CAPS = [
  "_dbg_active_cap", "_dbg_cap_visible_vmin", "_dbg_cap_occl_vmin", "_dbg_cap_map_vmin",
  "_dbg_vtsc_cmd", "_dbg_tail_frac", "_dbg_s_tail", "_pre_cap_target_speed", "_dbg_pre_cap_target",
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


def _extract_occl_state(vtsc: VisionTurnController, only: Optional[set[str]] = None) -> Dict[str, Any]:
  occl = _safe_get(vtsc, "_occlusion_state")
  if not occl:
    return {}
  fields = [
    "vision_good", "vision_status", "smoothed_confidence",
    "last_valid_curvature", "est_curvature", "distance_since_m",
    "occluded_since_time", "reacquired_at",
    "good_threshold", "bad_threshold", "tail_started", "tail_start_time",
  ]
  d = {f: _safe_get(occl, f) for f in fields}
  return {k: v for k, v in d.items() if (only is None or k in only)}


def _extract_state(vtsc: VisionTurnController, only: Optional[set[str]] = None) -> Dict[str, Any]:
  state: Dict[str, Any] = {}
  # Core groups
  for name in ATTRS_CORE + ATTRS_CURV + ATTRS_GATING + ATTRS_CAPS + ATTRS_TUNABLES:
    if (only is None) or (name in only):
      state[name] = _safe_get(vtsc, name)
  # Units echo (opt-in if _v_ego present)
  v_ego = _safe_get(vtsc, "_v_ego")
  if v_ego is not None:
    if (only is None) or ("_v_ego_mph" in only):
      try:
        state["_v_ego_mph"] = float(v_ego) * CV.MS_TO_MPH
      except Exception:
        pass
  # Nested occlusion block under a namespaced key for clarity
  if (only is None) or ("occlusion_state" in only):
    state["occlusion_state"] = _extract_occl_state(vtsc, None if only is None else set())
  return state


def _wrap_method(vtsc: VisionTurnController, name: str, trace_list: List[Dict[str, Any]], only: Optional[set[str]]) -> None:
  orig = getattr(vtsc, name)

  def wrapped(*args, **kwargs):  # type: ignore
    try:
      trace_list.append({"phase": "pre", "method": name, "state": _extract_state(vtsc, only)})
    except Exception:
      pass
    ret = orig(*args, **kwargs)
    try:
      trace_list.append({"phase": "post", "method": name, "state": _extract_state(vtsc, only)})
    except Exception:
      pass
    return ret

  setattr(vtsc, name, wrapped)


def _mk_sm_from(ev_model: Any, gas_pressed: bool = False) -> Any:
  class SM:
    def __init__(self, model, gas):
      self.valid = {'modelV2': model is not None}
      self._data = {'modelV2': model, 'carState': SimpleNamespace(gasPressed=bool(gas))}
    def __getitem__(self, k):
      return self._data.get(k)
  return SM(ev_model, gas_pressed)


def replay_full_trace(rlog_path: str,
                      out_path: str,
                      cruise_mps: Optional[float] = None,
                      max_frames: Optional[int] = None,
                      disable_failopen: bool = False,
                      only_keys: Optional[Iterable[str]] = None,
                      emit_human: bool = False) -> None:
  """Run a full VTSC trace over an rlog and write JSONL + metrics TSV."""
  class CP: ...
  vtsc = VisionTurnController(CP())
  vtsc._is_enabled = True
  vtsc._op_enabled = True
  vtsc._gas_pressed = False
  # Replace Params with a benign stub: unknown keys return None/False instead of raising
  class _StubParams:
    def __init__(self, seed: Optional[Dict[str, bytes]] = None):
      self._d: Dict[str, bytes] = dict(seed or {})
    def get(self, k: str):
      return self._d.get(k, None)
    def get_bool(self, k: str) -> bool:
      v = self._d.get(k, b"0")
      try:
        s = v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v)
        return s not in ("0", "false", "False", "")
      except Exception:
        return False
  try:
    vtsc._params = _StubParams()
    vtsc._mem_params = vtsc._params
  except Exception:
    pass
  if disable_failopen:
    try:
      vtsc._freeway_failopen_active = False
    except Exception:
      pass

  # Parse only_keys (state filter)
  only: Optional[set[str]] = None
  if only_keys:
    only = set(str(k).strip() for k in only_keys)

  vtsc._full_trace: List[Dict[str, Any]] = []  # type: ignore
  for mname in ["_update_params", "_update_calculations", "_state_transition", "_update_solution", "_plan_advanced_speed_trajectory"]:
    if hasattr(vtsc, mname):
      _wrap_method(vtsc, mname, vtsc._full_trace, only)  # type: ignore

  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  # Simple metrics
  metric_total = 0
  cap_counts: Dict[str, int] = {}

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

        try:
          vtsc._full_trace.clear()  # type: ignore
        except Exception:
          vtsc._full_trace = []  # type: ignore
        if disable_failopen:
          try:
            vtsc._freeway_failopen_active = False
          except Exception:
            pass

        vtsc.update(sm, True, v_ego, a_ego, v_cruise)

        # Build record
        rec: Dict[str, Any] = {
          "idx": idx,
          "inputs": {"v_ego": v_ego, "a_ego": a_ego, "v_cruise": v_cruise},
          "trace": list(getattr(vtsc, "_full_trace", [])),
          "snapshot": vtsc.snapshot_debug_state() or {},
        }
        # optional timestamp
        try:
          rec["ts"] = float(getattr(ev, 'logMonoTime', 0)) * 1e-9
        except Exception:
          pass

        # Update metrics
        snap = rec.get("snapshot", {})
        cap = snap.get("_dbg_active_cap") or snap.get("active_cap")
        if isinstance(cap, str) and cap:
          cap_counts[cap] = cap_counts.get(cap, 0) + 1
        metric_total += 1

        # Optional human-readable summary (first N lines if caller pipes)
        if emit_human:
          try:
            v = float(snap.get('v', 0.0))
            final = float(snap.get('final', 0.0))
            cap_src = str(snap.get('_dbg_active_cap') or snap.get('active_cap') or '')
            print(f"[{idx:05d}] v={v*CV.MS_TO_MPH:5.1f} mph  final={final*CV.MS_TO_MPH:5.1f} mph  cap={cap_src}")
          except Exception:
            pass

        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        idx += 1
        if max_frames and idx >= max_frames:
          break

  # Metrics TSV
  try:
    mpath = out_path + ".metrics.tsv"
    with open(mpath, 'w') as mf:
      mf.write("metric\tvalue\n")
      mf.write(f"frames\t{metric_total}\n")
      for k, v in sorted(cap_counts.items()):
        mf.write(f"cap_{k}\t{v}\n")
  except Exception:
    pass


def main():
  ap = argparse.ArgumentParser(description='Full-trace VTSC replay over a recorded rlog, outputs JSONL (+metrics TSV)')
  ap.add_argument('rlog', help='Path to rlog.zst or rlog.bz2')
  ap.add_argument('--out', required=True, help='Output JSONL path')
  ap.add_argument('--cruise', type=float, default=None, help='Constant cruise setpoint (m/s), default uses v_ego')
  ap.add_argument('--max-frames', type=int, default=None, help='Limit number of model frames to process')
  ap.add_argument('--disable-failopen', action='store_true', help='Force-disable freeway fail-open gating during replay')
  ap.add_argument('--only-keys', type=str, default=None, help='Comma-separated subset of state keys to record')
  ap.add_argument('--emit-human', action='store_true', help='Print compact human-readable summaries (optional)')
  args = ap.parse_args()

  only_keys: Optional[List[str]] = None
  if args.only_keys:
    only_keys = [s for s in (args.only_keys or '').split(',') if s.strip()]

  replay_full_trace(args.rlog, args.out, args.cruise, args.max_frames, args.disable_failopen, only_keys, args.emit_human)


if __name__ == '__main__':
  main()
