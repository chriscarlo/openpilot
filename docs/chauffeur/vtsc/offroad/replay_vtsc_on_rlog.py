#!/usr/bin/env python3
"""
Replay VisionTurnController over a recorded rlog and emit VTSC snapshot JSON lines.

This runs the CURRENT controller code against recorded model/carState streams to
approximate on-road behavior with new arbitration changes. It does not require
device hardware and produces a compact JSONL suitable for quick flag counts.

Usage:
  python docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py \
         /data/media/0/realdata/<dongle>--<route>--<seg>/rlog.zst \
         --out docs/chauffeur/vtsc/offroad/replays/REPLAY.jsonl

Optional:
  --cruise <mps>   Constant cruise setpoint (default: use v_ego)
  --max-frames N   Limit frames for quick iteration
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Optional

try:
  from openpilot.tools.lib.logreader import LogReader
except Exception as e:
  from tools.lib.logreader import LogReader  # type: ignore

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController


def _mk_sm_from(ev_model: Any, gas_pressed: bool = False) -> Any:
  """Build a minimal sm-like object that VTS C.update expects."""
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


def replay(rlog_path: str, out_path: str, cruise_mps: Optional[float], max_frames: Optional[int], disable_failopen: bool) -> None:
  # Minimal CP
  class CP: pass
  vtsc = VisionTurnController(CP())
  # Force enabled
  vtsc._is_enabled = True
  vtsc._op_enabled = True
  vtsc._gas_pressed = False
  if disable_failopen:
    vtsc._freeway_failopen_active = False

  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  v_ego = 0.0
  a_ego = 0.0
  count = 0
  with open(out_path, 'w') as f:
    for ev in LogReader(rlog_path):
      w = ev.which()
      if w == 'carState':
        try:
          v_ego = float(ev.carState.vEgo)
        except Exception:
          pass
      elif w == 'modelV2':
        # Build minimal model struct the controller expects
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
        # Optionally force-disable freeway fail-open before update
        if disable_failopen:
          vtsc._freeway_failopen_active = False
        try:
          vtsc.update(sm, True, v_ego, a_ego, v_cruise)
        except Exception:
          continue
        snap = vtsc.snapshot_debug_state() or {}
        if snap:
          # Attach an approximate timestamp if present on event
          try:
            snap['ts'] = float(getattr(ev, 'logMonoTime', 0)) * 1e-9
          except Exception:
            pass
          f.write(json.dumps(snap, separators=(',', ':')) + "\n")
          count += 1
          if max_frames and count >= max_frames:
            break

  print(f"Wrote {count} VTSC snapshots to {out_path}")


def main():
  ap = argparse.ArgumentParser(description='Replay VTSC over a recorded rlog and emit snapshot JSONL')
  ap.add_argument('rlog', help='Path to rlog.zst or rlog.bz2')
  ap.add_argument('--out', required=True, help='Output JSONL path')
  ap.add_argument('--cruise', type=float, default=None, help='Constant cruise setpoint (m/s), default uses v_ego')
  ap.add_argument('--max-frames', type=int, default=None, help='Limit number of model frames to process')
  ap.add_argument('--disable-failopen', action='store_true', help='Force-disable freeway fail-open gating during replay')
  args = ap.parse_args()
  replay(args.rlog, args.out, args.cruise, args.max_frames, args.disable_failopen)


if __name__ == '__main__':
  main()
