#!/usr/bin/env python3
"""
Analyze VTSC target velocity vs. immediate model curvature-based velocity.

This script reads recent rlogs under /data/media/0/realdata and computes
times when VisionTurnSpeedControl.velocity is below both current vEgo and
the physics speed implied by the model's first predicted curvature sample.

Outputs a short summary per segment and a few example lines.

Usage:
  python tools/vtsc/analyze_vtsc_vs_vision.py [-n N]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from openpilot.tools.lib.logreader import LogReader
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed


def nearest(samples: List[Tuple[float, float]], t: float) -> Tuple[float, float] | None:
  if not samples:
    return None
  # Binary search for nearest timestamp
  lo, hi = 0, len(samples) - 1
  while lo < hi:
    mid = (lo + hi) // 2
    if samples[mid][0] < t:
      lo = mid + 1
    else:
      hi = mid
  idx = lo
  cands = [idx]
  if idx > 0:
    cands.append(idx - 1)
  best = min(cands, key=lambda i: abs(samples[i][0] - t))
  return samples[best]


def analyze_segment(seg: Path, sample_dt: float = 0.5) -> Tuple[int, int, List[Tuple[float, float, float, float, float]]]:
  rlog = seg / 'rlog.zst'
  if not rlog.exists():
    return 0, 0, []

  vego: List[Tuple[float, float]] = []
  vts: List[Tuple[float, float]] = []
  k0: List[Tuple[float, float]] = []

  for m in LogReader(str(rlog)):
    t = m.logMonoTime / 1e9
    which = m.which()
    if which == 'carState':
      vego.append((t, float(m.carState.vEgo)))
    elif which == 'longitudinalPlanSP':
      vts.append((t, float(m.longitudinalPlanSP.visionTurnSpeedControl.velocity)))
    elif which == 'modelV2':
      try:
        z = m.modelV2.orientationRate.z
        if z and len(z) > 0:
          k = abs(float(z[0]))
          k0.append((t, k))
      except Exception:
        pass

  if not vego or not vts or not k0:
    return 0, 0, []

  t0 = vego[0][0]
  t1 = vego[-1][0]
  t = t0
  decel_events = 0
  early = 0
  examples: List[Tuple[float, float, float, float, float]] = []
  while t <= t1:
    vn = nearest(vego, t)
    vt = nearest(vts, t)
    kk = nearest(k0, t)
    if not (vn and vt and kk):
      t += sample_dt
      continue
    v_ego = vn[1]
    v_vts = vt[1]
    k = max(1e-8, kk[1])
    v_vis = curvature_to_speed(k)
    if v_ego > 5.0 and v_vts < v_ego:
      decel_events += 1
      if v_vts < 0.9 * v_vis:
        early += 1
        if len(examples) < 8:
          examples.append((t, v_ego, v_vts, v_vis, k))
    t += sample_dt

  return decel_events, early, examples


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('-n', type=int, default=12, help='number of latest segments to scan')
  args = ap.parse_args()

  base = Path('/data/media/0/realdata')
  segs = sorted([p for p in base.glob('*--*--*') if p.is_dir()], key=lambda p: p.stat().st_mtime)[-args.n:]
  if not segs:
    print('No segments found under', base)
    return
  print('Scanning segments:', [s.name for s in segs])
  for seg in segs:
    decel, early, ex = analyze_segment(seg)
    if decel == early == 0:
      continue
    print(f"{seg.name}: decel events={decel} early_vs_vis={early}")
    for t, v, vts, vvis, k in ex:
      print(f"  t={t:.1f} v_ego={v:.1f} v_vts={vts:.1f} v_vis={vvis:.1f} k={k:.4g}")


if __name__ == '__main__':
  main()

