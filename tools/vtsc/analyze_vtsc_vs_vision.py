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


def _interp(samples: List[Tuple[float, float]], t: float) -> float | None:
  """Linear interpolation of (t, v) samples at time t; clamps at ends."""
  if not samples:
    return None
  n = len(samples)
  if t <= samples[0][0]:
    return samples[0][1]
  if t >= samples[-1][0]:
    return samples[-1][1]
  # binary search
  lo, hi = 0, n - 1
  while lo + 1 < hi:
    mid = (lo + hi) // 2
    if samples[mid][0] <= t:
      lo = mid
    else:
      hi = mid
  t0, v0 = samples[lo]
  t1, v1 = samples[hi]
  if t1 == t0:
    return v0
  a = (t - t0) / (t1 - t0)
  return v0 + a * (v1 - v0)


def analyze_segment(seg: Path, sample_dt: float = 0.5) -> Tuple[int, int, List[Tuple[float, float, float, float, float]]]:
  rlog = seg / 'rlog.zst'
  if not rlog.exists():
    return 0, 0, []

  vego: List[Tuple[float, float]] = []    # (time, v_ego m/s)
  vts: List[Tuple[float, float]] = []     # (time, v_vts m/s)
  yaw: List[Tuple[float, float]] = []     # (time, |yaw_rate| rad/s), min over first 3 model points

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
          # Use a conservative min across the first 3 points to reduce spikes
          zmin = min(abs(float(x)) for x in z[:3])
          yaw.append((t, zmin))
      except Exception:
        pass

  if not vego or not vts or not yaw:
    return 0, 0, []

  t0 = vego[0][0]
  t1 = vego[-1][0]
  t = t0
  decel_events = 0
  early = 0
  examples: List[Tuple[float, float, float, float, float]] = []
  # persistence: require >= 2 consecutive early flags (~1.0 s at 0.5 s dt)
  early_streak = 0
  while t <= t1:
    v_ego = _interp(vego, t)
    v_vts = _interp(vts, t)
    yaw_rt = _interp(yaw, t)
    if v_ego is None or v_vts is None or yaw_rt is None:
      t += sample_dt
      continue
    # Convert yaw rate (rad/s) to curvature (1/m): k = yaw_rate / v
    k = max(1e-8, yaw_rt / max(1.0, v_ego))
    v_vis = curvature_to_speed(k)
    # decel event: VTSC below current speed
    if v_ego > 5.0 and v_vts < v_ego:
      decel_events += 1
      # early if below both relative and absolute margins vs vision physics
      rel_ok = (v_vts <= 0.88 * v_vis)
      abs_ok = ((v_vis - v_vts) >= 0.8)
      is_early = rel_ok or abs_ok
      if is_early:
        early_streak += 1
        if early_streak >= 2:
          early += 1
          if len(examples) < 8:
            examples.append((t, v_ego, v_vts, v_vis, k))
      else:
        early_streak = 0
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
