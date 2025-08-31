#!/usr/bin/env python3
"""
Subscribe to mapTurnSpeedControlSP and print key fields at ~10 Hz.
Also prints deviceState.started to verify ForceOnroad.

Usage:
  PYTHONPATH=$PWD scripts/dev/sub_mtscd.py
"""
from __future__ import annotations

import argparse
import time

import cereal.messaging as messaging


def fmt_bool(b: bool) -> str:
  return "1" if b else "0"


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--hz", type=float, default=5.0, help="print frequency")
  args = ap.parse_args()

  sm = messaging.SubMaster([
    'deviceState',
    'mapTurnSpeedControlSP',
  ], poll='mapTurnSpeedControlSP')

  last_print = 0.0
  period = 1.0 / max(0.1, args.hz)

  start = time.monotonic()
  while True:
    sm.update(100)
    now = time.monotonic()
    if (now - last_print) < period:
      continue
    last_print = now

    ds = sm['deviceState']
    mtsc = sm['mapTurnSpeedControlSP']
    print(
      f"t+{now-start:6.1f}s | started={fmt_bool(ds.started)} | "
      f"alive={fmt_bool(sm.alive['mapTurnSpeedControlSP'])} upd={fmt_bool(sm.updated['mapTurnSpeedControlSP'])} | "
      f"avail={fmt_bool(mtsc.available)} conf={mtsc.confidence:4.2f} vT={mtsc.targetSpeedMps:5.2f} startM={mtsc.startDistanceM:5.1f} "
      f"cov={mtsc.horizonCoverage:4.2f} way={mtsc.matchedWayId} class={mtsc.roadClass} lvl={mtsc.levelSeparation} "
      f"hErr={mtsc.headingErrorDeg:4.1f} dC={mtsc.distanceToCenterlineM:4.1f} vis={mtsc.visHorizonM:4.1f}"
    )

  return 0


if __name__ == '__main__':
  raise SystemExit(main())

