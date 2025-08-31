#!/usr/bin/env python3
"""
Synthetic MTSC publisher to stress-test messaging off-car without a second publisher conflict.

Usage:
  # Disable mtscd first to avoid MultiplePublishersError
  PYTHONPATH=$PWD python - <<<'from openpilot.common.params import Params; Params().put_bool("MTSCEnabled", False)'

  # Run synthetic publisher at 10 Hz with large payloads
  PYTHONPATH=$PWD python -u scripts/dev/publish_mtsc_synthetic.py --hz 10 --len 60 --duration 120

This publishes mapTurnSpeedControlSP messages with debug arrays to maximize payload size.
"""
from __future__ import annotations

import argparse
import math
import time

import cereal.messaging as messaging


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument('--hz', type=float, default=10.0)
  ap.add_argument('--len', dest='vec_len', type=int, default=60, help='debug arrays length')
  ap.add_argument('--duration', type=float, default=60.0, help='seconds to run (<=0 for infinite)')
  args = ap.parse_args()

  pm = messaging.PubMaster(['mapTurnSpeedControlSP'])
  period = 1.0 / max(0.1, args.hz)
  end_time = time.monotonic() + (args.duration if args.duration > 0 else 10**9)
  t0 = time.monotonic()
  i = 0
  while time.monotonic() < end_time:
    now = time.monotonic()
    # Build a large message with arrays
    msg = messaging.new_message('mapTurnSpeedControlSP')
    msg.valid = True
    out = msg.mapTurnSpeedControlSP
    out.timeStamp = int(now * 1e9)
    out.available = True
    out.confidence = 0.85
    out.targetSpeedMps = 15.0 + 5.0 * math.sin(i * 0.1)
    out.startDistanceM = 50.0
    out.horizonCoverage = 0.95
    out.minSpeedMps = 8.0
    out.minSpeedAtDistanceM = 120.0
    out.matchedWayId = 123456789
    out.roadClass = 2  # primary
    out.levelSeparation = 0
    out.headingErrorDeg = 2.5
    out.distanceToCenterlineM = 1.2
    out.visHorizonM = 80.0

    # Debug vectors to stress payload (clamped by schema consumers)
    n = max(1, min(200, args.vec_len))
    xs = [float(k * 3.0) for k in range(n)]
    ks = [float(0.001 * (1 + math.sin(0.05 * k))) for k in range(n)]
    vs = [float(25.0 / (1 + 500.0 * kk)) for kk in ks]
    out.init('distancesM', n)
    out.init('kappasPerM', n)
    out.init('vSafeMps', n)
    for j in range(n):
      out.distancesM[j] = xs[j]
      out.kappasPerM[j] = ks[j]
      out.vSafeMps[j] = vs[j]

    try:
      pm.send('mapTurnSpeedControlSP', msg)
    except Exception:
      # Keep going even on transport errors
      pass

    i += 1
    # basic pacing
    dt = time.monotonic() - now
    if dt < period:
      time.sleep(period - dt)

  return 0


if __name__ == '__main__':
  raise SystemExit(main())

