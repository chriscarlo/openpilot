#!/usr/bin/env python3
"""
Monitor liveParameters and print key fields that affect lateral behavior.

This is an SSH-friendly helper for diagnosing:
- PRM readiness widget yellow/red (outer msg.valid / comm gating)
- steer ratio overrides / learning
- angle offset drift and roll validity

Example:
  ./selfdrive/debug/monitor_live_parameters.py --once
  ./selfdrive/debug/monitor_live_parameters.py --rate 5
"""

import argparse
import json
import time

import cereal.messaging as messaging


def snapshot(sm: messaging.SubMaster, timeout_ms: int):
  sm.update(timeout_ms)

  out = {}
  if sm.updated.get("carState", False):
    cs = sm["carState"]
    out["carState"] = {
      "vEgo": float(getattr(cs, "vEgo", 0.0)),
      "steeringAngleDeg": float(getattr(cs, "steeringAngleDeg", 0.0)),
      "steeringPressed": bool(getattr(cs, "steeringPressed", False)),
    }

  if sm.updated.get("liveParameters", False):
    lp = sm["liveParameters"]
    out["liveParameters"] = {
      "outer_msg_valid": bool(sm.valid.get("liveParameters", False)),
      "inner_valid": bool(getattr(lp, "valid", False)),
      "steerRatio": float(getattr(lp, "steerRatio", 0.0)),
      "stiffnessFactor": float(getattr(lp, "stiffnessFactor", 0.0)),
      "angleOffsetDeg": float(getattr(lp, "angleOffsetDeg", 0.0)),
      "angleOffsetAverageDeg": float(getattr(lp, "angleOffsetAverageDeg", 0.0)),
      "roll": float(getattr(lp, "roll", 0.0)),
      "steerRatioValid": bool(getattr(lp, "steerRatioValid", False)),
      "stiffnessFactorValid": bool(getattr(lp, "stiffnessFactorValid", False)),
      "angleOffsetValid": bool(getattr(lp, "angleOffsetValid", False)),
      "angleOffsetAverageValid": bool(getattr(lp, "angleOffsetAverageValid", False)),
      "posenetValid": bool(getattr(lp, "posenetValid", False)),
      "sensorValid": bool(getattr(lp, "sensorValid", False)),
    }

  return out


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
  parser.add_argument("--rate", type=float, default=2.0, help="Refresh rate in Hz (default: 2.0)")
  parser.add_argument("--timeout-ms", type=int, default=2000, help="SubMaster update timeout in ms (default: 2000)")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of formatted text")
  args = parser.parse_args()

  period_s = 1.0 / max(args.rate, 0.1)

  sm = messaging.SubMaster(["liveParameters", "carState"], poll="liveParameters")

  while True:
    data = snapshot(sm, args.timeout_ms)
    if not data:
      if args.json:
        print(json.dumps({"ok": False, "error": "no updates"}))
      else:
        print("no updates")
      if args.once:
        return 2
      time.sleep(period_s)
      continue

    if args.json:
      print(json.dumps({"ok": True, **data}))
    else:
      ts = time.strftime("%Y-%m-%d %H:%M:%S")
      print(f"[{ts}]")
      cs = data.get("carState")
      if cs is not None:
        print(f"  carState: vEgo={cs['vEgo']:.2f} m/s steerAngle={cs['steeringAngleDeg']:.2f} deg pressed={cs['steeringPressed']}")
      lp = data.get("liveParameters")
      if lp is not None:
        print(f"  liveParameters: outer_valid={lp['outer_msg_valid']} inner_valid={lp['inner_valid']}")
        print(f"    steerRatio={lp['steerRatio']:.3f} (valid={lp['steerRatioValid']})")
        print(f"    angleOffsetDeg={lp['angleOffsetDeg']:.3f} (avg={lp['angleOffsetAverageDeg']:.3f}, "
              f"valid={lp['angleOffsetValid']}, avgValid={lp['angleOffsetAverageValid']})")
        print(f"    roll={lp['roll']:.5f} rad sensorValid={lp['sensorValid']} posenetValid={lp['posenetValid']}")
        print(f"    stiffnessFactor={lp['stiffnessFactor']:.3f} (valid={lp['stiffnessFactorValid']})")

    if args.once:
      return 0
    time.sleep(period_s)


if __name__ == "__main__":
  raise SystemExit(main())
