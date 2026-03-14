#!/usr/bin/env python3
"""
Live monitor for the VTSC Rally Co-Pilot curve preview fields.

Use this when the HUD overlay isn't showing and you need to answer:
  - Is the producer publishing preview geometry?
  - Is curvature above the HUD's "more than slight" threshold?
  - Are we feeding sane distances/times/velocity?
"""

import argparse
import math
import time

try:
  import cereal.messaging as messaging
except ModuleNotFoundError as e:
  raise SystemExit(
    "Missing python dependencies (likely `capnp`). Run this on a machine/device with the openpilot python env."
  ) from e


def _fmt_float(v: float, nd: int = 2) -> str:
  if not math.isfinite(v):
    return "nan"
  fmt = f"{{:.{nd}f}}"
  return fmt.format(v)


def main() -> None:
  parser = argparse.ArgumentParser(description="Monitor longitudinalPlanSP VTSC curve preview fields.")
  parser.add_argument("--hz", type=float, default=10.0, help="Print rate (best-effort).")
  parser.add_argument("--once", action="store_true", help="Print one update and exit.")
  args = parser.parse_args()

  sm = messaging.SubMaster(["longitudinalPlanSP"])
  sleep_s = 1.0 / max(1.0, float(args.hz))

  while True:
    sm.update(1000)
    if not sm.updated["longitudinalPlanSP"]:
      time.sleep(sleep_s)
      continue

    lp = sm["longitudinalPlanSP"].longitudinalPlanSP
    vtsc = lp.visionTurnSpeedControl
    pts = list(vtsc.curvePreviewPoints)
    tiles = list(vtsc.curvePreviewTiles)

    # Core fields
    valid = bool(vtsc.curvePreviewValid)
    kappa = float(vtsc.curveMaxCurvature)
    dist_m = float(vtsc.curveDistanceM)
    t_s = float(vtsc.curveTimeToS)
    vel_mps = float(vtsc.velocity)
    direction = int(vtsc.curveDirection)
    severity = int(vtsc.curveSeverity)

    print(
      "curvePreviewValid=%d pts=%d tiles=%d dir=%d sev=%d kappa_max=%s dist_m=%s t_s=%s v_mps=%s"
      % (
        1 if valid else 0,
        len(pts),
        len(tiles),
        direction,
        severity,
        _fmt_float(kappa, 5),
        _fmt_float(dist_m, 1),
        _fmt_float(t_s, 1),
        _fmt_float(vel_mps, 2),
      )
    )

    # Geometry sanity
    if pts:
      xs = [float(p.xFwdM) for p in pts]
      ys = [float(p.yLeftM) for p in pts]
      x0, y0 = xs[0], ys[0]
      x1, y1 = xs[-1], ys[-1]
      print(
        "  x_fwd_m=[%s..%s] y_left_m=[%s..%s] first=(%s,%s) last=(%s,%s)"
        % (
          _fmt_float(min(xs), 1),
          _fmt_float(max(xs), 1),
          _fmt_float(min(ys), 1),
          _fmt_float(max(ys), 1),
          _fmt_float(x0, 1),
          _fmt_float(y0, 1),
          _fmt_float(x1, 1),
          _fmt_float(y1, 1),
        )
      )

    if tiles:
      tile = tiles[0]
      tile_pts = list(tile.points)
      print(
        "  tile0 id=%d dir=%d sev=%d dist_m=%s t_s=%s v_mps=%s tile_pts=%d"
        % (
          int(tile.tileId),
          int(tile.direction),
          int(tile.severity),
          _fmt_float(float(tile.distanceM), 1),
          _fmt_float(float(tile.timeToS), 1),
          _fmt_float(float(tile.advisorySpeedMps), 2),
          len(tile_pts),
        )
      )
      if tile_pts:
        tile_xs = [float(p.xFwdM) for p in tile_pts]
        tile_ys = [float(p.yLeftM) for p in tile_pts]
        print(
          "  tile0 x_fwd_m=[%s..%s] y_left_m=[%s..%s] first=(%s,%s) last=(%s,%s)"
          % (
            _fmt_float(min(tile_xs), 1),
            _fmt_float(max(tile_xs), 1),
            _fmt_float(min(tile_ys), 1),
            _fmt_float(max(tile_ys), 1),
            _fmt_float(tile_xs[0], 1),
            _fmt_float(tile_ys[0], 1),
            _fmt_float(tile_xs[-1], 1),
            _fmt_float(tile_ys[-1], 1),
          )
        )

    if args.once:
      return

    time.sleep(sleep_s)


if __name__ == "__main__":
  main()
