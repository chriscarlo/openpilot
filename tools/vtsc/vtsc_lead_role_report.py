#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

try:
  from openpilot.tools.lib.logreader import LogReader
except ModuleNotFoundError:
  # Support direct execution from repo root.
  sys.path.append(str(Path(__file__).resolve().parents[2]))
  from openpilot.tools.lib.logreader import LogReader


def _fmt(v: float | int, n: int = 2) -> str:
  try:
    return f"{float(v):.{n}f}"
  except Exception:
    return "?"


def scan_log(path: Path, y_thresh_m: float) -> dict[str, object]:
  latest_lead = {"status": False, "dRel": math.nan, "yRel": math.nan, "vRel": math.nan}
  stats = {
    "lp_count": 0,
    "lead_source_count": 0,
    "lead_source_adjacent_count": 0,
    "lead_source_missing_lead_count": 0,
    "lead_source_samples": [],
  }

  for msg in LogReader(str(path)):
    which = msg.which()
    if which == "radarState":
      lead = msg.radarState.leadOne
      latest_lead = {
        "status": bool(getattr(lead, "status", False)),
        "dRel": float(getattr(lead, "dRel", math.nan) or math.nan),
        "yRel": float(getattr(lead, "yRel", math.nan) or math.nan),
        "vRel": float(getattr(lead, "vRel", math.nan) or math.nan),
      }
    elif which == "longitudinalPlan":
      stats["lp_count"] += 1
      src = str(msg.longitudinalPlan.longitudinalPlanSource)
      if src.startswith("lead"):
        stats["lead_source_count"] += 1
        sample = {
          "mono": int(msg.logMonoTime),
          "src": src,
          "aTarget": float(msg.longitudinalPlan.aTarget),
          "lead_status": bool(latest_lead["status"]),
          "lead_dRel": float(latest_lead["dRel"]),
          "lead_yRel": float(latest_lead["yRel"]),
          "lead_vRel": float(latest_lead["vRel"]),
        }
        stats["lead_source_samples"].append(sample)
        if not latest_lead["status"]:
          stats["lead_source_missing_lead_count"] += 1
        elif abs(float(latest_lead["yRel"])) > y_thresh_m:
          stats["lead_source_adjacent_count"] += 1

  return stats


def main() -> int:
  ap = argparse.ArgumentParser(description="Summarize adjacent-lane lead-follow source usage from rlogs.")
  ap.add_argument("logs", nargs="+", help="Paths to rlog.zst/qlog.zst files")
  ap.add_argument("--y-thresh-m", type=float, default=3.0, help="Absolute yRel above this counts as adjacent (default: 3.0)")
  ap.add_argument("--show-worst", type=int, default=12, help="Show N most-negative aTarget lead-source rows (default: 12)")
  args = ap.parse_args()

  rows = []
  total = Counter()
  for p in args.logs:
    path = Path(p)
    if not path.exists():
      print(f"[skip] missing: {path}")
      continue
    s = scan_log(path, args.y_thresh_m)
    total.update({
      "lp_count": s["lp_count"],
      "lead_source_count": s["lead_source_count"],
      "lead_source_adjacent_count": s["lead_source_adjacent_count"],
      "lead_source_missing_lead_count": s["lead_source_missing_lead_count"],
    })
    for r in s["lead_source_samples"]:
      rr = dict(r)
      rr["path"] = str(path)
      rows.append(rr)

  print("Lead-Role Report")
  print(f"lp_count={total['lp_count']}")
  print(f"lead_source_count={total['lead_source_count']}")
  adj_pct = (100.0 * total["lead_source_adjacent_count"] / max(total["lead_source_count"], 1))
  print(f"lead_source_adjacent_count={total['lead_source_adjacent_count']} ({_fmt(adj_pct, 1)}%)")
  miss_pct = (100.0 * total["lead_source_missing_lead_count"] / max(total["lead_source_count"], 1))
  print(f"lead_source_missing_lead_count={total['lead_source_missing_lead_count']} ({_fmt(miss_pct, 1)}%)")

  if not rows:
    return 0

  print("\nWorst lead-source decel samples:")
  worst = sorted(rows, key=lambda r: float(r["aTarget"]))[:max(1, args.show_worst)]
  for r in worst:
    lead_flag = "Y" if r["lead_status"] else "N"
    parts = [
      f"path={r['path']}",
      f"mono={r['mono']}",
      f"src={r['src']}",
      f"aT={_fmt(r['aTarget'], 2)}",
      f"lead={lead_flag}",
      f"d={_fmt(r['lead_dRel'], 1)}",
      f"y={_fmt(r['lead_yRel'], 2)}",
      f"vRel={_fmt(r['lead_vRel'], 2)}",
    ]
    line = " ".join(parts)
    print(line)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
