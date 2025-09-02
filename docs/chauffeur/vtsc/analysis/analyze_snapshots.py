#!/usr/bin/env python3
"""
Lightweight analyzer for VTSC on-road snapshots (JSONL).

Usage:
  python docs/chauffeur/vtsc/analysis/analyze_snapshots.py /path/to/vtsc_snapshots.jsonl

No external deps; prints a concise summary and flags common issues.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
  out = []
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        out.append(json.loads(line))
      except Exception:
        continue
  return out


def summarize_basic(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
  if not rows:
    return {}
  ts0 = float(rows[0].get('ts', 0.0))
  ts1 = float(rows[-1].get('ts', ts0))
  dt = max(0.0, ts1 - ts0)
  vs = [float(r.get('v', 0.0)) for r in rows]
  avg_v = sum(vs) / max(1, len(vs))
  vision_counts = {}
  for r in rows:
    vstat = r.get('vision_status', 'UNKNOWN')
    vision_counts[vstat] = vision_counts.get(vstat, 0) + 1
  return {
    'points': len(rows),
    'duration_s': round(dt, 2),
    'avg_speed_mps': round(avg_v, 2),
    'vision_counts': vision_counts,
  }


def flag_straight_no_crawl(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
  bad = 0
  total = 0
  for r in rows:
    conf = float(r.get('conf', 0.0))
    v_base = float(r.get('v_base', 0.0))
    final = float(r.get('final', 0.0))
    if conf >= 0.8 and v_base > 0.0:
      total += 1
      if final < v_base - 2.0:
        bad += 1
  return bad, total


def flag_highway_bypass(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
  bad = 0
  total = 0
  for r in rows:
    v = float(r.get('v', 0.0))
    v_base = float(r.get('v_base', 0.0))
    final = float(r.get('final', 0.0))
    vision = r.get('vision_status', 'UNKNOWN')
    if v >= 29.0 and vision != 'FULL' and v_base > 0.0:
      total += 1
      if final < v_base - 1.0:
        bad += 1
  return bad, total


def flag_lead_bypass(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
  bad = 0
  total = 0
  for r in rows:
    lead = bool(r.get('lead', False))
    hw = float(r.get('hw', 99.0))
    vision = r.get('vision_status', 'UNKNOWN')
    a = float(r.get('a_cmd', 0.0))
    if lead and hw <= 3.0 and vision != 'FULL':
      total += 1
      if a <= 0.0:
        bad += 1
  return bad, total


def flag_map_misuse(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
  bad = 0
  total = 0
  for r in rows:
    active = bool(r.get('map_tail_active', False))
    cov = float(r.get('map_tail_coverage', 0.0))
    cap = float(r.get('map_tail_cap', 0.0))
    v_base = float(r.get('v_base', 0.0))
    if active:
      total += 1
      if (cov < 0.2 and cap < v_base - 1.0) or (v_base > 0.0 and cap < v_base - 2.0):
        bad += 1
  return bad, total


def flag_reacq_nudge(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
  # Count transitions from non-FULL to FULL and check a_cmd ≥ 0.18 within ~0.65s thereafter
  bad = 0
  total = 0
  # Build quick indices by time
  for i in range(1, len(rows)):
    prev = rows[i-1].get('vision_status', 'UNKNOWN')
    cur = rows[i].get('vision_status', 'UNKNOWN')
    if prev != 'FULL' and cur == 'FULL':
      total += 1
      t0 = float(rows[i].get('ts', 0.0))
      ok = False
      j = i
      while j < len(rows) and float(rows[j].get('ts', t0)) <= t0 + 0.65:
        if float(rows[j].get('a_cmd', 0.0)) >= 0.18:
          ok = True
          break
        j += 1
      if not ok:
        bad += 1
  return bad, total


def flag_jerk_or_comfort(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
  bad = 0
  total = 0
  for r in rows:
    jerk = float(r.get('jerk_cmd', 0.0))
    decel = float(r.get('decel_cmd', 0.0))
    comfort = float(r.get('comfort_decel', -1.47))
    total += 1
    if jerk < -6.5 or jerk > 3.0:
      bad += 1
    if decel < comfort - 0.2:  # more negative than comfort significantly
      bad += 1
  return bad, total


def main(path: str) -> None:
  rows = load_jsonl(path)
  if not rows:
    print("No data found. Make sure toggles are ON and drive for a few minutes.")
    return
  basic = summarize_basic(rows)
  print("VTSC Snapshot Summary")
  print(f"- Points: {basic.get('points')}  Duration: {basic.get('duration_s')} s  Avg v: {basic.get('avg_speed_mps')} m/s")
  print(f"- Vision states: {basic.get('vision_counts')}")

  checks = []
  snc_bad, snc_tot = flag_straight_no_crawl(rows)
  checks.append(("straight_no_crawl_fail", snc_bad, snc_tot, "final << v_base while vision FULL on straight"))

  hwy_bad, hwy_tot = flag_highway_bypass(rows)
  checks.append(("highway_bypass_fail", hwy_bad, hwy_tot, "occlusion depressing target at ≥~65 mph"))

  lb_bad, lb_tot = flag_lead_bypass(rows)
  checks.append(("lead_bypass_fail", lb_bad, lb_tot, "no positive accel with lead≤3s under occlusion"))

  map_bad, map_tot = flag_map_misuse(rows)
  checks.append(("map_cap_misuse", map_bad, map_tot, "map cap active with poor coverage or overshoot vs base"))

  rn_bad, rn_tot = flag_reacq_nudge(rows)
  checks.append(("reacq_nudge_fail", rn_bad, rn_tot, "no accel ≥0.18 m/s² within ~0.65s after FULL"))

  jc_bad, jc_tot = flag_jerk_or_comfort(rows)
  checks.append(("jerk_or_comfort_violations", jc_bad, jc_tot, "jerk bounds or comfort decel violations"))

  print("\nFlags")
  for name, bad, tot, hint in checks:
    ratio = (bad / tot) if tot else 0.0
    print(f"- {name}: {bad}/{tot}  ({ratio:.2%})  -- {hint}")

  print("\nRecommendations")
  for name, bad, tot, hint in checks:
    if bad == 0:
      continue
    if name == "straight_no_crawl_fail":
      print("- Straight road crawl flagged: inspect 'conf', 'vision', 'k_model', 'final-v_base'. Consider occlusion dwell or gating thresholds.")
    elif name == "highway_bypass_fail":
      print("- Highway bypass flagged: ensure occlusion gating relaxes ≥~65 mph. Check 'vision', 'final', 'v_base'.")
    elif name == "lead_bypass_fail":
      print("- Lead bypass flagged: check 'lead', 'hw', 'occl_lead_bypass_active', and 'a_cmd'.")
    elif name == "map_cap_misuse":
      print("- Map misuse flagged: check 'map_tail_coverage', 'map_tail_cap' vs 'v_base'; consider disabling lookahead for verification.")
    elif name == "reacq_nudge_fail":
      print("- Reacquisition nudge flagged: verify 'a_cmd' >= 0.18 within 0.65s after regaining FULL vision.")
    elif name == "jerk_or_comfort_violations":
      print("- Comfort bounds flagged: verify jerk limits and comfort decel caps under occlusion.")


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Usage: analyze_snapshots.py /path/to/vtsc_snapshots.jsonl")
    sys.exit(2)
  main(sys.argv[1])

