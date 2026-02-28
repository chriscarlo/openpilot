#!/usr/bin/env python3
"""
Lightweight analyzer for VTSC on-road snapshots (JSONL).

Adds LKG (last-known-good) channel and quantifies "vision floor saves" when
the occlusion cap is lifted to at least the LKG speed under occlusion.

Usage:
  python docs/chauffeur/vtsc/analysis/analyze_snapshots.py /path/to/vtsc_snapshots.jsonl [--dump-tsv out.tsv]

No external deps; prints a concise summary and flags common issues. Optional TSV
dump includes 17 columns: ts, v, v_base, v_vis, v_occ, v_lkg, cap_visible, cap_occl,
cap_map, map_cov, s_visible, v_target, a_req, comfort_margin, adaptive_margin,
active_cap, final.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, List, Tuple, Optional
import statistics


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


def _get_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
  try:
    v = row.get(key, default)
    return float(v)
  except Exception:
    return float(default)


def derive_v_lkg(row: Dict[str, Any]) -> Optional[float]:
  """Return LKG speed for this snapshot row.

  Priority:
    1) Use 'v_vis' if present (already curvature_to_speed(k_vis_last)).
    2) Else, None (we intentionally avoid re-implementing controller physics here).
  """
  v_vis = row.get('v_vis', None)
  try:
    if v_vis is None:
      return None
    return float(v_vis)
  except Exception:
    return None


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


def quantify_floor_saves(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Quantify cases where the occlusion cap appears lifted to LKG speed.

  Heuristic per row (occlusion-only):
    - occluded == true
    - v_occ < v_lkg - eps (occluded raw slower than LKG)
    - cap_occl_vmin >= v_lkg * mult_lo (cap lifted to at least LKG)

  Returns counts and median delta (v_lkg - v_occ) for such rows.
  """
  saves = 0
  occl_rows = 0
  deltas: List[float] = []
  examples: List[Tuple[float, float, float, float, float]] = []  # ts, v_occ, v_lkg, cap_occ, final
  for r in rows:
    if not bool(r.get('occluded', False)):
      continue
    occl_rows += 1
    v_occ = _get_float(r, 'v_occ', 0.0)
    v_lkg = derive_v_lkg(r)
    cap_occ = _get_float(r, 'cap_occl_vmin', 0.0)
    if v_lkg is None or v_occ <= 0.0:
      continue
    # Heuristics with small margins to avoid float/quantization noise
    if (v_occ < (v_lkg - 0.1)) and (cap_occ >= (0.98 * v_lkg)):
      saves += 1
      deltas.append((v_lkg - v_occ))
      if len(examples) < 6:
        examples.append((
          _get_float(r, 'ts', 0.0),
          v_occ,
          float(v_lkg),
          cap_occ,
          _get_float(r, 'final', 0.0),
        ))
  med_delta = statistics.median(deltas) if deltas else 0.0
  return {
    'occlusion_points': occl_rows,
    'floor_saves': saves,
    'floor_save_ratio': (saves / occl_rows) if occl_rows else 0.0,
    'median_saved_mps': med_delta,
    'examples': examples,
  }


def summarize_cap_transitions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
  if len(rows) < 2:
    return {'transitions': 0, 'transitions_per_s': 0.0, 'map_to_visible': 0, 'visible_to_map': 0}
  labels = [str(r.get('active_cap', '') or '') for r in rows]
  transitions = 0
  map_to_visible = 0
  visible_to_map = 0
  for a, b in zip(labels, labels[1:]):
    if a == b:
      continue
    transitions += 1
    if a == 'map' and b == 'visible':
      map_to_visible += 1
    elif a == 'visible' and b == 'map':
      visible_to_map += 1
  t0 = float(rows[0].get('ts', 0.0))
  t1 = float(rows[-1].get('ts', t0))
  if t1 > t0:
    dur = max(1e-3, t1 - t0)
  else:
    # Fallback for logs/snapshots that omit or flatten timestamps.
    dur = max(1e-3, 0.05 * (len(rows) - 1))
  return {
    'transitions': transitions,
    'transitions_per_s': transitions / dur,
    'map_to_visible': map_to_visible,
    'visible_to_map': visible_to_map,
  }


def flag_handoff_conflicts(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
  """Count map-vs-vision arbitration conflicts.

  Conflict condition:
  - vision is FULL and map lookahead is active
  - map cap is selected as active source
  - visible cap is at least as restrictive as map (within small epsilon)
  """
  bad = 0
  total = 0
  eps = 0.20
  for r in rows:
    if str(r.get('vision_status', 'UNKNOWN')) != 'FULL':
      continue
    if not bool(r.get('map_tail_active', False)):
      continue
    cap_vis = _get_float(r, 'cap_visible_vmin', 0.0)
    cap_map = _get_float(r, 'cap_map_vmin', 0.0)
    if cap_vis <= 0.0 or cap_map <= 0.0:
      continue
    total += 1
    active = str(r.get('active_cap', '') or '')
    if active == 'map' and cap_vis <= (cap_map + eps):
      bad += 1
  return bad, total


def _row_target_speed(r: Dict[str, Any]) -> float:
  vals = []
  for k in ('cap_visible_vmin', 'cap_occl_vmin', 'cap_map_vmin'):
    v = _get_float(r, k, 0.0)
    if v > 0.0:
      vals.append(v)
  if vals:
    return min(vals)
  return _get_float(r, 'final', 0.0)


def _required_decel(v_now: float, v_target: float, distance: float) -> float:
  d = max(1.0, float(distance))
  return (v_target * v_target - v_now * v_now) / (2.0 * d)


def quantify_required_decel_risk(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Estimate whether current decel demand exceeds comfort/adaptive limits.

  Uses visible horizon distance as a conservative proxy for remaining actionable distance.
  """
  evaluated = 0
  comfort_excess = 0
  adaptive_excess = 0
  comfort_margin_samples: List[float] = []
  adaptive_margin_samples: List[float] = []
  examples: List[Tuple[float, float, float, float, float, float]] = []
  for r in rows:
    v_now = _get_float(r, 'v', 0.0)
    s_vis = _get_float(r, 's_visible_m', 0.0)
    if v_now < 8.0 or s_vis < 8.0:
      continue
    v_target = _row_target_speed(r)
    if v_target <= 0.0 or v_target >= (v_now - 0.3):
      continue
    a_req = _required_decel(v_now, v_target, s_vis)
    comfort = -abs(_get_float(r, 'comfort_decel', -1.47))
    adaptive = -abs(_get_float(r, 'max_adaptive_decel', -6.0))
    evaluated += 1
    c_margin = comfort - a_req
    a_margin = adaptive - a_req
    comfort_margin_samples.append(c_margin)
    adaptive_margin_samples.append(a_margin)
    if c_margin > 0.20:
      comfort_excess += 1
    if a_margin > 0.20:
      adaptive_excess += 1
      if len(examples) < 6:
        examples.append((
          _get_float(r, 'ts', 0.0),
          v_now, v_target, s_vis, a_req, adaptive,
        ))
  return {
    'evaluated_rows': evaluated,
    'comfort_excess_count': comfort_excess,
    'adaptive_excess_count': adaptive_excess,
    'median_comfort_margin': statistics.median(comfort_margin_samples) if comfort_margin_samples else 0.0,
    'median_adaptive_margin': statistics.median(adaptive_margin_samples) if adaptive_margin_samples else 0.0,
    'examples': examples,
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
  # Only require a nudge when final is below ~98% of base at the moment of FULL.
  bad = 0
  total = 0
  for i in range(1, len(rows)):
    prev = rows[i-1].get('vision_status', 'UNKNOWN')
    cur = rows[i].get('vision_status', 'UNKNOWN')
    if prev != 'FULL' and cur == 'FULL':
      try:
        v_base_i = float(rows[i].get('v_base', 0.0))
        final_i = float(rows[i].get('final', 0.0))
      except Exception:
        v_base_i = final_i = 0.0
      # Skip if not a raise scenario
      if not (final_i < 0.98 * v_base_i):
        continue
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
  out_tsv: Optional[str] = None
  # Accept optional --dump-tsv argument
  if len(sys.argv) >= 3 and sys.argv[2] == '--dump-tsv':
    if len(sys.argv) >= 4:
      out_tsv = sys.argv[3]
    else:
      print('Error: --dump-tsv requires an output path')
      sys.exit(2)

  rows = load_jsonl(path)
  if not rows:
    print("No data found. Make sure toggles are ON and drive for a few minutes.")
    return
  basic = summarize_basic(rows)
  print("VTSC Snapshot Summary")
  print(f"- Points: {basic.get('points')}  Duration: {basic.get('duration_s')} s  Avg v: {basic.get('avg_speed_mps')} m/s")
  print(f"- Vision states: {basic.get('vision_counts')}")

  cap_trans = summarize_cap_transitions(rows)
  print("\nArbitration & Handoff")
  print(
    "- Active-cap transitions: "
    f"{cap_trans['transitions']} ({cap_trans['transitions_per_s']:.3f}/s), "
    f"map→visible={cap_trans['map_to_visible']}, visible→map={cap_trans['visible_to_map']}"
  )

  # LKG/floor saves
  q = quantify_floor_saves(rows)
  print("\nLKG + Vision Floor Saves")
  print(f"- Occlusion points: {q['occlusion_points']}  floor_saves: {q['floor_saves']}  ratio: {q['floor_save_ratio']:.2%}")
  print(f"- Median saved delta (v_lkg - v_occ): {q['median_saved_mps']:.2f} m/s")
  if q['examples']:
    print("- Examples (ts, v_occ, v_lkg, cap_occ, final):")
    for ts, v_occ, v_lkg, cap_occ, final in q['examples']:
      print(f"  {ts:.2f}\t{v_occ:.2f}\t{v_lkg:.2f}\t{cap_occ:.2f}\t{final:.2f}")

  checks = []
  snc_bad, snc_tot = flag_straight_no_crawl(rows)
  checks.append(("straight_no_crawl_fail", snc_bad, snc_tot, "final << v_base while vision FULL on straight"))

  hwy_bad, hwy_tot = flag_highway_bypass(rows)
  checks.append(("highway_bypass_fail", hwy_bad, hwy_tot, "occlusion depressing target at ≥~65 mph"))

  lb_bad, lb_tot = flag_lead_bypass(rows)
  checks.append(("lead_bypass_fail", lb_bad, lb_tot, "no positive accel with lead≤3s under occlusion"))

  map_bad, map_tot = flag_map_misuse(rows)
  checks.append(("map_cap_misuse", map_bad, map_tot, "map cap active with poor coverage or overshoot vs base"))

  handoff_bad, handoff_tot = flag_handoff_conflicts(rows)
  checks.append(("handoff_conflict_fail", handoff_bad, handoff_tot, "map active while FULL-vision cap is equally/more restrictive"))

  rn_bad, rn_tot = flag_reacq_nudge(rows)
  checks.append(("reacq_nudge_fail", rn_bad, rn_tot, "no accel ≥0.18 m/s² within ~0.65s after FULL"))

  jc_bad, jc_tot = flag_jerk_or_comfort(rows)
  checks.append(("jerk_or_comfort_violations", jc_bad, jc_tot, "jerk bounds or comfort decel violations"))

  req = quantify_required_decel_risk(rows)
  checks.append(("comfort_margin_excess", req['comfort_excess_count'], req['evaluated_rows'], "required decel exceeds comfort envelope"))
  checks.append(("late_brake_risk", req['adaptive_excess_count'], req['evaluated_rows'], "required decel exceeds adaptive envelope"))

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
    elif name == "handoff_conflict_fail":
      print("- Handoff conflict flagged: inspect active_cap vs cap_visible_vmin/cap_map_vmin when vision_status is FULL.")
    elif name == "reacq_nudge_fail":
      print("- Reacquisition nudge flagged: verify 'a_cmd' >= 0.18 within 0.65s after regaining FULL vision.")
    elif name == "jerk_or_comfort_violations":
      print("- Comfort bounds flagged: verify jerk limits and comfort decel caps under occlusion.")
    elif name == "comfort_margin_excess":
      print("- Comfort margin excess flagged: map/vision onset may be late for comfort-only braking; review early anticipation and map handoff.")
    elif name == "late_brake_risk":
      print("- Late-brake risk flagged: required decel exceeded adaptive envelope; inspect map coverage, GPS latency, and arbitration timing.")

  print("\nDecel Margin Summary")
  print(
    f"- Evaluated rows: {req['evaluated_rows']}  "
    f"median comfort margin: {req['median_comfort_margin']:.3f}  "
    f"median adaptive margin: {req['median_adaptive_margin']:.3f}"
  )
  if req['examples']:
    print("- Late-brake examples (ts, v_now, v_target, s_visible_m, a_req, a_adaptive_limit):")
    for ts, v_now, v_target, s_vis, a_req, a_lim in req['examples']:
      print(f"  {ts:.2f}\t{v_now:.2f}\t{v_target:.2f}\t{s_vis:.1f}\t{a_req:.2f}\t{a_lim:.2f}")

  # Optional TSV dump for quick plotting
  if out_tsv:
    try:
      with open(out_tsv, 'w', encoding='utf-8') as f:
        f.write("# ts\tv\tv_base\tv_vis\tv_occ\tv_lkg\tcap_visible\tcap_occl\tcap_map\tmap_cov\ts_visible\tv_target\ta_req\tcomfort_margin\tadaptive_margin\tactive_cap\tfinal\n")
        for r in rows:
          ts = _get_float(r, 'ts', 0.0)
          v = _get_float(r, 'v', 0.0)
          v_base = _get_float(r, 'v_base', 0.0)
          v_vis = _get_float(r, 'v_vis', 0.0)
          v_occ = _get_float(r, 'v_occ', 0.0)
          v_lkg = derive_v_lkg(r) or 0.0
          cap_v = _get_float(r, 'cap_visible_vmin', 0.0)
          cap_o = _get_float(r, 'cap_occl_vmin', 0.0)
          cap_m = _get_float(r, 'cap_map_vmin', 0.0)
          map_cov = _get_float(r, 'map_tail_coverage', 0.0)
          s_vis = _get_float(r, 's_visible_m', 0.0)
          v_target = _row_target_speed(r)
          a_req = _required_decel(v, v_target, s_vis) if (v_target > 0.0 and s_vis > 0.0) else 0.0
          comfort = -abs(_get_float(r, 'comfort_decel', -1.47))
          adaptive = -abs(_get_float(r, 'max_adaptive_decel', -6.0))
          c_margin = comfort - a_req
          a_margin = adaptive - a_req
          active_cap = str(r.get('active_cap', '') or '')
          final = _get_float(r, 'final', 0.0)
          f.write(
            f"{ts:.3f}\t{v:.3f}\t{v_base:.3f}\t{v_vis:.3f}\t{v_occ:.3f}\t{v_lkg:.3f}\t"
            f"{cap_v:.3f}\t{cap_o:.3f}\t{cap_m:.3f}\t{map_cov:.3f}\t{s_vis:.3f}\t{v_target:.3f}\t{a_req:.3f}\t"
            f"{c_margin:.3f}\t{a_margin:.3f}\t{active_cap}\t{final:.3f}\n"
          )
      print(f"\nTSV written: {out_tsv}")
      print("Plot tip (gnuplot):")
      print("  gnuplot -e \"set key left; plot 'OUT.tsv' u 1:3 w l t 'v_base', '' u 1:4 w l t 'v_vis', '' u 1:5 w l t 'v_occ', '' u 1:7 w l t 'cap_vis', '' u 1:9 w l t 'cap_map', '' u 1:17 w l t 'final'\"")
    except Exception as e:
      print("TSV write failed:", e)


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Usage: analyze_snapshots.py /path/to/vtsc_snapshots.jsonl [--dump-tsv out.tsv]")
    sys.exit(2)
  main(sys.argv[1])
