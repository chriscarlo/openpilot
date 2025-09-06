#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze a VTSC full-trace JSONL and print key metrics + top anomalies.

Usage:
  python docs/chauffeur/vtsc/fullTrace/analyze_trace.py <trace.jsonl> [--top N]

Outputs:
  - Summary counts and rates:
      frames, overslow (v - final >= 2.0 m/s), cap distribution, psi mismatches,
      double-cap violations
  - Top-N worst overslow frames with critical fields for investigation
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Any, List


def _load_jsonl(path: str):
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      try:
        yield json.loads(line)
      except Exception:
        continue


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('trace', help='Path to JSONL produced by full-trace harness')
  ap.add_argument('--top', type=int, default=10, help='Top-N worst overslow frames to print')
  args = ap.parse_args()

  frames = 0
  overslow = 0
  caps: Dict[str, int] = {}
  psi_mismatch = 0
  doublecap_viol = 0
  items: List[Dict[str, Any]] = []

  for rec in _load_jsonl(args.trace):
    frames += 1
    s = rec.get('snapshot', {})
    v = float(s.get('v', 0.0))
    final = float(s.get('final', v))
    cap = str(s.get('_dbg_active_cap') or s.get('active_cap') or '')
    caps[cap] = caps.get(cap, 0) + 1
    dv = v - final
    if v > 0.1 and dv >= 2.0:
      overslow += 1
      items.append({
        'idx': rec.get('idx'), 'dv': dv,
        'cap': cap,
        'v': v, 'final': final,
        'psi_vis': s.get('_dbg_psi_vis'), 'psi_thresh': s.get('_dbg_psi_thresh'),
        'pre_cap': s.get('_pre_cap_target_speed') or s.get('_dbg_pre_cap_target'),
        'vmin_vis': s.get('_dbg_cap_visible_vmin'),
        'vmin_occl': s.get('_dbg_cap_occl_vmin'),
        'vmin_map': s.get('_dbg_cap_map_vmin'),
        'k_vis': s.get('kappa_vis'), 's_visible_m': s.get('s_visible_m'),
        'conf': s.get('path_conf'), 'fail_open': s.get('fail_open'),
        'occluded': s.get('occluded'), 'reason': s.get('occlusion_reason'),
      })
    # PSI mismatch
    psi_vis = s.get('_dbg_psi_vis')
    psi_thr = s.get('_dbg_psi_thresh')
    if cap == 'occlusion' and psi_vis is not None and psi_thr is not None:
      if float(psi_vis) < float(psi_thr):
        psi_mismatch += 1
    # Double-cap guard
    pre = s.get('_pre_cap_target_speed') or s.get('_dbg_pre_cap_target')
    occl = s.get('_dbg_cap_occl_vmin')
    eps = float(s.get('_double_cap_eps_mps', 0.30))
    if cap == 'occlusion' and pre is not None and occl is not None:
      if float(pre) <= float(occl) + eps:
        doublecap_viol += 1

  print('Summary')
  print(f'- frames: {frames}')
  print(f'- overslow (>=2.0 m/s): {overslow} ({(overslow/max(1,frames)):.3f})')
  print('- cap distribution:')
  for k, v in sorted(caps.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f'  - {k or "<none>"}: {v} ({v/max(1,frames):.3f})')
  print(f'- psi_mismatch (cap=occlusion & psi_vis<psi_thresh): {psi_mismatch}')
  print(f'- double_cap_violations: {doublecap_viol}')

  items.sort(key=lambda it: it['dv'], reverse=True)
  print(f'\nTop {min(args.top, len(items))} overslow frames')
  for it in items[:args.top]:
    print(json.dumps(it, separators=(',', ':'), sort_keys=True))


if __name__ == '__main__':
  main()

