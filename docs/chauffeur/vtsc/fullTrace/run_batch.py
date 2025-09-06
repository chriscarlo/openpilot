#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the full-trace harness over many rlogs and aggregate metrics.

Usage examples:
  python docs/chauffeur/vtsc/fullTrace/run_batch.py \
    --glob 'docs/chauffeur/vtsc/**/rlog_*.zst' \
    --out-dir .cache/batch --max-frames 1200

  python docs/chauffeur/vtsc/fullTrace/run_batch.py \
    --files docs/chauffeur/vtsc/fullTrace/cases/**/rlog_*.zst \
    --out-dir .cache/batch

Produces per-rlog JSONL and a combined metrics TSV at <out-dir>/summary.tsv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

from docs.chauffeur.vtsc.fullTrace.gpt5pro_full_trace_harness import replay_full_trace


def _analyze(jsonl: str) -> Dict[str, float]:
  frames = 0
  overslow = 0
  caps: Dict[str, int] = {}
  psi_mismatch = 0
  doublecap_viol = 0
  with open(jsonl, 'r', encoding='utf-8') as f:
    for line in f:
      try:
        rec = json.loads(line)
      except Exception:
        continue
      frames += 1
      s = rec.get('snapshot', {})
      v = float(s.get('v', 0.0)); final = float(s.get('final', v))
      cap = str(s.get('_dbg_active_cap') or s.get('active_cap') or '')
      caps[cap] = caps.get(cap, 0) + 1
      if v > 0.1 and (v - final) >= 2.0: overslow += 1
      if cap == 'occlusion':
        psi_vis = s.get('_dbg_psi_vis'); psi_thr = s.get('_dbg_psi_thresh')
        if (psi_vis is not None) and (psi_thr is not None) and (float(psi_vis) < float(psi_thr)):
          psi_mismatch += 1
        pre = s.get('_pre_cap_target_speed') or s.get('_dbg_pre_cap_target')
        occl = s.get('_dbg_cap_occl_vmin')
        if (pre is not None) and (occl is not None) and (float(pre) <= float(occl) + 0.30):
          doublecap_viol += 1
  return {
    'frames': float(frames),
    'overslow': float(overslow),
    'overslow_rate': overslow / max(1, frames),
    'psi_mismatch': float(psi_mismatch),
    'doublecap_viol': float(doublecap_viol),
    **{f'cap_{k or "none"}': float(v) for k, v in caps.items()},
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--glob', dest='globpat', default=None, help='Glob for rlogs')
  ap.add_argument('--files', nargs='*', default=None, help='Explicit rlog paths')
  ap.add_argument('--out-dir', required=True, help='Output directory for JSONLs and summary.tsv')
  ap.add_argument('--max-frames', type=int, default=None)
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)

  paths: List[str] = []
  if args.globpat:
    paths.extend(glob.glob(args.globpat, recursive=True))
  if args.files:
    expanded: List[str] = []
    for p in args.files:
      expanded.extend(glob.glob(p, recursive=True))
    paths.extend(expanded)

  paths = sorted({p for p in paths if p.endswith('.zst')})

  rows: List[str] = ["file\tframes\toverslow\toverslow_rate\tpsi_mismatch\tdoublecap_viol"]
  for p in paths:
    base = os.path.basename(p).replace('.zst', '')
    out = os.path.join(args.out_dir, base + '.jsonl')
    replay_full_trace(p, out, cruise_mps=None, max_frames=args.max_frames,
                      disable_failopen=False, only_keys=None, emit_human=False)
    m = _analyze(out)
    rows.append("\t".join([
      p,
      str(int(m['frames'])),
      str(int(m['overslow'])),
      f"{m['overslow_rate']:.3f}",
      str(int(m['psi_mismatch'])),
      str(int(m['doublecap_viol'])),
    ]))

  with open(os.path.join(args.out_dir, 'summary.tsv'), 'w', encoding='utf-8') as f:
    f.write("\n".join(rows) + "\n")
  print(f"Wrote {len(paths)} rows to {args.out_dir}/summary.tsv")


if __name__ == '__main__':
  main()

