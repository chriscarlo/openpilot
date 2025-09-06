# -*- coding: utf-8 -*-
"""
Regression tests that reflect known bad on-road VTSC behavior.

These are intentionally strict and expected to FAIL today, serving as a red
bar to drive fixes. They are marked with pytest marker `regression` so they
run only when explicitly requested, e.g.:

  pytest -q -m regression docs/chauffeur/vtsc/fullTrace/tests/test_regressions_rlogs.py

They operate on committed rlogs (segments 67 and 80) and the full-trace harness.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from docs.chauffeur.vtsc.fullTrace.gpt5pro_full_trace_harness import replay_full_trace

RLOG_67 = "docs/chauffeur/vtsc/fullTrace/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_00000085--f247b281ca--67.zst"
RLOG_80 = "docs/chauffeur/vtsc/fullTrace/cases/overslow_2025-09-05/00000085--f247b281ca--80/rlog_00000085--f247b281ca--80.zst"


def _load_jsonl(path: str):
  with open(path) as f:
    for line in f:
      try:
        yield json.loads(line)
      except Exception:
        continue


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(RLOG_67), reason="rlog 67 missing")
def test_rlog67_overslow_rate_is_low():
  out = ".cache/test_rlog67_full_trace.jsonl"
  Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
  replay_full_trace(RLOG_67, out, cruise_mps=None, max_frames=300,
                    disable_failopen=False, only_keys=None, emit_human=False)

  total = 0
  overslow = 0
  for rec in _load_jsonl(out):
    s = rec.get('snapshot', {})
    v = float(s.get('v', 0.0))
    final = float(s.get('final', v))
    if v > 0.1:
      total += 1
      if (v - final) >= 2.0:
        overslow += 1

  assert total >= 50, "insufficient frames captured"
  rate = overslow / max(1, total)
  # Tightened: aim near 0%; allow temporary headroom at 2%
  assert rate <= 0.02, f"overslow rate too high: {rate:.3f} ({overslow}/{total})"


@pytest.mark.regression
@pytest.mark.skipif(not os.path.exists(RLOG_80), reason="rlog 80 missing")
def test_rlog80_psi_mismatch_and_doublecap_are_rare():
  out = ".cache/test_rlog80_full_trace.jsonl"
  Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
  replay_full_trace(RLOG_80, out, cruise_mps=None, max_frames=300,
                    disable_failopen=False, only_keys=None, emit_human=False)

  total = 0
  psi_mismatch = 0
  doublecap_viol = 0
  for rec in _load_jsonl(out):
    s = rec.get('snapshot', {})
    cap = str(s.get('_dbg_active_cap') or s.get('active_cap') or '')
    psi_vis = s.get('_dbg_psi_vis')
    psi_thr = s.get('_dbg_psi_thresh')
    pre = s.get('_pre_cap_target_speed') or s.get('_dbg_pre_cap_target')
    occl = s.get('_dbg_cap_occl_vmin')
    eps = s.get('_double_cap_eps_mps', 0.30)
    total += 1
    if cap == 'occlusion' and psi_vis is not None and psi_thr is not None:
      if float(psi_vis) < float(psi_thr):
        psi_mismatch += 1
    if cap == 'occlusion' and pre is not None and occl is not None:
      if float(pre) <= float(occl) + float(eps):
        doublecap_viol += 1

  assert total >= 50, "insufficient frames captured"
  psi_rate = psi_mismatch / max(1, total)
  # Strict: psi mismatch must be virtually zero
  assert psi_rate <= 0.01, f"psi mismatch rate too high: {psi_rate:.3f} ({psi_mismatch}/{total})"
  # Strict: double-cap guard violations must be zero
  assert doublecap_viol == 0, f"double-cap violations: {doublecap_viol}/{total}"
  # Also ensure overslow is negligible on this segment
  overslow = 0
  for rec in _load_jsonl(out):
    s = rec.get('snapshot', {})
    v = float(s.get('v', 0.0)); final = float(s.get('final', v))
    if v > 0.1 and (v - final) >= 2.0: overslow += 1
  assert overslow <= 2, f"overslow frames too high on seg80: {overslow}/{total}"
