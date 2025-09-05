# VTSC Metrics Summary — 2025-09-05 (Drive 06:12–06:30 local) — Fix A

Sources analyzed (post‑patch):
- Replay snapshots: docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330_fixA/combined.jsonl
- Analyzer summary: docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330_fixA/combined_summary_after_analyzer_update.txt

Highlights (Replay Analyzer with refined reacq condition)
- Vision states: FULL≈8.8k, PARTIAL≈2.9k, SEVERE≈11.1k (unchanged counts expected over same rlogs).
- LKG floor saves: small but present (2 examples in this tranche).
- Straight freeway crawl: 0/448.
- Highway bypass failure: 0/1967.
- Reacq nudge (only when final < 0.98×base at FULL): 0/0 in this slice (no events requiring a floor found by the condition).
- Comfort/jerk flags: similar to earlier (replay heuristic; on‑road may be lower).

What changed in code (Fix A)
- Status thresholds aligned to control gate: PARTIAL at 0.65; exit FULL at 0.70.
- “Enough vision” floor: skip occlusion capping when `vision_good` or `(s_visible_m ≥ ~35m and conf ≥ 0.60)`.
- Vision floor TTL: for ~2.5 s after occlusion onset, occl cap ≥ min(visible cap, 0.98×LKG).
- Reacq nudge: accel floor only if physics base supports a raise (base_cap > v_ego + 0.05 and target < 0.98×base), for ~0.9 s after FULL.
- Tail growth defaults reduced: gamma_per_m ~2.5e‑4; envelope_horizon_s ~1.0 (still tunable via Params).
- Faster LKG re‑anchor: _lkg_ramp_s 0.9 s.

Expected behavioral effect
- No undercut below sigmoid when model visibility is viable.
- Less stickiness immediately after occlusion; faster recovery.
- Maintains freeway fail‑open and hidden‑turn recall (psi/dwell continue to gate influence).

