# VTSC Overslow/Conservatism — Detailed Prompt for GPT‑5 Pro

Scope
- Repo: chriscarlo/chauffeur • Branch: `chubbs-merge` • Commit: `921fef98cf77` (2025‑09‑05)
- Window: one tranche representative of this morning’s drive, segments selected by file mtime ≈ 06:12–06:30 (local).

Goal
- Diagnose and propose code‑level changes to reduce over‑conservatism when any hint of curvature appears, while preserving freeway fail‑open and hidden‑turn recall. Prioritize on‑road responsiveness immediately after visibility returns to FULL, and ensure that short occlusion dips don’t starve target speed when the last‑known‑good path supports higher speeds.

Inputs (all in this repo)
- Replay outputs (current controller over rlogs):
  - JSONL: `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/combined.jsonl`
  - TSV:   `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/combined.tsv`
  - Analyzer summary: `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/combined_summary.txt`
- Off‑road FOV‑gate evaluator (tuned hysteresis):
  - Metrics: `docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_145922/metrics.json`
  - Per‑log: `docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_145922/by_log.jsonl`
- Context & prior patterns: `docs/chauffeur/vtsc/PRIMER_GPT5_PRO_EXEC.md`, `docs/chauffeur/vtsc/VTSC_FOV_Gating_Status.md`.

Observed (from our analysis)
- Freeway invariants hold: no straight‑road crawl; freeway occlusion ~0%.
- Reacquisition accel lag: ~52% of FULL reacquisitions lack ≥0.18 m/s² within ~0.65 s.
- Vision floor‑lift absent: 0/10,044 “saves” where occlusion cap is lifted toward LKG/visible floor.
- Hidden‑turn recall high with tuned hysteresis (n_on=2, n_off=12): ~98.9% median.

What to produce
1) Root‑cause analysis of slow post‑FULL acceleration and lack of floor‑lift using the JSONL/TSV fields (`active_cap, v_base, v_vis, v_occ, cap_*_vmin, final, a_cmd, jerk_cmd, path_conf, kappa_vis, s_visible_m, psi_vis, psi_thresh, occlusion_reason, vision_status`). Quote 6–10 representative rows.
2) Concrete code changes in `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` that:
   - Add a bounded “reacq nudge” window after FULL (e.g., ensure a_cmd ≥ 0.18 m/s² for ~0.65 s when `final < 0.98×v_base`), respecting comfort/jerk limits.
   - Implement a time‑to‑live vision floor under occlusion: cap ≥ max(v_occ, 0.98×v_lkg) for ~2–3 s.
   - Keep freeway fail‑open and hidden‑turn recall; use hysteresis ~N_on=2, N_off=12 and consider a slightly larger/dynamic psi margin at high confidence and long visibility.
3) Guardrails: prevent double‑capping (raw ≈ occl_vmin when occlusion wins) and ensure psi‑gate consistency (avoid occlusion wins when `psi_vis < psi_thresh` except during explicit dwell).
4) A short test plan (off‑road/replay & on‑road toggles) to validate no freeway regressions and improved reacq behavior.

Notes
- Replay JSONL path above contains all fields required to craft conditions and compute deltas; prefer quoting exact lines with minimal rounding.
- Where adding tunables, propose param names and defaults; keep code snippets localized and comment‑sparse, matching repo style.

