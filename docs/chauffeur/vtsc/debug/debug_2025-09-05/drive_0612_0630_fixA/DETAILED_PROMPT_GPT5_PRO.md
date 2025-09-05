# VTSC Overslow/Conservatism — Detailed Prompt for GPT‑5 Pro — Fix A

Scope
- Repo: chriscarlo/chauffeur • Branch: `chubbs-merge` (post‑Fix A changes pending push)
- Window: representative tranche for 2025‑09‑05, segments selected by mtime ≈ 06:12–06:30 (local)

Goal
- Validate and refine changes to avoid undercutting the sigmoid when there is “enough” vision, while preserving freeway fail‑open and hidden‑turn recall, and improving post‑FULL reacquisition.

Inputs
- Replay outputs (Fix A):
  - JSONL: `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330_fixA/combined.jsonl`
  - Analyzer summary (refined reacq rule): `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330_fixA/combined_summary_after_analyzer_update.txt`
- Prior tranche (pre‑Fix A) for comparison:
  - JSONL/TSV: `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/`
  - Off‑road FOV: `docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_145922/{metrics.json,by_log.jsonl}`
- Controller code under review: `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`, `vision_turn_params.py`.

What changed in Fix A (code-level)
- Aligned status thresholds to control gate: `CONFIDENCE_ENTER_PARTIAL = 0.65`, `CONFIDENCE_EXIT_TO_FULL = 0.70`.
- “Enough vision” gating: skip occlusion cap when `vision_good` OR `(s_visible_m ≥ ~35 m AND conf ≥ 0.60)`.
- Vision floor TTL: during ~2.5 s after occlusion onset, enforce occl cap ≥ min(visible cap, 0.98×LKG speed).
- Reacq nudge: accel floor (≥0.18 m/s²) only when physics base supports a raise and target is <0.98×base, for ~0.9 s after FULL.
- Tail growth defaults reduced: gamma_per_m ≈ 2.5e‑4; envelope_horizon_s ≈ 1.0.
- Faster LKG re‑anchor: ramp 0.9 s.

Tasks
1) Cross‑check the “enough vision” predicate against JSONL evidence. Confirm that frames with `vision_status=FULL` or `(s_visible_m ≥ ~35 m AND conf ≥ 0.60)` no longer present `active_cap=occlusion` or `final < v_base` due to occlusion.
2) Validate Vision floor TTL: in early occlusion windows, confirm `cap_occl_vmin ≥ 0.98×v_vis` (bounded by visible cap) and that `final` doesn’t drop below this floor.
3) Reacq behavior: with the refined analyzer (nudge only when `final < 0.98×v_base`), verify we satisfy the floor within ~0.65 s or justify cases where base ≤ v_ego.
4) Hidden‑turn recall + freeway invariants: spot‑check off‑road FOV metrics from prior tranche; confirm Fix A doesn’t re‑introduce freeway occlusion or reduce recall notably.
5) Suggest small, safe follow‑ups:
   - Tune `_vision_floor_ttl_s` (2.0–3.0 s) and `floor_mult` (0.97–1.00) if needed.
   - Consider slightly higher PSI hysteresis or margin at high `s_visible_m` and `conf`.
   - Optional: nudge `gamma_per_m` further if you still see overslow below base in partial occlusion.

Deliverables
- Code deltas (patch‑style) if further tightening is needed, with 5–8 precise quoted JSONL rows.
- Updated quick metrics (counts for `active_cap`, psi‑gate consistency, double‑cap suspects, and any residual undercut of `final` below `v_base` under “enough vision”).

Notes
- Analyzer’s reacq metric now counts only when a raise is warranted.
- Prefer quoting exact numbers with minimal rounding; reference the file paths above.

