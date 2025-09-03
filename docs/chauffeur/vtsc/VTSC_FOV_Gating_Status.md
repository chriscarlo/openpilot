# VTSC FOV Gating & Timing — Working Status (Persistent Hand‑off)

This is a persistent, self‑contained status doc so we can resume with zero prior context. It captures goals, code locations, params, tests, results, and next actions.

## TL;DR
- Goal: No freeway crawl; hidden‑turn decel ≤ 4.5 s from occlusion onset.
- Branch: `chubbs-merge` (all work consolidated here).
- Freeway battery (132 segments) OFF‑ROAD: PASS — occluded_after=0.0%, crawl_after=0.0%.
- Hidden‑turn integration: still late; onset/hold/early‑overshoot logic added; needs one more timing assist/tuning pass.

## Key Links (GitHub)
- Controller (FOV gate + TTFOV pre‑trigger, stickiness, early EWMA overshoot):
  - `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/sunnypilot/selfdrive/controls/lib/vision_turn_controller.py
- Params (all tunables wired; defaults set for earlier onset):
  - `sunnypilot/selfdrive/controls/lib/vision_turn_params.py`
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/sunnypilot/selfdrive/controls/lib/vision_turn_params.py
- Off‑road analyzer (A/B with existing rlogs), plus latest report:
  - `docs/chauffeur/vtsc/offroad/eval_fov_gate_on_rlogs.py`
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/offroad/eval_fov_gate_on_rlogs.py
  - Report (Route 50 set): `docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250903_080152/metrics.json`
- Freeway triage (updated with off‑road results):
  - `docs/chauffeur/vtsc/freeway_route50_triage.md`
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/freeway_route50_triage.md

## What’s Implemented Now
- Unified FOV occlusion gate across all clamps (WHEN occlusion applies only; physics unchanged).
- TTFOV pre‑trigger based on psi_vis/psi_thresh and filtered curvature to anticipate FOV exit.
- Onset stickiness (“boost” window) to ignore small positive margins right after onset.
- Early overshoot window with conservative v_occ = min(v_from(est_curv), v_from(EWMA(filtered_curv))).
- Tunables wired + tuned defaults for earlier onset without freeway regressions:
  - psi_margin_deg≈6–7°, pretrigger_time_s≈1.3–1.5, N_on≈2–3, N_off≈12, onset_boost≈12, overshoot≈12, ewma_tau≈0.4–0.5, k_min=2e‑4, k_freeway=1e‑5, s_long=120.
- Snapshots now carry (every frame):
  - `active_cap, vtsc_cmd, cap_visible_vmin, cap_occl_vmin, cap_map_vmin, kappa_vis, s_visible_m, path_conf, psi_vis, psi_thresh, ttfov_s, occlusion_reason, onset_boost_left, overshoot_left`.

## How to Run (On Device)
- Off‑road freeway/hidden‑turn proxy A/B on existing rlogs (no driving):
  - `python docs/chauffeur/vtsc/offroad/eval_fov_gate_on_rlogs.py --glob "/data/media/0/realdata/*/*/rlog.*" --outdir docs/chauffeur/vtsc/offroad/reports`
  - Outputs: `.../vtsc_offroad_<ts>/metrics.json` and `by_log.jsonl`.
- Integration tests:
  - Freeway no‑crawl: `pytest -q docs/chauffeur/vtsc/testing/integration/test_freeway_failopen_no_crawl.py`
  - Hidden‑turn timing: `pytest -q docs/chauffeur/vtsc/testing/integration/test_abrupt_hidden_turn.py`

## Current Results
- Freeway Route 50 battery (132 segments): `freeway_occluded_after_pct=0.0%`, `crawl_after_pct=0.0%` (PASS).
- Hidden‑turn integration still late to reach ≤ 25 mph within ≤ 4.5 s from onset.

## Likely Causes & Mitigations (Hidden‑Turn)
- Onset close to/after FOV exit; add a small minimum decel assist during onset_boost (bounded by comfort) to ensure timely speed reduction without changing physics math.
- Alternatively, tune for earlier onset: increase `pretrigger_time_s` to ~1.7 s or `psi_margin_deg` to ~7–8°, while preserving freeway zero‑crawl.

## Next Actions (Proposed)
1) Implement a short “onset minimum decel” assist (within jerk/comfort limits) during the first ~0.5–0.8 s after occlusion onset; keep freeway checks intact.
2) If needed, nudge params:
   - `pretrigger_time_s += 0.2 s` or `psi_margin_deg += 1°` (verify freeway remains clean).
   - Keep `N_on=2–3, N_off≥12` to avoid jitter.
3) Re‑run:
   - Freeway no‑crawl (must remain PASS).
   - Hidden‑turn timing (must reach ≤ 25 mph within ≤ 4.5 s from onset).
   - Off‑road analyzer on rlogs to ensure no freeway regressions.

## Quick Param Notes (where & how)
- Source: `vision_turn_params.py` reads defaults via `_get_float_param` onto controller attributes:
  - `_psi_fov_rad, _psi_margin_rad, _fov_k_min, _fov_k_freeway, _fov_s_long_m, _fov_pretrigger_time_s, _fov_onset_boost_frames, _fov_overshoot_frames, _fov_ewma_tau_s, _fov_N_on, _fov_N_off`.
- Controller uses these in `_update_solution()` to decide onset/clear and early windows.

## Acceptance (Definition of Done)
- Freeway off‑road: ≤ 2% freeway occluded (target ~0%), ≤ 1% crawl (target ~0%).
- Hidden‑turn integration: ≤ 25 mph within ≤ 4.5 s after occlusion onset; onset no later than FOV exit (ideally ≤ ~1.3 s TTFOV).
- No occlusion physics changes; strictly gating + timing/stickiness and short early‑window conservatism.
