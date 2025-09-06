VTSC Full‑Trace Analysis Prompt (for GPT‑5 Pro)

Context
- This repository contains a Vision Turn Speed Control (VTSC) module that computes a target speed for upcoming curves using model curvature, visibility, and occlusion logic. Recent behavior reports: overly sensitive early slowing, “double‑capping” under occlusion, and inconsistent gating.
- You have two inputs:
  1) A small, concrete full‑trace replay JSONL from a recorded rlog: `docs/chauffeur/vtsc/fullTrace/examples/vtsc_full_trace_sample.jsonl` (20 frames).
  2) The normalized trace schema example: `docs/chauffeur/vtsc/fullTrace/examples/trace_event_schema.json`.

Your Goal
Analyze the full‑trace JSONL to identify: (a) the exact phase(s) and conditions where VTSC clamps or slows too early, (b) any duplicated effects between visible and occlusion caps, (c) inconsistencies in PSI/FOV gating, and (d) concrete, code‑level remediation with validation steps.

Key Concepts & Signals
- Snapshot fields (per frame):
  - `v`: ego speed (m/s), `raw`/`final`: VTSC computed speed target (m/s), `v_base`: physics/visible baseline, `v_vis`: speed from last‑visible curvature, `v_occ`: speed from occlusion estimate, `active_cap`: visible|occlusion|map|none
  - `cap_visible_vmin`, `cap_occl_vmin`, `cap_map_vmin`: cap minima (m/s)
  - `psi_vis`, `psi_thresh`: FOV gating; occlusion allowed when `psi_vis ≥ psi_thresh`
  - `occlusion_reason`: pretrigger|onset|tail|fov_exit, etc.
  - `tail_frac`, `s_tail`: occlusion “tail” bookkeeping
- Method trace entries (ordered): pre/post snapshots of `_update_params`, `_update_calculations`, `_state_transition`, `_update_solution`, `_plan_advanced_speed_trajectory` with selected internal fields:
  - `_filtered_curvature`, `_current_lat_acc`, `_max_pred_lat_acc`
  - `_fov_occluded`, `_dbg_psi_vis`, `_dbg_psi_thresh`, `_psi_fov_rad`, `_psi_margin_rad`
  - `_dbg_active_cap`, `_dbg_cap_visible_vmin`, `_dbg_cap_occl_vmin`, `_dbg_vtsc_cmd`
  - `_prev_target_speed`, `_current_accel`, `_a_target`
  - `occlusion_state.{vision_good, vision_status, smoothed_confidence, last_valid_curvature, est_curvature, distance_since_m, occluded_since_time, reacquired_at}`

What to Produce
1) Summary diagnosis
   - Occurrence and timing of early slowdowns: which methods/states cause target speed to drop early?
   - Evidence of double‑cap: frames where `active_cap = occlusion` AND `raw ≈ cap_occl_vmin` while `cap_visible_vmin` also undercuts `v_vis` (esp. during occlusion bypass/gating).
   - PSI/FOV gating mismatches: any frames with `active_cap = occlusion` while `psi_vis < psi_thresh`.
   - Any suspicious EMA or curvature inflation that precedes caps arbitration.

2) Concrete remediation
   - Code‑level diffs or precise edits: which functions/lines to change, and how.
   - Tunable recommendations (Params): values and expected effect (e.g., `VisHorizonS`, anticipation depth, psi thresholds).
   - Justify each change with frame indices from the trace and before/after expected impact on `final`.

3) Validation plan
   - Exact commands to replay multiple rlogs (list at end) with the full‑trace harness and compare aggregate metrics: cap winner distribution, psi gating, delta between `v_vis` and `cap_visible_vmin` under occlusion, rate and magnitude of `current_accel` vs. distance to apex.
   - Stopping conditions: thresholds for acceptable false positives (e.g., occlusion cap when `psi_vis < psi_thresh` must be < 1% of frames).

Ground Truth / Expectations
- Visible cap (when occluded) should reflect last‑visible curvature, not model/filtered curvature.
- Curvature EMA should not track horizon max; it should track near‑term curvature to avoid sticky overslow.
- PSI margin is unified to 0.087 rad (~5°) in all codepaths.
- On straight, long‑visibility, high‑confidence highway frames, occlusion must fail‑open (no occlusion cap wins).

Data to Analyze
- Full trace sample: `docs/chauffeur/vtsc/fullTrace/examples/vtsc_full_trace_sample.jsonl`
  - Each line is a JSON object with: `idx`, `ts` (optional), `inputs`, `trace[]`, and `snapshot`.
  - Use `trace` to locate the exact method and state where decisions change, then corroborate with `snapshot`.

Useful commands (replay and metrics)
- Full trace (custom rlog):
  - `python docs/chauffeur/vtsc/fullTrace/full_trace_replay.py /path/to/rlog.zst --out OUT.jsonl --max-frames 300`
- Quick replay (snapshots only):
  - `python docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py /path/to/rlog.zst --out OUT.jsonl --max-frames 300`
- Candidate rlogs in repo:
  - `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_*.zst`
  - `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/rlog_*.zst`
  - `docs/chauffeur/vtsc/debug/debug_2025-09-04/artifacts/rlog_00000084--9a63e66bc3--12.zst`
  - `docs/chauffeur/vtsc/vtsc_good_data_20250903_013935/logs/*_rlog.zst`

Deliverables
- A short written report answering (1)‑(3) above with:
  - A concise set of code diffs or line‑level edits
  - Tunable updates + justifications
  - Verification metrics on at least two rlogs (one “overslow” case and one “good” case)
  - Any diagnostics/plots that visualize cap winners and gating transitions over time

Appendix: Schema reference
- See `docs/chauffeur/vtsc/fullTrace/examples/trace_event_schema.json` for a compact example record and core keys.
