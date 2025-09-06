VTSC Full‑Trace Test Harness Spec (for GPT‑5 Pro)

Objective
- Produce a full‑trace, on‑parity offline test harness that replays real rlogs through VTSC and emits exhaustive, machine‑parsable debug output for every critical calculation, threshold, gate, and decision inside `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`.
- The output must make it trivial to pinpoint unit/scale bugs, duplicated/overlapping caps, inconsistent PSI/FOV gating, or hidden state overwrites that are not caught by existing tests.

Key Deliverables
- A Python CLI tool placed at `docs/chauffeur/vtsc/fullTrace/gpt5pro_full_trace_harness.py` that:
  - Ingests a recorded `rlog.zst` via `openpilot.tools.lib.logreader.LogReader`.
  - Constructs a `VisionTurnController` instance and simulates the minimal OpenPilot state (carState, modelV2), matching production update cadence.
  - Wraps and instruments every critical VTSC method to emit pre/post state including all relevant internal attributes and intermediate values.
  - Emits JSONL lines per model frame with stable keys and timestamps; each line includes inputs, per‑phase traces, and a final snapshot.
  - Supports flags: `--cruise MPS`, `--max-frames N`, `--disable-failopen`, `--only-keys key1,key2,...` (filter fields). Human-readable output is not required.
- A minimal pytest suite under `docs/chauffeur/vtsc/fullTrace/tests/` validating the harness on the included sample rlog. Focus on invariants that would catch unit errors and double‑capping:
  - Unit sanity: MPH↔MPS conversions consistent, non‑negative speeds, finite curvature, psi math monotonicity.
  - Gating consistency: no `active_cap=occlusion` when `psi_vis < psi_thresh` unless explicitly justified; `double_cap_guard` honored.
  - Fail‑open coverage: on straight + long‑visibility + good confidence, visible cap wins or no cap.
- Clear, local run instructions in the file header and a short `README` note if needed.

Scope of Instrumentation (minimum)
- Wrap the following VTSC methods and record pre/post state: `_update_params`, `_update_calculations`, `_state_transition`, `_update_solution`, `_plan_advanced_speed_trajectory`.
- Inside each phase capture:
  - Kinematics: `_v_ego`, `_a_ego`, `_prev_target_speed`, `_current_accel`, `_a_target`.
  - Curvature/physics: `_filtered_curvature`, `_current_lat_acc`, `_max_pred_lat_acc`, `_max_v_for_current_curvature`.
  - Occlusion model state: `vision_good`, `vision_status`, `smoothed_confidence`, `last_valid_curvature`, `est_curvature`, `distance_since_m`, `occluded_since_time`, `reacquired_at`, trend/tail timers if present.
  - PSI/FOV inputs: `_dbg_psi_vis`, `_dbg_psi_thresh` (or recompute `psi_vis = |kappa_vis|*s_visible_m`, `psi_thresh = psi_fov - psi_margin`), `_psi_fov_rad`, `_psi_margin_rad`, `_fov_occluded`, `_freeway_failopen_active`.
  - Cap arbitration: `_dbg_cap_visible_vmin`, `_dbg_cap_occl_vmin`, `_dbg_cap_map_vmin`, `_dbg_active_cap`, `_dbg_vtsc_cmd`, `pre_cap_target`.
  - Tunables/thresholds actually used: `_vis_horizon_s`, `_lat_jerk_cap`, `_anticipation_*`, occlusion dwell timers, hysteresis thresholds, highway min speed gates, any `*_EPS` values.
  - Unit provenance for every calculation: record which inputs are m/s, mph, 1/m, rad, etc., and any conversion coefficients applied (e.g., `CV.MPH_TO_MS`).
- If a field does not yet exist, compute it explicitly for the trace rather than changing controller behavior. Keep the harness read‑only to VTSC logic.

Output Record Shape (per model frame)
- Top level: `{ idx, ts, inputs:{v_ego,a_ego,v_cruise}, trace:[{phase, method, state:{...}}], snapshot:{...} }`.
- Provide a compact schema example (see links) and align keys with what’s already used in `full_trace_replay.py` to ease downstream tooling.

Required Data Sources (GitHub permalinks)
- Controller under test:
  - vision_turn_controller.py
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/sunnypilot/selfdrive/controls/lib/vision_turn_controller.py
  - vision_turn_params.py
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/sunnypilot/selfdrive/controls/lib/vision_turn_params.py
- Harness references already in repo:
  - Full‑trace harness (baseline):
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/fullTrace/full_trace_replay.py
  - Snapshot replay (lighter):
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py
  - LogReader utility:
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/openpilot/tools/lib/logreader.py
  - Unit‑test harness for VTSC Params (reference):
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/sunnypilot/selfdrive/controls/lib/tests/vtsc/harness.py
  - Example schema and sample JSONL (20 frames):
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/fullTrace/examples/trace_event_schema.json
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/fullTrace/examples/vtsc_full_trace_sample.jsonl
- Case rlog for development and validation:
  - Overslow case (segment 67) rlog (12 MB):
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_00000085--f247b281ca--67.zst
  - Same artifacts mirrored in fullTrace (path will exist after commit):
    - `docs/chauffeur/vtsc/fullTrace/cases/overslow_2025-09-05/00000085--f247b281ca--67/`
  - Additional overslow case (occlusion heavy, segment 80) rlog (9 MB):
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/rlog_00000085--f247b281ca--80.zst
  - Prior off‑road replay JSONLs (reference outputs for multiple segments):
    - Summary: https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/offroad/replays/REPLAY_SUMMARY.tsv
    - Segment 67: https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/offroad/replays/replay_00000085--f247b281ca--67.jsonl
    - Segment 80: https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/offroad/replays/replay_00000085--f247b281ca--80.jsonl
    - Segment 80 (fail‑open disabled variant): https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/offroad/replays/replay_00000085--f247b281ca--80_nofo.jsonl
    - Normal baseline examples (60–66): e.g., https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/offroad/replays/replay_00000085--f247b281ca--60.jsonl
- Additional reference (for context, optional to read programmatically):
  - VTSC overview and parameters:
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/documentation/VTSC_Overview.md
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/documentation/paramBaseline.md
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/reference/vtscTunableParams.md
  - FOV gating notes:
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/VTSC_FOV_Gating_Status.md
  - Debug snapshots and summaries (ground truth symptoms):
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/debug/debug_2025-09-05/DIAGNOSTICS_OVERVIEW.md
    - https://raw.githubusercontent.com/chriscarlo/chauffeur/6284595518800b2c33287cd8fcb882272a391da2/docs/chauffeur/vtsc/debug/debug_2025-09-05/analyze_vtsc_vs_vision.txt

Harness Requirements (detailed)
- Simulation:
  - Reconstruct minimal `SM` shim providing `modelV2` (fields used by VTSC) and `carState.gasPressed`.
  - Feed `carState` updates for `vEgo`/`aEgo`; on each `modelV2` event invoke `vtsc.update(sm, True, v_ego, a_ego, v_cruise)`.
  - Compute `v_cruise` as specified by `--cruise` or default to `v_ego`.
- Instrumentation:
  - Monkey‑patch target methods to push `{phase, method, state}` before and after execution.
  - State encoder must be resilient to missing attrs; use safe getters and record `None` rather than failing.
  - Include a top‑level `snapshot` using `VisionTurnController.snapshot_debug_state()` for stable downstream consumers.
  - Provide an opt‑in `--emit-human` mode that prints readable, concise summaries per frame (last‑stage sanity) without replacing JSONL.
- Exhaustive debug keys:
  - Every threshold/gate value used in a decision must be captured alongside its inputs and boolean result (e.g., `psi_gate_est`, `psi_gate_thresh`, `psi_gate_on`), plus the winning cap and its competitors.
  - Record units explicitly for critical fields: `{value, unit}` or parallel `*_unit` keys.
  - Record conversion constants actually applied in that step (e.g., `CV.MPH_TO_MS`, `CV.MS_TO_MPH`).
- CLI/IO:
  - `--out OUT.jsonl` required; create parent directories.
  - Support `--max-frames` to bound runtime and file size.
  - `--disable-failopen` must flip any freeway fail‑open flag the controller would otherwise set.

Validation Invariants (express as pytest tests)
- Units:
  - `v_base_mph ≈ v_base_mps * 2.236936` within 1e‑3 relative when both are present.
  - No negative speeds; all curvature finite; `psi_vis >= 0`, `psi_thresh >= 0`.
- Gating:
  - If `active_cap == 'occlusion'` then `psi_vis >= psi_thresh - 1e-6` unless an explicit override flag is set; otherwise mark failure.
  - When double‑cap guard is true and `pre_cap_target <= cap_occl_vmin + eps`, `active_cap` must not be `occlusion`.
- Fail‑open:
  - When `|kappa_vis| < FREEWAY_CURV_EPS` and `s_visible_m >= FREEWAY_MIN_VISIBLE_M` and `path_conf >= FREEWAY_MIN_CONF`, `active_cap` is not `occlusion`.

Scenario Synthesis (generate synthetic modelV2 + carState streams)
- Provide a `docs/chauffeur/vtsc/fullTrace/scenario_generator.py` used by tests to synthesize sequences that hit problematic logic consistently. Do not modify VTSC; synthesize inputs.
- Implement helpers to emit evenly spaced horizons of length `N_POINTS` (≤ len(ModelConstants.T_IDXS)) for:
  - Straight, high‑confidence, long‑visibility (fail‑open baseline)
  - Gentle curves with no occlusion (good behavior baseline)
  - Gradual occlusion onset (crest), then recovery (fov_exit)
  - Lead‑vehicle occlusion (vary headway), including spurious pretrigger with high confidence
  - Mountain‑side occlusion: low curvature but poor visibility from terrain, psi close to threshold
  - Multi‑apex (S‑curve), increasing and decreasing curvature trends to exercise tail growth/decay
  - Short‑visibility urban segments with frequent gating toggles
  - Edge cases: near‑zero curvature noise, unit conversion stress (mph↔m/s), zero/NaN guards
- Each scenario yields a deterministic list of pseudo‑`modelV2` objects and kinematics `(v_ego, a_ego)` over time; tests feed these through the harness’s VTSC invocation loop.

Metrics & Aggregations (produce in harness or tests)
- For both rlog replays and synthetic scenarios, compute counts and rates for:
  - `active_cap` distribution; overslow events (`v - final ≥ 2.0 m/s`) by cap
  - PSI mismatches: `active_cap=occlusion` while `psi_vis < psi_thresh`
  - Double‑cap guard violations and pretrigger_with_high_conf occurrences
  - Recovery speed: frames from fov_exit to cap switch away from occlusion under stable visibility
- Emit TSV summaries next to JSONL (optional): `<run>.metrics.tsv` to ease spreadsheet review.

GitHub Large‑File Note
- All linked artifacts are ≤ ~12 MB and accessible via `raw.githubusercontent.com`. Process JSONL line‑by‑line to avoid memory spikes. Use `--max-frames` when demonstrating long runs.

Example CLI
- Full trace on the committed overslow case (segment 67), 300 frames:
  - `python docs/chauffeur/vtsc/fullTrace/gpt5pro_full_trace_harness.py docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_00000085--f247b281ca--67.zst --out .cache/vtsc_full_trace_67.jsonl --max-frames 300`
- Lighter snapshot‑only replay (reference):
  - `python docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_00000085--f247b281ca--67.zst --out .cache/vtsc_snap_67.jsonl --max-frames 300`

Acceptance Criteria
- Harness runs without modifying VTSC logic and produces deterministic JSONL on the provided rlog.
- Tests assert the invariants above and fail if gating/units/double‑cap protections regress.
- The record includes enough breadcrumbs to reconstruct any speed target change back to the contributing thresholds and inputs.

Notes
- Keep all new code self‑contained in `docs/chauffeur/vtsc/fullTrace/` and tests nearby; do not edit production VTSC code as part of this harness.
- If a calculation is deeply internal, prefer wrapping surrounding methods and computing the extra diagnostics externally in the harness rather than patching VTSC.

What To Return
- The complete `gpt5pro_full_trace_harness.py` file content and the small `pytest` files.
- A short explanation of how to run them locally (commands above are sufficient).
