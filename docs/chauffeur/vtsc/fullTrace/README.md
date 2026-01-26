VTSC Full‑Trace Replay Harness

Purpose
- Replays recorded rlogs through the current VisionTurnController (VTSC), capturing method‑level pre/post state changes and the final VTSC snapshot for each model frame.
- Enables precise, reproducible debugging of early/over‑slowing, gating decisions, and caps arbitration without on‑road testing.

Data provenance (important)
- For VTSC behavior regressions, rlog input data should originate from a real drive on the target device.
  - In this repo/workspace that means **TICI / comma3x**.
- Some rlogs may appear “local” in this git checkout because they were copied over from the device and committed as fixtures.
  - Treat them as **tici-sourced** even when running analysis/tests on a dev laptop.
  - Avoid substituting sim/desktop-generated logs for these cases; timing/message-shape differences can mask overslow/recovery issues.

Files
- `full_trace_replay.py`: CLI tool to run a full VTSC trace over an rlog and write JSONL output.
- `examples/trace_event_schema.json`: Example of one replay step’s JSON shape for downstream tooling (e.g., GPT‑5 Pro prompt or notebooks).
- `examples/vtsc_full_trace_sample.jsonl`: Sample full‑trace JSONL (20 frames) from an overslow case rlog segment.
- `prompt_full_trace_gpt5pro.md`: Ready‑to‑use analysis prompt for GPT‑5 Pro.

Quick Start
- Run with one of the committed rlogs:

  ```
  python docs/chauffeur/vtsc/fullTrace/full_trace_replay.py \
    docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_00000085--f247b281ca--67.zst \
    --out .cache/vtsc_full_trace_67.jsonl --max-frames 300
  ```

- Optional flags:
  - `--cruise <mps>`: Constant cruise setpoint (default: use v_ego)
  - `--max-frames N`: Limit frames for quick iteration
  - `--disable-failopen`: Force‑disable freeway fail‑open gating

Output JSONL structure (per model frame)
- Top‑level fields:
  - `idx`: 0‑based frame index
  - `ts`: seconds (float) from rlog event (if available)
  - `inputs`: {`v_ego`, `a_ego`, `v_cruise`}
  - `trace`: ordered list of method entries, each with:
    - `phase`: `pre` or `post`
    - `method`: `_update_params|_update_calculations|_state_transition|_update_solution|_plan_advanced_speed_trajectory`
    - `state`: selected controller attributes, occlusion state, gating and cap diagnostics
  - `snapshot`: `VisionTurnController.snapshot_debug_state()` output (compact, stable for analyzers)

What’s captured in `state`
- Kinematics: `_v_ego, _a_ego, _prev_target_speed, _current_accel, _a_target`
- Curvature/physics: `_filtered_curvature, _current_lat_acc, _max_pred_lat_acc`
- Occlusion model: `vision_good, vision_status, smoothed_confidence, last_valid_curvature, est_curvature, distance_since_m, occluded_since_time, reacquired_at`
- Gating & FOV: `_fov_occluded, _dbg_psi_vis, _dbg_psi_thresh, _psi_fov_rad, _psi_margin_rad, _freeway_failopen_active`
- Caps arbitration: `_dbg_active_cap, _dbg_cap_visible_vmin, _dbg_cap_occl_vmin, _dbg_cap_map_vmin, _dbg_vtsc_cmd, _dbg_tail_frac, _dbg_s_tail, _dbg_pre_cap_target`
- Tunables & windows: `_vis_horizon_s, _vis_margin_m, _lat_jerk_cap, _anticipation_target_reduction, _anticipation_max_reduction_mps, _onset_no_raise_active, _occlusion_onset_timer_s, _v_cap_active_at_onset_mps`

Notes
- Harness uses method wrapping (monkey‑patch) to capture `pre`/`post` state without modifying VTSC logic.
- This is read‑only with respect to rlogs; produced JSONL can be large — use `--max-frames` for development.
- Pair with `docs/chauffeur/vtsc/analysis/analyze_snapshots.py` or your own notebooks to aggregate decisions and flags.
