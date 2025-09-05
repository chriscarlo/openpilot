# VTSC Diagnostics Artifacts — 2025-09-05 (Drive 06:12–06:30 local)

Scope: branch `chubbs-merge` @ 921fef98cf77 • window processed via file mtimes ~13:12–13:30 local (≈06:12–06:30 in user’s TZ)

Artifacts in this repo for offline analysis:

- Replay outputs (JSONL/TSV) for 19 segments:
  - `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/combined.jsonl`
  - `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/combined.tsv`
  - Per-segment JSONL under the same folder.

- Snapshot analyzer summary:
  - `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330/combined_summary.txt`

- Off-road FOV-gate reports (aggregates and per-log):
  - Static gate (controller default hysteresis): see latest timestamped folder under `docs/chauffeur/vtsc/offroad/reports/` prior to 14:59 UTC on 2025-09-05.
  - Tuned hysteresis (n_on=2, n_off=12, force fallback path):
    - `docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_145922/metrics.json`
    - `docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_145922/by_log.jsonl`

Notes:
- Replay JSONL fields include: `v, v_base, v_vis, v_occ, cap_visible_vmin, cap_occl_vmin, vtsc_cmd, final, active_cap, path_conf, kappa_vis, s_visible_m, psi_vis, psi_thresh, a_cmd, jerk_cmd, decel_cmd, vision_status, occlusion_reason`.
- The TSV mirrors a subset for quick plotting and awk/jq filtering.

