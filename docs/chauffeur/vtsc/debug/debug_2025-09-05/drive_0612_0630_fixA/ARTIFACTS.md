# VTSC Diagnostics Artifacts — 2025-09-05 (Drive 06:12–06:30 local) — Fix A

Scope: branch `chubbs-merge` • window processed via file mtimes ~13:12–13:30 local (≈06:12–06:30 in user’s TZ)

This tranche packages replay outputs after implementing Fix A (see prompt) for GPT‑5 Pro analysis.

Artifacts in this repo for offline analysis:

- Replay outputs (JSONL) for 19 segments and combined view:
  - `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330_fixA/combined.jsonl`
  - Per-segment JSONL under the same folder.
- Analyzer summary (updated reacq condition):
  - `docs/chauffeur/vtsc/offroad/replays/run_20250905_1312_1330_fixA/combined_summary_after_analyzer_update.txt`

Notes:
- Replay JSONL fields include: `v, v_base, v_vis, v_occ, cap_visible_vmin, cap_occl_vmin, vtsc_cmd, final, a_cmd, active_cap, path_conf, kappa_vis, s_visible_m, psi_vis, psi_thresh, vision_status`.
- The analyzer’s reacq_nudge check only triggers when `final < 0.98×v_base` at the FULL transition.

