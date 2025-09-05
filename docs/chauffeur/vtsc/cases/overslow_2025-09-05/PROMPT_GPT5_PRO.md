# Prompt: VTSC Overslow Diagnostics (Request for Analysis)

We’re investigating persistent VTSC overslow behavior during the ~17:30–18:30 local driving window (≈ 00:43–01:43 UTC). This repo contains an offline analysis and a representative rlog segment with derived diagnostics that can be reviewed directly on GitHub (no code execution needed).

## Artifacts to Review
- Diagnostics overview:
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/debug/debug_2025-09-05/DIAGNOSTICS_OVERVIEW.md
- Cross‑window summary (hour):
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/debug/debug_2025-09-05/vtsc_watch_offline_2025-09-05_004324UTC_to_2025-09-05_014324UTC_summary.txt
- Per‑segment metrics (hour window):
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/debug/debug_2025-09-05/window_segment_metrics.tsv
- VTSC vs Vision scan (latest 20 segs):
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/debug/debug_2025-09-05/analyze_vtsc_vs_vision.txt
  
### Off‑road FOV Gate Evaluation (per‑segment)
- 00000085--f247b281ca--80
  - metrics.json: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_060843/metrics.json
  - by_log.jsonl: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_060843/by_log.jsonl
- 00000085--f247b281ca--67
  - metrics.json: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_060856/metrics.json
  - by_log.jsonl: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/offroad/reports/vtsc_offroad_20250905_060856/by_log.jsonl
- Case A — visible‑cap overslow (top visible overslow):
  - Folder: https://github.com/chriscarlo/chauffeur/tree/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67
  - Report: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/CASE_REPORT.md
  - Rlog: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_00000085--f247b281ca--67.zst
  - TSV: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/vtsc_events_00000085--f247b281ca--67.tsv
  - Flagged: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/flagged_00000085--f247b281ca--67.log
- Case B — occlusion‑cap focus (psi_below_thresh cluster):
  - Folder: https://github.com/chriscarlo/chauffeur/tree/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80
  - Report: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/CASE_REPORT.md
  - Rlog: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/rlog_00000085--f247b281ca--80.zst
  - TSV: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/vtsc_events_00000085--f247b281ca--80.tsv
  - Flagged: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/flagged_00000085--f247b281ca--80.log
- TSV Field Reference:
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/cases/overslow_2025-09-05/DATA_DICTIONARY.md

## What the Data Shows (high level)
- Hour summary: 1612 matched VTSCDBG events; 190 flagged by the watcher.
  - Flags: `pretrigger_with_high_conf` (103), `double_occl_cap_suspect` (53), `psi_below_thresh` (37).
  - Overslow (v − final ≥ 2.0 m/s): 1112/1406 events; by cap: visible=819, occlusion=293.
  - Top reasons during overslow: `fov_exit` (925), `pretrigger` (176), `short_vis` (11).
- Segment `00000085--f247b281ca--80` (00:56:54–00:57:38 UTC):
  - VTSCDBG events: 84
  - Overslow count: 11 (all under `occlusion` with reason `fov_exit`)
  - Watcher flags: `psi_below_thresh` (36), `double_occl_cap_suspect` (7)
  
- Segment `00000085--f247b281ca--67` (00:43:54–00:44:54 UTC):
  - VTSCDBG events: 113
  - Overslow count: 112 (visible=110, occlusion=2)
  - Overslow reasons: `fov_exit` (75), `pretrigger` (37)
  - Watcher flags: `pretrigger_with_high_conf` (11), `double_occl_cap_suspect` (1)

## Patch Under Test (what changed)
We’ve applied a targeted VTSC update in this branch (see source files below) to address overslow under occlusion:
- PSI‑gated occlusion arbitration: occlusion cap participates only when `psi_vis ≥ psi_thresh − hyst`.
- Double‑cap guard: skip occlusion when the pre‑cap target is already ≤ occl vmin + ε.
- fov_exit relax: after short dwell at near‑zero confidence, nudge occl vmin up toward visible vmin (bounded by visible) to avoid crawl.
- Telemetry: added `_dbg_psi_est`, `_dbg_psi_thresh`, `_dbg_consider_occl`, `_dbg_double_cap_guard`, `_dbg_pre_cap_target` in VTSCDBG.

Source (for reference):
- `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
- `sunnypilot/selfdrive/controls/lib/vision_turn_params.py`

Expected impact to verify on future captures:
- Fewer `psi_below_thresh` coincident with `active_cap=occlusion`.
- Fewer `double_occl_cap_suspect` events.
- Reduced `fov_exit` overslow under occlusion (especially in segment‑like scenarios similar to ...--80).

## Hypotheses / Suspicions
1) Psi gating may be too sticky or inconsistently applied: numerous frames show `psi_below_thresh` while occlusion cap remains active with very low vmin (~2.7 m/s), conf near 0.0.
2) Possible double‑capping: `raw`≈`cap_occl_vmin` concurrent with `active_cap=occlusion`, suggesting redundant limiting suppresses recovery.
3) FOV exit handling may be overly conservative, maintaining occlusion vmin lock even as conditions improve (tail/visibility dynamics may be slow to relax).

## What We’re Asking You To Do
Please analyze the provided artifacts and:
- Identify gating inconsistencies between `psi_vis` and `psi_thresh` when `active_cap=occlusion` and reason=`fov_exit`.
- Determine if our occlusion min velocity computation and application are causing double limitation (e.g., raw already at occl vmin while occlusion cap also wins), and propose a single‑source‑of‑truth or priority rule to avoid double‑capping.
- Suggest tuning or conditional logic for faster recovery from `fov_exit` with low confidence (e.g., minimal safe vmin, hysteresis, or decay of occlusion influence) without compromising safety.
- Recommend additional derived metrics we should log to make future diagnosis easier (e.g., explicit occlusion arbitration inputs/outputs, hysteresis states, transition timers).

## Constraints / Context
- This analysis is offline; you cannot run code in GitHub. Use the TSV, logs, and rlog.zst for reference.
- Offroad toggles for VTSC debug were enabled during collection.
- Safety requirements apply: changes must preserve safe decel behavior and avoid high‑risk fail‑open scenarios.
- Branch: `chubbs-merge` at commit `aa3b5d957a3c`.
 - Branch: `chubbs-merge` at commit `520889f04`.
 - Environment snapshot: https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/docs/chauffeur/vtsc/debug/debug_2025-09-05/ENVIRONMENT.md

## Preferred Output
- A written analysis with concrete findings tied to fields in the TSV/flagged logs.
- A precise set of code changes or a PR against branch `chubbs-merge` improving:
  - Psi gating consistency and/or thresholds
  - Occlusion cap arbitration to prevent double‑capping
  - Recovery dynamics post‑`fov_exit`
- If changes are non‑trivial, include tests or a stepwise rollout plan with guardrails.

Thank you.

## Code Pointers
- Controller: vision turn logic and cap arbitration
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/sunnypilot/selfdrive/controls/lib/vision_turn_controller.py
  - Helpful identifiers to search:
    - Psi gating and clear: `psi_vis`, `psi_thresh`, `onset`, `clear` (e.g., around gating helpers and debug fields).
    - Cap computation and arbitration: `cap_visible_vmin`, `cap_occl_vmin`, `caps = [("visible", ...)]`, `active_cap = min(caps, ...)`.
    - Debug fields published to VTSCDBG: `_dbg_active_cap`, `_dbg_cap_visible_vmin`, `_dbg_cap_occl_vmin`.
- Params and thresholds (lookahead, confidence hysteresis):
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/sunnypilot/selfdrive/controls/lib/vision_turn_params.py
- Watcher/parser used to render logs (for field mapping):
  - https://github.com/chriscarlo/chauffeur/blob/chubbs-merge/tools/vtsc/vtsc_watch.py
