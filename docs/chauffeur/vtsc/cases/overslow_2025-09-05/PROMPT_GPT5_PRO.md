# Prompt: VTSC Overslow Diagnostics (Request for Analysis)

We’re investigating persistent VTSC overslow behavior during the ~17:30–18:30 local driving window (≈ 00:43–01:43 UTC). This repo contains an offline analysis and a representative rlog segment with derived diagnostics that can be reviewed directly on GitHub (no code execution needed).

## Artifacts to Review
- Cross‑window summary (hour):
  - `docs/chauffeur/vtsc/debug/debug_2025-09-05/vtsc_watch_offline_2025-09-05_004324UTC_to_2025-09-05_014324UTC_summary.txt`
- Selected “worst‑offender” segment package (highest flagged count):
  - `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/CASE_REPORT.md`
  - `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/rlog_00000085--f247b281ca--80.zst`
  - `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/vtsc_events_00000085--f247b281ca--80.tsv`
  - `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/flagged_00000085--f247b281ca--80.log`

## What the Data Shows (high level)
- Hour summary: 1612 matched VTSCDBG events; 190 flagged by the watcher.
  - Flags: `pretrigger_with_high_conf` (103), `double_occl_cap_suspect` (53), `psi_below_thresh` (37).
  - Overslow (v − final ≥ 2.0 m/s): 1112/1406 events; by cap: visible=819, occlusion=293.
  - Top reasons during overslow: `fov_exit` (925), `pretrigger` (176), `short_vis` (11).
- Segment `00000085--f247b281ca--80` (00:56:54–00:57:38 UTC):
  - VTSCDBG events: 84
  - Overslow count: 11 (all under `occlusion` with reason `fov_exit`)
  - Watcher flags: `psi_below_thresh` (36), `double_occl_cap_suspect` (7)

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

## Preferred Output
- A written analysis with concrete findings tied to fields in the TSV/flagged logs.
- A precise set of code changes or a PR against branch `chubbs-merge` improving:
  - Psi gating consistency and/or thresholds
  - Occlusion cap arbitration to prevent double‑capping
  - Recovery dynamics post‑`fov_exit`
- If changes are non‑trivial, include tests or a stepwise rollout plan with guardrails.

Thank you.

