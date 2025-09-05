# VTSC Diagnostics Overview — 2025-09-05 (17:30–18:30 local)

Window: 2025-09-05 00:43:24–01:43:24 UTC

## Hour Summary (from watcher)

window: 2025-09-05_004324UTC -> 2025-09-05_014324UTC
files: 16
events_matched: 1612
flagged_events: 190
pretrigger_with_high_conf: 103
double_occl_cap_suspect: 53
psi_below_thresh: 37

[derived_overslow_metrics]
threshold: v - final >= 2.0 m/s
events: 1406
overslow_total: 1112
overslow_by_cap: visible=819 occlusion=293 other=0
overslow_by_reason_top: fov_exit=925, pretrigger=176, short_vis=11


## Top Segments by Overslow

- 00000085--f247b281ca--68: overslow=113 visible=73 occlusion=40 reasons={"fov_exit":77,"pretrigger":36}
- 00000085--f247b281ca--70: overslow=113 visible=98 occlusion=15 reasons={"fov_exit":113}
- 00000085--f247b281ca--67: overslow=112 visible=110 occlusion=2 reasons={"fov_exit":75,"pretrigger":37}
- 00000085--f247b281ca--74: overslow=110 visible=109 occlusion=1 reasons={"fov_exit":56,"pretrigger":49,"short_vis":5}
- 00000085--f247b281ca--69: overslow=109 visible=85 occlusion=24 reasons={"fov_exit":77,"pretrigger":29,"short_vis":3}
- 00000085--f247b281ca--75: overslow=104 visible=84 occlusion=20 reasons={"fov_exit":104}
- 00000085--f247b281ca--73: overslow=101 visible=98 occlusion=3 reasons={"fov_exit":101}
- 00000085--f247b281ca--71: overslow=97 visible=75 occlusion=22 reasons={"fov_exit":69,"pretrigger":25,"short_vis":3}

## Cases

- Case A (visible-cap heavy): 00000085--f247b281ca--67
  - Case folder: cases/overslow_2025-09-05/00000085--f247b281ca--67
  - See CASE_REPORT.md and TSV/flagged files therein.
- Case B (occlusion-cap focus): 00000085--f247b281ca--80
  - Case folder: cases/overslow_2025-09-05/00000085--f247b281ca--80
  - See CASE_REPORT.md and TSV/flagged files therein.

## Additional Artifacts

- Per-segment metrics TSV: debug/debug_2025-09-05/window_segment_metrics.tsv
- VTSC vs Vision scan: debug/debug_2025-09-05/analyze_vtsc_vs_vision.txt
- TSV Field Reference: cases/overslow_2025-09-05/DATA_DICTIONARY.md
- Environment snapshot: debug/debug_2025-09-05/ENVIRONMENT.md

## Snapshot Analysis (on-device JSONL)

- Source: `/data/media/0/VTSCDebug/vtsc_snapshots.jsonl` (221 points)
- Summary:
  - Vision states: FULL=60, PARTIAL=16, SEVERE=145
  - straight_no_crawl_fail: 16/66 (24.24%)
  - highway_bypass_fail: 0/16 (0.00%)
  - lead_bypass_fail: 6/8 (75.00%)
  - map_cap_misuse: 16/16 (100.00%)
  - reacq_nudge_fail: 1/20 (5.00%)
  - jerk_or_comfort_violations: 24/221 (10.86%)
- TSV: debug/debug_2025-09-05/vtsc_snapshots.tsv

## Off-road FOV Gate Evaluation (rlogs)

- Segment 00000085--f247b281ca--80 report:
  - metrics.json: offroad/reports/vtsc_offroad_20250905_060843/metrics.json
  - by_log.jsonl: offroad/reports/vtsc_offroad_20250905_060843/by_log.jsonl
- Segment 00000085--f247b281ca--67 report:
  - metrics.json: offroad/reports/vtsc_offroad_20250905_060856/metrics.json
  - by_log.jsonl: offroad/reports/vtsc_offroad_20250905_060856/by_log.jsonl

Notes: Off-road gate evaluation replays gating heuristics on existing VTSCDBG frames and does not reflect on-device controller arbitration changes beyond gating.
