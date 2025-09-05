# VTSC Overslow Case: 00000085--f247b281ca--67 (2025-09-05)

This package contains the rlog segment and derived diagnostics for investigating VTSC overslow in this segment.

## Files
- rlog: `rlog_00000085--f247b281ca--67.zst`
- TSV: `vtsc_events_00000085--f247b281ca--67.tsv`
- Flagged: `flagged_00000085--f247b281ca--67.log`

## Segment Metrics
- Events (VTSCDBG): 113
- Overslow (v - final ≥ 2.0 m/s): 112
- Overslow by cap: {'visible': 110, 'occlusion': 2}
- Overslow reasons: {'fov_exit': 75, 'pretrigger': 37}
- Watcher flags: {'pretrigger_with_high_conf': 11, 'double_occl_cap_suspect': 1}

## Observations
- This segment is a visible-cap overslow heavy case within the hour window.
- Focus on transitions around fov_exit and pretrigger, and verify raw vs cap_visible_vmin behavior.
