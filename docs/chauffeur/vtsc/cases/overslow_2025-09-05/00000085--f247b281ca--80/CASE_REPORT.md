# VTSC Overslow Case: 00000085--f247b281ca--80 (2025-09-05)

This package contains a single rlog segment and derived diagnostics for investigating VTSC overslow behavior observed around 00:56:54–00:57:38 UTC.

## Files
- `rlog_00000085--f247b281ca--80.zst`: Copied rlog for this segment.
- `vtsc_events_00000085--f247b281ca--80.tsv`: Per-event VTSCDBG extraction (human time, key fields, overslow flag).
- `flagged_00000085--f247b281ca--80.log`: Flagged lines from the offline watcher for this segment.
- Cross‑window summary for the hour: `../../../../debug/debug_2025-09-05/vtsc_watch_offline_2025-09-05_004324UTC_to_2025-09-05_014324UTC_summary.txt`.

## Segment Metrics
- Segment: `00000085--f247b281ca--80`
- Window: 2025-09-05 00:56:54 → 00:57:38 UTC
- Events (VTSCDBG): 84
- Overslow (v - final ≥ 2.0 m/s): 11
- Overslow by cap: occlusion=11, visible=0, other=0
- Overslow reasons: fov_exit=11
- Watcher flags in this segment:
  - psi_below_thresh: 36
  - double_occl_cap_suspect: 7

## Observations
- Dominant gating state during overslow is `occlusion` with reason `fov_exit`; confidence is frequently near 0.0.
- Many frames show `psi_below_thresh` while occlusion cap is active. Review whether occlusion gating behavior fully respects psi thresholding, or if there’s intentional hysteresis that is too sticky.
- Several `double_occl_cap_suspect` cases indicate raw target ≈ occlusion vmin while occlusion cap is concurrently active in `active_cap`, suggesting potential double-capping or redundant limiting logic.
- Tail and visibility indicators often show a high tail fraction and capped visibility velocity near ~2.7 m/s, hinting at a low occlusion vmin lock while vehicle speed and base targets are substantially higher.

## How These Were Produced (offline)
- VTSC watcher run offline over 17:30–18:30 local (≈ 00:43–01:43 UTC):
  - Flagged summary: `docs/chauffeur/vtsc/debug/debug_2025-09-05/..._summary.txt`
  - Flagged lines: `docs/chauffeur/vtsc/debug/debug_2025-09-05/..._flagged.log`
- Worst offender segment selected by highest flagged count in the window: `00000085--f247b281ca--80` (42 flagged lines; psi_below_thresh=36).
- TSV extracted directly from the selected `rlog.zst` with key VTSC fields and an `overslow` indicator.

## Suggested Lines of Inquiry
- Psi gating: verify consistency between `psi_vis` vs `psi_thresh`; ensure occlusion cap disengages promptly when `psi_vis` < threshold unless other guards apply.
- FOV exit handling: validate rationale for `fov_exit` maintaining occlusion cap at very low vmin while confidence is ~0.0; consider fail‑open behavior and minimal safe vmin.
- Double-capping: audit how `raw`, `cap_occl_vmin`, and `active_cap` interact; avoid redundant limiting that suppresses recovery.
- Transition smoothing: check whether tail dynamics or smoothing are overly conservative post‑occlusion leading to sustained undershoot.

## Pointers
- Segment rlog: `rlog_00000085--f247b281ca--80.zst`
- Segment TSV: `vtsc_events_00000085--f247b281ca--80.tsv`
- Segment flagged lines: `flagged_00000085--f247b281ca--80.log`
- Window summary: `../../../../debug/debug_2025-09-05/vtsc_watch_offline_2025-09-05_004324UTC_to_2025-09-05_014324UTC_summary.txt`

