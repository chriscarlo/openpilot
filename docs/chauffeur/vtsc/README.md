MTSC Diagnostics — Snapshot Index

- Latest run: `mtsc_diag_20250903_005236`
- Summary verdict: service_schema

What this means
- Only `mapTurnSpeedControlSP` is stale in the 15 s live probe; other live topics are healthy.
- `mtscd` is not running as RT (SCHED_OTHER, rtprio 0) and shows low CPU.
- Kernel log sample has no RT throttling or watchdog stalls.
- `loggerd` CPU is elevated (>25%), but no cross-topic stalls were observed.

Key files
- Final report JSON: `mtsc_diag_20250903_005236/F_final_report.json`
- Live liveness probe: `mtsc_diag_20250903_005236/C_live_probe.json`
- Service entry: `mtsc_diag_20250903_005236/B_services.json`
- Scheduler/load snapshots: files prefixed with `A_` and `E_` in the same folder
- Rlog rates (if found): `mtsc_diag_20250903_005236/D_rlog_rates.json`

Next steps
- Verify the MTSC publisher is launched/enabled on chubbs-merge.
- Check params/toggles gating `mapTurnSpeedControlSP` publication.
- If needed, expand rlog analysis to more segments to compare presence/absence.

