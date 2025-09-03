MTSC Diagnostic Findings — 2025-09-03 00:52:36

- Verdict: service_schema
- Rationale:
  - Only `mapTurnSpeedControlSP` is stale in the 15 s live probe; other focus topics are healthy.
  - `mtscd` is non-RT (`SCHED_OTHER`, `rtprio=0`), CPU ~1.4%, affinity `0-7`.
  - No kernel RT throttling or watchdog issues in sampled dmesg.
  - `loggerd` CPU elevated (>25%) without cross-topic stalls → unlikely root cause.

Pointers
- Final JSON report: `F_final_report.json`
- Live probe: `C_live_probe.json`
- Services entry: `B_services.json`
- Scheduler/load: `A_*` and `E_*`
- Rlogs: `D_rlog_rates.json` (none found in last 3 by mtime during this snapshot)

Next actions
- Verify MTSC publisher is started/enabled on chubbs-merge.
- Check params/toggles gating `mapTurnSpeedControlSP` publication.
- Optionally extend rlog window for presence/absence comparison.

