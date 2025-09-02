# VTSC Tests Overview

This page summarizes the developer-facing VTSC test suites, how to run them locally, and how to capture on-road snapshots for diagnosing discrepancies between dev tests and real-world behavior.

## Quick Start
- Install dev deps: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[testing,dev]"`
- Build native targets (if needed): `scons -j$(nproc)`
- Fast tests: `pytest -m 'not slow'`

## Module Co-Located Scenario Tests
Location: `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py`

Scenarios covered:
- Straight road, clear vision: validates no crawl and healthy target speeds.
- Highway bypass under partial occlusion: ensures occlusion gates relax ≥ ~65 mph.
- Lead-aware bypass: with a lead ≤ ~3 s time headway under occlusion, allows positive accel when margin exists.
- Partial occlusion without lead: blocks positive accel unless positive margin; decel clamped to comfort caps.
- Severe occlusion → reacquisition: adds a small positive nudge on reacquisition for quick reconvergence.
- Hidden turn/tail growth: early decel under tightening curvature with jerk/comfort caps respected.
- Map lookahead present: applies a cap with coverage; cap stays inactive when absent (negative case).
- Occlusion dwell/hysteresis stability: bounces confidence around thresholds to verify dwell times prevent oscillation.

Run just these: `pytest sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py -q`

## Extended VTSC Testbenches (Docs Area)
Location: `docs/chauffeur/vtsc/testing/`

Useful entry points:
- `harness/` and `integration/`: synthetic time-stepped simulations with metrics windows (apex/reacq).
- `filtering/` and `adaptive_deceleration/`: EMA and decel subsystem checks, including fast-reacq.
- `acceptance/test_vtsc_acceptance.py`: invariants like monotonic-while-occluded and jerk caps.
- `run_all_tests.py` / `run_all_from_root.sh`: convenience runners from repo root.

Example: `pytest docs/chauffeur/vtsc/testing -q`

## On-Road Snapshot Debugging

To understand dev vs. real-world discrepancies, VTSC can emit compact snapshots with key decision variables.

Toggles (persistent; Survive reboot): Settings → Cruise → VTSC → Settings
- Map Lookahead for VTSC (`MTSCLookaheadEnabled`): include map tail cap in planning.
- Occlusion Bypass When Following Lead (`VisionTurnSpeedControlOcclBypassWithLead`): relaxes occlusion under a lead.
- Verbose VTSC Debug Logging (`VTSCVerboseDebug`): logs `VTSCDBG` JSON to standard logs at ~2 Hz.
- Write Onroad VTSC Snapshots (JSONL) (`VTSCWriteSnapshotFile`): appends snapshots to a small rotating file.

Snapshot file path (when enabled): `/data/media/0/VTSCDebug/vtsc_snapshots.jsonl`

Tail and filter on-device:
- `adb shell "tail -f /data/media/0/VTSCDebug/vtsc_snapshots.jsonl"`
- Pretty-print: `... | jq '
  {ts, v: .v, cruise, conf, vision: .vision_status, lead: .lead, hw: .hw,
   v_base: .v_base, v_occ: .v_occ, final, map_cap: .map_tail_cap,
   occl_margin: .occl_positive_margin, bypass: .occl_lead_bypass_active,
   decel: .decel_cmd, jerk: .jerk_cmd}'
`

Primary fields:
- Vision: `conf`, `vision_status`, occlusion dwell and tail context (`s_tail`, `tail_frac`).
- Speed sources: `v_base` (physics base), `v_occ` (occlusion-estimated), `final` (after caps), `map_tail_cap`.
- Lead context: `lead`, `hw`, `occl_lead_bypass_active`, `occl_positive_margin`.
- Limits and commands: `comfort_decel`, `decel_cmd`, `jerk_cmd`.

Tips for discrepancy investigation:
- Confirm whether slowdowns correlate with `vision_status != FULL` despite straight roads (check `conf`, `k_model`).
- Verify `map_tail_active` and `map_tail_cap` only engage with reasonable coverage and ahead of visible horizon.
- For crawl with a lead, check `occl_lead_bypass_active` and `occl_positive_margin`.
- After reacquisition, confirm `a_last` ≥ ~0.18 for nudge, and quick reconvergence of `final → v_base`.

## Notes
- Snapshots are compact and rate-limited; the JSONL file auto-rotates (small size).
- All tests run locally via `pytest` without device dependencies.
- For broad sweeps/tuning, see `docs/chauffeur/vtsc/testing/sweep/`.

