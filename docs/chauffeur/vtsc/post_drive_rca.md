# VTSC Post‑Drive RCA Guide (Zero Context)

This guide is for a clean, future session with no prior context. It explains exactly where to look and what to analyze after a short drive with the following toggles enabled in Offroad UI → Settings → Cruise → VTSC → Settings:

- Map Lookahead for VTSC (`MTSCLookaheadEnabled`) → ON
- Verbose VTSC Debug Logging (`VTSCVerboseDebug`) → ON
- Write Onroad VTSC Snapshots (JSONL) (`VTSCWriteSnapshotFile`) → ON

The controller writes compact snapshots of key decisions at ~2 Hz to:
- `/data/media/0/VTSCDebug/vtsc_snapshots.jsonl` (one JSON object per line, auto‑rotated)

You only need this file for analysis.

## 1) Retrieve Snapshot File
- ADB: `adb pull /data/media/0/VTSCDebug/vtsc_snapshots.jsonl ./vtsc_snapshots.jsonl`
- Or, SSH/SCP: `scp device:/data/media/0/VTSCDebug/vtsc_snapshots.jsonl ./`

If the file is empty, confirm the three toggles above are ON and drive for 2–5 minutes.

## 2) Run the Analyzer (local)
We provide a simple analyzer script. Run:

- `python docs/chauffeur/vtsc/analysis/analyze_snapshots.py ./vtsc_snapshots.jsonl`

It summarizes core metrics and flags common mismatch patterns. If you cannot run it, proceed with Section 3 using `jq`.

## 3) Manual Quick Triage (jq)
Useful projections while tailing or after pulling:

- Basic fields: `jq '{ts, v: .v, cruise, conf, vision: .vision_status, lead, hw, final, v_base, v_occ, map_cap: .map_tail_cap, map_active: .map_tail_active, cov: .map_tail_coverage, bypass: .occl_lead_bypass_active, margin: .occl_positive_margin, decel: .decel_cmd, jerk: .jerk_cmd, a: .a_cmd}' vtsc_snapshots.jsonl | less`

Focus on these checks:

- Straight‑road crawl:
  - Look for `conf ≥ 0.8`, `vision == FULL`, near‑zero curvature context (`v_base` near `cruise`).
  - Problem: `final < v_base - 2.0` m/s for many consecutive samples.

- Highway occlusion bypass:
  - Look at samples with `v ≥ 29 m/s` and `vision != FULL`.
  - Problem: `final < v_base - 1.0` m/s (occlusion depressing target when it should be gated at highway speeds).

- Lead‑aware bypass:
  - With `lead == true` and `hw ≤ ~3.0`, during occlusion (`vision != FULL`).
  - Problem: `bypass == false` or `a (a_cmd) ≤ 0` for extended periods (no positive accel allowed despite margin).

- Map lookahead misuse:
  - When `map_active == true` but `cov < 0.2` (poor coverage) or on straight segments (compare `v_base`).
  - Problem: `map_cap` significantly below `v_base` on straight road or with poor coverage.

- Reacquisition nudge:
  - Find transitions where `vision` goes from not `FULL` to `FULL`.
  - Expect within ~0.65s: `a (a_cmd) ≥ 0.18` at least once (small positive push toward base).

- Jerk/comfort guardrails:
  - `decel` should be ≥ `comfort_decel` when occluded (not exceeding comfort decel in magnitude).
  - `jerk` bounded roughly in `[-6.0, +2.5]` m/s³ during typical operation.

## 4) Automated Analyzer Output
`analyze_snapshots.py` computes and prints:
- Time span, points, avg speed, share of `vision` states
- Flags:
  - `straight_no_crawl_fail` (count/ratio)
  - `highway_bypass_fail`
  - `lead_bypass_fail`
  - `map_cap_misuse`
  - `reacq_nudge_fail`
  - `jerk_or_comfort_violations`

It also prints short “why” hints per flag to accelerate RCA.

## 5) What To Send Back
Provide either:
- The analyzer summary output, or
- A short note listing which checks above tripped (e.g., “highway_bypass_fail with vision=PARTIAL at 30–33 m/s; final 3–5 m/s under v_base; map_active=false”).

Given only that summary, we can propose targeted fixes or new tests that mirror your on‑road conditions.

## Appendix: Snapshot Fields
Each JSON line contains:
- Inputs/context: `ts`, `v`, `cruise`, `conf`, `vision_status`, `lead`, `hw`
- Curvature: `k_model`, `k_occ`, `k_vis_last`, `is_easing`, `abs_curv_rate`
- Speeds: `v_base`, `v_occ`, `v_vis`, `raw`, `final`
- Occlusion gating: `occl_positive_margin`, `occl_lead_bypass_active`, `vis_horizon_s`, `tail_frac`, `s_tail`, `early_no_raise`
- Map: `map_tail_active`, `map_tail_cap`, `map_tail_start_m`, `map_tail_coverage`
- Limits/commands: `comfort_decel`, `max_adaptive_decel`, `decel_cmd`, `jerk_cmd`, `a_cmd`

