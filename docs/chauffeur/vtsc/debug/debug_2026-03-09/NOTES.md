# VTSC Debug Notes — 2026-03-09

## Scope

- Repo: `/projects/chauffeur/data/openpilot`
- Skill used: local `vtsc-tuner`
- Route analyzed: `0000009d--378452f068`
- Raw pulled artifacts: `.cache/vtsc/commaCar_20260309_route9d/`

## Environment / commands

- Pulled qlogs and rlogs from `commaCar:/data/media/0/realdata/0000009d--378452f068--<seg>/`
- Replays run with repo venv:
  - `PYTHONPATH=$PWD .venv/bin/python docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py ... --cruise 25 --max-frames 1200`
  - `PYTHONPATH=$PWD .venv/bin/python docs/chauffeur/vtsc/analysis/analyze_snapshots.py ...`
- VTSC tests run with repo venv:
  - `PYTHONPATH=$PWD .venv/bin/pytest sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_map_timing.py`

## Route reconstruction

- Approx route span on device: `2026-03-09T22:43:41Z` to `2026-03-09T23:14:41Z`
- Engaged freeway: roughly segments `0..6` on `Route 50`
- Engaged mountain section: segments `7..12`
  - `7..8`: `Greenstone Road`
  - `9..12`: mostly `Old French Town Road` with transition through `Mother Lode Drive`
- Segment `13` is partial disengage / transition home
- Segments `14..21` are manual local roads / neighborhood

Mountain segment summary from qlogs (`selfdriveStateSP.mads` as engagement source):

| seg | road | avg m/s | max m/s | mads_active | gas frames | brake frames | steer override frames |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | Greenstone Road | 23.35 | 25.34 | 1.00 | 0 | 0 | 0 |
| 8 | Greenstone Road | 15.37 | 16.53 | 1.00 | 63 | 8 | 74 |
| 9 | Old French Town Road | 10.80 | 17.08 | 1.00 | 149 | 79 | 181 |
| 10 | Old French Town Road | 13.35 | 21.30 | 1.00 | 0 | 0 | 19 |
| 11 | Old French Town Road | 16.36 | 22.13 | 1.00 | 127 | 68 | 117 |
| 12 | Old French Town Road | 16.00 | 17.72 | 1.00 | 0 | 0 | 26 |

## Manual interventions worth explaining

Mountain gas/brake rising-edge events extracted from qlogs:

- seg 8 `Greenstone Road` gas at `v=11.07`, `vt=8.72`, `delta=-2.35`, `pred_lat=2.43`
- seg 8 `Greenstone Road` brake at `v=14.88`, `vt=9.76`, `delta=-5.11`, `pred_lat=3.08`
- seg 8 `Greenstone Road` gas at `v=13.24`, `vt=14.20`, `delta=+0.95`, `pred_lat=2.20`
- seg 11 `Old French Town Road` brake at `v=17.83`, `vt=18.92`, `delta=+1.09`, `pred_lat=1.45`
- seg 11 `Old French Town Road` gas at `v=20.99`, `vt=19.75`, `delta=-1.24`, `pred_lat=0.64`, `curve_distance=101.2 m`
- seg 13 `Old French Town Road` gas at `v=13.01`, `vt=13.02`, `delta≈0`

The strongest mountain tuning signals remain seg `8` and seg `11`.

## What the last ~25 minutes looked like

- Freeway was stable and fast. No VTSC tuning priority surfaced there.
- The mountain section was good overall, but it showed two distinct non-freeway behaviors:
  1. Release/re-apply spikes between close bends
  2. Mild acceleration drag under degraded vision confidence even when explicit FOV occlusion did not fire

### Pattern 1: release spike between close curves

Segment `11` is the clearest example.

- Around the interesting window, qlog `curveDistanceM` steps down roughly `101 m -> 79 m -> 59 m -> 38.6 m -> 18.6 m`
- VTSC target (`vt_vel`) briefly relaxes back toward cruise near the `38.6 m` point, then tightens again by `18.6 m`
- This matches the user complaint: the map knows another curve is close, but the handoff lets vision relax too early while still exiting the previous bend

Code-level cause found:

- Strategic release used `s_visible + vis_margin` as the takeover boundary
- At mountain speeds this lets map release occur when the controlling anchor is only barely inside the padded zone, not clearly inside the true visible horizon

### Pattern 2: acceleration drag from degraded confidence

Replays for segments `8`, `11`, and `12` with `--cruise 25` showed:

- lots of `PARTIAL` and `SEVERE` vision states
- zero explicit `occluded=true` frames
- zero `active_cap` handoffs; replay stayed on visible cap throughout

Interpretation:

- The explicit FOV occlusion latch does **not** look like the main culprit on this drive
- The draggy feeling is more likely from degraded vision confidence affecting the visible-cap path, plus the severe-confidence no-raise behavior

## Resulting tuning direction

### Implemented in this branch

1. Strategic chained-curve envelope

- Added a cheap backward reachability envelope on the strategic map frontier
- This adds a cap that can limit exit acceleration even when the next curve's target speed is at or slightly above current speed
- It uses planner comfort decel plus actuation delay to keep enough room for the next curve

2. Stricter strategic release boundary

- Strategic takeover/release now waits for the controlling map anchor to be inside the actual visible horizon, not merely `visible + margin`
- This targets the seg `11` style release spike

### Tests added

- `test_strategic_chain_envelope_limits_accel_for_same_speed_next_curve`
- `test_strategic_counterevidence_dwell_waits_for_anchor_visibility`

Verification passed:

- `46 passed` across `test_scenarios.py` and `test_longitudinal_planner_vtsc_map_timing.py`

## Open questions / next drive

- The severe-confidence no-raise path may still be too sticky on mountain roads with good maps and rock walls. It should be evaluated separately from the explicit FOV occlusion gate.
- A future refinement could identify "windy-road mode" from repeated map-curvature density or repeated strategic anchor cadence, then use that only as a classifier/telemetry signal before any behavior change.
- The desired human feel still likely needs some blend work:
  - slow slightly later into a visible bend
  - avoid hard acceleration between bends
  - release more smoothly once the next anchor is genuinely visible and benign
