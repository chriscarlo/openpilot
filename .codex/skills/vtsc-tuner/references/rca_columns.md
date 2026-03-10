# VTSC RCA Workbook Columns

This reference explains the fields written by `scripts/vtsc_rca_workbook.py`.

## Summary Sheet

The summary is intended for quick compare/contrast across multiple interventions.

- `event_id`: Stable identifier for the intervention window.
- `route`, `seg`, `t0`: Route name, segment index, and intervention timestamp (`logMonoTime/1e9` seconds).
- `vEgo@0`: Vehicle speed at the intervention edge (`dt=0`).
- `vEgo@-0.5`: Speed 0.5s before intervention.
- `vtscVel@-0.5`: `longitudinalPlanSP.visionTurnSpeedControl.velocity` at `dt=-0.5s`.
- `dv_vtsc@-0.5`: `vEgo@-0.5 - vtscVel@-0.5` (overspeed vs VTSC cap).
- `vtscVel@-2`: VTSC velocity at `dt=-2s`.
- `vtscVelDrop(-2->-0.5)`: `vtscVel@-2 - vtscVel@-0.5`.
  - Large positive values mean the cap tightened late (within ~2s of the intervention).
- `predLatAccMax[-2..0]`, `predLatAccDtMax[-2..0]`: Max `vtscMaxPredLatAcc` in the 2s pre-window.
  - A quick proxy for “how turny” VTSC thought it was.
- `llProbMin[-2..0]`, `llProbDtMin[-2..0]`: Min `mean(modelV2.laneLineProbs)` and when it occurred.
  - A crude proxy for lane-line visibility / confidence (use alongside `modelConf` in Trace).
- `modelFrameDropMax[-2..0]`: Max `modelV2.frameDropPerc` in the pre-window.
  - If near 0.0, late cap changes are less likely to be caused by dropped model frames.
- `aTarget@-0.5`: `longitudinalPlan.aTarget` 0.5s before intervention.
- `aTargetFirstNegDt[-2..0]`: First `dt` where `aTarget < -0.10` in the pre-window.
  - Values close to `0` indicate “planner started braking very late”.
- `actAccel@-0.5`: `carControl.actuators.accel` at `dt=-0.5s`.
- `actAccelFirstNegDt[-2..0]`: First `dt` where commanded accel `< -0.10` in the pre-window.
- `gpsAge@-0.5`, `gpsAgeMax[-2..0]`: Age (seconds) of latest `gpsLocation` sample used at those times.
  - Near `1.0s` implies ~1 Hz GPS feed for this segment.
- `mapDataAge@-0.5`, `mapDataAgeMax[-2..0]`: Age (seconds) of latest `liveMapDataSP` sample.
  - Near `1.0s` implies ~1 Hz mapd publication cadence.
- `mapRoadNameChanges[-2..0]`: Count of `liveMapDataSP.roadName` transitions in pre-window.
  - Non-zero near intervention suggests map matching instability at split/merge transitions.
- `mapSpeedValidChanges[-2..0]`: Count of `liveMapDataSP.speedLimitValid` toggles in pre-window.
- `mapRoadGeomValidAny[-2..0]`: Whether `liveMapDataSP.roadGeometryValid` was ever true in pre-window.

## Trace Sheet

The Trace sheet is a stacked time series (all events concatenated) intended for plotting/filtering.

Key fields:
- `dt`: Relative time in seconds (`-10.0` .. `+10.0`).
- `vtscVelMps`, `vtscMaxPredLatAcc`, `vtscState`: VTSC cap and its state machine.
- `llProbMean`, `modelConf`, `modelFrameDropPerc`: quick perception confidence indicators.
- `lpATarget`, `lpV0`, `lpVMin`, `lpSource`: longitudinal planner outputs.
- `actAccel`, `uiAccelCmd`, `ufAccelCmd`, `upAccelCmd`, `forceDecel`: control outputs.
- `gpsLat`, `gpsLon`, `gpsSpeed`, `gpsBearingDeg`, `gpsAge`: latest GPS sample and staleness at each row.
- `mapSpeedLimitValid`, `mapSpeedLimit`, `mapSpeedLimitAhead`, `mapSpeedLimitAheadDistance`, `mapRoadName`, `mapRoadGeometryValid`, `mapDataAge`: latest map publication fields and staleness.

## Interpretation Heuristics (Quick)

- **Late VTSC cap collapse**:
  - `vtscVelDrop(-2->-0.5)` is large, AND `aTargetFirstNegDt` is close to `0`.
  - Often points to “VTSC didn’t confidently commit early enough” (perception/occlusion gating).

- **Planner decel delay**:
  - `vtscVel@-2` is already low (cap active early), but `aTargetFirstNegDt` is still close to `0`.
  - Points to “VTSC cap exists but isn’t getting enforced early”.

- **Confidence collapse correlation**:
  - `llProbMin[-2..0]` drops sharply in the same window as the VTSC cap tightening.
  - Points to “the inputs to VTSC changed abruptly”, not necessarily a bad curve.
