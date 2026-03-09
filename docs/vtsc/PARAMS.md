# VTSC Params (Keys, Defaults Source, UI Mapping)

This doc indexes VTSC-related `Params` keys, where they are defined, and where they are consumed.

## Source of Truth for Defaults

All default values and declared types live in:
- `common/params_keys.h`

VTSC code is robust to malformed/missing values (falls back to defaults), but for “clean install” expectations, `common/params_keys.h` is canonical.

## Where Params Are Read

Primary read path:
- `sunnypilot/selfdrive/controls/lib/vision_turn_params.py`
  - `update_vtsc_params(ctrl, force=False)` refreshes most tunables (debounced ~2s).

Telemetry/debug toggles are polled by the controller:
- `VTSCVerboseDebug` (enables `VTSCDBG` logs)
- `VTSCWriteSnapshotFile` (enables `/data/media/0/VTSCDebug/vtsc_snapshots.jsonl`)
- `VTSCInterventionRecorderEnabled` (manager-controlled onroad event recorders to `/data/media/0/VTSCTuner/events` and `/data/media/0/VTSCTuner/comm_issues`)

## Key Param Groups (High Level)

Enable / main knobs:
- `VisionTurnSpeedControl` (bool)
- `VisionTurnSpeedControlAggressiveness` (default `1`)
- `VisionTurnSpeedControlFixedLeadTimeSeconds` (default `0.0`)
- `VisionTurnSpeedControlCurvePhaseOffsetS` (default `0.0`)
- `VisionTurnSpeedControlOvershootPhaseOffsetS` (default `0.0`)
- `VisionTurnSpeedControlApexExitPhaseOffsetS` (default `0.0`)

Adaptive decel + safety shaping:
- `VisionTurnSpeedControlFilterAlpha`
- `VisionTurnSpeedControlHysteresisThreshold`
- `VisionTurnSpeedControlSafetyBias`
- `VisionTurnSpeedControlComfortDecelLimit`
- `VisionTurnSpeedControlComfortJerkLimit`
- `VisionTurnSpeedControlMaxAdaptiveDecel`
- `VisionTurnSpeedControlMaxAdaptiveJerk`

Smoothing / jerk limits:
- `VisionTurnSpeedControlCurvatureEMAFactor`
- `VisionTurnSpeedControlSmoothingMaxDecel`
- `VisionTurnSpeedControlSmoothingMaxJerk`
- `VisionTurnSpeedControlAccelToDecelRatio`
- `VisionTurnSpeedControlJerkAccelMultiplier`

Planning / anticipation / overshoot:
- `VisionTurnSpeedControlPlanningDecelLimit`
- `VisionTurnSpeedControlOvershootSafetyMargin`
- `VisionTurnSpeedControlOvershootMinDistance`
- `VisionTurnSpeedControlAnticipationTargetReduction`

Apex detection + boost:
- `VisionTurnSpeedControlApexThreshold`
- `VisionTurnSpeedControlApexProminence`
- `VisionTurnSpeedControlApexHysteresisTime`
- `VisionTurnSpeedControlApexMetersPerIndex`
- `VisionTurnSpeedControlApexNearIndex`
- `VisionTurnSpeedControlApexBoostDistance`
- `VisionTurnSpeedControlApexBoostFactor`
- `VisionTurnSpeedControlApexBoostMinLatAccel`
- `VisionTurnSpeedControlApexBoostCenter`
- `VisionTurnSpeedControlApexBoostWidth`
- `VisionTurnSpeedControlBoostSafetyCurvatureScale`

Vision confidence + occlusion behavior:
- `VisionTurnSpeedControlVisionConfAlpha`
- `VisionTurnSpeedControlVisionConfGoodThreshold`
- `VisionTurnSpeedControlVisionConfBadThreshold`
- `VisionTurnSpeedControlOcclEnterDwellS`
- `VisionTurnSpeedControlOcclExitDwellS`
- `VisionTurnSpeedControlCurvatureGrowthPerMeter`
- `VisionTurnSpeedControlEnvelopeHorizonS`
- `VisionTurnSpeedControlOcclusionDecayTauFastS`
- `VisionTurnSpeedControlOcclusionDecayTauSlowS`
- `VisionTurnSpeedControlOcclusionMinFrac`
- `VisionTurnSpeedControlFastReacqAlpha`
- `VisionTurnSpeedControlFastReacqWindowS`

Visibility barrier + occlusion envelope:
- `VisionTurnSpeedControlVisHorizonS`
- `VisionTurnSpeedControlVisMarginM`
- `VisionTurnSpeedControlGammaPerMeter`
- `VisionTurnSpeedControlLatJerkCap`

FOV gating / geometry:
- `VisionTurnSpeedControlPsiFOVRad`
- `VisionTurnSpeedControlPsiMarginRad`
- `VisionTurnSpeedControlFOVKMin`
- `VisionTurnSpeedControlFOVKFreeway`
- `VisionTurnSpeedControlFOVSLongM`
- `VisionTurnSpeedControlFOVPretriggerTimeS`
- `VisionTurnSpeedControlFOVOnsetBoostFrames`
- `VisionTurnSpeedControlFOVOvershootFrames`
- `VisionTurnSpeedControlFOVEWMATauS`
- `VisionTurnSpeedControlFOVNOn`
- `VisionTurnSpeedControlFOVNOff`

Vision “floor” / dropout discrimination:
- `VisionTurnSpeedControlVisionFloorTtlS`
- `VisionTurnSpeedControlVisionFloorMult`
- `VisionTurnSpeedControlDropoutGraceS`

Physics model parameters (curvature→speed envelope):
- `VisionTurnSpeedControlPhysicsBaseline`
- `VisionTurnSpeedControlPhysicsAmplitude`
- `VisionTurnSpeedControlPhysicsSteepness`
- `VisionTurnSpeedControlPhysicsCenter`
- `VisionTurnSpeedControlPhysicsMinLatAccel`
- `VisionTurnSpeedControlPhysicsMaxLatAccel`

Lead-aware occlusion bypass:
- `VisionTurnSpeedControlOcclBypassWithLead`
- `VisionTurnSpeedControlOcclBypassHeadwayS`

Map lookahead (consumed by VTSC; MTSC publisher process is disabled):
- `MTSCLookaheadEnabled`

Debug / triage knobs:
- `VTSCFailOpen`
- `VTSCVerboseDebug`
- `VTSCWriteSnapshotFile`
- `VTSC.PsiThreshRad`
- `VTSC.PsiHystRad`
- `VTSC.DoubleCapEpsMps`
- `VTSC.OcclConfFloor`
- `VTSC.FovExitRelaxS`
- `VTSC.OcclVminNudgeMps`

## Where Params Are Set (UI)

Offroad settings:
- `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal_panel.cc` (main toggle + submenu)
- `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_*_panel.cc` (individual tunables)

UI/UX design notes:
- `docs/chauffeur/ui/bsg/offroad/vtsc_menu_framework.md`

## Known Gotcha: Declared Param Types vs. Runtime Usage

At least one VTSC Param is declared with a type that does not match how the VTSC controller treats it:
- `VisionTurnSpeedControlAggressiveness` is declared as `INT` in `common/params_keys.h`,
  but VTSC reads it as a float and clamps it in a floating range (currently `0.5..2.0` in `vision_turn_params.py`).

If this Param is written with a non-integer string (example: `"1.00"`), the Param system may log cast errors like:
`Failed to cast param ... from type ... INT`.

When debugging “why is VTSC ignoring my knob?” or noisy startup logs, check:
- the declared types in `common/params_keys.h`, and
- what the UI writes (string formatting) in the VTSC settings panels.
