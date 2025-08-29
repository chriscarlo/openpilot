# VTSC Parameters Baseline (as of current branch)

This document lists the Vision Turn Speed Control (VTSC) parameter keys and their current baseline defaults in this branch. Values shown are the defaults compiled into `common/params_keys.h` and represent the expected starting configuration on a clean install.

Notes:
- Types are shown for clarity; values are stored as strings in the Param store unless otherwise noted.
- If a device already has user‑saved values, those will override these defaults at runtime.

## Enablement
- `VisionTurnSpeedControl` (BOOL): 0

## Driving Style / Anticipation
- `VisionTurnSpeedControlAggressiveness` (INT): 1
- `VisionTurnSpeedControlFixedLeadTimeSeconds` (FLOAT): 0.0

## Adaptive Deceleration & Filtering
- `VisionTurnSpeedControlFilterAlpha` (FLOAT): 0.3
- `VisionTurnSpeedControlHysteresisThreshold` (FLOAT): 0.2
- `VisionTurnSpeedControlSafetyBias` (FLOAT): 0.1

## Smoothing Limits
- `VisionTurnSpeedControlSmoothingMaxDecel` (FLOAT): 3.5
- `VisionTurnSpeedControlSmoothingMaxJerk` (FLOAT): 6.0
- `VisionTurnSpeedControlAccelToDecelRatio` (FLOAT): 1.3
- `VisionTurnSpeedControlJerkAccelMultiplier` (FLOAT): 2.0

## Planning & Overshoot Control
- `VisionTurnSpeedControlPlanningDecelLimit` (FLOAT): 3.5
- `VisionTurnSpeedControlOvershootSafetyMargin` (FLOAT): 1.2
- `VisionTurnSpeedControlOvershootMinDistance` (FLOAT): 10.0
- `VisionTurnSpeedControlAnticipationTargetReduction` (FLOAT): 0.95

## Apex Detection & Exit Boost
- `VisionTurnSpeedControlApexThreshold` (FLOAT): 0.00005
- `VisionTurnSpeedControlApexProminence` (FLOAT): 0.0001
- `VisionTurnSpeedControlApexHysteresisTime` (FLOAT): 2.0
- `VisionTurnSpeedControlApexMetersPerIndex` (FLOAT): 2.0
- `VisionTurnSpeedControlApexNearIndex` (INT): 3
- `VisionTurnSpeedControlApexBoostDistance` (FLOAT): 50.0
- `VisionTurnSpeedControlApexBoostFactor` (FLOAT): 0.1
- `VisionTurnSpeedControlApexBoostMinLatAccel` (FLOAT): 1.0
- `VisionTurnSpeedControlApexBoostCenter` (FLOAT): 2.0
- `VisionTurnSpeedControlApexBoostWidth` (FLOAT): 0.5
- `VisionTurnSpeedControlBoostSafetyCurvatureScale` (FLOAT): 0.7

## Comfort & Adaptive Hard Limits
- `VisionTurnSpeedControlComfortDecelLimit` (FLOAT): -1.47
- `VisionTurnSpeedControlComfortJerkLimit` (FLOAT): -2.0
- `VisionTurnSpeedControlMaxAdaptiveDecel` (FLOAT): -6.0
- `VisionTurnSpeedControlMaxAdaptiveJerk` (FLOAT): -6.0

## Vision Confidence & Occlusion
- `VisionTurnSpeedControlVisionConfAlpha` (FLOAT): 0.28
- `VisionTurnSpeedControlVisionConfGoodThreshold` (FLOAT): 0.70
- `VisionTurnSpeedControlVisionConfBadThreshold` (FLOAT): 0.65

Occlusion dwell / decay / envelope:
- `VisionTurnSpeedControlOcclEnterDwellS` (FLOAT): 0.20
- `VisionTurnSpeedControlOcclExitDwellS` (FLOAT): 0.10
- `VisionTurnSpeedControlCurvatureGrowthPerMeter` (FLOAT): 0.0005
- `VisionTurnSpeedControlEnvelopeHorizonS` (FLOAT): 1.2
- `VisionTurnSpeedControlOcclusionDecayTauFastS` (FLOAT): 1.2
- `VisionTurnSpeedControlOcclusionDecayTauSlowS` (FLOAT): 2.0
- `VisionTurnSpeedControlOcclusionMinFrac` (FLOAT): 0.20

Fast reacquisition window:
- `VisionTurnSpeedControlFastReacqAlpha` (FLOAT): 0.85
- `VisionTurnSpeedControlFastReacqWindowS` (FLOAT): 0.90

## Curvature & Physics
- `VisionTurnSpeedControlCurvatureEMAFactor` (FLOAT): 0.3

Physics model (sigmoid lat accel clamps):
- `VisionTurnSpeedControlPhysicsBaseline` (FLOAT): 3.144734
- `VisionTurnSpeedControlPhysicsAmplitude` (FLOAT): -1.1751
- `VisionTurnSpeedControlPhysicsSteepness` (FLOAT): -2000.0
- `VisionTurnSpeedControlPhysicsCenter` (FLOAT): 0.004778
- `VisionTurnSpeedControlPhysicsMinLatAccel` (FLOAT): 1.8
- `VisionTurnSpeedControlPhysicsMaxLatAccel` (FLOAT): 3.12

## Speed Scaling & Limits
- `VisionTurnSpeedControlSpeedIncreaseFactor` (FLOAT): 1.0
- `VisionTurnSpeedControlMaxSpeed` (FLOAT): 70.0   (m/s)
- `VisionTurnSpeedControlMinOperatingSpeed` (FLOAT): 2.24  (m/s)
- `VisionTurnSpeedControlLowSpeedSpeedBiasMph` (FLOAT): 0.0
- `VisionTurnSpeedControlLowSpeedBiasEndMph` (FLOAT): 50.0

