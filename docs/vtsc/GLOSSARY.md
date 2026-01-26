# VTSC Glossary

This glossary standardizes VTSC terms used across code, tests, and docs.

## Acronyms

- **VTSC**: Vision Turn Speed Control (a.k.a. “Vision Turn Speed Controller” in UI strings).
- **MTSC**: Map Turn Speed Control (map-derived turn speed; in this branch the publisher process is disabled).
- **SLC**: Speed Limit Control (another speed source the planner may min() with VTSC).
- **RTI**: Realtime Traffic Intelligence (another speed recommendation source).
- **DEC**: Dynamic Experimental Control (longitudinal mode behavior).

## Signals / Variables

- `v_ego`: vehicle speed (m/s).
- `v_cruise`: cruise setpoint (m/s).
- `v_turn`: VTSC’s recommended/commanded speed target for turns (m/s).
- **curvature / kappa (`k`)**: road curvature (1/m). Higher magnitude = tighter turn.
- **lateral acceleration (`a_lat`)**: `v^2 * k` (m/s²) (approx), used to compute a safe speed envelope.

## VTSC State Machine

Defined in `cereal/custom.capnp`:
- `disabled`: feature off or no meaningful turn predicted.
- `entering`: turn predicted ahead, speed adapting down.
- `turning`: actively turning.
- `leaving`: exiting turn, allowing speed to rise again.

## Occlusion / Vision Confidence (VTSC context)

- **vision confidence**: a confidence signal derived from model outputs; VTSC smooths it (EMA) and uses thresholds + dwell times.
- **occlusion**: a state where confidence is low enough (for long enough) that VTSC behaves conservatively:
  - blocks positive acceleration (“monotonic while occluded”),
  - may use an envelope/tail estimate of curvature.

## FOV / Psi (as used by VTSC)

- **FOV**: Field of View (controller considers whether a turn is “in view” or “behind the horizon”).
- **psi**: a geometry-derived quantity used in gating (see Params `VisionTurnSpeedControlPsiFOVRad`, `VisionTurnSpeedControlPsiMarginRad`, and `VTSC.PsiThreshRad`).

