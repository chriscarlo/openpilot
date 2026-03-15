# Longitudinal Pipeline

## Core Ownership Map

- `sunnypilot/modeld_v2/modeld.py`
  produces `modelV2.action.desiredAcceleration` and `modelV2.action.shouldStop`.
  This is the model-driven action surface.
- `selfdrive/controls/lib/longitudinal_planner.py`
  owns the main planner loop and publishes `longitudinalPlan`.
- `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
  wraps the planner with sunnypilot overlays and publishes `longitudinalPlanSP`.
- `selfdrive/controls/controlsd.py`
  consumes `longitudinalPlan.aTarget` and runs `LongControl`.
- Brand `CarController` implementations translate actuator accel into CAN.
  Hyundai is special because sunnypilot adds another longitudinal shaping layer
  inside the Hyundai controller.

## ACC Path vs Blended Path

- Initial planner mode comes from `selfdriveState.experimentalMode`:
  `acc` when off, `blended` when on.
- `DynamicExperimentalController` in
  `sunnypilot/selfdrive/controls/lib/dec/dec.py`
  can override that planner mode between `acc` and `blended` while
  Experimental Mode is active.
- In `acc` mode, `LongitudinalMpc.update(...)` uses cruise and radar leads as
  obstacles. The planner output is deterministic MPC output.
- In `blended` mode, `LongitudinalMpc.update(...)` also consumes the model path
  (`x`, `v`, `a`) and can mark the source as `e2e` or `cruise`.
- Final planner output is not always raw model accel:
  `selfdrive/controls/lib/longitudinal_planner.py` compares the MPC output and
  `modelV2.action.desiredAcceleration`.
  In blended mode it either:
  - uses `min(mpc, e2e)` by default, or
  - applies the DEC transition blend when DEC is enabled.

## Important Subtlety: `mlsim`

- `sunnypilot/modeld_v2/modeld.py` treats bundle generation `>= 11` as `mlsim`.
- `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py` treats missing
  generation or generation `>= 11` as `mlsim`.
- If “the model should be driving longitudinal but it is not” looks confusing,
  check the active model bundle first instead of guessing from UI state.

## Sunnypilot Longitudinal Overlays

- `LongitudinalPlannerSP.update_v_cruise(...)` takes the minimum of:
  cruise set speed, VTSC, SLC, RTI, and Weather.
- VTSC keeps computing onroad even when openpilot is not engaged so the HUD can
  keep showing previews, but it only becomes a speed source when
  `carControl.longActive` is true.
- `longitudinalPlanSP` does not expose every cap candidate together:
  VTSC and SLC live there, RTI is published separately on `rtiStateSP`, and
  Weather is internal-only to the planner. A passive monitor can get close, but
  it cannot prove weather attribution without an internal hook.
- `longitudinalPlan.allowThrottle` is influenced by the model’s gas-press
  probability. The planner can clamp positive acceleration even in ACC mode if
  the model thinks the driver is likely to press the accelerator.
- `VibePersonalityController` can change max accel, min accel, and following
  distance independently of the stock personality curves.

## `LongControl` Layer

- `selfdrive/controls/lib/longcontrol.py`
  turns `longitudinalPlan.aTarget` into actuator accel with a small state
  machine: `off`, `stopping`, `starting`, `pid`.
- This layer uses `aTarget - aEgo` as the PID error and publishes the PID
  terms to `controlsState.upAccelCmd`, `uiAccelCmd`, and `ufAccelCmd`.
- If `aTarget` is wrong, stay in the planner.
- If `aTarget` is right but `carControl.actuators.accel` is wrong, the issue is
  in `LongControl` or its inputs.

## Hyundai-Specific Command Path

- Hyundai `CarController` does not just pass through the controlsd accel.
- `opendbc/sunnypilot/car/hyundai/longitudinal/controller.py`
  can reshape the command into:
  `desired_accel`, `actual_accel`, `jerk_upper`, `jerk_lower`,
  `comfort_band_upper`, and `comfort_band_lower`.
- CAN message creation uses those shaped values:
  `opendbc/car/hyundai/hyundaican.py` and
  `opendbc/car/hyundai/hyundaicanfd.py`.
- `carOutput.actuatorsOutput.accel`
  reflects the brand-controller output, not just the planner target.

## Tuning Surfaces

- Planner and state-machine parameters:
  `CarParams.longitudinalActuatorDelay`, `vEgoStopping`, `vEgoStarting`,
  `stoppingDecelRate`, `startAccel`.
- Model / blended behavior:
  Experimental Mode, Dynamic Experimental Control, active model bundle
  generation, and `modelV2.action.*`.
- Overlay caps:
  VTSC, SLC, RTI, Weather, and throttle gating.
- Brand command shaping:
  Hyundai `HyundaiLongitudinalTuning` and `LongTuning*` params.
- Personality overlays:
  Vibe accel, brake, and follow-distance params.

## Symptom-to-Layer Mapping

- `longitudinalPlan.aTarget` already low:
  planner or an overlay cap is responsible.
- `aTarget` is healthy, but `carControl.actuators.accel` is low:
  `LongControl` or its limits are responsible.
- `carControl.actuators.accel` is healthy, but
  `carOutput.actuatorsOutput.accel` is lower:
  brand controller shaping is responsible.
- `carOutput.actuatorsOutput.accel` is healthy, but delayed `carState.aEgo`
  does not track:
  actuator delay, stock ECU interference, or vehicle-response mismatch.
- `allowThrottle` is false with positive targets:
  the model throttle gate is clamping positive accel.
- `longitudinalPlanSource` flips between `cruise`, `lead0`, and `e2e`:
  that is planner-source arbitration, not necessarily actuator instability.
