# VTSC Testing Strategy (Emulating “Live” Behavior)

This document describes how to test **Vision Turn Speed Control (VTSC)** end-to-end, with an emphasis on catching:
- **over-slowing** (cap becomes too low too early), and
- **failure to recover** (cap stays low after the curve/occlusion ends), and
- **pipeline deviation** (VTSC computes one thing but the longitudinal stack consumes/commands another).

It complements `docs/vtsc/TESTS.md` (which is a runnable index).

## 1) What We’re Verifying (Top-Level)

VTSC influences the car through a specific chain:

1. `VisionTurnController` computes a recommended speed (`v_turn`)
2. `LongitudinalPlannerSP.update_v_cruise(...)` selects the **minimum** of:
   - driver cruise setpoint
   - VTSC (`v_turn`)
   - Speed Limit Control (SLC)
   - RTI (Realtime Traffic Intelligence)
3. The main `LongitudinalPlanner` uses that “final” cruise speed to generate `longitudinalPlan.aTarget`
4. `controlsd` feeds `aTarget` to `LongControl` which becomes `CarControl.actuators.accel`

The most important “deviation points” to test are:
- **VTSC → min-of-sources selection** (units, gating, latching, stale values)
- **min-of-sources → MPC input** (the exact value passed to MPC)
- **MPC output → `aTarget`** (the plan actually commands decel/accel appropriately)

### Common “Not Actually VTSC” Confounders
These are system-level behaviors that can look like “VTSC won’t recover”, even when
`v_turn` has already released back to cruise:

- **`allow_throttle` gating (planner accel clip)**  
  In the main longitudinal planner, low model throttle probability can reduce the maximum
  allowed acceleration (even if cruise is higher), and it ramps down smoothly over time.
  - Signal: `LongitudinalPlanner.allow_throttle` becomes `False`
  - Input: `modelV2.meta.disengagePredictions.gasPressProbs[1]` (“throttle_prob”)
  - Test: `test_throttle_prob_gate_can_prevent_accel_after_vtsc_release`

- **MPC internal cruise clipping (`v_cruise_clipped`)**  
  The long MPC may clip `v_cruise` inside an accel/decel envelope; sharp VTSC cap steps can
  be softened or delayed by this envelope.
  - Test: `test_mpc_cruise_clipping_softens_large_vtsc_step_down`

- **Blended/E2E acceleration selection**  
  In blended mode, the planner may blend or min with `modelV2.action.desiredAcceleration`,
  so a conservative model action can keep accel low even after the VTSC cap is gone.

- **Actuation limits downstream (LongControl + CarController)**  
  Per-car accel limits, stopping/shouldStop logic, or CarController rate limits can prevent
  the commanded accel from being achieved.

## 2) Test Layers (Recommended)

### Layer A — Controller Behavior (Unit / Scenario)
Goal: validate `VisionTurnController` invariants and edge cases independent of the planner.

Typical assertions:
- No crawl on straights with good confidence
- No positive acceleration while occluded without margin
- Dwell/hysteresis stability (enter/exit occlusion)
- Steering-curvature fail-open guard: when confidence is SEVERE/LOST and model curvature is flat, steering-derived curvature must still drop the cap (`test_severe_confidence_model_flat_steering_fallback_slows_for_sharp_curve`)
- Map lookahead caps apply only when available and covered
- Off-ramp “vision lost” fallback: when confidence is SEVERE/LOST and model curvature is unreliable, map lookahead must still cap short, tight curves inside the normal visible horizon (`test_offramp_short_tight_curve_map_cap_applies_when_vision_lost`, GPS seed 38°43'54.0"N 120°47'20.2"W)

Location:
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py`

### Layer B — Planner Ingestion (VTSC → v_cruise_final)
Goal: validate the **exact** cruise-speed value selected by the planner and published to UI telemetry.

Key assertions:
- If VTSC is enabled + longActive, `update_v_cruise` returns `min(v_cruise, v_turn)` (assuming other sources disabled)
- If longActive is false, VTSC should not constrain `v_cruise`
- `longitudinalPlanSP.visionTurnSpeedControl.velocity` equals `v_tsc.v_turn`

Location:
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_pipeline_integration.py`

### Layer C — End-to-End-ish Flow (v_cruise_final → MPC → aTarget)
Goal: emulate the “live” behavior of the **longitudinal planner** without requiring an Acados build.

Approach:
- Inject a **fake long MPC** module (pure Python) to avoid `c_generated_code` build dependency.
- Assert that the MPC receives the exact `v_cruise_final` selected by the planner.

Location:
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_flow.py`

### Layer D — True Live Replay (Optional / Slower)
Goal: replay real rlogs/qlogs through VTSC and/or planner to validate behavior against recorded conditions.

Data requirement (important):
- For “live replay” regressions, rlogs should originate from **TICI / comma3x** drives.
- Some rlogs live in this git checkout on a dev laptop because they were manually copied from the device and committed as fixtures.
  - Treat them as **tici-sourced**, not “desktop/sim logs”.

This repo already has tooling and larger benches here:
- `docs/chauffeur/vtsc/`

Entry points:
- Replay scripts:
  - `docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py` (snapshot-only)
  - `docs/chauffeur/vtsc/fullTrace/full_trace_replay.py` (method-level trace)
- Regression tests (strict; opt-in):
  - `pytest -q -m regression docs/chauffeur/vtsc/fullTrace/tests/test_regressions_rlogs.py`

Provenance + fixture guidance:
- `docs/vtsc/RLOGS.md`

## 3) Known-Bug Tests (`xfail`)

Some tests may be marked `xfail` to track currently-known issues without breaking CI.

As of 2026-01-26, the VTSC suites under `sunnypilot/selfdrive/controls/lib/tests/vtsc/` have **no** `xfail` tests.

Once fixed:
- remove the `xfail`,
- tighten assertions (timing windows, minimum recovery rate, etc.),
- add additional scenarios based on real logs.

## 4) Practical Guidance for Adding New Tests

When adding a new VTSC behavior test, record:
- the scenario description (what road/visibility condition it represents),
- the invariant it protects (what must never happen),
- the signal(s) you observe:
  - `v_tsc.v_turn`
  - `v_cruise_final` (planner selection)
  - MPC input / output (if using Layer C/D)
  - `aTarget` / `actuators.accel` (if integrating further)

Prefer adding “pipeline deviation” tests to Layer B/C so it’s obvious whether the issue is:
- inside VTSC (computation wrong), or
- in integration (computed correctly but not consumed correctly).

## Appendix: Key Code Pointers (Where to Inspect When a Test Fails)

- VTSC core logic: `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
- VTSC Params refresh: `sunnypilot/selfdrive/controls/lib/vision_turn_params.py`
- VTSC ingestion into planner (min-of sources): `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
- Main longitudinal planner (MPC + `longitudinalPlan.aTarget`): `selfdrive/controls/lib/longitudinal_planner.py`
- Final “passed to car” accel command: `selfdrive/controls/controlsd.py` (`LongControl.update(..., long_plan.aTarget, ...)`)
