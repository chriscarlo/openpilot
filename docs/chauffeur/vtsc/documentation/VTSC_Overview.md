# Vision Turn Speed Control (VTSC): Practical Overview

This document explains how VTSC behaves on-road, in practical terms, with pointers to the key code paths. It is not a deep internal design doc; instead it describes what users should expect, and where those behaviors live in code.

- Primary file: `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
- Primary class: `VisionTurnController`

## High‑Level Summary
VTSC computes a safe target speed for upcoming curves from model‑predicted curvature, smooths that target into a realizable acceleration, and defers to the longitudinal planner to choose the minimum of all sources (cruise, speed limit, VTSC). It actively avoids accelerating while vision confidence is poor (monotonic‑while‑occluded), and uses comfort limits unless it must escalate to adaptive deceleration to make the curve in time.

## Step‑By‑Step Behavior

1) Update & Inputs (20 Hz)
- Code: `VisionTurnController.update(...)`
- Inputs: enabled state, current speed/accel, cruise setpoint, and model outputs via `Sm['modelV2']`.
- What happens: The controller refreshes runtime params and calls the calculation pipeline (`_update_calculations` → `_state_transition` → `_update_solution`).

2) Curvature & Confidence
- Code: `_update_calculations`, `_update_vision_occlusion`, `VisionOcclusionState`
- Curvature: Taken from model orientation rate and predicted velocity (signed), smoothed by an EMA.
- Confidence: A smoothed confidence signal with a tri‑state gate:
  - Good (≥ good threshold): use live curvature.
  - Borderline (between bad and good): partially update curvature to avoid freezing.
  - Occluded (≤ bad threshold with dwell): hold/estimate curvature, do not accelerate.
- Practical effect: Enter occlusion only after sustained low confidence (enter dwell). Exit faster after sustained good confidence (exit dwell). During occlusion, target speed won’t increase; when just borderline, it still refreshes to prevent “sticky holds”.

3) Anticipatory Deceleration
- Code: `_plan_advanced_speed_trajectory`
- Behavior: Before a curve, VTSC anticipates and gradually reduces the target speed (physics‑based from curvature). This anticipatory reduction is moderated:
  - Near the confidence threshold and flattening curves: it stops digging the hole (no extra reductions).
  - A small budget caps how much extra reduction can be added in a short window.
- Practical effect: You typically reach a safe speed before the apex without overslowing, and you won’t pre‑brake too deeply right as confidence worsens.

4) Physics Target & Exit Boost
- Code: `curvature_to_speed`, `_physics_based_lateral_acceleration`, `_plan_advanced_speed_trajectory`
- Physics: A sigmoid maps curvature → allowed lateral acceleration → target speed. Clamped by min/max lat accel.
- Exit boost: After passing the apex (or near it), VTSC can gently nudge the target up via a smooth function of lateral acceleration (bounded by physics and a safety curvature scale).
- Practical effect: A steady, predictable decel into curves and a natural, bounded pickup on the way out.

5) Smoothing & Jerk Limits
- Code: `_update_solution`
- Behavior: The difference between successive acceleration commands is bounded (jerk), with separate limits for accel/decel, and an overall decel cap. When vision is occluded, positive acceleration is blocked (monotonic behavior), and further decel is capped to comfort when appropriate to avoid crawl.
- Practical effect: Target speed feels composed. No snap‑accel while vision is questionable; braking ramps remain in comfortable bounds.

6) Confidence, Occlusion & Reacquisition
- Code: `VisionOcclusionState.update`
- Behavior while occluded:
  - Before apex (tightening): conservative growth envelope for curvature for a short horizon.
  - After apex (easing): two‑stage decay of held curvature toward a lower floor to shorten crawl.
  - Fast reacquisition window briefly increases filter responsiveness once vision is good again (still jerk‑bounded).
- Practical effect: No blind acceleration, shorter post‑apex crawl, and a quick—but smooth—return to clean‑vision behavior.

7) Planner Interaction (min‑of sources)
- Code: `VisionTurnController.v_turn`, main longitudinal planner
- VTSC always provides a physics‑based target (or cruise on straights). The main longitudinal planner selects the minimum of VTSC, speed limits, and any other sources.
- Practical effect: VTSC never forces speeding; it only lowers the allowed target when curves demand it.

## What To Expect On‑Road
- Approaching curves: A calm, progressive decel to a physics‑valid speed, beginning neither too early nor late (tunable via aggressiveness and planning limits).
- During borderline vision: VTSC keeps curvature estimates fresh and avoids entering a full freeze unless confidence truly goes low.
- If vision degrades: It will not accelerate blind, and will moderate further decel to avoid prolonged crawl tails.
- Past the apex: A modest, bounded boost helps the car feel natural exiting, while still respecting physics and safety margins.
- Overall ride: Smoothing and jerk limits enforce consistent, comfortable ramps without sacrificing responsiveness where it matters.

## Where To Tune (at a glance)
- Comfort & jerk: Comfort decel/jerk and smoothing limits control “feel.”
- Anticipation: Aggressiveness, planning decel limit, overshoot margin/distance, anticipation reduction.
- Vision & occlusion: Confidence EMA and thresholds, dwell times, curvature growth γ, envelope horizon, decay time constants, minimum fraction, fast‑reacq window.
- Physics: Min/max lateral acceleration and sigmoid parameters set the envelope.

## Key Code Entrypoints (for reference)
- `VisionTurnController.update(...)` — the main entrypoint (20 Hz)
- `_update_calculations(...)` — curvature + confidence handling
- `_plan_advanced_speed_trajectory()` — physics target, anticipatory reduction, apex boost
- `_update_solution()` — smoothing, jerk limits, monotonic occlusion rule
- `curvature_to_speed(...)`, `_physics_based_lateral_acceleration(...)` — physics mapping

For the exact parameter keys and their default values, see `paramBaseline.md` in this folder.

