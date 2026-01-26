# VTSC Occlusion Logic (What It Does, Why, and How to Test It)

This document explains the **VTSC occlusion subsystem** in this fork:
- what the occlusion logic is trying to protect against,
- how it is implemented (major signals + state machines),
- how lead-vehicle interactions are handled (and why),
- and what tests exist to prevent regressions.

## 1) What “Occlusion” Means (Intent)

In VTSC, “occlusion” is meant to represent **real visibility limits**:
- mountain roads where a graded hillside blocks view around a bend,
- tall freeway dividers/walls that block far-ahead line-of-sight,
- sharp bends where the predicted path leaves the camera/model field-of-view before we have enough “visible distance”.

Occlusion logic is *not* intended to activate just because:
- there is a normal lead vehicle ahead, and/or
- lane line confidence is reduced due to the lead partially blocking lane markings.

The key safety goal is:
- **don’t accelerate into a region we can’t see well enough to brake for**.

The key usability goal is:
- **don’t get “stuck” at a low speed after the curve ends**, especially in benign conditions (straight road, good vision).

## 2) Where It Lives (Code Pointers)

Core implementation:
- `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
  - `VisionOcclusionState` (confidence + dwell + curvature extrapolation)
  - `_update_vision_occlusion(...)`
  - `_update_solution(...)` (FOV gate + arbitration + “no-raise” behavior)

Param refresh:
- `sunnypilot/selfdrive/controls/lib/vision_turn_params.py`

Param defaults:
- `common/params_keys.h`

## 3) Major Inputs Used by Occlusion Logic

### 3.1 Model (vision)
VTSC primarily uses a lightweight subset of `modelV2`:
- `modelV2.orientationRate.z` (yaw rate, rad/s)
- `modelV2.velocity.x` (m/s)
- `modelV2.laneLineProbs` (used as a proxy for “vision confidence”)

Important unit note:
- curvature κ (1/m) is derived as: `κ = yaw_rate / speed`.

### 3.2 Visibility horizon
VTSC uses a simple visibility horizon model:
- visible distance `s_visible_m ≈ v_ego * VisionTurnSpeedControlVisHorizonS`
- plus a margin `VisionTurnSpeedControlVisMarginM` in some checks.

### 3.3 Lead (radar)
Lead presence can be consumed from:
- `radarState.leadOne.status` and `radarState.leadOne.dRel`

VTSC uses these only for **occlusion bypass** (see below), not for curve physics.

## 4) Two “Occlusion” Layers (Confidence + Geometry)

Occlusion behavior in this fork is effectively the combination of:

### 4.1 Confidence / dwell state (`VisionOcclusionState`)
This layer:
- smooths “vision confidence” (mean of `laneLineProbs`),
- applies **enter dwell** and **exit dwell** timers,
- maintains a `last_valid_curvature` snapshot when vision is “good”,
- can extrapolate curvature when vision is “not good”.

Key Params:
- `VisionTurnSpeedControlVisionConfAlpha`
- `VisionTurnSpeedControlVisionConfGoodThreshold`
- `VisionTurnSpeedControlVisionConfBadThreshold`
- `VisionTurnSpeedControlOcclEnterDwellS`
- `VisionTurnSpeedControlOcclExitDwellS`
- `VisionTurnSpeedControlDropoutGraceS` (short grace after model dropouts)

### 4.2 Field-of-view (FOV) gating (geometry-based)
This layer detects when the path likely leaves the usable forward view.

It uses a “heading” proxy:
- `psi_vis ≈ |κ_gate| * s_visible_m`
- `psi_thresh = PsiFOVRad - PsiMarginRad`

When `psi_vis` crosses `psi_thresh` (plus some hysteresis/counters), VTSC latches:
- `self._fov_occluded = True`

Key Params:
- `VisionTurnSpeedControlPsiFOVRad`
- `VisionTurnSpeedControlPsiMarginRad`
- `VisionTurnSpeedControlFOVPretriggerTimeS`
- `VisionTurnSpeedControlFOVNOn`
- `VisionTurnSpeedControlFOVNOff`

The controller also computes a time-to-FOV-exit (`ttfov_s`) and can “pretrigger”
occlusion when the exit is imminent *and* confidence is already degraded.

**Clear behavior (human-like recovery)**  
Once geometry indicates the path is safely within FoV again (`psi_vis < psi_thresh` for the clear hysteresis),
the FOV-occlusion latch is cleared even if lane-line confidence remains mediocre.

Rationale:
- In real-world driving, lane-line confidence can stay low for benign reasons (wear, glare, lead artifacts).
- VTSC is an assistant; it should not “stick” in occlusion on a straight just because confidence is subpar.

## 5) “Fail-Open” Guard (Don’t Occlude a Clean Freeway)

There is a sanity guard intended to prevent occlusion logic from depressing speed on
straight freeway conditions with decent confidence and long visible horizon.

The guard sets:
- `self._freeway_failopen_active = True`

Key Params:
- `VTSCFailOpen` (forces fail-open; developer triage)

## 6) Lead-Vehicle Bypass (Critical Behavior)

Problem this addresses:
- A lead vehicle can reduce `laneLineProbs` without actually representing a *visibility* occlusion.
- Historically this caused VTSC’s occlusion/no-raise logic to “stick” and refuse to accelerate.

Solution in this fork:
- VTSC has a **lead-aware occlusion bypass** that disables occlusion effects when:
  - a lead is present, and
  - headway is within a configurable threshold.

Key Params:
- `VisionTurnSpeedControlOcclBypassWithLead` (default `1`)
- `VisionTurnSpeedControlOcclBypassHeadwayS` (default `3.0`)

Design note:
- This bypass is intentionally aimed at the “lead is causing lane-line confidence artifacts” case.
- It is not a general replacement for real occlusion handling (hillside/wall).

## 7) “No-Raise” Behavior (What It Is and Why It Exists)

When occlusion effects are active (FOV-occluded, not fail-open, not lead-bypassed), VTSC may:
- **forbid increasing speed** unless a “positive margin” criterion is met.

Intuition:
- If we don’t have enough visible distance to guarantee comfortable braking for a tighter curve
  just outside view, accelerating is risky.

In code, this is expressed as:
- “no positive accel” gates (and in the current implementation, equivalently a speed-cap clamp so
  the planner won’t accelerate).

The most important invariant (covered by tests):
- In occluded/no-margin cases **we should not accelerate**.
- With a close lead vehicle, this no-raise behavior should **not** activate just because
  lane-line confidence is low.

### 7.1 Severe-confidence (non-FOV) no-raise
In addition to FOV-gated occlusion, this fork also applies a conservative guard when
vision confidence is *extremely low* (`SEVERE` / `LOST`):
- **do not allow increasing speed** even if the FOV gate does not trigger.

Why:
- gentle curvature (or low psi) can avoid the FOV gate, but very low confidence still
  indicates we do not have a reliable vision signal to justify accelerating.

Important:
- The **lead-bypass** mechanism overrides this (close headway lead should not, by itself,
  suppress acceleration due to lane-line artifacts).
- This guard is only applied when the path curvature is non-trivial (i.e., when VTSC is relevant).
  On straight roads, low lane-line confidence can happen for reasons unrelated to “can’t see around a bend”
  (glare, worn paint, lead vehicle partially covering markings) and VTSC should not interfere.

## 8) Why the Logic Can Feel Over-Engineered (And How to Evaluate It)

There are multiple interacting mechanisms:
- confidence smoothing,
- enter/exit dwell timing,
- dropout grace periods,
- geometry-based FOV gating with pretrigger,
- occlusion cap arbitration gates (psi gate, “double-cap guard”),
- and no-raise windows.

This can create “self-interference” failure modes:
- e.g., a conservative gate blocks recovery because a state variable latches longer than intended.

The recommended way to evaluate whether a piece of logic is *still appropriate*:
1) Identify the intended real-world scenario it protects (hillside/wall/true occlusion).
2) Add a synthetic test representing a benign scenario (lead on straight) where it must *not* activate.
3) Add a replay regression based on a **tici/comma3x rlog** if the bug was observed on-road.
4) Prefer simpler gating/threshold/hysteresis where possible; avoid overlapping “guard rails” that
   all clamp in the same direction.

## 9) How We Test Occlusion Behavior (Synthetic + Pipeline)

### 9.1 Fast scenario tests (controller-level)
Run:
```bash
pytest -q sunnypilot/selfdrive/controls/lib/tests/vtsc
```

Notable tests:
- `test_fov_occlusion_clears_on_straight_even_with_mediocre_confidence`
  - ensures FOV-occlusion clears on the straight (even if confidence stays mediocre), with or without a lead.
- `test_lead_bypass_only_applies_at_close_headway`
  - ensures a far lead behaves like “no lead” for occlusion purposes.
- `test_severe_confidence_on_straight_does_not_block_raise`
  - ensures severe confidence alone does not block raising on a straight (curvature ~0).
- `test_severe_confidence_model_flat_steering_fallback_slows_for_sharp_curve`
  - ensures severe confidence + model-flat curvature does not “fail open” when steering clearly indicates a real curve.
- `test_partial_occlusion_no_lead_blocks_raise_when_no_margin`
  - ensures occluded/no-margin cases do not accelerate.

### 9.2 Pipeline ingestion tests (planner + MPC input)
These tests ensure VTSC’s computed cap is what the planner actually uses:
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_pipeline_integration.py`
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_flow.py`

## 10) Rlog Regressions (Closest to “Live”)

When adding rlog-based regressions for occlusion/overslow/recovery:
- rlog data must originate from **TICI / comma3x** on-road drives, even if later copied to a dev laptop.

See:
- `docs/vtsc/RLOGS.md`
