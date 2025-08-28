# Vision Turn Speed Controller — Dev Tunables (Curated)

Developer‑focused subset of VTSC parameters that are either wired today or high‑leverage to tune defaults without rebuild. Everything else has been removed to avoid noise.

Notes
- Units: speeds in m/s unless noted, accelerations in m/s², jerks in m/s³.
- Refresh: controller polls Params every ~5 s (`_update_params`).
- Status: Wired = used today; Not wired (planned) = worth exposing next for dev iterations.

## 1) Core Switches

Parameter: `VisionTurnSpeedControl`
- Default: true | Type: bool | Status: Wired
- Effect: Master enable for VTSC.

Parameter: `VisionTurnSpeedControlAggressiveness`
- Default: 1.0 | Range: 0.5–2.0 | Type: float | Status: Wired
- Effect: Scales pre‑curve anticipation time (lower = earlier, gentler slowing).

Parameter: `VisionTurnSpeedControlFixedLeadTimeSeconds`
- Default: 0.0 (off) | Range: 0.0–10.0 | Type: float | Status: Wired
- Effect: Hard override of dynamic anticipation time when >0. Use for controlled A/B.

Parameter: `VisionTurnSpeedControlSpeedIncreaseFactor`
- Default: 1.0 | Range: 0.9–1.2 | Type: float | Status: Not wired (planned)
- Effect: Global multiplier on physics‑computed curve speed. Keep near 1.0.

## 2) Adaptive Decel & Filtering

Parameter: `VisionTurnSpeedControlFilterAlpha`
- Default: 0.3 | Range: 0.1–0.9 | Type: float | Status: Wired
- Effect: EMA smoothing on required decel signal (lower = smoother, slower response).

Parameter: `VisionTurnSpeedControlHysteresisThreshold`
- Default: 0.2 | Range: 0.1–0.5 | Type: float | Status: Wired
- Effect: Band to exit adaptive decel back to comfort.

Parameter: `VisionTurnSpeedControlSafetyBias`
- Default: 0.1 | Range: 0.0–0.5 | Type: float | Status: Wired
- Effect: Bias on physics decel requirement (ensures hitting target before apex).

Parameter: `VisionTurnSpeedControlComfortDecelLimit`
- Default: −1.47 | Range: −1.0 to −2.0 | Type: float | Status: Not wired (planned)
- Effect: Comfort decel target before escalating to adaptive.

Parameter: `VisionTurnSpeedControlComfortJerkLimit`
- Default: −2.0 | Range: −1.0 to −3.0 | Type: float | Status: Not wired (planned)
- Effect: Comfort jerk cap when ramping braking.

Parameter: `VisionTurnSpeedControlMaxAdaptiveDecel`
- Default: −6.0 | Range: −3.5 to −7.0 | Type: float | Status: Not wired (planned)
- Effect: Hard floor for adaptive decel.

Parameter: `VisionTurnSpeedControlMaxAdaptiveJerk`
- Default: −6.0 | Range: −3.0 to −8.0 | Type: float | Status: Not wired (planned)
- Effect: Hard floor for adaptive jerk during decel.

## 3) Anticipation & Overshoot Planning

Parameter: `VisionTurnSpeedControlPlanningDecelLimit`
- Default: 3.5 | Range: 2.0–5.0 | Units: m/s² | Status: Not wired (planned)
- Effect: Decel used to decide “start slowing now”. Higher = starts later.

Parameter: `VisionTurnSpeedControlOvershootSafetyMargin`
- Default: 1.2 | Range: 1.0–1.5 | Type: float | Status: Not wired (planned)
- Effect: Multiplier on required decel distance for safety.

Parameter: `VisionTurnSpeedControlOvershootMinDistance`
- Default: 10.0 | Range: 5–25 | Units: m | Status: Not wired (planned)
- Effect: Minimum distance considered for overshoot logic.

Parameter: `VisionTurnSpeedControlAnticipationTargetReduction`
- Default: 0.95 | Range: 0.90–1.00 | Type: float | Status: Not wired (planned)
- Effect: Slight reduction while decelerating to ensure hitting target pre‑apex.

## 4) Curve Detection & Physics

Parameter: `VisionTurnSpeedControlCurvatureEMAFactor`
- Default: 0.3 | Range: 0.1–0.5 | Type: float | Status: Not wired (planned)
- Effect: EMA ratio for smoothing predicted curvature.

Physics model (sigmoid; expert but useful now)
- `VisionTurnSpeedControlPhysicsBaseline`
  - Default: 3.1447 | Range: 2.7–3.4 | Units: m/s² | Status: Not wired (planned)
  - Effect: Baseline lateral accel on easy curves; higher = higher speeds broadly.
- `VisionTurnSpeedControlPhysicsAmplitude`
  - Default: −1.1751 | Range: −0.8 to −1.5 | Units: m/s² | Status: Not wired (planned)
  - Effect: Depth of reduction as curvature tightens; more negative = stronger slow‑down.
- `VisionTurnSpeedControlPhysicsSteepness`
  - Default: −2000.0 | Range: −1000 to −3000 | Status: Not wired (planned)
  - Effect: Transition steepness; closer to 0 flattens, more negative sharpens the drop‑off.
- `VisionTurnSpeedControlPhysicsCenter`
  - Default: 0.004778 | Range: 0.003–0.007 | Units: 1/m | Status: Not wired (planned)
  - Effect: Curvature where the transition centers; lower shifts speed reduction to tighter curves.
- `VisionTurnSpeedControlPhysicsMinLatAccel`
  - Default: 1.8 | Range: 1.5–2.2 | Units: m/s² | Status: Not wired (planned)
  - Effect: Floor clamp on allowed lateral accel for very tight curves.
- `VisionTurnSpeedControlPhysicsMaxLatAccel`
  - Default: 3.12 | Range: 2.8–3.5 | Units: m/s² | Status: Not wired (planned)
  - Effect: Ceiling clamp on allowed lateral accel for straight/easy curves.

Simple low‑speed bias (quick “+Δ mph” under ~50 mph)
- `VisionTurnSpeedControlLowSpeedSpeedBiasMph`
  - Default: 0.0 | Range: −3.0 to +3.0 | Units: mph | Status: Not wired (planned)
  - Effect: Adds a small speed bias to the physics target at low speeds.
  - Notes: Applied with a smooth taper that goes to zero by `LowSpeedBiasEndMph`.
- `VisionTurnSpeedControlLowSpeedBiasEndMph`
  - Default: 50.0 | Range: 40–60 | Units: mph | Status: Not wired (planned)
  - Effect: Upper end of the taper; above this, bias ≈ 0.

Parameter: `VisionTurnSpeedControlSmoothingMaxDecel`
- Default: 3.5 | Range: 2.0–5.0 | Type: float | Status: Not wired (planned)
- Effect: Bound on negative accel changes in internal smoothing.

Parameter: `VisionTurnSpeedControlSmoothingMaxJerk`
- Default: 6.0 | Range: 3.0–10.0 | Type: float | Status: Not wired (planned)
- Effect: Bound on jerk used in target accel updates.

Parameter: `VisionTurnSpeedControlAccelToDecelRatio`
- Default: 1.3 | Range: 1.0–1.6 | Type: float | Status: Not wired (planned)
- Effect: Positive accel limit relative to decel limit.

Parameter: `VisionTurnSpeedControlJerkAccelMultiplier`
- Default: 2.0 | Range: 1.0–3.0 | Type: float | Status: Not wired (planned)
- Effect: Positive vs negative jerk ratio for smoother exits.

Parameter: `VisionTurnSpeedControlMaxSpeed`
- Default: 70.0 | Range: 50–85 | Units: m/s | Status: Not wired (planned)
- Effect: Straight‑road speed ceiling when curvature is ~0.

Parameter: `VisionTurnSpeedControlMinOperatingSpeed`
- Default: 2.24 | Range: 1.5–3.5 | Units: m/s | Status: Not wired (planned)
- Effect: Floor for speed clamps; avoids fighting at walking speeds.

## 5) Apex Detection & Exit Boost

Detection
- `VisionTurnSpeedControlApexThreshold`: 5e‑5 | Range: 1e‑5–1e‑4 | Type: float | Status: Not wired (planned)
- `VisionTurnSpeedControlApexProminence`: 1e‑4 | Range: 5e‑5–5e‑4 | Type: float | Status: Not wired (planned)
- `VisionTurnSpeedControlApexHysteresisTime`: 2.0 | Range: 0.5–5.0 | Units: s | Status: Not wired (planned)
- `VisionTurnSpeedControlApexMetersPerIndex`: 2.0 | Range: 1.0–3.0 | Units: m | Status: Not wired (planned)
- `VisionTurnSpeedControlApexNearIndex`: 3 | Range: 1–6 | Type: int | Status: Not wired (planned)

Boost behavior
- `VisionTurnSpeedControlApexBoostDistance`: 50.0 | Range: 20–100 | Units: m | Status: Not wired (planned)
- `VisionTurnSpeedControlApexBoostFactor`: 0.1 | Range: 0.0–0.2 | Type: float | Status: Not wired (planned)
- `VisionTurnSpeedControlApexBoostMinLatAccel`: 1.0 | Range: 0.5–1.5 | Units: m/s² | Status: Not wired (planned)
- `VisionTurnSpeedControlApexBoostCenter`: 2.0 | Range: 1.5–3.0 | Units: m/s² | Status: Not wired (planned)
- `VisionTurnSpeedControlApexBoostWidth`: 0.5 | Range: 0.2–1.0 | Units: m/s² | Status: Not wired (planned)
- `VisionTurnSpeedControlBoostSafetyCurvatureScale`: 0.7 | Range: 0.6–0.9 | Type: float | Status: Not wired (planned)

Why: Small post‑apex lift reduces the perception of “dragging speed out of the turn”.

## 6) Vision Occlusion

Parameter: `VisionTurnSpeedControlVisionConfAlpha`
- Default: 0.1 | Range: 0.05–0.4 | Type: float | Status: Not wired (planned)
- Effect: EMA smoothing for vision confidence.

Parameter: `VisionTurnSpeedControlVisionConfGoodThreshold`
- Default: 0.75 | Range: 0.6–0.9 | Type: float | Status: Not wired (planned)
- Effect: Threshold to (re)enter good‑vision mode.

Parameter: `VisionTurnSpeedControlVisionConfBadThreshold`
- Default: 0.70 | Range: 0.5–0.8 | Type: float | Status: Not wired (planned)
- Effect: Threshold to leave good‑vision mode (begin holding last curvature).

---

Deliberately trimmed
- Removed preset profiles, dynamic scaling group, UI state thresholds, alternative nonlinear model, and low‑leverage physics sigmoid internals (amplitude/steepness/center/baseline). These add cognitive overhead with little day‑to‑day tuning value.

Param usage
- Use `Params().put(key, str(value))`; VTSC re‑reads about every 5 s.
- Validate in safe conditions; comfort/adaptive limits interact with physics‑based targets.
