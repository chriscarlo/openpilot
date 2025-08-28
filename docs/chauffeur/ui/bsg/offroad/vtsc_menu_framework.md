Title: VTSC Offroad Menu — Framework & Layout (Dev)

Purpose: Provide a clear, driver‑friendly menu structure for Vision Turn Speed Control (VTSC) that’s easy to tune while testing. Aligns with Offroad BSG patterns and avoids overwhelming options. Focus on grouped, intuitive controls with concise microcopy.

Entry & Navigation
- Path: Settings → Cruise → VTSC
- Hub row: gear + toggle row per Feature Hub pattern
  - Title: Vision Turn Speed Control
  - Description (collapsed): Sets target speeds for upcoming curves from vision + physics.
  - Toggle: `VisionTurnSpeedControl` (enabled → gear active)
  - Settings gear: opens this VTSC sub‑panel

Sub‑Panel Header (Feature Menu pattern)
- Back button (top‑left), Title centered: Vision Turn Speed Control
- One‑line description: Tune curve speed behavior. Changes apply live (~5 s).

Section Overview
Group controls into a few logical cards. Keep the most used items first; place advanced controls behind a reveal toggle where noted.

1) Driving Style (Card)
- VisionTurnSpeedControlAggressiveness (range)
  - Label: Aggressiveness
  - Desc: How early to slow before curves. Lower = earlier, calmer; higher = later, firmer.
  - Units: × (no unit); Default: 1.0; Range: 0.5–2.0; Step: 0.1
- VisionTurnSpeedControlFixedLeadTimeSeconds (range)
  - Label: Fixed Lead Time
  - Desc: Force the same heads‑up time before curves (seconds). Set to 0 to let VTSC decide.
  - Units: s; Default: 0.0; Range: 0.0–10.0; Step: 0.5
- VisionTurnSpeedControlLowSpeedSpeedBiasMph (range) [planned]
  - Label: Low‑Speed Bias
  - Desc: Add/subtract a few mph under town speeds. Use small nudges.
  - Units: mph; Default: 0.0; Range: −3.0–+3.0; Step: 0.5
- VisionTurnSpeedControlLowSpeedBiasEndMph (range) [planned]
  - Label: Bias Ends By
  - Desc: Bias tapers to zero by this speed. Above it, no extra mph.
  - Units: mph; Default: 50; Range: 40–60; Step: 5
- VisionTurnSpeedControlSpeedIncreaseFactor (range, Advanced) [planned]
  - Label: Global Speed Bias
  - Desc: Nudge speeds up/down everywhere. Keep near 1.0.
  - Units: ×; Default: 1.0; Range: 0.9–1.2; Step: 0.02

2) Curve Physics (Card, Advanced reveal)
- Header hint: Expert — affects speed on all curves
- VisionTurnSpeedControlPhysicsBaseline (range) [planned]
  - Label: Baseline Cornering Force
  - Desc: Top cornering force on easy bends. Higher = higher speeds broadly.
  - Units: m/s²; Default: 3.1447; Range: 2.7–3.4; Step: 0.05
- VisionTurnSpeedControlPhysicsCenter (range) [planned]
  - Label: Transition Curvature
  - Desc: Where slowing becomes noticeable. Lower = holds speed longer on mild curves.
  - Units: 1/m; Default: 0.004778; Range: 0.003–0.007; Step: 0.0002
- VisionTurnSpeedControlPhysicsSteepness (range) [planned]
  - Label: Transition Sharpness
  - Desc: How quickly speeds drop as curves tighten. More negative = sharper drop‑off.
  - Units: (unitless); Default: −2000; Range: −1000 to −3000; Step: 100
- VisionTurnSpeedControlPhysicsAmplitude (range) [planned]
  - Label: Tight‑Curve Slow‑Down
  - Desc: Total slow‑down on tight curves. More negative = slower on tight bends.
  - Units: m/s²; Default: −1.1751; Range: −0.8 to −1.5; Step: 0.05
- VisionTurnSpeedControlPhysicsMaxLatAccel (range) [planned]
  - Label: Max Lateral Accel
  - Desc: Ceiling on cornering force for easy/straight roads.
  - Units: m/s²; Default: 3.12; Range: 2.8–3.5; Step: 0.05
- VisionTurnSpeedControlPhysicsMinLatAccel (range) [planned]
  - Label: Min Lateral Accel
  - Desc: Floor on cornering force for very tight curves.
  - Units: m/s²; Default: 1.8; Range: 1.5–2.2; Step: 0.05

3) Adaptive Braking & Filtering (Card)
- VisionTurnSpeedControlFilterAlpha (range)
  - Label: Decel Filter
  - Desc: How fast braking responds to new info. Lower = smoother, slower.
  - Units: ×; Default: 0.3; Range: 0.1–0.9; Step: 0.05
- VisionTurnSpeedControlHysteresisThreshold (range)
  - Label: Hysteresis
  - Desc: Prevents bouncing between gentle/hard braking. Higher = stays hard longer.
  - Units: ×; Default: 0.2; Range: 0.1–0.5; Step: 0.05
- VisionTurnSpeedControlSafetyBias (range)
  - Label: Safety Margin
  - Desc: Small extra braking to hit target before the apex.
  - Units: ×; Default: 0.1; Range: 0.0–0.5; Step: 0.05
- Reveal “Comfort & Limits” (Advanced)
  - VisionTurnSpeedControlComfortDecelLimit (range) [planned]
    - Label: Comfort Decel Limit
    - Desc: Preferred braking strength before going harder.
    - Units: m/s²; Default: −1.47; Range: −1.0 to −2.0; Step: 0.1
  - VisionTurnSpeedControlComfortJerkLimit (range) [planned]
    - Label: Comfort Jerk Limit
    - Desc: How quickly braking ramps in. Higher magnitude = snappier.
    - Units: m/s³; Default: −2.0; Range: −1.0 to −3.0; Step: 0.2
  - VisionTurnSpeedControlMaxAdaptiveDecel (range) [planned]
    - Label: Max Adaptive Decel
    - Desc: Hard cap for late/tight cases. Use carefully.
    - Units: m/s²; Default: −6.0; Range: −3.5 to −7.0; Step: 0.5
  - VisionTurnSpeedControlMaxAdaptiveJerk (range) [planned]
    - Label: Max Adaptive Jerk
    - Desc: Hard cap for how quickly braking ramps in adaptively.
    - Units: m/s³; Default: −6.0; Range: −3.0 to −8.0; Step: 0.5

4) Anticipation & Overshoot (Card)
- VisionTurnSpeedControlPlanningDecelLimit (range) [planned]
  - Label: Planning Decel
  - Desc: How hard we plan to brake when deciding “start braking now”. Higher = later starts.
  - Units: m/s²; Default: 3.5; Range: 2.0–5.0; Step: 0.5
- VisionTurnSpeedControlOvershootSafetyMargin (range) [planned]
  - Label: Safety Distance
  - Desc: Extra room to avoid arriving too fast.
  - Units: ×; Default: 1.2; Range: 1.0–1.5; Step: 0.05
- VisionTurnSpeedControlOvershootMinDistance (range) [planned]
  - Label: Min Considered Distance
  - Desc: Ignore very small distances when planning.
  - Units: m; Default: 10; Range: 5–25; Step: 1
- VisionTurnSpeedControlAnticipationTargetReduction (range) [planned]
  - Label: Target Reduction While Braking
  - Desc: Nudge target down to reach it before the apex.
  - Units: ×; Default: 0.95; Range: 0.90–1.00; Step: 0.01

5) Apex & Exit Boost (Card)
- VisionTurnSpeedControlApexBoostDistance (range) [planned]
  - Label: Boost Window
  - Desc: How far past the apex we keep the extra push.
  - Units: m; Default: 50; Range: 20–100; Step: 5
- VisionTurnSpeedControlApexBoostFactor (range) [planned]
  - Label: Boost Amount
  - Desc: Extra speed after the apex (as a fraction). Keep small.
  - Units: ×; Default: 0.10; Range: 0.00–0.20; Step: 0.02
- Reveal “Detection & Onset” (Advanced)
  - VisionTurnSpeedControlApexThreshold (range) [planned]
    - Label: Apex Curvature Threshold
    - Desc: Minimum curvature to count as an apex.
    - Units: 1/m; Default: 5e−5; Range: 1e−5–1e−4; Step: 1e−5
  - VisionTurnSpeedControlApexProminence (range) [planned]
    - Label: Apex Prominence
    - Desc: How much higher the peak must be vs neighbors.
    - Units: 1/m; Default: 1e−4; Range: 5e−5–5e−4; Step: 5e−5
  - VisionTurnSpeedControlApexHysteresisTime (range) [planned]
    - Label: Re‑trigger Cooldown
    - Desc: Don’t re‑detect the same apex too soon.
    - Units: s; Default: 2.0; Range: 0.5–5.0; Step: 0.5
  - VisionTurnSpeedControlApexMetersPerIndex (range) [planned]
    - Label: Meters Per Index
    - Desc: Spacing used for proximity checks.
    - Units: m; Default: 2.0; Range: 1.0–3.0; Step: 0.5
  - VisionTurnSpeedControlApexNearIndex (range) [planned]
    - Label: Near‑Apex Indices
    - Desc: How close counts as “at/past” the apex.
    - Units: idx; Default: 3; Range: 1–6; Step: 1
  - VisionTurnSpeedControlApexBoostMinLatAccel (range) [planned]
    - Label: Boost Min Lateral Accel
    - Desc: Only boost when cornering force exceeds this.
    - Units: m/s²; Default: 1.0; Range: 0.5–1.5; Step: 0.1
  - VisionTurnSpeedControlApexBoostCenter (range) [planned]
    - Label: Boost Midpoint
    - Desc: Lateral accel where boost is about half.
    - Units: m/s²; Default: 2.0; Range: 1.5–3.0; Step: 0.1
  - VisionTurnSpeedControlApexBoostWidth (range) [planned]
    - Label: Boost Ramp Width
    - Desc: Smoothness of the boost ramp.
    - Units: m/s²; Default: 0.5; Range: 0.2–1.0; Step: 0.1
  - VisionTurnSpeedControlBoostSafetyCurvatureScale (range) [planned]
    - Label: Physics Clamp Margin
    - Desc: Safety factor when clamping boosted speed to physics.
    - Units: ×; Default: 0.7; Range: 0.6–0.9; Step: 0.05

6) Vision Occlusion (Card)
- VisionTurnSpeedControlVisionConfAlpha (range) [planned]
  - Label: Vision Confidence Filter
  - Desc: Smooths confidence to avoid flicker.
  - Units: ×; Default: 0.1; Range: 0.05–0.4; Step: 0.05
- VisionTurnSpeedControlVisionConfGoodThreshold (range) [planned]
  - Label: Good Vision Threshold
  - Desc: Confidence to (re)enter “good vision”.
  - Units: ×; Default: 0.75; Range: 0.6–0.9; Step: 0.05
- VisionTurnSpeedControlVisionConfBadThreshold (range) [planned]
  - Label: Leave‑Good Threshold
  - Desc: Confidence to leave “good vision” and hold last curve.
  - Units: ×; Default: 0.70; Range: 0.5–0.8; Step: 0.05

7) Limits (Card, Optional)
- VisionTurnSpeedControlMaxSpeed (range) [planned]
  - Label: Straight‑Road Ceiling
  - Desc: Max speed when the road is essentially straight.
  - Units: mph (display); Default: 156 (≈70 m/s); Range: 110–190; Step: 5
- VisionTurnSpeedControlMinOperatingSpeed (range) [planned]
  - Label: VTSC Minimum Speed
  - Desc: Don’t engage below this. Avoids fighting at walking speeds.
  - Units: mph (display); Default: 5; Range: 3–8; Step: 1

Footer Utilities
- Reset All to Defaults (button)
  - Style: secondary button in a card; confirm dialog
- Export/Import Preset (optional dev utility)
  - Simple file picker to save/load a JSON of Params keys (defer implementation until requested)

Interaction & Persistence
- Use Range Controls with ± and Reset; show units; immediate Params persistence on every tap.
- Show “(Default)” vs “(Modified)” under each value per BSG.
- Hide advanced controls behind a small “Advanced” reveal toggle inside their card.
- Disable all controls if `VisionTurnSpeedControl` is OFF (mirror the hub toggle).

Microcopy Principles (driver‑friendly)
- Short, plain language; directionality and effect first.
- Avoid jargon (e.g., say “extra push after the apex” not “post‑apex velocity bias”).
- Use examples sparingly (e.g., “Use small nudges”).
- Always show units explicitly in values.

Minimal Spec (wiring checklist)
- For each control specify: key, default, unit (display vs stored), range, step, epsilon for float equality.
- Ensure conversions: mph↔m/s for speed values.
- Add per‑section “Reset” (optional) and a global “Reset All”.
- Respect Offroad BSG tokens: padding, radii, type scale, colors.

Notes
- This framework is intentionally concise: core dials first, expert physics and detection behind reveals.
- The plan avoids exposing every model knob; instead, focuses on those most likely to be tuned on‑road.

