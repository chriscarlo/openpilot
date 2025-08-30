**MTSC (Map Turn Speed Controller) — Implementation Plan**

- Owner: Chauffeur VTSC/Maps
- Process name: `mtscd` (Map Turn Speed Controller daemon)
- Cereal service: `mapTurnSpeedControlSP`
- Message struct: `Custom.MapTurnSpeedControlSP`
- Feature flag: Param `MTSCEnabled` (default: false)
- Rate: 10 Hz publish; ≤2 ms avg compute at 10 Hz on device

**Purpose & Scope**
- Goal: Provide strategic, map-derived turn speed recommendations that enable early, comfort-limited slowing for edge cases (hidden ramps, hairpins, switchbacks), while leaving tactical second-to-second control to vision.
- Non-goal: Replace VTSC or model-based tactical control; do not command aggressive near-field decel; do not require network during drive (offline-friendly).

**Guiding Principles**
- Vision-first: Vision/model horizon is the source of truth tactically; MTSC only adds far-field context as an upper bound for early decel.
- Conservative by design: Bound deceleration to comfort when occluded; never escalate based on map alone.
- Robust alignment: Use heading, centerline proximity, continuity, road class, and level separation to avoid overpass/parallel-road mistakes.
- Graceful degradation: If maps/GPS are unavailable, MTSC publishes unavailable with no effect on planning.

**High-Level Architecture**
- Producer: `mtscd` subscribes to `liveMapDataSP`, `carState`, `selfdriveState`, and the platform GPS service (for freshness), computes a curvature/speed horizon, and publishes `mapTurnSpeedControlSP` (MTSC output).
- Consumer: Longitudinal stack (VTSC fusion path) reads `mapTurnSpeedControlSP` and takes the min() of speeds under strict gating (beyond visible horizon, sub-65 mph, sufficient confidence). Vision remains tactical authority.
- Manager wiring: Separate process started onroad (and when `MTSCEnabled`), independent from VTSC for safety and observability.

**Interfaces**
- Inputs (SubMaster):
  - `liveMapDataSP` (1 Hz): current and nearby `RoadSegment`s with centerline samples, roadClass, levelSeparation, optional maxSpeed.
  - `carState` (100 Hz → sampled): `vEgo`, gear, standstill.
  - `selfdriveState` (100 Hz → sampled): enabled state for gating.
  - `gps_location_service` (platform-specific): for staleness checks only.
- Outputs (PubMaster):
  - `mapTurnSpeedControlSP` (10 Hz): MTSC recommendation + diagnostics (see Schema).
- Params (runtime tunables):
  - `MTSCEnabled` (bool; default false)
  - `MTSCMinConfidence` (float; default 0.70)
  - `MTSCSpeedGateMps` (float; default 29.06 ≈ 65 mph)
  - `MTSCHorizonMinS`/`MTSCHorizonMaxS` (floats; default 15/30)
  - `MTSCResampleM` (float; default 3.0)
  - `MTSCMinCoverage` (float; default 0.60)
  - `MTSCHeadingMaxDeg` (float; default 12.0)
  - `MTSCMaxCenterlineDistanceM` (float; default 10.0)
  - `MTSCConfidenceWeights` (JSON: distance, heading, continuity, class, level)
  - `MTSCCurvSmoothingAlpha` (float; default 0.25)
  - `MTSCComfortDecelMps2` (float; default 1.47)
  - `MTSCJerkLimitMps3` (float; default 2.0)
  - `MTSCLogDetail` (int; 0..2)

**Cereal Schema**
- File: `cereal/custom.capnp`
- Add struct:
  - `struct MapTurnSpeedControlSP {`
  - `  timeStamp @0 :UInt64;`
  - `  available @1 :Bool;`
  - `  confidence @2 :Float32;`
  - `  targetSpeedMps @3 :Float32;`
  - `  startDistanceM @4 :Float32;`
  - `  horizonCoverage @5 :Float32;`
  - `  minSpeedMps @6 :Float32;`
  - `  minSpeedAtDistanceM @7 :Float32;`
  - `  matchedWayId @8 :UInt64;`
  - `  roadClass @9 :Custom.LiveMapDataSP.RoadSegment.RoadClass;`
  - `  levelSeparation @10 :Int8;`
  - `  headingErrorDeg @11 :Float32;`
  - `  distanceToCenterlineM @12 :Float32;`
  - `  visHorizonM @13 :Float32;`
  - `  distancesM @14 :List(Float32);   # optional, ≤30 samples (debug)`
  - `  kappasPerM @15 :List(Float32);   # optional, ≤30 samples (debug)`
  - `  vSafeMps @16 :List(Float32);     # optional, ≤30 samples (debug)`
  - `}`
- File: `cereal/services.py`
  - Add: `'mapTurnSpeedControlSP': (True, 10., 10)`
- File: `cereal/log.capnp`
  - Replace one `customReserved*` union field with: `mapTurnSpeedControlSP @136 :Custom.MapTurnSpeedControlSP;` (choose the first free reserved slot; keep ID stable).

**Process & Manager Wiring**
- New module: `sunnypilot/selfdrive/controls/mtsc/mtscd.py` (daemon)
  - Sub: `['liveMapDataSP','carState','selfdriveState', gps_location_service]`
  - Pub: `['mapTurnSpeedControlSP']`
  - Ratekeeper at 10 Hz
  - Enabled onroad if `MTSCEnabled` and OSM path exists
- Manager: `system/manager/process_config.py`
  - `PythonProcess("mtscd", "sunnypilot.selfdrive.controls.mtsc.mtscd", only_onroad, enabled_by_param=MTSCEnabled)`
  - Use a small predicate in-line: returns onroad AND Params.get_bool("MTSCEnabled")

**Core Algorithm (mtscd)**
- Way selection:
  - Prefer `liveMapDataSP.currentRoadSegment` if present; otherwise select best from `nearbyRoadSegments`.
  - Compute:
    - `d_center`: distance to centerline at projected point.
    - `heading_err`: |ego heading − segment roadDirection|, normalized to 0..180.
    - `class_ok`: motorway/trunk/primary and *_link only (initial rollout).
    - `level_ok`: prefer `levelSeparation==0`; drop if toggling levels without continuity.
    - Continuity: keep last matched `wayId` unless new candidate is better for ≥0.4 s.
  - Confidence score in [0..1] from weighted sum; require ≥ `MTSCMinConfidence`.
- GPS/time alignment:
  - Treat `currentRoadSegment` projection as ground-truth alignment; if GPS stale (>0.5 s), propagate along-track by `v_ego*Δt` (clamped to small window).
- Horizon construction:
  - Visible horizon `s_vis = v_ego * vis_horizon_s` (pull from VTSC if available; else assume 1.2–1.4 s).
  - Target horizon length `S = clamp(v_ego*T, min=80 m, max=450 m)` with T ∈ [`MTSCHorizonMinS`,`MTSCHorizonMaxS`].
  - Resample centerline every `MTSCResampleM` meters beyond the projection; compute coverage fraction in S (require ≥ `MTSCMinCoverage`).
- Curvature & speed:
  - Curvature: 3‑point chord area formula; smooth EMA with `MTSCCurvSmoothingAlpha`.
  - Speed: reuse VTSC’s `curvature_to_speed(k)` to compute physics envelope per sample.
  - Strategic target: jerk/comfort-limited decel (≤ `MTSCComfortDecelMps2`, jerk ≤ `MTSCJerkLimitMps3`) toward the minimum physics speed in the far horizon; do not induce decel inside `s_vis` purely from map.
- Overpass/false positives:
  - Penalize confidence if `levelSeparation` differs from last or fluctuates; reject if heading_err large while `d_center` small (parallel above/below).
  - Class gate avoids residential/service complexity early on.
- Output policy:
  - `available=true` only if gates pass: speed ≤ `MTSCSpeedGateMps`, confidence ≥ min, coverage ≥ min, geometry valid, inputs fresh.
  - `startDistanceM = max(s_vis + margin, 0)` where margin ≈ 10 m; consumer must only apply beyond this.
  - Include diag vectors (decimated ≤30 samples) when `MTSCLogDetail>=2`.

**Consumer Integration (Vision/Planner)**
- Injectable points (choose one during implementation):
  - LongitudinalPlannerSP.update_v_cruise: read `mapTurnSpeedControlSP`; compute `v_cruise_mtsc = targetSpeedMps` if `available && confidence≥min`; then `v_cruise_final = min(v_cruise_final, v_cruise_mtsc)`.
  - VisionTurnController: optionally read MTSC to refine far-field bound; still keep final min() in planner for transparency.
- Gating at consumer:
  - Enforce `startDistanceM` and visible-horizon rules; if VTSC says positive margin near-field, ignore map for near-field.
  - Never allow map to increase decel beyond comfort while occluded; VTSC already caps — consumer keeps min().

**Telemetry & Logging**
- Publish diag fields in `mapTurnSpeedControlSP`: `confidence`, `horizonCoverage`, `matchedWayId`, `roadClass`, `levelSeparation`, `headingErrorDeg`, `distanceToCenterlineM`, `visHorizonM`.
- Cloudlog key transitions: way switch, confidence drop, overpass rejection, coverage fail, enable/disable.
- Optional CSV ring buffer for dev builds (behind `MTSCLogDetail`).

**Performance Targets**
- ≤ 2 ms avg at 10 Hz on device for typical horizon lengths; memory stable; no allocations > 1 MB per tick.
- Message size bounded via decimation limits.

**Security/Privacy**
- No network usage onroad; offline tiles only. Do not log raw GPS traces beyond standard services. MTSC outputs are derived and low-rate.

**Failure Modes & Fallbacks**
- No OSM/`liveMapDataSP` invalid/stale: publish `available=false`.
- Confidence < min or coverage < min: `available=false`.
- GPS stale > 1 s: `available=false` (or reduced confidence if along-track propagation still reliable at low Δt).
- Any exception: swallow, log, and publish `available=false`.

**Milestones & Checkpoints**
- M0 — Schema & Plumbing
  - Add capnp struct, services entry, log union field.
  - Build passes; service visible to messaging.
  - Acceptance: basic publisher test sends/receives empty/unavailable messages.
- M1 — Process Skeleton
  - `mtscd` loop with subscriptions, freshness checks, and empty recommendation logic; feature flag gating.
  - Acceptance: process starts/stops with Param; publishes `available=false`; no crashes over 1 hr idle.
- M2 — Matching & Confidence
  - Implement way selection, heading/centerline checks, continuity/hysteresis, class/level gates.
  - Unit tests: overpass vs same-road, parallel roads, heading flips; confidence behaves as expected.
  - Acceptance: confidence ≥ 0.7 on clearly matched highways; < 0.5 on overpass/parallel.
- M3 — Horizon & Curvature
  - Implement resampling, coverage, curvature, smoothing; publish diag vectors in dev mode.
  - Unit tests: curvature computation matches expected values on synthetic arcs; coverage math correct.
  - Acceptance: Placerville ramp curvature min matches README estimates ±15%.
- M4 — Speed Profile & Comfort Bounds
  - Use VTSC `curvature_to_speed`; generate strategic target with jerk/comfort bounds and start distance ≥ `s_vis`.
  - Unit tests: comfort-limited decel profile monotonic and starts beyond `s_vis`.
  - Acceptance: simulated hidden-turn shows early, gentle slowing vs. baseline.
- M5 — Consumer Integration (flagged)
  - Add planner min() with MTSC target; gated by `startDistanceM` and confidence.
  - Tests: regression — highway straight unaffected; occlusion hidden-turn improved.
  - Acceptance: integration tests green; off-by-default.
- M6 — Onroad Pilot & Telemetry
  - Limited rollout with telemetry; collect stats on activations, decel magnitude, overrides.
  - Acceptance: no regressions; positive early-slowing in targeted routes.
- M7 — Harden & Enable (optional)
  - Tuning of thresholds; expanded roadClass set; enable by default if success criteria met.

**Test Plan (Details)**
- Unit
  - Geometry: projection to centerline, along-track interpolation, distance/heading computations.
  - Curvature: chord-based curvature vs. analytic arcs; smoothing EMA response.
  - Confidence: scoring under distance/heading/class/level perturbations.
  - Speed conversion: `curvature_to_speed` envelopes and monotonic decel profile.
- Integration (simulation/harness)
  - Hidden ramp (Placerville) at various approach speeds.
  - Hairpin/switchbacks: S-curves requiring ≤25 mph mid-corner.
  - Overpass above/parallel service road: ensure `available=false` or high `startDistanceM` with low confidence.
  - Highway straight 70–75 mph: no effect (available=false due to speed gate or target≈cruise).
  - Occlusion: vision low-confidence segment with upcoming curve beyond FoV → early comfort decel only.
- Onroad (pilot)
  - 5–10 handpicked routes covering ramps, mountain roads, and mixed urban highways.
  - Record: `confidence`, `coverage`, `targetSpeedMps`, `startDistanceM`, applied planner min(), jerk/decel usage.

**Acceptance Criteria (Key)**
- No change in throttle/brake actuation on highway straights at 70–75 mph.
- Hidden-turn scenario: MTSC causes ≥1.5 s earlier decel vs. baseline, decel magnitude ≤ comfort cap while occluded.
- Overpass/parallel: MTSC either unavailable or startDistanceM beyond any near-field impact; target not below vision envelope.
- Time alignment: no oscillatory advice switch; way continuity holds in steady driving.

**Risks & Mitigations**
- Overpass false matching → mitigated by level separation, heading, continuity, and class gates; require high confidence.
- GPS staleness → along-track propagation + freshness checks; drop to unavailable beyond thresholds.
- Map geometry errors → require coverage; fall back to unavailable.
- Overly conservative speeds → comfort bounds + gating; optional param for min confidence and coverage.

**Developer How-To**
- Enable: `params put MTSCEnabled 1`.
- Inspect: subscribe to `mapTurnSpeedControlSP`; check `confidence`, `horizonCoverage`, `matchedWayId`.
- Tuning: adjust `MTSCMinConfidence`, `MTSCHorizon*`, `MTSCResampleM`, `MTSCMinCoverage`.
- Logs: cloudlog for transitions; set `MTSCLogDetail=2` for diag vectors (dev only).

**Future Extensions**
- Use elevation/grade to adjust comfort decel on steep ramps.
- Lane-level path selection when lane centerlines are available.
- Persist matched way across drives to improve continuity on reconnect.

**Checklist (Per Milestone)**
- [ ] Schema added (capnp, services, union) and compiled
- [ ] Process created, manager wired, flag gated
- [ ] Matching+confidence unit-tested
- [ ] Horizon+curvature unit-tested
- [ ] Speed profile with comfort bounds unit-tested
- [ ] Consumer min() integration off-by-default
- [ ] Integration scenarios validated (hidden ramp, overpass, hairpins, highway straight)
- [ ] Onroad pilot complete with telemetry
- [ ] Thresholds tuned; docs updated

