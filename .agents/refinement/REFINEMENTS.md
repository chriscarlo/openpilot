# Refinement Log

### VTSC Orphan Cleanup With Compatibility Guardrails
- **Date:** 2026-02-11
- **Classification:** A
- **Category:** Code
- **Status:** applied
- **Approval:** na (explicitly requested by user)
- **User-visible change:** no
- **Behavior/semantics change:** yes (legacy VTSC state machine no longer transitions; state held disabled)
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** no
- **Diverges from user proposal:** no
- **Files touched:** `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`, `sunnypilot/selfdrive/controls/lib/vision_turn_params.py`
- **Notes:** Removed dead constants/functions/branches and orphan controller fields while preserving MTSC map-lookahead code path and debug snapshot schema used by tooling.

### VTSC Follow-on Reference Cleanup
- **Date:** 2026-02-11
- **Classification:** A
- **Category:** Code
- **Status:** applied
- **Approval:** na (explicitly requested by user)
- **User-visible change:** no
- **Behavior/semantics change:** no
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** no
- **Diverges from user proposal:** no
- **Files touched:** `common/params_keys.h`, `docs/vtsc/PARAMS.md`, `docs/chauffeur/vtsc/debug/debug_2025-09-04/NOTES.md`, `docs/chauffeur/vtsc/debug/debug_2025-09-04/monitor_vtsc_checklist.md`, `docs/chauffeur/vtsc/testing/unit/test_dynamic_decel_scale_behavior.py`
- **Notes:** Removed stale references/imports to deleted VTSC internals and removed unused VisionFloorMode param key/docs references.

### RTI Same-Road Hardening (Street Mismatch + Side-Street Bearing Gate)
- **Date:** 2026-02-12
- **Classification:** C
- **Category:** Code
- **Status:** applied
- **Approval:** user-requested in-thread (explicit RTI behavior refinement request)
- **User-visible change:** yes (fewer RTI interventions for nearby off-road/side-street threats)
- **Behavior/semantics change:** yes
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** no
- **Diverges from user proposal:** no
- **Files touched:** `sunnypilot/rtid/threat_detector.py`, `sunnypilot/rtid/tests/test_threat_detector.py`
- **Notes:** Prevented distance-only fallback from overriding explicit street-name mismatches and added heading-gated rejection for likely cross-street threats when names are unavailable.

### RTI Controller Same-Road Enforcement For Longitudinal Action
- **Date:** 2026-02-12
- **Classification:** C
- **Category:** Code
- **Status:** applied
- **Approval:** user-requested in-thread (explicit RTI behavior refinement request)
- **User-visible change:** yes (off-road threats no longer drive speed recommendations)
- **Behavior/semantics change:** yes
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** no
- **Diverges from user proposal:** no
- **Files touched:** `sunnypilot/selfdrive/controls/lib/rti_controller.py`, `sunnypilot/selfdrive/controls/lib/tests/rti/test_rti_rampdown_unit.py`
- **Notes:** Enforced `onSameRoad` in controller threat selection and normalized direction parsing for robustness; added explicit regression test for off-road ahead threats.

### OpenAI-Assisted Threat Adjudication (Concept)
- **Date:** 2026-02-12
- **Classification:** D
- **Category:** Architecture
- **Status:** proposed
- **Approval:** pending
- **User-visible change:** potentially yes (different intervention decisions, latency sensitivity)
- **Behavior/semantics change:** yes
- **Concurrency/threading change:** possible (bounded async API call path required)
- **Bounded?** pending design
- **Structured?** pending design
- **Potential downstream load increase:** yes (external API traffic)
- **Diverges from user proposal:** no
- **Files touched:** none
- **Notes:** Candidate design is a gated, fail-open, low-frequency secondary classifier; not implemented in this pass due safety/latency and dependency implications.

### No Refinements (VTSC steering fallback + recorder summaries)
- **Date:** 2026-02-14
- **Classification:** A
- **Category:** Code
- **Status:** applied
- **Approval:** na
- **User-visible change:** no
- **Behavior/semantics change:** no
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** no
- **Diverges from user proposal:** no
- **Files touched:** `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`, `tools/vtsc/vtsc_intervention_recorder.py`, `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py`
- **Notes:** No separate refinement beyond the requested VTSC fixes and diagnostics.

### VTSC Map-Lookahead Inactive Reason Diagnostics
- **Date:** 2026-02-25
- **Classification:** A
- **Category:** Code
- **Status:** applied
- **Approval:** na (user requested patch)
- **User-visible change:** no
- **Behavior/semantics change:** no (telemetry-only)
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** negligible (added snapshot keys only)
- **Diverges from user proposal:** no
- **Files touched:** `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`, `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py`
- **Notes:** Added explicit map lookahead reason fields (`map_tail_reason`, `map_tail_compute_reason`) to VTSC debug snapshots to disambiguate inactive map-tail causes (`toggle_off`, `no_gps`, `no_map_curvatures`, `vision_suppressed`, etc.).

### VTSC Intervention Recorder: Fix vEgo/vCruise Source (Deprecated ControlsState)
- **Date:** 2026-02-26
- **Classification:** C
- **Category:** Code
- **Status:** applied
- **Approval:** na (bugfix for requested offline RCA capture)
- **User-visible change:** yes (event bundles can now trigger when the toggle is enabled)
- **Behavior/semantics change:** yes (source arbitration now uses correct speed values)
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** bounded (more events may be recorded, but guarded by toggle + size cap)
- **Diverges from user proposal:** no
- **Files touched:** `tools/vtsc/vtsc_intervention_recorder.py`
- **Notes:** `controlsState.*DEPRECATED` speed fields are 0 in this branch; recorder now uses `carState.vEgo/aEgo` and `carControl.hudControl.setSpeed` (m/s) so `vtscLimiting` arbitration and triggers work as intended.

### Mapd: Include Bearing In LastGPSPosition For Way Matching
- **Date:** 2026-02-26
- **Classification:** C
- **Category:** Code
- **Status:** applied
- **Approval:** implicit (user requested MTSC/map lookahead investigation and suspected mapd path issue)
- **User-visible change:** potentially yes (may enable MapCurvatures population and MTSC/map-tail lookahead once deployed)
- **Behavior/semantics change:** yes (extends LastGPSPosition JSON payload)
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** no (same write rate, small extra field)
- **Diverges from user proposal:** small (not a directory change; fixes a likely input mismatch for mapd)
- **Files touched:** `sunnypilot/mapd/live_map_data/osm_map_data.py`
- **Notes:** `openpilot-mapd` uses `bearing` to disambiguate forward direction on one-way roads. Without this field, map matching can fail and keep `MapCurvatures` empty (`[]`), disabling VTSC map-tail lookahead.

### VTSC + Mapd Stabilization For Commit Gate (2026-02-26)
- **Classification:** A
- **Category:** Code
- **Status:** applied
- **Approval:** na (internal equivalent refactors/bugfixes to satisfy requested test/lint gate)
- **User-visible change:** no direct UI/workflow changes
- **Behavior/semantics change:** yes (fixes two VTSC regression paths and hardens mapd geometry fallback)
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** no
- **Diverges from user proposal:** no
- **Files touched:** `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`, `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py`, `sunnypilot/mapd/live_map_data/osm_map_data.py`, `sunnypilot/mapd/tests/test_integration.py`, `tools/vtsc/vtsc_intervention_recorder.py`
- **Notes:** Restored missing hold-threshold constant used by occluded hold gating, added map-tail reason diagnostics/tests, fixed mapd integration test patch targets/param fixtures, and hardened `update_location()` against geometry-update exceptions while preserving legacy behavior.

### Paramsd Roll-Confidence A/B Validation Harness
- **Date:** 2026-02-27
- **Classification:** A
- **Category:** Reliability
- **Status:** applied
- **Approval:** na (validation-only work requested by user)
- **User-visible change:** no
- **Behavior/semantics change:** no
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** no
- **Diverges from user proposal:** no
- **Files touched:** `.agents/verification/VERIFICATION.md`, `.agents/refinement/REFINEMENTS.md`
- **Notes:** Added deterministic A/B proof harness and command evidence to validate root cause and non-regression claims for `paramsd` roll-confidence gating without introducing runtime code changes.

### VTSC Rally Co-Pilot Strip-Map Refinement
- **Date:** 2026-03-05
- **Classification:** C
- **Category:** UI + Telemetry
- **Status:** applied
- **Approval:** user-requested in-thread (explicit strip-map rendering, realism, smoothness, and nav-arrow changes)
- **User-visible change:** yes (fixed 10-second horizon, smoother road geometry, nav arrow replacing red dot, more stable 3D road preview)
- **Behavior/semantics change:** yes (HUD preview now uses ego-pose-aligned, resampled map geometry instead of sparse segment-to-segment decimation and speed-dependent zoom)
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** negligible (slightly denser preview preprocessing inside existing VTSC update loop)
- **Diverges from user proposal:** no
- **Files touched:** `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`, `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`, `selfdrive/ui/sunnypilot/qt/onroad/hud.h`, `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_map_curve_preview.py`, `.agents/refinement/REFINEMENTS.md`
- **Notes:** Switched the HUD preview to a fixed 10-second strip-map, aligned map geometry to the live ego pose/bearing, added densify + smoothing + resampling so coarse OSM segments render as continuous bends, and replaced the red ego marker with a classic navigation arrow.
