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

### VTSC Intervention Recorder Cruise Source + Engagement Window Hardening
- **Date:** 2026-02-16
- **Classification:** C
- **Category:** Data capture / tooling
- **Status:** applied
- **Approval:** user-requested in-thread (autonomous bug hunt and closure)
- **User-visible change:** yes (intervention recorder can now capture scenarios previously missed)
- **Behavior/semantics change:** yes
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** low (same daemon cadence, broader eligibility)
- **Diverges from user proposal:** no
- **Files touched:** `tools/vtsc/vtsc_intervention_recorder.py`, `tools/vtsc/tests/test_vtsc_intervention_recorder.py`
- **Notes:** Root cause was cruise setpoint sourced from `controlsState.vCruiseDEPRECATED` which is zero on current stack; switched to robust source fallback with `carState.vCruise` primary and excluded invalid zero cruise from limiter arbitration. Also made engagement lookback configurable and increased default to 3.0s to tolerate timing skew between control state and pedal override events.

### MTSC Health Telemetry In VTSCDBG + MTSCHEALTH Stream
- **Date:** 2026-02-16
- **Classification:** A
- **Category:** Observability
- **Status:** applied
- **Approval:** user-requested in-thread (`add the health log`)
- **User-visible change:** no (developer telemetry only)
- **Behavior/semantics change:** no longitudinal decision-policy change intended
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** minimal (debug-only log emission at existing VTSC debug cadence)
- **Diverges from user proposal:** no
- **Files touched:** `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`, `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_mtsc_health_logging.py`
- **Notes:** Added MTSC health status derivation (`disabled`, `missing_gps_bearing`, `no_curvature_while_moving`, `no_curvature`, `ok`) and embedded fields in `VTSCDBG` snapshots (`mtsc_*` keys). Added dedicated `MTSCHEALTH` debug line with minimal key indicators for live triage and grepability.

### MTSC Root-Cause Refinement: Bearing In LastGPSPosition For mapd Matching
- **Date:** 2026-02-16
- **Classification:** C
- **Category:** Runtime behavior / map integration
- **Status:** applied
- **Approval:** user-requested in-thread (MTSC root-cause and fix)
- **User-visible change:** yes (MTSC map curvature should populate reliably on device)
- **Behavior/semantics change:** yes (mapd matching now has heading signal)
- **Concurrency/threading change:** no
- **Bounded?** na
- **Structured?** na
- **Potential downstream load increase:** no material increase
- **Diverges from user proposal:** no
- **Files touched:** `sunnypilot/mapd/live_map_data/base_map_data.py`, `sunnypilot/mapd/live_map_data/osm_map_data.py`, `sunnypilot/mapd/tests/test_base_map_data_bearing.py`
- **Notes:** mapd consumes `LastGPSPosition.bearing`; previous bridge omitted it, causing `MapCurvatures=[]` despite valid offline map tiles. Added robust bearing extraction/fallback and propagated bearing into memparam payload.
