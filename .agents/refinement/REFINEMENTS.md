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
