# VTSC Tests — What Exists Today

This doc is an index of existing VTSC-related tests/harnesses and how to run them.

For deeper testbench docs, also see `docs/chauffeur/vtsc/tests_overview.md`.

## Fast, Co-Located Scenario Tests (Recommended Starting Point)

Location:
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/`

Files:
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/harness.py`
  - Minimal SubMaster-like stubs and a deterministic time-stepped simulator for `VisionTurnController`.
  - Includes `simulate_sequence_trace(...)` for per-step timing assertions (cap release timing, bypass thresholds, etc.).
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py`
  - Scenario-style tests (straight/no-crawl, occlusion hysteresis, lead bypass, map lookahead cap, etc.).
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_sweep_matrix.py`
  - Small parameterized sweeps that stress key invariants across multiple values (speed/curvature/headway).
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_pipeline_integration.py`
  - Planner-ingestion focused tests:
    - verifies VTSC’s `v_turn` is selected by `update_v_cruise` min-of logic,
    - verifies `longitudinalPlanSP.visionTurnSpeedControl.velocity` matches `v_tsc.v_turn`,
    - includes recovery regressions (cap must return to cruise after curve/occlusion ends).
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_flow.py`
  - End-to-end-ish tests using a **fake MPC module** (no Acados build required):
    - verifies the **exact** value passed to MPC equals min(cruise sources),
    - verifies recovery to cruise *and* that actuator-level accel can go positive after VTSC releases,
    - verifies VTSC does not remain “stuck occluded” after a curve ends (even with mediocre confidence, with or without a lead),
    - includes “confounder” tests showing how `allow_throttle` and MPC cruise-envelope clipping can delay or prevent accel even after VTSC releases.
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/pipeline_harness.py`
  - Shared helpers for the pipeline tests (FakeSubMaster, synthetic model messages, optional fake MPC injection).

Run:
```bash
pytest sunnypilot/selfdrive/controls/lib/tests/vtsc -q
```

Why these matter:
- They’re closest to production code.
- They’re deterministic and don’t require device logs.
- They cover important VTSC invariants (no positive accel when occluded, dwell stability, etc.).

### About `xfail` VTSC tests
There are currently **no** VTSC tests marked `xfail` in `sunnypilot/selfdrive/controls/lib/tests/vtsc/`.

If a future change introduces a known regression that needs tracking without breaking CI, prefer:
- adding a focused `xfail` with a specific reason, and
- filing the follow-up work to remove the `xfail` once fixed.

## Larger Testbenches (Docs Area)

Location:
- `docs/chauffeur/vtsc/testing/`

What’s in there:
- Unit/integration/acceptance/e2e-style simulations that exercise more of the controller surface area.
- Harness utilities: `docs/chauffeur/vtsc/testing/harness/`

Run:
```bash
pytest docs/chauffeur/vtsc/testing -q
```

Notes:
- These are often “research-grade” tests; some may be slower or assume extra fixtures.
- Prefer keeping new, stable regressions co-located (next to the controller), and keep log-heavy tests in docs.

## Rlog-Based Regressions / Full Trace

Location:
- `docs/chauffeur/vtsc/fullTrace/`

Includes:
- Replay tooling and trace schema.
- Regression tests under `docs/chauffeur/vtsc/fullTrace/tests/` (may rely on recorded logs being present).

Data provenance requirement:
- rlogs for VTSC regressions should originate from a real drive on **TICI / comma3x** (even if the files are later copied to a dev laptop).
- This repo includes a small set of committed, tici-sourced rlogs as fixtures (see `docs/vtsc/RLOGS.md`).

Run (opt-in regressions):
```bash
pytest -q -m regression docs/chauffeur/vtsc/fullTrace/tests/test_regressions_rlogs.py
```

## Minimal Import/Instantiation Smoke Script

Location:
- `test_vtsc_minimal.py`

Purpose:
- Quick import/instantiate checks for `VisionTurnController`, plus a couple physics monotonicity assertions.

## Where To Extend Next (Suggested Structure)

If you are adding more comprehensive VTSC behavior tests:
- Keep **fast, deterministic, “invariant” tests** in:
  - `sunnypilot/selfdrive/controls/lib/tests/vtsc/`
- Keep **log replay tests**, trace generation, and large scenario suites in:
  - `docs/chauffeur/vtsc/`

When adding new tests, record:
- the scenario name and what invariant it protects,
- any Params assumed/forced,
- any dependencies on model message shapes (orientationRate.z, laneLineProbs, etc.),
and update `docs/vtsc/INVENTORY.md` + `docs/vtsc/CHANGELOG.md`.
