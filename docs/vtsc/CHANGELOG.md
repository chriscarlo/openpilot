# Changelog — `docs/vtsc/`

This changelog tracks **documentation/inventory changes** made inside `docs/vtsc/`.

## 2026-01-26
- Added `docs/vtsc/OCCLUSION.md` (occlusion logic deep dive + lead-bypass expectations).
- Updated `docs/vtsc/README.md`, `docs/vtsc/AGENTS.md`, and `docs/vtsc/INVENTORY.md` to reference the occlusion doc.
- Updated `docs/vtsc/TESTS.md` and `docs/vtsc/TESTING_STRATEGY.md` to reflect current `xfail` status (none remaining in the co-located VTSC suites).
- Expanded occlusion docs to cover severe-confidence (non-FOV) no-raise behavior and its interaction with lead-bypass.
- Updated `docs/vtsc/TESTS.md` + `docs/vtsc/INVENTORY.md` to include the new VTSC sweep suite (`test_sweep_matrix.py`) and trace helper (`simulate_sequence_trace`).
- Updated `docs/vtsc/OCCLUSION.md` + `docs/vtsc/TESTING_STRATEGY.md` to document recent behavior refinements and key non-VTSC confounders (notably `allow_throttle`).
- Simplified FOV-occlusion clear behavior to be geometry-based (reduces “stuck occluded” recovery failures on real roads with mediocre confidence).
- Added a regression to distinguish VTSC issues from long MPC internal cruise clipping (`v_cruise_clipped`) when VTSC requests large cap step-downs.
- Clarified `docs/vtsc/TESTS.md` end-to-end suite description to match current recovery behavior and confounder coverage.
- Updated map lookahead behavior: when vision is SEVERE/LOST, map tail capping now considers near-horizon curvature so short off-ramp curves aren’t ignored (adds a synthetic regression seeded by 38°43'54.0"N 120°47'20.2"W).
- Added a controller-level regression for “model-flat + severe confidence” fail-open using a steering-curvature fallback (`test_severe_confidence_model_flat_steering_fallback_slows_for_sharp_curve`).

## 2026-01-24
- Created `docs/vtsc/` baseline VTSC index folder.
- Added initial inventory of VTSC code, Params, UI, tools, and tests.
- Added agent maintenance instructions and a stable docs schema.
- Updated test index to include the new VTSC pipeline and planner-ingestion tests under `sunnypilot/selfdrive/controls/lib/tests/vtsc/`.
- Added `docs/vtsc/TESTING_STRATEGY.md` to describe an end-to-end VTSC test layering approach and pipeline deviation checks.
- Added `docs/vtsc/RLOGS.md` and clarified that VTSC rlog regressions require **tici / comma3x**-sourced data (even if copied to a dev laptop).
