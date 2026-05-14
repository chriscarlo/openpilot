# Longitudinal Windows Test Usability

- [x] Repair tracked symlinks for the Windows GUI checkout.
- [x] Add Windows-only fallbacks for native params and msgq imports.
- [x] Keep full acados-backed longitudinal simulations on the Linux path.
- [x] Run focused Windows-native longitudinal tests.
- [x] Review git diff for intended source/test/task changes.

## Review

- `python -m pytest -q selfdrive/controls/tests/test_longitudinal_windows_compat.py selfdrive/controls/tests/test_longitudinal_live_tune.py selfdrive/controls/lib/tests/test_lead_role_classifier.py selfdrive/controls/tests/test_lead_interactions.py selfdrive/controls/tests/test_hyundai_ai_lead_stability.py selfdrive/controls/tests/test_following_distance.py selfdrive/test/longitudinal_maneuvers/test_longitudinal.py` passed: 54 passed, 25 skipped.
- `python -m pytest -q opendbc_repo/opendbc/sunnypilot/car/hyundai/tests/test_tuning_controller.py opendbc_repo/opendbc/sunnypilot/car/hyundai/tests/test_lead_data_car_controller.py sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_response_model.py` passed: 18 passed.
- The skipped cases are full acados-backed longitudinal simulations on Windows; the Linux CLI path remains available for those tests.

# AI Lead dRel Noise Baseline

- [x] Check existing plant, maneuver, radard, and Hyundai lead-stability tests before adding a new harness.
- [x] Add a reusable source-side model-lead dRel noise harness.
- [x] Run the harness until post-filter `dRel` movement reaches the 1-15 m baseline band.
- [x] Run focused regression coverage for the reused longitudinal path.
- [x] Record baseline result and harness limitations.

## Review

- Existing plant/maneuver simulation was not reused as the baseline source-noise harness because `plant.py` publishes `radarState` directly and still has `TODO use real radard logic for this`; that bypasses `modelV2.leadsV3 -> radard.get_lead -> radarState.leadOne`.
- The new helper `.codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py` injects noise into model lead `x`, converts it through `radard.get_lead()` with no radar tracks, then feeds the actual Hyundai virtual-lead/source-stability path in `LongitudinalMpc`.
- EV6/no-radar model leads do not use radard `Track` Kalman filtering in this path; radar `Track` Kalman filtering only applies when real radar points exist. The measured post-source filter is the Hyundai virtual-lead `dRel` filter.
- Baseline run: `python .codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py --duration-s 60 --seed 7 --source-noise-std-m 5 --spike-prob-per-s 0 --white-noise-std-m 0.25 --json-out .cache/ai_lead_noise_baseline_std5_seed7.json --csv-out .cache/ai_lead_noise_baseline_std5_seed7.csv --print-samples --sample-period-s 5`.
- Baseline result: raw p95 absolute error `10.93 m`, post-filter p95 absolute error `11.83 m`, post-filter 3 s rolling p95 range `10.25 m`, post-filter 3 s rolling max range `12.18 m`, one init snap, and no dRel jump reset after init.
- Duplicate-lead variant with the same source noise produced 53 `source_switch` resets in 30 s and post-filter 3 s rolling max range `17.85 m`, so duplicate source churn is a plausible amplifier beyond plain dRel noise.
- Focused verification passed: `python -m pytest -q selfdrive/controls/tests/test_radard_path_metrics.py selfdrive/controls/tests/test_hyundai_ai_lead_stability.py` -> 29 passed.

# EV6 No-Radar AI Lead dRel Remediation

- [ ] Map current `radard.py`, LongMPC, live-tune, and noise-harness behavior.
- [ ] Add source-side model-lead tracking/filtering before no-radar `radarState` publication.
- [ ] Preserve stable negative synthetic model lead ids across `leadsV3` slot churn.
- [ ] Keep LongMPC virtual-lead filtering from resetting when the same synthetic model track changes slot.
- [ ] Add live-tunable critical model-lead filter/association knobs and document them.
- [ ] Extend tests and the simulation harness for steady noise, duplicate hypotheses, opening jumps, and closer safety events.
- [ ] Tune defaults with harness sweeps until steady-follow noise improves while closer hazards still adopt quickly.
- [ ] Run focused regression tests and review the final diff for intended source/test/docs only.
