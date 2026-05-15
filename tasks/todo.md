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

- [x] Map current `radard.py`, LongMPC, live-tune, and noise-harness behavior.
- [x] Add source-side model-lead tracking/filtering before no-radar `radarState` publication.
- [x] Preserve stable negative synthetic model lead ids across `leadsV3` slot churn.
- [x] Keep LongMPC virtual-lead filtering from resetting when the same synthetic model track changes slot.
- [x] Add live-tunable critical model-lead filter/association knobs and document them.
- [x] Extend tests and the simulation harness for steady noise, duplicate hypotheses, opening jumps, and closer safety events.
- [x] Tune defaults with harness sweeps until steady-follow noise improves while closer hazards still adopt quickly.
- [x] Run focused regression tests and review the final diff for intended source/test/docs only.

## Review

- Added `ModelLeadTracker` in `selfdrive/controls/radard.py` for model-only no-radar leads. Real radar `Track` behavior is unchanged.
- The tracker emits stable negative synthetic `radarTrackId` values and filters dRel before `radarState.leadOne/leadTwo` reaches planner/MPC.
- LongMPC now preserves the Hyundai virtual-lead filter state when `lead0`/`lead1` changes but the synthetic model track id is the same.
- Added live-tunable `Longitudinal.LiveTune.ModelLeadFilter*` params and documented them in `docs/chauffeur/live_tunable_params.md`.
- Updated `.codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py` to report raw source, radard-tracked, and LongMPC-filtered dRel; `--disable-model-lead-tracker` compares the old raw path.
- Single-lead baseline, old path: raw p95 `10.93 m`, radard 3 s rolling p95 `21.19 m`, LongMPC-filtered 3 s rolling p95 `10.25 m`.
- Single-lead baseline, tracker enabled: raw p95 `10.93 m`, radard 3 s rolling p95 `2.95 m`, LongMPC-filtered 3 s rolling p95 `2.94 m`.
- Duplicate baseline, old path: `53` `source_switch` resets in 30 s, LongMPC-filtered 3 s rolling p95 `13.43 m`.
- Duplicate baseline, tracker enabled: only init reset, LongMPC-filtered 3 s rolling p95 `2.55 m`.
- Verification passed: `python -m pytest -q selfdrive/controls/tests/test_radard_model_lead_filter.py selfdrive/controls/tests/test_radard_path_metrics.py selfdrive/controls/tests/test_hyundai_ai_lead_stability.py selfdrive/controls/tests/test_lead_interactions.py selfdrive/controls/tests/test_longitudinal_live_tune.py selfdrive/controls/lib/tests/test_lead_role_classifier.py` -> 61 passed, 3 skipped.
- Full-MPC smoke passed: `python .codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py --duration-s 5 --seed 7 --source-noise-std-m 5 --spike-prob-per-s 0 --white-noise-std-m 0.25 --full-mpc`.

# EV6 No-Radar AI Lead Higher-Noise Tuning

- [x] Re-run steady-follow source-noise simulation at higher `std=8` and `std=10`.
- [x] Stress duplicate model hypotheses at `std=10`.
- [x] Probe accel/decel latency separately from steady noise.
- [x] Tighten duplicate-collapse behavior for same-frame model hypotheses.
- [x] Reduce model-lead vRel smoothing latency without increasing dRel admission.
- [x] Re-run focused regression tests and commit tuning pass.

## Review

- At `source-noise-std-m=8`, raw source range was `44.34 m`; tracked radard 3 s p95 range was `3.25 m`, LongMPC-filtered 3 s p95 range was `2.79 m`.
- At `source-noise-std-m=10`, raw source range was `55.43 m`; tracked radard 3 s p95 range was `3.29 m`, LongMPC-filtered 3 s p95 range was `2.89 m`.
- The first `std=10` duplicate run exposed separate synthetic track creation and 19 `source_switch` resets; same-frame duplicate collapse now keeps that run to only init reset, with LongMPC-filtered 3 s p95 range `2.48 m`.
- Decel/accel probe: lowering model-lead vRel tau from `0.55 s` to `0.40 s` reduced decel overestimate from about `2.17 m` to `1.80 m` and 1 s decel overestimate from `0.72 m` to `0.63 m`, with no meaningful dRel noise increase in the probe.
- Increasing close-side dRel slew did not improve the decel probe and increased steady-noise dRel movement, so it was left unchanged.
- Verification passed after the tuning edits: focused longitudinal/radard suite -> 62 passed, 3 skipped.
- Full-MPC smoke at `source-noise-std-m=10` passed with LongMPC-filtered 3 s p95 range `3.11 m`.

# EV6 No-Radar AI Lead Windows Pre-Merge Gate

- [x] Run broad Windows longitudinal regression suite.
- [x] Run Hyundai controller-side Windows tests.
- [x] Stress simulator at `source-noise-std-m=12`.
- [x] Fix distance-only close-dip fast-adopt edge case found by `std=12`.
- [x] Re-run final Windows regression and simulator checks.

## Review

- Broad Windows longitudinal suite passed before the final edge-case patch: 65 passed, 25 skipped.
- Hyundai controller-side suite passed: 18 passed.
- Initial `std=12` simulator stress exposed pure distance-noise close dips causing `drel_jump_closer` resets and large filtered range.
- Fast-close admission now requires low TTC, strong closing speed, or cut-in lateral support; distance-only same-speed close dips are rate-limited instead.
- After the gate change, `std=12` single-lead stress: raw range `64.27 m`, radard 3 s p95 `3.17 m`, LongMPC-filtered 3 s p95 `2.87 m`, only init reset.
- After the gate change, `std=12` duplicate stress: raw range `56.43 m`, radard 3 s p95 `2.99 m`, LongMPC-filtered 3 s p95 `2.83 m`, only init reset.
- Final broad Windows longitudinal suite passed: 66 passed, 25 skipped.
- Final Hyundai controller-side suite passed: 18 passed.
- Final `std=12` full-MPC smoke passed: LongMPC-filtered 3 s p95 range `0.68 m`, only init reset.

# EV6 No-Radar AI Lead Param Audit

- [x] Confirm Windows params fallback reads `common/params_keys.h`.
- [x] Add live params for model-lead vRel smoothing values tuned for accel/decel latency.
- [x] Leave structural tracker gates as code constants.
- [x] Remove unused close-confirm constant after distance-only fast-admit hardening.
- [x] Run focused params/radard verification and push.

## Review

- Added `Longitudinal.LiveTune.ModelLeadFilterVRelTauS` default `0.40`.
- Added `Longitudinal.LiveTune.ModelLeadFilterFastVRelTauS` default `0.16`.
- Did not expose `MODEL_LEAD_SAME_SLOT_RECOVER_DREL_GATE_M`; it is an identity-recovery guard, not a roadside behavior knob.
- Focused verification passed: live-tune default/spec checks plus radard model-lead filter tests -> 9 passed.
- `std=12` simulator smoke passed with radard 3 s p95 `3.08 m` and LongMPC-filtered 3 s p95 `2.19 m`.

# EV6 No-Radar Freeway Brake-Tap / Low-Speed Launch Follow-up

- [x] Patch lead slowdown ceiling release so close/closing lead braking does not pulse between hard and soft every frame.
- [x] Extend lead-to-cruise transition accel cap so cruise does not surge immediately after lead release.
- [x] Require a real standstill gap before lead-launch stop release can command starting accel.
- [x] Add regression tests from captured freeway and stop/go trace shapes.
- [ ] Run focused tests, commit, push, pull to tici, build/reboot.

## Review

- Lowered the lead slowdown speed gate so close/closing low-speed leads still get an accel ceiling below 2 m/s.
- Added release-rate limiting for `lead_slowdown_accel_ceiling` while a lead is still close or closing, so braking can deepen immediately but relaxes gradually.
- Extended the Hyundai lead-to-cruise accel transition from 1.0 s to 3.0 s and lowered its initial cap from 0.45 to 0.25 m/s^2.
- Required at least 5.0 m dRel before standstill lead-launch release can leave `shouldStop` and enter full starting accel.
- Focused verification passed: `python -m pytest -q selfdrive/controls/tests/test_longitudinal_planner_stop_release.py selfdrive/controls/tests/test_lead_interactions.py selfdrive/controls/tests/test_hyundai_ai_lead_stability.py` -> 71 passed, 3 skipped.
