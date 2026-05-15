# Handoff Prompt: EV6 No-Radar AI Lead `dRel` Noise Remediation

You are taking over an openpilot/Sunnypilot/Chauffeur longitudinal behavior task in this worktree:

`C:\Users\crimoldi\Documents\codex\chauffeur`

Use `$openpilot-longitudinal-tuner` first. The goal is to build, test, tune, and validate a robust fix for no-radar Hyundai/Kia EV6 AI/model lead distance noise that currently survives into longitudinal planning/control and feels like light brake taps, throttle taps, or a nauseating mix of both while following a lead.

Do not stop at architecture. Implement the fix in this worktree, add or modify comprehensive tests and simulation coverage, run the relevant checks, tune the new behavior until it is smoother in the worktree, and leave the branch ready to merge back into `chauffeur-dev4` once validated.

## Current User Goal

The user believes the raw AI/model-generated lead distance estimate is noisy on their EV6 because the car radar cannot be used without allowing HKG longitudinal control. They need that noise tamped down before it creates jumpy following behavior.

The fix must:

- Reduce post-filter lead-distance movement that reaches planning/control during steady following.
- Preserve fast response to real closer leads, cut-ins, low-TTC events, and stopped/slow leads.
- Avoid masking safety-critical closing events.
- Avoid solving this only at the final car-controller layer.
- Improve reliability for no-radar Hyundai/Kia EV6, while preserving real-radar behavior.
- Make critical knobs live-tunable where feasible, and document them.
- Include tests and a simulation harness that prove the new behavior and guard against regressions.

## Critical Branch Facts

Use current branch code as source of truth. Older notes may be stale.

The current no-radar lead path is:

1. `modeld` publishes `modelV2.leadsV3[*]`.
2. `radard.get_lead(...)` converts model lead `x[0]` to `radarState.leadOne.dRel`.
3. With no real radar tracks, `radard.get_RadarState_from_vision(...)` is used.
4. `LongitudinalMpc.update(...)` classifies `radarState.leadOne/leadTwo`.
5. Hyundai-only virtual lead filtering/source stability happens inside `LongitudinalMpc`.
6. MPC output feeds planner and later Hyundai longitudinal controller shaping.

Important files:

- `selfdrive/controls/radard.py`
  - Existing real-radar `Track` class.
  - Existing `get_RadarState_from_vision(...)`.
  - Existing `get_lead(...)`.
  - Current no-radar path bypasses `Track` because `tracks={}`.
- `common/simple_kalman.py`
  - Existing `KF1D`.
  - This is currently used by real-radar `Track` to filter `vLeadK/aLeadK`, not `dRel`.
- `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py`
  - Existing `LeadDistanceFilter`.
  - Existing Hyundai virtual lead state and source stability.
  - Existing `LEADROLEDBG` debug payload.
- `selfdrive/controls/lib/lead_role_classifier.py`
  - Lead role classification and duplicate handling before MPC source choice.
- `selfdrive/controls/lib/longitudinal_live_tune.py`
  - Live-tunable longitudinal lead-response metadata.
- `common/params_keys.h`
  - Register any new live-tune params here.
- `docs/chauffeur/live_tunable_params.md`
  - Document new knobs here.
- `.codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py`
  - Existing live-tune CLI.
- `.codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py`
  - New baseline harness added in this worktree.
- `.codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py`
  - Live watcher for device testing.

## Historical Context

The user asked whether the "Kalman" built earlier is filtering the no-radar path. Be precise: the code currently has a prediction-corrector EMA-style filter, not a formal Kalman filter for no-radar `dRel`.

Relevant commits:

- `fc371bdc1` - 2026-03-27 - `Stabilize Hyundai AI lead ownership`
  - Added Hyundai virtual lead and the old simple asymmetric EMA:
    `filtered.dRel = self._filter_metric(prev.dRel, raw_lead.dRel, dt_s, danger_if_lower=True)`.
- `dacdf89a6` - 2026-03-30 - `Add prediction-corrector dRel filter for AI-model lead noise`
  - Replaced the simple `dRel` EMA in the Hyundai virtual lead path with `LeadDistanceFilter`.
  - This is a prediction-corrector EMA with gates, not a formal Kalman filter.
- `514f6d317` - 2026-03-30 - `Make dRel filter params live-tunable, add tunable params reference`
  - Made several `DRelFilter*` values live-tunable.
- `3eb4ab654` - 2026-03-31 - `longitudinal: improve EV6 lead acquisition and smoothing`
  - Added deadband, opening slew clamp, better debug fields, split reset reasons, and related EV6 lead behavior improvements.
- `4f6617a0c` - 2026-03-30 - `Smooth lead chevron display with EMA filter on dRel/yRel/vRel`
  - UI-only smoothing. Does not affect control.

## What Exists Today

### Real-radar `Track`

`selfdrive/controls/radard.py` has:

- `KalmanParams`
- `Track`
- `KF1D`

But `Track.update(...)` copies `dRel` directly:

```python
self.dRel = d_rel
```

The existing `KF1D` in `Track` filters `vLeadK` and `aLeadK`, not `dRel`.

Do not simply route model-only leads through the current `Track` class and assume that solves `dRel` noise. It does not.

### Model-only no-radar path

`get_RadarState_from_vision(...)` currently does:

```python
"dRel": float(lead_msg.x[0] - RADAR_TO_CAMERA)
```

That raw model lead distance then becomes `radarState.leadOne.dRel`.

### Hyundai virtual lead filter

`LongitudinalMpc` has `LeadDistanceFilter`, which helps, but it is late in the pipeline. It can still allow or amplify occupant-noticeable movement because:

- Closer `dRel` movement is admitted quickly for safety.
- Some closer jumps reset/snap.
- `lead0`/`lead1` source switches reset virtual-lead state.
- Duplicate model-lead hypotheses can churn selected source.
- Several ownership/safety decisions still look at raw lead metrics.
- Final controller shaping cannot distinguish real lead movement from source noise.

## Baseline Work Already Done

A new helper script was added:

` .codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py`

Purpose:

- Inject source-side model lead `dRel` noise into `modelV2.leadsV3`-style input.
- Convert it through `radard.get_lead(...)` with no radar tracks.
- Feed the actual Hyundai virtual-lead/source-stability path.
- Measure raw and post-filter `dRel` movement.
- Optionally run the slower full-MPC solver path with `--full-mpc`.

Useful baseline commands:

```powershell
python .codex\skills\openpilot-longitudinal-tuner\scripts\simulate_ai_lead_noise.py --duration-s 60 --seed 7 --source-noise-std-m 5 --spike-prob-per-s 0 --white-noise-std-m 0.25 --json-out .cache\ai_lead_noise_baseline_std5_seed7.json --csv-out .cache\ai_lead_noise_baseline_std5_seed7.csv --print-samples --sample-period-s 5
```

Observed baseline:

- Raw p95 absolute error: `10.93 m`.
- Post-filter p95 absolute error: `11.83 m`.
- Post-filter 3 s rolling p95 range: `10.25 m`.
- Post-filter 3 s rolling max range: `12.18 m`.
- One init snap.
- No repeated `dRel` jump resets after init.

Duplicate-lead variant:

```powershell
python .codex\skills\openpilot-longitudinal-tuner\scripts\simulate_ai_lead_noise.py --duration-s 30 --seed 7 --source-noise-std-m 5 --spike-prob-per-s 0 --white-noise-std-m 0.25 --duplicate --json-out .cache\ai_lead_noise_baseline_std5_duplicate_seed7.json --csv-out .cache\ai_lead_noise_baseline_std5_duplicate_seed7.csv --print-samples --sample-period-s 5
```

Observed duplicate variant:

- `source_switch` resets: `53` in 30 s.
- Post-filter 3 s rolling max range: `17.85 m`.

Interpretation:

- Plain source `dRel` noise is bad enough.
- Duplicate model hypotheses / slot churn can make it worse.

Existing artifacts under `.cache/` are raw/supporting artifacts and should stay untracked.

## Recommended Architecture

Implement a hybrid fix:

1. Add a model-only lead tracker/filter in `radard.py` before model-only leads are published as `radarState`.
2. Keep the existing Hyundai virtual-lead filter in `LongitudinalMpc` as a second-stage guard.
3. Preserve model-lead identity across `lead0`/`lead1` slot churn so source switches do not reset filtering when the physical lead is the same.
4. Add tests and harness coverage that exercise the full no-radar source path.
5. Make important thresholds/noise parameters live-tunable where feasible.

### Do Not Reuse Existing `Track` Directly

The existing real-radar `Track` is not the correct abstraction for model-only leads:

- It assumes radar/liveTracks identity.
- It does not filter `dRel`.
- Its `KF1D` setup observes lead speed, not distance.
- It should stay untouched for real-radar cars unless tests prove a local change is safe.

### Build a New Model-Lead Tracker in `radard.py`

Create a new model-only tracker, likely named something like:

- `VisionLeadTrack`
- `VisionLeadTracker`
- `ModelLeadTrack`
- `ModelLeadTracker`

This tracker should run only when:

- `track is None`
- `ready is True`
- `lead_msg.prob > .5`
- the lead is coming from model/vision, not radar

Suggested state:

- Filtered `dRel`.
- Filtered/estimated `vRel` or `vLead`.
- Estimated `aLeadK`.
- Filtered `yRel` / `dPath` enough for stable identity and cut-in logic.
- `modelProb`.
- Track age.
- Miss count.
- Stable synthetic model track id.
- Last raw slot / model hypothesis slot.
- Debug fields for innovation, gating, snap, source association, and confidence.

Suggested measurement inputs:

- `lead_msg.x[0] - RADAR_TO_CAMERA`
- `lead_msg.v[0] - model_v_ego`
- `lead_msg.a[0]`
- `lead_msg.y[0]`
- `lead_msg.xStd[0]`, `vStd[0]`, `yStd[0]` when available
- `lead_msg.prob`
- path-relative metrics from `add_path_relative_lead_metrics(...)`

### Filter Choice

Consider these routes before finalizing:

1. **Alpha-beta tracker**
   - State: position and velocity.
   - Simpler than Kalman.
   - Good first step if tuned with innovation gates and measurement noise weighting.

2. **Small Kalman for model leads**
   - State: distance and relative velocity, optionally acceleration.
   - Measurement: model distance and model relative velocity.
   - Measurement noise can use `xStd/vStd` plus probability scaling.
   - Better long-term fit if you can keep it readable and tested.

3. **Current `LeadDistanceFilter` moved into `radard`**
   - Faster to implement.
   - Does not handle identity/source as well by itself.
   - Might be acceptable as part of the tracker but not as the whole fix.

Recommended: build a small model-lead tracker in `radard.py`. It can be Kalman-like if you keep it bounded and testable, but the important properties are source identity, measurement-noise gating, and asymmetric safety behavior.

### Safety Gating

The tracker must not suppress real hazards. It should classify raw innovations before deciding how much to trust them.

Closer raw measurement:

- Fast-adopt when:
  - TTC is low.
  - Raw gap is at or below safe headway.
  - Lead is centered or cutting in.
  - Closing speed supports the closer distance.
  - The closer measurement persists across frames.
- Slow-admit or require confirmation when:
  - The lead is still comfortably beyond desired headway.
  - Relative speed does not support sudden closing.
  - Path/slot/probability is jittery.
  - It is a one-frame or short-burst `dRel` dip.

Opening raw measurement:

- Slew-limit opening distance.
- Avoid instant "lead teleported away" acceleration permission.
- Preserve identity during brief opening bursts.

Source/duplicate behavior:

- Associate model leads across `leadsV3` slots using distance, velocity, y/path offset, probability, and predicted state.
- Emit stable synthetic negative `radarTrackId` values for model-only leads, e.g. `-1001`, `-1002`.
- Keep `radar=False`.
- Let downstream code distinguish "same model track changed slot" from "new physical lead".

### LongitudinalMpc Follow-Up

After adding stable synthetic model track ids in `radard`, update `LongitudinalMpc._should_reset_hyundai_virtual_lead(...)` so it does not reset on `lead0`/`lead1` source switch when both old and new leads represent the same synthetic model track.

Keep `LeadDistanceFilter` as a second-stage guard, but do not require it to fix every source-noise problem.

Consider adding live-tunable params for:

- Model lead distance filter process noise.
- Model lead distance measurement noise scale.
- Model lead velocity measurement noise scale.
- Closer innovation confirmation frames or dwell time.
- Closer safe-adopt TTC threshold.
- Opening slew max.
- Track association max `dRel` delta.
- Track association max `vRel` delta.
- Track association max path/y delta.
- Track miss hold frames.

Register feasible knobs in:

- `common/params_keys.h`
- `selfdrive/controls/lib/longitudinal_live_tune.py` or a new small radard-specific tune module if that is a better boundary.
- `.codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py` if appropriate.
- `docs/chauffeur/live_tunable_params.md`.

Prefer live-tunable knobs for parameters that the user may reasonably need to adjust after a drive. Do not expose every internal constant if it would create a tuning maze.

## Test and Simulation Requirements

Create or modify tests so this cannot silently regress.

Recommended new tests:

1. `selfdrive/controls/tests/test_radard_model_lead_filter.py`
   - Model-only lead `dRel` noise is filtered before publishing `radarState`.
   - Real closer cut-in / low TTC is adopted quickly.
   - Opening jumps are slew-limited.
   - Track identity persists across slot reorder / slot jitter.
   - Synthetic `radarTrackId` is stable and negative for model-only tracks.
   - Real radar `Track` behavior is not changed.

2. Extend `selfdrive/controls/tests/test_hyundai_ai_lead_stability.py`
   - Same synthetic model track switching from lead0 to lead1 does not reset virtual lead.
   - Duplicate model hypotheses collapse without repeated `source_switch`.
   - Post-filter `dRel` movement stays below a defined range under representative source noise.
   - Real closer jump still snaps/adopts when safety conditions are met.

3. Extend `.codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py`
   - Add an option to run through the new radard model-lead tracker.
   - Emit before/after summary for old raw model path versus new tracked path.
   - Include source-id/source-switch counts.
   - Include raw/model-tracked/LongMPC-filtered `dRel` ranges.

4. Add a small full-MPC smoke test, but keep it short on Windows.
   - Use `--full-mpc` for a few seconds to prove solver path still works.
   - Keep long sweeps in fast mode.

Useful existing tests:

```powershell
python -m pytest -q selfdrive/controls/tests/test_radard_path_metrics.py
python -m pytest -q selfdrive/controls/tests/test_hyundai_ai_lead_stability.py
python -m pytest -q selfdrive/controls/tests/test_lead_interactions.py
python -m pytest -q selfdrive/controls/tests/test_longitudinal_live_tune.py
python -m pytest -q selfdrive/controls/lib/tests/test_lead_role_classifier.py
```

Windows GUI test context:

- Some full acados-backed plant tests are skipped on Windows after recent local changes.
- The Windows fallback work added files such as `common/params_pyx.py`, `msgq_repo/msgq/ipc_pyx.py`, `selfdrive/controls/lib/longitudinal_mpc_lib/windows_acados_stub.py`, and `selfdrive/controls/tests/test_longitudinal_windows_compat.py`.
- Keep Linux/WSL tests intact.

## Metrics to Hit

Tune until representative no-radar steady-follow noise improves materially.

Suggested targets:

- Steady follow, single model lead, source noise similar to current harness:
  - Post-radard tracked `dRel` 3 s rolling p95 range: target around `2-4 m`, lower if safety response remains good.
  - Post-LongMPC filtered `dRel` 3 s rolling p95 range: target around `1-3 m`.
  - No repeated snaps/resets after init.
- Duplicate model hypotheses:
  - Near-zero `source_switch` resets for the same physical lead.
  - Stable synthetic model track id.
  - Post-filter 3 s rolling max should not blow past the single-lead case by large multiples.
- Safety:
  - Real closer cut-in and low TTC events must adopt quickly.
  - Slow or stopped close lead still produces timely braking behavior.
  - No delayed response that would fail existing FCW or lead-interaction tests.

## Live / Device Validation Plan

After worktree tests pass, use a bounded live capture on the tici.

Existing skill command:

```bash
cd /data/openpilot
/usr/local/venv/bin/python3 .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --duration 10 --only-alerts --show-live-tune
```

If adding new debug fields, update the monitor script to show:

- raw model `dRel`
- radard tracked model `dRel`
- LongMPC virtual lead filtered `dRel`
- synthetic model track id
- source switch count
- snap/reset reason
- planner accel
- carControl accel
- carOutput accel
- carState `aEgo`

Keep live captures bounded and put raw artifacts under `.cache/`.

## Worktree Cautions

The current worktree may already contain unrelated or prior-task changes. Do not revert user changes or unrelated local work.

Known current context from the previous session:

- There are many `D .../claude.md` status entries from Windows/case/symlink weirdness. Treat them as existing noise unless the user explicitly asks to address instruction files.
- Symlink repair was part of the Windows GUI migration. Do not dismiss symlink status as irrelevant.
- `.cache/` contains generated noise-baseline artifacts and is gitignored.
- There are untracked files from Windows test-environment work.

Run `git status --short` before editing and keep your final diff focused on the source/test/docs required for this task.

## Suggested Implementation Sequence

1. Re-read the current production path:
   - `selfdrive/controls/radard.py`
   - `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py`
   - `selfdrive/controls/lib/lead_role_classifier.py`
   - `.codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py`
2. Add model-lead tracker state in `radard.py` without touching real-radar `Track` behavior.
3. Add unit tests around the tracker itself.
4. Wire tracker output into `get_lead(...)` / `RadarD.update(...)` for model-only no-radar leads.
5. Add stable synthetic model `radarTrackId`.
6. Update LongMPC reset logic to preserve virtual-lead state across same-model-track slot switches.
7. Add live-tunable knobs for the parameters that matter most.
8. Extend the noise harness to compare old and new behavior.
9. Run targeted tests and harness sweeps.
10. Tune defaults until steady-follow noise is reduced and safety cases still pass.
11. Update docs:
    - `docs/chauffeur/live_tunable_params.md`
    - `$openpilot-longitudinal-tuner` skill notes if workflow changes.
    - This handoff doc if important decisions change during implementation.

## Definition of Done

Do not call the work complete until:

- Source-side model-only no-radar `dRel` is filtered/tracked before `radarState` publication.
- Same physical model lead keeps identity across slot churn.
- LongMPC does not reset virtual lead for same synthetic model track id.
- Representative source-noise simulation shows a substantial reduction versus baseline.
- Real closer lead / cut-in / low TTC cases still adopt quickly.
- Focused tests pass.
- Live-tunable knobs are registered, script-accessible, and documented where feasible.
- The final diff contains intended source/test/docs only.
