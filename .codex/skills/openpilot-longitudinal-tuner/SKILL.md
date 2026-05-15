---
name: openpilot-longitudinal-tuner
description: >
  Research, debug, tune, and monitor openpilot longitudinal behavior across the
  planner/MPC path, model-driven experimental path, brand-specific carcontroller
  actuation, and sunnypilot overlays. Use when investigating acceleration or
  braking anomalies, stop-and-go behavior, lead following, speed-limit or
  curve-based slowdowns, Experimental Mode / end-to-end longitudinal behavior,
  Hyundai CAN FD or Kia EV6 longitudinal issues, or when you need a live
  longitudinal anomaly monitor with code-verified file paths and tests.
---

# Openpilot Longitudinal Tuner

## Guardrails

- Treat branch code as source of truth. Some older docs and tests still describe
  deprecated MTSC publishing paths or stale EV6 speed-limit flags.
- Split the problem by layer before tuning anything:
  planner target generation, `LongControl`, brand `CarController`, and actual
  vehicle response are different failure surfaces.
- **The watcher only produces data on the tici.** There is no cereal bridge
  on the dev box — `.venv/bin/python .../monitor_longitudinal_anomalies.py`
  will sit silent forever and look like "no messages." Always run the live
  monitor on-device via `ssh commaCar ...`. Dev-box invocation is for static
  analysis/help only.
- **Always pass `python3 -u` over SSH.** Without `-u`, stdout is block-buffered
  because SSH without a TTY is a pipe, and the output file stays empty even
  while the watcher is running fine. This silent-fail costs entire driving
  windows — make `-u` non-optional.
- On the dev machine, `.venv/bin/python` is only correct for scripts that do
  not need live cereal data (static imports, help, parsing). Repo-root
  `python3` can miss `capnp`. On tici, use `/usr/local/venv/bin/python3`.
- While the user is actively driving, prefer bounded captures with
  `--duration ...` or short one-off probes. Open-ended SSH tails are easy to
  leave hanging and are rarely the fastest way to isolate a longitudinal bug.
- On Hyundai CAN FD, do not assume `carControl.actuators.accel` is the final
  car command. Sunnypilot's Hyundai `LongitudinalController` can reshape accel
  and jerk before CAN transmission.
- On EV6, do not assume one topology. LKA-steering and LFA-steering variants
  move CAN FD traffic to different buses and change which ECU must be disabled
  for openpilot longitudinal to own the car.

## Quick Start

### "Live monitor" = run this, right now, no questions

When the user says "live monitor the longitudinal tuner" (or any variant —
"start the monitor", "watch the longitudinal telemetry"), do this
**immediately**. Do not probe SSH profiles, do not try the dev box, do not ask
for confirmation.

```bash
ssh commaCar 'cd /data/openpilot && /usr/local/venv/bin/python3 -u .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --show-live-tune'
```

Run it with `run_in_background: true` so output streams to a task file you can
tail on demand. The two non-obvious must-haves:

- **`commaCar` is the default.** The tici is normally on the car hotspot when
  the user asks to monitor. `commaHome` and `commaAdb` are fallbacks — do not
  probe them first. If `commaCar` fails, *then* ask the user which network.
- **`python3 -u` is mandatory over SSH.** Without `-u`, Python line-buffers
  to a pipe (no TTY) and the output file stays empty indefinitely. This looks
  identical to "no data flowing" and wastes a driving session debugging it.
- **Dev-box invocations (`.venv/bin/python ...`) produce no data** on this
  workstation — there is no cereal bridge from the tici. Ignore any older
  Quick Start bullet suggesting otherwise.

### Bounded probe (5-10s) to verify the pipeline before a long watch

```bash
ssh commaCar 'cd /data/openpilot && /usr/local/venv/bin/python3 -u .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --duration 5 --show-live-tune'
```

### Alert-only mode (quieter, long drives)

```bash
ssh commaCar 'cd /data/openpilot && /usr/local/venv/bin/python3 -u .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --only-alerts --show-live-tune'
```

### Iterating on live lead-response knobs — print the effective tune first

On-device (read-only) so the values match what controls is actually using:
```bash
ssh commaCar 'cd /data/openpilot && /usr/local/venv/bin/python3 .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py show'
```

### Baseline no-radar AI/model lead `dRel` noise before changing tune values

```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/simulate_ai_lead_noise.py --duration-s 60 --source-noise-std-m 5 --spike-prob-per-s 0 --white-noise-std-m 0.25
```

Add `--disable-model-lead-tracker` to compare against the old raw model-lead
radard path.

### On-device bounded monitor uses the tici venv explicitly

```bash
ssh commaCar 'cd /data/openpilot && /usr/local/venv/bin/python3 .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --duration 10 --only-alerts --show-live-tune'
```

- If the issue is Hyundai CAN FD or EV6 specific, read
  `references/hyundai_canfd_ev6.md` before changing planner code.
- If the issue is “why did it choose ACC vs e2e / blended,” read
  `references/pipeline.md`.
- If you are live-tuning the new lead preview / gap reclaim path, also read
  `references/live_lead_tuning.md`.

## Workflow Decision Tree

- Unexpected slowdown or refusal to accelerate:
  check external caps first. Inspect `longitudinalPlanSP` VTSC, SLC, and RTI
  state plus `longitudinalPlan.allowThrottle` before changing PID or car tuning.
- Lead follows too loosely on pull-away, or reacts too late to a newly
  recognized lead:
  first prove whether ownership is already stable. If `source` is steady and
  the problem is just amplitude/timing, inspect the live lead-tune values.
  The helper script and runtime refresh path let you change those heuristics
  without restarting services.
- A valid lead is still present, but `source=cruise` and accel spikes positive:
  inspect `LEADROLEDBG.source_hysteresis` and
  `lead_present_cruise_accel_cap` before touching reclaim knobs. That is a
  different failure mode from duplicate-lead chatter.
- A freeway cut-in still causes a hard gap snap-back:
  inspect the cut-in settle live knobs and the `LEADROLEDBG` cut-in fields
  before assuming the classifier is late. If the lead becomes control early but
  braking still spikes, the issue is usually the post-recognition gap recovery
  path rather than lead selection.
- Hyundai no-radar AI lead following pulses or chatters near settled headway:
  inspect `LEADROLEDBG.virtual_duplicate`,
  `LEADROLEDBG.filtered_virtual_lead`, and
  `LEADROLEDBG.source_hysteresis` before changing gap reclaim or cut-in
  settings. On this path the primary failure mode is often duplicate model
  hypotheses for one physical car plus noisy model kinematics causing ACC
  ownership to bounce between lead-follow and cruise.
- Source flickers rapidly between `cruise` and `lead` at the edge of lead
  acquisition (subjectively feels like rapid brake-taps on an EV):
  this is the source-stability layer, not the reclaim / cut-in / or older
  Hyundai hysteresis paths. Tune the stability-layer knobs on-device
  (`LeadProbEnter/Exit`, `LeadSourceAcquireFrames/ReleaseFrames`,
  `PhantomLeadHoldS`, `FlutterClampJerkMps3`) before changing MPC or vibe
  settings. Read `references/live_lead_tuning.md` -> Source Stability Layer
  for the layered attack model and recommended enable order.
- Hyundai follow overshoots, then coasts or lightly slows too long while
  `source` stays on the lead:
  inspect `gap_reclaim_obstacle_push_m`,
  `gap_reclaim_projection_scale`, `raw_reclaim_safety_override`, and the
  reclaim-lead state in `LongitudinalMpc`. That symptom usually lives in the
  fixed Hyundai reclaim path, not in the live tune knobs.
- Hard accel behind a slow close lead in stop-and-go:
  inspect `LEADROLEDBG.source_hysteresis.low_speed_queue_hold` before touching
  reclaim or the Hyundai controller overlay. That is a lead-ownership bug
  surface, not a comfort-tuning surface.
- Planner target looks reasonable, but the car command does not:
  compare `longitudinalPlan.aTarget`, `carControl.actuators.accel`,
  `carOutput.actuatorsOutput.accel`, and delayed `carState.aEgo`. If the first
  two diverge, controlsd is involved. If the second and third diverge, the
  brand controller is shaping commands. If the third and fourth diverge after
  actuator delay, look for stock ECU interference or a vehicle-response issue.
- Experimental Mode / AI-vs-deterministic confusion:
  inspect `selfdriveState.experimentalMode`,
  `longitudinalPlanSP.dec.{enabled,active,state}`, and the active model bundle
  generation. The planner can start in blended mode from Experimental Mode and
  still be pushed back toward ACC by Dynamic Experimental Control.
- Hyundai/EV6 engagement or stock ACC fight:
  confirm topology, correct CAN FD bus mapping, and correct ECU disable target
  before touching tuning params.
- Hyundai jerk or stop/go complaints:
  inspect `LongControl` state and the Hyundai
  `opendbc/sunnypilot/car/hyundai/longitudinal/controller.py` overlay before
  concluding the planner is wrong. Hyundai EVs already have separate actuator
  jerk/lookahead tuning in that overlay, so not every follow feel complaint
  should be solved in MPC.

## Verification Commands

- Hyundai no-radar AI lead path:
```bash
pytest selfdrive/controls/tests/test_hyundai_ai_lead_stability.py -q
```

```bash
pytest selfdrive/controls/tests/test_lead_interactions.py -q
```

- Cruise-response and steady-state following regressions:
```bash
pytest selfdrive/controls/tests/test_following_distance.py -q
```

- Live tuning / runtime-refresh coverage:
```bash
pytest selfdrive/controls/tests/test_longitudinal_live_tune.py -q
```

- Source-stability filter and handoff jerk limits:
```bash
pytest selfdrive/controls/tests/test_lead_stability_filter.py selfdrive/controls/tests/test_cruise_reacquire_jerk_limit.py -q
```

- Hyundai longitudinal tuning overlay:
```bash
pytest opendbc/sunnypilot/car/hyundai/tests/test_tuning_controller.py -q
```

- Lead classification regressions:
```bash
pytest selfdrive/controls/lib/tests/test_lead_role_classifier.py -q
```

## References

- Read `references/pipeline.md` for the end-to-end data and control flow,
  planner mode selection, tunable surfaces, and symptom-to-layer mapping.
- Read `references/hyundai_canfd_ev6.md` for EV6-specific topology and Hyundai
  CAN FD longitudinal ownership details.
- Read `references/live_monitoring.md` for the watcher field guide and alert
  meanings.
- Read `references/live_lead_tuning.md` for the runtime ACC lead-response tune,
  the source-stability layer (Schmitt, dwell, phantom, flutter clamp,
  cruise-reacquire jerk), helper script, and no-restart workflow.
- Read `docs/chauffeur/live_tunable_params.md` for the complete catalog of
  live-tunable `Longitudinal.LiveTune.*` params with defaults, ranges, and SSH
  usage examples.

## Skill Maintenance

- Update this skill after real longitudinal debugging sessions.
- Replace stale guidance instead of appending contradictory bullets.
- Keep Hyundai/EV6 notes code-verified; do not copy stale implementation docs
  forward.
- Keep only reusable workflow in `SKILL.md`; move dense branch facts into the
  reference files.
- Treat maintenance as a kaizen loop:
  when a session proves an older branch-specific hypothesis wrong, replace or
  delete the old workflow in the same pass so the next invocation starts from
  the corrected mental model.
- When adding any new tuning parameter that affects longitudinal behavior,
  always register it in `common/params_keys.h`, add a `LeadResponseTuneSpec`
  in `longitudinal_live_tune.py`, and update
  `docs/chauffeur/live_tunable_params.md`. All tuning params must be
  live-adjustable via the params system — never leave them as hardcoded
  constants.
