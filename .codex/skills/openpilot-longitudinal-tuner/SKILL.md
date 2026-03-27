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
- On Hyundai CAN FD, do not assume `carControl.actuators.accel` is the final
  car command. Sunnypilot's Hyundai `LongitudinalController` can reshape accel
  and jerk before CAN transmission.
- On EV6, do not assume one topology. LKA-steering and LFA-steering variants
  move CAN FD traffic to different buses and change which ECU must be disabled
  for openpilot longitudinal to own the car.

## Quick Start

- If the behavior is happening live, run the watcher first:
```bash
python3 .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5
```

- If you are iterating on the new lead-response heuristics, print the effective
  live tune first:
```bash
python3 .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py show
```

- On-device, the same script works from `/data/openpilot`:
```bash
cd /data/openpilot
python3 .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --only-alerts
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
  inspect the live lead-tune values before editing code. The helper script and
  runtime refresh path let you change those heuristics without restarting
  services.
- A freeway cut-in still causes a hard gap snap-back:
  inspect the cut-in settle live knobs and the `LEADROLEDBG` cut-in fields
  before assuming the classifier is late. If the lead becomes control early but
  braking still spikes, the issue is usually the post-recognition gap recovery
  path rather than lead selection.
- Hyundai no-radar AI lead following pulses or chatters near settled headway:
  inspect `LEADROLEDBG.virtual_duplicate` and `LEADROLEDBG.source_hysteresis`
  before changing gap reclaim or cut-in settings. On this path the primary
  failure mode is often duplicate model hypotheses for one physical car causing
  the active ACC obstacle to bounce between lead-follow and cruise.
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

- Core longitudinal behavior:
```bash
pytest selfdrive/controls/tests/test_following_distance.py -q
```

```bash
pytest selfdrive/controls/tests/test_longitudinal_live_tune.py -q
```

```bash
pytest selfdrive/car/tests/test_cruise_speed.py -q
```

- Hyundai longitudinal tuning overlay:
```bash
pytest opendbc/sunnypilot/car/hyundai/tests/test_tuning_controller.py -q
```

- EV6 CAN FD speed-limit ingestion path:
```bash
pytest selfdrive/car/hyundai/tests/test_ev6_dashboard_speed_limit.py -q
```

## References

- Read `references/pipeline.md` for the end-to-end data and control flow,
  planner mode selection, tunable surfaces, and symptom-to-layer mapping.
- Read `references/hyundai_canfd_ev6.md` for EV6-specific topology and Hyundai
  CAN FD longitudinal ownership details.
- Read `references/live_monitoring.md` for the watcher field guide and alert
  meanings.
- Read `references/live_lead_tuning.md` for the runtime ACC lead-response tune,
  helper script, and no-restart workflow.

## Skill Maintenance

- Update this skill after real longitudinal debugging sessions.
- Replace stale guidance instead of appending contradictory bullets.
- Keep Hyundai/EV6 notes code-verified; do not copy stale implementation docs
  forward.
- Keep only reusable workflow in `SKILL.md`; move dense branch facts into the
  reference files.
