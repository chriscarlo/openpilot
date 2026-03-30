# Live Monitoring

## Watcher Script

- From the dev box, prefer the repo venv:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5
```

- For a bounded live sample while the user is driving:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --duration 10 --show-live-tune
```

- Alert-only mode:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --only-alerts
```
- Include the current live lead-response tune in the startup header:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --show-live-tune
```
- Save rendered samples:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --jsonl-out .cache/longitudinal_watch.jsonl
```

- On tici, use the device venv explicitly:
```bash
cd /data/openpilot
/usr/local/venv/bin/python3 .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --duration 10 --only-alerts
```

## What It Compares

- Planner target:
  `longitudinalPlan.aTarget`
- Model target:
  `modelV2.action.desiredAcceleration`
- Controlsd actuator command:
  `carControl.actuators.accel`
- Brand-controller command actually sent downstream:
  `carOutput.actuatorsOutput.accel`
- Vehicle response:
  delayed comparison between `carOutput.actuatorsOutput.accel` and
  `carState.aEgo`
- Overlay caps:
  VTSC, SLC, and RTI
- Lead presence:
  `radarState.leadOne`

## Important Limitation

- The watcher is best at regime attribution. It does not print the full
  Hyundai `LEADROLEDBG` internals by itself. Use the watcher to prove whether
  the car is in `cruise`, `lead0`, `lead1`, or a cap-limited regime first,
  then inspect `LEADROLEDBG` if the root cause still appears Hyundai-specific.
- Weather-aware slowdown is internal-only to the planner. The watcher cannot
  attribute weather caps from published buses alone.
- `longitudinalPlanSP` does not include RTI state. The watcher reads RTI from
  `rtiStateSP` separately.
- Because of those two facts, the watcher is best for rapid attribution and
  consistency checks, not for proving every possible slowdown source from
  published telemetry alone.

## Alert Meanings

- `throttle-block`
  `allowThrottle` is false while the planner still wants positive accel.
  Check model gas-press probability and throttle gating before changing PID.
- `plan-vs-model`
  planner `aTarget` and model desired accel are materially different while the
  system is in Experimental / blended behavior. This is a mode-arbitration clue,
  not automatically a bug.
- `shape`
  controlsd accel and brand-controller accel are materially different.
  On Hyundai this often means the custom Hyundai longitudinal tuner is shaping
  the command as designed.
- `tracking-gap`
  delayed car-controller accel still does not match `aEgo`. That points away
  from the planner and toward actuation delay, stock ECU interference, or bad
  vehicle-response assumptions.
- `stop-mismatch`
  `shouldStop` and `LongControl` stopping state disagree near low speed.
  Check stop-state transitions and brand stop request handling.

## Reading Cap Notes

- `cap=vtsc:<speed>`
  VTSC is currently the slowest visible external cap.
- `cap=slc:<speed>`
  Speed Limit Control is the slowest visible external cap.
- `cap=rti:<speed>`
  RTI is currently providing the slowest visible threat-based cap.
- No cap note does not prove “no overlay.” It only means no visible VTSC, SLC,
  or RTI cap is lower than the others at that instant.

## Reading Hyundai Notes

- On Hyundai with openpilot longitudinal, `shape` can be normal because
  `LongitudinalController` intentionally jerks and smooths the raw accel
  request.
- A persistent `tracking-gap` after the configured actuator delay is more
  suspicious than a transient `shape` note.
- If `src=cruise` while `lead` stays valid and the car accelerates too hard,
  the next check is `LEADROLEDBG.source_hysteresis`, not the Hyundai actuator
  overlay.
- If `src=lead0` or `src=lead1` stays stable but the car overshoots and then
  coasts or lightly slows too long, the next check is the Hyundai reclaim path
  in `LongitudinalMpc`, especially
  `gap_reclaim_obstacle_push_m`,
  `gap_reclaim_projection_scale`, and
  `raw_reclaim_safety_override`.
- If the watcher says `op_long=N`, controlsd is not owning longitudinal, so the
  script is mainly useful for planner and overlay observation.
