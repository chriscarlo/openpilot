# Live Monitoring

## Watcher Script

- Run:
```bash
python3 .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5
```
- Alert-only mode:
```bash
python3 .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --only-alerts
```
- Include the current live lead-response tune in the startup header:
```bash
python3 .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --show-live-tune
```
- Save rendered samples:
```bash
python3 .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --jsonl-out .cache/longitudinal_watch.jsonl
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

## Important Limitation

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
- If the watcher says `op_long=N`, controlsd is not owning longitudinal, so the
  script is mainly useful for planner and overlay observation.
