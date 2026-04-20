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

- Diagnose adjacent-lead / path-relative classification (adds per-row
  `yRel`, `dPath`, `vLat` for both leads to the stdout line):
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 10 --show-lateral
```

- Enable classifier cloudlog emission for a session (sets
  `VTSC.Expert.AdjLeadDebugLogEnabled=1`, restores on exit):
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 10 --duration 60 --enable-lead-role-log
```
  Then tail the cloudlog on-device with
  `tail -F /data/log/cloudlog | grep LEADROLEDBG`.

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
- Lead lateral (written to JSONL always under `lead0Lat` / `lead1Lat`;
  shown in stdout with `--show-lateral`):
  `radarState.leadOne.{yRel, dPath, vLat, vRel, vLead, aLeadK}` and the
  same for `leadTwo`. Use these to distinguish source flicker driven by
  `prob` edge cases from flicker driven by lateral-classification
  ambiguity (in-lane leads measuring as adjacent or vice versa).

## Important Limitation

- The watcher is best at regime attribution. It does not print the full
  Hyundai `LEADROLEDBG` internals by itself. Use the watcher to prove whether
  the car is in `cruise`, `lead0`, `lead1`, or a cap-limited regime first,
  then inspect `LEADROLEDBG` if the root cause still appears Hyundai-specific.
- `LEADROLEDBG` is off by default. `--enable-lead-role-log` toggles it on
  for the current session only and restores on clean exit. If the monitor
  is killed hard, the flag stays set — disable it manually via
  `/usr/local/venv/bin/python3 -c "from openpilot.common.params import Params; Params().put_bool('VTSC.Expert.AdjLeadDebugLogEnabled', False)"`.
- When the source-stability layer is active, `LEADROLEDBG` entries also
  include a top-level `lead_stability` dict with per-slot `latched`,
  `valid_streak`, `invalid_streak`, `latched_valid_streak`,
  `phantom_age_s`, and `out_status`. That is where to look when the
  stabilized lead status differs from the raw vision lead.
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
