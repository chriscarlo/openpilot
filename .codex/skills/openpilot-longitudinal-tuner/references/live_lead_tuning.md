# Live Lead Tuning

## Scope

- This tuning surface is only for the new ACC lead-response heuristics:
  early preview for a newly recognized slower lead, safe gap reclaim when a
  lead pulls away, and benign cut-in settle behavior.
- It is not the tuning surface for Hyundai fixed-code lead ownership or
  lead-stability logic. Duplicate-lead collapse, filtered virtual lead
  ownership, low-speed queue hold, cruise-owned lead accel taper, and the
  reclaim-lead optimism release all live in fixed `LongitudinalMpc` code.
- It does not change `forceDecel` behavior. That path is still the normal
  planner stop request used for DM / soft-disable handling.
- It is assistant-oriented in v1. There is no offroad UI for these knobs.

## Runtime Path

- Param keys live in `common/params_keys.h` under the `Longitudinal.LiveTune.*`
  namespace.
- The shared metadata and clamping logic live in
  `selfdrive/controls/lib/longitudinal_live_tune.py`.
- `LongitudinalMpc` refreshes those params every `0.5 s` inside
  `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py`.
- The live config feeds three heuristics:
  `get_lead_approach_preview_buffer()`,
  `get_gap_reclaim_accel_floor()`, and the cut-in settle decel cap applied
  after MPC.
- Hyundai-only duplicate-lead stabilization and ACC source hysteresis also live
  in `LongitudinalMpc`, along with a filtered virtual lead used for stable
  lead-vs-cruise ownership. Those Hyundai source-stability paths are fixed code
  in v1, not live knobs.
- The current Hyundai fixed-code layer also includes:
  low-speed queue hold for slow close leads,
  a cruise-owned lead accel cap when a valid lead still exists,
  and a reclaim-lead state that drops optimistic pull-away dynamics quickly
  once the real lead stops pulling away.
- On Hyundai EVs, final accel and jerk still pass through the separate
  `opendbc/sunnypilot/car/hyundai/longitudinal/controller.py` overlay, which
  already has EV-specific shaping from
  `opendbc/sunnypilot/car/hyundai/longitudinal/config.py`.

## Knobs

- `LeadPreviewStrength`
  Scale how early the planner starts backing out of throttle for a newly
  recognized slower lead that is still outside nominal headway.
- `LeadPreviewGapMinM`
  Minimum extra slack above nominal headway before the preview path can engage.
- `LeadPreviewMaxBufferM`
  Hard cap on how much the preview logic can pull the lead obstacle closer.
- `GapReclaimStrength`
  Scale how eagerly ACC closes a safe extra gap when the lead is pulling away.
- `GapReclaimGapMinM`
  Minimum extra slack above nominal headway before reclaim is allowed.
- `GapReclaimMaxAccel`
  Hard cap on the positive accel floor used for safe gap reclaim.
- `CutInSettleDurationS`
  How long a benign cut-in gets a gradual headway-recovery grace window.
- `CutInSettleMaxDecel`
  Strongest braking the planner is allowed to ask for during that grace window.
- `CutInSettleMaxClosingSpeedMps`
  Highest closing speed that still counts as a benign cut-in rather than a
  situation that should brake normally.

## Helper Script

- From the dev box, use the repo venv. On tici, swap in
  `/usr/local/venv/bin/python3`.

- Show current values:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py show
```

- Show a one-line summary:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py show --shell-summary
```

- Set one or more knobs without restarting services:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py set \
  --gap-reclaim-strength 1.20 \
  --gap-reclaim-max-accel 0.42
```

- Remove overrides for this feature and fall back to defaults:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py reset
```

## Verification

- While `plannerd` is running, change a knob with the helper script and wait
  about one second. The running `LongitudinalMpc` instance should pick up the
  new effective value on its next refresh.
- For live observation, start the watcher with:
```bash
.venv/bin/python .codex/skills/openpilot-longitudinal-tuner/scripts/monitor_longitudinal_anomalies.py --hz 5 --duration 10 --show-live-tune
```
- If behavior becomes springy or late, reset the overrides first and confirm
  the defaults are active before changing more than one knob at once.
- If `LEADROLEDBG` shows `virtual_duplicate.active=true`, debug lead stability
  through `virtual_duplicate`, `filtered_virtual_lead`, and
  `source_hysteresis` before touching live reclaim or cut-in knobs. Those
  knobs cannot fix duplicate same-car hypotheses or noisy lead-vs-cruise
  ownership in the fixed Hyundai source-stability path.
- If `source=cruise` while a valid lead is still present and accel spikes
  positive, debug `source_hysteresis.lead_present_cruise_accel_cap` first.
  Live reclaim knobs are not the root-cause surface for that behavior.
- If `source` stays on the lead but the car overshoots and then coasts or
  lightly slows for too long, debug the Hyundai reclaim path first. That is
  usually stale optimistic reclaim state, not a missing live tune knob.
