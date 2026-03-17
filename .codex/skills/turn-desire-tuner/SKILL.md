---
name: turn-desire-tuner
description: >
  Trace, debug, and tune the low-speed turn-desire path that converts a blinker
  into `turnLeft` or `turnRight` model input. Use when investigating
  `DesireHelper`, blinker-triggered intersection turns, `turnLeft` /
  `turnRight`, low-speed turn behavior, why the model did or did not react to a
  turn signal, or which stock/SNPE/tinygrad `modeld` runner currently owns the
  desire input path.
---

# Turn Desire Tuner

## Guardrails

- Treat branch code as source of truth. `docs/chauffeur/turn_desires/` is
  useful background, but it does not capture the current runner split or the
  fact that only the stock `selfdrive/modeld/modeld.py` path logs
  `"Sending turn desire to model ..."`.
- Separate three different surfaces before changing anything:
  `DesireHelper` activation, desire transport into the active `modeld` runner,
  and the model or car's downstream behavior. A correct input pulse does not
  prove the model will actually steer onto the side street.
- Keep the two speed thresholds straight:
  `LANE_CHANGE_SPEED_MIN` is `20 mph`, while `TURN_DESIRE_SPEED_MAX` is
  `35 mph`. Bugs and false assumptions usually come from mixing those gates.
- Do not treat `selfdrive/debug/test_turn_desires.py` as proof that the input
  desire fired. That script reads `modelV2.meta.desireState`, which is model
  output probability, not the `DesireHelper.desire` input enum.

## Quick Loop

- Start at `selfdrive/controls/lib/desire_helper.py`. That is where the branch
  adds the low-speed turn override on top of the lane-change state machine.
- Then map the active runner:
  `system/manager/process_config.py` plus `sunnypilot/models/helpers.py`.
  The shared turn logic lives in `DesireHelper`, but the active process is one
  of:
  - `selfdrive/modeld/modeld.py` for stock models
  - `sunnypilot/modeld/modeld.py` for SNPE models
  - `sunnypilot/modeld_v2/modeld.py` for tinygrad models
- For live reaction checks, use `selfdrive/debug/test_turn_desires.py`.
  For dense pipeline facts and edge cases, read `references/pipeline.md`.

## Workflow Decision Tree

- Turn desire did not arm:
  inspect `lateral_active`, `one_blinker`, and `v_ego < TURN_DESIRE_SPEED_MAX`
  first. If those are true, verify whether you are looking at the input-side
  log or the next model cycle's output.
- Lane-change UI or events still fired during a turn attempt:
  inspect `lane_change_state` and `lane_change_direction`. Between `20 mph` and
  `35 mph`, the lane-change state machine can progress while the final
  `self.desire` is still overridden to `turnLeft` or `turnRight`.
- Monitor script did not show `turnLeft` or `turnRight`:
  remember that `selfdrive/debug/test_turn_desires.py` reads
  `modelV2.meta.desireState`, not the input desire. Confirm the shared
  `"Turn desire activated:"` cloudlog line first. Only the stock modeld path
  adds the extra `"Sending turn desire to model ..."` line.
- Need to change the desire transport path:
  edit the currently active `modeld` implementation after mapping the runner.
  Do not change only `selfdrive/modeld/modeld.py` if the device is actually
  running SNPE or tinygrad.
- Behavior appears one frame late:
  that is expected. `DH.update(...)` runs after the current model inference
  because it consumes `lane_change_prob` derived from the current
  `modelV2.meta.desireState`. The updated desire is sent on the next inference.

## Recommended Verification Commands

Use exact checks instead of vague "test turn desires."

### Live model-output monitor

```bash
.venv/bin/python selfdrive/debug/test_turn_desires.py
```

### Direct turn-desire unit coverage

```bash
pytest sunnypilot/selfdrive/controls/lib/tests/test_turn_desires.py -q
```

### Adjacent lane-change safety net

This still matters because turn desire shares the same helper and lane-change
state machine:

```bash
pytest sunnypilot/selfdrive/controls/lib/tests/test_auto_lane_change.py -q
```

### Code search to confirm current branch facts

```bash
rg -n "TURN_DESIRE_SPEED_MAX|turn_desire_active|Sending turn desire to model|desire = DH.desire|DH.update\\(" \
  selfdrive/controls/lib/desire_helper.py \
  selfdrive/modeld/modeld.py \
  sunnypilot/modeld/modeld.py \
  sunnypilot/modeld_v2/modeld.py
```

## References

- Read `references/pipeline.md` for the code-verified file map, speed-band
  behavior, runner split, observability caveats, and the exact two-cycle state
  probe results.
- Use `docs/chauffeur/turn_desires/documentation/implementation-summary.md` and
  `docs/chauffeur/turn_desires/testing/quick-reference.md` as supplemental
  background, not as source of truth.

## Skill Maintenance

- Update this skill only after verifying behavior in branch code or with a
  reproducible probe.
- If SNPE or tinygrad gains its own turn-desire logging, update the runner
  caveat instead of stacking another note.
- Keep reusable workflow in `SKILL.md`; move dense branch facts to
  `references/pipeline.md`.
