# Turn Desire Pipeline

## Source Files

- `selfdrive/controls/lib/desire_helper.py`
  - Defines both speed thresholds:
    - `LANE_CHANGE_SPEED_MIN = 20 mph`
    - `TURN_DESIRE_SPEED_MAX = 35 mph`
  - Computes the normal lane-change state machine and then overrides the final
    `self.desire` to `log.Desire.turnLeft` or `log.Desire.turnRight` when:
    - lateral control is active
    - exactly one blinker is active
    - `v_ego < TURN_DESIRE_SPEED_MAX`
  - Tracks `turn_desire_active` and logs `"Turn desire activated: ..."` and
    `"Turn desire deactivated"`.

- `selfdrive/modeld/modeld.py`
- `sunnypilot/modeld/modeld.py`
- `sunnypilot/modeld_v2/modeld.py`
  - All three import `DesireHelper`, instantiate `DH = DesireHelper()`, and
    read `desire = DH.desire` before running model inference.
  - All three convert the enum into an 8-slot one-hot vector.
  - All three turn the input into a rising-edge pulse before appending it to the
    temporal model context.
  - Only the stock `selfdrive/modeld/modeld.py` path adds the extra debug log:
    `"Sending turn desire to model: turnLeft/turnRight"`.

- `selfdrive/modeld/constants.py`
  - `DESIRE_LEN = 8`
  - `DESIRE_PRED_WIDTH = 8`

- `cereal/log.capnp`
  - Current enum order:
    - `0 = none`
    - `1 = turnLeft`
    - `2 = turnRight`
    - `3 = laneChangeLeft`
    - `4 = laneChangeRight`
    - `5 = keepLeft`
    - `6 = keepRight`
  - The desire vector width is still 8, so tools that decode the highest-prob
    slot may show an extra "unknown" bucket for index `7`.

- `selfdrive/modeld/fill_model_msg.py`
- `sunnypilot/modeld/fill_model_msg.py`
- `sunnypilot/modeld_v2/fill_model_msg.py`
  - Populate `meta.desireState` and `meta.desirePrediction` from model output,
    not from the input desire sent by `DesireHelper`.

- `selfdrive/debug/test_turn_desires.py`
  - Monitors `modelV2.meta.desireState` and prints the highest-probability
    desire class. This is useful for model reaction, but it is not direct proof
    that the input pulse was sent.

- `system/manager/process_config.py`
- `sunnypilot/models/helpers.py`
  - Map the active `modeld` implementation:
    - stock runner -> `selfdrive.modeld.modeld`
    - SNPE runner -> `sunnypilot/modeld`
    - tinygrad runner -> `sunnypilot/modeld_v2`

## Execution Order

The temporal ordering matters.

1. `modeld` starts a loop iteration and reads `desire = DH.desire`.
2. `modeld` builds a one-hot desire vector and passes it into the active model
   runner.
3. The runner converts that vector into a rising-edge pulse and writes it into
   the temporal desire history buffer.
4. The model runs and produces `modelV2.meta.desireState`.
5. `modeld` derives `lane_change_prob` from the current model output.
6. `DH.update(carState, latActive, lane_change_prob)` runs and computes the
   next loop iteration's `DH.desire`, `lane_change_state`, and
   `lane_change_direction`.

Implication:
- the desire you compute now is consumed on the next inference, not the current
  one
- a live monitor that only watches `modelV2.meta.desireState` can lag the input
  activation by a cycle and can also show that the model ignored the input

## Speed Bands And State Interaction

The branch has two overlapping behaviors:

- Below `20 mph`
  - lane-change state machine stays `off`
  - turn-desire override can still arm
  - result: turn desire can be active while `lane_change_state` is still `off`

- From `20 mph` up to `35 mph`
  - lane-change state machine can enter `preLaneChange` and later
    `laneChangeStarting`
  - final `self.desire` is still overridden to `turnLeft` or `turnRight`
  - result: lane-change metadata can say "lane change" while model input is
    actually a turn desire

- At or above `35 mph`
  - no turn-desire override
  - behavior falls back to normal lane-change desire logic

## Two-Cycle Probe Results

These were reproduced on 2026-03-17 with `.venv/bin/python` by instantiating
`DesireHelper` and feeding a synthetic `carstate`.

Reproduce with a minimal probe like:

```bash
.venv/bin/python - <<'PY'
from types import SimpleNamespace
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.common.constants import CV

def cs(v_mph, left=False, right=False, steeringPressed=False, steeringTorque=0.0):
  return SimpleNamespace(
    vEgo=v_mph * CV.MPH_TO_MS,
    leftBlinker=left,
    rightBlinker=right,
    steeringPressed=steeringPressed,
    steeringTorque=steeringTorque,
    leftBlindspot=False,
    rightBlindspot=False,
    brakePressed=False,
  )

dh = DesireHelper()
for i, state in enumerate([cs(25, left=True), cs(25, left=True, steeringPressed=True, steeringTorque=1.0)], start=1):
  dh.update(state, lateral_active=True, lane_change_prob=0.5)
  print(i, int(dh.lane_change_state), int(dh.lane_change_direction), int(dh.desire), dh.turn_desire_active)
PY
```

### 15 mph, left blinker, two updates

- cycle 1:
  - `lane_change_state = off`
  - `lane_change_direction = none`
  - `desire = turnLeft`
  - `turn_desire_active = True`
- cycle 2:
  - same result

Interpretation:
- below `20 mph`, turn desire can stay active with no lane-change state

### 25 mph, left blinker, no steering torque, two updates

- cycle 1:
  - `lane_change_state = preLaneChange`
  - `lane_change_direction = none`
  - `desire = turnLeft`
  - `turn_desire_active = True`
- cycle 2:
  - `lane_change_state = preLaneChange`
  - `lane_change_direction = left`
  - `desire = turnLeft`
  - `turn_desire_active = True`

Interpretation:
- the first blinker-edge cycle enters `preLaneChange`, but direction is still
  `none`
- direction is assigned on the next update

### 25 mph, left blinker, steering torque on second update

- cycle 1:
  - same as the no-torque case above
- cycle 2:
  - `lane_change_state = laneChangeStarting`
  - `lane_change_direction = left`
  - `desire = turnLeft`
  - `turn_desire_active = True`

Interpretation:
- the lane-change state machine can advance, but the final desire is still the
  turn override

### 40 mph, left blinker, first update

- cycle 1:
  - `lane_change_state = preLaneChange`
  - `lane_change_direction = none`
  - `desire = none`
  - `turn_desire_active = False`

Interpretation:
- above `35 mph`, only the normal lane-change path remains

## Downstream Consumers

- `selfdrive/controls/controlsd.py`
  - uses `modelV2.meta.laneChangeState` and `laneChangeDirection` to keep
    blinkers active during lane changes
  - does not consume the input turn desire directly

- `selfdrive/selfdrived/selfdrived.py`
  - raises `preLaneChangeLeft`, `preLaneChangeRight`, and `laneChange` events
    from `modelV2.meta.laneChangeState` and `laneChangeDirection`
  - does not consume the input turn desire directly

Implication:
- from `20 mph` to `35 mph`, UI or event behavior can still look like a lane
  change while the model input is a turn desire

## Observability Rules

- Shared input-side proof across all runners:
  - `DesireHelper` cloudlog line: `"Turn desire activated: LEFT/RIGHT at ..."`

- Stock-only extra proof:
  - `selfdrive/modeld/modeld.py` logs
    `"Sending turn desire to model: turnLeft/turnRight"`

- Output-side reaction:
  - `selfdrive/debug/test_turn_desires.py`
  - `modelV2.meta.desireState`

Treat those as different signals:
- activation log proves the helper armed
- stock modeld log proves the stock runner saw the input enum
- `desireState` only proves what the model predicted afterward

## Coverage Status

Code search on 2026-03-17 found:

- direct pytest coverage in:
  `sunnypilot/selfdrive/controls/lib/tests/test_turn_desires.py`
- one live debug helper:
  `selfdrive/debug/test_turn_desires.py`
- adjacent lane-change coverage in:
  `sunnypilot/selfdrive/controls/lib/tests/test_auto_lane_change.py`

Use the direct turn-desire test as the primary regression net for this feature.
Use the adjacent lane-change test to protect the shared state-machine path.

## Existing Repo Docs

Supplemental docs already exist:

- `docs/chauffeur/turn_desires/documentation/implementation-summary.md`
- `docs/chauffeur/turn_desires/testing/quick-reference.md`

They are useful as background, but they do not capture all current branch facts.
Most importantly:

- they assume the stock `modeld` debug log as if it were universal
- they do not call out that `test_turn_desires.py` watches model output rather
  than the input desire
