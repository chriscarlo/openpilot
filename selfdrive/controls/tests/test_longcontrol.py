from cereal import car
from opendbc.car.hyundai.values import CAR
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, long_control_state_trans




class TestLongControlStateTransition:

  def test_stay_stopped(self):
    CP = car.CarParams.new_message()
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

def test_engage():
  CP = car.CarParams.new_message()
  active = True
  current_state = LongCtrlState.off
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid

def test_starting():
  CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
  active = True
  current_state = LongCtrlState.starting
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.starting
  next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid


class _TestParams:
  def __init__(self, values=None):
    self.values = {} if values is None else dict(values)

  def get(self, key, block=False, encoding=None, return_default=False):
    return self.values.get(key)


def _road_rollout_long_control(*, starting_state=True, brand="hyundai", fingerprint=CAR.KIA_EV6, jerk_mps3=None):
  """Minimal CP for the 2026-07-16 09:47 PDT EV6 rollout transition."""
  CP = car.CarParams.new_message(
    brand=brand,
    carFingerprint=fingerprint,
    startingState=starting_state,
    vEgoStarting=0.1,
    startAccel=1.0,
    stopAccel=-2.0,
    stoppingDecelRate=0.8,
  )
  CP.longitudinalTuning.kpBP = [0.0]
  CP.longitudinalTuning.kpV = [0.0]
  CP.longitudinalTuning.kiBP = [0.0]
  CP.longitudinalTuning.kiV = [0.0]
  CP.longitudinalTuning.kf = 1.0
  values = {}
  if jerk_mps3 is not None:
    values["Longitudinal.LiveTune.StoppingReleaseJerkMps3"] = str(jerk_mps3)
  return CP, LongControl(CP, _TestParams(values))


def test_road_rollout_stopping_release_is_jerk_contained():
  """09:47:22: stopping -2.003 must not jump directly to starting/PID +1.0.

  The road transition happened while ego was still rolling at about 0.54 m/s
  and planner aTarget had just become slightly positive. The first cycle went
  from the full terminal hold to startAccel, then PID kept the positive demand.
  Bound the complete release, including the immediate starting->PID change,
  without weakening a later negative (safety) request.
  """
  _, long_control = _road_rollout_long_control()
  CS = car.CarState.new_message(vEgo=0.54, aEgo=-1.0)
  CS.brakePressed = False
  CS.cruiseState.standstill = False
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  outputs = [-2.0]
  for _ in range(int(round(0.60 / DT_CTRL))):
    outputs.append(long_control.update(
      active=True,
      CS=CS,
      a_target=0.957,
      should_stop=False,
      accel_limits=(-3.5, 2.0),
    ))

  release_deltas = [cur - prev for prev, cur in zip(outputs[:-1], outputs[1:], strict=True)]
  # Six m/s^3 takes the road's -2 -> +1 reversal across 0.50 s instead of one
  # 10 ms control tick, while retaining the existing launch-follow demand.
  assert max(release_deltas) <= (6.0 * DT_CTRL) + 1e-9
  assert long_control.long_control_state == LongCtrlState.pid
  assert outputs[-1] >= 0.95


def test_ev6_starting_state_true_negative_then_positive_stays_bounded():
  _, long_control = _road_rollout_long_control()
  CS = car.CarState.new_message(vEgo=0.54, aEgo=0.0)
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  released = long_control.update(True, CS, a_target=0.025, should_stop=False, accel_limits=(-3.5, 2.0))
  assert abs(released - (-1.94)) < 1e-9
  assert long_control.stopping_release_active

  # A new negative target bypasses the comfort-only upward bound, but keeps the
  # rollout episode armed in case PID immediately reverses direction again.
  braking = long_control.update(True, CS, a_target=-3.0, should_stop=False, accel_limits=(-3.5, 2.0))
  assert braking < released
  assert long_control.stopping_release_active

  within_bound = long_control.update(True, CS, a_target=-2.98, should_stop=False, accel_limits=(-3.5, 2.0))
  assert within_bound == -2.98
  assert long_control.stopping_release_active

  resumed = long_control.update(True, CS, a_target=0.957, should_stop=False, accel_limits=(-3.5, 2.0))
  assert resumed == within_bound + 6.0 * DT_CTRL
  assert long_control.stopping_release_active

  while long_control.stopping_release_active:
    long_control.update(True, CS, a_target=0.957, should_stop=False, accel_limits=(-3.5, 2.0))
  assert long_control.last_output_accel >= 0.0


def test_ev6_starting_state_false_direct_pid_release_is_bounded():
  _, long_control = _road_rollout_long_control(starting_state=False)
  CS = car.CarState.new_message(vEgo=0.54, aEgo=0.0)
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  released = long_control.update(True, CS, a_target=0.957, should_stop=False, accel_limits=(-3.5, 2.0))

  assert long_control.long_control_state == LongCtrlState.pid
  assert released == -2.0 + 6.0 * DT_CTRL
  assert long_control.stopping_release_active


def test_ev6_stopping_release_uses_bounded_harness_param():
  _, long_control = _road_rollout_long_control(starting_state=False, jerk_mps3=2.0)
  CS = car.CarState.new_message(vEgo=0.54, aEgo=0.0)
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  released = long_control.update(True, CS, a_target=0.957, should_stop=False, accel_limits=(-3.5, 2.0))

  assert long_control.stopping_release_jerk_mps3 == 2.0
  assert released == -2.0 + 2.0 * DT_CTRL


def test_ev6_live_param_refresh_uses_injected_replay_clock():
  CP, _ = _road_rollout_long_control(starting_state=False)
  params = _TestParams({"Longitudinal.LiveTune.StoppingReleaseJerkMps3": "6.0"})
  replay_time_s = [0.0]
  long_control = LongControl(CP, params, time_fn=lambda: replay_time_s[0])
  CS = car.CarState.new_message(vEgo=0.54, aEgo=0.0)
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  first = long_control.update(True, CS, a_target=0.957, should_stop=False, accel_limits=(-3.5, 2.0))
  params.values["Longitudinal.LiveTune.StoppingReleaseJerkMps3"] = "2.0"
  replay_time_s[0] = 0.49
  before_refresh = long_control.update(True, CS, a_target=0.957, should_stop=False, accel_limits=(-3.5, 2.0))
  replay_time_s[0] = 0.50
  after_refresh = long_control.update(True, CS, a_target=0.957, should_stop=False, accel_limits=(-3.5, 2.0))

  assert first == -2.0 + 6.0 * DT_CTRL
  assert before_refresh == first + 6.0 * DT_CTRL
  assert after_refresh == before_refresh + 2.0 * DT_CTRL


def test_non_ev6_stop_release_is_exact_legacy_behavior():
  _, long_control = _road_rollout_long_control(fingerprint=CAR.HYUNDAI_IONIQ_5)
  CS = car.CarState.new_message(vEgo=0.54, aEgo=0.0)
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  starting = long_control.update(True, CS, a_target=0.025, should_stop=False, accel_limits=(-3.5, 2.0))
  braking = long_control.update(True, CS, a_target=-3.0, should_stop=False, accel_limits=(-3.5, 2.0))
  resumed = long_control.update(True, CS, a_target=0.957, should_stop=False, accel_limits=(-3.5, 2.0))

  assert starting == 1.0
  assert braking == -3.0
  assert resumed == 0.957
  assert not long_control.stopping_release_active


def test_wrong_brand_same_fingerprint_is_exact_legacy_behavior():
  _, long_control = _road_rollout_long_control(brand="mock")
  CS = car.CarState.new_message(vEgo=0.54, aEgo=0.0)
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  output = long_control.update(True, CS, a_target=0.025, should_stop=False, accel_limits=(-3.5, 2.0))

  assert output == 1.0
  assert not long_control.stopping_release_active


def test_stationary_stop_release_preserves_launch_floor():
  """Do not reintroduce the road-205 weak launch behind a departing lead."""
  _, long_control = _road_rollout_long_control()
  CS = car.CarState.new_message(vEgo=0.0, aEgo=0.0)
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  output = long_control.update(True, CS, a_target=0.2, should_stop=False, accel_limits=(-3.5, 2.0))
  assert long_control.long_control_state == LongCtrlState.starting
  assert output == 1.0
  assert not long_control.stopping_release_active


def test_rolling_stop_release_hands_off_if_car_reaches_standstill():
  """A comfort ramp must not keep braking after the rolling car stops."""
  _, long_control = _road_rollout_long_control()
  CS = car.CarState.new_message(vEgo=0.100001, aEgo=0.0)
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  released = long_control.update(True, CS, a_target=0.957, should_stop=False, accel_limits=(-3.5, 2.0))
  assert released == -1.94
  assert long_control.stopping_release_active

  CS.vEgo = 0.0
  CS.cruiseState.standstill = True
  launch = long_control.update(True, CS, a_target=0.2, should_stop=False, accel_limits=(-3.5, 2.0))
  assert launch == 1.0
  assert not long_control.stopping_release_active


def test_terminal_stop_hold_is_unchanged_by_rollout_guard():
  _, long_control = _road_rollout_long_control()
  CS = car.CarState.new_message(vEgo=0.0, aEgo=0.0)
  long_control.long_control_state = LongCtrlState.stopping
  long_control.last_output_accel = -2.0

  output = long_control.update(True, CS, a_target=0.5, should_stop=True, accel_limits=(-3.5, 2.0))
  assert long_control.long_control_state == LongCtrlState.stopping
  assert output == -2.0
