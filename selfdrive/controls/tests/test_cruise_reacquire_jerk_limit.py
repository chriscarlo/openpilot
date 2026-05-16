import types
from collections import deque
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner


def _make_stub(jerk_limit: float = 0.6, window_s: float = 1.5,
               dt: float = 0.05, a_max: float = 2.0) -> SimpleNamespace:
  stub = SimpleNamespace(
    output_a_target=0.0,
    dt=dt,
    _planner_output_accel_limits=(-3.0, a_max),
    _prev_mpc_source="",
    _cruise_pos_jerk_frames_left=0,
    _cruise_pos_jerk_prev_a=0.0,
    mpc=SimpleNamespace(_live_tune_cfg=SimpleNamespace(
      cruise_reacquire_pos_jerk_limit=jerk_limit,
      cruise_reacquire_jerk_window_s=window_s,
    )),
  )
  stub._apply_cruise_reacquire_jerk_limit = types.MethodType(
    LongitudinalPlanner._apply_cruise_reacquire_jerk_limit, stub,
  )
  return stub


def _make_flutter_stub(jerk_cap: float = 0.8, n_trans: float = 2.0, window_s: float = 1.0,
                       bypass_decel: float = 1.5, dt: float = 0.05) -> SimpleNamespace:
  stub = SimpleNamespace(
    output_a_target=0.0,
    dt=dt,
    _flutter_prev_source="",
    _source_transition_frames=deque(),
    _flutter_clamp_prev_a=0.0,
    _flutter_mode_active=False,
    mpc=SimpleNamespace(_live_tune_cfg=SimpleNamespace(
      flutter_detect_transitions=n_trans,
      flutter_detect_window_s=window_s,
      flutter_clamp_jerk_mps3=jerk_cap,
      flutter_clamp_bypass_decel_mps2=bypass_decel,
    )),
  )
  stub._apply_flutter_mode_clamp = types.MethodType(
    LongitudinalPlanner._apply_flutter_mode_clamp, stub,
  )
  return stub


class TestCruiseReacquireJerkLimit:
  def test_lead_follow_is_untouched(self):
    stub = _make_stub()
    stub.output_a_target = -0.3
    stub._apply_cruise_reacquire_jerk_limit("lead0")
    assert stub.output_a_target == pytest.approx(-0.3)
    stub.output_a_target = 0.8
    stub._apply_cruise_reacquire_jerk_limit("lead0")
    assert stub.output_a_target == pytest.approx(0.8)

  def test_cruise_without_prior_lead_is_untouched(self):
    stub = _make_stub()
    stub.output_a_target = 0.5
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    assert stub.output_a_target == pytest.approx(0.5)

  def test_lead_to_cruise_transition_clamps_positive_jerk(self):
    stub = _make_stub(jerk_limit=0.6, window_s=1.5, dt=0.05)
    stub.output_a_target = -0.2
    stub._apply_cruise_reacquire_jerk_limit("lead0")  # establish lead history

    stub.output_a_target = 1.2  # MPC wants big positive accel
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    # Transition arms the clamp and slews from the prior lead-follow value.
    # max_up = 0.6 * 0.05 = 0.03, so ceiling = -0.2 + 0.03 = -0.17.
    assert stub.output_a_target == pytest.approx(-0.17, abs=1e-6)

  def test_braking_is_never_clamped(self):
    stub = _make_stub()
    stub.output_a_target = 0.5
    stub._apply_cruise_reacquire_jerk_limit("lead0")
    stub.output_a_target = -1.2  # MPC wants brake post-transition
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    assert stub.output_a_target == pytest.approx(-1.2)

  def test_window_expires_after_configured_duration(self):
    stub = _make_stub(jerk_limit=0.6, window_s=1.0, dt=0.05)
    stub.output_a_target = 0.0
    stub._apply_cruise_reacquire_jerk_limit("lead0")

    stub.output_a_target = 1.5
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    # frames_left starts at ceil(1.0 / 0.05) = 20; decremented on the same call.
    assert stub._cruise_pos_jerk_frames_left == 19

    # After 19 more frames, the window should be exhausted.
    for _ in range(19):
      stub.output_a_target = 1.5
      stub._apply_cruise_reacquire_jerk_limit("cruise")
    assert stub._cruise_pos_jerk_frames_left == 0

    # Next frame is unclipped.
    stub.output_a_target = 1.5
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    assert stub.output_a_target == pytest.approx(1.5)

  def test_window_closes_when_output_reaches_cap(self):
    stub = _make_stub(jerk_limit=5.0, window_s=1.5, dt=0.05, a_max=0.8)
    stub.output_a_target = 0.0
    stub._apply_cruise_reacquire_jerk_limit("lead0")
    stub.output_a_target = 2.0
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    # ceiling = 0 + 5.0 * 0.05 = 0.25. Not yet at cap (0.8).
    assert stub.output_a_target == pytest.approx(0.25)
    assert stub._cruise_pos_jerk_frames_left > 0

    # Drive it up over several frames until we hit the cap.
    for _ in range(5):
      stub.output_a_target = 2.0
      stub._apply_cruise_reacquire_jerk_limit("cruise")
    # After enough frames, output should have reached the cap and window closed.
    assert stub.output_a_target >= 0.8 - 1e-3
    assert stub._cruise_pos_jerk_frames_left == 0

  def test_returning_to_lead_clears_window(self):
    stub = _make_stub(jerk_limit=0.6, window_s=1.5, dt=0.05)
    stub.output_a_target = 0.0
    stub._apply_cruise_reacquire_jerk_limit("lead0")
    stub.output_a_target = 1.0
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    assert stub._cruise_pos_jerk_frames_left > 0
    stub.output_a_target = 0.8
    stub._apply_cruise_reacquire_jerk_limit("lead0")
    assert stub._cruise_pos_jerk_frames_left == 0
    assert stub.output_a_target == pytest.approx(0.8)

  def test_disabled_when_jerk_limit_zero(self):
    stub = _make_stub(jerk_limit=0.0, window_s=1.5)
    stub.output_a_target = -0.2
    stub._apply_cruise_reacquire_jerk_limit("lead0")
    stub.output_a_target = 1.2
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    assert stub.output_a_target == pytest.approx(1.2)

  def test_disabled_when_window_zero(self):
    stub = _make_stub(jerk_limit=0.6, window_s=0.0)
    stub.output_a_target = -0.2
    stub._apply_cruise_reacquire_jerk_limit("lead0")
    stub.output_a_target = 1.2
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    assert stub.output_a_target == pytest.approx(1.2)


class TestFlutterModeClamp:
  def test_no_clamp_when_single_transition(self):
    stub = _make_flutter_stub()
    stub.output_a_target = 0.5
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    stub.output_a_target = 1.5  # big jump, but only 1 transition
    stub._apply_flutter_mode_clamp("cruise", 0.0)
    assert stub.output_a_target == pytest.approx(1.5)
    assert stub._flutter_mode_active is False

  def test_clamp_activates_on_second_transition(self):
    stub = _make_flutter_stub(jerk_cap=0.8, n_trans=2.0, dt=0.05)
    stub.output_a_target = 0.0
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    stub._apply_flutter_mode_clamp("cruise", 0.0)  # 1st transition
    stub.output_a_target = 1.0  # big jump
    stub._apply_flutter_mode_clamp("lead0", 0.0)  # 2nd transition -> flutter mode
    # max_step = 0.8 * 0.05 = 0.04. Prev was 0.0 (set by first clamp call).
    assert stub.output_a_target == pytest.approx(0.04, abs=1e-6)
    assert stub._flutter_mode_active is True

  def test_clamps_both_directions(self):
    stub = _make_flutter_stub(jerk_cap=0.8, n_trans=2.0, dt=0.05)
    # Trigger flutter mode with 2 real transitions (init-""→x is not counted).
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    stub._apply_flutter_mode_clamp("cruise", 0.0)  # 1st real transition
    stub._apply_flutter_mode_clamp("lead0", 0.0)   # 2nd real transition -> flutter on
    assert stub._flutter_mode_active
    # Push in negative direction; should also clamp.
    stub.output_a_target = -1.5
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    # prev was 0.0 from last call; max_step = 0.8 * 0.05 = 0.04 downward.
    assert stub.output_a_target == pytest.approx(-0.04, abs=1e-6)

  def test_same_source_goldilocks_twitch_is_not_hidden_by_flutter_clamp(self):
    stub = _make_flutter_stub(jerk_cap=0.8, n_trans=2.0, dt=0.05)

    for commanded_accel in (0.22, -0.18, 0.16, -0.12, 0.10):
      stub.output_a_target = commanded_accel
      stub._apply_flutter_mode_clamp("lead0", 0.0)
      assert stub.output_a_target == pytest.approx(commanded_accel)

    assert stub._flutter_mode_active is False
    assert len(stub._source_transition_frames) == 0

  def test_bypass_on_strong_model_decel(self):
    stub = _make_flutter_stub(jerk_cap=0.8, n_trans=2.0, bypass_decel=1.5, dt=0.05)
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    stub._apply_flutter_mode_clamp("cruise", 0.0)
    # In flutter mode; model commands hard brake.
    stub.output_a_target = -2.5
    stub._apply_flutter_mode_clamp("lead0", model_accel=-2.0)  # 2nd transition + big brake
    # Bypass means no clamp: output remains -2.5.
    assert stub.output_a_target == pytest.approx(-2.5)

  def test_disabled_when_jerk_cap_zero(self):
    stub = _make_flutter_stub(jerk_cap=0.0)
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    stub._apply_flutter_mode_clamp("cruise", 0.0)
    stub.output_a_target = 2.0
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    assert stub.output_a_target == pytest.approx(2.0)
