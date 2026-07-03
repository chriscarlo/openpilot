import types
from collections import deque
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner


def _make_stub(jerk_limit: float = 0.6, window_s: float = 1.5,
               dt: float = 0.05, a_max: float = 2.0,
               jerk_ramp: float | None = None) -> SimpleNamespace:
  # jerk_ramp=None omits the attr (legacy fallback path); a float exercises the
  # ramped-ceiling branch.
  cfg = SimpleNamespace(
    cruise_reacquire_pos_jerk_limit=jerk_limit,
    cruise_reacquire_jerk_window_s=window_s,
  )
  if jerk_ramp is not None:
    cfg.cruise_reacquire_jerk_ramp_mps3_per_s = jerk_ramp
  stub = SimpleNamespace(
    output_a_target=0.0,
    dt=dt,
    _planner_output_accel_limits=(-3.0, a_max),
    _prev_mpc_source="",
    _cruise_pos_jerk_frames_left=0,
    _cruise_pos_jerk_prev_a=0.0,
    # CD5 exit-cause classifier + collapse-holdback state (additive; these unit
    # tests exercise the positive-jerk clamp only, so published_lead is None and
    # the classifier stays in its fail-safe "departure" path — full ramp).
    _exit_lookback=deque(maxlen=32),
    _exit_max_prob_drop=0.0,
    _exit_prev_pub_prob=None,
    _exit_drel_dropout=False,
    _exit_peak_prob=0.0,
    _exit_last_status_true_prob=None,
    _exit_lead_track_id=-1,
    _exit_lead_last_drel=None,
    _last_lead_owned_track_id=-1,
    _last_lead_owned_drel=None,
    _reacquire_armed_pending=False,
    _reacquire_exit_cause="none",
    _collapse_holdback_frames_left=0,
    cruise_reacquire_debug={},
    mpc=SimpleNamespace(_live_tune_cfg=cfg),
  )
  stub._apply_cruise_reacquire_jerk_limit = types.MethodType(
    LongitudinalPlanner._apply_cruise_reacquire_jerk_limit, stub,
  )
  # _lead_owned_slot is a staticmethod: assign the plain function (no self bind).
  stub._lead_owned_slot = LongitudinalPlanner._lead_owned_slot
  stub._classify_exit_cause = types.MethodType(
    LongitudinalPlanner._classify_exit_cause, stub,
  )
  return stub


def _make_flutter_stub(jerk_cap: float = 0.8, n_trans: float = 2.0, window_s: float = 1.0,
                       bypass_decel: float = 1.5, dt: float = 0.05,
                       brake_jerk_cap: float | None = None) -> SimpleNamespace:
  # brake_jerk_cap=None omits the attr (legacy symmetric fallback path); a float
  # exercises the asymmetric downward-allowance branch.
  cfg = SimpleNamespace(
    flutter_detect_transitions=n_trans,
    flutter_detect_window_s=window_s,
    flutter_clamp_jerk_mps3=jerk_cap,
    flutter_clamp_bypass_decel_mps2=bypass_decel,
  )
  if brake_jerk_cap is not None:
    cfg.flutter_clamp_brake_jerk_mps3 = brake_jerk_cap
  stub = SimpleNamespace(
    output_a_target=0.0,
    dt=dt,
    _flutter_prev_source="",
    _source_transition_frames=deque(),
    _flutter_clamp_prev_a=0.0,
    _flutter_mode_active=False,
    mpc=SimpleNamespace(_live_tune_cfg=cfg),
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


# Shipped live-tune values for the ramped/asymmetric branches
# (common/params_keys.h defaults, mirrored by the device snapshot).
SHIPPED_REACQUIRE_JERK = 0.08
SHIPPED_REACQUIRE_WINDOW_S = 3.0
SHIPPED_REACQUIRE_RAMP = 0.8
SHIPPED_FLUTTER_JERK = 0.12
SHIPPED_FLUTTER_BRAKE_JERK = 1.5
DT = 0.05


def _run_reacquire_sequence(stub: SimpleNamespace, demand: float, frames: int) -> list[float]:
  outputs = []
  for _ in range(frames):
    stub.output_a_target = demand
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    outputs.append(stub.output_a_target)
  return outputs


class TestCruiseReacquireJerkRamp:
  def _armed_stub(self, jerk_ramp: float | None, a_max: float = 10.0) -> SimpleNamespace:
    stub = _make_stub(jerk_limit=SHIPPED_REACQUIRE_JERK, window_s=SHIPPED_REACQUIRE_WINDOW_S,
                      dt=DT, a_max=a_max, jerk_ramp=jerk_ramp)
    stub.output_a_target = -0.1  # pre-departure follow accel
    stub._apply_cruise_reacquire_jerk_limit("lead0")
    return stub

  def test_first_frame_allowance_equals_base_limit_exactly(self):
    # On the arming frame elapsed=0, so the ramp must contribute nothing: the
    # handoff is exactly as soft as the shipped fixed limit.
    stub = self._armed_stub(jerk_ramp=SHIPPED_REACQUIRE_RAMP)
    stub.output_a_target = 1.2
    stub._apply_cruise_reacquire_jerk_limit("cruise")
    assert stub.output_a_target == -0.1 + SHIPPED_REACQUIRE_JERK * DT

  def test_allowance_grows_monotonically_with_elapsed_frames(self):
    stub = self._armed_stub(jerk_ramp=SHIPPED_REACQUIRE_RAMP)
    window_frames = int(SHIPPED_REACQUIRE_WINDOW_S / DT)
    outputs = _run_reacquire_sequence(stub, demand=8.0, frames=window_frames)
    steps = [b - a for a, b in zip(outputs[:-1], outputs[1:], strict=True)]
    assert all(b > a for a, b in zip(steps[:-1], steps[1:], strict=True)), "per-frame allowance must escalate"
    # Each step matches (jerk_limit + ramp * elapsed) * dt for its frame.
    for frame_idx, step in enumerate(steps, start=1):
      expected = (SHIPPED_REACQUIRE_JERK + SHIPPED_REACQUIRE_RAMP * frame_idx * DT) * DT
      assert step == pytest.approx(expected, abs=1e-9)

  def test_ramp_zero_reproduces_legacy_fixed_ceiling_bit_for_bit(self):
    stub_ramp0 = self._armed_stub(jerk_ramp=0.0)
    stub_legacy = self._armed_stub(jerk_ramp=None)  # absent attr = legacy fallback
    demands = [1.2, 0.9, 1.5, -0.4, 1.1] * 12
    for demand in demands:
      stub_ramp0.output_a_target = demand
      stub_legacy.output_a_target = demand
      stub_ramp0._apply_cruise_reacquire_jerk_limit("cruise")
      stub_legacy._apply_cruise_reacquire_jerk_limit("cruise")
      assert stub_ramp0.output_a_target == stub_legacy.output_a_target
      assert stub_ramp0._cruise_pos_jerk_frames_left == stub_legacy._cruise_pos_jerk_frames_left

  def test_shipped_ramp_reaches_meaningful_accel_within_two_seconds(self):
    # The R7 repro bound: with the fixed 0.08 allowance the ceiling gains only
    # +0.24 m/s^2 across the whole 3 s window; the shipped ramp must clear
    # +0.25 m/s^2 within 2 s of the handoff.
    stub = self._armed_stub(jerk_ramp=SHIPPED_REACQUIRE_RAMP)
    outputs = _run_reacquire_sequence(stub, demand=2.0, frames=int(2.0 / DT))
    assert max(outputs) >= 0.25


class TestFlutterBrakeJerkAllowance:
  def _latched_stub(self, brake_jerk_cap: float | None) -> SimpleNamespace:
    stub = _make_flutter_stub(jerk_cap=SHIPPED_FLUTTER_JERK, n_trans=2.0, dt=DT,
                              brake_jerk_cap=brake_jerk_cap)
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    stub._apply_flutter_mode_clamp("cruise", 0.0)  # 1st real transition
    stub._apply_flutter_mode_clamp("lead0", 0.0)   # 2nd real transition -> flutter on
    assert stub._flutter_mode_active
    return stub

  def test_downward_step_uses_brake_allowance_at_shipped_defaults(self):
    stub = self._latched_stub(brake_jerk_cap=SHIPPED_FLUTTER_BRAKE_JERK)
    stub.output_a_target = -1.0  # moderate brake, above the -1.5 bypass
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    assert stub.output_a_target == pytest.approx(-SHIPPED_FLUTTER_BRAKE_JERK * DT, abs=1e-9)

  def test_upward_step_unchanged_by_brake_allowance(self):
    stub = self._latched_stub(brake_jerk_cap=SHIPPED_FLUTTER_BRAKE_JERK)
    stub.output_a_target = 1.0
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    assert stub.output_a_target == pytest.approx(SHIPPED_FLUTTER_JERK * DT, abs=1e-9)

  def test_brake_cap_zero_reproduces_symmetric_clamp(self):
    stub_zero = self._latched_stub(brake_jerk_cap=0.0)
    stub_legacy = self._latched_stub(brake_jerk_cap=None)  # absent attr = legacy fallback
    for stub in (stub_zero, stub_legacy):
      stub.output_a_target = -1.0
      stub._apply_flutter_mode_clamp("lead0", 0.0)
    assert stub_zero.output_a_target == pytest.approx(-SHIPPED_FLUTTER_JERK * DT, abs=1e-9)
    assert stub_zero.output_a_target == stub_legacy.output_a_target

  def test_brake_allowance_never_tighter_than_upward_cap(self):
    # Effective downward cap is max(brake, up): a brake cap below the upward cap
    # must not throttle braking harder than the symmetric clamp did.
    stub = self._latched_stub(brake_jerk_cap=0.05)
    stub.output_a_target = -1.0
    stub._apply_flutter_mode_clamp("lead0", 0.0)
    assert stub.output_a_target == pytest.approx(-SHIPPED_FLUTTER_JERK * DT, abs=1e-9)
