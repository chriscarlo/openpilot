import math
import time

import numpy as np
from cereal import car
from opendbc.car.hyundai.values import CAR
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LEAD_RESPONSE_TUNE_SPECS_BY_ATTR
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

_STOPPING_RELEASE_JERK_SPEC = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["stopping_release_jerk_mps3"]

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, active, long_control_state, v_ego,
                             should_stop, brake_pressed, cruise_standstill):
  stopping_condition = should_stop
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        else:
          long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state

class LongControl:
  LIVE_TUNE_REFRESH_DT_S = 0.50

  def __init__(self, CP, params=None, time_fn=None):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             k_f=CP.longitudinalTuning.kf, rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.stopping_release_active = False
    self._ev6_stopping_release_enabled = (
      getattr(CP, "brand", None) == "hyundai" and
      getattr(CP, "carFingerprint", None) == CAR.KIA_EV6
    )
    self._live_tune_params = params
    self._time_fn = time.monotonic if time_fn is None else time_fn
    self._last_live_tune_refresh_t = float("-inf")
    self.stopping_release_jerk_mps3 = _STOPPING_RELEASE_JERK_SPEC.default
    self._refresh_live_tune(force=True)

  def reset(self):
    self.pid.reset()

  def _refresh_live_tune(self, force: bool = False) -> None:
    now = self._time_fn()
    if not force and (now - self._last_live_tune_refresh_t) < self.LIVE_TUNE_REFRESH_DT_S:
      return
    self._last_live_tune_refresh_t = now
    if self._live_tune_params is None:
      self.stopping_release_jerk_mps3 = _STOPPING_RELEASE_JERK_SPEC.default
      return

    try:
      raw = self._live_tune_params.get(_STOPPING_RELEASE_JERK_SPEC.key)
      value = _STOPPING_RELEASE_JERK_SPEC.default if raw is None else float(raw)
      if not math.isfinite(value):
        value = _STOPPING_RELEASE_JERK_SPEC.default
    except Exception:
      value = _STOPPING_RELEASE_JERK_SPEC.default
    self.stopping_release_jerk_mps3 = _STOPPING_RELEASE_JERK_SPEC.clamp(value)

  def update(self, active, CS, a_target, should_stop, accel_limits):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self._refresh_live_tune()
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    previous_state = self.long_control_state
    self.long_control_state = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill)
    if (self._ev6_stopping_release_enabled
        and previous_state == LongCtrlState.stopping
        and self.long_control_state in (LongCtrlState.starting, LongCtrlState.pid)
        and CS.vEgo > self.CP.vEgoStarting):
      self.stopping_release_active = True
    elif self.long_control_state in (LongCtrlState.off, LongCtrlState.stopping):
      self.stopping_release_active = False

    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        output_accel -= self.CP.stoppingDecelRate * DT_CTRL
      self.reset()

    elif self.long_control_state == LongCtrlState.starting:
      # Never command LESS than the plan while starting: on a launch behind a
      # departing lead the planner's launch-follow floor can exceed the flat
      # startAccel, and holding the smaller value through the standstill
      # actuation dead zone (EV6: ~0.85 s of brake bleed + torque build) is
      # part of what let the road 205-13 launch fall behind. With the launch
      # floor disabled (its 0.0 sentinel) a_target sits below startAccel here
      # and this reduces to the legacy constant exactly.
      output_accel = max(self.CP.startAccel, a_target)
      self.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo,
                                     feedforward=a_target)

    # If the rolling release reaches the launch threshold (or the cruise stack
    # declares standstill), hand control back to the ordinary starting state.
    # Continuing to meter out negative acceleration here would create the very
    # stop this comfort guard is meant to avoid and delay a departing lead.
    if self.stopping_release_active and (CS.vEgo <= self.CP.vEgoStarting or CS.cruiseState.standstill):
      self.stopping_release_active = False

    stopping_release_target_met = False
    if self.stopping_release_active:
      # Bound only upward release. A new/downward braking request passes in the
      # same tick, but it does not erase the rollout episode: a later positive
      # PID request remains bounded until the actual command reaches coast.
      release_upper = self.last_output_accel + self.stopping_release_jerk_mps3 * DT_CTRL
      if output_accel > release_upper:
        output_accel = release_upper
      else:
        # End only after an unconstrained nonnegative request actually fits
        # inside the bound. Merely crossing zero on a still-limited ramp would
        # reintroduce a one-tick coast-to-accel jump on the following cycle.
        stopping_release_target_met = output_accel >= 0.0

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    if self.stopping_release_active and stopping_release_target_met and self.last_output_accel >= 0.0:
      self.stopping_release_active = False
    return self.last_output_accel
