"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np
from dataclasses import dataclass

from opendbc.car import structs, DT_CTRL
from opendbc.car.interfaces import CarStateBase
from opendbc.car.hyundai.values import CarControllerParams, CAR, HyundaiFlags
from opendbc.sunnypilot.car import get_param
from opendbc.sunnypilot.car.hyundai.longitudinal.helpers import get_car_config, jerk_limited_integrator, ramp_update, \
                                                                LongitudinalTuningType


LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

COMFORT_BAND_VAL = 0.01

COMFORT_BAND_ACCEL_BP = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
COMFORT_BAND_DECEL_BP = [-3.5, -2.5, -1.5, -0.75, -0.25, -0.05]
COMFORT_BAND_V = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]

DYNAMIC_LOWER_JERK_BP = [-2.0, -1.5, -1.0, -0.25, -0.1, -0.025, -0.01, -0.005]
DYNAMIC_LOWER_JERK_V  = [3.3,  2.5,  2.0,   1.9,  1.8,   1.65,  1.15,    0.5]
BRAKE_TRACKING_GAP_JERK_SCALE_MPS2 = 1.4
FOLLOW_REQUIRED_DECEL_SCALE_MPS2 = 1.6
FOLLOW_LEAD_BRAKE_SCALE_MPS2 = 2.0
FOLLOW_DISTANCE_SCALE_M = 55.0
FOLLOW_PATH_SCALE_M = 0.9
FOLLOW_VLAT_SCALE_MPS = 1.5
FOLLOW_URGENCY_POWER = 1.5

SPEED_BP = [0.0, 5.0, 20.0]


@dataclass
class LongitudinalState:
  desired_accel: float = 0.0
  actual_accel: float = 0.0
  accel_last: float = 0.0
  jerk_upper: float = 0.0
  jerk_lower: float = 0.0
  comfort_band_upper: float = 0.0
  comfort_band_lower: float = 0.0
  stopping: bool = False


class LongitudinalController:
  """Longitudinal controller which gets injected into CarControllerParams."""

  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP) -> None:
    self.CP = CP
    self.CP_SP = CP_SP

    self.long_tuning_param = LongitudinalTuningType.OFF

    self.tuning = LongitudinalState()
    self.car_config = get_car_config(CP)
    self.long_control_state_last = LongCtrlState.off
    self.stopping_count = 0

    self.accel_cmd = 0.0
    self.desired_accel = 0.0
    self.actual_accel = 0.0
    self.accel_last = 0.0
    self.jerk_upper = 0.0
    self.jerk_lower = 0.0
    self.comfort_band_upper = 0.0
    self.comfort_band_lower = 0.0
    self.stopping = False

    self._last_tuning_params: tuple = ()
    self._tuning_params_dict: dict[str, str] = {}
    self._param_update_counter = 0

  def _get_tuning_params_dict(self, CC_SP) -> None:
    """Update car config when tuning parameters change."""
    params_list = CC_SP.params
    self.long_tuning_param = int(get_param(params_list, "HyundaiLongitudinalTuning", str(LongitudinalTuningType.OFF)))
    if self.long_tuning_param != LongitudinalTuningType.OFF:
      tuning_values = tuple(getattr(p, 'value', '') for p in params_list if getattr(p, 'key', '').startswith('LongTuning'))
      if tuning_values != self._last_tuning_params:
        self._last_tuning_params = tuning_values
        self._tuning_params_dict = {p.key: p.value for p in params_list if p.key.startswith('LongTuning')}
        self.car_config = get_car_config(self.CP, self._tuning_params_dict)

  def _update_tuning_params(self, CC_SP) -> None:
    """Update tuning parameters every 3 seconds."""
    if self._param_update_counter % int(3.0 / (DT_CTRL * 2)) == 0:
      self._get_tuning_params_dict(CC_SP)
    self._param_update_counter = (self._param_update_counter + 1) % 1000000

  @property
  def enabled(self) -> bool:
    return self.long_tuning_param != LongitudinalTuningType.OFF

  def fcw(self, CC: structs.CarControl) -> bool:
    return bool(CC.hudControl.visualAlert == VisualAlert.fcw)

  def get_stopping_state(self, actuators: structs.CarControl.Actuators) -> None:
    stopping = actuators.longControlState == LongCtrlState.stopping

    # If custom tuning is not enabled, use upstream stopping logic
    if not self.enabled:
      self.stopping = stopping
      self.stopping_count = 0
      return

    # Reset stopping state when not in stopping mode
    if not stopping:
      self.stopping = False
      self.stopping_count = 0
      return

    # When transitioning from off state to stopping
    if self.long_control_state_last == LongCtrlState.off:
      self.stopping = True
      return

    # Keep track of time in stopping state (in control cycles)
    if self.CP.carFingerprint == CAR.KIA_NIRO_EV:
      if self.stopping_count > 1.0 / DT_CTRL:
        self.stopping = True
    else:
      if self.stopping_count > 1 / (DT_CTRL * 2):
        self.stopping = True

    self.stopping_count += 1

  def _calculate_speed_based_jerk_limits(self, velocity: float, long_control_state: LongCtrlState) -> tuple[float, float]:
    """Calculate jerk limits based on vehicle speed according to ISO 15622:2018.

    Args:
        velocity: Current vehicle speed (m/s)
        long_control_state: Current longitudinal control state

    Returns:
        Tuple of (upper_limit, lower_limit) in m/s³
    """

    # Upper jerk limit varies based on speed and control state
    if long_control_state == LongCtrlState.pid:
      upper_limit = float(np.interp(velocity, SPEED_BP, self.car_config.upper_jerk_v))
    else:
      upper_limit = 0.5  # Default for non-PID states

    # Lower jerk limit varies based on speed
    lower_limit = float(np.interp(velocity, SPEED_BP, self.car_config.lower_jerk_v))

    return upper_limit, lower_limit

  def _calculate_lookahead_jerk(self, accel_error: float, velocity: float) -> tuple[float, float]:
    """Calculate lookahead jerk needed to reach target acceleration.

    Args:
        accel_error: Difference between target and current acceleration (m/s²)
        velocity: Current vehicle speed (m/s)

    Returns:
        Tuple of (upper_jerk, lower_jerk) in m/s³
    """

    # Time window to reach target acceleration, varies with speed
    future_t_upper = float(np.interp(velocity, self.car_config.lookahead_jerk_bp, self.car_config.lookahead_jerk_upper_v))
    future_t_lower = float(np.interp(velocity, self.car_config.lookahead_jerk_bp, self.car_config.lookahead_jerk_lower_v))

    # Required jerk to reach target acceleration in lookahead window
    j_ego_upper = accel_error / future_t_upper
    j_ego_lower = accel_error / future_t_lower

    return j_ego_upper, j_ego_lower

  def _calculate_dynamic_lower_jerk(self, accel_error: float, velocity: float) -> float:
    """Calculate dynamic jerk for braking based on acceleration error.

    Used for the dynamic tuning approach (non-predictive).

    Args:
        accel_error: Difference between actual and previous acceleration (m/s²)
        velocity: Current vehicle speed (m/s)

    Returns:
        Dynamic lower jerk limit (m/s³)
    """

    if accel_error < 0:
      # Scale the brake jerk values based on car config
      if self.CP.radarUnavailable:
        lower_max = 5.0
      else:
        lower_max = self.car_config.jerk_limits
      original_values = np.array(DYNAMIC_LOWER_JERK_V)
      scaled_values = original_values * (lower_max / original_values[0])

      # Interpolate based on acceleration error
      dynamic_lower_jerk = float(np.interp(accel_error, DYNAMIC_LOWER_JERK_BP, scaled_values))
    else:
      dynamic_lower_jerk = 0.5

    return dynamic_lower_jerk

  @staticmethod
  def _lead_field(lead, field_name: str, default):
    if lead is None:
      return default
    if isinstance(lead, dict):
      return lead.get(field_name, default)
    return getattr(lead, field_name, default)

  def _calculate_follow_urgency(self, CC_SP: structs.CarControlSP) -> float:
    """Continuously score how much the current brake request is tied to a centered slowing lead."""
    best_urgency = 0.0
    for slot_name in ("leadOne", "leadTwo"):
      lead = getattr(CC_SP, slot_name, None)
      if not bool(self._lead_field(lead, "status", False)):
        continue

      d_rel = max(0.0, float(self._lead_field(lead, "dRel", 0.0)))
      v_rel = float(self._lead_field(lead, "vRel", 0.0))
      a_lead_k = float(self._lead_field(lead, "aLeadK", 0.0))
      d_path = abs(float(self._lead_field(lead, "dPath", self._lead_field(lead, "yRel", 0.0))))
      v_lat = abs(float(self._lead_field(lead, "vLat", 0.0)))

      closing_speed = max(0.0, -v_rel)
      required_decel = (closing_speed ** 2) / max(2.0 * max(d_rel, 1.0), 1.0)
      closing_urgency = 1.0 - np.exp(-required_decel / FOLLOW_REQUIRED_DECEL_SCALE_MPS2)

      lead_brake_urgency = 1.0 - np.exp(-max(0.0, -a_lead_k) / FOLLOW_LEAD_BRAKE_SCALE_MPS2)
      closeness_weight = 1.0 / (1.0 + (d_rel / FOLLOW_DISTANCE_SCALE_M))
      path_weight = np.exp(-((d_path / FOLLOW_PATH_SCALE_M) ** 2)) * np.exp(-((v_lat / FOLLOW_VLAT_SCALE_MPS) ** 2))

      closing_term = closing_urgency * closeness_weight
      brake_term = lead_brake_urgency * closeness_weight
      slot_urgency = float(path_weight * (1.0 - ((1.0 - closing_term) * (1.0 - brake_term))))
      best_urgency = max(best_urgency, slot_urgency)

    return best_urgency

  def _calculate_brake_tracking_lower_jerk(self, CC_SP: structs.CarControlSP) -> float:
    """Raise brake-entry jerk only when the controller lags a centered follow-braking demand."""
    lower_max = 5.0 if self.CP.radarUnavailable else self.car_config.jerk_limits
    brake_tracking_gap = max(0.0, self.accel_last - self.accel_cmd)
    tracking_urgency = 1.0 - np.exp(-brake_tracking_gap / BRAKE_TRACKING_GAP_JERK_SCALE_MPS2)
    follow_urgency = self._calculate_follow_urgency(CC_SP)
    urgency = tracking_urgency * (follow_urgency ** FOLLOW_URGENCY_POWER)
    return self.car_config.min_lower_jerk + (lower_max - self.car_config.min_lower_jerk) * urgency

  def calculate_jerk(self, CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: CarStateBase, long_control_state: LongCtrlState) -> None:
    """Calculate appropriate jerk limits for smooth acceleration/deceleration.

    Args:
        CC: Car control signals
        CS: Car state
        long_control_state: Current longitudinal control state
    """

    # If custom tuning is disabled, use upstream fixed values
    if not self.enabled:
      if self.CP.flags & HyundaiFlags.CANFD:
        self.jerk_lower = 5.0 if CC.enabled else 1.0
        self.jerk_upper = 3.0
      else:
        self.jerk_upper = 3.0 if long_control_state == LongCtrlState.pid else 1.0
        self.jerk_lower = 5.0
      return

    velocity = CS.out.vEgo
    accel_error = self.accel_cmd - self.accel_last

    # Calculate jerk limits based on speed
    upper_speed_factor, lower_speed_factor = self._calculate_speed_based_jerk_limits(velocity, long_control_state)

    # Calculate lookahead jerk
    j_ego_upper, j_ego_lower = self._calculate_lookahead_jerk(accel_error, velocity)

    # Calculate lower jerk limit
    lower_jerk = max(-j_ego_lower, self.car_config.min_lower_jerk)

    # Final jerk limits with thresholds
    desired_jerk_upper = min(max(j_ego_upper, self.car_config.min_upper_jerk), upper_speed_factor)
    desired_jerk_lower = min(lower_jerk, lower_speed_factor)

    # Calculate dynamic lower jerk for non-predictive tuning
    a_ego_blended = float(np.interp(velocity, [1.0, 2.0], [CS.aBasis, CS.out.aEgo]))
    dynamic_accel_error = a_ego_blended - self.accel_last
    dynamic_lower_jerk = self._calculate_dynamic_lower_jerk(dynamic_accel_error, velocity)
    brake_tracking_lower_jerk = self._calculate_brake_tracking_lower_jerk(CC_SP)
    dynamic_desired_lower_jerk = max(self.car_config.min_lower_jerk,
                                     min(max(dynamic_lower_jerk, brake_tracking_lower_jerk), lower_speed_factor))

    # Apply jerk limits based on tuning approach
    self.jerk_upper = ramp_update(self.jerk_upper, desired_jerk_upper, self.car_config.min_upper_jerk)

    # Predictive tuning uses calculated desired jerk directly
    # Dynamic tuning applies a ramped approach for smoother transitions
    if self.long_tuning_param == LongitudinalTuningType.PREDICTIVE:
      self.jerk_lower = desired_jerk_lower
    elif self.long_tuning_param == LongitudinalTuningType.DYNAMIC:
      self.jerk_lower = ramp_update(self.jerk_lower, dynamic_desired_lower_jerk, self.car_config.min_lower_jerk)

    # Disable jerk when longitudinal control is inactive
    if not CC.longActive:
      self.jerk_upper = 0.0
      self.jerk_lower = 0.0

  def calculate_accel(self, CC: structs.CarControl) -> None:
    """Calculate commanded acceleration using jerk-limited approach.

    Args:
        CC: Car control signals
    """

    # Skip custom processing if tuning is disabled or radar unavailable
    if not self.enabled or self.CP.radarUnavailable:
      if not CC.longActive:
        self.desired_accel = 0.0
        self.actual_accel = 0.0
        self.accel_last = 0.0
        return
      self.desired_accel = self.accel_cmd
      # No-radar AI lead: low-pass to damp EV micro-flutter.
      # alpha=0.12 at 100 Hz → ~2 Hz corner, attenuates 10+ Hz jitter by ~10x.
      # 0.019 m/s² planner jitter → ~0.002 m/s² at CAN (below 1 LSB = 0.01).
      # No deadband — it blocks gentle gap-closing accels and causes
      # the car to follow further back than the target headway.
      if self.CP.radarUnavailable:
        self.actual_accel = float(0.12 * self.accel_cmd + 0.88 * self.accel_last)
      else:
        self.actual_accel = self.accel_cmd
      self.accel_last = self.actual_accel
      return

    # Reset acceleration when control is inactive
    if not CC.longActive:
      self.desired_accel = 0.0
      self.actual_accel = 0.0
      self.accel_last = 0.0
      return

    # Force zero acceleration during stopping
    if self.stopping:
      self.desired_accel = 0.0
    else:
      self.desired_accel = float(np.clip(self.accel_cmd, self.car_config.accel_min, self.car_config.accel_max))

    # Apply jerk-limited integration to get smooth acceleration
    self.actual_accel = jerk_limited_integrator(self.desired_accel, self.accel_last, self.jerk_upper, self.jerk_lower)

    self.accel_last = self.actual_accel

  def calculate_comfort_band(self, CC: structs.CarControl, CS: CarStateBase) -> None:
    if not self.enabled or self.CP.radarUnavailable or not CC.longActive:
      self.comfort_band_upper = 0.0
      self.comfort_band_lower = 0.0
      return

    accel = CS.out.aEgo
    self.comfort_band_upper = float(np.interp(accel, COMFORT_BAND_ACCEL_BP, COMFORT_BAND_V))
    self.comfort_band_lower = float(np.interp(accel, COMFORT_BAND_DECEL_BP, COMFORT_BAND_V[::-1]))

  def get_tuning_state(self) -> None:
    """Update the tuning state object with current control values.

    External components depend on this state for longitudinal control.
    """

    self.tuning = LongitudinalState(
      desired_accel=self.desired_accel,
      actual_accel=self.actual_accel,
      accel_last=self.accel_last,
      jerk_upper=self.jerk_upper,
      jerk_lower=self.jerk_lower,
      comfort_band_upper=self.comfort_band_upper,
      comfort_band_lower=self.comfort_band_lower,
      stopping=self.stopping,
    )

  def emergency_control(self, CC: structs.CarControl) -> None:
    """Handle FCW situations with emergency braking jerk allowed."""
    if not CC.longActive:
      self.actual_accel = 0.0
      self.accel_last = 0.0
      self.comfort_band_upper = 0.0
      self.comfort_band_lower = 0.0
      self.desired_accel = 0.0
      self.jerk_upper = 0.0
      self.jerk_lower = 0.0
      return

    self.comfort_band_upper = 0.0
    self.comfort_band_lower = 0.0
    accel = max(CarControllerParams.ACCEL_MIN, self.car_config.accel_min)
    self.desired_accel = accel
    self.actual_accel = accel
    self.accel_last = self.actual_accel
    self.jerk_upper = 0.5
    self.jerk_lower = 8.0

  def update(self, CC: structs.CarControl, CC_SP: structs.CarControlSP, CS: CarStateBase) -> None:
    """Update longitudinal control calculations.

    This is the main entry point called externally.

    Args:
        CC: Car control signals including actuators
        CC_SP: sunnypilot car control signals including longitudinal tuning parameters and flags
        CS: Car state information
    """
    actuators = CC.actuators
    long_control_state = actuators.longControlState
    self.accel_cmd = CC.actuators.accel

    self._update_tuning_params(CC_SP)
    self.get_stopping_state(actuators)

    if self.fcw(CC):
      self.emergency_control(CC)
    else:
      self.calculate_jerk(CC, CC_SP, CS, long_control_state)
      self.calculate_accel(CC)
      self.calculate_comfort_band(CC, CS)

    self.get_tuning_state()
    self.long_control_state_last = long_control_state
