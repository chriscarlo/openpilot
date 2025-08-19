import numpy as np
import time
import math

from cereal import custom
from openpilot.common.params import Params
from openpilot.common.conversions import Conversions as CV
from openpilot.common.numpy_fast import clip
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.modeld.constants import ModelConstants

VisionTurnControllerState = custom.LongitudinalPlanSP.VisionTurnSpeedControl.VisionTurnSpeedControlState

N_POINTS = int(min(33, len(ModelConstants.T_IDXS)))  # Use available trajectory points


_MIN_V = 5.6  # Do not operate under 20km/h

_ENTERING_PRED_LAT_ACC_TH = 1.3  # Predicted Lat Acc threshold to trigger entering turn state.
_ABORT_ENTERING_PRED_LAT_ACC_TH = 1.1  # Predicted Lat Acc threshold to abort entering state if speed drops.

_TURNING_LAT_ACC_TH = 1.6  # Lat Acc threshold to trigger turning turn state.

_LEAVING_LAT_ACC_TH = 1.3  # Lat Acc threshold to trigger leaving turn state.
_FINISH_LAT_ACC_TH = 1.1  # Lat Acc threshold to trigger end of turn cycle.


_NO_OVERSHOOT_TIME_HORIZON = 4.  # s. Time to use for velocity desired based on a_target when not overshooting.

# Lookup table for the minimum smooth deceleration during the ENTERING state
# depending on the actual maximum absolute lateral acceleration predicted on the turn ahead.
_ENTERING_SMOOTH_DECEL_V = [-0.2, -1.]  # min decel value allowed on ENTERING state
_ENTERING_SMOOTH_DECEL_BP = [1.3, 3.]  # absolute value of lat acc ahead

# Lookup table for the acceleration for the TURNING state
# depending on the current lateral acceleration of the vehicle.
_TURNING_ACC_V = [0.5, 0., -0.4]  # acc value
_TURNING_ACC_BP = [1.5, 2.3, 3.]  # absolute value of current lat acc

_LEAVING_ACC = 0.5  # Confortble acceleration to regain speed while leaving a turn.


_DEBUG = False


# Advanced vision-based functions extracted from chauffeur_vtsc.py

# Constants for advanced curvature-based speed calculation
CURV_CORR_FACTOR = (CV.MS_TO_MPH ** 2)  # Correction factor for lat accel function
MAX_SPEED_DEFAULT = 70.0  # m/s, fallback for straight roads
SPEED_INCREASE_FACTOR = 1.0

def _original_curvature_based_lat_accel(abs_curvature_scaled: float) -> float:
    """Internal function replicating the tuned lateral accel logic."""
    high_accel = 3.12
    low_accel = 1.5
    span = high_accel - low_accel
    center_curvature = 0.060
    k = 75
    reduction = span / (1.0 + math.exp(-k * (abs_curvature_scaled - center_curvature)))
    lat_acc = high_accel - reduction
    return clip(lat_acc, low_accel, high_accel)

def _physics_based_lateral_acceleration(curvature: float) -> float:
    """
    FIXED sports car sigmoid for curvature-based lateral acceleration
    NO SCALING HACKS - Takes curvature in SI units (1/meters)
    
    Returns LOWER lateral acceleration for HIGHER curvature (inverse relationship)
    This ensures sharp curves get appropriate low speeds while maintaining safety.
    """
    # Physical constants - no arbitrary scaling
    a_max = 3.12   # Maximum lateral acceleration (m/s²) - safety limit
    a_min = 1.2    # Minimum lateral acceleration (m/s²) - tight curve limit
    alpha = 24.3   # Decay rate - tuned for sports car performance
    beta = 0.78    # Power law exponent - tuned for sports car performance

    # Power law sigmoid: exponential decay for smooth inverse relationship
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))

    # Safety bounds (function naturally stays within bounds)
    return max(a_min, min(lateral_acceleration, a_max))

def curvature_to_speed(abs_curvature_meters: float) -> float:
    """FIXED: Calculates target speed (m/s) directly from curvature with NO SCALING HACK"""
    if abs_curvature_meters < 1e-7:  # Handle straight roads
        return MAX_SPEED_DEFAULT

    # Get safe lateral acceleration using FIXED sigmoid (NO SCALING!)
    safe_lat_accel = _physics_based_lateral_acceleration(abs_curvature_meters)

    # Calculate speed using physics formula v = sqrt(a / k) with CONSISTENT curvature
    try:
        base_speed_mps = math.sqrt(safe_lat_accel / abs_curvature_meters)
    except (ValueError, ZeroDivisionError):
        base_speed_mps = 0.0

    # Apply speed increase factor and clip to reasonable maximum
    target_speed_mps = base_speed_mps * SPEED_INCREASE_FACTOR
    return clip(target_speed_mps, 0.0, MAX_SPEED_DEFAULT)

def dynamic_decel_scale(v_ego_ms: float) -> float:
    """Dynamic deceleration scaling based on speed."""
    min_speed = 3.0
    max_speed = 35.0
    if v_ego_ms <= min_speed:
        scale = 9.0
    elif v_ego_ms >= max_speed:
        scale = 2.0
    else:
        ratio = (v_ego_ms - min_speed) / (max_speed - min_speed)
        scale = 9.0 + (2.0 - 9.0) * ratio
    return min(scale, 3.0)

def find_apexes(curv_array: np.ndarray, threshold: float = 5e-5) -> list:
    """Identify indices where curvature spikes above threshold and is a local maximum."""
    apex_indices = []
    for i in range(1, len(curv_array) - 1):
        if (curv_array[i] > threshold and
            curv_array[i] >= curv_array[i + 1] and
            curv_array[i] > curv_array[i - 1]):
            apex_indices.append(i)
    return apex_indices

def nonlinear_lat_accel(v_ego_ms: float, turn_aggressiveness: float = 1.0) -> float:
    """Compute lateral acceleration limit based on speed and aggressiveness."""
    v_ego_mph = v_ego_ms * CV.MS_TO_MPH
    base = 1.5
    span = 2.18
    center = 25.0
    k = 0.10
    lat_acc = base + span / (1.0 + math.exp(-k * (v_ego_mph - center)))
    return lat_acc * turn_aggressiveness

def margin_time_fn(v_ego_ms: float) -> float:
    """Returns a 'margin time' used in backward-pass speed planning."""
    v_low = 0.0
    t_low = 1.0
    v_med = 15.0     # ~34 mph
    t_med = 3.0
    v_high = 31.3    # ~70 mph
    t_high = 5.0

    if v_ego_ms <= v_low:
        return t_low
    elif v_ego_ms >= v_high:
        return t_high
    elif v_ego_ms <= v_med:
        ratio = (v_ego_ms - v_low) / (v_med - v_low)
        return t_low + ratio * (t_med - t_low)
    else:
        ratio = (v_ego_ms - v_med) / (v_high - v_med)
        return t_med + ratio * (t_high - t_med)

def calculate_anticipation_time(v_ego_ms: float, target_speed_ms: float, max_pred_lat_acc: float) -> float:
    """Calculate how early (in seconds) to reach target speed before apex.
    
    This creates a more comfortable deceleration profile by reaching the target
    speed before it's strictly necessary, rather than "just in time".
    
    Returns a time between 1-3 seconds based on:
    - Current speed (higher speeds need more anticipation)
    - Speed reduction required (larger reductions need more time)
    - Curve severity (sharper curves need more anticipation)
    """
    # Base anticipation time
    base_time = 1.5  # Base 1.5 seconds early

    # Speed factor: Higher speeds need more anticipation
    # Normalize around 20 m/s (~45 mph)
    speed_factor = clip(v_ego_ms / 20.0, 0.7, 1.5)

    # Speed reduction factor: Larger speed changes need more anticipation
    if v_ego_ms > 0.1:  # Avoid division by zero
        delta_ratio = (v_ego_ms - target_speed_ms) / v_ego_ms
        delta_factor = clip(1.0 + delta_ratio * 0.5, 1.0, 1.5)
    else:
        delta_factor = 1.0

    # Curve severity factor: Sharper curves need more anticipation
    # Normalize around 1.5 m/s² lateral acceleration
    severity_factor = clip(max_pred_lat_acc / 1.5, 0.8, 1.3)

    # Calculate total anticipation time
    anticipation_time = base_time * speed_factor * delta_factor * severity_factor

    # Clip to reasonable range (1-3 seconds)
    return clip(anticipation_time, 1.0, 3.0)

def _debug(msg):
  if not _DEBUG:
    return
  print(msg)




def _description_for_state(turn_controller_state):
  if turn_controller_state == VisionTurnControllerState.disabled:
    return 'DISABLED'
  if turn_controller_state == VisionTurnControllerState.entering:
    return 'ENTERING'
  if turn_controller_state == VisionTurnControllerState.turning:
    return 'TURNING'
  if turn_controller_state == VisionTurnControllerState.leaving:
    return 'LEAVING'


class VisionTurnController:
  def __init__(self, CP):
    self._params = Params()
    self._CP = CP
    self._op_enabled = False
    self._gas_pressed = False
    self._is_enabled = self._params.get_bool("VisionTurnSpeedControl")
    self._last_params_update = 0.
    self._v_cruise_setpoint = 0.
    self._v_ego = 0.
    self._a_ego = 0.
    self._a_target = 0.
    self._v_overshoot = 0.
    self._state = VisionTurnControllerState.disabled

    # Advanced controller state
    self._planned_speeds = np.zeros(N_POINTS, dtype=float)
    self._current_accel = 0.0
    self._prev_target_speed = 0.0
    self._max_decel = 3.5
    self._max_jerk = 6.0
    self._max_accel = 1.3 * self._max_decel
    self._max_jerk_accel = 2.0 * self._max_jerk

    # EMA filtering for curvature
    self._curvature_ema_ratio = 0.3
    self._filtered_curvature = 0.0

    # Anticipatory deceleration state
    self._is_decelerating_for_curve = False
    self._anticipation_start_time = 0.0
    self._curve_detection_distance = 0.0

    self._reset()

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self, value):
    if value != self._state:
      _debug(f'TVC: TurnVisionController state: {_description_for_state(value)}')
      if value == VisionTurnControllerState.disabled:
        self._reset()
    self._state = value

  @property
  def a_target(self):
    return self._a_target if self.is_active else self._a_ego

  @property
  def v_turn(self):
    # Key change: Don't let state machine disable speed planning
    # Always use advanced planning if we have curvature data, regardless of state

    # Only use cruise setpoint if we truly have no curvature data
    if self._filtered_curvature < 1e-7 and self._max_pred_lat_acc < 0.5:
      return self._v_cruise_setpoint

    # Use advanced controller's target speed when we have meaningful curvature
    if self._lat_acc_overshoot_ahead:
      return self._v_overshoot
    else:
      # Return the planned target speed from advanced controller
      # This allows apex acceleration even when state machine shows "disabled"
      return self._prev_target_speed

  @property
  def current_lat_acc(self):
    return self._current_lat_acc

  @property
  def max_pred_lat_acc(self):
    return self._max_pred_lat_acc

  @property
  def is_active(self):
    # Key change: Active for planning purposes if we have meaningful curvature data,
    # regardless of state machine state (which is only for UI/logging)
    has_meaningful_curvature = (self._filtered_curvature > 1e-7 or self._max_pred_lat_acc > 0.5)
    return has_meaningful_curvature or (self._state != VisionTurnControllerState.disabled)

  @property
  def is_entering(self):
    return self._state == VisionTurnControllerState.entering

  @property
  def is_turning(self):
    return self._state == VisionTurnControllerState.turning

  @property
  def is_leaving(self):
    return self._state == VisionTurnControllerState.leaving

  @property
  def distance(self):
    """Distance to lateral acceleration overshoot point."""
    return self._v_overshoot_distance if hasattr(self, '_v_overshoot_distance') else 200.0

  def getCurrentLateralAccel(self):
    """Return current lateral acceleration for HUD display."""
    return self._current_lat_acc

  def _reset(self):
    self._current_lat_acc = 0.
    self._max_v_for_current_curvature = 0.
    self._max_pred_lat_acc = 0.
    self._v_overshoot_distance = 200.
    self._lat_acc_overshoot_ahead = False

    # Reset advanced controller state
    self._planned_speeds[:] = self._v_ego if hasattr(self, '_v_ego') else 0.0
    self._current_accel = 0.0
    self._prev_target_speed = self._v_ego if hasattr(self, '_v_ego') else 0.0
    self._filtered_curvature = 0.0


    # Reset anticipatory deceleration state
    self._is_decelerating_for_curve = False
    self._anticipation_start_time = 0.0
    self._curve_detection_distance = 0.0
  def _update_params(self):
    tm = time.time()
    if tm > self._last_params_update + 5.0:
      self._is_enabled = self._params.get_bool("VisionTurnSpeedControl")
      self._last_params_update = tm

  def _update_calculations(self, sm):
    """Advanced vision-based curvature calculation using direct model outputs."""
    model_data = sm['modelV2'] if sm.valid.get('modelV2', False) else None

    # Initialize defaults for edge cases
    current_curvature_signed = 0.0
    current_curvature = 0.0
    max_pred_curvature = 0.0

    # Use advanced method: direct model data access
    if (model_data is not None and
        hasattr(model_data, 'orientationRate') and hasattr(model_data, 'velocity') and
        model_data.orientationRate.z is not None and model_data.velocity.x is not None):

      orientation_rate_raw = model_data.orientationRate.z
      velocity_pred_raw = model_data.velocity.x

      MIN_POINTS = 3
      if (len(orientation_rate_raw) >= MIN_POINTS and len(velocity_pred_raw) >= MIN_POINTS):
        # Use direct model outputs for curvature calculation
        n_points = int(min(len(orientation_rate_raw), len(velocity_pred_raw), N_POINTS))
        # Ensure n_points is a pure Python int for Cap'n Proto compatibility
        n_points = int(n_points)
        orientation_rate = np.abs(np.array(list(orientation_rate_raw)[:n_points], dtype=float))
        velocity_pred = np.array(list(velocity_pred_raw)[:n_points], dtype=float)

        # Compute curvature array: curvature = orientation_rate / velocity
        eps = 1e-9
        curvature_array = orientation_rate / np.clip(velocity_pred, eps, None)
        max_pred_curvature = float(np.max(curvature_array))

        # Calculate lateral acceleration using model-predicted curvature
        # This is more accurate than steering angle at highway speeds
        # Use the current model-predicted curvature instead of steering angle
        if len(curvature_array) > 0:
          current_curvature = float(curvature_array[0])  # Use model's current prediction
          current_curvature_signed = current_curvature  # Preserve sign from orientation rate sign
          # Get the original signed orientation rate to preserve direction
          orientation_rate_signed = model_data.orientationRate.z[0] if len(model_data.orientationRate.z) > 0 else 0.0
          if orientation_rate_signed < 0:
            current_curvature_signed = -current_curvature

        # Update filtered curvature using EMA
        self._filtered_curvature = ((1 - self._curvature_ema_ratio) * self._filtered_curvature +
                                   self._curvature_ema_ratio * max_pred_curvature)

        # Calculate lateral accelerations using model predictions (not steering angle)
        self._current_lat_acc = current_curvature_signed * self._v_ego**2
        self._max_pred_lat_acc = self._v_ego**2 * max_pred_curvature

        # Calculate safe speed using advanced physics-based method
        self._max_v_for_current_curvature = curvature_to_speed(current_curvature) if current_curvature > 0 else V_CRUISE_MAX * CV.KPH_TO_MS

        # Check for overshoot using curvature_to_speed method
        safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in curvature_array])
        overshoot_mask = safe_speeds < self._v_ego
        self._lat_acc_overshoot_ahead = np.any(overshoot_mask)

        if self._lat_acc_overshoot_ahead:
          overshoot_idx = np.where(overshoot_mask)[0][0]
          self._v_overshoot = min(safe_speeds[overshoot_idx], self._v_cruise_setpoint)
          # Estimate distance using time indices and current velocity
          times = np.array(ModelConstants.T_IDXS[:n_points])
          self._v_overshoot_distance = max(times[overshoot_idx] * self._v_ego, 10.0)
          # Calculate anticipation time for early deceleration
          anticipation_time = calculate_anticipation_time(
              self._v_ego,
              self._v_overshoot,
              max_pred_curvature * self._v_ego**2
          )

          # Adjust the overshoot distance to start deceleration earlier
          # This makes us reach target speed BEFORE the apex
          anticipation_distance = anticipation_time * self._v_ego
          self._v_overshoot_distance = max(self._v_overshoot_distance - anticipation_distance, 10.0)

          _debug(f'TVC: Advanced High LatAcc. Dist: {self._v_overshoot_distance:.2f}, v: {self._v_overshoot * CV.MS_TO_KPH:.2f}, anticipation: {anticipation_time:.1f}s')

        return  # Successfully processed vision data

    # If model data is not available, use safe defaults
    _debug('TVC: Model data not available, using safe defaults')
    self._current_lat_acc = 0.0
    self._max_pred_lat_acc = 0.0
    self._max_v_for_current_curvature = V_CRUISE_MAX * CV.KPH_TO_MS
    self._lat_acc_overshoot_ahead = False
    self._filtered_curvature = 0.0


  def _state_transition(self):
    # In any case, if system is disabled or the feature is disabled or gas is pressed, disable.
    if not self._op_enabled or not self._is_enabled or self._gas_pressed:
      self.state = VisionTurnControllerState.disabled
      return

    # DISABLED
    if self.state == VisionTurnControllerState.disabled:
      # Do not enter a turn control cycle if speed is low.
      if self._v_ego <= _MIN_V:
        pass
      # If substantial lateral acceleration is predicted ahead, then move to Entering turn state.
      elif self._max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
        self.state = VisionTurnControllerState.entering
    # ENTERING
    elif self.state == VisionTurnControllerState.entering:
      # Transition to Turning if current lateral acceleration is over the threshold.
      if self._current_lat_acc >= _TURNING_LAT_ACC_TH:
        self.state = VisionTurnControllerState.turning
      # Abort if the predicted lateral acceleration drops
      elif self._max_pred_lat_acc < _ABORT_ENTERING_PRED_LAT_ACC_TH:
        self.state = VisionTurnControllerState.disabled
    # TURNING
    elif self.state == VisionTurnControllerState.turning:
      # Transition to Leaving if current lateral acceleration drops drops below threshold.
      if self._current_lat_acc <= _LEAVING_LAT_ACC_TH:
        self.state = VisionTurnControllerState.leaving
    # LEAVING
    elif self.state == VisionTurnControllerState.leaving:
      # Transition back to Turning if current lateral acceleration goes back over the threshold.
      if self._current_lat_acc >= _TURNING_LAT_ACC_TH:
        self.state = VisionTurnControllerState.turning
      # Finish if current lateral acceleration goes below threshold.
      elif self._current_lat_acc < _FINISH_LAT_ACC_TH:
        self.state = VisionTurnControllerState.disabled

  def _update_solution(self):
    """Advanced solution calculation with multi-pass planning while maintaining state machine."""
    dt = 0.05  # 20Hz

    # Key change: Use advanced planning logic when we have meaningful curvature,
    # regardless of state machine state (which is only for UI/logging)
    has_meaningful_curvature = (self._filtered_curvature > 1e-7 or self._max_pred_lat_acc > 0.5)

    if has_meaningful_curvature:
      # Calculate target speed using advanced planning
      raw_target = self._plan_advanced_speed_trajectory()

      # Apply dynamic scaling
      scale_decel = dynamic_decel_scale(self._v_ego)
      scale_jerk = dynamic_decel_scale(self._v_ego)  # Use same scaling for jerk

      # Compute acceleration command
      accel_cmd = (raw_target - self._prev_target_speed) / dt

      # Apply acceleration and jerk limits with dynamic scaling
      pos_limit = self._max_accel
      neg_limit = self._max_decel * scale_decel
      accel_cmd = clip(accel_cmd, -neg_limit, pos_limit)

      # Jerk-limit the change in acceleration
      accel_diff = accel_cmd - self._current_accel

      if accel_diff > 0:
        max_delta = (self._max_jerk_accel * scale_jerk) * dt
        if accel_diff > max_delta:
          self._current_accel += max_delta
        else:
          self._current_accel = accel_cmd
      elif accel_diff < 0:
        max_delta = (self._max_jerk * scale_jerk) * dt
        if accel_diff < -max_delta:
          self._current_accel -= max_delta
        else:
          self._current_accel = accel_cmd
      else:
        self._current_accel = accel_cmd

      # Update target acceleration for compatibility
      self._a_target = self._current_accel

      # Update previous target speed to the actual planned target, not ego-relative
      # This allows proper acceleration when the vision controller is active
      self._prev_target_speed = raw_target

    else:
      # No meaningful curvature - use basic logic
      self._a_target = self._a_ego
      self._current_accel = self._a_ego
      self._prev_target_speed = self._v_ego

  def _find_time_index(self, times: np.ndarray, target_time: float, clip_high=False) -> int:
    """Helper to find an index in 'times' that is closest to 'target_time'."""
    n = len(times)
    if target_time <= times[0]:
        return 0
    if target_time >= times[-1] and clip_high:
        return n - 1
    for i in range(n - 1):
        if times[i] <= target_time < times[i + 1]:
            if (target_time - times[i]) < (times[i + 1] - target_time):
                return i
            else:
                return i + 1
    return n - 1 if clip_high else n - 2

  def _plan_advanced_speed_trajectory(self) -> float:
    """Advanced speed planning using chauffeur_vtsc.py multi-pass logic with apex detection."""

    # Use existing model data that was processed in _update_calculations
    # Get orientation rate and velocity prediction arrays from model data
    # This replicates the sophisticated planning from chauffeur_vtsc.py

    # If we don't have sufficient curvature data, fall back to simple logic
    if self._filtered_curvature < 1e-7:
      return min(self._v_cruise_setpoint, self._max_v_for_current_curvature)

    # For now, use a simple trajectory planning approach that mimics chauffeur_vtsc.py
    # but adapted to work with the existing data structures

    # Calculate safe speed using curvature_to_speed (physics-based)
    physics_safe_speed = curvature_to_speed(self._filtered_curvature)

    # Apply a simple trajectory that allows acceleration above cruise setpoint
    # when appropriate (mimicking the chauffeur_vtsc.py behavior)
    base_target = min(self._v_cruise_setpoint, physics_safe_speed)

    # Key change: Allow speeds above cruise setpoint for acceleration out of apexes
    # This mimics the chauffeur_vtsc.py logic where final_target_speed is only
    # clamped to cruise setpoint at the very end

    # Simple apex detection: if current curvature is decreasing from max predicted,
    # we're likely past an apex and should allow more aggressive acceleration
    if self._v_ego > 1e-3:  # Avoid division by zero
      curvature_ratio = self._filtered_curvature / max(self._max_pred_lat_acc / (self._v_ego**2), 1e-6)
    else:
      curvature_ratio = 1.0  # Conservative default when stopped or nearly stopped

    # Detect if we're approaching or past the apex
    is_past_apex = curvature_ratio < 0.7  # We're past the peak curvature

    if is_past_apex:
      # CRITICAL: Preserve original acceleration behavior after apex
      # Allow aggressive acceleration out of apex - key difference from state-based logic
      apex_recovery_factor = 1.25  # Allow 25% above normal speeds
      target_speed = base_target * apex_recovery_factor
      # Allow temporary overshoot above cruise for smooth apex exit
      target_speed = clip(target_speed, _MIN_V, self._v_cruise_setpoint * 1.2)

      # Clear deceleration state when past apex
      self._is_decelerating_for_curve = False
    else:
      # BEFORE APEX: Apply anticipatory deceleration
      # Check if we should start decelerating early
      if self._lat_acc_overshoot_ahead and not self._is_decelerating_for_curve:
        # Mark that we've started anticipatory deceleration
        self._is_decelerating_for_curve = True
        self._curve_detection_distance = self._v_overshoot_distance

      # Use the physics-based calculation
      target_speed = base_target

      # If we're in anticipatory deceleration mode and haven't reached target yet
      if self._is_decelerating_for_curve and self._v_ego > base_target + 0.5:
        # Apply slightly more aggressive deceleration to reach target early
        # This ensures we hit the target speed before the apex
        reduction_factor = 0.95  # Reduce target by 5% to decelerate faster
        target_speed = base_target * reduction_factor

      target_speed = clip(target_speed, _MIN_V, self._v_cruise_setpoint)

    return target_speed

  def update(self, sm, enabled, v_ego, a_ego, v_cruise_setpoint, v_cruise_cluster_setpoint=None):
    self._op_enabled = enabled
    self._gas_pressed = sm['carState'].gasPressed
    self._v_ego = v_ego
    self._a_ego = a_ego
    # Use cluster speed as source of truth if available, otherwise fall back to v_cruise
    self._v_cruise_setpoint = v_cruise_cluster_setpoint if v_cruise_cluster_setpoint is not None else v_cruise_setpoint

    # Initialize advanced controller state on first run or when speed changes significantly
    if (self._prev_target_speed == 0.0 or
        abs(self._prev_target_speed - v_ego) > 5.0):
      self._prev_target_speed = v_ego
      self._planned_speeds[:] = v_ego
      self._current_accel = a_ego

    self._update_params()
    self._update_calculations(sm)
    self._state_transition()
    self._update_solution()
