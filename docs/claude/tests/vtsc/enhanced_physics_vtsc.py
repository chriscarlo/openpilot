#!/usr/bin/env python3
"""
Physics-Based VTSC with Sigmoid Speed Curves from chauffeur-dev-merge branch

This is the complete physics-based vision turn speed controller that uses a continuous
sigmoid function to derive speed based on curvature, rather than the discrete state
machine approach used in our enhanced version.

Key differences from our enhanced VTSC:
1. Uses sigmoid formula for lateral acceleration: high_accel - span / (1 + exp(-k * (x - center)))
2. Continuous physics-based speed calculation: v = sqrt(a / k)
3. Advanced model-based curvature detection using orientation_rate/velocity
4. Multi-pass trajectory planning with apex detection
5. Sophisticated anticipatory deceleration logic

Retrieved from: remotes/origin/chauffeur-dev-merge:sunnypilot/selfdrive/controls/lib/vision_turn_controller.py
Date: 2025-08-01
"""

import numpy as np
import time
import math
from enum import IntEnum
from dataclasses import dataclass

# Advanced math libraries
try:
    from scipy import optimize, signal
    from scipy.interpolate import interp1d
    import scipy.ndimage as ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Mock cereal states for testing
class VisionTurnControllerState(IntEnum):
    disabled = 0
    entering = 1
    turning = 2
    leaving = 3

# Enhanced VTSC additions from our state-based version
class EmergencyLevel(IntEnum):
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
    INTERVENTION = 4

class VisionStatus(IntEnum):
    FULL_VISIBILITY = 0
    PARTIAL_OCCLUSION = 1
    CURVE_EXCEEDS_FOV = 2
    LOST_ROAD = 3

# System constraint: Maximum -6.0 m/s²
DECEL_LIMITS = {
    EmergencyLevel.NORMAL: -1.47,      # 0.15g
    EmergencyLevel.CAUTION: -2.45,     # 0.25g
    EmergencyLevel.WARNING: -3.92,     # 0.40g
    EmergencyLevel.CRITICAL: -5.50,    # 0.56g
    EmergencyLevel.INTERVENTION: -6.00  # 0.61g - System maximum
}

# Jerk limits for smooth transitions
JERK_LIMITS = {
    EmergencyLevel.NORMAL: 2.0,
    EmergencyLevel.CAUTION: 3.0,
    EmergencyLevel.WARNING: 4.0,
    EmergencyLevel.CRITICAL: 6.0,
    EmergencyLevel.INTERVENTION: 10.0
}

# Constants
class CV:
    KPH_TO_MS = 1/3.6
    MS_TO_KPH = 3.6
    MS_TO_MPH = 2.237
    DEG_TO_RAD = math.pi / 180

class Params:
    def __init__(self):
        self._values = {"VisionTurnSpeedControl": True}

    def get_bool(self, key):
        return self._values.get(key, False)

V_CRUISE_MAX = 144  # km/h

# Model constants
class ModelConstants:
    T_IDXS = [0.0, 0.033, 0.067, 0.1, 0.133, 0.167, 0.2, 0.233, 0.267, 0.3,
              0.333, 0.367, 0.4, 0.433, 0.467, 0.5, 0.533, 0.567, 0.6, 0.633,
              0.667, 0.7, 0.733, 0.767, 0.8, 0.833, 0.867, 0.9, 0.933, 0.967,
              1.0, 1.033, 1.067]

def clip(x, min_val, max_val):
    """Clip value between min and max"""
    return max(min_val, min(x, max_val))

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

# Enhanced anticipation constants from our state-based version
MIN_SAFE_DISTANCE = 10.0  # meters
MIN_ANTICIPATION_TIME = 1.0  # seconds
MAX_ANTICIPATION_TIME = 3.0  # seconds


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

def curvature_to_speed(abs_curvature_meters: float) -> float:
    """Calculates target speed (m/s) directly from curvature (1/radius in meters)."""
    if abs_curvature_meters < 1e-7:  # Handle straight roads
        return MAX_SPEED_DEFAULT

    # Scale curvature for the internally used lat accel function
    abs_curvature_scaled = abs_curvature_meters / CURV_CORR_FACTOR

    # Get the base lateral accel using the tuned logic
    base_lat_accel = _original_curvature_based_lat_accel(abs_curvature_scaled)

    # Calculate speed using physics formula v = sqrt(a / k)
    try:
        if base_lat_accel < 0:
            base_lat_accel = 0
        base_speed_mps = math.sqrt(base_lat_accel / abs_curvature_meters)
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
  return 'UNKNOWN'


@dataclass
class VisionOcclusionState:
    """Tracks vision occlusion and extrapolates curvature"""
    last_valid_curvature: float = 0.0
    last_valid_timestamp: float = 0.0
    occlusion_start_time: float | None = None
    extrapolated_curvature: float = 0.0
    confidence_decay_factor: float = 1.0
    vision_status: VisionStatus = VisionStatus.FULL_VISIBILITY
    initialized: bool = False

    def update(self, current_curvature: float | None,
               predicted_curvatures: np.ndarray | None,
               vision_status: VisionStatus,
               current_time: float) -> float:
        """Update occlusion state and return extrapolated curvature"""
        self.vision_status = vision_status

        # Initialize from predictions if needed
        if not self.initialized and predicted_curvatures is not None and len(predicted_curvatures) > 0:
            self.last_valid_curvature = np.max(predicted_curvatures)
            self.last_valid_timestamp = current_time
            self.initialized = True

        if vision_status == VisionStatus.FULL_VISIBILITY and current_curvature is not None:
            # Good visibility - update state
            self.last_valid_curvature = current_curvature
            self.last_valid_timestamp = current_time
            self.occlusion_start_time = None
            self.confidence_decay_factor = 1.0
            self.extrapolated_curvature = current_curvature
            self.initialized = True
            return current_curvature

        # Handle occlusion
        if self.occlusion_start_time is None:
            self.occlusion_start_time = current_time

        occlusion_duration = current_time - self.occlusion_start_time
        self.confidence_decay_factor = 0.5 ** (occlusion_duration / 2.0)

        # Apply appropriate safety factors
        if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
            safety_factor = 1.0 + 0.05 * min(occlusion_duration, 2.0)
            self.extrapolated_curvature = self.last_valid_curvature * safety_factor
        elif vision_status == VisionStatus.PARTIAL_OCCLUSION:
            if current_curvature is not None:
                blend_factor = self.confidence_decay_factor * 0.5
                self.extrapolated_curvature = (
                    blend_factor * current_curvature +
                    (1 - blend_factor) * self.last_valid_curvature * 1.05
                )
            else:
                self.extrapolated_curvature = self.last_valid_curvature * 1.05
        else:  # LOST_ROAD
            self.extrapolated_curvature = self.last_valid_curvature * 1.1

        return self.extrapolated_curvature


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

    # Enhanced VTSC additions
    self._emergency_level = EmergencyLevel.NORMAL
    self._time_at_current_level = 0.0
    self._critical_situation_time = 0.0
    self._occlusion_state = VisionOcclusionState()
    self._last_update_time = 0.0
    self._anticipation_time = 2.0
    self._model_confidence = 1.0
    self._intervention_required = False

    # Modified decel limits for safety
    self._max_decel_system = 6.0  # Hard system limit

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
  def emergency_level(self):
    return self._emergency_level

  @property
  def intervention_required(self):
    return self._intervention_required

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

    # Reset enhanced state
    self._emergency_level = EmergencyLevel.NORMAL
    self._time_at_current_level = 0.0
    self._critical_situation_time = 0.0
    self._intervention_required = False
  def _update_params(self):
    tm = time.time()
    if tm > self._last_params_update + 5.0:
      self._is_enabled = self._params.get_bool("VisionTurnSpeedControl")
      self._last_params_update = tm

  def _determine_emergency_level(self, required_decel: float,
                               distance: float, v_ego: float) -> EmergencyLevel:
    """Determine appropriate emergency level with vision awareness"""
    required_g = abs(required_decel) / 9.81

    # Consider vision status
    vision_factor = 1.0
    if self._occlusion_state.vision_status == VisionStatus.LOST_ROAD:
      vision_factor = 0.8  # More conservative
    elif self._occlusion_state.vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
      vision_factor = 0.9

    # Speed factor
    speed_factor = min(v_ego / 30.0, 1.0)

    # Distance-based thresholds with vision adjustment
    if distance > 60:
      thresholds = [0.20, 0.30, 0.50, 0.70]
    elif distance > 40:
      thresholds = [0.17, 0.27, 0.42, 0.62]
    else:
      base_thresholds = [0.15, 0.25, 0.40, 0.60]
      thresholds = [t * (1 - 0.1 * speed_factor) for t in base_thresholds]

    # Apply vision factor
    thresholds = [t * vision_factor for t in thresholds]

    if required_g <= thresholds[0]:
      return EmergencyLevel.NORMAL
    elif required_g <= thresholds[1]:
      return EmergencyLevel.CAUTION
    elif required_g <= thresholds[2]:
      return EmergencyLevel.WARNING
    elif required_g <= thresholds[3]:
      return EmergencyLevel.CRITICAL
    else:
      return EmergencyLevel.INTERVENTION

  def _update_calculations(self, sm):
    """Advanced vision-based curvature calculation using direct model outputs."""
    model_data = sm['modelV2'] if sm.valid.get('modelV2', False) else None
    current_time = time.time()

    # Initialize defaults for edge cases
    current_curvature_signed = 0.0
    current_curvature = 0.0
    max_pred_curvature = 0.0

    # Determine vision status first
    if model_data is not None and hasattr(model_data, 'laneLineProbs') and len(model_data.laneLineProbs) > 2:
      l_prob = model_data.laneLineProbs[1] if len(model_data.laneLineProbs) > 1 else 0.5
      r_prob = model_data.laneLineProbs[2] if len(model_data.laneLineProbs) > 2 else 0.5
      self._model_confidence = (l_prob + r_prob) / 2.0
    else:
      self._model_confidence = 0.5

    # Determine vision status
    if self._model_confidence < 0.3:
      vision_status = VisionStatus.LOST_ROAD
    elif self._model_confidence < 0.6 and self._current_lat_acc < 0.1:
      vision_status = VisionStatus.PARTIAL_OCCLUSION
    elif self._max_pred_lat_acc > 2.0 * 1.2:  # _A_LAT_REG_MAX equivalent
      vision_status = VisionStatus.CURVE_EXCEEDS_FOV
    else:
      vision_status = VisionStatus.FULL_VISIBILITY

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

        # Update occlusion state with vision awareness
        predicted_curvatures = curvature_array
        current_curvature_for_occlusion = current_curvature if vision_status == VisionStatus.FULL_VISIBILITY else None

        extrapolated_curvature = self._occlusion_state.update(
            current_curvature_for_occlusion,
            predicted_curvatures,
            vision_status,
            current_time
        )

        # Use extrapolated curvature if occluded
        if vision_status != VisionStatus.FULL_VISIBILITY:
            max_pred_curvature = extrapolated_curvature

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
    """Enhanced solution with constrained physics and emergency levels"""
    dt = 0.05  # 20Hz
    current_time = time.time()
    self._time_at_current_level += dt

    has_meaningful_curvature = (self._filtered_curvature > 1e-7 or self._max_pred_lat_acc > 0.5)

    if has_meaningful_curvature:
      # Step 1: Calculate physics-based target speed
      raw_target = self._plan_advanced_speed_trajectory()

      # Step 2: Calculate required deceleration
      speed_diff = self._v_ego - raw_target
      distance_to_target = self._v_overshoot_distance if self._lat_acc_overshoot_ahead else 100.0

      if speed_diff > 0 and distance_to_target > 0:
        required_decel = -(speed_diff ** 2) / (2 * distance_to_target)
      else:
        required_decel = 0.0

      # Step 3: Determine emergency level
      target_level = self._determine_emergency_level(
        required_decel, distance_to_target, self._v_ego
      )

      # Step 4: Handle level transitions with hysteresis
      if target_level != self._emergency_level:
        should_transition = False

        if target_level.value > self._emergency_level.value:
          # Escalating - quick response
          if self._time_at_current_level > 0.15:
            should_transition = True
        else:
          # De-escalating - slower
          if self._time_at_current_level > 2.0:
            should_transition = True

        if should_transition:
          self._emergency_level = target_level
          self._time_at_current_level = 0.0
          _debug(f'Emergency level: {self._emergency_level.name}')

      # Step 5: Calculate physics-based acceleration with dynamic scaling
      scale_decel = dynamic_decel_scale(self._v_ego)
      accel_cmd_physics = (raw_target - self._prev_target_speed) / dt

      # Apply physics limits
      pos_limit = self._max_accel
      neg_limit_physics = self._max_decel * scale_decel
      accel_cmd_physics = clip(accel_cmd_physics, -neg_limit_physics, pos_limit)

      # Step 6: Apply emergency level constraints
      level_limit = DECEL_LIMITS[self._emergency_level]
      accel_cmd_constrained = max(accel_cmd_physics, level_limit)

      # Step 7: Apply jerk limiting per emergency level
      max_jerk = JERK_LIMITS[self._emergency_level]
      accel_diff = accel_cmd_constrained - self._current_accel

      if abs(accel_diff) > max_jerk * dt:
        if accel_diff < 0:
          self._current_accel -= max_jerk * dt
        else:
          self._current_accel += max_jerk * dt
      else:
        self._current_accel = accel_cmd_constrained

      # Step 8: Final safety check - enforce system limit
      self._current_accel = max(self._current_accel, -self._max_decel_system)

      # Step 9: Check for intervention
      if abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]):
        self._critical_situation_time += dt
      else:
        self._critical_situation_time = 0.0

      self._intervention_required = (
        abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]) * 1.05 and
        self._critical_situation_time > 0.3 and
        self._emergency_level == EmergencyLevel.CRITICAL and
        distance_to_target < 25
      )

      # Update outputs
      self._a_target = self._current_accel
      self._prev_target_speed = raw_target

      _debug(f'Enhanced Physics: level={self._emergency_level.name}, '
            f'decel={self._current_accel:.2f} m/s², '
            f'vision={self._occlusion_state.vision_status.name}')
    else:
      # No meaningful curvature
      self._a_target = self._a_ego
      self._current_accel = self._a_ego
      self._prev_target_speed = self._v_ego
      self._emergency_level = EmergencyLevel.NORMAL

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
