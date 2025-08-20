import numpy as np
import time
import math
from enum import IntEnum
from dataclasses import dataclass

from cereal import custom
from openpilot.common.params import Params
from opendbc.car.common.conversions import Conversions as CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N

VisionTurnSpeedControlState = custom.LongitudinalPlanSP.VisionTurnSpeedControl.VisionTurnSpeedControlState

TRAJECTORY_SIZE = 33

_MIN_V = 20 * CV.KPH_TO_MS  # Do not operate under 20 km/h

_ENTERING_PRED_LAT_ACC_TH = 1.3  # Predicted Lat Acc threshold to trigger entering turn state.
_ABORT_ENTERING_PRED_LAT_ACC_TH = 1.1  # Predicted Lat Acc threshold to abort entering state if speed drops.

_TURNING_LAT_ACC_TH = 1.6  # Lat Acc threshold to trigger turning state.

_LEAVING_LAT_ACC_TH = 1.3  # Lat Acc threshold to trigger leaving turn state.
_FINISH_LAT_ACC_TH = 1.1  # Lat Acc threshold to trigger the end of the turn cycle.

_EVAL_STEP = 5.  # mts. Resolution of the curvature evaluation.
_EVAL_START = 20.  # mts. Distance ahead where to start evaluating vision curvature.
_EVAL_LENGTH = 150.  # mts. Distance ahead where to stop evaluating vision curvature.
_EVAL_RANGE = np.arange(_EVAL_START, _EVAL_LENGTH, _EVAL_STEP)

_A_LAT_REG_MAX = 3.12  # Maximum lateral acceleration (restored from original)

_NO_OVERSHOOT_TIME_HORIZON = 4.  # s. Time to use for velocity desired based on a_target when not overshooting.

# Lookup table for the minimum smooth deceleration during the ENTERING state
# depending on the actual maximum absolute lateral acceleration predicted on the turn ahead.
_ENTERING_SMOOTH_DECEL_V = [-0.2, -1.]  # min decel value allowed on ENTERING state
_ENTERING_SMOOTH_DECEL_BP = [1.3, 3.]  # absolute value of lat acc ahead

# Lookup table for the acceleration for the TURNING state
# depending on the current lateral acceleration of the vehicle.
_TURNING_ACC_V = [0.5, 0., -0.4]  # acc value
_TURNING_ACC_BP = [1.5, 2.3, 3.]  # absolute value of current lat acc

_LEAVING_ACC = 0.5  # Conformable acceleration to regain speed while leaving a turn.

_MIN_LANE_PROB = 0.6  # Minimum lanes probability to allow curvature prediction based on lanes.

_DEBUG = False


# Enhanced VTSC additions
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

# Constants for anticipatory control
MIN_SAFE_DISTANCE = 10.0  # meters
MIN_ANTICIPATION_TIME = 1.0  # seconds
MAX_ANTICIPATION_TIME = 3.0  # seconds


def _debug(msg):
  if not _DEBUG:
    return
  print(msg)


def eval_curvature(poly, x_vals):
  """
  This function returns a vector with the curvature based on a path defined by `poly`
  evaluated on distance vector `x_vals`
  """

  # https://en.wikipedia.org/wiki/Curvature# Local_expressions
  def curvature(x):
    a = abs(2 * poly[1] + 6 * poly[0] * x) / (1 + (3 * poly[0] * x ** 2 + 2 * poly[1] * x + poly[2]) ** 2) ** 1.5
    return a

  return np.vectorize(curvature)(x_vals)


def eval_lat_acc(v_ego, x_curv):
  """
  This function returns a vector with the lateral acceleration based
  for the provided speed `v_ego` evaluated over curvature vector `x_curv`
  """

  def lat_acc(curv):
    a = v_ego ** 2 * curv
    return a

  return np.vectorize(lat_acc)(x_curv)


def _description_for_state(turn_controller_state):
  if turn_controller_state == VisionTurnSpeedControlState.disabled:
    return 'DISABLED'
  if turn_controller_state == VisionTurnSpeedControlState.entering:
    return 'ENTERING'
  if turn_controller_state == VisionTurnSpeedControlState.turning:
    return 'TURNING'
  if turn_controller_state == VisionTurnSpeedControlState.leaving:
    return 'LEAVING'
  return NotImplementedError("v-tsc: state not supported")


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
    self._state = VisionTurnSpeedControlState.disabled

    # Enhanced VTSC additions - always active when VTSC is enabled
    self._emergency_level = EmergencyLevel.NORMAL
    self._current_decel = 0.0
    self._time_at_current_level = 0.0
    self._critical_situation_time = 0.0
    self._occlusion_state = VisionOcclusionState()
    self._last_update_time = 0.0
    self._anticipation_time = 2.0  # seconds to reach target speed before physically necessary
    self._model_confidence = 1.0
    self._intervention_required = False

    self._reset()

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self, value):
    if value != self._state:
      _debug(f'TVC: TurnVisionController state: {_description_for_state(value)}')
      if value == VisionTurnSpeedControlState.disabled:
        self._reset()
    self._state = value

  @property
  def a_target(self):
    if not self.is_active:
      return self._a_ego

    # Use post-apex acceleration when active
    if self._apex_acceleration_active:
      return self._apex_acceleration_value

    # Otherwise use enhanced deceleration
    return self._current_decel

  @property
  def v_turn(self):
    if not self.is_active:
      return self._v_cruise_setpoint

    if self._lat_acc_overshoot_ahead:
      return self._v_overshoot
    else:
      normal_calc = self._v_ego + self._a_target * _NO_OVERSHOOT_TIME_HORIZON
      return normal_calc

  @property
  def current_lat_acc(self):
    return self._current_lat_acc

  @property
  def max_pred_lat_acc(self):
    return self._max_pred_lat_acc

  @property
  def is_active(self):
    return self._state != VisionTurnSpeedControlState.disabled

  @property
  def emergency_level(self):
    return self._emergency_level

  @property
  def intervention_required(self):
    return self._intervention_required

  # Physics script properties for external access
  @property
  def apex_detected(self):
    return self._apex_detected

  @property
  def past_apex(self):
    return self._past_apex

  @property
  def acceleration_embargo_lifted(self):
    return self._acceleration_embargo_lifted

  def _reset(self):
    self._current_lat_acc = 0.
    self._max_v_for_current_curvature = 0.
    self._max_pred_lat_acc = 0.
    self._v_overshoot_distance = 200.
    self._lat_acc_overshoot_ahead = False

    # Enhanced VTSC reset
    self._emergency_level = EmergencyLevel.NORMAL
    self._current_decel = 0.0
    self._time_at_current_level = 0.0
    self._critical_situation_time = 0.0
    self._intervention_required = False

    # Physics script features
    self.curvature_trajectory = []
    self.target_speeds_trajectory = []
    self.filtered_curvature = 0.0
    self.physics_max_pred_lat_acc = 0.0
    self._apex_detected = False
    self._past_apex = False
    self._acceleration_embargo_lifted = False
    self._last_filtered_curvature = 0.0
    self._curvature_ema_ratio = 0.3
    self._past_apex_steps = 0  # Count steps since past apex
    self._apex_acceleration_active = False
    self._apex_acceleration_value = 0.0

  def _update_params(self):
    tm = time.time()
    if tm > self._last_params_update + 5.0:
      self._is_enabled = self._params.get_bool("VisionTurnSpeedControl")
      self._last_params_update = tm

  def _calculate_safe_speed_for_curve(self, curvature: float, lateral_acc_limit: float) -> float:
    """Calculate safe speed for given curvature"""
    if curvature <= 0:
      return 100.0  # m/s (effectively no limit)
    return math.sqrt(lateral_acc_limit / curvature)

  def _calculate_anticipation_distance(self, v_ego: float, v_target: float,
                                     distance_to_critical: float,
                                     comfort_decel_g: float = 0.15) -> float:
    """Calculate where to start deceleration for anticipatory control"""
    if v_ego <= v_target:
      return 0.0

    # Convert g to m/s²
    comfort_decel = comfort_decel_g * 9.81

    # Distance needed to decelerate comfortably
    decel_distance = (v_ego**2 - v_target**2) / (2 * comfort_decel)

    # Time at target speed before curve
    anticipation_time = np.clip(self._anticipation_time,
                               MIN_ANTICIPATION_TIME,
                               MAX_ANTICIPATION_TIME)
    anticipation_distance = v_target * anticipation_time

    # Total distance needed
    total_distance_needed = decel_distance + anticipation_distance + MIN_SAFE_DISTANCE

    return total_distance_needed

  def _calculate_required_deceleration(self, v_ego: float, v_target: float,
                                     distance: float) -> float:
    """Calculate required deceleration with safety margin"""
    if distance <= 0 or v_ego <= v_target:
      return 0.0

    # Dynamic safety margin
    if distance < 20:
      safety_margin = 0.9
    elif distance < 40:
      safety_margin = 0.93
    else:
      safety_margin = 0.95

    safety_distance = distance * safety_margin
    return -(v_ego**2 - v_target**2) / (2 * safety_distance)

  def _determine_emergency_level(self, required_decel: float,
                                distance: float, v_ego: float) -> EmergencyLevel:
    """Determine appropriate emergency level"""
    required_g = abs(required_decel) / 9.81

    # Consider speed factor
    speed_factor = min(v_ego / 30.0, 1.0)

    if distance > 60:
      # Far - relaxed thresholds
      if required_g <= 0.20:
        return EmergencyLevel.NORMAL
      elif required_g <= 0.30:
        return EmergencyLevel.CAUTION
      elif required_g <= 0.50:
        return EmergencyLevel.WARNING
      elif required_g <= 0.70:
        return EmergencyLevel.CRITICAL
      else:
        return EmergencyLevel.INTERVENTION
    elif distance > 40:
      # Medium - balanced thresholds
      if required_g <= 0.17:
        return EmergencyLevel.NORMAL
      elif required_g <= 0.27:
        return EmergencyLevel.CAUTION
      elif required_g <= 0.42:
        return EmergencyLevel.WARNING
      elif required_g <= 0.62:
        return EmergencyLevel.CRITICAL
      else:
        return EmergencyLevel.INTERVENTION
    else:
      # Close - standard thresholds
      base_thresholds = [0.15, 0.25, 0.40, 0.60]
      adjusted_thresholds = [t * (1 - 0.1 * speed_factor) for t in base_thresholds]

      if required_g <= adjusted_thresholds[0]:
        return EmergencyLevel.NORMAL
      elif required_g <= adjusted_thresholds[1]:
        return EmergencyLevel.CAUTION
      elif required_g <= adjusted_thresholds[2]:
        return EmergencyLevel.WARNING
      elif required_g <= adjusted_thresholds[3]:
        return EmergencyLevel.CRITICAL
      else:
        return EmergencyLevel.INTERVENTION

  def _get_optimal_deceleration(self, level: EmergencyLevel,
                              required_decel: float) -> float:
    """Get optimal deceleration for level"""
    level_limit = DECEL_LIMITS[level]

    # Use only what's needed with small buffer
    if abs(required_decel) <= abs(level_limit):
      buffer_factor = 1.1
      target = required_decel * buffer_factor
      if abs(target) > abs(level_limit):
        return level_limit
      return target
    else:
      return level_limit

  def _physics_based_lateral_acceleration(self, curvature: float) -> float:
    """
    Pure mathematical sigmoid for curvature-based lateral acceleration
    NO SCALING HACKS - Takes curvature in SI units (1/meters)
    
    Returns LOWER lateral acceleration for HIGHER curvature (inverse relationship)
    This ensures sharp curves get appropriate low speeds while maintaining safety.
    
    Mathematical Formula: a = a_min + (a_max - a_min) * exp(-alpha * curvature^beta)
    
    Args:
        curvature: Curvature in SI units (1/meters)
        
    Returns:
        lateral_acceleration: Safe lateral acceleration in m/s²
    """
    # Physical constants - no arbitrary scaling
    a_max = 3.12   # Maximum lateral acceleration (m/s²) - safety limit
    a_min = 1.2    # Minimum lateral acceleration (m/s²) - tight curve limit
    alpha = 24.3   # Decay rate - tuned for sports car performance (was 80.0)
    beta = 0.78    # Power law exponent - tuned for sports car performance (was 0.8)

    # Power law sigmoid: exponential decay for smooth inverse relationship
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))

    # Safety bounds (function naturally stays within bounds)
    return max(a_min, min(lateral_acceleration, a_max))

  def _curvature_to_speed(self, curvature: float) -> float:
    """Convert curvature to safe speed - exact physics script implementation"""
    if curvature <= 1e-7:
      return 200.0  # Very high speed for straight roads

    # Get safe lateral acceleration for this curvature (NO SCALING!)
    safe_lat_accel = self._physics_based_lateral_acceleration(abs(curvature))

    # Calculate safe speed: v = sqrt(a_lat / curvature)
    safe_speed = math.sqrt(safe_lat_accel / abs(curvature))

    # Apply reasonable limits
    final_speed = max(5.6, min(safe_speed, 50.0))  # 5.6 m/s = 20 km/h minimum


    return final_speed

  def _update_enhanced_calculations(self, sm, pred_curvatures):
    """Enhanced calculations for anticipatory control"""
    current_time = time.time()
    dt = current_time - self._last_update_time if self._last_update_time > 0 else 0.05
    self._last_update_time = current_time
    self._time_at_current_level += dt

    # PHYSICS SCRIPT INTEGRATION - Model data ingestion from modelV2
    model_data = sm['modelV2'] if sm.valid.get('modelV2', False) else None
    if model_data is not None:
      # Extract orientation rates and velocities - EXACT PHYSICS SCRIPT METHOD
      orientation_rates = np.array(model_data.orientationRate.z)
      velocities = np.array(model_data.velocity.x)

      # Calculate curvature trajectory: curvature = orientation_rate / velocity
      self.curvature_trajectory = []
      self.target_speeds_trajectory = []

      for i in range(min(len(orientation_rates), len(velocities), 33)):
        if abs(velocities[i]) > 0.1:  # Avoid division by zero
          curvature = abs(orientation_rates[i] / velocities[i])
          target_speed = self._curvature_to_speed(curvature)
        else:
          curvature = 0.0
          target_speed = 50.0

        self.curvature_trajectory.append(curvature)
        self.target_speeds_trajectory.append(target_speed)

      # EMA filtering of current curvature - EXACT PHYSICS SCRIPT METHOD
      if len(self.curvature_trajectory) > 0:
        current_curvature = self.curvature_trajectory[0]
        self.filtered_curvature = (self._curvature_ema_ratio * current_curvature +
                                 (1 - self._curvature_ema_ratio) * self._last_filtered_curvature)
        self._last_filtered_curvature = self.filtered_curvature

      # APEX DETECTION - EXACT PHYSICS SCRIPT LOGIC
      if len(self.curvature_trajectory) > 0 and self._v_ego > 0.1:
        # Calculate maximum predicted lateral acceleration
        self.physics_max_pred_lat_acc = max([curvature * (self._v_ego ** 2) for curvature in self.curvature_trajectory])

        # Apex detection using curvature ratio - EXACT PHYSICS SCRIPT METHOD
        curvature_ratio = self.filtered_curvature / max(self.physics_max_pred_lat_acc / (self._v_ego**2), 1e-6)

        # Update apex detection state
        if not self._apex_detected and curvature_ratio >= 0.7:
          self._apex_detected = True

        # Past apex detection - EXACT PHYSICS SCRIPT LOGIC
        if self._apex_detected and curvature_ratio < 0.7:
          if not self._past_apex:
            self._past_apex = True
            self._past_apex_steps = 0  # Reset counter when first detecting past apex

        # Count steps since past apex
        if self._past_apex:
          self._past_apex_steps += 1

        # Acceleration embargo lifting - PHYSICS SCRIPT LOGIC
        if self._past_apex and not self._acceleration_embargo_lifted:
          # Simple approach: lift embargo after 3 steps past apex
          if self._past_apex_steps >= 3:
            self._acceleration_embargo_lifted = True

    # Get model confidence
    model_data = sm['modelV2'] if sm.valid.get('modelV2', False) else None
    if model_data is not None:
      # Use lane line probabilities as proxy for model confidence
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
    elif self._max_pred_lat_acc > _A_LAT_REG_MAX * 1.2:
      vision_status = VisionStatus.CURVE_EXCEEDS_FOV
    else:
      vision_status = VisionStatus.FULL_VISIBILITY

    # Update occlusion state with current curvature
    current_curvature = abs(sm['carState'].steeringAngleDeg * CV.DEG_TO_RAD /
                           (self._CP.steerRatio * self._CP.wheelbase))

    extrapolated_curvature = self._occlusion_state.update(
      current_curvature if vision_status == VisionStatus.FULL_VISIBILITY else None,
      pred_curvatures,
      vision_status,
      current_time
    )

    # Determine planning curvature
    if vision_status != VisionStatus.FULL_VISIBILITY:
      planning_curvature = extrapolated_curvature
    else:
      planning_curvature = current_curvature

    # Calculate safe speeds
    v_safe_current = self._calculate_safe_speed_for_curve(
      planning_curvature, _A_LAT_REG_MAX
    )

    # Analyze predicted path
    critical_distance = 100.0  # Default far distance
    v_safe_ahead = v_safe_current
    max_curvature = planning_curvature

    if pred_curvatures is not None and len(pred_curvatures) > 0:
      # Apply vision safety factors
      if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
        safety_factor = 1.0 + 0.05 * (1 - self._occlusion_state.confidence_decay_factor)
        adjusted_curvatures = pred_curvatures * safety_factor
      else:
        adjusted_curvatures = pred_curvatures

      # Find most critical curvature
      max_curve_idx = np.argmax(adjusted_curvatures)
      max_curvature = adjusted_curvatures[max_curve_idx]

      if max_curve_idx < len(_EVAL_RANGE):
        critical_distance = _EVAL_RANGE[max_curve_idx]

      # Target speed for critical curvature
      v_safe_ahead = self._calculate_safe_speed_for_curve(
        max_curvature, _A_LAT_REG_MAX * 0.97
      )

    # Determine target speed
    v_target_physics = min(v_safe_current, v_safe_ahead, self._v_cruise_setpoint)

    # Apply vision safety margins
    if vision_status == VisionStatus.LOST_ROAD:
      v_target_physics *= 0.95
    elif vision_status == VisionStatus.CURVE_EXCEEDS_FOV and self._model_confidence < 0.5:
      v_target_physics *= 0.98

    # Calculate anticipatory deceleration point
    anticipation_distance = self._calculate_anticipation_distance(
      self._v_ego, v_target_physics, critical_distance
    )

    # Check if we should use anticipatory control
    using_anticipation = critical_distance <= anticipation_distance

    # Calculate required deceleration
    remaining_distance = critical_distance
    required_decel = self._calculate_required_deceleration(
      self._v_ego, v_target_physics, remaining_distance
    )

    # Determine emergency level
    target_level = self._determine_emergency_level(
      required_decel, remaining_distance, self._v_ego
    )

    # Handle level transitions
    if target_level != self._emergency_level:
      should_transition = False

      if target_level.value > self._emergency_level.value:
        # Escalating
        if self._time_at_current_level > 0.15:
          should_transition = True
      else:
        # De-escalating
        if self._time_at_current_level > 2.0:
          should_transition = True

      if should_transition:
        self._emergency_level = target_level
        self._time_at_current_level = 0.0

    # Get optimal deceleration
    target_decel = self._get_optimal_deceleration(self._emergency_level, required_decel)

    # Apply jerk limiting
    max_jerk = JERK_LIMITS[self._emergency_level]
    max_change = max_jerk * dt

    decel_error = target_decel - self._current_decel

    if abs(decel_error) > max_change:
      if decel_error < 0:
        self._current_decel -= max_change
      else:
        self._current_decel += max_change
    else:
      self._current_decel = target_decel

    # Track critical situations
    if abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]):
      self._critical_situation_time += dt
    else:
      self._critical_situation_time = 0.0

    # Intervention logic
    self._intervention_required = False

    if (abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]) * 1.05 and
        self._critical_situation_time > 0.3 and
        self._emergency_level == EmergencyLevel.CRITICAL and
        remaining_distance < 25):
      self._intervention_required = True

    _debug(f'Enhanced VTSC: level={self._emergency_level.name}, '
           f'decel={self._current_decel:.2f}, anticipation={using_anticipation}')

  def _update_calculations(self, sm):
    # Get path polynomial approximation for curvature estimation from model data.
    path_poly = None
    model_data = sm['modelV2'] if sm.valid.get('modelV2', False) else None
    lat_planner_data = sm['lateralPlan'] if sm.valid.get('lateralPlan', False) else None

    # 1. When the probability of lanes is good enough, compute polynomial from lanes as they are way more stable
    # on current mode than a driving path.
    if model_data is not None and len(model_data.laneLines) == 4 and len(model_data.laneLines[0].t) == TRAJECTORY_SIZE:
      ll_x = model_data.laneLines[1].x  # left and right ll x is the same
      lll_y = np.array(model_data.laneLines[1].y)
      rll_y = np.array(model_data.laneLines[2].y)
      l_prob = model_data.laneLineProbs[1]
      r_prob = model_data.laneLineProbs[2]
      lll_std = model_data.laneLineStds[1]
      rll_std = model_data.laneLineStds[2]

      # Reduce reliance on lanelines that are too far apart or will be in a few seconds
      width_pts = rll_y - lll_y
      prob_mods = []
      for t_check in [0.0, 1.5, 3.0]:
        width_at_t = np.interp(t_check * (self._v_ego + 7), ll_x, width_pts)
        prob_mods.append(np.interp(width_at_t, [4.0, 5.0], [1.0, 0.0]))
      mod = min(prob_mods)
      l_prob *= mod
      r_prob *= mod

      # Reduce reliance on uncertain lanelines
      l_std_mod = np.interp(lll_std, [.15, .3], [1.0, 0.0])
      r_std_mod = np.interp(rll_std, [.15, .3], [1.0, 0.0])
      l_prob *= l_std_mod
      r_prob *= r_std_mod

      # Find a path from lanes as the average center lane only if min probability on both lanes is above a threshold.
      if l_prob > _MIN_LANE_PROB and r_prob > _MIN_LANE_PROB:
        c_y = width_pts / 2 + lll_y
        path_poly = np.polyfit(ll_x, c_y, 3)

    # 2. If not polynomially derived from lanes, then derive it from a driving path as provided by `lateralPlanner`.
    if path_poly is None and lat_planner_data is not None and len(lat_planner_data.psis) == CONTROL_N \
       and lat_planner_data.dPathPoints[0] > 0:
      yData = list(lat_planner_data.dPathPoints)
      path_poly = np.polyfit(lat_planner_data.psis, yData[0:CONTROL_N], 3)

    # 3. If no polynomial derived from lanes or driving path, then provide a straight line poly.
    if path_poly is None:
      path_poly = np.array([0., 0., 0., 0.])

    current_curvature = abs(
      sm['carState'].steeringAngleDeg * CV.DEG_TO_RAD / (self._CP.steerRatio * self._CP.wheelbase))
    self._current_lat_acc = current_curvature * self._v_ego ** 2
    self._max_v_for_current_curvature = math.sqrt(_A_LAT_REG_MAX / current_curvature) if current_curvature > 0 \
        else V_CRUISE_MAX * CV.KPH_TO_MS

    pred_curvatures = eval_curvature(path_poly, _EVAL_RANGE)
    max_pred_curvature = np.amax(pred_curvatures)
    self._max_pred_lat_acc = self._v_ego ** 2 * max_pred_curvature

    max_curvature_for_vego = _A_LAT_REG_MAX / max(self._v_ego, 0.1) ** 2

    lat_acc_overshoot_idxs = np.nonzero(pred_curvatures >= max_curvature_for_vego)[0]
    self._lat_acc_overshoot_ahead = len(lat_acc_overshoot_idxs) > 0


    if self._lat_acc_overshoot_ahead:
      raw_v_overshoot = math.sqrt(_A_LAT_REG_MAX / max_pred_curvature)
      self._v_overshoot = min(raw_v_overshoot, self._v_cruise_setpoint)
      self._v_overshoot_distance = max(float(lat_acc_overshoot_idxs[0] * _EVAL_STEP + _EVAL_START), _EVAL_STEP)
      _debug(f'TVC: High LatAcc. Dist: {self._v_overshoot_distance:.2f}, v: {self._v_overshoot * CV.MS_TO_KPH:.2f}')

    # Run enhanced calculations if enabled
    self._update_enhanced_calculations(sm, pred_curvatures)

  def _state_transition(self):
    # In any case, if a system is disabled or the feature is disabled or gas is pressed, disable.
    if not self._op_enabled or not self._is_enabled or self._gas_pressed:
      self.state = VisionTurnSpeedControlState.disabled
      return

    # DISABLED
    if self.state == VisionTurnSpeedControlState.disabled:
      # Do not enter a turn control cycle if the speed is low.
      if self._v_ego <= _MIN_V:
        pass
      # If significant lateral acceleration is predicted ahead, then move to Entering turn state.
      elif self._max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
        self.state = VisionTurnSpeedControlState.entering
    # ENTERING
    elif self.state == VisionTurnSpeedControlState.entering:
      # Transition to Turning if current lateral acceleration is over the threshold.
      if self._current_lat_acc >= _TURNING_LAT_ACC_TH:
        self.state = VisionTurnSpeedControlState.turning
      # Abort if the predicted lateral acceleration drops
      elif self._max_pred_lat_acc < _ABORT_ENTERING_PRED_LAT_ACC_TH:
        self.state = VisionTurnSpeedControlState.disabled
    # TURNING
    elif self.state == VisionTurnSpeedControlState.turning:
      # Transition to Leaving if current lateral acceleration drops below a threshold.
      if self._current_lat_acc <= _LEAVING_LAT_ACC_TH:
        self.state = VisionTurnSpeedControlState.leaving
    # LEAVING
    elif self.state == VisionTurnSpeedControlState.leaving:
      # Transition back to Turning if current lateral acceleration goes back over the threshold.
      if self._current_lat_acc >= _TURNING_LAT_ACC_TH:
        self.state = VisionTurnSpeedControlState.turning
      # Finish if current lateral acceleration goes below a threshold.
      elif self._current_lat_acc < _FINISH_LAT_ACC_TH:
        self.state = VisionTurnSpeedControlState.disabled

  def _update_solution(self):
    # The anticipatory deceleration is calculated in `_update_enhanced_calculations` and
    # stored in `self._current_decel`. This value is used by default.

    # This method now *only* handles overriding the default deceleration with
    # post-apex acceleration logic.

    # Default to no acceleration override
    self._apex_acceleration_active = False

    # Only consider accelerating after the apex and if the embargo is lifted.
    if self._past_apex and self._acceleration_embargo_lifted and self.filtered_curvature > 1e-7:
      # Calculate target speed using physics script method
      physics_safe_speed = self._curvature_to_speed(self.filtered_curvature)

      # Apply apex recovery factor
      apex_recovery_factor = 1.4
      target_speed = physics_safe_speed * apex_recovery_factor

      # Conservative cruise speed estimate
      estimated_cruise = min(self._v_cruise_setpoint, 30.0)
      target_speed = max(5.6, min(target_speed, estimated_cruise * 1.2))

      # Calculate acceleration command based on speed difference
      speed_error = target_speed - self._v_ego

      # Apply apex acceleration only if there's a significant speed error and
      # the anticipatory logic isn't already commanding a strong deceleration.
      if speed_error > 0.2 and self._current_decel >= -0.5:
        self._apex_acceleration_active = True
        dt = 0.05  # 20Hz matching physics script
        acceleration_command = min(speed_error / dt, 2.0)  # Limit to 2.0 m/s² max

        # Store the post-apex acceleration value (don't override _current_decel)
        self._apex_acceleration_value = acceleration_command

    # If we are in the 'leaving' state but not actively accelerating via apex logic,
    # apply a gentle default acceleration.
    elif self.state == VisionTurnSpeedControlState.leaving:
      self._current_decel = _LEAVING_ACC

    # If disabled, reset to ego acceleration.
    elif self.state == VisionTurnSpeedControlState.disabled:
      self._current_decel = self._a_ego

    # Update self._a_target for v_turn calculation, respecting apex acceleration
    if self._apex_acceleration_active:
      self._a_target = self._apex_acceleration_value
    else:
      self._a_target = self._current_decel

  def update(self, sm, enabled, v_ego, a_ego, v_cruise_setpoint):
    self._op_enabled = enabled
    self._gas_pressed = sm['carState'].gasPressed
    self._v_ego = v_ego
    self._a_ego = a_ego
    self._v_cruise_setpoint = v_cruise_setpoint

    self._update_params()
    self._update_calculations(sm)
    self._state_transition()
    self._update_solution()
