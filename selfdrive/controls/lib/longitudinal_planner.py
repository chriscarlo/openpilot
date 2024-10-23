#!/usr/bin/env python3
import math
import numpy as np
from openpilot.common.numpy_fast import clip, interp

import cereal.messaging as messaging
from openpilot.common.conversions import Conversions as CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.simple_kalman import KF1D
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc,
  T_IDXS as T_IDXS_MPC,  # Add this import
  LEAD_ACCEL_TAU
)
from openpilot.selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX, V_CRUISE_UNSET, CONTROL_N, get_speed_error
from openpilot.common.swaglog import cloudlog

LON_MPC_STEP = 0.2  # first step is 0.2s
A_CRUISE_MIN = -8.0
A_CRUISE_MAX_VALS = [4.2, 3.5, 2.8, 1.6, 0.8]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40., 50.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.5
ACCEL_LIMIT_MARGIN = 0.05

# Lookup table for turns
# _A_TOTAL_MAX_V = [1.7, 3.2]
# _A_TOTAL_MAX_BP = [20., 40.]

# Kalman filter states enum
LEAD_KALMAN_SPEED, LEAD_KALMAN_ACCEL = 0, 1

def get_max_accel(v_ego):
  return interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

"""
def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  #This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  #this should avoid accelerating when losing the target in turns
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
  return [a_target[0], min(a_target[1], a_x_allowed)]
"""

def get_accel_from_plan(CP, speeds, accels):
  if len(speeds) == CONTROL_N:
    v_target_now = interp(DT_MDL, CONTROL_N_T_IDX, speeds)
    a_target_now = interp(DT_MDL, CONTROL_N_T_IDX, accels)

    v_target = interp(CP.longitudinalActuatorDelay + DT_MDL, CONTROL_N_T_IDX, speeds)
    a_target = 2 * (v_target - v_target_now) / CP.longitudinalActuatorDelay - a_target_now

    v_target_1sec = interp(CP.longitudinalActuatorDelay + DT_MDL + 1.0, CONTROL_N_T_IDX, speeds)
  else:
    v_target = 0.0
    v_target_1sec = 0.0
    a_target = 0.0
  should_stop = (v_target < CP.vEgoStopping and
                 v_target_1sec < CP.vEgoStopping)
  return a_target, should_stop


def lead_kf(v_lead: float, dt: float = 0.05):
  # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
  # hardcoding a lookup table to compute K for values of radar_ts between 0.01s and 0.2s
  assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
  A = [[1.0, dt], [0.0, 1.0]]
  C = [1.0, 0.0]
  #Q = np.matrix([[10., 0.0], [0.0, 100.]])
  #R = 1e3
  #K = np.matrix([[ 0.05705578], [ 0.03073241]])
  dts = [dt * 0.01 for dt in range(1, 21)]
  # Original K0 and K1 arrays from your code
  K0_base = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
                0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
                0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
                0.35353899, 0.36200124]

  K1_base = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
                0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
                0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
                0.26393339, 0.26278425]

  # Scale factors increased for faster response
  scale_factor_K0 = 2.0  # Increased from 1.5
  scale_factor_K1 = 2.0  # Increased from 1.5

  # Original K0 and K1 arrays with higher gains
  K0 = [k * scale_factor_K0 for k in K0_base]
  K1 = [k * scale_factor_K1 for k in K1_base]

  # Interpolate K0 and K1 based on dt
  K0_interp = interp(dt, dts, K0)
  K1_interp = interp(dt, dts, K1)
  K = [[K0_interp], [K1_interp]]  # Ensure matrix dimensions are correct

  kf = KF1D([[v_lead], [0.0]], A, C, K)
  return kf


class Lead:
  def __init__(self, sm=None):
    self.dRel = 0.0
    self.yRel = 0.0
    self.vLead = 0.0
    self.aLead = 0.0
    self.vLeadK = 0.0
    self.aLeadK = 0.0
    self.aLeadTau = LEAD_ACCEL_TAU
    self.prob = 0.0
    self.status = False
    self.sm = sm  # Initialize sm
    self.curve_urgency = 0.0  # Add missing attribute
    self.safe_speed = 0.0     # Add missing attribute

    self.kf: KF1D | None = None
    self.prev_vLead = 0.0
    self.prev_dRel = 0.0

  def reset(self):
    self.status = False
    self.kf = None
    self.aLeadTau = LEAD_ACCEL_TAU
    self.prev_vLead = 0.0
    self.prev_dRel = 0.0

  def update(self, dRel, yRel, vLead, aLead, prob, sm=None):
    self.sm = sm  # Update state manager reference
    self.dRel = dRel
    self.yRel = yRel
    self.vLead = vLead
    self.aLead = aLead
    self.prob = prob
    self.status = True

    if self.kf is None:
      self.kf = lead_kf(self.vLead)
    else:
      self.kf.update(self.vLead)

    self.vLeadK = float(self.kf.x[LEAD_KALMAN_SPEED][0])
    self.aLeadK = float(self.kf.x[LEAD_KALMAN_ACCEL][0])

    # Enhanced adaptive behavior for rapid deceleration
    relative_velocity = self.vLead - self.prev_vLead
    distance_change = self.dRel - self.prev_dRel

    # Lower thresholds for quicker response
    acceleration_threshold = 0.15  # Reduced from 0.2
    relative_velocity_threshold = 0.15  # Reduced from 0.2
    distance_change_threshold = 0.15  # Reduced from 0.2

    if (abs(self.aLeadK) < acceleration_threshold and
        abs(relative_velocity) < relative_velocity_threshold and
        abs(distance_change) < distance_change_threshold):
      # Slower increase in steady state
      self.aLeadTau = min(self.aLeadTau * 1.02, LEAD_ACCEL_TAU)
    else:
      # More aggressive reduction for rapid changes
      reduction_factor = 0.7  # Increased from 0.8
      self.aLeadTau *= reduction_factor

      # Lower minimum tau for faster response
      self.aLeadTau = max(self.aLeadTau, 0.05 * LEAD_ACCEL_TAU)  # Reduced from 0.1

    self.prev_vLead = self.vLead
    self.prev_dRel = self.dRel

class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.v_model_error = 0.0

    self.lead_one = Lead()
    self.lead_two = Lead()

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.solverExecutionTime = 0.0

    self.prev_dRel = 0.0
    self.prev_v_rel = 0.0
    self.prev_a = init_a
    self.x0 = np.zeros(3)
    self.x0[1] = init_v
    self.x0[2] = init_a

    # Add new curve tracking variables
    self.prev_curvature = 0.0
    self.curvature_rate = 0.0
    self.curve_confidence = 1.0
    self.decel_profile = None
    self.target_speed = 0.0
    self.completion_time = 0.0
    self.curve_weights = 1.0
    self.combined_urgency = 0.0
    self.safe_speed = 0.0

    # Add this to track the model type
    self.secretgoodopenpilot_model = False  # Default to False

    # Remove self.sm = None
    self.lead_states = []  # Add this line

  @staticmethod
  def parse_model(model_msg, model_error, v_ego, taco_tune):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
       len(model_msg.velocity.x) == ModelConstants.IDX_N and
       len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x) - model_error * T_IDXS_MPC
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x) - model_error
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))

    if taco_tune:
      max_lat_accel = interp(v_ego, [5, 10, 20], [2.0, 2.5, 3.0])
      curvatures = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.orientationRate.z) / np.clip(v, 0.3, 100.0)
      max_v = np.sqrt(max_lat_accel / (np.abs(curvatures) + 1e-3)) - 2.0
      v = np.minimum(max_v, v)

    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def calculate_dynamic_response(self, lead):
    """
    Model expert driver response incorporating rate of change
    and vehicle capabilities
    """
    v_ego = self.x0[1]
    v_rel = lead.vLead - v_ego

    # Calculate derivatives
    d_rel_rate = (lead.dRel - self.prev_dRel) / self.dt
    v_rel_rate = (v_rel - self.prev_v_rel) / self.dt

    # Time to collision with dynamic adjustment
    if v_rel < 0 and lead.dRel > 0.5:  # Added minimal distance threshold
        ttc = lead.dRel / abs(v_rel)
    else:
        ttc = float('inf')

    # Further safety checks
    if ttc < 0:
        ttc = float('inf')

    # Required deceleration including velocity trend
    req_decel = (v_ego**2 - lead.vLead**2) / (2 * lead.dRel)
    if v_rel_rate < 0:  # Closure rate increasing
        req_decel *= (1 + abs(v_rel_rate) * 0.5)

    # Vehicle capability factor
    g = 9.81
    max_brake_decel = abs(ACCEL_MIN)
    capability_factor = np.clip(max_brake_decel / req_decel, 0, 1) if req_decel > 0 else 1.0

    # Anticipation factor - earlier response when closing quickly
    anticipation = 1 - np.exp(-abs(v_rel) / 5.0)

    # Combine into urgency metric with continuous derivatives
    tau = 2.0 * (1 + capability_factor)

    # Core urgency calculation incorporating all factors
    urgency = (1 - np.exp(-req_decel/g)) * \
              np.exp(-ttc/tau) * \
              (1 + anticipation) * \
              capability_factor

    # Add lead acceleration influence with hysteresis
    accel_factor = np.clip((-lead.aLeadK / g) * (1.5 if lead.aLeadK < 0 else 1.0), 0, 1)

    # Smooth response using geometric mean
    combined_urgency = np.sqrt(urgency * (1 + accel_factor))

    # Calculate optimal jerk profile
    optimal_jerk = -combined_urgency * max_brake_decel * \
                   (1 - np.exp(-abs(self.prev_a - req_decel)))

    return combined_urgency, optimal_jerk

  def get_safe_speed(self, curve_urgency):
    """
    Calculate a safe speed based on curve urgency.

    Args:
        curve_urgency (float): The urgency factor based on the curve.

    Returns:
        float: The calculated safe speed in m/s.
    """
    # Define speed constants (~120 mph max, ~11 mph min)
    MAX_SAFE_SPEED = 54.0  # m/s (~120 mph)
    MIN_SAFE_SPEED = 5.0   # m/s (~11 mph)

    # Make safety margin proportional to curve urgency
    # Reduced base safety margin to allow higher speeds on straights
    SAFETY_MARGIN = 1.5 * curve_urgency  # More margin in curves, less on straights

    # Calculate safe speed inversely proportional to curve urgency
    safe_speed = MAX_SAFE_SPEED - (curve_urgency * (MAX_SAFE_SPEED - MIN_SAFE_SPEED))
    safe_speed = np.clip(safe_speed, MIN_SAFE_SPEED, MAX_SAFE_SPEED)

    # Apply dynamic safety margin
    return max(safe_speed - SAFETY_MARGIN, MIN_SAFE_SPEED)

  def calculate_curve_response(self, model_position, frogpilot_toggles):
    """
    Calculate curve response with professional driver-like behavior.
    """
    x_points = model_position.x
    y_points = model_position.y

    # Constants for professional driving behavior
    COMFORT_DECEL = 0.6  # Target deceleration rate (m/s²)
    MIN_CURVE_SPEED = 5.0  # Minimum speed in curves (m/s)
    PLANNING_HORIZON = 50.0  # How far ahead to look (m)
    ENTRY_MARGIN = 10.0  # Distance before curve for settling (m)

    max_curvature = 0.0
    distance_to_curve = float('inf')
    curve_entry_point = 0.0

    # First pass: Identify significant curves and their entry points
    for i in range(len(x_points) - 2):
        if x_points[i] > PLANNING_HORIZON:
            break

        # Calculate local curvature using three points
        if i < len(x_points) - 2:
            dx1 = x_points[i+1] - x_points[i]
            dx2 = x_points[i+2] - x_points[i+1]
            dy1 = y_points[i+1] - y_points[i]
            dy2 = y_points[i+2] - y_points[i+1]

            if abs(dx1) > 0.1 and abs(dx2) > 0.1:
                k1 = dy1 / dx1
                k2 = dy2 / dx2
                local_curvature = abs(k2 - k1) / ((dx1 + dx2) / 2.0)

                # If this is the sharpest curve we've seen and it's significant
                if local_curvature > max_curvature and local_curvature > 0.05:
                    max_curvature = local_curvature
                    distance_to_curve = x_points[i] - ENTRY_MARGIN
                    curve_entry_point = x_points[i]

    if max_curvature == 0.0:
        return 0.0, float('inf')  # No significant curves found

    # Calculate required speed for curve
    safe_curve_speed = self.get_safe_speed(max_curvature)

    # Current speed from vehicle state
    v_current = self.x0[1]

    if v_current <= safe_curve_speed:
        return 0.0, float('inf')  # No deceleration needed

    # Calculate deceleration profile
    # Use s = ut + (1/2)at² to determine when to start braking
    # Rearranged to solve for required distance
    speed_delta = v_current - safe_curve_speed

    # Required distance to achieve speed reduction with comfort decel
    required_distance = (v_current * v_current - safe_curve_speed * safe_curve_speed) / (2 * COMFORT_DECEL)

    # Calculate urgency based on available vs required distance
    distance_ratio = distance_to_curve / max(required_distance, 0.1)

    # Progressive urgency that ramps up if we're getting close to required braking distance
    if distance_ratio > 1.5:  # We have plenty of distance
        curve_urgency = 0.0
    elif distance_ratio < 0.8:  # We're behind schedule
        curve_urgency = min((0.8 - distance_ratio) * 2.0, 1.0)
    else:  # We're in the sweet spot for initiating deceleration
        curve_urgency = math.pow((1.5 - distance_ratio) / 0.7, 2.0)

    # Calculate target speed based on urgency and curve speed
    target_speed = safe_curve_speed + (v_current - safe_curve_speed) * (1 - curve_urgency)

    # Add hysteresis to prevent oscillation
    if abs(v_current - target_speed) < 1.0:  # 1 m/s deadband
        target_speed = v_current

    return curve_urgency, target_speed

  def update_curvature(self, x_points, y_points):
    """
    Enhanced version of current VTSC curvature calculation with improved confidence metrics
    """
    # Calculate curvature using three-point method
    if len(x_points) >= 3 and len(y_points) >= 3:
        x1, x2, x3 = x_points[0:3]
        y1, y2, y3 = y_points[0:3]

        # Enhanced curvature calculation with better noise handling
        dx1 = x2 - x1
        dx2 = x3 - x2
        dy1 = y2 - y1
        dy2 = y3 - y2

        # More stringent division checks
        if abs(dx1) > 1e-6 and abs(dx2) > 1e-6 and abs((dx1 + dx2) / 2.0) > 1e-6:
            k1 = dy1 / dx1
            k2 = dy2 / dx2
            curvature = (k2 - k1) / ((dx1 + dx2) / 2.0)

            # Update curvature rate
            self.curvature_rate = (curvature - self.prev_curvature) / self.dt
            self.prev_curvature = curvature

            # Enhanced confidence metrics
            # 1. Noise factor based on rate of change
            noise_factor = np.clip(1.0 - abs(self.curvature_rate), 0.5, 1.0)

            # 2. Path straightness metric
            path_variance = np.std(y_points) / (np.std(x_points) + 1e-6)
            straightness_factor = np.clip(1.0 - path_variance, 0.3, 1.0)

            # 3. Point density metric
            path_length = x_points[-1] - x_points[0]
            if path_length > 0:
                point_density = len(x_points) / path_length
                density_factor = np.clip(point_density / 10.0, 0.5, 1.0)
            else:
                density_factor = 0.5

            # 4. Temporal consistency
            temporal_factor = np.clip(1.0 - min(abs(self.curvature_rate), 1.0), 0.4, 1.0)

            # Combine confidence metrics
            self.curve_confidence = np.mean([
                noise_factor * 1.0,
                straightness_factor * 1.2,
                density_factor * 0.8,
                temporal_factor * 1.0
            ])

            return curvature

    self.curve_confidence = 0.0
    return 0.0

  def calculate_deceleration_profile(self, current_speed, target_speed, distance):
    """
    Calculate smooth deceleration profile
    """
    if distance <= 0 or current_speed <= target_speed:
        return 0.0, 0.0

    # Constants for smooth profile
    COMFORT_JERK = 0.3  # m/s³ - Rate of decel increase
    MAX_DECEL = 0.7  # m/s² - Upper limit
    MIN_DECEL = 0.5  # m/s² - Lower limit

    # Calculate required average deceleration
    avg_decel = (current_speed * current_speed - target_speed * target_speed) / (2.0 * distance)

    # Clamp to comfortable limits
    decel_rate = np.clip(avg_decel, MIN_DECEL, MAX_DECEL)

    # Calculate time to complete maneuver
    completion_time = 2.0 * distance / (current_speed + target_speed)

    return decel_rate, completion_time

  def update(self, radarless_model, secretgoodopenpilot_model, sm, frogpilot_toggles):
    # Store the model type
    self.secretgoodopenpilot_model = secretgoodopenpilot_model

    self.mpc.mode = 'blended' if sm['controlsState'].experimentalMode else 'acc'

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['controlsState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['controlsState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['controlsState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    accel_limits = [sm['frogpilotPlan'].minAcceleration, sm['frogpilotPlan'].maxAcceleration]
    # if self.mpc.mode == 'acc':
      # accel_limits_turns = limit_accel_in_turns(v_ego, sm['carState'].steeringAngleDeg, accel_limits, self.CP)
    # else:
    accel_limits_turns = [ACCEL_MIN, ACCEL_MAX]

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Allow stronger deceleration even when becoming active
      self.a_desired = clip(sm['carState'].aEgo, accel_limits[0] * 1.2, accel_limits[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    # Compute model v_ego error
    self.v_model_error = get_speed_error(sm['modelV2'], v_ego)
    x, v, a, j, throttle_prob = self.parse_model(sm['modelV2'], self.v_model_error, v_ego, frogpilot_toggles.taco_tune)
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD

    if not self.allow_throttle and v_ego > 5.0 and self.secretgoodopenpilot_model:  # Don't clip at low speeds since throttle_prob doesn't account for creep
      # MPC breaks when accel limits would cause negative velocity within the MPC horizon, so we clip the max accel limit at vEgo/T_MAX plus a bit of margin
      clipped_accel_coast = max(accel_coast, accel_limits_turns[0], -v_ego / T_IDXS_MPC[-1] + ACCEL_LIMIT_MARGIN)
      accel_limits_turns[1] = min(accel_limits_turns[1], clipped_accel_coast)

    if force_slow_decel:
      v_cruise = 0.0
    # clip limits, cannot init MPC outside of bounds
    accel_limits_turns[0] = min(accel_limits_turns[0], self.a_desired + 0.1)  # Increased from 0.05
    accel_limits_turns[1] = max(accel_limits_turns[1], self.a_desired - 0.1)  # Increased from 0.05

    if radarless_model:
      model_leads = list(sm['modelV2'].leadsV3)
      while len(self.lead_states) < len(model_leads):
          self.lead_states.append(Lead(sm))
      while len(self.lead_states) > len(model_leads):
          self.lead_states.pop()

      for index, model_lead in enumerate(model_leads):
          self.lead_states[index].update(
              model_lead.x[0],
              model_lead.y[0],
              model_lead.v[0],
              model_lead.a[0],
              model_lead.prob,
              sm
          )
      self.lead_one = self.lead_states[0] if self.lead_states else Lead(sm)
      self.lead_two = self.lead_states[1] if len(self.lead_states) > 1 else Lead(sm)
    else:
      self.lead_one = sm['radarState'].leadOne
      self.lead_two = sm['radarState'].leadTwo

    if self.lead_one.status:
        # Only calculate dynamic response for radarless mode where lead_one is our custom Lead class
        if radarless_model:
            urgency, optimal_jerk = self.calculate_dynamic_response(self.lead_one)
            v_rel = self.lead_one.vLead - self.x0[1]

            # Dynamic weight adjustment
            acceleration_jerk = 1.0 / (1 + urgency**2)
            danger_jerk = 1.0 + urgency * 2.0
            speed_jerk = np.clip(1.0 - urgency, 0.5, 1.0)

            # Update MPC weights
            self.mpc.set_weights(
                acceleration_jerk=acceleration_jerk,
                danger_jerk=danger_jerk,
                speed_jerk=speed_jerk,
                personality=sm['controlsState'].personality
            )

            # Update lead tau with hysteresis - only in radarless mode
            target_tau = LEAD_ACCEL_TAU * (1 - 0.8 * urgency)
            if target_tau < self.lead_one.aLeadTau:
                self.lead_one.aLeadTau = target_tau
            else:
                self.lead_one.aLeadTau += (target_tau - self.lead_one.aLeadTau) * 0.2

            # Store states for next iteration
            self.prev_dRel = self.lead_one.dRel
            self.prev_v_rel = v_rel
        else:
            # For radar-based system, use simpler logic
            v_rel = self.lead_one.vRel
            self.prev_dRel = self.lead_one.dRel
            self.prev_v_rel = v_rel

    # Add curve handling section
    if len(sm['modelV2'].position.x) > 0:
        curve_urgency, safe_speed = self.calculate_curve_response(
            sm['modelV2'].position,
            frogpilot_toggles
        )

        # Integrate curve response with MPC weights
        self.mpc.set_weights(
            acceleration_jerk=sm['frogpilotPlan'].accelerationJerk * (1.0 + curve_urgency),
            danger_jerk=sm['frogpilotPlan'].dangerJerk * (1.0 + curve_urgency * 0.5),
            speed_jerk=sm['frogpilotPlan'].speedJerk,
            personality=sm['controlsState'].personality
        )

        # Adjust speed target for curves
        v_target = min(v_cruise, safe_speed)
    else:
        v_target = v_cruise

    # Update MPC with more aggressive jerk factors for rapid response
    self.mpc.set_weights(
        acceleration_jerk=sm['frogpilotPlan'].accelerationJerk * 1.2,  # Increase base jerk
        danger_jerk=sm['frogpilotPlan'].dangerJerk * 1.5,              # Increase danger jerk
        speed_jerk=sm['frogpilotPlan'].speedJerk,
        personality=sm['controlsState'].personality
    )
    self.mpc.set_accel_limits(accel_limits_turns[0], accel_limits_turns[1])
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(self.lead_one, self.lead_two, v_target, x, v, a, j, radarless_model,
                    sm['frogpilotPlan'].tFollow, sm['frogpilotCarState'].trafficModeActive,
                    frogpilot_toggles, personality=sm['controlsState'].personality)

    self.a_desired_trajectory_full = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = self.lead_one.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    a_target, should_stop = get_accel_from_plan(self.CP, longitudinalPlan.speeds, longitudinalPlan.accels)
    longitudinalPlan.aTarget = a_target
    longitudinalPlan.shouldStop = should_stop
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = self.allow_throttle

    pm.send('longitudinalPlan', plan_send)
