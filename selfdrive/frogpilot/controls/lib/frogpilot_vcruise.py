from openpilot.common.conversions import Conversions as CV
from openpilot.common.numpy_fast import clip
from openpilot.common.realtime import DT_MDL

from openpilot.selfdrive.controls.controlsd import ButtonType
from openpilot.selfdrive.controls.lib.drive_helpers import V_CRUISE_UNSET

from openpilot.selfdrive.frogpilot.controls.lib.map_turn_speed_controller import MapTurnSpeedController
from openpilot.selfdrive.frogpilot.controls.lib.speed_limit_controller import SpeedLimitController
from openpilot.selfdrive.frogpilot.frogpilot_variables import CRUISING_SPEED, PLANNER_TIME

import numpy as np
import cereal.messaging as messaging
from cereal import log

LaneChangeState = log.LaneChangeState

TARGET_LAT_A = 2.0

class VTSCKalmanFilter:
  def __init__(self, dt=0.1):
    self.dt = dt

    # State: [target_speed, speed_rate_of_change]
    self.A = np.array([[1.0, dt],    # State transition matrix
                      [0.0, 1.0]])

    self.C = np.array([1.0, 0.0])    # Measurement matrix

    # Initial state uncertainty
    self.P = np.array([[10.0, 0.0],  # Higher initial uncertainty
                      [0.0, 10.0]])

    # Process noise (tune these)
    self.Q = np.array([[0.1, 0.0],   # Speed process noise
                      [0.0, 0.5]])   # Acceleration process noise

    # Measurement noise (tune this)
    self.R = 5.0                     # Higher value = more smoothing

    self.x = np.array([[0.0],        # Initial speed
                      [0.0]])        # Initial rate of change

  def update(self, measured_speed, curve_confidence):
    # Adjust measurement noise based on curve confidence
    R_adjusted = self.R / max(curve_confidence, 0.1)

    # Predict step
    self.x = np.dot(self.A, self.x)
    self.P = np.dot(np.dot(self.A, self.P), np.transpose(self.A)) + self.Q

    # Update step
    S = np.dot(np.dot(self.C, self.P), np.transpose(self.C)) + R_adjusted
    K = np.dot(self.P, np.transpose(self.C).reshape(-1, 1)) / S

    y = measured_speed - np.dot(self.C, self.x)
    self.x = self.x + K * y
    self.P = self.P - np.dot(np.dot(K, self.C.reshape(1, -1)), self.P)

    return float(self.x[0])

class FrogPilotVCruise:
  def __init__(self, FrogPilotPlanner):
    self.frogpilot_planner = FrogPilotPlanner

    self.adjusted_target_lat_a = TARGET_LAT_A

    self.params_memory = self.frogpilot_planner.params_memory

    self.mtsc = MapTurnSpeedController()
    self.slc = SpeedLimitController()

    self.sm = messaging.SubMaster(['modelV2'])
    self.lane_change_state = LaneChangeState.off
    self.base_curvature = 0.0
    self.lc_curvature_offset = 0.0
    self.curvature_rate = 0.0
    self.last_curvature = 0.0
    self.curvature_confidence = 1.0

    self.forcing_stop = False
    self.override_force_stop = False
    self.override_slc = False
    self.speed_limit_changed = False

    self.model_length = 0
    self.mtsc_target = 0
    self.overridden_speed = 0
    self.previous_speed_limit = 0
    self.slc_target = 0
    self.speed_limit_timer = 0
    self.tracked_model_length = 0
    self.vtsc_target = 0

    # Initialize variables for VTSC rate limiting and apex detection
    self.initial_v_cruise = V_CRUISE_UNSET  # Stores the original cruise speed before curve
    self.vtsc_rate_limited_target = 0.0     # Rate-limited target speed
    self.vtsc_max_accel = 0.5               # Maximum acceleration in m/s^2 (adjust as needed)
    self.curvature_derivative = 0.0         # Derivative of curvature
    self.prev_estimated_base_curvature = 0.0
    self.apex_reached = False               # Flag to indicate if apex has been reached
    self.apex_speed = 0.0                   # Speed at the apex
    self.time_since_apex = 0.0
    self.deceleration_distance = float('inf')  # Initialize with infinity
    self.deceleration_total_time = 0.0
    self.deceleration_initial_speed = 0.0

    # Initialize variables for deceleration easing
    self.deceleration_started = False       # Flag indicating if deceleration has started
    self.time_since_deceleration = 0.0      # Timer for deceleration easing

    self.kf = VTSCKalmanFilter()
    self.curve_confidence = 1.0

    # Add new attributes for curve handling
    self.prev_curvature = 0.0
    self.curvature_rate = 0.0
    self.curve_confidence = 1.0

  def estimate_base_curvature(self, current_curvature, dt):
    self.curvature_rate = (current_curvature - self.last_curvature) / dt
    self.last_curvature = current_curvature

    alpha = 0.2
    self.base_curvature = alpha * current_curvature + (1 - alpha) * self.base_curvature

    self.curvature_derivative = (self.base_curvature - self.prev_estimated_base_curvature) / dt
    self.prev_estimated_base_curvature = self.base_curvature

    return max(self.base_curvature, 0.0001)

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

  def get_safe_speed(self, curve_urgency):
    """
    Calculate a safe speed based on curve urgency.
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

  def calculate_deceleration_rate(self, curvature):
    base_deceleration = -0.25
    curvature_factor = min(abs(curvature) * 1000, 1)  # Normalize curvature influence
    return base_deceleration * (1 + curvature_factor)

  def find_critical_curve(self, curvatures, distances, v_ego, frogpilot_toggles):
    critical_curve_index = 0
    min_safe_speed = float('inf')
    self.adjusted_target_lat_a = TARGET_LAT_A * frogpilot_toggles.turn_aggressiveness

    for i, (curve, dist) in enumerate(zip(curvatures, distances)):
      lookahead_time = 6.0  #seconds
      max_lookahead_distance = v_ego * lookahead_time

      if dist > max_lookahead_distance:
        break

      safe_speed = np.sqrt(self.adjusted_target_lat_a / max(curve, 0.0001))
      if safe_speed < min_safe_speed:
        min_safe_speed = safe_speed
        critical_curve_index = i

    return critical_curve_index, min_safe_speed

  def smooth_target_speed(self, current_target, new_target, smoothing_factor=0.1):
    return current_target + smoothing_factor * (new_target - current_target)

  def exponential_decay_ease(self, current_speed, target_speed, time_elapsed, rate_constant):
    speed_diff = current_speed - target_speed
    eased_speed = target_speed + speed_diff * np.exp(-rate_constant * time_elapsed)
    return eased_speed

  def sigmoid_ease(self, current_speed, target_speed, time_elapsed, total_time):
    speed_diff = current_speed - target_speed
    midpoint = total_time / 2.0
    k = 10.0 / total_time  # Adjust k for desired smoothness
    eased_speed = target_speed + speed_diff / (1 + np.exp(k * (time_elapsed - midpoint)))
    return eased_speed

  def update(self, carState, controlsState, frogpilotCarControl, frogpilotCarState, frogpilotNavigation, modelData, v_cruise, v_ego, frogpilot_toggles):
    self.override_force_stop |= carState.gasPressed
    self.override_force_stop |= frogpilot_toggles.force_stops and carState.standstill and self.frogpilot_planner.tracking_lead
    self.override_force_stop |= frogpilotCarControl.resumePressed

    road_curvature = self.frogpilot_planner.road_curvature * frogpilot_toggles.curve_sensitivity

    v_cruise_cluster = max(controlsState.vCruiseCluster, v_cruise) * CV.KPH_TO_MS
    v_cruise_diff = v_cruise_cluster - v_cruise

    v_ego_cluster = max(carState.vEgoCluster, v_ego)
    v_ego_diff = v_ego_cluster - v_ego

    # Pfeiferj's Map Turn Speed Controller
    if frogpilot_toggles.map_turn_speed_controller and v_ego > CRUISING_SPEED and controlsState.enabled:
      mtsc_active = self.mtsc_target < v_cruise
      self.mtsc_target = clip(self.mtsc.target_speed(v_ego, carState.aEgo, frogpilot_toggles), CRUISING_SPEED, v_cruise)

      curve_detected = (1 / road_curvature)**0.5 < v_ego
      if curve_detected and mtsc_active:
        self.mtsc_target = self.frogpilot_planner.v_cruise
      elif not curve_detected and frogpilot_toggles.mtsc_curvature_check:
        self.mtsc_target = v_cruise

      if self.mtsc_target == CRUISING_SPEED:
        self.mtsc_target = v_cruise
    else:
      self.mtsc_target = v_cruise if v_cruise != V_CRUISE_UNSET else 0

    # SLC
    if frogpilot_toggles.speed_limit_controller:
      self.slc.update(frogpilotCarState.dashboardSpeedLimit, controlsState.enabled, frogpilotNavigation.navigationSpeedLimit, v_cruise, v_ego, frogpilot_toggles)
      unconfirmed_slc_target = self.slc.desired_speed_limit

      if (frogpilot_toggles.speed_limit_confirmation_lower or frogpilot_toggles.speed_limit_confirmation_higher) and self.slc_target != 0:
        self.speed_limit_changed = unconfirmed_slc_target != self.previous_speed_limit and abs(self.slc_target - unconfirmed_slc_target) > 1

        speed_limit_decreased = self.speed_limit_changed and self.slc_target > unconfirmed_slc_target
        speed_limit_increased = self.speed_limit_changed and self.slc_target < unconfirmed_slc_target

        accepted_via_ui = self.params_memory.get_bool("SLCConfirmedPressed") and self.params_memory.get_bool("SLCConfirmed")
        denied_via_ui = self.params_memory.get_bool("SLCConfirmedPressed") and not self.params_memory.get_bool("SLCConfirmed")

        speed_limit_accepted = frogpilotCarControl.resumePressed or accepted_via_ui
        speed_limit_denied = any(be.type == ButtonType.decelCruise for be in carState.buttonEvents) or denied_via_ui or self.speed_limit_timer >= 10

        if speed_limit_accepted or speed_limit_denied:
          self.previous_speed_limit = unconfirmed_slc_target
          self.params_memory.put_bool("SLCConfirmed", False)
          self.params_memory.put_bool("SLCConfirmedPressed", False)

        if speed_limit_decreased:
          speed_limit_confirmed = not frogpilot_toggles.speed_limit_confirmation_lower or speed_limit_accepted
        elif speed_limit_increased:
          speed_limit_confirmed = not frogpilot_toggles.speed_limit_confirmation_higher or speed_limit_accepted
        else:
          speed_limit_confirmed = False

        if self.speed_limit_changed:
          self.speed_limit_timer += DT_MDL
        else:
          self.speed_limit_timer = 0

        if speed_limit_confirmed:
          self.slc_target = unconfirmed_slc_target
          self.speed_limit_changed = False
      else:
        self.slc_target = unconfirmed_slc_target

      self.override_slc = self.overridden_speed > self.slc_target
      self.override_slc |= carState.gasPressed and v_ego > self.slc_target
      self.override_slc &= controlsState.enabled

      if self.override_slc:
        if frogpilot_toggles.speed_limit_controller_override_manual:
          if carState.gasPressed:
            self.overridden_speed = v_ego + v_ego_diff
          self.overridden_speed = clip(self.overridden_speed, self.slc_target, v_cruise + v_cruise_diff)
        elif frogpilot_toggles.speed_limit_controller_override_set_speed:
          self.overridden_speed = v_cruise + v_cruise_diff
      else:
        self.overridden_speed = 0
    else:
      self.slc_target = 0

    # VTSC
    self.sm.update()
    prev_lane_change_state = self.lane_change_state
    self.lane_change_state = self.sm['modelV2'].meta.laneChangeState

    dt = 0.1  # Time step
    current_curvature = self.frogpilot_planner.road_curvature
    estimated_base_curvature = self.estimate_base_curvature(current_curvature, dt)

    if frogpilot_toggles.vision_turn_controller and v_ego > CRUISING_SPEED and controlsState.enabled:
      # Get future positions from the model
      positions = self.sm['modelV2'].position
      x = np.array(positions.x)
      y = np.array(positions.y)

      # Dynamically update curvature
      curvature = self.update_curvature(x, y)

      # Check if curvature data is valid
      if curvature is not None:
        # Calculate cumulative distances
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        cumulative_distances = np.cumsum(distances)
        cumulative_distances = np.insert(cumulative_distances, 0, 0)  # Add 0 at the beginning for the current position

        # Detect upcoming curves
        curvature_threshold = 0.0001  # Adjust as needed
        curve_indices = np.where(abs(curvature) > curvature_threshold)[0]

        if len(curve_indices) > 0:
          critical_index, desired_vtsc_target = self.find_critical_curve(
            curvature[curve_indices], cumulative_distances[curve_indices], v_ego, frogpilot_toggles)
          curve_start_index = curve_indices[critical_index]
          distance_to_curve = cumulative_distances[curve_start_index]

          # Adjusted curvature based on user settings
          adjusted_base_curvature = curvature[curve_start_index] * frogpilot_toggles.curve_sensitivity
          self.adjusted_target_lat_a = TARGET_LAT_A * frogpilot_toggles.turn_aggressiveness

          # Calculate desired speed for the curve
          raw_vtsc_target = np.sqrt(self.adjusted_target_lat_a / max(adjusted_base_curvature, 0.0001))

          # Calculate curve confidence and apply Kalman filter
          distance_factor = np.clip(1.0 - (distance_to_curve / self.deceleration_distance), 0.1, 1.0)
          curvature_factor = np.clip(abs(adjusted_base_curvature) * 1000, 0.1, 1.0)
          self.curve_confidence = distance_factor * curvature_factor

          # Use Kalman filter to smooth the target speed
          filtered_target = self.kf.update(raw_vtsc_target, self.curve_confidence)
          desired_vtsc_target = clip(filtered_target, CRUISING_SPEED, v_cruise)

          v_initial = v_ego
          v_final = desired_vtsc_target
          speed_diff = v_initial - v_final

          # Determine if we're approaching the apex or exiting the curve
          curvature_derivative = np.gradient(curvature)
          apex_index = np.argmin(curvature_derivative[curve_indices])

          # Calculate distance to apex
          apex_distance = cumulative_distances[curve_indices[apex_index]]

          # === Modified Section Start ===
          # Increase the deceleration_total_time by 1 second for earlier and gentler deceleration
          base_deceleration_total_time = max(5.0, min(15.0, abs(speed_diff) / 0.1)) + 1.0  # Added 1.0 second
          # === Modified Section End ===

          # Calculate adjusted deceleration_total_time to hit target speed 1 second before critical curve
          adjusted_deceleration_total_time = base_deceleration_total_time
          # Compensate for adjusted decel time
          additional_distance = v_ego * 1.0  # distance = speed * time
          # Update deceleration_distance to include the additional distance
          self.deceleration_distance = (v_initial + v_final) / 2.0 * adjusted_deceleration_total_time + additional_distance

          if distance_to_curve <= self.deceleration_distance and not self.apex_reached:
            if not self.deceleration_started:
              self.time_since_deceleration = 0.0  # Reset timer
              self.deceleration_total_time = adjusted_deceleration_total_time
              self.deceleration_initial_speed = v_ego
              # self.deceleration_distance already includes additional_distance
            self.deceleration_started = True
            self.time_since_deceleration += dt

            # Apply sigmoid easing function for deceleration
            self.vtsc_rate_limited_target = self.sigmoid_ease(
              current_speed=self.deceleration_initial_speed,
              target_speed=desired_vtsc_target,
              time_elapsed=self.time_since_deceleration,
              total_time=self.deceleration_total_time
            )
            # Ensure target speed does not fall below desired speed
            self.vtsc_rate_limited_target = max(self.vtsc_rate_limited_target, desired_vtsc_target)

            # Check if we've reached the apex
            if distance_to_curve <= apex_distance:
              self.apex_reached = True
              self.apex_speed = v_ego
              self.time_since_apex = 0.0
          elif self.apex_reached:
            # Acceleration phase after the apex
            self.time_since_apex += dt
            acceleration_time = 2.0  # Time to accelerate back to cruise speed

            # Apply sigmoid easing for acceleration
            self.vtsc_rate_limited_target = self.sigmoid_ease(
              current_speed=self.apex_speed,
              target_speed=v_cruise,
              time_elapsed=self.time_since_apex,
              total_time=acceleration_time
            )
            self.vtsc_rate_limited_target = min(self.vtsc_rate_limited_target, v_cruise)

            # Reset deceleration variables if we have reached cruise speed
            if self.vtsc_rate_limited_target >= v_cruise:
              self.apex_reached = False
              self.deceleration_started = False
              self.time_since_deceleration = 0.0
              self.deceleration_distance = float('inf')  # Reset deceleration distance
          else:
            # Maintain current speed until it's time to decelerate
            self.vtsc_rate_limited_target = v_cruise
            self.deceleration_started = False
            self.apex_reached = False
            self.time_since_deceleration = 0.0
            self.time_since_apex = 0.0
            self.deceleration_distance = float('inf')  # Reset deceleration distance

          # Update the target speed
          self.vtsc_target = self.vtsc_rate_limited_target
        else:
          # No curve ahead detected
          self.vtsc_target = v_cruise
          self.vtsc_rate_limited_target = self.vtsc_target
          self.deceleration_started = False
          self.apex_reached = False
          self.time_since_deceleration = 0.0
          self.time_since_apex = 0.0
          self.deceleration_distance = float('inf')  # Reset deceleration distance
      else:
        # VTSC not active or conditions not met
        self.vtsc_target = v_cruise if v_cruise != V_CRUISE_UNSET else float('inf')
        self.vtsc_rate_limited_target = self.vtsc_target
        self.deceleration_started = False
        self.apex_reached = False
        self.time_since_deceleration = 0.0
        self.time_since_apex = 0.0
        self.deceleration_distance = float('inf')  # Reset deceleration distance

    if frogpilot_toggles.force_standstill and carState.standstill and not self.override_force_stop and controlsState.enabled:
      self.forcing_stop = True
      v_cruise = -1

    elif frogpilot_toggles.force_stops and self.frogpilot_planner.cem.stop_light_detected and not self.override_force_stop and controlsState.enabled:
      if self.tracked_model_length == 0:
        self.tracked_model_length = self.model_length

      self.forcing_stop = True
      self.tracked_model_length -= v_ego * DT_MDL
      v_cruise = min((self.tracked_model_length / PLANNER_TIME) - 1, v_cruise)

    else:
      if not self.frogpilot_planner.cem.stop_light_detected:
        self.override_force_stop = False

      self.forcing_stop = False
      self.tracked_model_length = 0

      targets = [self.mtsc_target, max(self.overridden_speed, self.slc_target) - v_ego_diff, self.vtsc_target]
      v_cruise = float(min([target if target > CRUISING_SPEED else v_cruise for target in targets]))

    # Add curve handling section
    if len(modelData.position.x) > 0:
        curve_urgency, safe_speed = self.calculate_curve_response(
            modelData.position,
            frogpilot_toggles
        )

        # Adjust v_cruise based on curve response
        v_cruise = min(v_cruise, safe_speed)

    return v_cruise
