from openpilot.common.numpy_fast import clip, interp

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import COMFORT_BRAKE, get_jerk_factor, get_safe_obstacle_distance, get_stopped_equivalence_factor, get_T_FOLLOW

from openpilot.selfdrive.frogpilot.frogpilot_variables import CITY_SPEED_LIMIT, CRUISING_SPEED

TRAFFIC_MODE_BP = [0., CITY_SPEED_LIMIT]

class FrogPilotFollowing:
  def __init__(self, FrogPilotPlanner):
    self.frogpilot_planner = FrogPilotPlanner

    self.following_lead = False
    self.slower_lead = False

    self.acceleration_jerk = 0
    self.base_acceleration_jerk = 0
    self.base_speed_jerk = 0
    self.danger_jerk = 0
    self.safe_obstacle_distance = 0
    self.safe_obstacle_distance_stock = 0
    self.speed_jerk = 0
    self.stopped_equivalence_factor = 0
    self.t_follow = 0

  def update(self, aEgo, controlsState, frogpilotCarState, lead_distance, stopping_distance, v_ego, v_lead, frogpilot_toggles):
    # Determine jerk factors based on traffic mode and vehicle acceleration
    if frogpilotCarState.trafficModeActive:
      if aEgo >= 0:
        self.base_acceleration_jerk = interp(v_ego, TRAFFIC_MODE_BP, frogpilot_toggles.traffic_mode_jerk_acceleration)
        self.base_speed_jerk = interp(v_ego, TRAFFIC_MODE_BP, frogpilot_toggles.traffic_mode_jerk_speed)
      else:
        self.base_acceleration_jerk = interp(v_ego, TRAFFIC_MODE_BP, frogpilot_toggles.traffic_mode_jerk_deceleration)
        self.base_speed_jerk = interp(v_ego, TRAFFIC_MODE_BP, frogpilot_toggles.traffic_mode_jerk_speed_decrease)

      self.base_danger_jerk = interp(v_ego, TRAFFIC_MODE_BP, frogpilot_toggles.traffic_mode_jerk_danger)
      self.t_follow = interp(v_ego, TRAFFIC_MODE_BP, frogpilot_toggles.traffic_mode_t_follow)

    else:
      if aEgo >= 0:
        self.base_acceleration_jerk, self.base_danger_jerk, self.base_speed_jerk = get_jerk_factor(
          frogpilot_toggles.aggressive_jerk_acceleration,
          frogpilot_toggles.aggressive_jerk_danger,
          frogpilot_toggles.aggressive_jerk_speed,
          frogpilot_toggles.standard_jerk_acceleration,
          frogpilot_toggles.standard_jerk_danger,
          frogpilot_toggles.standard_jerk_speed,
          frogpilot_toggles.relaxed_jerk_acceleration,
          frogpilot_toggles.relaxed_jerk_danger,
          frogpilot_toggles.relaxed_jerk_speed,
          frogpilot_toggles.custom_personalities,
          controlsState.personality
        )
      else:
        self.base_acceleration_jerk, self.base_danger_jerk, self.base_speed_jerk = get_jerk_factor(
          frogpilot_toggles.aggressive_jerk_deceleration,
          frogpilot_toggles.aggressive_jerk_danger,
          frogpilot_toggles.aggressive_jerk_speed_decrease,
          frogpilot_toggles.standard_jerk_deceleration,
          frogpilot_toggles.standard_jerk_danger,
          frogpilot_toggles.standard_jerk_speed_decrease,
          frogpilot_toggles.relaxed_jerk_deceleration,
          frogpilot_toggles.relaxed_jerk_danger,
          frogpilot_toggles.relaxed_jerk_speed_decrease,
          frogpilot_toggles.custom_personalities,
          controlsState.personality
        )

      self.t_follow = get_T_FOLLOW(
        frogpilot_toggles.aggressive_follow,
        frogpilot_toggles.standard_follow,
        frogpilot_toggles.relaxed_follow,
        frogpilot_toggles.custom_personalities,
        controlsState.personality
      )

    # Set current jerk values
    self.acceleration_jerk = self.base_acceleration_jerk
    self.danger_jerk = self.base_danger_jerk
    self.speed_jerk = self.base_speed_jerk

    # Determine if the ego vehicle is following the lead
    self.following_lead = self.frogpilot_planner.tracking_lead and lead_distance < (self.t_follow + 1) * v_ego

    if self.frogpilot_planner.tracking_lead:
      self.safe_obstacle_distance = int(get_safe_obstacle_distance(v_ego, self.t_follow, dynamic_brake))
      self.safe_obstacle_distance_stock = self.safe_obstacle_distance

      # Retrieve lead acceleration (a_lead) from frogpilotCarState
      if hasattr(frogpilotCarState, 'lead_vehicle') and frogpilotCarState.lead_vehicle.status:
        a_lead = frogpilotCarState.lead_vehicle.aLeadK  # Replace with the correct attribute if different
      else:
        a_lead = 0.0  # Default value when no lead vehicle is detected

      # Update stopped equivalence factor with both v_lead and a_lead
      try:
        self.stopped_equivalence_factor = int(get_stopped_equivalence_factor(v_lead, a_lead))
      except Exception as e:
        # Handle the exception, possibly log it and set a default value
        from openpilot.common.swaglog import cloudlog
        cloudlog.error(f"Error calculating stopped_equivalence_factor: {e}")
        self.stopped_equivalence_factor = int(get_stopped_equivalence_factor(v_lead, COMFORT_BRAKE))

      # Update follow-related values
      self.update_follow_values(lead_distance, stopping_distance, v_ego, v_lead, frogpilot_toggles)
    else:
      self.safe_obstacle_distance = 0
      self.safe_obstacle_distance_stock = 0
      self.stopped_equivalence_factor = 0

  def update_follow_values(self, lead_distance, stopping_distance, v_ego, v_lead, frogpilot_toggles):
    # Disabled FrogAI modifications while testing core following behavior
    if frogpilot_toggles.human_following and v_lead > v_ego:
      pass  # Removed faster lead logic

    if (frogpilot_toggles.conditional_slower_lead or frogpilot_toggles.human_following) and v_lead < v_ego:
      pass  # Removed slower lead logic
      self.slower_lead = False  # Maintain expected state variable
