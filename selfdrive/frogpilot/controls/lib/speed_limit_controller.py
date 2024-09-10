# PFEIFER - SLC - Modified by FrogAi for FrogPilot, graceful decel logic by chris.carlo
import json
import math
import time

from openpilot.common.conversions import Conversions as CV
from openpilot.common.params import Params

from openpilot.selfdrive.frogpilot.controls.lib.frogpilot_variables import FrogPilotVariables

R = 6373000.0  # approximate radius of earth in meters
TO_RADIANS = math.pi / 180

# points should be in radians
# output is meters
def distance_to_point(ax, ay, bx, by):
  a = math.sin((bx - ax) / 2) * math.sin((bx - ax) / 2) + math.cos(ax) * math.cos(bx) * math.sin((by - ay) / 2) * math.sin((by - ay) / 2)
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  return R * c  # in meters

class SpeedLimitController:
  def __init__(self):
    self.frogpilot_toggles = FrogPilotVariables.toggles
    FrogPilotVariables.update_frogpilot_params()

    self.params = Params()
    self.params_memory = Params("/dev/shm/params")

    self.car_speed_limit = 0  # m/s
    self.map_speed_limit = 0  # m/s
    self.max_speed_limit = 0  # m/s
    self.nav_speed_limit = 0  # m/s
    self.prv_speed_limit = self.params.get_float("PreviousSpeedLimit")
    self.transition_start_time = 0
    self.transition_start_speed = 0
    self.transition_target_speed = 0
    self.gentle_decel_rate = 0.67  # m/s

  def get_param_memory(self, key, is_json=False):
    param_value = self.params_memory.get(key)
    if param_value is None:
      return {} if is_json else 0.0
    return json.loads(param_value) if is_json else float(param_value)

  def update_previous_limit(self, speed_limit):
    if self.prv_speed_limit != speed_limit:
      self.params.put_float_nonblocking("PreviousSpeedLimit", speed_limit)
      self.prv_speed_limit = speed_limit

  def update(self, dashboardSpeedLimit, enabled, navigationSpeedLimit, v_cruise, v_ego, frogpilot_toggles):
    self.car_speed_limit = dashboardSpeedLimit
    self.write_map_state(v_ego)
    self.nav_speed_limit = navigationSpeedLimit

    self.max_speed_limit = v_cruise if enabled else 0

    self.frogpilot_toggles = frogpilot_toggles

    new_limit = self.speed_limit
    if new_limit > 1:
      target_speed = new_limit + self.offset
      if target_speed < v_ego and (self.transition_start_time == 0 or target_speed < self.transition_target_speed):
        self.transition_start_speed = v_ego
        self.transition_target_speed = target_speed
        self.transition_start_time = time.time()

        speed_diff = self.transition_start_speed - self.transition_target_speed
        self.transition_duration = speed_diff / self.gentle_decel_rate

  def write_map_state(self, v_ego):
    self.map_speed_limit = self.get_param_memory("MapSpeedLimit")

    next_map_speed_limit = self.get_param_memory("NextMapSpeedLimit", is_json=True)
    next_map_speed_limit_value = next_map_speed_limit.get("speedlimit", 0)
    next_map_speed_limit_lat = next_map_speed_limit.get("latitude", 0)
    next_map_speed_limit_lon = next_map_speed_limit.get("longitude", 0)

    position = self.get_param_memory("LastGPSPosition", is_json=True)
    lat = position.get("latitude", 0)
    lon = position.get("longitude", 0)

    if next_map_speed_limit_value > 1:
      d = distance_to_point(lat * TO_RADIANS, lon * TO_RADIANS, next_map_speed_limit_lat * TO_RADIANS, next_map_speed_limit_lon * TO_RADIANS)

      if self.prv_speed_limit < next_map_speed_limit_value:
        max_d = self.frogpilot_toggles.map_speed_lookahead_higher * v_ego
      else:
        max_d = self.frogpilot_toggles.map_speed_lookahead_lower * v_ego

      if d < max_d:
        self.map_speed_limit = next_map_speed_limit_value

  @property
  def experimental_mode(self):
    return self.speed_limit == 0 and self.frogpilot_toggles.use_experimental_mode

  @property
  def desired_speed_limit(self):
    new_limit = self.speed_limit
    if new_limit > 1:
      target_speed = new_limit + self.offset

      if target_speed < self.transition_start_speed:
        current_time = time.time()
        if current_time > self.transition_start_time:
          # Calculate the ideal speed based on gentle deceleration
          time_elapsed = current_time - self.transition_start_time
          ideal_speed_change = self.gentle_decel_rate * time_elapsed
          ideal_current_speed = max(self.transition_start_speed - ideal_speed_change, target_speed)

          return ideal_current_speed

      else:
        # If the new limit is higher or equal, update immediately
        self.update_previous_limit(new_limit)
        self.transition_start_speed = target_speed
        self.transition_target_speed = target_speed
        self.transition_start_time = time.time()

      return target_speed

    return 0

  @property
  def offset(self):
    if self.speed_limit < 13.5:
      return self.frogpilot_toggles.offset1
    if self.speed_limit < 24:
      return self.frogpilot_toggles.offset2
    if self.speed_limit < 29:
      return self.frogpilot_toggles.offset3
    return self.frogpilot_toggles.offset4

  @property
  def speed_limit(self):
    limits = [self.car_speed_limit, self.map_speed_limit, self.nav_speed_limit]
    filtered_limits = [float(limit) for limit in limits if limit > 1]

    if self.frogpilot_toggles.speed_limit_priority_highest and filtered_limits:
      return max(filtered_limits)
    if self.frogpilot_toggles.speed_limit_priority_lowest and filtered_limits:
      return min(filtered_limits)

    speed_limits = {
      "Dashboard": self.car_speed_limit,
      "Offline Maps": self.map_speed_limit,
      "Navigation": self.nav_speed_limit,
    }

    for priority in [
      self.frogpilot_toggles.speed_limit_priority1,
      self.frogpilot_toggles.speed_limit_priority2,
      self.frogpilot_toggles.speed_limit_priority3,
    ]:
      if speed_limits.get(priority, 0) in filtered_limits:
        return speed_limits[priority]

    if self.frogpilot_toggles.use_previous_limit:
      return self.prv_speed_limit

    if self.frogpilot_toggles.use_set_speed:
      return self.max_speed_limit

    return 0
