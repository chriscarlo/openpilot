from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_T_FOLLOW
import numpy as np
import cereal.messaging as messaging
from cereal import log

LaneChangeState = log.LaneChangeState

class DynamicFollow:
  def __init__(self):
    self.sm = messaging.SubMaster(['modelV2'])
    self.t_follow = None  # Initialize previous t_follow

  def sigmoid(self, x, L, k, x0):
    epsilon = 1e-6
    return L / (1 + np.exp(-k * (x - x0)) + epsilon)

  def calculate_dynamic_follow(self, base_follow, v_ego, personality):
    v_ego_mph = v_ego * 2.23694

    self.sm.update()
    lane_change_state = self.sm['modelV2'].meta.laneChangeState

    if personality == "aggressive":
      min_follow = 0.7
      max_follow = 1.5
      L = max_follow - min_follow
      k = 0.05
      x0 = 35
    elif personality == "standard":
      min_follow = 0.8
      max_follow = 1.7
      L = max_follow - min_follow
      k = 0.05
      x0 = 25
    elif personality == "relaxed":
      min_follow = 0.9
      max_follow = 2.0
      L = max_follow - min_follow
      k = 0.05
      x0 = 18
    else:
      # Fallback to original calculation
      return np.clip(base_follow * (v_ego / (60 * 0.44704)), 0.8, 2.5)

    target_follow = min_follow + self.sigmoid(v_ego_mph, L, k, x0)

    # Reduce follow distance by 20% if changing lanes
    if lane_change_state in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing):
      target_follow *= 0.8

    return np.clip(target_follow, min_follow, max_follow)

  def get_dynamic_follow(self, aggressive_follow, standard_follow, relaxed_follow, custom_personalities, personality, v_ego):
    base_follow = get_T_FOLLOW(aggressive_follow, standard_follow, relaxed_follow, custom_personalities, personality)
    target_follow = self.calculate_dynamic_follow(base_follow, v_ego, personality)

    # Apply exponential smoothing to t_follow
    alpha = 0.5  # Smoothing factor (adjust between 0 and 1)
    if self.t_follow is None:
      self.t_follow = target_follow
    else:
      self.t_follow = alpha * target_follow + (1 - alpha) * self.t_follow

    return self.t_follow

dynamic_follow = DynamicFollow()
get_dynamic_follow = dynamic_follow.get_dynamic_follow
