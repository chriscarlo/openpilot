#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


CRUISE_ENVELOPE_SAFETY_FACTOR = 1.05
DEFAULT_CRUISE_MIN_ACCEL = -6.0
DEFAULT_CRUISE_MAX_ACCEL = 5.0
DEFAULT_COMFORT_BRAKE = 2.5


@dataclass(frozen=True)
class CruiseResponseModel:
  min_accel_mps2: float = DEFAULT_CRUISE_MIN_ACCEL
  max_accel_mps2: float = DEFAULT_CRUISE_MAX_ACCEL
  comfort_brake_mps2: float = DEFAULT_COMFORT_BRAKE
  actuation_delay_s: float = 0.0
  speed_safety_factor: float = CRUISE_ENVELOPE_SAFETY_FACTOR

  @property
  def effective_min_decel_mps2(self) -> float:
    return max(1e-3, abs(min(0.0, float(self.min_accel_mps2))) * float(self.speed_safety_factor))

  @property
  def planning_decel_mps2(self) -> float:
    return max(1e-3, min(self.effective_min_decel_mps2, max(1e-3, float(self.comfort_brake_mps2))))


def build_cruise_response_model(
  *,
  min_accel_mps2: float = DEFAULT_CRUISE_MIN_ACCEL,
  max_accel_mps2: float = DEFAULT_CRUISE_MAX_ACCEL,
  comfort_brake_mps2: float = DEFAULT_COMFORT_BRAKE,
  actuation_delay_s: float = 0.0,
  speed_safety_factor: float = CRUISE_ENVELOPE_SAFETY_FACTOR,
) -> CruiseResponseModel:
  return CruiseResponseModel(
    min_accel_mps2=float(min_accel_mps2),
    max_accel_mps2=float(max_accel_mps2),
    comfort_brake_mps2=max(1e-3, float(comfort_brake_mps2)),
    actuation_delay_s=max(0.0, float(actuation_delay_s)),
    speed_safety_factor=max(1.0, float(speed_safety_factor)),
  )


def compute_cruise_speed_bounds(
  *,
  v_ego: float,
  t_idxs,
  response_model: CruiseResponseModel,
):
  t = np.asarray(t_idxs, dtype=float)
  v_ego_f = float(v_ego)
  min_accel = min(0.0, float(response_model.min_accel_mps2)) * float(response_model.speed_safety_factor)
  max_accel = max(0.0, float(response_model.max_accel_mps2)) * float(response_model.speed_safety_factor)
  v_lower = v_ego_f + (t * min_accel)
  v_upper = v_ego_f + (t * max_accel)
  return v_lower, v_upper


def clip_cruise_speed_profile(
  *,
  v_ego: float,
  v_cruise: float,
  t_idxs,
  response_model: CruiseResponseModel,
):
  v_lower, v_upper = compute_cruise_speed_bounds(
    v_ego=v_ego,
    t_idxs=t_idxs,
    response_model=response_model,
  )
  v_cruise_clipped = np.clip(np.full_like(v_lower, float(v_cruise)), v_lower, v_upper)
  return v_lower, v_upper, v_cruise_clipped


def distance_needed_to_reach_speed(
  *,
  v_ego: float,
  target_speed: float,
  response_model: CruiseResponseModel,
) -> float:
  v_ego_f = max(0.0, float(v_ego))
  target_f = max(0.0, float(target_speed))
  delay_distance = v_ego_f * max(0.0, float(response_model.actuation_delay_s))
  if target_f >= v_ego_f:
    return delay_distance
  a_eff = float(response_model.planning_decel_mps2)
  return delay_distance + max(0.0, (v_ego_f * v_ego_f - target_f * target_f) / (2.0 * a_eff))


def reachable_speed_for_target_at_distance(
  *,
  v_ego: float,
  target_speed: float,
  distance_m: float,
  response_model: CruiseResponseModel,
) -> float:
  v_ego_f = max(0.0, float(v_ego))
  target_f = max(0.0, float(target_speed))
  distance_f = max(0.0, float(distance_m))
  delay_distance = v_ego_f * max(0.0, float(response_model.actuation_delay_s))
  braking_distance = max(0.0, distance_f - delay_distance)
  a_eff = float(response_model.planning_decel_mps2)
  return math.sqrt(max(0.0, target_f * target_f + 2.0 * a_eff * braking_distance))
