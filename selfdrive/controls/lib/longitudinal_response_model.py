#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.planner_lag_debug import (
  SPAN_HELPER_CRUISE_CAP,
  SPAN_HELPER_PREDICT,
  end_span,
  start_span,
)


CRUISE_ENVELOPE_SAFETY_FACTOR = 1.05
DEFAULT_CRUISE_MIN_ACCEL = -6.0
DEFAULT_CRUISE_MAX_ACCEL = 5.0
DEFAULT_COMFORT_BRAKE = 2.5
CURVE_DECEL_PROBE_DURATION_S = 2.0
CRUISE_CAP_REQUIRED_DECEL_TOL_MPS2 = 0.02


@dataclass(frozen=True)
class CruiseResponseModel:
  min_accel_mps2: float = DEFAULT_CRUISE_MIN_ACCEL
  max_accel_mps2: float = DEFAULT_CRUISE_MAX_ACCEL
  comfort_brake_mps2: float = DEFAULT_COMFORT_BRAKE
  actuation_delay_s: float = 0.0
  speed_safety_factor: float = CRUISE_ENVELOPE_SAFETY_FACTOR
  planner_output_min_accel_mps2: float = ACCEL_MIN
  planner_output_max_accel_mps2: float = ACCEL_MAX

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
  planner_output_min_accel_mps2: float = ACCEL_MIN,
  planner_output_max_accel_mps2: float = ACCEL_MAX,
) -> CruiseResponseModel:
  return CruiseResponseModel(
    min_accel_mps2=float(min_accel_mps2),
    max_accel_mps2=float(max_accel_mps2),
    comfort_brake_mps2=max(1e-3, float(comfort_brake_mps2)),
    actuation_delay_s=max(0.0, float(actuation_delay_s)),
    speed_safety_factor=max(1.0, float(speed_safety_factor)),
    planner_output_min_accel_mps2=float(planner_output_min_accel_mps2),
    planner_output_max_accel_mps2=float(planner_output_max_accel_mps2),
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


def _planner_step_accel(
  *,
  v_ego: float,
  cruise_cap: float,
  response_model: CruiseResponseModel,
  dt_s: float = DT_MDL,
  v_ego_stopping: float = 0.25,
  t_idxs = None,
) -> float:
  t = np.asarray(ModelConstants.T_IDXS if t_idxs is None else t_idxs, dtype=float)
  _, _, v_cruise_clipped = clip_cruise_speed_profile(
    v_ego=float(v_ego),
    v_cruise=float(cruise_cap),
    t_idxs=t,
    response_model=response_model,
  )
  a_profile = np.gradient(v_cruise_clipped, t)
  a_target, _ = get_accel_from_plan(
    v_cruise_clipped,
    a_profile,
    t,
    action_t=max(float(dt_s), float(response_model.actuation_delay_s)),
    vEgoStopping=v_ego_stopping,
  )
  amin = float(response_model.planner_output_min_accel_mps2)
  amax = float(response_model.planner_output_max_accel_mps2)
  return float(np.clip(a_target, amin, amax))


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


def predict_speed_with_cruise_cap(
  *,
  v_ego: float,
  cruise_cap: float,
  distance_m: float,
  response_model: CruiseResponseModel,
  dt_s: float = DT_MDL,
  v_ego_stopping: float = 0.25,
  t_idxs = None,
  max_steps: int = 600,
) -> float:
  v = max(0.0, float(v_ego))
  cruise = max(0.0, float(cruise_cap))
  remaining = max(0.0, float(distance_m))
  if remaining <= 0.0:
    return v

  dt = max(1e-3, float(dt_s))

  for _ in range(int(max_steps)):
    if remaining <= 0.0:
      break
    a_cmd = _planner_step_accel(
      v_ego=v,
      cruise_cap=cruise,
      response_model=response_model,
      dt_s=dt,
      v_ego_stopping=v_ego_stopping,
      t_idxs=t_idxs,
    )
    v = max(0.0, v + a_cmd * dt)
    remaining -= v * dt
    if v <= cruise + 1e-3 and a_cmd >= -1e-3:
      break
  return v


def predict_average_decel_for_cruise_cap(
  *,
  v_ego: float,
  cruise_cap: float,
  response_model: CruiseResponseModel,
  probe_duration_s: float = CURVE_DECEL_PROBE_DURATION_S,
  dt_s: float = DT_MDL,
  v_ego_stopping: float = 0.25,
  t_idxs = None,
) -> float:
  total_span = start_span(SPAN_HELPER_PREDICT)
  try:
    v = max(0.0, float(v_ego))
    cruise = max(0.0, float(cruise_cap))
    dt = max(1e-3, float(dt_s))
    duration = max(dt, float(probe_duration_s))
    steps = max(1, int(math.ceil(duration / dt)))
    a_sum = 0.0

    for _ in range(steps):
      a_cmd = _planner_step_accel(
        v_ego=v,
        cruise_cap=cruise,
        response_model=response_model,
        dt_s=dt,
        v_ego_stopping=v_ego_stopping,
        t_idxs=t_idxs,
      )
      a_sum += a_cmd
      v = max(0.0, v + a_cmd * dt)

    return max(0.0, -(a_sum / float(steps)))
  finally:
    end_span(total_span)


def cruise_cap_for_required_average_decel(
  *,
  v_ego: float,
  required_decel_mps2: float,
  response_model: CruiseResponseModel,
  v_cruise_upper: float,
  probe_duration_s: float = CURVE_DECEL_PROBE_DURATION_S,
  dt_s: float = DT_MDL,
  v_ego_stopping: float = 0.25,
  t_idxs = None,
  iterations: int = 10,
  decel_tol_mps2: float = CRUISE_CAP_REQUIRED_DECEL_TOL_MPS2,
) -> float:
  total_span = start_span(SPAN_HELPER_CRUISE_CAP)
  try:
    v_ego_f = max(0.0, float(v_ego))
    required = max(0.0, float(required_decel_mps2))
    upper = max(0.0, float(v_cruise_upper))
    if required <= decel_tol_mps2:
      return upper

    cache: dict[float, float] = {}

    def avg_decel(cap: float) -> float:
      key = round(float(cap), 4)
      if key not in cache:
        cache[key] = predict_average_decel_for_cruise_cap(
          v_ego=v_ego_f,
          cruise_cap=key,
          response_model=response_model,
          probe_duration_s=probe_duration_s,
          dt_s=dt_s,
          v_ego_stopping=v_ego_stopping,
          t_idxs=t_idxs,
        )
      return cache[key]

    lo = 0.0
    hi = upper
    if avg_decel(hi) >= required - decel_tol_mps2:
      return hi
    if avg_decel(lo) < required - decel_tol_mps2:
      return lo

    for _ in range(int(iterations)):
      mid = 0.5 * (lo + hi)
      if avg_decel(mid) >= required - decel_tol_mps2:
        lo = mid
      else:
        hi = mid

    return max(0.0, float(lo))
  finally:
    end_span(total_span)


def required_cruise_cap_for_target_at_distance(
  *,
  v_ego: float,
  target_speed: float,
  distance_m: float,
  response_model: CruiseResponseModel,
  v_cruise_upper: float,
  dt_s: float = DT_MDL,
  v_ego_stopping: float = 0.25,
  t_idxs = None,
  max_steps: int = 600,
  iterations: int = 8,
  target_tol_mps: float = 0.25,
) -> float:
  upper = max(0.0, float(v_cruise_upper))
  target = max(0.0, float(target_speed))
  lower = min(target, upper)
  distance = max(0.0, float(distance_m))

  if predict_speed_with_cruise_cap(
    v_ego=v_ego,
    cruise_cap=upper,
    distance_m=distance,
    response_model=response_model,
    dt_s=dt_s,
    v_ego_stopping=v_ego_stopping,
    t_idxs=t_idxs,
    max_steps=max_steps,
  ) <= target + float(target_tol_mps):
    return upper

  if predict_speed_with_cruise_cap(
    v_ego=v_ego,
    cruise_cap=lower,
    distance_m=distance,
    response_model=response_model,
    dt_s=dt_s,
    v_ego_stopping=v_ego_stopping,
    t_idxs=t_idxs,
    max_steps=max_steps,
  ) > target + float(target_tol_mps):
    return lower

  for _ in range(int(iterations)):
    mid = 0.5 * (lower + upper)
    predicted = predict_speed_with_cruise_cap(
      v_ego=v_ego,
      cruise_cap=mid,
      distance_m=distance,
      response_model=response_model,
      dt_s=dt_s,
      v_ego_stopping=v_ego_stopping,
      t_idxs=t_idxs,
      max_steps=max_steps,
    )
    if predicted <= target + float(target_tol_mps):
      lower = mid
    else:
      upper = mid
  return max(0.0, float(lower))
