#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np
import pytest

import openpilot.selfdrive.controls.lib.longitudinal_response_model as response_model
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.longitudinal_response_model import (
  CruiseResponseModel,
  build_cruise_response_model,
  clip_cruise_speed_profile,
  cruise_cap_for_required_average_decel,
  predict_average_decel_for_cruise_cap,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


def _legacy_planner_step_accel(
  *,
  v_ego: float,
  cruise_cap: float,
  response_model_obj: CruiseResponseModel,
  dt_s: float = response_model.DT_MDL,
  v_ego_stopping: float = 0.25,
  t_idxs = None,
) -> float:
  t = np.asarray(ModelConstants.T_IDXS if t_idxs is None else t_idxs, dtype=float)
  _, _, v_cruise_clipped = clip_cruise_speed_profile(
    v_ego=float(v_ego),
    v_cruise=float(cruise_cap),
    t_idxs=t,
    response_model=response_model_obj,
  )
  a_profile = np.gradient(v_cruise_clipped, t)
  a_target, _ = get_accel_from_plan(
    v_cruise_clipped,
    a_profile,
    t,
    action_t=max(float(dt_s), float(response_model_obj.actuation_delay_s)),
    vEgoStopping=v_ego_stopping,
  )
  amin = float(response_model_obj.planner_output_min_accel_mps2)
  amax = float(response_model_obj.planner_output_max_accel_mps2)
  return float(np.clip(a_target, amin, amax))


def _legacy_predict_average_decel_for_cruise_cap(
  *,
  v_ego: float,
  cruise_cap: float,
  response_model_obj: CruiseResponseModel,
  probe_duration_s: float = response_model.CURVE_DECEL_PROBE_DURATION_S,
  dt_s: float = response_model.DT_MDL,
  v_ego_stopping: float = 0.25,
  t_idxs = None,
) -> float:
  v = max(0.0, float(v_ego))
  cruise = max(0.0, float(cruise_cap))
  dt = max(1e-3, float(dt_s))
  duration = max(dt, float(probe_duration_s))
  steps = max(1, int(math.ceil(duration / dt)))
  a_sum = 0.0

  for _ in range(steps):
    a_cmd = _legacy_planner_step_accel(
      v_ego=v,
      cruise_cap=cruise,
      response_model_obj=response_model_obj,
      dt_s=dt,
      v_ego_stopping=v_ego_stopping,
      t_idxs=t_idxs,
    )
    a_sum += a_cmd
    v = max(0.0, v + a_cmd * dt)

  return max(0.0, -(a_sum / float(steps)))


def _legacy_cruise_cap_for_required_average_decel(
  *,
  v_ego: float,
  required_decel_mps2: float,
  response_model_obj: CruiseResponseModel,
  v_cruise_upper: float,
  probe_duration_s: float = response_model.CURVE_DECEL_PROBE_DURATION_S,
  dt_s: float = response_model.DT_MDL,
  v_ego_stopping: float = 0.25,
  t_idxs = None,
  iterations: int = 10,
  decel_tol_mps2: float = response_model.CRUISE_CAP_REQUIRED_DECEL_TOL_MPS2,
) -> float:
  v_ego_f = max(0.0, float(v_ego))
  required = max(0.0, float(required_decel_mps2))
  upper = max(0.0, float(v_cruise_upper))
  if required <= decel_tol_mps2:
    return upper

  cache: dict[float, float] = {}

  def avg_decel(cap: float) -> float:
    key = round(float(cap), 4)
    if key not in cache:
      cache[key] = _legacy_predict_average_decel_for_cruise_cap(
        v_ego=v_ego_f,
        cruise_cap=key,
        response_model_obj=response_model_obj,
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


def _response_models() -> list[CruiseResponseModel]:
  return [
    build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, comfort_brake_mps2=2.5, actuation_delay_s=0.0, planner_output_min_accel_mps2=-3.5),
    build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, comfort_brake_mps2=2.5, actuation_delay_s=0.2, planner_output_min_accel_mps2=-2.0),
    build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, comfort_brake_mps2=2.5, actuation_delay_s=0.4, planner_output_min_accel_mps2=-3.5),
    build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, comfort_brake_mps2=2.5, actuation_delay_s=0.6, planner_output_min_accel_mps2=-1.5),
  ]


def test_planner_step_accel_matches_legacy_profile_path():
  for response_model_obj in _response_models():
    for v_ego in (1.0, 5.0, 10.0, 15.0, 24.0, 30.0):
      for cruise_cap in (0.0, 0.5, 2.0, 4.0, 8.0, 12.0, 18.0, 22.0, 28.0, 35.0):
        exact = response_model._planner_step_accel(
          v_ego=v_ego,
          cruise_cap=cruise_cap,
          response_model=response_model_obj,
        )
        legacy = _legacy_planner_step_accel(
          v_ego=v_ego,
          cruise_cap=cruise_cap,
          response_model_obj=response_model_obj,
        )
        assert exact == pytest.approx(legacy, abs=1e-9)


def test_predict_average_decel_matches_legacy_profile_path():
  for response_model_obj in _response_models():
    for v_ego in (5.0, 10.0, 15.0, 24.0, 30.0):
      for cruise_cap in (0.0, 2.0, 5.0, 8.0, 12.0, 18.0, 22.0, 25.0, 30.0):
        exact = predict_average_decel_for_cruise_cap(
          v_ego=v_ego,
          cruise_cap=cruise_cap,
          response_model=response_model_obj,
        )
        legacy = _legacy_predict_average_decel_for_cruise_cap(
          v_ego=v_ego,
          cruise_cap=cruise_cap,
          response_model_obj=response_model_obj,
        )
        assert exact == pytest.approx(legacy, abs=1e-9)


def test_cruise_cap_inverse_matches_legacy_profile_path():
  for response_model_obj in _response_models():
    for v_ego in (10.0, 15.0, 24.0, 30.0):
      for required_decel in (0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        exact = cruise_cap_for_required_average_decel(
          v_ego=v_ego,
          required_decel_mps2=required_decel,
          response_model=response_model_obj,
          v_cruise_upper=33.0,
        )
        legacy = _legacy_cruise_cap_for_required_average_decel(
          v_ego=v_ego,
          required_decel_mps2=required_decel,
          response_model_obj=response_model_obj,
          v_cruise_upper=33.0,
        )
        assert exact == pytest.approx(legacy, abs=1e-9)


def test_predict_average_decel_avoids_legacy_profile_build(monkeypatch):
  def fail(*args, **kwargs):
    raise AssertionError("slow profile build should not run in the helper probe")

  monkeypatch.setattr(response_model, "clip_cruise_speed_profile", fail)

  rm = build_cruise_response_model(
    min_accel_mps2=-6.0,
    max_accel_mps2=5.0,
    comfort_brake_mps2=2.5,
    actuation_delay_s=0.4,
    planner_output_min_accel_mps2=-3.5,
  )
  decel = predict_average_decel_for_cruise_cap(
    v_ego=24.0,
    cruise_cap=7.5,
    response_model=rm,
  )
  assert decel > 0.0
