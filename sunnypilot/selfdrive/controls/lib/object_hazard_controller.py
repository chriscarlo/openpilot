#!/usr/bin/env python3
from __future__ import annotations

import math
import time

from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET

COMFORT_DECEL_MPS2 = 1.8
STOP_BUFFER_M = 3.0
STOP_DISTANCE_M = 6.0
STALE_STATE_MAX_AGE_S = 1.25


def compute_hazard_speed_recommendation(distance_m: float, v_ego_mps: float, cruise_cap_mps: float | None) -> float:
  distance = max(float(distance_m), 0.0)
  available_distance = max(distance - STOP_BUFFER_M, 0.0)
  recommended = math.sqrt(2.0 * COMFORT_DECEL_MPS2 * available_distance)

  ego_speed = max(float(v_ego_mps), 0.0)
  if ego_speed > recommended and recommended > 1e-3:
    # If we're already above the comfort-stop envelope, push the cap down harder than the
    # raw stopping-speed bound so MPC sees a meaningful decel request immediately.
    recommended = recommended * (recommended / ego_speed)

  if cruise_cap_mps is not None and cruise_cap_mps != V_CRUISE_UNSET:
    recommended = min(recommended, float(cruise_cap_mps))
  return recommended


def should_stop_for_hazard(distance_m: float, recommended_speed_mps: float) -> bool:
  return bool(float(distance_m) <= STOP_DISTANCE_M or float(recommended_speed_mps) <= 0.5)


def get_fresh_object_hazard_state(sm):
  if not all(hasattr(sm, attr) for attr in ("valid", "alive", "recv_time")):
    return None
  if not sm.valid.get("objectHazardStateSP", False):
    return None
  if not sm.alive.get("objectHazardStateSP", False):
    return None
  recv_time = float(sm.recv_time.get("objectHazardStateSP", 0.0))
  if recv_time <= 0.0 or (time.monotonic() - recv_time) > STALE_STATE_MAX_AGE_S:
    return None
  return sm["objectHazardStateSP"]


class ObjectHazardController:
  def __init__(self):
    self.reset()

  def reset(self) -> None:
    self.enabled = False
    self.is_active = False
    self.speed_recommendation = V_CRUISE_UNSET
    self.stop_required = False
    self.hazard_distance_m = 0.0
    self.hazard_confidence = 0.0
    self.hazard_class = ""

  def update(self, sm, v_ego: float, _a_ego: float, v_cruise: float) -> None:
    self.reset()
    state = get_fresh_object_hazard_state(sm)
    if state is None:
      return

    self.enabled = bool(state.enabled)
    if not (self.enabled and state.modelReady and state.active and state.hazardOnPath):
      return

    self.is_active = True
    self.hazard_distance_m = float(state.hazardDistanceM)
    self.hazard_confidence = float(state.hazardConfidence)
    self.hazard_class = str(state.hazardClass)
    self.speed_recommendation = compute_hazard_speed_recommendation(self.hazard_distance_m, v_ego, v_cruise)
    self.stop_required = should_stop_for_hazard(self.hazard_distance_m, self.speed_recommendation)
