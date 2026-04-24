from __future__ import annotations

import time
from types import SimpleNamespace

from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.object_hazard_controller import (
  ObjectHazardController,
  STALE_STATE_MAX_AGE_S,
  compute_hazard_speed_recommendation,
  should_stop_for_hazard,
)


class _FakeSubMaster:
  def __init__(self, state, valid: bool, *, alive: bool = True, recv_time: float | None = None):
    self._state = state
    self.valid = {"objectHazardStateSP": valid}
    self.alive = {"objectHazardStateSP": alive}
    self.recv_time = {"objectHazardStateSP": time.monotonic() if recv_time is None else recv_time}

  def __getitem__(self, key: str):
    assert key == "objectHazardStateSP"
    return self._state


def test_compute_hazard_speed_recommendation_clamps_to_cruise():
  recommended = compute_hazard_speed_recommendation(40.0, 8.0, 10.0)
  assert recommended == 10.0


def test_should_stop_for_hazard_triggers_for_close_object():
  assert should_stop_for_hazard(5.5, 3.0) is True
  assert should_stop_for_hazard(15.0, 4.0) is False


def test_compute_hazard_speed_recommendation_penalizes_overspeed():
  fast_recommended = compute_hazard_speed_recommendation(40.0, 20.0, 25.0)
  slow_recommended = compute_hazard_speed_recommendation(40.0, 8.0, 25.0)

  assert fast_recommended < slow_recommended


def test_object_hazard_controller_ignores_invalid_or_inactive_state():
  controller = ObjectHazardController()
  sm = _FakeSubMaster(None, False)
  controller.update(sm, 20.0, 0.0, 25.0)

  assert controller.enabled is False
  assert controller.is_active is False
  assert controller.speed_recommendation == V_CRUISE_UNSET


def test_object_hazard_controller_ignores_plain_dict_harness_sm():
  controller = ObjectHazardController()
  controller.update({}, 20.0, 0.0, 25.0)

  assert controller.enabled is False
  assert controller.is_active is False
  assert controller.speed_recommendation == V_CRUISE_UNSET


def test_object_hazard_controller_ignores_stale_or_dead_messages():
  controller = ObjectHazardController()
  state = SimpleNamespace(
    enabled=True,
    modelReady=True,
    active=True,
    hazardOnPath=True,
    hazardDistanceM=12.0,
    hazardConfidence=0.88,
    hazardClass="person",
  )

  stale_sm = _FakeSubMaster(state, True, recv_time=time.monotonic() - STALE_STATE_MAX_AGE_S - 0.1)
  controller.update(stale_sm, 12.0, 0.0, 20.0)
  assert controller.is_active is False

  dead_sm = _FakeSubMaster(state, True, alive=False)
  controller.update(dead_sm, 12.0, 0.0, 20.0)
  assert controller.is_active is False


def test_object_hazard_controller_generates_speed_cap_without_stop():
  controller = ObjectHazardController()
  state = SimpleNamespace(
    enabled=True,
    modelReady=True,
    active=True,
    hazardOnPath=True,
    hazardDistanceM=18.0,
    hazardConfidence=0.88,
    hazardClass="person",
  )
  sm = _FakeSubMaster(state, True)

  controller.update(sm, 20.0, 0.0, 25.0)

  assert controller.enabled is True
  assert controller.is_active is True
  assert controller.stop_required is False
  assert controller.speed_recommendation < 25.0
  assert controller.hazard_class == "person"


def test_object_hazard_controller_requests_stop_for_near_object():
  controller = ObjectHazardController()
  state = SimpleNamespace(
    enabled=True,
    modelReady=True,
    active=True,
    hazardOnPath=True,
    hazardDistanceM=4.0,
    hazardConfidence=0.92,
    hazardClass="person",
  )
  sm = _FakeSubMaster(state, True)

  controller.update(sm, 8.0, 0.0, 15.0)

  assert controller.is_active is True
  assert controller.stop_required is True
  assert controller.speed_recommendation >= 0.0
