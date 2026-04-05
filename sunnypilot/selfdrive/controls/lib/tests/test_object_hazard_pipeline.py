#!/usr/bin/env python3
from __future__ import annotations

import time
import importlib
import sys
from types import SimpleNamespace

import pytest

from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.sunnypilot.selfdrive.controls.lib.object_hazard_controller import compute_hazard_speed_recommendation
from openpilot.sunnypilot.selfdrive.controls.lib.tests.vtsc.pipeline_harness import (
  FakeSubMaster,
  import_longitudinal_planner_sp,
  install_fake_long_mpc,
  make_car_control,
  make_model_v2,
  make_radar_state,
)


class _NoOpController:
  def __init__(self, *args, **kwargs):
    pass

  def update(self, *args, **kwargs) -> None:
    pass


class _NoOpSLC(_NoOpController):
  state = 0
  is_enabled = False
  is_active = False
  speed_limit_offseted = V_CRUISE_UNSET
  speed_limit = 0.0
  speed_limit_offset = 0.0
  distance = 0.0
  source = 0


class _NoOpRTI(_NoOpController):
  is_active = False
  speed_recommendation = V_CRUISE_UNSET


class _NoOpWeather(_NoOpController):
  is_active = False
  speed_recommendation = V_CRUISE_UNSET


class _NoOpDEC(_NoOpController):
  def active(self) -> bool:
    return False

  def enabled(self) -> bool:
    return False

  def mode(self) -> str:
    return "acc"


class _NoOpVibe(_NoOpController):
  def is_accel_enabled(self) -> bool:
    return False

  def get_accel_limits(self, _v_ego):
    return None


class _NoOpVTSC(_NoOpController):
  is_active = False
  v_turn = V_CRUISE_UNSET
  state = 0
  current_lat_acc = 0.0
  max_pred_lat_acc = 0.0
  curve_preview_valid = False
  curve_preview_distance_m = 0.0
  curve_preview_time_to_s = 0.0
  curve_preview_kappa_max = 0.0
  curve_preview_direction = 0
  curve_preview_severity = 0
  curve_preview_points = []
  curve_preview_tiles = []
  curve_preview_branch_stubs = []

  def set_longitudinal_response_model(self, _response_model) -> None:
    pass


class _MockCP:
  pass


class _MockPlannerCP:
  openpilotLongitudinalControl = True
  longitudinalActuatorDelay = 0.15
  vEgoStopping = 0.25
  notCar = False


def _import_longitudinal_planner(monkeypatch):
  install_fake_long_mpc()

  sp_mod = importlib.import_module("openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner")
  monkeypatch.setattr(sp_mod, "SpeedLimitController", _NoOpSLC, raising=True)
  monkeypatch.setattr(sp_mod, "RTIController", _NoOpRTI, raising=True)
  monkeypatch.setattr(sp_mod, "WeatherController", _NoOpWeather, raising=True)
  monkeypatch.setattr(sp_mod, "DynamicExperimentalController", _NoOpDEC, raising=True)
  monkeypatch.setattr(sp_mod, "VibePersonalityController", _NoOpVibe, raising=True)
  monkeypatch.setattr(sp_mod, "VisionTurnController", _NoOpVTSC, raising=True)

  sys.modules.pop("openpilot.selfdrive.controls.lib.longitudinal_planner", None)
  return importlib.import_module("openpilot.selfdrive.controls.lib.longitudinal_planner")


@pytest.fixture()
def planner_sp(monkeypatch):
  mod = import_longitudinal_planner_sp()
  monkeypatch.setattr(mod, "SpeedLimitController", _NoOpSLC, raising=True)
  monkeypatch.setattr(mod, "RTIController", _NoOpRTI, raising=True)
  monkeypatch.setattr(mod, "WeatherController", _NoOpWeather, raising=True)
  monkeypatch.setattr(mod, "DynamicExperimentalController", _NoOpDEC, raising=True)
  monkeypatch.setattr(mod, "VibePersonalityController", _NoOpVibe, raising=True)
  monkeypatch.setattr(mod, "VisionTurnController", _NoOpVTSC, raising=True)

  return mod.LongitudinalPlannerSP(_MockCP(), object())


def test_object_hazard_speed_cap_wins_min_of_sources(planner_sp):
  sm = FakeSubMaster(
    data={
      "carControl": SimpleNamespace(longActive=True),
      "controlsState": SimpleNamespace(),
      "liveMapDataSP": SimpleNamespace(),
      "rtiStateSP": SimpleNamespace(),
      "objectHazardStateSP": SimpleNamespace(
        enabled=True,
        modelReady=True,
        active=True,
        hazardOnPath=True,
        recommendedSpeed=11.0,
        stopRequired=False,
        hazardDistanceM=12.0,
        hazardConfidence=0.91,
        hazardClass="person",
      ),
    },
    valid={"objectHazardStateSP": True},
    alive={"objectHazardStateSP": True},
    recv_time={"objectHazardStateSP": time.monotonic()},
  )

  v_cruise_final = planner_sp.update_v_cruise(sm, 20.0, 0.0, 25.0)

  expected = compute_hazard_speed_recommendation(12.0, 20.0, 25.0)
  assert planner_sp.object_hazard.is_active is True
  assert planner_sp.object_hazard.speed_recommendation == pytest.approx(expected, abs=1e-6)
  assert v_cruise_final == pytest.approx(expected, abs=1e-6)


def test_publish_longitudinal_plan_sp_includes_object_hazard_summary(planner_sp):
  class _FakePM:
    def __init__(self):
      self.sent = {}
    def send(self, name, msg) -> None:
      self.sent[name] = msg

  planner_sp.object_hazard.enabled = True
  planner_sp.object_hazard.is_active = True
  planner_sp.object_hazard.speed_recommendation = 9.5
  planner_sp.object_hazard.stop_required = True
  planner_sp.object_hazard.hazard_distance_m = 5.0
  planner_sp.object_hazard.hazard_confidence = 0.93
  planner_sp.object_hazard.hazard_class = "person"

  sm = FakeSubMaster(
    data={
      "controlsState": SimpleNamespace(),
    },
    valid={},
  )
  pm = _FakePM()

  planner_sp.publish_longitudinal_plan_sp(sm, pm)

  msg = pm.sent["longitudinalPlanSP"].longitudinalPlanSP.objectHazardControl
  assert bool(msg.active) is True
  assert bool(msg.stopRequired) is True
  assert float(msg.recommendedSpeed) == pytest.approx(9.5, abs=1e-6)
  assert float(msg.hazardDistanceM) == pytest.approx(5.0, abs=1e-6)
  assert str(msg.hazardClass) == "person"


def test_object_hazard_stop_request_reaches_main_longitudinal_planner(monkeypatch):
  mod = _import_longitudinal_planner(monkeypatch)
  planner = mod.LongitudinalPlanner(_MockPlannerCP(), init_v=8.0, init_a=0.0)

  sm = FakeSubMaster(
    data={
      "modelV2": make_model_v2(curvature=0.0, v_pred=8.0, confidence=0.9),
      "radarState": make_radar_state(lead_d_rel_m=None),
      "carState": SimpleNamespace(
        vEgo=8.0,
        aEgo=0.0,
        standstill=False,
        vCruise=30.0 * CV.MS_TO_KPH,
        brakePressed=False,
        gasPressed=False,
        cruiseState=SimpleNamespace(standstill=False),
      ),
      "carControl": make_car_control(long_active=True),
      "controlsState": SimpleNamespace(longControlState=LongCtrlState.pid, forceDecel=False),
      "selfdriveState": SimpleNamespace(experimentalMode=False, personality=0, enabled=True),
      "liveParameters": SimpleNamespace(angleOffsetDeg=0.0),
      "objectHazardStateSP": SimpleNamespace(
        enabled=True,
        modelReady=True,
        active=True,
        hazardOnPath=True,
        hazardDistanceM=4.0,
        hazardConfidence=0.95,
        hazardClass="person",
      ),
    },
    valid={"modelV2": True, "radarState": True, "objectHazardStateSP": True},
    alive={"objectHazardStateSP": True},
    recv_time={"objectHazardStateSP": time.monotonic()},
  )

  planner.update(sm)

  assert planner.object_hazard.is_active is True
  assert planner.output_should_stop is True
