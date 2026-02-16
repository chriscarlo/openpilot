#!/usr/bin/env python3
"""
Unit tests for VTSC intervention recorder helper logic.
"""

from dataclasses import dataclass

from opendbc.car.common.conversions import Conversions as CV

from tools.vtsc.vtsc_intervention_recorder import _read_cruise_set_speed_mps


@dataclass
class DummyCarState:
  vCruise: float = 0.0
  vCruiseCluster: float = 0.0


@dataclass
class DummyControlsState:
  vCruiseDEPRECATED: float = 0.0
  vCruiseClusterDEPRECATED: float = 0.0


def test_prefers_car_state_vcruise():
  car_state = DummyCarState(vCruise=80.0, vCruiseCluster=70.0)
  ctrls = DummyControlsState(vCruiseDEPRECATED=90.0, vCruiseClusterDEPRECATED=85.0)

  speed_mps, src = _read_cruise_set_speed_mps(car_state, ctrls)

  assert abs(speed_mps - (80.0 * CV.KPH_TO_MS)) < 1e-6
  assert src == "carState.vCruise"


def test_falls_back_to_controls_when_car_state_missing():
  ctrls = DummyControlsState(vCruiseDEPRECATED=72.0)

  speed_mps, src = _read_cruise_set_speed_mps(None, ctrls)

  assert abs(speed_mps - (72.0 * CV.KPH_TO_MS)) < 1e-6
  assert src == "controlsState.vCruiseDEPRECATED"


def test_returns_none_for_all_non_positive_sources():
  car_state = DummyCarState(vCruise=0.0, vCruiseCluster=0.0)
  ctrls = DummyControlsState(vCruiseDEPRECATED=0.0, vCruiseClusterDEPRECATED=0.0)

  speed_mps, src = _read_cruise_set_speed_mps(car_state, ctrls)

  assert speed_mps == 0.0
  assert src == "none"
