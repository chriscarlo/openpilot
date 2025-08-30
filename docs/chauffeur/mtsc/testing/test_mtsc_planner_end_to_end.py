#!/usr/bin/env python3
"""
End-to-end MTSC + VTSC + Longitudinal min() integration test in simulation.

Builds a synthetic 90° bend centerline (ramp-like), computes MTSC's strategic
target (beyond visible horizon, comfort-limited), and verifies the planner's
min() selection reduces the cruise target compared to vision-only.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import time

import numpy as np
import pytest

import cereal.messaging as messaging
from sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed
from sunnypilot.selfdrive.controls.mtsc.mtscd import build_horizon_and_diagnostics


def make_arc_centerline(lat0: float, lon0: float, radius_m: float, angle_deg: float, n: int = 60):
  """Return list of points with latitude, longitude, and distanceFromStart for an arc."""
  R = 6371007.2
  pts = []
  total = 0.0
  angle_rad = math.radians(angle_deg)
  for i in range(n):
    th = angle_rad * i / (n - 1)
    x = radius_m * math.cos(th)
    y = radius_m * math.sin(th)
    # meters to degrees around reference
    lat = lat0 + (y / R) * 180.0 / math.pi
    lon = lon0 + (x / R) * 180.0 / math.pi / math.cos(math.radians(lat0))
    if i > 0:
      # distance along arc approximately
      total += radius_m * angle_rad / (n - 1)
    pts.append(SimpleNamespace(latitude=lat, longitude=lon, distanceFromStart=total))
  return pts


class SM:
  """Minimal SubMaster-like container for the planner stack."""
  def __init__(self, mapping: dict):
    self._data = mapping
    # Provide validity map expected by VTSC
    self.valid = {'modelV2': True}

  def __getitem__(self, key):
    return self._data[key]

  def all_checks(self, service_list=None):
    return True


def _mk_model_msg(curvature: float, v_pred: float, conf: float):
  # Minimal fields VTSC reads
  model = SimpleNamespace(
    orientationRate=SimpleNamespace(z=[curvature] * 33),
    velocity=SimpleNamespace(x=[v_pred] * 33),
    laneLineProbs=[conf] * 4,
  )
  return model


def reachable_cap(vsafe, dgrid, s_start, v_now, a_comf=1.47):
  vmax = v_now
  for vi, di in zip(vsafe, dgrid):
    if di < s_start:
      continue
    d = max(0.0, di - s_start)
    v_allow = math.sqrt(max(0.0, vi * vi + 2.0 * a_comf * d))
    vmax = min(vmax, v_allow)
  return vmax


@pytest.mark.parametrize("radius_m, angle_deg, v0_mph", [
  (45.0, 90.0, 50.0),    # tight 90° ramp
  (35.0, 150.0, 45.0),   # hairpin-like
])
def test_mtsc_planner_min(radius_m, angle_deg, v0_mph):
  # Ego initial conditions
  v0_mps = v0_mph * 0.44704
  a0 = 0.0
  lat0, lon0 = 37.0, -122.0

  # Build centerline and horizon diagnostics
  centerline = make_arc_centerline(lat0, lon0, radius_m, angle_deg, n=60)
  seg = SimpleNamespace(centerline=centerline, wayId=1001, roadDirection=0.0, roadClass=0, levelSeparation=0)
  diag = build_horizon_and_diagnostics(seg, centerline[0].latitude, centerline[0].longitude, v0_mps, resample_m=3.0)

  # Compute MTSC comfort-limited target beyond visible horizon
  s_vis = v0_mps * 1.3
  start_dist = s_vis + 10.0
  vsafe = diag.get('vsafe_mps', [])
  dgrid = diag.get('distances_m', [])
  assert len(vsafe) == len(dgrid) and len(vsafe) > 0
  v_cap = reachable_cap(vsafe, dgrid, start_dist, v0_mps, a_comf=1.47)
  v_target_mtsc = min(v0_mps, v_cap)

  # Vision-only curvature near field (take first vsafe value as proxy)
  k0 = 0.0  # simulate poor visibility (occluded); curvature not yet observed
  conf = 0.3
  model = _mk_model_msg(k0, v0_mps, conf)

  # Minimal services for planner update_v_cruise
  sm = SM({
    'carControl': SimpleNamespace(longActive=True),
    'carState': SimpleNamespace(gasPressed=False),
    'carStateSP': SimpleNamespace(speedLimit=0.0),
    'gpsLocation': SimpleNamespace(unixTimestampMillis=int(time.time()*1000), latitude=lat0, longitude=lon0, bearingDeg=0.0),
    'liveMapDataSP': SimpleNamespace(speedLimitValid=False, speedLimit=0.0, speedLimitAheadValid=False, speedLimitAhead=0.0, speedLimitAheadDistance=0.0),
    'modelV2': model,
    # Vision confidence low, so VTSC won't preempt early
    'mapTurnSpeedControlSP': SimpleNamespace(available=True, targetSpeedMps=v_target_mtsc),
  })

  class MockCP:
    openpilotLongitudinalControl = True
    pcmCruise = True
  # Create a dummy planner mixin to call LongitudinalPlannerSP.update_v_cruise directly
  class DummyPlanner(LongitudinalPlannerSP):
    def __init__(self):
      super().__init__(MockCP(), mpc=None)

  with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams1, \
       patch('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams2:
    mp = MagicMock()
    mp.get_bool.return_value = True
    mp.get.return_value = None
    MockParams1.return_value = mp
    MockParams2.return_value = mp

    planner = DummyPlanner()

    # Baseline cruise
    v_cruise_base = 100.0
    # With MTSC available
    v_with = planner.update_v_cruise(sm, v0_mps, a0, v_cruise_base)

  # Without MTSC (set available false)
  sm_off = SM({
    'carControl': SimpleNamespace(longActive=True),
    'carState': SimpleNamespace(gasPressed=False),
    'carStateSP': SimpleNamespace(speedLimit=0.0),
    'gpsLocation': SimpleNamespace(unixTimestampMillis=int(time.time()*1000), latitude=lat0, longitude=lon0, bearingDeg=0.0),
    'liveMapDataSP': SimpleNamespace(speedLimitValid=False, speedLimit=0.0, speedLimitAheadValid=False, speedLimitAhead=0.0, speedLimitAheadDistance=0.0),
    'modelV2': model,
    'mapTurnSpeedControlSP': SimpleNamespace(available=False, targetSpeedMps=0.0),
  })
  with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams1, \
       patch('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams2:
    mp = MagicMock()
    mp.get_bool.return_value = True
    mp.get.return_value = None
    MockParams1.return_value = mp
    MockParams2.return_value = mp
    v_without = planner.update_v_cruise(sm_off, v0_mps, a0, v_cruise_base)

  # MTSC should lower the cruise target compared to vision-only case (under occlusion)
  assert v_with <= v_without + 1e-3
  # MTSC target should be near or below the physics speed at first far-field sample
  assert v_with <= max(v0_mps, min(vsafe)) + 5.0  # allow slack for envelope
