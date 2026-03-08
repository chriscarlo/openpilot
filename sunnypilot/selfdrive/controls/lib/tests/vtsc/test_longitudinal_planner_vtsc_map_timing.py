#!/usr/bin/env python3
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState

from .pipeline_harness import (
  FakeSubMaster,
  make_car_control,
  make_model_v2,
)
from .test_longitudinal_planner_vtsc_flow import (
  _NoOpDEC,
  _NoOpRTI,
  _NoOpSLC,
  _NoOpVibe,
)
from openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed


class _MockCP:
  openpilotLongitudinalControl = True
  longitudinalActuatorDelay = 0.15
  vEgoStopping = 0.25
  notCar = False
  pcmCruise = False
  steerRatio = 15.0
  wheelbase = 2.75


def _make_radar_state():
  lead = SimpleNamespace(status=False, dRel=1e9, vLead=0.0, aLeadK=0.0, aLeadTau=1.5)
  return SimpleNamespace(leadOne=lead, leadTwo=lead)


def _mk_sm(*, v_ego: float, a_ego: float, v_cruise_mps: float, long_active: bool = True,
           curvature: float = 0.0, confidence: float = 0.95):
  car_state = SimpleNamespace(
    vEgo=float(v_ego),
    aEgo=float(a_ego),
    standstill=bool(v_ego < 0.01),
    vCruise=float(v_cruise_mps * CV.MS_TO_KPH),
    gasPressed=False,
    brakePressed=False,
    cruiseState=SimpleNamespace(standstill=False),
  )
  controls_state = SimpleNamespace(
    longControlState=LongCtrlState.pid,
    forceDecel=False,
  )
  selfdrive_state = SimpleNamespace(
    experimentalMode=False,
    personality=0,
    enabled=True,
  )
  model = make_model_v2(curvature=curvature, curvature_ahead=curvature, v_pred=v_ego, confidence=confidence)
  return FakeSubMaster(
    data={
      'modelV2': model,
      'radarState': _make_radar_state(),
      'carState': car_state,
      'carControl': make_car_control(long_active=long_active),
      'controlsState': controls_state,
      'selfdriveState': selfdrive_state,
      'liveParameters': SimpleNamespace(angleOffsetDeg=0.0),
    },
    valid={'modelV2': True, 'radarState': True},
  )


def _build_map_profile_polyline(
  lat0: float,
  lon0: float,
  curvature_profile: list[float],
  *,
  profile_start_m: float,
  total_m: float = 1500.0,
  step_m: float = 10.0,
):
  pts = []
  n = int(total_m // step_m)
  profile_start_idx = int(round(float(profile_start_m) / float(step_m)))
  for i in range(n + 1):
    profile_idx = i - profile_start_idx
    if 0 <= profile_idx < len(curvature_profile):
      k = float(max(0.0, curvature_profile[profile_idx]))
    else:
      k = 0.0
    pts.append((lat0 + (i * step_m) / 111000.0, lon0, k))
  return pts


def _run_hidden_apex_profile(monkeypatch, *, fixed_lead_time_s: float):
  sp_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner')
  monkeypatch.setattr(sp_mod, 'SpeedLimitController', _NoOpSLC, raising=True)
  monkeypatch.setattr(sp_mod, 'RTIController', _NoOpRTI, raising=True)
  monkeypatch.setattr(sp_mod, 'DynamicExperimentalController', _NoOpDEC, raising=True)
  monkeypatch.setattr(sp_mod, 'VibePersonalityController', _NoOpVibe, raising=True)
  main_mod = importlib.import_module('openpilot.selfdrive.controls.lib.longitudinal_planner')

  p = Params()
  p.put_bool('VisionTurnSpeedControl', True)
  p.put_bool('VTSCVerboseDebug', False)
  p.put_bool('VTSCWriteSnapshotFile', False)

  planner = main_mod.LongitudinalPlanner(_MockCP(), init_v=22.0, init_a=0.0)

  orig_get_bool = planner.v_tsc._get_bool_param
  orig_get_string = planner.v_tsc._get_string_param
  planner.v_tsc._map_strategy_mode = 'strategic'
  planner.v_tsc._fixed_lead_time_s = float(fixed_lead_time_s)
  planner.v_tsc._curve_phase_offset_s = 0.0
  planner.v_tsc._overshoot_phase_offset_s = 0.0
  planner.v_tsc._apex_exit_phase_offset_s = 0.0
  planner.v_tsc._get_bool_param = lambda key, default=False: True if key == 'MTSCLookaheadEnabled' else bool(orig_get_bool(key, default))
  planner.v_tsc._get_string_param = lambda key, default='': 'strategic' if key == 'VTSCMapStrategy' else str(orig_get_string(key, default))

  lat0, lon0 = 37.0, -122.0
  apex_profile = [
    0.0,
    0.0015,
    0.0030,
    0.0200,
    0.0080,
    0.0030,
    0.0,
  ]
  profile_start_m = 250.0
  anchor_s = profile_start_m + 30.0
  pts = _build_map_profile_polyline(lat0, lon0, apex_profile, profile_start_m=profile_start_m)

  state = {'d': 0.0}
  planner.v_tsc._get_last_gps_pose = lambda: (lat0 + state['d'] / 111000.0, lon0, None)
  planner.v_tsc._load_map_curvatures = lambda: pts

  vtc_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller')
  v_ego = 22.0
  a_ego = 0.0
  v_cruise = 27.0
  dt = 0.05
  t = 0.0
  hist: list[dict[str, float]] = []

  from unittest.mock import patch
  for _ in range(500):
    sm = _mk_sm(v_ego=v_ego, a_ego=a_ego, v_cruise_mps=v_cruise)
    with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
      planner.update(sm)
    snap = planner.v_tsc.snapshot_debug_state() or {}
    hist.append({
      't': float(t),
      'distance_traveled_m': float(state['d']),
      'dist_to_apex_m': float(anchor_s - state['d']),
      'v_ego': float(v_ego),
      'a_target': float(planner.output_a_target),
      'v_turn': float(planner.v_tsc.v_turn),
      'map_strategic_cap': float(snap.get('map_strategic_cap', 0.0)),
    })
    a_ego = float(planner.output_a_target)
    v_ego = max(0.0, v_ego + a_ego * dt)
    state['d'] += v_ego * dt
    t += dt

  return hist


def test_strategic_hidden_apex_hits_target_across_fixed_lead_array_real_mpc(monkeypatch):
  target_speed = float(curvature_to_speed(0.02))
  target_tol = 0.75
  results = {lead_s: _run_hidden_apex_profile(monkeypatch, fixed_lead_time_s=lead_s) for lead_s in (0.0, 1.0, 2.0, 3.0)}

  hit_distances = []
  speeds_100m = []
  for lead_s, hist in results.items():
    at_100m = min(hist, key=lambda row: abs(row['dist_to_apex_m'] - 100.0))
    speeds_100m.append(at_100m['v_ego'])

    first_hit = next(row for row in hist if row['v_ego'] <= target_speed + target_tol)
    assert first_hit['dist_to_apex_m'] >= 0.0, f"lead {lead_s} did not reach target until after the apex"
    hit_distances.append(first_hit['dist_to_apex_m'])

  # Zero lead should still behave like a normal approach, not crawl toward the curve far too early.
  assert speeds_100m[0] >= target_speed + 2.5
  # Larger fixed lead times should only move target attainment earlier, not be required to hit it.
  assert hit_distances[0] <= hit_distances[1] <= hit_distances[2] <= hit_distances[3]
