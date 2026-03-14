#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

import openpilot.sunnypilot.selfdrive.controls.lib.vtsc_map_strategy as map_strategy
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


def _curve_phase_raw_for_effective(effective_s: float) -> float:
  return float(effective_s) - float(map_strategy.CURVE_PHASE_OFFSET_ZERO_BASELINE_S)


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
  sys.modules.pop('openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner', None)
  sys.modules.pop('openpilot.selfdrive.controls.lib.longitudinal_planner', None)
  sys.modules.pop('openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc', None)
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
  planner.v_tsc._update_params = lambda: None
  planner.v_tsc._is_enabled = True
  planner.v_tsc._map_strategy_mode = 'strategic'
  planner.v_tsc._fixed_lead_time_s = float(fixed_lead_time_s)
  planner.v_tsc._curve_phase_offset_s = float(_curve_phase_raw_for_effective(0.0))
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


def _run_chained_winding_profile(monkeypatch, *, winding_profile_level: int):
  sys.modules.pop('openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner', None)
  sys.modules.pop('openpilot.selfdrive.controls.lib.longitudinal_planner', None)
  sys.modules.pop('openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc', None)
  sp_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner')
  monkeypatch.setattr(sp_mod, 'SpeedLimitController', _NoOpSLC, raising=True)
  monkeypatch.setattr(sp_mod, 'RTIController', _NoOpRTI, raising=True)
  monkeypatch.setattr(sp_mod, 'DynamicExperimentalController', _NoOpDEC, raising=True)
  monkeypatch.setattr(sp_mod, 'VibePersonalityController', _NoOpVibe, raising=True)
  main_mod = importlib.import_module('openpilot.selfdrive.controls.lib.longitudinal_planner')
  vtc_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller')

  p = Params()
  p.put_bool('VisionTurnSpeedControl', True)
  p.put_bool('VTSCVerboseDebug', False)
  p.put_bool('VTSCWriteSnapshotFile', False)

  planner = main_mod.LongitudinalPlanner(_MockCP(), init_v=22.0, init_a=0.0)
  orig_get_bool = planner.v_tsc._get_bool_param
  orig_get_string = planner.v_tsc._get_string_param
  planner.v_tsc._update_params = lambda: None
  planner.v_tsc._is_enabled = True
  planner.v_tsc._map_strategy_mode = 'strategic'
  planner.v_tsc._fixed_lead_time_s = 0.0
  planner.v_tsc._curve_phase_offset_s = float(_curve_phase_raw_for_effective(0.0))
  planner.v_tsc._overshoot_phase_offset_s = 0.0
  planner.v_tsc._apex_exit_phase_offset_s = 0.0
  planner.v_tsc._get_bool_param = lambda key, default=False: True if key == 'MTSCLookaheadEnabled' else bool(orig_get_bool(key, default))
  planner.v_tsc._get_string_param = lambda key, default='': 'strategic' if key == 'VTSCMapStrategy' else str(orig_get_string(key, default))

  selected_profile = map_strategy.WINDING_BEHAVIOR_PROFILES[int(winding_profile_level)]

  def _resolve_profile(**kwargs):
    if int(winding_profile_level) <= 0:
      return map_strategy.DEFAULT_WINDING_BEHAVIOR_PROFILE
    if bool(kwargs.get('active')) or bool(kwargs.get('local_active')) or int(kwargs.get('mapd_level', 0) or 0) > 0:
      return selected_profile
    return map_strategy.DEFAULT_WINDING_BEHAVIOR_PROFILE

  monkeypatch.setattr(vtc_mod, 'resolve_winding_behavior_profile', _resolve_profile, raising=True)

  lat0, lon0 = 37.0, -122.0
  chained_profile = [
    0.0,
    0.0015,
    0.0040,
    0.0180,
    0.0100,
    0.0040,
    0.0025,
    0.0070,
    0.0210,
    0.0120,
    0.0040,
    0.0,
  ]
  profile_start_m = 220.0
  first_anchor_s = profile_start_m + 30.0
  second_anchor_s = profile_start_m + 80.0
  second_target_speed = float(curvature_to_speed(0.0210))
  pts = _build_map_profile_polyline(lat0, lon0, chained_profile, profile_start_m=profile_start_m)

  state = {'d': 0.0}
  planner.v_tsc._get_last_gps_pose = lambda: (lat0 + state['d'] / 111000.0, lon0, None)
  planner.v_tsc._load_map_curvatures = lambda: pts

  v_ego = 22.0
  a_ego = 0.0
  v_cruise = 27.0
  dt = 0.05
  t = 0.0
  hist: list[dict[str, float]] = []

  from unittest.mock import patch
  for _ in range(600):
    sm = _mk_sm(v_ego=v_ego, a_ego=a_ego, v_cruise_mps=v_cruise)
    with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
      planner.update(sm)
    snap = planner.v_tsc.snapshot_debug_state() or {}
    hist.append({
      't': float(t),
      'distance_traveled_m': float(state['d']),
      'dist_to_first_apex_m': float(first_anchor_s - state['d']),
      'dist_to_second_apex_m': float(second_anchor_s - state['d']),
      'v_ego': float(v_ego),
      'a_target': float(planner.output_a_target),
      'v_turn': float(planner.v_tsc.v_turn),
      'map_floor_active': bool(snap.get('map_floor_active', False)),
      'map_strategic_cap': float(snap.get('map_strategic_cap', 0.0)),
      'winding_profile_level': float(snap.get('winding_profile_level', 0.0)),
    })
    a_ego = float(planner.output_a_target)
    v_ego = max(0.0, v_ego + a_ego * dt)
    state['d'] += v_ego * dt
    t += dt

  return hist, second_target_speed, float(planner.v_tsc._longitudinal_response_model.planner_output_min_accel_mps2)


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


def test_winding_profile_chained_curves_smooths_gap_release_and_hits_second_anchor(monkeypatch):
  baseline, second_target_speed, planner_min_accel = _run_chained_winding_profile(monkeypatch, winding_profile_level=0)
  winding, _second_target_speed, _planner_min_accel = _run_chained_winding_profile(monkeypatch, winding_profile_level=4)

  assert second_target_speed == pytest.approx(_second_target_speed, abs=1e-9)
  assert planner_min_accel == pytest.approx(_planner_min_accel, abs=1e-9)

  baseline_gap = [row for row in baseline if row['dist_to_first_apex_m'] <= 0.0 and row['dist_to_second_apex_m'] >= 0.0]
  winding_gap = [row for row in winding if row['dist_to_first_apex_m'] <= 0.0 and row['dist_to_second_apex_m'] >= 0.0]
  assert baseline_gap and winding_gap

  baseline_peak_step = max(max(0.0, nxt['v_turn'] - cur['v_turn']) for cur, nxt in zip(baseline_gap, baseline_gap[1:], strict=False))
  winding_peak_step = max(max(0.0, nxt['v_turn'] - cur['v_turn']) for cur, nxt in zip(winding_gap, winding_gap[1:], strict=False))
  baseline_peak_gap_cap = max(row['v_turn'] for row in baseline_gap)
  winding_peak_gap_cap = max(row['v_turn'] for row in winding_gap)
  baseline_second_hit = next(row for row in baseline if row['v_ego'] <= second_target_speed + 0.75 and row['dist_to_second_apex_m'] >= 0.0)
  winding_second_hit = next(row for row in winding if row['v_ego'] <= second_target_speed + 0.75 and row['dist_to_second_apex_m'] >= 0.0)

  # Winding mode should keep some VTSC shaping in the gap, but it should sit much closer
  # to the more freely releasing baseline and brake later into the second anchor.
  assert winding_peak_step <= baseline_peak_step + 5e-4
  assert baseline_peak_gap_cap > winding_peak_gap_cap
  assert 0.20 < (baseline_peak_gap_cap - winding_peak_gap_cap) < 0.80
  assert winding_second_hit['dist_to_second_apex_m'] + 4.0 < baseline_second_hit['dist_to_second_apex_m']

  for hist in (baseline, winding):
    second_hit = next(row for row in hist if row['v_ego'] <= second_target_speed + 0.75 and row['dist_to_second_apex_m'] >= 0.0)
    assert second_hit['dist_to_second_apex_m'] >= 0.0
    assert min(row['a_target'] for row in hist) >= planner_min_accel - 1e-6
