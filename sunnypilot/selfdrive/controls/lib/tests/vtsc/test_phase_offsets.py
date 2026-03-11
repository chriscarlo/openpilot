#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

import sunnypilot.selfdrive.controls.lib.vtsc_map_strategy as map_strategy

from .harness import mk_vtsc_with_params


def _mk_sm_from_k_points(k_points: list[float], v_pred: float, confidence: float = 0.95):
  model = SimpleNamespace(
    orientationRate=SimpleNamespace(z=[float(k) * float(v_pred) for k in k_points]),
    velocity=SimpleNamespace(x=[float(v_pred)] * len(k_points)),
    laneLineProbs=[float(confidence)] * 4,
  )

  class SM:
    def __init__(self, m):
      self.valid = {'modelV2': True}
      self._data = {
        'modelV2': m,
        'carState': SimpleNamespace(gasPressed=False, steeringAngleDeg=0.0),
      }

    def __getitem__(self, key):
      return self._data.get(key)

  return SM(model)


def test_curve_phase_offset_changes_curvature_sample_and_vturn_cap():
  k_points = np.linspace(0.0005, 0.0200, 33).tolist()
  sm = _mk_sm_from_k_points(k_points, v_pred=25.0)

  vtsc_early = mk_vtsc_with_params()
  vtsc_early._curve_phase_offset_s = -2.0
  vtsc_early._overshoot_phase_offset_s = 0.0
  vtsc_early._apex_exit_phase_offset_s = 0.0

  vtsc_late = mk_vtsc_with_params()
  vtsc_late._curve_phase_offset_s = 2.0
  vtsc_late._overshoot_phase_offset_s = 0.0
  vtsc_late._apex_exit_phase_offset_s = 0.0

  with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: 0.0), \
       patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: 0.0):
    vtsc_early.update(sm, True, 25.0, 0.0, 33.0)
    vtsc_late.update(sm, True, 25.0, 0.0, 33.0)

  snap_early = vtsc_early.snapshot_debug_state()
  snap_late = vtsc_late.snapshot_debug_state()

  assert int(snap_early['curve_sample_idx']) > int(snap_late['curve_sample_idx'])
  assert float(vtsc_early.v_turn) < float(vtsc_late.v_turn) - 0.5


def test_overshoot_phase_offset_shifts_cap_activation_timing():
  # Ramp ahead-curvature so overshoot starts partway through the sequence.
  k_ahead_profile = np.linspace(0.0010, 0.0065, 300)

  def first_cap_time(offset_s: float) -> float | None:
    vtsc = mk_vtsc_with_params()
    vtsc._curve_phase_offset_s = 0.0
    vtsc._overshoot_phase_offset_s = float(offset_s)
    vtsc._apex_exit_phase_offset_s = 0.0

    t = 0.0
    first = None
    last_snap = None
    for k_ahead in k_ahead_profile:
      sm = _mk_sm_from_k_points([0.0007] + [float(k_ahead)] * 32, v_pred=27.0)
      with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda t=t: t), \
           patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda t=t: t):
        vtsc.update(sm, True, 27.0, 0.0, 33.0)
      snap = vtsc.snapshot_debug_state()
      last_snap = snap
      if first is None and bool(snap.get('overshoot_cap_active', False)):
        first = t
        break
      t += 0.05
    return first, last_snap or {}

  t_early, _snap_early = first_cap_time(-3.0)
  t_late, snap_late = first_cap_time(3.0)

  assert t_early is not None
  if t_late is None:
    assert bool(snap_late.get('apex_exit_ready', False)) is True
    assert bool(snap_late.get('overshoot_cap_active', False)) is False
  else:
    assert t_early < t_late - 2.0


def test_apex_exit_phase_offset_controls_overshoot_cap_release():
  def run_once(apex_offset_s: float):
    vtsc = mk_vtsc_with_params()
    vtsc._curve_phase_offset_s = 0.0
    vtsc._overshoot_phase_offset_s = 0.0
    vtsc._apex_exit_phase_offset_s = float(apex_offset_s)

    # Precondition state: overshoot cap would be active unless apex-release logic gates it off.
    vtsc._v_ego = 20.0
    vtsc._a_ego = 0.0
    vtsc._v_cruise_setpoint = 33.0
    vtsc._filtered_curvature = 0.01
    vtsc._current_lat_acc = vtsc._filtered_curvature * vtsc._v_ego * vtsc._v_ego
    vtsc._apex_indices = [5]
    vtsc._is_easing = True
    vtsc._lat_acc_overshoot_ahead = True
    vtsc._v_overshoot = 12.0
    vtsc._v_overshoot_distance = 20.0
    vtsc._overshoot_trigger_in_s = -0.5
    vtsc._occlusion_state.vision_good = True
    vtsc._occlusion_state.smoothed_confidence = 0.95

    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: 100.0), \
         patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: 100.0):
      vtsc._update_solution()
    return vtsc.snapshot_debug_state()

  snap_early = run_once(-2.0)
  snap_late = run_once(2.0)

  assert bool(snap_early['apex_exit_ready']) is True
  assert bool(snap_late['apex_exit_ready']) is False
  assert bool(snap_early['overshoot_cap_active']) is False
  assert bool(snap_late['overshoot_cap_active']) is True
  assert float(snap_early['vtsc_cmd']) > float(snap_late['vtsc_cmd'])


def test_apex_exit_phase_offset_biases_near_apex_release_threshold():
  def run_once(apex_offset_s: float):
    vtsc = mk_vtsc_with_params()
    vtsc._set_winding_behavior_profile(map_strategy.WINDING_BEHAVIOR_PROFILES[4], source='mapd')
    vtsc._apex_exit_phase_offset_s = float(apex_offset_s)
    vtsc._filtered_curvature = 0.01
    vtsc._current_lat_acc = 1.20
    vtsc._max_pred_lat_acc = 2.50
    vtsc._lat_acc_overshoot_ahead = True
    ready = vtsc._is_near_apex_release_ready()
    return ready, vtsc.snapshot_debug_state()

  ready_early, snap_early = run_once(-2.0)
  ready_late, snap_late = run_once(2.0)

  assert ready_early is True
  assert ready_late is False
  assert bool(snap_early['near_apex_release_ready']) is True
  assert bool(snap_late['near_apex_release_ready']) is False
  assert float(snap_early['apex_release_lat_acc_ratio']) < float(snap_late['apex_release_lat_acc_ratio'])
