#!/usr/bin/env python3
import json
import math
import random

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cereal import log
import sunnypilot.selfdrive.controls.lib.vtsc_map_strategy as map_strategy
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
  curvature_to_speed,
  SEVERE_OVERSHOOT_SPEED_SCALE_MIN,
  VISIBLE_MAINLINE_RELAX_DWELL_S,
  VTURN_HOLD_S,
  VisionStatus,
)
from sunnypilot.selfdrive.controls.lib.vtsc_map_strategy import (
  MapCapCandidate,
  MapStrategyState,
  compute_map_cap_candidate,
  evaluate_map_strategy,
)
from openpilot.selfdrive.controls.lib.longitudinal_response_model import (
  CRUISE_CAP_REQUIRED_DECEL_TOL_MPS2,
  build_cruise_response_model,
  predict_average_decel_for_cruise_cap,
)
from pathlib import Path

from .harness import Step, simulate_sequence, simulate_sequence_trace, mk_vtsc_with_params, load_steps_from_rlog


WINDING_PROFILE_FIXTURE = Path(__file__).with_name("fixtures") / "winding_road_profiles" / "eldorado_representatives.json"


def _load_winding_profile_fixture():
  return json.loads(WINDING_PROFILE_FIXTURE.read_text())["profiles"]


def _steps_constant(curvature: float, confidence: float, n: int, lead_d: float | None = None, curvature_ahead: float | None = None):
  for _ in range(n):
    yield Step(curvature=curvature, curvature_ahead=curvature_ahead, confidence=confidence, lead_d_rel_m=lead_d)


def test_straight_clear_no_crawl():
  # Straight road, strong vision, cruise above physics base
  v0 = 25.0
  v_cruise = 30.0
  # curvature ~ 0 => physics base near MAX; controller should not crawl
  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.0, confidence=0.95, n=80),
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap, "Snapshot missing"
  final = float(snap['final'])
  # No occlusion flags
  assert snap['vision_status'] == 'FULL'
  # Target stays healthy (≥ 24 m/s) and near base
  assert final >= 24.0
  

def test_highway_bypass_partial_occlusion():
  # Highway speed (~30 m/s), partial occlusion should not depress target
  v0 = 30.0
  v_cruise = 32.0
  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.0, confidence=0.55, n=80),
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap, "Snapshot missing"
  final = float(snap['final'])
  # At ≥ ~65 mph gate the occlusion; ensure no undue slow-down
  assert final >= v0 - 0.5


def test_freeway_cap_hold_does_not_latch_preview_only_single_frame_pulse():
  # Regression (freeway): recent rlogs showed model-only preview spikes latching a deep held cap
  # even though current curvature/steering still looked straight. A one-frame horizon pulse should
  # not stay latched once the next frame is back to straight.
  v0 = 32.0
  v_cruise = 33.0
  dt = 0.05

  # One-frame "curve ahead" pulse (horizon only), then straight.
  steps = [Step(curvature=0.0, curvature_ahead=0.012, confidence=0.95)]
  steps += [Step(curvature=0.0, curvature_ahead=0.0, confidence=0.95) for _ in range(int(VTURN_HOLD_S / dt) + 12)]

  trace = simulate_sequence_trace(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=dt, integrate_ego=False)
  assert trace and len(trace) >= 3

  # First frame can still publish the preview reduction.
  assert float(trace[0]['v_turn']) <= v_cruise - 1.0
  # But the next frame should recover promptly instead of latching the preview-only dip.
  assert float(trace[1]['v_turn']) >= v_cruise - 1e-3
  assert float(trace[2]['v_turn']) >= v_cruise - 1e-3
  assert trace[1]['cap_hold_active'] is False
  assert trace[1]['winding_release_shape_active'] is False


def test_freeway_cap_hold_preserves_corroborated_entry_reduction():
  # When current curvature already agrees a turn is underway, the same one-frame preview pulse
  # should still be held long enough for the planner to react.
  v0 = 32.0
  v_cruise = 33.0
  dt = 0.05
  hold_release_idx = math.ceil(VTURN_HOLD_S / dt)

  steps = [Step(curvature=0.0, curvature_ahead=0.012, confidence=0.95, desired_curvature=0.0014, actual_curvature=0.0012)]
  steps += [
    Step(curvature=0.0, curvature_ahead=0.0, confidence=0.95, desired_curvature=0.0014, actual_curvature=0.0012)
    for _ in range(hold_release_idx + 100)
  ]

  trace = simulate_sequence_trace(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=dt, integrate_ego=False)
  assert trace and len(trace) > hold_release_idx + 2

  assert float(trace[0]['v_turn']) <= v_cruise - 1.0
  assert float(trace[1]['v_turn']) <= v_cruise - 1.0
  assert trace[1]['cap_hold_active'] is True
  assert trace[1]['winding_release_shape_active'] is True
  assert trace[hold_release_idx - 1]['cap_hold_active'] is True
  assert trace[hold_release_idx]['cap_hold_active'] is False
  assert trace[hold_release_idx]['winding_release_shape_active'] is True
  assert float(trace[hold_release_idx]['v_turn']) > float(trace[hold_release_idx - 1]['v_turn'])
  assert float(trace[-1]['v_turn']) >= v_cruise - 1e-3


def test_low_confidence_hold_stabilizes_single_frame_pulse_without_occlusion_state():
  # Low-confidence cap-hold is still useful for anti-flap stabilization, but it now keys directly
  # off raw confidence rather than entering an occluded controller state.
  v0 = 15.0
  v_cruise = 24.0
  dt = 0.05
  conf = 0.52

  steps = [Step(curvature=0.0, curvature_ahead=0.0, confidence=conf) for _ in range(10)]
  steps += [Step(curvature=0.0, curvature_ahead=0.012, confidence=conf)]
  steps += [Step(curvature=0.0, curvature_ahead=0.0, confidence=conf) for _ in range(int(VTURN_HOLD_S / dt) + 20)]

  trace = simulate_sequence_trace(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=dt, integrate_ego=False)
  assert trace and len(trace) >= 3

  first = 10
  assert float(trace[first]['v_turn']) <= v_cruise - 1.0
  assert float(trace[first + 1]['v_turn']) <= v_cruise - 1.0
  assert float(trace[-1]['v_turn']) >= v_cruise - 1e-3


def test_freeway_low_confidence_hold_requires_local_curve_corroboration():
  # At freeway speeds, low-confidence preview pulses should not latch by themselves. They need
  # current steering/lateral corroboration just like the high-confidence freeway path.
  v0 = 32.0
  v_cruise = 33.0
  dt = 0.05
  conf = 0.69

  steps = [Step(curvature=0.0, curvature_ahead=0.012, confidence=conf)]
  steps += [Step(curvature=0.0, curvature_ahead=0.0, confidence=conf) for _ in range(int(VTURN_HOLD_S / dt) + 12)]

  trace = simulate_sequence_trace(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=dt, integrate_ego=False)
  assert trace and len(trace) >= 3

  assert float(trace[0]['v_turn']) <= v_cruise - 1.0
  assert float(trace[1]['v_turn']) >= v_cruise - 1e-3
  assert float(trace[2]['v_turn']) >= v_cruise - 1e-3
  assert trace[1]['cap_hold_active'] is False
  assert trace[1]['winding_release_shape_active'] is False


def test_low_confidence_with_lead_stays_at_visible_cap_without_bypass():
  # Low confidence plus a close lead should not activate any occlusion-specific bypass path.
  # The controller should simply follow the visible-cap result.
  v0 = 20.0
  v_cruise = 22.0
  k = 0.001
  headway_s = 2.4
  d_rel = v0 * headway_s
  vtsc = mk_vtsc_with_params()
  steps = list(_steps_constant(curvature=k, confidence=0.4, n=40, lead_d=None))
  steps += list(_steps_constant(curvature=k, confidence=0.4, n=40, lead_d=d_rel))
  snap = simulate_sequence(steps=steps, vtsc=vtsc, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  assert snap, "Snapshot missing"
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(0.4, abs=1e-6)
  assert bool(snap['occl_lead_bypass_active']) is False
  assert float(snap['final']) >= v_cruise - 1e-3


def test_curve_exit_never_latches_fov_occlusion_even_with_mediocre_confidence():
  # Low-confidence curve exit should recover through the normal visible-cap path.
  # No FOV occlusion latch or lead-specific bypass should appear in the trace.
  v0 = 25.0
  v_cruise = 30.0
  dt = 0.05

  k_curve = 0.012
  k_ahead = 0.004
  conf = 0.55
  lead_d = 30.0
  n_curve = 10
  n_straight = 60

  def run(lead_d_rel_m: float | None):
    vtsc = mk_vtsc_with_params()
    steps = list(_steps_constant(curvature=k_curve, confidence=conf, n=n_curve, lead_d=lead_d_rel_m))
    steps += list(_steps_constant(curvature=0.0, confidence=conf, n=n_straight, lead_d=lead_d_rel_m, curvature_ahead=k_ahead))
    return simulate_sequence_trace(steps=steps, vtsc=vtsc, v0_mps=v0, v_cruise_mps=v_cruise, dt=dt)

  def first_clear_idx(trace):
    for i in range(n_curve, len(trace)):
      if trace[i]['fov_occluded'] is False:
        return i
    return None

  def v_turn_recovers_smoothly(trace, start_idx: int) -> bool:
    window = trace[start_idx:start_idx + int(3.0 / dt) + 1]
    if not window:
      return False
    if float(window[0]['v_turn']) >= v_cruise - 1.0:
      return False
    net_rise = float(window[-1]['v_turn']) - float(window[0]['v_turn'])
    peak_step = max(
      max(0.0, float(nxt['v_turn']) - float(cur['v_turn']))
      for cur, nxt in zip(window, window[1:], strict=False)
    )
    return bool(peak_step <= 0.25 and net_rise >= 3.0)

  trace_no_lead = run(None)
  assert all(not s['fov_occluded'] for s in trace_no_lead)
  clear_idx = first_clear_idx(trace_no_lead)
  assert clear_idx is not None
  assert clear_idx == n_curve
  assert v_turn_recovers_smoothly(trace_no_lead, clear_idx) is True

  trace_lead = run(lead_d)
  assert all(not s['fov_occluded'] for s in trace_lead)
  assert all(not s['occl_lead_bypass_active'] for s in trace_lead)
  clear_idx_lead = first_clear_idx(trace_lead)
  assert clear_idx_lead is not None
  assert clear_idx_lead == n_curve
  assert v_turn_recovers_smoothly(trace_lead, clear_idx_lead) is True

  assert abs(clear_idx_lead - clear_idx) == 0


def test_low_speed_calibration_relaxes_clean_low_headroom_curve():
  v0 = 11.5
  v_cruise = 16.0
  steps_neutral = [
    Step(curvature=0.02, curvature_ahead=0.02, confidence=0.95)
    for _ in range(220)
  ]
  steps_relax = [
    Step(
      curvature=0.02,
      curvature_ahead=0.02,
      confidence=0.95,
      desired_curvature=0.020,
      actual_curvature=0.0194,
      lateral_output=0.24,
      lateral_saturated=False,
    )
    for _ in range(220)
  ]

  snap_neutral = simulate_sequence(steps=steps_neutral, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  snap_relax = simulate_sequence(steps=steps_relax, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  assert float(snap_relax['low_speed_calibration_state']) > 0.01
  assert float(snap_relax['low_speed_calibration_scale']) > 1.0
  assert str(snap_relax['low_speed_calibration_reason']) == 'relax_clean'
  assert float(snap_relax['v_base']) > float(snap_neutral['v_base']) + 0.05


def test_low_speed_calibration_driver_override_weights_larger_divergence_more_heavily():
  v0 = 10.5
  v_cruise = 16.0
  common = dict(
    curvature=0.02,
    curvature_ahead=0.02,
    confidence=0.95,
    desired_curvature=0.020,
    actual_curvature=0.0195,
    lateral_output=0.52,
    lateral_saturated=False,
    gas_pressed=True,
  )
  steps_mild = [Step(**common, applied_accel=0.12) for _ in range(160)]
  steps_strong = [Step(**common, applied_accel=0.42) for _ in range(160)]

  snap_mild = simulate_sequence(steps=steps_mild, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  snap_strong = simulate_sequence(steps=steps_strong, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  # The retuned source sigmoid already covers more of the former relax gap, so the residual
  # low-speed calibration state is smaller. The key invariant is still relative weighting:
  # larger driver override divergence should produce a measurably larger relax state.
  assert float(snap_strong['low_speed_calibration_state']) > float(snap_mild['low_speed_calibration_state']) + 0.002
  assert float(snap_strong['low_speed_calibration_override_ema']) > float(snap_mild['low_speed_calibration_override_ema']) + 0.20
  assert float(snap_strong['low_speed_calibration_divergence_mps']) > float(snap_mild['low_speed_calibration_divergence_mps']) + 1.0
  assert str(snap_strong['low_speed_calibration_reason']) == 'relax_override'


def test_low_speed_calibration_does_not_tighten_during_driver_override_saturation():
  v0 = 10.5
  v_cruise = 16.0
  steps_override = [
    Step(
      curvature=0.02,
      curvature_ahead=0.02,
      confidence=0.95,
      desired_curvature=0.020,
      actual_curvature=0.0195,
      lateral_output=0.52,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.42,
    )
    for _ in range(160)
  ]
  steps_saturated = [
    Step(
      curvature=0.02,
      curvature_ahead=0.02,
      confidence=0.95,
      desired_curvature=0.024,
      actual_curvature=0.016,
      lateral_output=0.96,
      lateral_saturated=True,
      gas_pressed=True,
      applied_accel=0.42,
    )
    for _ in range(160)
  ]

  snap_override = simulate_sequence(steps=steps_override, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  snap_saturated = simulate_sequence(steps=steps_saturated, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  # With the higher base freeway/55-65 mph sigmoid shoulder, the override relax state remains
  # positive but no longer needs to climb as high before the source curve covers the gap itself.
  assert float(snap_override['low_speed_calibration_state']) > 0.002
  assert str(snap_override['low_speed_calibration_reason']) == 'relax_override'
  assert float(snap_saturated['low_speed_calibration_base_state']) > -0.002
  assert float(snap_saturated['low_speed_calibration_state']) > -0.002
  assert float(snap_saturated['low_speed_calibration_override_ema']) == pytest.approx(0.0, abs=1e-6)
  assert str(snap_saturated['low_speed_calibration_reason']) == 'driver_override_passthrough'


def test_low_speed_calibration_tightens_on_sustained_saturation():
  v0 = 11.5
  v_cruise = 16.0
  steps_neutral = [
    Step(curvature=0.02, curvature_ahead=0.02, confidence=0.95)
    for _ in range(180)
  ]
  steps_tighten = [
    Step(
      curvature=0.02,
      curvature_ahead=0.02,
      confidence=0.95,
      desired_curvature=0.024,
      actual_curvature=0.016,
      lateral_output=0.96,
      lateral_saturated=True,
    )
    for _ in range(180)
  ]

  snap_neutral = simulate_sequence(steps=steps_neutral, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  snap_tighten = simulate_sequence(steps=steps_tighten, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  assert float(snap_tighten['low_speed_calibration_state']) < -0.02
  assert float(snap_tighten['low_speed_calibration_scale']) < 1.0
  assert str(snap_tighten['low_speed_calibration_reason']) == 'tighten_saturated'
  assert float(snap_tighten['v_base']) < float(snap_neutral['v_base']) - 0.10


def test_low_speed_calibration_does_not_tighten_on_high_effort_without_saturation():
  v0 = 11.5
  v_cruise = 16.0
  steps = [
    Step(
      curvature=0.02,
      curvature_ahead=0.02,
      confidence=0.95,
      desired_curvature=0.024,
      actual_curvature=0.014,
      lateral_output=0.97,
      lateral_saturated=False,
    )
    for _ in range(220)
  ]

  snap = simulate_sequence(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  assert float(snap['low_speed_calibration_state']) > -0.005
  assert float(snap['low_speed_calibration_scale']) >= 0.999
  assert str(snap['low_speed_calibration_reason']) == 'decay_ambiguous'


def test_low_speed_calibration_loads_and_persists_learned_state():
  ctrl = mk_vtsc_with_params(value_overrides={"VisionTurnSpeedControlLowSpeedLearnedState": "0.025"})
  assert float(getattr(ctrl, '_low_speed_calibration_state', 0.0)) == pytest.approx(0.025, abs=1e-9)

  relax_steps = [
    Step(
      curvature=0.02,
      curvature_ahead=0.02,
      confidence=0.95,
      desired_curvature=0.020,
      actual_curvature=0.0195,
      lateral_output=0.20,
      lateral_saturated=False,
    )
    for _ in range(220)
  ]
  simulate_sequence(steps=relax_steps, vtsc=ctrl, v0_mps=11.5, v_cruise_mps=16.0, dt=0.05)

  persist_calls = [
    call for call in getattr(ctrl._params, 'put_nonblocking', MagicMock()).call_args_list
    if call.args and call.args[0] == 'VisionTurnSpeedControlLowSpeedLearnedState'
  ]
  assert persist_calls
  persisted_value = float(persist_calls[-1].args[1])
  assert persisted_value > 0.025


def test_low_speed_calibration_toggle_disables_live_learning_and_persistence():
  ctrl = mk_vtsc_with_params(
    bool_overrides={"VisionTurnSpeedControlLowSpeedLearningEnabled": False},
    value_overrides={"VisionTurnSpeedControlLowSpeedLearnedState": "0.040"},
  )
  assert float(getattr(ctrl, '_low_speed_calibration_state', 0.0)) == pytest.approx(0.0, abs=1e-9)

  steps_override = [
    Step(
      curvature=0.02,
      curvature_ahead=0.02,
      confidence=0.95,
      desired_curvature=0.020,
      actual_curvature=0.0195,
      lateral_output=0.24,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.42,
    )
    for _ in range(220)
  ]
  snap = simulate_sequence(steps=steps_override, vtsc=ctrl, v0_mps=10.5, v_cruise_mps=16.0, dt=0.05)

  assert str(snap['low_speed_calibration_reason']) == 'disabled_by_toggle'
  assert float(snap['low_speed_calibration_state']) == pytest.approx(0.0, abs=1e-9)
  assert float(snap['low_speed_calibration_scale']) == pytest.approx(1.0, abs=1e-9)
  persist_calls = [
    call for call in getattr(ctrl._params, 'put_nonblocking', MagicMock()).call_args_list
    if call.args and call.args[0] in (
      'VisionTurnSpeedControlLowSpeedLearnedState',
      'VisionTurnSpeedControlDriverOverrideCurveProfile',
    )
  ]
  assert not persist_calls


def test_low_speed_calibration_high_end_param_limits_sigmoid_range():
  k_curve = next(
    k for k in (0.012, 0.011, 0.010, 0.009, 0.008)
    if float(curvature_to_speed(k)) * 2.2369362920544 > 26.0
  )
  ctrl_default = mk_vtsc_with_params(value_overrides={
    "VisionTurnSpeedControlLowSpeedLearnedState": "0.040",
  })
  scale_default = float(ctrl_default._low_speed_calibration_scale(k_curve))

  ctrl_limited = mk_vtsc_with_params(value_overrides={
    "VisionTurnSpeedControlLowSpeedLearnedState": "0.040",
    "VisionTurnSpeedControlLowSpeedLearnedHighEndMph": "25.0",
  })
  scale_limited = float(ctrl_limited._low_speed_calibration_scale(k_curve))

  assert scale_default > 1.03
  assert scale_limited == pytest.approx(1.0, abs=1e-6)


def test_driver_override_learning_applies_above_legacy_low_speed_band():
  v0 = 31.5
  v_cruise = 35.0
  ctrl = mk_vtsc_with_params()
  steps = [
    Step(
      curvature=0.0048,
      curvature_ahead=0.0048,
      confidence=0.95,
      desired_curvature=0.0048,
      actual_curvature=0.00475,
      lateral_output=0.52,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.0,
    )
    for _ in range(30)
  ]

  snap = simulate_sequence(steps=steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  assert float(snap['low_speed_calibration_curve_mph']) > 60.0
  assert float(snap['low_speed_calibration_override_state']) > 0.02
  assert float(snap['low_speed_calibration_scale']) > 1.02
  assert str(snap['low_speed_calibration_reason']) == 'relax_override'


def test_driver_override_learning_only_relaxes_local_curvature_neighborhood():
  v0 = 31.5
  v_cruise = 35.0
  local_k = 0.0048
  near_k = 0.0055
  far_k = 0.02
  ctrl = mk_vtsc_with_params()
  steps = [
    Step(
      curvature=local_k,
      curvature_ahead=local_k,
      confidence=0.95,
      desired_curvature=local_k,
      actual_curvature=0.00475,
      lateral_output=0.52,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.0,
    )
    for _ in range(30)
  ]

  snap = simulate_sequence(steps=steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  local_scale = float(ctrl._low_speed_calibration_scale(local_k))
  near_scale = float(ctrl._low_speed_calibration_scale(near_k))
  far_scale = float(ctrl._low_speed_calibration_scale(far_k))

  assert str(snap['low_speed_calibration_reason']) == 'relax_override'
  assert local_scale > 1.02
  assert near_scale > 1.01
  assert near_scale < local_scale
  assert far_scale == pytest.approx(1.0, abs=5e-3)
  assert local_scale > far_scale + 0.015
  assert near_scale > far_scale + 0.008


def test_driver_override_learning_persists_curvature_local_profile():
  v0 = 31.5
  v_cruise = 35.0
  local_k = 0.0048
  ctrl = mk_vtsc_with_params()
  steps = [
    Step(
      curvature=local_k,
      curvature_ahead=local_k,
      confidence=0.95,
      desired_curvature=local_k,
      actual_curvature=0.00475,
      lateral_output=0.52,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.0,
    )
    for _ in range(220)
  ]

  simulate_sequence(steps=steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  persist_calls = [
    call for call in getattr(ctrl._params, 'put_nonblocking', MagicMock()).call_args_list
    if call.args and call.args[0] == 'VisionTurnSpeedControlDriverOverrideCurveProfile'
  ]
  assert persist_calls
  persisted_profile = json.loads(persist_calls[-1].args[1])
  assert isinstance(persisted_profile, dict)
  assert max(float(v) for v in persisted_profile['values']) > 0.02
  assert sum(1 for v in persisted_profile['values'] if float(v) > 0.005) >= 3


def test_driver_override_learning_reaches_material_relax_after_three_short_bursts():
  v0 = 12.8
  v_cruise = 16.0
  burst = Step(
    curvature=0.02,
    curvature_ahead=0.02,
    confidence=0.95,
    desired_curvature=0.020,
    actual_curvature=0.0195,
    lateral_output=0.52,
    lateral_saturated=False,
    gas_pressed=True,
    applied_accel=0.42,
  )
  settle = Step(
    curvature=0.02,
    curvature_ahead=0.02,
    confidence=0.95,
    desired_curvature=0.020,
    actual_curvature=0.0195,
    lateral_output=0.52,
    lateral_saturated=False,
    gas_pressed=False,
    applied_accel=0.0,
  )
  steps = [
    burst, burst, burst, burst, burst,
    settle, settle, settle,
    burst, burst, burst, burst, burst,
    settle, settle, settle,
    burst, burst, burst, burst, burst,
  ]

  snap = simulate_sequence(steps=steps, vtsc=mk_vtsc_with_params(), v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  assert float(snap['low_speed_calibration_override_state']) > 0.025
  assert float(snap['low_speed_calibration_state']) > 0.025
  assert float(snap['low_speed_calibration_scale']) > 1.025
  assert str(snap['low_speed_calibration_reason']) == 'relax_override'


# ===== Map-aware calibration tests =====


def test_map_aware_calibration_gas_override_during_approach():
  """Gas override on a straight approach with strategic map active should learn at the map anchor curvature."""
  v0 = 15.0
  v_cruise = 20.0
  map_anchor_k = 0.015  # map sees a curve ahead
  map_cap_mps = 12.0    # map cap speed
  ctrl = mk_vtsc_with_params()
  steps = [
    Step(
      curvature=0.0,            # straight road (approach phase)
      curvature_ahead=0.0,
      confidence=0.95,
      desired_curvature=0.0005,
      actual_curvature=0.0004,
      lateral_output=0.10,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.0,        # hold speed above map cap
      map_anchor_k=map_anchor_k,
      map_cap_mps=map_cap_mps,
    )
    for _ in range(160)
  ]

  snap = simulate_sequence(steps=steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  # The override profile should have learned at the map anchor curvature
  scale_at_anchor = float(ctrl._low_speed_calibration_scale(map_anchor_k))
  assert scale_at_anchor > 1.005, f"scale at anchor {scale_at_anchor} should be > 1.005"
  assert bool(snap['low_speed_calibration_map_active']) is True
  assert float(snap['low_speed_calibration_map_anchor_k']) == pytest.approx(map_anchor_k, abs=1e-6)
  assert str(snap['low_speed_calibration_reason']) == 'relax_override'


def test_map_aware_calibration_does_not_contaminate_zero_curvature_bin():
  """Learning at map anchor curvature should not bleed into near-zero curvature bins."""
  v0 = 15.0
  v_cruise = 20.0
  map_anchor_k = 0.015
  map_cap_mps = 12.0
  ctrl = mk_vtsc_with_params()
  steps = [
    Step(
      curvature=0.0,
      curvature_ahead=0.0,
      confidence=0.95,
      desired_curvature=0.0005,
      actual_curvature=0.0004,
      lateral_output=0.10,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.0,
      map_anchor_k=map_anchor_k,
      map_cap_mps=map_cap_mps,
    )
    for _ in range(160)
  ]

  simulate_sequence(steps=steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  # Near-zero curvature should remain at baseline (Gaussian spread shouldn't reach it)
  scale_near_zero = float(ctrl._low_speed_calibration_scale(0.001))
  assert scale_near_zero == pytest.approx(1.0, abs=0.005), \
    f"scale at k=0.001 should be ~1.0 but got {scale_near_zero}"


def test_map_aware_calibration_dual_bin_when_curvatures_differ():
  """When experienced curvature differs from map anchor by >1 sigma, both bins should learn."""
  v0 = 12.0
  v_cruise = 18.0
  experienced_k = 0.005   # in the curve, model sees this
  map_anchor_k = 0.015    # map anchor is tighter (different part of curve)
  map_cap_mps = 10.0
  ctrl = mk_vtsc_with_params()
  steps = [
    Step(
      curvature=experienced_k,
      curvature_ahead=experienced_k,
      confidence=0.95,
      desired_curvature=experienced_k,
      actual_curvature=experienced_k * 0.97,
      lateral_output=0.30,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.0,
      map_anchor_k=map_anchor_k,
      map_cap_mps=map_cap_mps,
    )
    for _ in range(160)
  ]

  snap = simulate_sequence(steps=steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  scale_at_anchor = float(ctrl._low_speed_calibration_scale(map_anchor_k))
  scale_at_experienced = float(ctrl._low_speed_calibration_scale(experienced_k))

  # log10(0.015) - log10(0.005) = 0.48, sigma=0.12, so 4 sigma apart → dual-bin
  assert bool(snap['low_speed_calibration_dual_bin']) is True
  assert scale_at_anchor > 1.005, f"map anchor bin should have learned: {scale_at_anchor}"
  assert scale_at_experienced > 1.002, f"experienced bin should have learned (secondary): {scale_at_experienced}"
  # Primary bin (map anchor) should have learned more than secondary
  assert scale_at_anchor > scale_at_experienced


def test_map_aware_calibration_no_dual_bin_when_curvatures_close():
  """When experienced curvature is close to map anchor (<1 sigma), no dual-bin."""
  v0 = 12.0
  v_cruise = 18.0
  experienced_k = 0.014
  map_anchor_k = 0.015    # very close to experienced
  map_cap_mps = 10.0
  ctrl = mk_vtsc_with_params()
  steps = [
    Step(
      curvature=experienced_k,
      curvature_ahead=experienced_k,
      confidence=0.95,
      desired_curvature=experienced_k,
      actual_curvature=experienced_k * 0.97,
      lateral_output=0.30,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.0,
      map_anchor_k=map_anchor_k,
      map_cap_mps=map_cap_mps,
    )
    for _ in range(80)
  ]

  snap = simulate_sequence(steps=steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  # log10(0.015) - log10(0.014) ≈ 0.03, less than 1 sigma (0.12) → no dual-bin
  assert bool(snap['low_speed_calibration_dual_bin']) is False


def test_map_aware_calibration_tighten_uses_experienced_curvature():
  """Saturation tighten should write to the experienced curvature bin, not the map anchor.

  Use curvatures far enough apart (>3σ in log10 space) that the Gaussian tails
  from the tighten at the experienced bin don't cross-contaminate the anchor bin.
  σ=0.12, 3σ cutoff → curvatures must be >10^0.36 ≈ 2.3× apart.
  k=0.003 vs k=0.015 → 5× apart (log10 distance=0.70, 5.8σ) — well beyond cutoff.
  """
  v0 = 12.0
  v_cruise = 18.0
  experienced_k = 0.003   # far from anchor (beyond Gaussian cutoff)
  map_anchor_k = 0.015
  map_cap_mps = 10.0
  ctrl = mk_vtsc_with_params()

  # First, learn some relax at the map anchor bin via gas override
  relax_steps = [
    Step(
      curvature=experienced_k,
      curvature_ahead=experienced_k,
      confidence=0.95,
      desired_curvature=experienced_k,
      actual_curvature=experienced_k * 0.97,
      lateral_output=0.30,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.0,
      map_anchor_k=map_anchor_k,
      map_cap_mps=map_cap_mps,
    )
    for _ in range(200)
  ]
  simulate_sequence(steps=relax_steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  scale_anchor_after_relax = float(ctrl._low_speed_calibration_scale(map_anchor_k))

  # Now saturate at experienced curvature (no gas, with map still active)
  tighten_steps = [
    Step(
      curvature=experienced_k,
      curvature_ahead=experienced_k,
      confidence=0.95,
      desired_curvature=experienced_k * 1.2,
      actual_curvature=experienced_k * 0.8,
      lateral_output=0.96,
      lateral_saturated=True,
      gas_pressed=False,
      applied_accel=0.0,
      map_anchor_k=map_anchor_k,
      map_cap_mps=map_cap_mps,
    )
    for _ in range(200)
  ]
  simulate_sequence(steps=tighten_steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  scale_anchor_after_tighten = float(ctrl._low_speed_calibration_scale(map_anchor_k))

  # Map anchor bin should be preserved (tighten targets experienced curvature,
  # and at 5.8σ distance the Gaussian bleed is zero)
  assert scale_anchor_after_tighten >= scale_anchor_after_relax - 0.005, \
    f"anchor bin should be preserved: {scale_anchor_after_relax} -> {scale_anchor_after_tighten}"


def test_steering_headroom_per_curvature_relax_accumulates():
  """Sustained low steering effort at a curve should slowly relax the per-curvature override profile."""
  v0 = 10.5
  v_cruise = 16.0
  curve_k = 0.02  # maps to ~24 mph → within taper band
  ctrl = mk_vtsc_with_params()
  steps = [
    Step(
      curvature=curve_k,
      curvature_ahead=curve_k,
      confidence=0.95,
      desired_curvature=curve_k,
      actual_curvature=curve_k * 0.97,
      lateral_output=0.15,       # low effort → headroom
      lateral_saturated=False,
      gas_pressed=False,
    )
    for _ in range(400)  # ~20 seconds
  ]

  simulate_sequence(steps=steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  scale_at_k = float(ctrl._low_speed_calibration_scale(curve_k))
  override_state = float(ctrl._low_speed_calibration_override_state_for_curvature(curve_k))
  # Per-curvature override should have accumulated a small positive value
  assert override_state > 0.003, \
    f"steering headroom should have relaxed per-curvature profile: {override_state}"
  assert scale_at_k > 1.003, f"scale should reflect per-curvature relax: {scale_at_k}"


def test_no_regression_without_map_context():
  """Without map context, gas override should behave identically to before (no map_active)."""
  v0 = 10.5
  v_cruise = 16.0
  steps = [
    Step(
      curvature=0.02,
      curvature_ahead=0.02,
      confidence=0.95,
      desired_curvature=0.020,
      actual_curvature=0.0195,
      lateral_output=0.52,
      lateral_saturated=False,
      gas_pressed=True,
      applied_accel=0.42,
      # No map_anchor_k / map_cap_mps → no map context
    )
    for _ in range(160)
  ]

  snap = simulate_sequence(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  assert bool(snap['low_speed_calibration_map_active']) is False
  assert float(snap['low_speed_calibration_override_ema']) > 0.20
  assert str(snap['low_speed_calibration_reason']) == 'relax_override'
  assert float(snap['low_speed_calibration_state']) > 0.002


def test_physics_sigmoid_lifts_freeway_sweeper_band_without_bloating_sub_50_curve():
  mph = 2.2369362920544

  def pure_sigmoid_speed_mph(curvature: float, *, a: float, b: float, c: float, d: float, max_lat: float) -> float:
    result = float(a) / (1.0 + math.exp(float(b) * (float(curvature) - float(c)))) + float(d)
    result = max(0.1, min(result, float(max_lat)))
    return math.sqrt(result / float(curvature)) * mph

  # Keep the regression deterministic even if other tests instantiated a controller and refreshed
  # the module-level knobs from Params earlier in the same process.
  phys_a = -3.26
  phys_b = -6270.0
  phys_c = 0.00501
  phys_d = 5.607
  phys_max_lat = 4.478

  # Regression target from the seg-10 "should be high-60s/low-70s" sweeper band.
  k_ref = 0.004567423064561596
  k_band_hi = 0.0048

  # Anchor curvatures taken from the current curve's ~40/45/50 mph region. Keep these close.
  k_40 = 0.006808786689291953
  k_45 = 0.0058203472225662614
  k_50 = 0.00526743605396956

  v_ref_mph = pure_sigmoid_speed_mph(k_ref, a=phys_a, b=phys_b, c=phys_c, d=phys_d, max_lat=phys_max_lat)
  v_band_hi_mph = pure_sigmoid_speed_mph(k_band_hi, a=phys_a, b=phys_b, c=phys_c, d=phys_d, max_lat=phys_max_lat)
  v_50_mph = pure_sigmoid_speed_mph(k_50, a=phys_a, b=phys_b, c=phys_c, d=phys_d, max_lat=phys_max_lat)
  v_45_mph = pure_sigmoid_speed_mph(k_45, a=phys_a, b=phys_b, c=phys_c, d=phys_d, max_lat=phys_max_lat)
  v_40_mph = pure_sigmoid_speed_mph(k_40, a=phys_a, b=phys_b, c=phys_c, d=phys_d, max_lat=phys_max_lat)

  assert 68.0 <= v_ref_mph <= 72.0
  assert 68.0 <= v_band_hi_mph <= 72.0
  assert v_50_mph <= 53.0
  assert v_45_mph <= 46.0
  assert v_40_mph <= 42.0


def test_runtime_param_refresh_preserves_freeway_sweeper_sigmoid_defaults(monkeypatch):
  import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vtc_mod

  monkeypatch.setattr(vtc_mod, "PHYSICS_A", -3.26, raising=False)
  monkeypatch.setattr(vtc_mod, "PHYSICS_B", -6270.0, raising=False)
  monkeypatch.setattr(vtc_mod, "PHYSICS_C", 0.00501, raising=False)
  monkeypatch.setattr(vtc_mod, "PHYSICS_D", 5.607, raising=False)
  monkeypatch.setattr(vtc_mod, "PHYSICS_MAX_LAT_ACCEL", 4.478, raising=False)

  ctrl = mk_vtsc_with_params()
  ctrl._update_params()

  assert float(vtc_mod.PHYSICS_A) == pytest.approx(-3.26, abs=1e-6)
  assert float(vtc_mod.PHYSICS_B) == pytest.approx(-6270.0, abs=1e-6)
  assert float(vtc_mod.PHYSICS_C) == pytest.approx(0.00501, abs=1e-9)
  assert float(vtc_mod.PHYSICS_D) == pytest.approx(5.607, abs=1e-6)
  assert float(vtc_mod.PHYSICS_MAX_LAT_ACCEL) == pytest.approx(4.478, abs=1e-6)


def test_low_speed_calibration_decays_back_toward_neutral_when_curve_feedback_disappears():
  v0 = 11.5
  v_cruise = 16.0
  ctrl = mk_vtsc_with_params()
  relax_steps = [
    Step(
      curvature=0.02,
      curvature_ahead=0.02,
      confidence=0.95,
      desired_curvature=0.020,
      actual_curvature=0.0194,
      lateral_output=0.24,
      lateral_saturated=False,
    )
    for _ in range(220)
  ]
  decay_steps = [Step(curvature=0.0, confidence=0.95) for _ in range(220)]

  snap_relax = simulate_sequence(steps=relax_steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  snap_decay = simulate_sequence(steps=decay_steps, vtsc=ctrl, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)

  assert float(snap_relax['low_speed_calibration_state']) > 0.01
  assert float(snap_decay['low_speed_calibration_state']) < float(snap_relax['low_speed_calibration_state'])
  assert float(snap_decay['low_speed_calibration_scale']) == pytest.approx(1.0, abs=1e-6)
  assert str(snap_decay['low_speed_calibration_reason']) in ('decay_no_feedback', 'decay_not_relevant', 'decay_ambiguous')


def test_lead_bypass_only_applies_at_close_headway():
  # Sanity check: a distant lead should behave like "no lead" for occlusion logic.
  v0 = 25.0
  v_cruise = 30.0
  conf = 0.40
  k = 0.002
  far_lead_d = 180.0  # headway > 3s at these speeds

  vtsc_far = mk_vtsc_with_params()
  snap = simulate_sequence(
    steps=_steps_constant(curvature=k, confidence=conf, n=40, lead_d=far_lead_d),
    vtsc=vtsc_far,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap and bool(snap['lead']) is True
  assert float(snap['hw']) >= 3.0
  assert bool(snap['occl_lead_bypass_active']) is False


def test_low_confidence_visible_curve_tracks_visible_cap_without_occlusion_state():
  # Low confidence on a visible curve should still produce the visible-cap result, but must not
  # create an occluded state or comfort-limited no-raise behavior.
  v0 = 20.0
  v_cruise = 22.0
  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.008, confidence=0.4, n=80),
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(0.4, abs=1e-6)
  assert bool(snap['occluded']) is False
  assert bool(snap['occl_lead_bypass_active']) is False
  assert bool(snap['occl_positive_margin']) is False
  assert float(snap['vtsc_cmd']) < v_cruise - 1.0
  assert str(snap['active_cap']) == 'visible'
  # Under the retuned source sigmoid, the visible-curve cap itself now asks for decel here.
  # The invariant we care about is that the controller stays on the visible path without any
  # occlusion-state takeover or bypass behavior.
  assert float(snap['decel_cmd']) < -0.1


def test_severe_occlusion_reacquisition_adds_nudge():
  # Drop vision to severe occlusion, then recover; expect a small positive accel nudge on reacquisition
  v0 = 20.0
  v_cruise = 25.0
  steps = list(_steps_constant(curvature=0.002, confidence=0.1, n=40))
  steps += list(_steps_constant(curvature=0.0, confidence=0.95, n=20))
  snap, history = simulate_sequence(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05, capture_history=True)
  assert snap
  assert snap['vision_status'] == 'FULL'
  first_full = None
  for idx, entry in enumerate(history):
    if entry.get('vision_status') == 'FULL':
      first_full = idx
      break
  assert first_full is not None, "Expected to reacquire full vision"
  window = history[first_full:first_full + 6]
  assert any(entry.get('a_cmd', 0.0) >= (0.18 - 1e-3) for entry in window)


def test_severe_confidence_on_straight_does_not_block_raise():
  # Very low lane-line confidence should remain observable in telemetry, but must not create a
  # separate VTSC visibility state or block acceleration on a straight road.
  v0 = 20.0
  v_cruise = 25.0
  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.0, confidence=0.10, n=15),
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert snap['vision_status'] == 'FULL'
  assert bool(snap['occluded']) is False
  assert float(snap['raw_path_conf']) == pytest.approx(0.1, abs=1e-6)
  assert float(snap['a_last']) >= 0.05


def test_severe_confidence_low_speed_lead_does_not_freeze():
  # Crawling behind a lead with terrible lane confidence should still resume through the normal
  # controller path. No lead-bypass or occluded state should be needed.
  v0 = 1.0
  v_cruise = 15.0
  conf = 0.10
  k = 5e-4  # tiny curvature (above k_min) where physics cap is effectively cruise
  lead_d = 15.0

  snap = simulate_sequence(
    steps=_steps_constant(curvature=k, confidence=conf, n=25, lead_d=lead_d),
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(conf, abs=1e-6)
  assert bool(snap['lead']) is True
  assert bool(snap['occl_lead_bypass_active']) is False
  assert float(snap['vtsc_cmd']) > float(snap['v']) + 0.5
  assert float(snap['a_last']) >= 0.05


def test_severe_confidence_model_flat_steering_fallback_slows_for_sharp_curve():
  # Regression reproducer (non-map):
  #
  # Some real-world off-ramps can cause model confidence to drop to SEVERE/LOST and
  # simultaneously produce a near-zero curvature estimate from the model ("fail open").
  #
  # In that situation, if the driver is clearly steering into a real curve, VTSC should
  # not keep publishing a near-cruise cap just because the model is flat.
  v0 = 26.8  # ~60 mph
  v_cruise = 26.8
  dt = 0.05
  conf = 0.05
  k_curve = 0.025  # ~20 mph safe speed for curvature_to_speed()

  vtsc = mk_vtsc_with_params()
  assert getattr(vtsc, '_vm', None) is not None, "VehicleModel required for steering-curvature fallback"

  # Compute the steering angle that corresponds to the target curvature at this speed.
  sa_rad = float(vtsc._vm.get_steer_from_curvature(k_curve, v0, 0.0))
  sa_deg = float(math.degrees(sa_rad))

  n_pre = 20   # 1.0 s of model-flat severe confidence (still straight)
  n_turn = 12  # 0.6 s of sharp turn, model remains flat but steering indicates curvature
  steps = [Step(curvature=0.0, curvature_ahead=0.0, confidence=conf, steering_angle_deg=0.0) for _ in range(n_pre)]
  steps += [Step(curvature=0.0, curvature_ahead=0.0, confidence=conf, steering_angle_deg=sa_deg) for _ in range(n_turn)]

  trace = simulate_sequence_trace(
    steps=steps,
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
    integrate_ego=False,  # keep v_ego constant for deterministic curvature mapping
  )
  assert trace

  # Before the turn, the model is flat so VTSC should remain near cruise.
  assert trace[n_pre - 1]['v_turn'] >= v_cruise - 0.2

  # As soon as steering indicates a sharp curve, VTSC must drop its cap rapidly.
  assert trace[n_pre]['v_turn'] < v_cruise - 1.0

  # Within ~0.5s, cap should be near the physics safe speed for the curve.
  v_safe = float(curvature_to_speed(k_curve))
  idx_check = n_pre + int(0.5 / dt) - 1
  assert trace[idx_check]['v_turn'] <= v_safe + 2.5


def test_steering_fallback_ignored_below_min_speed():
  # Guard: steering fallback should not engage at low speeds.
  v0 = 10.0
  v_cruise = 12.0
  dt = 0.05
  conf = 0.95
  k_curve = 0.02

  vtsc = mk_vtsc_with_params()
  assert getattr(vtsc, '_vm', None) is not None, "VehicleModel required for steering-curvature fallback"

  sa_rad = float(vtsc._vm.get_steer_from_curvature(k_curve, v0, 0.0))
  sa_deg = float(math.degrees(sa_rad))

  steps = [Step(curvature=0.0, curvature_ahead=0.0, confidence=conf, steering_angle_deg=sa_deg) for _ in range(10)]
  trace = simulate_sequence_trace(
    steps=steps,
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
    integrate_ego=False,
  )
  assert trace
  assert trace[0]['v_turn'] >= v_cruise - 0.2


def test_confirmed_lane_change_suppresses_model_only_curve_cap():
  # Offline RCA reproducer from 2026-03-13 route 000000af--0e810c91d5--31:
  # laneChangeStarting with tiny actual curvature and small steering, but the model horizon still
  # arced enough to drag VTSC far below cruise. During a confirmed lane change, suppress that
  # model-only cap when steering does not corroborate a real road curve.
  v0 = 22.5
  v_cruise = 24.7
  trace = simulate_sequence_trace(
    steps=[Step(
      curvature=0.0,
      curvature_ahead=0.02,
      confidence=0.95,
      steering_angle_deg=0.0,
      lane_change_state=int(log.LaneChangeState.laneChangeStarting),
      lane_change_direction=int(log.LaneChangeDirection.left),
      left_blinker=True,
    ) for _ in range(10)],
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
    integrate_ego=False,
  )
  assert trace
  assert min(float(row['v_turn']) for row in trace) >= v_cruise - 0.2


def test_real_curve_still_slows_during_confirmed_lane_change():
  # Safety guard: if the car is genuinely turning while lane-change state is active, VTSC should
  # still respect the real curve via steering corroboration.
  v0 = 22.5
  v_cruise = 24.7
  k_curve = 0.02
  vtsc = mk_vtsc_with_params()
  assert getattr(vtsc, '_vm', None) is not None, "VehicleModel required for lane-change curve guard"
  sa_rad = float(vtsc._vm.get_steer_from_curvature(k_curve, v0, 0.0))
  sa_deg = float(math.degrees(sa_rad))

  trace = simulate_sequence_trace(
    steps=[Step(
      curvature=0.0,
      curvature_ahead=0.02,
      confidence=0.95,
      steering_angle_deg=sa_deg,
      lane_change_state=int(log.LaneChangeState.laneChangeStarting),
      lane_change_direction=int(log.LaneChangeDirection.left),
      left_blinker=True,
    ) for _ in range(10)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
    integrate_ego=False,
  )
  assert trace
  assert min(float(row['v_turn']) for row in trace) < v_cruise - 1.0


def test_severe_confidence_overshoot_is_mildly_conservative_for_blind_curves():
  # Regression-style guard:
  #
  # On some blind off-ramps, lane-line confidence can be extremely low for several seconds
  # before the model curvature spikes. Without any conservatism, overshoot detection may only
  # trigger at the last moment.
  #
  # Ensure that in SEVERE/LOST confidence (and without a lead-bypass), the overshoot detector is
  # slightly conservative, producing an early (small) cap reduction for moderate curvature.
  v0 = 26.8  # ~60 mph
  v_cruise = 26.8
  dt = 0.05
  conf = 0.05
  # Choose curvature just past the point where the severe-confidence overshoot scaling flips a
  # "no cap" case into a small cap reduction. This keeps the test tied to the behavior it intends
  # to guard even if the global curvature->speed fit changes slightly.
  target_scaled_speed = v0 - 0.15
  lo, hi = 1e-4, 0.02
  for _ in range(80):
    mid = 0.5 * (lo + hi)
    if float(SEVERE_OVERSHOOT_SPEED_SCALE_MIN) * float(curvature_to_speed(mid)) <= target_scaled_speed:
      hi = mid
    else:
      lo = mid
  k = hi
  assert float(curvature_to_speed(k)) > v0
  assert float(SEVERE_OVERSHOOT_SPEED_SCALE_MIN) * float(curvature_to_speed(k)) < v0

  trace = simulate_sequence_trace(
    steps=[Step(curvature=k, curvature_ahead=k, confidence=conf) for _ in range(5)],
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
    integrate_ego=False,
  )
  assert trace
  assert float(trace[0]['v_turn']) <= v0 - 0.10


def test_tightening_visible_curvature_can_use_full_decel_budget_without_occlusion_clamp():
  # With occlusion removed, tightening visible curvature is allowed to use the controller's full
  # decel budget instead of being pinned to the old comfort-decel occlusion clamp.
  v0 = 22.0
  v_cruise = 24.0
  ks = [0.002 + i * (0.006 - 0.002) / 20 for i in range(20)]
  steps = [Step(curvature=k, confidence=0.4) for k in ks]
  steps += [Step(curvature=0.006, confidence=0.4) for _ in range(20)]
  snap = simulate_sequence(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  assert snap
  assert float(snap['decel_cmd']) <= 0.0
  assert float(snap['decel_cmd']) < float(snap['comfort_decel']) - 1e-6
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(0.4, abs=1e-6)
  assert -7.0 <= float(snap['jerk_cmd']) <= 3.0


def _mk_sm_for_map_latency(curvature: float, curvature_ahead: float, v_pred: float, confidence: float):
  """Create a minimal modelV2-only SM for map/arbitration timing tests."""
  v_pred = float(max(0.0, v_pred))
  k_now = float(curvature)
  k_ahead = float(curvature_ahead)
  k_points = [k_now] + [k_ahead] * 32
  yaw_rate_points = [k * v_pred for k in k_points]
  model = SimpleNamespace(
    orientationRate=SimpleNamespace(z=yaw_rate_points),
    velocity=SimpleNamespace(x=[v_pred] * 33),
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


def _build_map_polyline(
  lat0: float, lon0: float,
  hairpin_start_m: float = 500.0, hairpin_end_m: float = 580.0,
  k_hairpin: float = 0.025, total_m: float = 1500.0, step_m: float = 10.0,
  curvature_offset_m: float = 0.0,
):
  pts = []
  n = int(total_m // step_m)
  for i in range(n + 1):
    dist_m = i * step_m + float(curvature_offset_m)
    k = float(k_hairpin) if (hairpin_start_m <= dist_m <= hairpin_end_m) else 0.0
    pts.append((lat0 + (i * step_m) / 111000.0, lon0, k))
  return pts


def _build_map_profile_polyline(
  lat0: float, lon0: float,
  curvature_profile: list[float],
  *,
  profile_start_m: float = 0.0,
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


def _run_map_latency_trace(
  gps_delay_s: float = 0.0,
  map_hold_s: float = 0.0,
  map_stale_offset_m: float = 0.0,
  vision_lookahead_m: float = 140.0,
  v_ego_mps: float = 29.0,
  total_s: float = 24.0,
  dt: float = 0.05,
):
  """Run a deterministic straight→hairpin approach with map lookahead enabled.

  The scenario keeps ego speed fixed to isolate arbitration behavior:
  - Map can constrain early from far-horizon geometry.
  - Vision begins "seeing" the curve at `vision_lookahead_m`.
  - Optional GPS lag and stale map window inject latency faults.
  """
  lat0, lon0 = 37.0, -122.0
  hairpin_start_m = 500.0
  k_hairpin = 0.025

  vtsc = mk_vtsc_with_params()
  # Use comfort-like decel for lookahead reachability in this synthetic case.
  vtsc._max_decel = 1.47
  orig_get_bool = vtsc._get_bool_param
  def _get_bool(key: str, default: bool = False) -> bool:
    if key == 'MTSCLookaheadEnabled':
      return True
    return bool(orig_get_bool(key, default))
  vtsc._get_bool_param = _get_bool

  map_base = _build_map_polyline(lat0, lon0, k_hairpin=k_hairpin)
  map_stale = _build_map_polyline(lat0, lon0, k_hairpin=k_hairpin, curvature_offset_m=map_stale_offset_m)

  state = {'t': 0.0, 'd': 0.0}

  def _gps():
    d = max(0.0, state['d'] - float(gps_delay_s) * float(v_ego_mps))
    return (lat0 + d / 111000.0, lon0, None)

  def _map_pts():
    return map_stale if state['t'] < float(map_hold_s) else map_base

  vtsc._get_last_gps_pose = _gps
  vtsc._load_map_curvatures = _map_pts

  trace = []
  n_steps = int(total_s / dt)
  for _ in range(n_steps):
    dist_to_curve = float(hairpin_start_m - state['d'])
    if dist_to_curve > float(vision_lookahead_m):
      k_now = 0.0
      k_ahead = 0.0
    elif dist_to_curve > 0.0:
      k_now = 0.0
      k_ahead = k_hairpin
    else:
      k_now = k_hairpin
      k_ahead = k_hairpin

    sm = _mk_sm_for_map_latency(k_now, k_ahead, float(v_ego_mps), 0.95)
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: state['t']), \
         patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: state['t']):
      vtsc.update(sm, True, float(v_ego_mps), 0.0, float(v_ego_mps))

    snap = vtsc.snapshot_debug_state() or {}
    trace.append({
      't': float(state['t']),
      'dist_to_curve_m': dist_to_curve,
      'active_cap': str(snap.get('active_cap', '') or ''),
      'map_tail_active': bool(snap.get('map_tail_active', False)),
      'map_tail_cap': float(snap.get('map_tail_cap', 0.0) or 0.0),
      'cap_visible_vmin': float(snap.get('cap_visible_vmin', 0.0) or 0.0),
      'cap_map_vmin': float(snap.get('cap_map_vmin', 0.0) or 0.0),
      'a_target': float(getattr(vtsc, 'a_target', 0.0)),
      'v_turn': float(getattr(vtsc, 'v_turn', 0.0)),
    })

    state['t'] += float(dt)
    state['d'] += float(v_ego_mps) * float(dt)

  return trace


def _count_active_cap_transitions(trace):
  labels = [str(r.get('active_cap', '') or '') for r in trace]
  return sum(1 for a, b in zip(labels, labels[1:]) if a != b)


def _count_accel_sign_flips(trace, eps: float = 0.05):
  signs = []
  for r in trace:
    a = float(r.get('a_target', 0.0))
    if a > eps:
      signs.append(1)
    elif a < -eps:
      signs.append(-1)
  return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def test_map_lookahead_gps_delay_does_not_cause_cap_flapping():
  base = _run_map_latency_trace(gps_delay_s=0.0)
  delayed = _run_map_latency_trace(gps_delay_s=1.0)

  # Map should engage in both runs before vision sees the hairpin.
  base_first_map = next((r['dist_to_curve_m'] for r in base if r['active_cap'] == 'map'), None)
  delayed_first_map = next((r['dist_to_curve_m'] for r in delayed if r['active_cap'] == 'map'), None)
  assert base_first_map is not None
  assert delayed_first_map is not None
  # 1s GPS lag should not cause map engagement to happen materially *earlier*.
  # Discrete 10m map sampling can quantize both cases to the same first engagement point.
  assert delayed_first_map <= base_first_map + 1e-6

  # In the pre-entry approach window, cap arbitration should remain stable.
  for tr in (base, delayed):
    win = [r for r in tr if 0.0 <= r['dist_to_curve_m'] <= 260.0]
    assert _count_active_cap_transitions(win) <= 3
    assert _count_accel_sign_flips(win) <= 1


def test_map_to_vision_handoff_prefers_visible_cap():
  trace = _run_map_latency_trace(gps_delay_s=0.6)
  onset_idx = next((i for i, r in enumerate(trace) if r['dist_to_curve_m'] <= 0.0), None)
  assert onset_idx is not None

  post = trace[onset_idx:onset_idx + 40]  # ~2.0 s after entering curve zone
  first_visible = next((i for i, r in enumerate(post) if r['active_cap'] == 'visible'), None)
  assert first_visible is not None, "Visible cap should take over after curve entry"
  assert first_visible <= 15  # <= ~0.75 s at 20 Hz
  # After visible takes over, map should not steal cap authority.
  tail = post[first_visible:]
  map_frames = sum(1 for r in tail if r['active_cap'] == 'map')
  assert map_frames == 0


def test_stale_map_recovery_avoids_brake_accel_oscillation():
  # Hold stale map geometry for 3s, then recover to fresh geometry.
  trace = _run_map_latency_trace(gps_delay_s=0.8, map_hold_s=3.0, map_stale_offset_m=150.0)
  release_idx = next((i for i, r in enumerate(trace) if r['t'] >= 3.0), None)
  assert release_idx is not None

  pre = trace[max(0, release_idx - 20):release_idx]       # ~1.0 s before release
  post = trace[release_idx:release_idx + 200]             # allow longer handoff under delayed/stale map

  # Stale map should be active before release and hand off cleanly after release.
  assert any(r['active_cap'] == 'map' for r in pre)
  assert any(r['active_cap'] == 'visible' for r in post)
  assert _count_active_cap_transitions(pre + post) <= 4
  assert _count_accel_sign_flips(pre + post) <= 1


def _enable_map_lookahead(vtsc, monkeypatch):
  orig_get_bool = vtsc._get_bool_param

  def _get_bool(key: str, default: bool = False) -> bool:
    if key == 'MTSCLookaheadEnabled':
      return True
    return bool(orig_get_bool(key, default))

  monkeypatch.setattr(vtsc, "_get_bool_param", _get_bool, raising=True)


def _set_map_strategy(vtsc, monkeypatch, mode: str):
  orig_get_string = vtsc._get_string_param

  def _get_string(key: str, default: str = "") -> str:
    if key == 'VTSCMapStrategy':
      return mode
    return str(orig_get_string(key, default))

  vtsc._map_strategy_mode = mode
  monkeypatch.setattr(vtsc, "_get_string_param", _get_string, raising=True)


def _curve_phase_raw_for_effective(effective_s: float) -> float:
  return float(effective_s) - float(map_strategy.CURVE_PHASE_OFFSET_ZERO_BASELINE_S)


def _set_longitudinal_response_model(vtsc, *, min_accel: float = -6.0, max_accel: float = 5.0, delay_s: float = 0.0):
  vtsc.set_longitudinal_response_model(build_cruise_response_model(
    min_accel_mps2=min_accel,
    max_accel_mps2=max_accel,
    actuation_delay_s=delay_s,
  ))


def _patch_map_tail_inputs(vtsc, monkeypatch, lat0: float, lon0: float, pts):
  monkeypatch.setattr(vtsc, "_get_last_gps_pose", lambda: (lat0, lon0, None), raising=True)
  monkeypatch.setattr(vtsc, "_load_map_curvatures", lambda: pts, raising=True)


def _run_strategic_map_snapshot(
  monkeypatch,
  *,
  fixed_lead_time_s: float = 0.0,
  curve_phase_s: float = 0.0,
  overshoot_phase_s: float = 0.0,
  apex_exit_phase_s: float = 0.0,
  v0: float = 24.0,
  v_cruise: float = 27.0,
  current_curve: float = 0.0,
  curvature_ahead: float = 0.0,
  confidence: float = 0.95,
  map_curve_start_m: float = 50.0,
  map_curve_end_m: float = 70.0,
  map_curve_k: float = 0.02,
  planner_delay_s: float = 0.35,
):
  lat0, lon0 = 37.0, -122.0
  step_deg = 10.0 / 111000.0

  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, 'strategic')
  _set_longitudinal_response_model(vtsc, min_accel=-6.0, delay_s=planner_delay_s)
  vtsc._fixed_lead_time_s = float(fixed_lead_time_s)
  vtsc._curve_phase_offset_s = float(_curve_phase_raw_for_effective(curve_phase_s))
  vtsc._overshoot_phase_offset_s = float(overshoot_phase_s)
  vtsc._apex_exit_phase_offset_s = float(apex_exit_phase_s)

  pts = []
  for i in range(80):
    dist_m = float(i * 10.0)
    k = map_curve_k if (map_curve_start_m <= dist_m <= map_curve_end_m) else 0.0
    pts.append((lat0 + i * step_deg, lon0, k))
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

  snap = simulate_sequence(
    steps=[Step(curvature=current_curve, curvature_ahead=curvature_ahead, confidence=confidence) for _ in range(10)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert snap['strategy_mode'] == 'strategic'
  return snap


def _run_profile_map_snapshot(
  monkeypatch,
  *,
  mode: str,
  map_profile: list[float],
  profile_start_m: float = 0.0,
  v0: float = 22.0,
  v_cruise: float = 27.0,
  current_curve: float = 0.0015,
  curvature_ahead: float = 0.0025,
  confidence: float = 0.95,
  planner_delay_s: float = 0.35,
):
  lat0, lon0 = 37.0, -122.0
  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, mode)
  _set_longitudinal_response_model(vtsc, min_accel=-6.0, delay_s=planner_delay_s)

  pts = _build_map_profile_polyline(lat0, lon0, map_profile, profile_start_m=profile_start_m)
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

  snap = simulate_sequence(
    steps=[Step(curvature=current_curve, curvature_ahead=curvature_ahead, confidence=confidence) for _ in range(10)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert snap['strategy_mode'] == mode
  return snap


def test_map_lookahead_cap_applies_when_available(monkeypatch):
  # Patch controller to provide GPS + synthetic map tail; assert map tail active and cap < cruise
  v0 = 25.0
  v_cruise = 30.0
  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, 'advisory')
  # Enable map lookahead via get_bool and patch data providers
  lat0, lon0 = 37.0, -122.0
  def _map_pts():
    # Construct ~10 points with increasing curvature ahead (~0.0 near, then 0.008 farther)
    base_lat, base_lon = lat0, lon0
    pts = []
    for i in range(10):
      lat = base_lat + 0.0001 * i
      lon = base_lon
      k = 0.0 if i < 3 else 0.008
      pts.append((lat, lon, k))
    return pts
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, _map_pts())
  # Run straight/clear steps; map tail should cap
  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.0, confidence=0.95, n=80),
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert bool(snap['map_tail_active']) is True
  # Cap should be below cruise when curvature ahead is present
  assert float(snap['map_tail_cap']) <= v_cruise + 1e-6
  assert snap['strategy_mode'] == 'advisory'
  assert float(snap['map_advisory_cap']) <= v_cruise + 1e-6
  assert float(snap['map_strategic_cap']) <= float(snap['map_advisory_cap']) + 1e-6


def test_offramp_short_tight_curve_map_cap_applies_when_vision_lost(monkeypatch):
  # Regression reproducer (real-world style):
  #
  # A freeway off-ramp can contain a short, tight advisory-speed curve very shortly after leaving
  # the freeway. If model confidence drops hard (SEVERE/LOST) and the model's curvature estimate
  # is unreliable (kappa ~ 0), we still want map lookahead to prevent VTSC from effectively
  # "failing open" at ~60 mph.
  #
  # This uses the user-reported location as a stable GPS seed for the synthetic map polyline:
  #   38°43'54.0"N 120°47'20.2"W
  v0 = 26.8  # ~60 mph
  v_cruise = 26.8
  dt = 0.05

  # GPS seed from report (degrees)
  lat0 = 38.0 + 43.0 / 60.0 + 54.0 / 3600.0
  lon0 = -(120.0 + 47.0 / 60.0 + 20.2 / 3600.0)

  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)

  # Synthetic map curvature polyline:
  # - Straight segment
  # - Short tight curve from 20m..40m (advisory ~20 mph)
  # - Straight again (so the curve is *inside* the typical "visible horizon" distance)
  step_deg = 10.0 / 111000.0  # ~10m north per sample
  k_curve = 0.02
  pts = []
  for i in range(80):
    dist_m = float(i * 10.0)
    k = k_curve if (20.0 <= dist_m <= 40.0) else 0.0
    pts.append((lat0 + i * step_deg, lon0, k))
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

  # Simulate a brief "vision lost" window with near-zero model curvature (can't see the ramp curve).
  trace = simulate_sequence_trace(
    steps=[Step(curvature=0.0, curvature_ahead=0.0, confidence=0.05) for _ in range(5)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
    integrate_ego=False,
  )
  assert trace

  # With severe vision confidence, the cap should immediately reflect map curvature inside ~20m,
  # not stay at cruise.
  v_safe = float(curvature_to_speed(k_curve))
  a_comf = float(getattr(vtsc, '_max_decel', 3.5))
  v_expected_10m = math.sqrt(max(0.0, v_safe * v_safe + 2.0 * a_comf * 10.0))
  assert trace[0]['v_turn'] < v_cruise - 1e-3
  # Depending on local map sample spacing, first-step effective distance can quantize shorter than 10m.
  assert trace[0]['v_turn'] >= v_safe - 0.5
  assert trace[0]['v_turn'] <= v_expected_10m + 0.75


def test_low_confidence_does_not_let_advisory_map_beat_visible_cap_for_hidden_curve(monkeypatch):
  v0 = 18.0
  v_cruise = 22.0
  dt = 0.05
  lat0, lon0 = 37.0, -122.0
  step_deg = 10.0 / 111000.0
  k_curve = 0.02

  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, 'advisory')

  pts = []
  for i in range(80):
    dist_m = float(i * 10.0)
    k = k_curve if (20.0 <= dist_m <= 30.0) else 0.0
    pts.append((lat0 + i * step_deg, lon0, k))
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

  trace = simulate_sequence_trace(
    steps=[Step(curvature=0.0, curvature_ahead=0.0, confidence=0.95) for _ in range(10)] +
          [Step(curvature=0.0, curvature_ahead=0.0, confidence=0.60) for _ in range(8)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
    integrate_ego=False,
  )
  assert trace

  snap = vtsc.snapshot_debug_state()
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(0.60, abs=1e-6)
  assert bool(snap['map_tail_active']) is True
  assert snap['map_tail_reason'] == 'applied'
  assert float(snap['map_tail_start_m']) >= (v0 * float(snap['vis_horizon_s']) + 10.0) - 1e-3
  assert snap['active_cap'] == 'visible'
  assert float(snap['map_tail_cap']) >= v_cruise - 1e-3


def test_low_confidence_still_allows_tighter_advisory_map_cap_when_visible_curve_is_looser(monkeypatch):
  v0 = 18.0
  v_cruise = 22.0
  dt = 0.05
  lat0, lon0 = 37.0, -122.0
  step_deg = 10.0 / 111000.0
  k_curve = 0.02

  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, 'advisory')

  pts = []
  for i in range(80):
    dist_m = float(i * 10.0)
    k = k_curve if (20.0 <= dist_m <= 40.0) else 0.0
    pts.append((lat0 + i * step_deg, lon0, k))
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

  trace = simulate_sequence_trace(
    steps=[Step(curvature=0.0, curvature_ahead=0.0, confidence=0.95) for _ in range(10)] +
          [Step(curvature=0.0, curvature_ahead=0.008, confidence=0.60) for _ in range(8)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
    integrate_ego=False,
  )
  assert trace

  snap = vtsc.snapshot_debug_state()
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(0.60, abs=1e-6)
  assert bool(snap['map_tail_active']) is True
  assert snap['map_tail_reason'] == 'applied'
  assert float(snap['cap_map_vmin']) > 0.0
  assert float(snap['cap_map_vmin']) < float(snap['cap_visible_vmin']) - 1e-3
  assert snap['active_cap'] == 'map'
  assert float(snap['vtsc_cmd']) < float(snap['cap_visible_vmin']) - 1e-3


def test_strategic_mode_applies_map_floor_inside_visible_horizon(monkeypatch):
  v0 = 18.0
  v_cruise = 22.0
  dt = 0.05
  lat0, lon0 = 37.0, -122.0
  step_deg = 10.0 / 111000.0
  k_curve = 0.02

  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, 'strategic')
  _set_longitudinal_response_model(vtsc, min_accel=-6.0, delay_s=0.35)

  pts = []
  for i in range(80):
    dist_m = float(i * 10.0)
    k = k_curve if (20.0 <= dist_m <= 30.0) else 0.0
    pts.append((lat0 + i * step_deg, lon0, k))
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

  snap = simulate_sequence(
    steps=[Step(curvature=0.0, curvature_ahead=0.0, confidence=0.95) for _ in range(12)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
  )
  assert snap
  assert snap['strategy_mode'] == 'strategic'
  assert bool(snap['map_tail_active']) is True
  assert bool(snap['map_floor_active']) is True
  assert float(snap['map_tail_start_m']) <= 1e-6
  assert float(snap['map_strategic_cap']) < v_cruise - 1e-3
  assert float(snap['map_advisory_cap']) >= v_cruise - 1e-3
  assert snap['active_cap'] == 'map'
  assert float(snap['vtsc_cmd']) < v_cruise - 1e-3
  assert float(snap['planner_min_accel_mps2']) == pytest.approx(-6.0, abs=1e-6)
  assert float(snap['planner_response_delay_s']) == pytest.approx(0.35, abs=1e-6)


def test_strategic_mode_releases_map_floor_on_counterevidence_dwell(monkeypatch):
  v0 = 18.0
  v_cruise = 22.0
  dt = 0.05
  lat0, lon0 = 37.0, -122.0
  step_deg = 10.0 / 111000.0
  k_curve = 0.02

  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, 'strategic')
  _set_longitudinal_response_model(vtsc, min_accel=-6.0, delay_s=0.35)

  pts = []
  for i in range(80):
    dist_m = float(i * 10.0)
    k = k_curve if (20.0 <= dist_m <= 40.0) else 0.0
    pts.append((lat0 + i * step_deg, lon0, k))
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

  snap = simulate_sequence(
    steps=[Step(curvature=0.0, curvature_ahead=0.008, confidence=0.95) for _ in range(22)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
  )
  assert snap
  assert snap['strategy_mode'] == 'strategic'
  assert bool(snap['vision_relax_allowed']) is True
  assert snap['vision_relax_reason'] in ('counterevidence_dwell', 'post_apex_release')
  assert bool(snap['map_floor_active']) is False
  assert snap['strategy_state'] == 'vision_owns'
  assert snap['active_cap'] == 'visible'
  assert float(snap['vision_local_cap']) > float(snap['map_strategic_cap']) + 0.5


def test_strategic_mode_uses_planner_response_model_not_vtsc_max_decel(monkeypatch):
  v0 = 24.0
  v_cruise = 27.0
  dt = 0.05
  lat0, lon0 = 37.0, -122.0
  step_deg = 10.0 / 111000.0
  k_curve = 0.02

  def run_case(vtsc_max_decel: float):
    vtsc = mk_vtsc_with_params()
    _enable_map_lookahead(vtsc, monkeypatch)
    _set_map_strategy(vtsc, monkeypatch, 'strategic')
    _set_longitudinal_response_model(vtsc, min_accel=-6.0, delay_s=0.40)
    vtsc._max_decel = float(vtsc_max_decel)

    pts = []
    for i in range(80):
      dist_m = float(i * 10.0)
      k = k_curve if (50.0 <= dist_m <= 70.0) else 0.0
      pts.append((lat0 + i * step_deg, lon0, k))
    _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

    snap = simulate_sequence(
      steps=[Step(curvature=0.0, curvature_ahead=0.0, confidence=0.95) for _ in range(8)],
      vtsc=vtsc,
      v0_mps=v0,
      v_cruise_mps=v_cruise,
      dt=dt,
    )
    return snap

  snap_low = run_case(1.0)
  snap_high = run_case(7.0)
  assert float(snap_low['map_strategic_cap']) == pytest.approx(float(snap_high['map_strategic_cap']), abs=1e-6)
  assert float(snap_low['planner_min_accel_mps2']) == pytest.approx(-6.0, abs=1e-6)
  assert float(snap_high['planner_min_accel_mps2']) == pytest.approx(-6.0, abs=1e-6)


def test_strategic_mode_tightens_when_planner_delay_increases(monkeypatch):
  v0 = 24.0
  v_cruise = 27.0
  dt = 0.05
  lat0, lon0 = 37.0, -122.0
  step_deg = 10.0 / 111000.0
  k_curve = 0.02

  def run_case(delay_s: float):
    vtsc = mk_vtsc_with_params()
    _enable_map_lookahead(vtsc, monkeypatch)
    _set_map_strategy(vtsc, monkeypatch, 'strategic')
    _set_longitudinal_response_model(vtsc, min_accel=-6.0, delay_s=delay_s)
    vtsc._curve_phase_offset_s = float(_curve_phase_raw_for_effective(0.0))

    pts = []
    for i in range(80):
      dist_m = float(i * 10.0)
      k = k_curve if (50.0 <= dist_m <= 70.0) else 0.0
      pts.append((lat0 + i * step_deg, lon0, k))
    _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

    snap = simulate_sequence(
      steps=[Step(curvature=0.0, curvature_ahead=0.0, confidence=0.95) for _ in range(8)],
      vtsc=vtsc,
      v0_mps=v0,
      v_cruise_mps=v_cruise,
      dt=dt,
    )
    return snap

  snap_fast = run_case(0.10)
  snap_slow = run_case(1.00)
  assert float(snap_slow['map_strategic_cap']) < float(snap_fast['map_strategic_cap']) - 1e-3
  assert float(snap_slow['planner_response_delay_s']) == pytest.approx(1.0, abs=1e-6)


def test_strategic_mode_curve_phase_offset_shifts_map_floor_timing(monkeypatch):
  snap_early = _run_strategic_map_snapshot(monkeypatch, curve_phase_s=-2.0, overshoot_phase_s=0.0)
  snap_late = _run_strategic_map_snapshot(monkeypatch, curve_phase_s=2.0, overshoot_phase_s=0.0)

  assert float(snap_early['map_strategic_cap']) < float(snap_late['map_strategic_cap']) - 0.5
  assert float(snap_early['vtsc_cmd']) < float(snap_late['vtsc_cmd']) - 0.5


def test_strategic_mode_fixed_lead_time_tightens_map_floor(monkeypatch):
  snap_zero = _run_strategic_map_snapshot(monkeypatch, fixed_lead_time_s=0.0, curve_phase_s=0.0, overshoot_phase_s=0.0)
  snap_one = _run_strategic_map_snapshot(monkeypatch, fixed_lead_time_s=1.0, curve_phase_s=0.0, overshoot_phase_s=0.0)
  snap_two = _run_strategic_map_snapshot(monkeypatch, fixed_lead_time_s=2.0, curve_phase_s=0.0, overshoot_phase_s=0.0)

  assert float(snap_one['map_strategic_cap']) < float(snap_zero['map_strategic_cap']) - 0.5
  # Once the requested lead time is large enough that the strategic solver must ask for the
  # anchor speed immediately, additional lead time should saturate at that target rather than
  # manufacture an artificial distinction below it.
  assert float(snap_two['map_strategic_cap']) <= float(snap_one['map_strategic_cap']) + 1e-6
  assert float(snap_two['map_strategic_cap']) <= float(curvature_to_speed(0.02)) + 0.25


def test_strategic_mode_overshoot_phase_offset_shifts_tighter_map_floor_timing(monkeypatch):
  snap_early = _run_strategic_map_snapshot(monkeypatch, curve_phase_s=0.0, overshoot_phase_s=-2.0)
  snap_late = _run_strategic_map_snapshot(monkeypatch, curve_phase_s=0.0, overshoot_phase_s=2.0)

  assert float(snap_early['map_strategic_cap']) < float(snap_late['map_strategic_cap']) - 0.5
  assert float(snap_early['vtsc_cmd']) < float(snap_late['vtsc_cmd']) - 0.5


def test_strategic_mode_uses_hidden_apex_profile_for_blind_rising_curve(monkeypatch):
  # Synthetic blind-curve profile:
  # - shallow entry geometry
  # - much tighter hidden apex at ~30 m
  # - unwind after the apex
  # Vision is still effectively blind in this frame, so the map profile is the only
  # source that can expose the hidden entry->apex delta. Advisory mode only plans from
  # beyond the visible-handoff window, so it largely sees the unwind. Strategic mode
  # should still hold the tighter hidden-apex floor.
  map_profile = [
    0.0,    # ego point
    0.0015, # 10 m: shallow entry
    0.0030, # 20 m: shallow entry
    0.0200, # 30 m: hidden apex
    0.0080, # 40 m: unwind
    0.0030, # 50 m: unwind
    0.0,
  ]

  advisory_snap = _run_profile_map_snapshot(
    monkeypatch,
    mode='advisory',
    map_profile=map_profile,
    v0=22.0,
    v_cruise=27.0,
    current_curve=0.0,
    curvature_ahead=0.0,
  )
  strategic_snap = _run_profile_map_snapshot(
    monkeypatch,
    mode='strategic',
    map_profile=map_profile,
    v0=22.0,
    v_cruise=27.0,
    current_curve=0.0,
    curvature_ahead=0.0,
  )

  # The synthetic map profile should actually create a materially tighter hidden-apex floor.
  assert float(advisory_snap['map_strategic_cap']) < float(advisory_snap['map_advisory_cap']) - 1.0
  # Strategic mode should select that tighter floor, while advisory stays materially looser.
  assert float(strategic_snap['vtsc_cmd']) < float(advisory_snap['vtsc_cmd']) - 1.0
  assert bool(strategic_snap['map_floor_active']) is True
  assert float(strategic_snap['map_floor_anchor_dist_m']) <= 35.0
  assert float(strategic_snap['map_floor_anchor_k']) >= 0.018


def test_strategic_mode_lane_change_map_ambiguity_relaxes_to_advisory(monkeypatch):
  v0 = 15.0
  v_cruise = 22.0
  dt = 0.05
  lat0, lon0 = 37.0, -122.0
  step_deg = 10.0 / 111000.0
  k_curve = 0.02

  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, 'strategic')
  _set_longitudinal_response_model(vtsc, min_accel=-6.0, delay_s=0.35)

  pts = []
  for i in range(80):
    dist_m = float(i * 10.0)
    k = k_curve if (60.0 <= dist_m <= 80.0) else 0.0
    pts.append((lat0 + i * step_deg, lon0, k))
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

  snap = simulate_sequence(
    steps=[Step(
      curvature=0.0,
      curvature_ahead=0.0,
      confidence=0.95,
      live_map_data={'roadGeometryValid': False, 'windingRoadValid': False},
      lane_change_state=int(log.LaneChangeState.laneChangeStarting),
      lane_change_direction=int(log.LaneChangeDirection.left),
      left_blinker=True,
    ) for _ in range(12)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
  )
  assert snap
  assert float(snap['map_strategic_cap']) == pytest.approx(0.0, abs=1e-6)
  assert float(snap['map_advisory_cap']) == pytest.approx(0.0, abs=1e-6)
  assert str(snap['map_tail_compute_reason']) == 'road_geometry_invalid'
  assert float(snap['vtsc_cmd']) >= v_cruise - 1e-6


def test_lane_change_map_ambiguity_guard_requires_large_raw_map_mismatch():
  vtsc = mk_vtsc_with_params()
  vtsc._road_geometry_valid = False
  vtsc._mapd_winding_valid = False
  vtsc._lane_change_active = True
  vtsc._single_blinker_active = True

  vtsc._dbg_k_model = 0.008
  vtsc._dbg_k_steer = 0.0
  vtsc._filtered_curvature = 0.008

  mild_candidate = SimpleNamespace(cap_mps=16.0, anchor_dist_m=90.0, anchor_curvature=0.020)
  tight_candidate = SimpleNamespace(cap_mps=10.0, anchor_dist_m=90.0, anchor_curvature=0.040)

  assert vtsc._should_relax_strategic_map_candidate(mild_candidate) is False
  assert vtsc._should_relax_strategic_map_candidate(tight_candidate) is True


def test_visible_mainline_counterevidence_guard_detects_map_overestimate():
  vtsc = mk_vtsc_with_params()
  vtsc._road_geometry_valid = True
  vtsc._mapd_winding_valid = False
  vtsc._winding_context_active = False
  vtsc._lane_change_active = False
  vtsc._single_blinker_active = False
  vtsc._curve_preview_valid = True
  vtsc._curve_preview_distance_m = 0.0
  vtsc._curve_preview_branch_stubs = []
  vtsc._v_ego = 33.5
  vtsc._dbg_k_model = 0.0012
  vtsc._dbg_k_steer = 0.0
  vtsc._filtered_curvature = 0.0012
  vtsc._current_lat_acc = 0.31
  vtsc._max_pred_lat_acc = 1.31
  vtsc._occlusion_state.vision_status = VisionStatus.FULL_VISIBILITY

  candidate = SimpleNamespace(cap_mps=27.25, anchor_dist_m=69.2, anchor_curvature=0.0042)

  assert vtsc._should_suppress_map_candidate_for_visible_mainline_counterevidence(candidate) is True


def test_strategic_mode_visible_mainline_counterevidence_suppresses_map_floor(monkeypatch):
  v0 = 33.5
  v_cruise = 65.0
  dt = 0.05
  lat0, lon0 = 37.0, -122.0
  step_deg = 10.0 / 111000.0
  map_curve_k = 0.0042

  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, 'strategic')
  _set_longitudinal_response_model(vtsc, min_accel=-6.0, delay_s=0.35)

  pts = []
  for i in range(80):
    dist_m = float(i * 10.0)
    k = map_curve_k if (0.0 <= dist_m <= 20.0) else 0.0
    pts.append((lat0 + i * step_deg, lon0, k))
  _patch_map_tail_inputs(vtsc, monkeypatch, lat0, lon0, pts)

  dwell_steps = int(math.ceil(VISIBLE_MAINLINE_RELAX_DWELL_S / dt)) + 3
  snap = simulate_sequence(
    steps=[Step(
      curvature=0.0,
      curvature_ahead=0.0012,
      confidence=0.95,
      live_map_data={'roadGeometryValid': True, 'windingRoadValid': False},
      applied_accel=0.0,
    ) for _ in range(dwell_steps)],
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=dt,
  )
  assert snap
  assert snap['strategy_mode'] == 'strategic'
  assert snap['map_tail_compute_reason'] == 'visible_mainline_counterevidence'
  assert bool(snap['map_floor_active']) is False
  assert snap['active_cap'] == 'visible'
  assert float(snap['map_floor_anchor_k']) >= 0.004
  assert float(snap['vision_local_cap']) > float(snap['map_strategic_cap']) + 5.0
  assert float(snap['vtsc_cmd']) > float(snap['map_strategic_cap']) + 5.0


def test_strategic_mode_overshoot_phase_offset_ignored_when_reference_speed_is_not_tighter():
  k_curve = 0.02
  vsafe = float(curvature_to_speed(k_curve))
  response_model = build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, actuation_delay_s=0.35)

  candidate_early = compute_map_cap_candidate(
    mode='strategic',
    s_list=[60.0],
    k_list=[k_curve],
    vsafe_list=[vsafe],
    abs_indices=[6],
    v_ego=24.0,
    v_cruise=27.0,
    vis_horizon_s=1.4,
    vis_margin_m=10.0,
    severe_vision=False,
    partial_vision=False,
    vision_confidence=0.95,
    conf_lo=0.55,
    conf_hi=0.85,
    max_decel=3.5,
    horizon_limit_m=250.0,
    response_model=response_model,
    curve_phase_offset_s=_curve_phase_raw_for_effective(0.0),
    overshoot_phase_offset_s=-2.0,
    reference_speed_mps=vsafe + 0.5,
  )
  candidate_late = compute_map_cap_candidate(
    mode='strategic',
    s_list=[60.0],
    k_list=[k_curve],
    vsafe_list=[vsafe],
    abs_indices=[6],
    v_ego=24.0,
    v_cruise=27.0,
    vis_horizon_s=1.4,
    vis_margin_m=10.0,
    severe_vision=False,
    partial_vision=False,
    vision_confidence=0.95,
    conf_lo=0.55,
    conf_hi=0.85,
    max_decel=3.5,
    horizon_limit_m=250.0,
    response_model=response_model,
    curve_phase_offset_s=_curve_phase_raw_for_effective(0.0),
    overshoot_phase_offset_s=2.0,
    reference_speed_mps=vsafe + 0.5,
  )

  assert float(candidate_early.cap_mps) == pytest.approx(float(candidate_late.cap_mps), abs=1e-6)


def test_strategic_response_probe_runs_once_for_controlling_constraint(monkeypatch):
  response_model = build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, actuation_delay_s=0.35)
  probe_calls = []
  threshold_calls = []

  def fake_probe(*, v_ego, required_decel_mps2, response_model, v_cruise_upper, **kwargs):
    probe_calls.append(float(required_decel_mps2))
    return 16.5

  def fake_threshold(*, v_ego, cruise_cap, response_model, **kwargs):
    threshold_calls.append(float(cruise_cap))
    return 2.01

  monkeypatch.setattr(map_strategy, 'cruise_cap_for_required_average_decel', fake_probe)
  monkeypatch.setattr(map_strategy, 'predict_average_decel_for_cruise_cap', fake_threshold)

  candidate = compute_map_cap_candidate(
    mode='strategic',
    s_list=[10.0, 80.0, 120.0],
    k_list=[0.020, 0.030, 0.025],
    vsafe_list=[22.0, 17.0, 15.0],
    abs_indices=[3, 6, 9],
    v_ego=24.0,
    v_cruise=27.0,
    vis_horizon_s=1.4,
    vis_margin_m=10.0,
    severe_vision=False,
    partial_vision=False,
    vision_confidence=0.95,
    conf_lo=0.55,
    conf_hi=0.85,
    max_decel=3.5,
    horizon_limit_m=250.0,
    response_model=response_model,
    fixed_lead_time_s=0.0,
    curve_phase_offset_s=_curve_phase_raw_for_effective(0.0),
    overshoot_phase_offset_s=0.0,
    reference_speed_mps=24.0,
  )

  assert threshold_calls == [0.0]
  assert len(probe_calls) == 1
  assert float(candidate.cap_mps) == pytest.approx(16.5, abs=1e-6)
  assert float(candidate.anchor_dist_m) == pytest.approx(80.0, abs=1e-6)
  assert int(candidate.anchor_index) == 6


def test_strategic_chain_envelope_limits_accel_for_same_speed_next_curve():
  response_model = build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, actuation_delay_s=0.35)
  target_speed = 10.0
  target_dist = 60.0

  candidate = compute_map_cap_candidate(
    mode='strategic',
    s_list=[target_dist],
    k_list=[0.02],
    vsafe_list=[target_speed],
    abs_indices=[6],
    v_ego=10.0,
    v_cruise=25.0,
    vis_horizon_s=1.4,
    vis_margin_m=10.0,
    severe_vision=False,
    partial_vision=False,
    vision_confidence=0.95,
    conf_lo=0.55,
    conf_hi=0.85,
    max_decel=3.5,
    horizon_limit_m=250.0,
    response_model=response_model,
    fixed_lead_time_s=0.0,
    curve_phase_offset_s=_curve_phase_raw_for_effective(0.0),
    overshoot_phase_offset_s=0.0,
    reference_speed_mps=10.0,
  )

  a_plan = float(response_model.planning_decel_mps2)
  delay = float(response_model.actuation_delay_s)
  expected_cap = -a_plan * delay + math.sqrt((a_plan * delay) ** 2 + target_speed ** 2 + 2.0 * a_plan * target_dist)

  assert float(candidate.cap_mps) == pytest.approx(float(expected_cap), abs=1e-6)
  assert float(candidate.anchor_dist_m) == pytest.approx(target_dist, abs=1e-6)
  assert int(candidate.anchor_index) == 6


def test_strategic_compute_feeds_chain_envelope_with_reaccelerate_retighten_anchors(monkeypatch):
  response_model = build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, actuation_delay_s=0.35)
  captured = {}

  def fake_envelope(*, frontier_points, response_model, v_cruise):
    captured['frontier_points'] = list(frontier_points)
    return float(v_cruise), None

  monkeypatch.setattr(map_strategy, '_strategic_chain_envelope_cap', fake_envelope)

  compute_map_cap_candidate(
    mode='strategic',
    s_list=[5.0, 15.0, 30.0, 45.0, 60.0, 80.0],
    k_list=[0.020, 0.012, 0.001, 0.009, 0.001, 0.030],
    vsafe_list=[10.0, 12.0, 17.0, 13.0, 16.0, 8.0],
    abs_indices=[1, 2, 3, 4, 5, 6],
    v_ego=10.0,
    v_cruise=25.0,
    vis_horizon_s=1.4,
    vis_margin_m=10.0,
    severe_vision=False,
    partial_vision=False,
    vision_confidence=0.95,
    conf_lo=0.55,
    conf_hi=0.85,
    max_decel=3.5,
    horizon_limit_m=250.0,
    response_model=response_model,
    fixed_lead_time_s=0.0,
    curve_phase_offset_s=_curve_phase_raw_for_effective(0.0),
    overshoot_phase_offset_s=0.0,
    reference_speed_mps=10.0,
  )

  points = captured['frontier_points']
  assert [float(row[1]) for row in points] == pytest.approx([5.0, 45.0, 80.0], abs=1e-6)
  assert [float(row[2]) for row in points] == pytest.approx([10.0, 13.0, 8.0], abs=1e-6)
  assert [int(row[4]) for row in points] == [1, 4, 6]


def test_winding_road_context_detects_dense_curve_cluster():
  vsafe_list = [23.0, 20.0, 13.0, 16.0, 21.0, 18.0, 12.0, 16.0, 20.0, 17.0, 11.0, 15.0, 21.0, 23.0]
  s_list = [10.0 * (idx + 1) for idx in range(len(vsafe_list))]
  abs_indices = list(range(len(vsafe_list)))

  ctx = map_strategy.classify_winding_road_context(
    s_list=s_list,
    vsafe_list=vsafe_list,
    abs_indices=abs_indices,
  )

  assert bool(ctx.active) is True
  assert float(ctx.score) >= 0.55
  assert int(ctx.anchor_count) == 3
  assert int(ctx.short_gap_count) == 2
  assert float(ctx.curve_distance_m) >= 70.0


def test_winding_road_context_rejects_single_offramp():
  vsafe_list = [27.0, 27.0, 26.0, 22.0, 14.0, 10.0, 14.0, 22.0, 26.0, 27.0, 27.0]
  s_list = [10.0 * (idx + 1) for idx in range(len(vsafe_list))]
  abs_indices = list(range(len(vsafe_list)))

  ctx = map_strategy.classify_winding_road_context(
    s_list=s_list,
    vsafe_list=vsafe_list,
    abs_indices=abs_indices,
  )

  assert bool(ctx.active) is False
  assert int(ctx.anchor_count) == 1
  assert int(ctx.short_gap_count) == 0


def test_winding_road_context_rejects_sparse_freeway_bends():
  vsafe_list = [27.0, 26.0, 19.0, 23.0, 27.0, 27.0, 27.0, 27.0, 27.0, 26.0, 18.0, 23.0, 27.0, 27.0]
  s_list = [25.0 * (idx + 1) for idx in range(len(vsafe_list))]
  abs_indices = list(range(len(vsafe_list)))

  ctx = map_strategy.classify_winding_road_context(
    s_list=s_list,
    vsafe_list=vsafe_list,
    abs_indices=abs_indices,
  )

  assert bool(ctx.active) is False
  assert int(ctx.anchor_count) == 2
  assert int(ctx.short_gap_count) == 0


def test_winding_road_context_rejects_shallow_rolling_road():
  vsafe_list = [27.0, 26.2, 24.7, 25.6, 26.8, 25.2, 24.4, 25.5, 26.6, 25.4, 24.8, 26.3, 27.0]
  s_list = [10.0 * (idx + 1) for idx in range(len(vsafe_list))]
  abs_indices = list(range(len(vsafe_list)))

  ctx = map_strategy.classify_winding_road_context(
    s_list=s_list,
    vsafe_list=vsafe_list,
    abs_indices=abs_indices,
  )

  assert bool(ctx.active) is False
  assert int(ctx.anchor_count) == 0
  assert float(ctx.score) < 0.55


def test_winding_road_context_stays_active_under_small_vsafe_noise():
  baseline_vsafe = [23.0, 20.0, 13.0, 16.0, 21.0, 18.0, 12.0, 16.0, 20.0, 17.0, 11.0, 15.0, 21.0, 23.0]
  noisy_vsafe = [v + dv for v, dv in zip(
    baseline_vsafe,
    [0.10, -0.15, 0.18, -0.12, 0.05, -0.10, 0.22, -0.08, 0.12, -0.18, 0.16, -0.05, 0.09, -0.04],
    strict=False,
  )]
  s_list = [10.0 * (idx + 1) for idx in range(len(baseline_vsafe))]
  abs_indices = list(range(len(baseline_vsafe)))

  ctx = map_strategy.classify_winding_road_context(
    s_list=s_list,
    vsafe_list=noisy_vsafe,
    abs_indices=abs_indices,
  )

  assert bool(ctx.active) is True
  assert int(ctx.anchor_count) == 3
  assert int(ctx.short_gap_count) == 2


def test_winding_road_context_real_profiles_hold_activation_boundary():
  profiles = _load_winding_profile_fixture()

  for row in profiles:
    ctx = map_strategy.classify_winding_road_context(
      s_list=row["profile_s_m"],
      vsafe_list=row["profile_vsafe_mps"],
      abs_indices=list(range(len(row["profile_s_m"]))),
    )
    expected_active = row["level"] >= 3
    assert bool(ctx.active) is expected_active, row["label"]


def test_winding_road_context_real_profiles_score_increases_with_level():
  profiles = _load_winding_profile_fixture()
  scores = []
  for row in profiles:
    ctx = map_strategy.classify_winding_road_context(
      s_list=row["profile_s_m"],
      vsafe_list=row["profile_vsafe_mps"],
      abs_indices=list(range(len(row["profile_s_m"]))),
    )
    scores.append(float(ctx.score))

  assert scores == sorted(scores)
  assert scores[2] < float(map_strategy.WINDING_ROAD_ACTIVE_SCORE)
  assert scores[3] > float(map_strategy.WINDING_ROAD_ACTIVE_SCORE)
  assert scores[-1] >= 0.95


def test_snapshot_exposes_winding_road_context(monkeypatch):
  map_profile = [
    0.0005, 0.0040, 0.0140, 0.0080, 0.0020,
    0.0100, 0.0200, 0.0090, 0.0020,
    0.0130, 0.0240, 0.0100, 0.0020, 0.0005,
  ]
  snap = _run_profile_map_snapshot(
    monkeypatch,
    mode='strategic',
    map_profile=map_profile,
    v0=18.0,
    v_cruise=24.0,
    current_curve=0.002,
    curvature_ahead=0.003,
  )

  assert bool(snap['winding_road_active']) is True
  assert float(snap['winding_road_score']) >= 0.55
  assert int(snap['winding_anchor_count']) >= 2
  assert int(snap['winding_short_gap_count']) >= 1
  assert float(snap['winding_curve_distance_m']) > 0.0


def test_snapshot_exposes_mapd_winding_summary():
  snap = simulate_sequence(
    steps=[
      Step(
        curvature=0.0010,
        curvature_ahead=0.0012,
        confidence=0.95,
        live_map_data={
          'windingRoadValid': True,
          'windingRoadLevel': 4,
          'windingRoadScore': 208,
          'windingRoadConfidence': 196,
          'windingRoadCurrentLevel': 2,
          'windingRoadCurrentScore': 124,
          'windingRoadCurrentConfidence': 180,
          'windingRoadWayCount': 3,
        },
      )
      for _ in range(6)
    ],
    v0_mps=18.0,
    v_cruise_mps=24.0,
    dt=0.05,
  )

  assert bool(snap['mapd_winding_valid']) is True
  assert int(snap['mapd_winding_level']) == 4
  assert int(snap['mapd_winding_score']) == 208
  assert int(snap['mapd_winding_confidence']) == 196
  assert int(snap['mapd_winding_current_level']) == 2
  assert int(snap['mapd_winding_current_score']) == 124
  assert int(snap['mapd_winding_current_confidence']) == 180
  assert int(snap['mapd_winding_way_count']) == 3


def test_winding_context_uses_mapd_when_local_detector_is_inactive():
  snap = simulate_sequence(
    steps=[
      Step(
        curvature=0.0008,
        curvature_ahead=0.0010,
        confidence=0.95,
        live_map_data={
          'windingRoadValid': True,
          'windingRoadLevel': 4,
          'windingRoadScore': 204,
          'windingRoadConfidence': 190,
          'windingRoadCurrentLevel': 2,
          'windingRoadCurrentScore': 110,
          'windingRoadCurrentConfidence': 170,
          'windingRoadWayCount': 3,
        },
      )
      for _ in range(6)
    ],
    v0_mps=18.0,
    v_cruise_mps=24.0,
    dt=0.05,
  )

  assert bool(snap['winding_road_active']) is False
  assert bool(snap['winding_context_active']) is True
  assert snap['winding_context_source'] == 'mapd'
  assert int(snap['winding_context_level']) == 4
  assert float(snap['winding_context_score']) >= 0.75


def test_winding_context_blends_local_and_mapd_detectors(monkeypatch):
  map_profile = [
    0.0005, 0.0040, 0.0140, 0.0080, 0.0020,
    0.0100, 0.0200, 0.0090, 0.0020,
    0.0130, 0.0240, 0.0100, 0.0020, 0.0005,
  ]
  vtsc = mk_vtsc_with_params()
  _enable_map_lookahead(vtsc, monkeypatch)
  _set_map_strategy(vtsc, monkeypatch, 'strategic')
  _set_longitudinal_response_model(vtsc, min_accel=-6.0, delay_s=0.35)
  pts = _build_map_profile_polyline(37.0, -122.0, map_profile, profile_start_m=0.0)
  _patch_map_tail_inputs(vtsc, monkeypatch, 37.0, -122.0, pts)

  snap = simulate_sequence(
    steps=[
      Step(
        curvature=0.0020,
        curvature_ahead=0.0030,
        confidence=0.95,
        live_map_data={
          'windingRoadValid': True,
          'windingRoadLevel': 5,
          'windingRoadScore': 230,
          'windingRoadConfidence': 210,
          'windingRoadCurrentLevel': 3,
          'windingRoadCurrentScore': 150,
          'windingRoadCurrentConfidence': 190,
          'windingRoadWayCount': 4,
        },
      )
      for _ in range(8)
    ],
    vtsc=vtsc,
    v0_mps=18.0,
    v_cruise_mps=24.0,
    dt=0.05,
  )

  assert bool(snap['winding_road_active']) is True
  assert bool(snap['winding_context_active']) is True
  assert snap['winding_context_source'] == 'blended'
  assert int(snap['winding_context_level']) == 5
  assert float(snap['winding_context_score']) >= float(snap['winding_road_score']) - 1e-6


def test_winding_behavior_profile_preserves_mapd_severity_levels():
  gentle = map_strategy.resolve_winding_behavior_profile(
    active=False,
    level=0,
    score=0.0,
    confidence=0.0,
    source='none',
    local_active=False,
    local_score=0.0,
    mapd_level=2,
    mapd_score=0.62,
    mapd_confidence=0.74,
  )
  tight = map_strategy.resolve_winding_behavior_profile(
    active=False,
    level=0,
    score=0.0,
    confidence=0.0,
    source='none',
    local_active=False,
    local_score=0.0,
    mapd_level=4,
    mapd_score=0.78,
    mapd_confidence=0.82,
  )

  assert int(gentle.level) == 2
  assert int(tight.level) == 4
  assert float(tight.v_turn_release_up_slew_mps2) < float(gentle.v_turn_release_up_slew_mps2)
  assert float(tight.apex_release_lat_acc_ratio) < float(gentle.apex_release_lat_acc_ratio)
  assert float(tight.counterevidence_dwell_s) > float(gentle.counterevidence_dwell_s)


def test_winding_profile_keeps_map_ownership_longer_between_chained_curves():
  candidate = MapCapCandidate(
    mode='strategic',
    cap_mps=15.0,
    start_m=0.0,
    coverage=0.8,
    reason='cap_available',
    anchor_dist_m=40.0,
    anchor_vsafe_mps=10.0,
    anchor_curvature=0.02,
    anchor_index=12,
  )

  baseline = evaluate_map_strategy(
    mode='strategic',
    state=MapStrategyState(),
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=30.0,
    vis_margin_m=10.0,
    now_s=0.10,
    apex_exit_ready=True,
    winding_profile=map_strategy.WINDING_BEHAVIOR_PROFILES[0],
  )
  winding = evaluate_map_strategy(
    mode='strategic',
    state=MapStrategyState(),
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=30.0,
    vis_margin_m=10.0,
    now_s=0.10,
    apex_exit_ready=True,
    winding_profile=map_strategy.WINDING_BEHAVIOR_PROFILES[4],
  )

  assert baseline.apply_map_cap is False
  assert baseline.vision_relax_allowed is True
  assert baseline.vision_relax_reason == 'post_apex_release'
  assert winding.apply_map_cap is True
  assert winding.map_floor_active is True
  assert winding.vision_relax_allowed is False
  assert winding.strategy_state == 'vision_clear_waiting'


def test_winding_profile_later_brake_cap_stays_planner_reachable():
  response_model = build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, actuation_delay_s=0.35)
  profile = map_strategy.WINDING_BEHAVIOR_PROFILES[5]
  v_ego = 24.0
  v_cruise = 27.0
  anchor_dist = 90.0
  anchor_vsafe = 12.0

  candidate = compute_map_cap_candidate(
    mode='strategic',
    s_list=[anchor_dist],
    k_list=[0.022],
    vsafe_list=[anchor_vsafe],
    abs_indices=[9],
    v_ego=v_ego,
    v_cruise=v_cruise,
    vis_horizon_s=1.4,
    vis_margin_m=10.0,
    severe_vision=False,
    partial_vision=False,
    vision_confidence=0.95,
    conf_lo=0.55,
    conf_hi=0.85,
    max_decel=3.5,
    horizon_limit_m=250.0,
    response_model=response_model,
    fixed_lead_time_s=0.0,
    curve_phase_offset_s=_curve_phase_raw_for_effective(0.0),
    overshoot_phase_offset_s=0.0,
    reference_speed_mps=v_ego,
    winding_profile=profile,
  )

  effective_distance = float(anchor_dist)
  effective_distance += float(profile.curve_phase_offset_adjust_s) * float(v_ego)
  effective_distance += float(profile.overshoot_phase_offset_adjust_s) * float(v_ego)
  braking_distance = max(0.0, effective_distance - float(v_ego) * float(response_model.actuation_delay_s))
  required_decel = max(0.0, (float(v_ego) * float(v_ego) - float(anchor_vsafe) * float(anchor_vsafe)) / (2.0 * braking_distance))
  supported_decel = predict_average_decel_for_cruise_cap(
    v_ego=v_ego,
    cruise_cap=float(candidate.cap_mps),
    response_model=response_model,
  )

  assert supported_decel + float(CRUISE_CAP_REQUIRED_DECEL_TOL_MPS2) >= required_decel
  assert float(candidate.anchor_dist_m) == pytest.approx(float(anchor_dist), abs=1e-6)


def test_winding_profile_release_slew_limits_upward_v_turn_step():
  vtsc = mk_vtsc_with_params()
  _set_longitudinal_response_model(vtsc, min_accel=-6.0, max_accel=5.0, delay_s=0.35)
  vtsc._max_accel = 3.0
  vtsc._v_cruise_setpoint = 25.0
  vtsc._v_turn_output = 10.0
  vtsc._map_tail_active = True

  vtsc._set_winding_behavior_profile(map_strategy.WINDING_BEHAVIOR_PROFILES[0], source='none')
  baseline_first = vtsc._apply_winding_v_turn_release_slew(20.0, 0.05)
  vtsc._v_turn_output = baseline_first
  vtsc._map_tail_active = False
  baseline_follow = vtsc._apply_winding_v_turn_release_slew(20.0, 0.05)

  vtsc._set_winding_behavior_profile(map_strategy.WINDING_BEHAVIOR_PROFILES[4], source='mapd')
  vtsc._v_turn_output = 10.0
  vtsc._v_turn_release_shape_active = False
  vtsc._map_tail_active = True
  limited = vtsc._apply_winding_v_turn_release_slew(20.0, 0.05)

  assert baseline_first == pytest.approx(10.225, abs=1e-6)
  assert baseline_follow == pytest.approx(10.45, abs=1e-6)
  assert limited == pytest.approx(10.14, abs=1e-6)
  assert limited < baseline_first
  assert bool(vtsc._dbg_winding_release_limited) is True
  assert bool(vtsc._dbg_winding_release_shape_active) is True


def test_winding_profile_pre_apex_release_rolls_on_gradually():
  def run_once(current_lat_acc: float, *, apex_exit_ready: bool):
    vtsc = mk_vtsc_with_params()
    _set_longitudinal_response_model(vtsc, min_accel=-6.0, max_accel=5.0, delay_s=0.35)
    vtsc._max_accel = 3.0
    vtsc._v_cruise_setpoint = 25.0
    vtsc._v_turn_output = 10.0
    vtsc._map_tail_active = True
    vtsc._set_winding_behavior_profile(map_strategy.WINDING_BEHAVIOR_PROFILES[4], source='mapd')
    vtsc._filtered_curvature = 0.01
    vtsc._current_lat_acc = float(current_lat_acc)
    vtsc._max_pred_lat_acc = 2.50
    vtsc._lat_acc_overshoot_ahead = True
    vtsc._apex_exit_ready = bool(apex_exit_ready)
    vtsc._v_turn_release_shape_active = False
    return vtsc._apply_winding_v_turn_release_slew(20.0, 0.05)

  early = run_once(1.02, apex_exit_ready=False)
  late = run_once(2.20, apex_exit_ready=False)
  post = run_once(2.20, apex_exit_ready=True)

  assert 10.05 < early < 10.10
  assert early < late < post
  assert post == pytest.approx(10.14, abs=1e-6)


def test_winding_profile_near_apex_release_helper_scales_with_severity():
  def run_once(profile):
    vtsc = mk_vtsc_with_params()
    vtsc._set_winding_behavior_profile(profile, source='mapd')
    vtsc._filtered_curvature = 0.01
    vtsc._current_lat_acc = 2.15
    vtsc._max_pred_lat_acc = 2.50
    vtsc._lat_acc_overshoot_ahead = True
    ready = vtsc._is_near_apex_release_ready()
    return ready, vtsc.snapshot_debug_state()

  baseline_ready, baseline = run_once(map_strategy.WINDING_BEHAVIOR_PROFILES[0])
  winding_ready, winding = run_once(map_strategy.WINDING_BEHAVIOR_PROFILES[4])

  assert baseline_ready is False
  assert bool(baseline['near_apex_release_ready']) is False

  assert winding_ready is True
  assert bool(winding['near_apex_release_ready']) is True
  assert float(winding['apex_release_lat_acc_ratio']) < float(baseline['apex_release_lat_acc_ratio'])


def test_winding_profile_near_apex_release_can_prespool_before_geometric_apex():
  def run_once(profile):
    vtsc = mk_vtsc_with_params()
    vtsc._set_winding_behavior_profile(profile, source='mapd')
    vtsc._filtered_curvature = 0.01
    vtsc._current_lat_acc = 1.00
    vtsc._max_pred_lat_acc = 2.50
    vtsc._lat_acc_overshoot_ahead = True
    ready = vtsc._is_near_apex_release_ready()
    return ready, vtsc.snapshot_debug_state()

  baseline_ready, baseline = run_once(map_strategy.WINDING_BEHAVIOR_PROFILES[0])
  winding_ready, winding = run_once(map_strategy.WINDING_BEHAVIOR_PROFILES[4])

  assert baseline_ready is False
  assert bool(baseline['near_apex_release_ready']) is False
  assert winding_ready is True
  assert bool(winding['near_apex_release_ready']) is True
  assert float(winding['apex_release_lat_acc_ratio']) == pytest.approx(0.40, abs=1e-6)


def test_near_apex_release_helper_disables_overshoot_cap_before_geometric_apex():
  vtsc = mk_vtsc_with_params()
  _set_longitudinal_response_model(vtsc, min_accel=-6.0, max_accel=5.0, delay_s=0.35)
  vtsc._v_ego = 20.0
  vtsc._a_ego = 0.0
  vtsc._v_cruise_setpoint = 33.0
  vtsc._filtered_curvature = 0.01
  vtsc._current_lat_acc = 2.15
  vtsc._max_pred_lat_acc = 2.50
  vtsc._apex_indices = []
  vtsc._apex_exit_ready = False
  vtsc._is_easing = False
  vtsc._lat_acc_overshoot_ahead = True
  vtsc._overshoot_cap_active = True
  vtsc._v_overshoot = 12.0
  vtsc._v_overshoot_distance = 20.0
  vtsc._overshoot_trigger_in_s = -0.5
  vtsc._occlusion_state.vision_good = True
  vtsc._occlusion_state.smoothed_confidence = 0.95

  with patch.object(vtsc, '_is_near_apex_release_ready', return_value=True), \
       patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: 100.0), \
       patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: 100.0):
    vtsc._update_solution()
  snap = vtsc.snapshot_debug_state()

  assert bool(snap['apex_exit_ready']) is False
  assert bool(snap['overshoot_cap_active']) is False
  assert float(snap['vtsc_cmd']) > 12.0


def test_strategic_response_bounded_probe_matches_bruteforce():
  response_model = build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, actuation_delay_s=0.35)
  s_list = [30.0, 60.0, 90.0, 120.0]
  k_list = [0.020, 0.030, 0.025, 0.010]
  vsafe_list = [20.0, 15.0, 14.0, 22.0]
  abs_indices = [3, 6, 9, 12]
  v_ego = 24.0
  v_cruise = 27.0

  def bruteforce():
    v_cap = float(v_cruise)
    anchor = None
    for di, ki, vi, abs_idx in zip(s_list, k_list, vsafe_list, abs_indices, strict=False):
      braking_distance = max(0.0, float(di) - float(v_ego) * float(response_model.actuation_delay_s))
      if braking_distance <= 1e-3:
        v_allow = float(vi)
      elif float(vi) >= float(v_ego) - 1e-6:
        v_allow = float(v_cruise)
      else:
        required_decel = max(0.0, (float(v_ego) * float(v_ego) - float(vi) * float(vi)) / (2.0 * braking_distance))
        v_allow = map_strategy.cruise_cap_for_required_average_decel(
          v_ego=float(v_ego),
          required_decel_mps2=required_decel,
          response_model=response_model,
          v_cruise_upper=float(v_cruise),
        )
        if v_allow <= 1e-3 and required_decel > 1e-3:
          v_allow = min(float(v_cruise), float(vi))
      if anchor is None or v_allow < v_cap - 1e-6:
        anchor = (float(di), float(vi), float(ki), int(abs_idx))
      v_cap = min(v_cap, v_allow)
    return v_cap, anchor

  expected_cap, expected_anchor = bruteforce()
  candidate = compute_map_cap_candidate(
    mode='strategic',
    s_list=s_list,
    k_list=k_list,
    vsafe_list=vsafe_list,
    abs_indices=abs_indices,
    v_ego=v_ego,
    v_cruise=v_cruise,
    vis_horizon_s=1.4,
    vis_margin_m=10.0,
    severe_vision=False,
    partial_vision=False,
    vision_confidence=0.95,
    conf_lo=0.55,
    conf_hi=0.85,
    max_decel=3.5,
    horizon_limit_m=250.0,
    response_model=response_model,
    fixed_lead_time_s=0.0,
    curve_phase_offset_s=_curve_phase_raw_for_effective(0.0),
    overshoot_phase_offset_s=0.0,
    reference_speed_mps=v_ego,
  )

  assert float(candidate.cap_mps) == pytest.approx(float(expected_cap), rel=1e-6)
  assert float(candidate.anchor_dist_m) == pytest.approx(float(expected_anchor[0]), abs=1e-6)
  assert int(candidate.anchor_index) == int(expected_anchor[3])


def test_strategic_response_single_threshold_matches_bruteforce_randomized():
  rng = random.Random(0)
  response_model = build_cruise_response_model(min_accel_mps2=-6.0, max_accel_mps2=5.0, actuation_delay_s=0.35)
  v_ego = 24.0
  v_cruise = 27.0

  def bruteforce(s_list, k_list, vsafe_list, abs_indices):
    v_cap = float(v_cruise)
    anchor = None
    for di, ki, vi, abs_idx in zip(s_list, k_list, vsafe_list, abs_indices, strict=False):
      braking_distance = max(0.0, float(di) - float(v_ego) * float(response_model.actuation_delay_s))
      if braking_distance <= 1e-3:
        v_allow = float(vi)
      elif float(vi) >= float(v_ego) - 1e-6:
        v_allow = float(v_cruise)
      else:
        required_decel = max(0.0, (float(v_ego) * float(v_ego) - float(vi) * float(vi)) / (2.0 * braking_distance))
        v_allow = map_strategy.cruise_cap_for_required_average_decel(
          v_ego=float(v_ego),
          required_decel_mps2=required_decel,
          response_model=response_model,
          v_cruise_upper=float(v_cruise),
        )
        if v_allow <= 1e-3 and required_decel > 1e-3:
          v_allow = min(float(v_cruise), float(vi))
      if anchor is None or v_allow < v_cap - 1e-6:
        anchor = (float(di), float(vi), float(ki), int(abs_idx))
      v_cap = min(v_cap, v_allow)
    return v_cap, anchor

  for _ in range(16):
    n = rng.randint(2, 8)
    s = []
    dist = 0.0
    for _ in range(n):
      dist += rng.uniform(8.0, 40.0)
      s.append(dist)
    vsafe = []
    # Keep this parity test in the "future target already below current speed" regime.
    # Chained-curve accel limiting above current speed is covered separately.
    current_vsafe = rng.uniform(12.0, 23.5)
    for _ in range(n):
      current_vsafe = max(4.0, current_vsafe - rng.uniform(0.0, 3.5))
      vsafe.append(current_vsafe)
    k = [rng.uniform(0.001, 0.03) for _ in range(n)]
    abs_indices = list(range(n))

    expected_cap, expected_anchor = bruteforce(s, k, vsafe, abs_indices)
    candidate = compute_map_cap_candidate(
      mode='strategic',
      s_list=s,
      k_list=k,
      vsafe_list=vsafe,
      abs_indices=abs_indices,
      v_ego=v_ego,
      v_cruise=v_cruise,
      vis_horizon_s=1.4,
      vis_margin_m=10.0,
      severe_vision=False,
      partial_vision=False,
      vision_confidence=0.95,
      conf_lo=0.55,
      conf_hi=0.85,
      max_decel=3.5,
      horizon_limit_m=250.0,
      response_model=response_model,
      fixed_lead_time_s=0.0,
      curve_phase_offset_s=_curve_phase_raw_for_effective(0.0),
      overshoot_phase_offset_s=0.0,
      reference_speed_mps=v_ego,
    )

    assert float(candidate.cap_mps) == pytest.approx(float(expected_cap), rel=1e-6)
    assert float(candidate.anchor_dist_m) == pytest.approx(float(expected_anchor[0]), abs=1e-6)
    assert int(candidate.anchor_index) == int(expected_anchor[3])


def test_strategic_post_apex_release_helper_state():
  state = MapStrategyState()
  candidate = MapCapCandidate(
    mode='strategic',
    cap_mps=15.0,
    start_m=0.0,
    coverage=0.8,
    reason='cap_available',
    anchor_dist_m=18.0,
    anchor_vsafe_mps=9.0,
    anchor_curvature=0.02,
    anchor_index=9,
  )

  decision = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=0.10,
    apex_exit_ready=True,
  )

  assert decision.apply_map_cap is False
  assert decision.map_floor_active is False
  assert decision.vision_relax_allowed is True
  assert decision.vision_relax_reason == 'post_apex_release'
  assert decision.strategy_state == 'vision_owns'


def test_strategic_counterevidence_dwell_releases_helper_state():
  state = MapStrategyState()
  candidate = MapCapCandidate(
    mode='strategic',
    cap_mps=15.0,
    start_m=0.0,
    coverage=0.8,
    reason='cap_available',
    anchor_dist_m=20.0,
    anchor_vsafe_mps=9.0,
    anchor_curvature=0.02,
    anchor_index=12,
  )

  # Frame 1: enter takeover zone, vision above cap (no approach)
  d0 = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=0.10,
    apex_exit_ready=False,
  )
  assert d0.apply_map_cap is True
  assert d0.vision_relax_allowed is False

  # Frame 2: zone dwell >= takeover_dwell_s (0.35), CE eligible and starts
  d1 = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=0.50,
    apex_exit_ready=False,
  )
  assert d1.apply_map_cap is True
  assert state.counterevidence_since == 0.50

  # Frame 3: CE dwell expired (0.50 + 0.75 = 1.25), release fires
  d2 = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=1.30,
    apex_exit_ready=False,
  )
  assert d2.apply_map_cap is False
  assert d2.vision_relax_allowed is True
  assert d2.vision_relax_reason == 'counterevidence_dwell'
  assert d2.strategy_state == 'vision_owns'


def test_strategic_counterevidence_dwell_waits_for_anchor_visibility():
  state = MapStrategyState()
  candidate = MapCapCandidate(
    mode='strategic',
    cap_mps=15.0,
    start_m=0.0,
    coverage=0.8,
    reason='cap_available',
    anchor_dist_m=38.0,
    anchor_vsafe_mps=9.0,
    anchor_curvature=0.02,
    anchor_index=12,
  )

  d0 = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=28.0,
    vis_margin_m=10.0,
    now_s=0.10,
    apex_exit_ready=False,
  )
  assert d0.apply_map_cap is True
  assert d0.vision_relax_allowed is False

  d1 = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=28.0,
    vis_margin_m=10.0,
    now_s=0.95,
    apex_exit_ready=False,
  )
  assert d1.apply_map_cap is True
  assert d1.map_floor_active is True
  assert d1.vision_relax_allowed is False
  assert d1.strategy_state == 'vision_clear_waiting'


def test_counterevidence_blocked_when_vision_approached_cap():
  """Counterevidence must not fire if vision ever got close to map cap in this zone."""
  state = MapStrategyState()
  candidate = MapCapCandidate(
    mode='strategic',
    cap_mps=15.0,
    start_m=0.0,
    coverage=0.8,
    reason='cap_available',
    anchor_dist_m=20.0,
    anchor_vsafe_mps=9.0,
    anchor_curvature=0.02,
    anchor_index=12,
  )

  # Frame 1: vision approaches cap (15.2 <= 15.0 + 0.25)
  evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=15.2,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=0.10,
    apex_exit_ready=False,
  )
  assert state.takeover_ever_approached is True

  # Frame 2: vision relaxes above cap + counterevidence_delta, wait long enough
  d = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=2.00,
    apex_exit_ready=False,
  )
  # Counterevidence must NOT fire because takeover_ever_approached is True
  assert state.counterevidence_since == 0.0
  assert d.apply_map_cap is True


def test_counterevidence_fires_after_zone_dwell_without_approach():
  """Counterevidence fires when vision never approached cap and zone dwell expired."""
  state = MapStrategyState()
  candidate = MapCapCandidate(
    mode='strategic',
    cap_mps=15.0,
    start_m=0.0,
    coverage=0.8,
    reason='cap_available',
    anchor_dist_m=20.0,
    anchor_vsafe_mps=9.0,
    anchor_curvature=0.02,
    anchor_index=12,
  )

  # Frame 1: enter zone, vision above cap (no approach)
  evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=0.10,
    apex_exit_ready=False,
  )
  assert state.takeover_ever_approached is False
  assert state.zone_entry_since == 0.10

  # Frame 2: zone_elapsed=0.30 (< takeover_dwell_s=0.35), CE not eligible
  d2 = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=0.40,
    apex_exit_ready=False,
  )
  assert d2.apply_map_cap is True
  assert state.counterevidence_since == 0.0

  # Frame 3: zone_elapsed=0.46s (>= 0.35), CE starts accumulating
  evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=0.56,
    apex_exit_ready=False,
  )
  assert state.counterevidence_since == 0.56

  # Frame 4: CE dwell expired (0.56 + 0.75 = 1.31)
  d4 = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=1.35,
    apex_exit_ready=False,
  )
  assert d4.apply_map_cap is False
  assert d4.vision_relax_allowed is True
  assert d4.vision_relax_reason == 'counterevidence_dwell'


def test_rearm_cooldown_prevents_instant_rearm():
  """After counterevidence release, rearm must not happen for at least
  counterevidence_dwell_s even if anchor changes to a far-away point."""
  state = MapStrategyState()
  candidate_near = MapCapCandidate(
    mode='strategic',
    cap_mps=15.0,
    start_m=0.0,
    coverage=0.8,
    reason='cap_available',
    anchor_dist_m=20.0,
    anchor_vsafe_mps=9.0,
    anchor_curvature=0.02,
    anchor_index=12,
  )

  # Drive into counterevidence release: zone entry → CE eligible → CE dwell expired
  evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate_near,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=0.10,
    apex_exit_ready=False,
  )
  evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate_near,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=0.56,
    apex_exit_ready=False,
  )
  d_release = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate_near,
    raw_target_pre_map=16.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=True,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=1.35,
    apex_exit_ready=False,
  )
  assert d_release.apply_map_cap is False
  assert state.release_latched is True
  assert state.release_at > 0.0

  # Next frame: anchor jumps far away, normally would rearm instantly
  candidate_far = MapCapCandidate(
    mode='strategic',
    cap_mps=18.0,
    start_m=0.0,
    coverage=0.8,
    reason='cap_available',
    anchor_dist_m=106.5,
    anchor_vsafe_mps=12.0,
    anchor_curvature=0.01,
    anchor_index=30,
  )
  d_after = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate_far,
    raw_target_pre_map=19.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=False,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=1.40,
    apex_exit_ready=False,
  )
  # Cooldown NOT expired (1.40 - 1.35 = 0.05 < 0.75), still released
  assert d_after.apply_map_cap is False
  assert state.release_latched is True

  # Much later: cooldown expired
  d_rearm = evaluate_map_strategy(
    mode='strategic',
    state=state,
    candidate=candidate_far,
    raw_target_pre_map=19.0,
    full_visibility=True,
    vision_good=True,
    turn_visible=False,
    s_visible_m=35.0,
    vis_margin_m=10.0,
    now_s=2.20,
    apex_exit_ready=False,
  )
  # Now rearm succeeds (2.20 - 1.35 = 0.85 >= 0.75)
  assert d_rearm.apply_map_cap is True
  assert state.release_latched is False


def test_map_lookahead_absent_no_cap(monkeypatch):
  # When GPS or map points are absent, map cap stays inactive
  v0 = 25.0
  v_cruise = 30.0
  vtsc = mk_vtsc_with_params()
  # Disable map by returning no GPS or empty points
  def _map_empty():
    return []
  monkeypatch.setattr(vtsc, "_get_last_gps_pose", lambda: None, raising=True)
  monkeypatch.setattr(vtsc, "_load_map_curvatures", _map_empty, raising=True)
  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.0, confidence=0.95, n=80),
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert bool(snap['map_tail_active']) is False


def test_map_lookahead_reason_toggle_off_by_default():
  # With map lookahead toggle OFF, diagnostics should make that explicit.
  v0 = 25.0
  v_cruise = 30.0
  vtsc = mk_vtsc_with_params()
  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.0, confidence=0.95, n=20),
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert snap.get('map_tail_reason') == 'toggle_off'
  assert bool(snap.get('map_tail_active')) is False


def test_map_lookahead_reason_no_gps_when_enabled(monkeypatch):
  # With map lookahead enabled but no GPS, diagnostics should report no_gps.
  v0 = 25.0
  v_cruise = 30.0
  vtsc = mk_vtsc_with_params()

  _enable_map_lookahead(vtsc, monkeypatch)
  monkeypatch.setattr(vtsc, "_get_last_gps_pose", lambda: None, raising=True)

  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.0, confidence=0.95, n=20),
    vtsc=vtsc,
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert bool(snap.get('map_tail_active')) is False
  assert snap.get('map_tail_reason') == 'no_gps'
  assert snap.get('map_tail_compute_reason') == 'no_gps'


def test_low_confidence_no_longer_changes_visibility_state():
  # Confidence swings should no longer transition VTSC into PARTIAL/SEVERE/LOST. The raw
  # confidence remains observable, but the controller stays in FULL visibility throughout.
  v0 = 20.0
  v_cruise = 25.0
  vtsc = mk_vtsc_with_params()

  def run(curv, conf, n):
    return simulate_sequence(
      steps=_steps_constant(curvature=curv, confidence=conf, n=n),
      vtsc=vtsc,
      v0_mps=v0,
      v_cruise_mps=v_cruise,
      dt=0.05,
    )

  # Start with good vision to stabilize
  snap = run(0.0, 0.95, 40)  # 2.0 s
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(0.95, abs=1e-6)

  snap = run(0.0, 0.64, 2)   # 0.10 s < 0.20 s dwell
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(0.64, abs=1e-6)

  snap = run(0.002, 0.40, 10)  # 0.50 s > 0.20 s dwell
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(0.40, abs=1e-6)

  snap = run(0.002, 0.72, 2)
  assert snap['vision_status'] == 'FULL'
  assert float(snap['raw_path_conf']) == pytest.approx(0.72, abs=1e-6)

  snap = run(0.0, 0.95, 3)     # 0.15 s >= 0.10 s exit dwell
  assert snap['vision_status'] == 'FULL'


def test_low_speed_occlusion_margin_relax(monkeypatch):
  # Force an occlusion scenario with tiny curvature at town speeds
  vtsc = mk_vtsc_with_params(bool_overrides={'VTSCFailOpen': False})
  # Warm the controller so internal buffers are initialized
  simulate_sequence(steps=_steps_constant(curvature=0.0, confidence=0.9, n=10), vtsc=vtsc, v0_mps=5.0, v_cruise_mps=8.0, dt=0.05)

  vtsc._fov_occluded = True
  vtsc._fov_on_cnt = 12
  vtsc._occl_lead_bypass_active = False
  vtsc._freeway_failopen_active = False
  vtsc._vis_horizon_s = 1.4
  vtsc._vis_margin_m = 10.0
  vtsc._v_ego = 4.5
  vtsc._prev_target_speed = 4.3
  vtsc._v_cruise_setpoint = 9.0
  vtsc._filtered_curvature = 0.010  # Force psi gate to open despite small occlusion curvature
  vtsc._dbg_target_raw = 8.0
  vtsc._dbg_target_final = 8.0
  vtsc._plan_advanced_speed_trajectory = lambda: 8.0
  occ = vtsc._occlusion_state
  occ.vision_good = False
  occ.est_curvature = 2.5e-4
  occ.last_valid_curvature = 2.0e-4
  occ.distance_since_m = 1.0
  occ.occluded_since_time = 0.0

  t0 = 100.0
  monkeypatch.setattr('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: t0)
  monkeypatch.setattr('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: t0)
  vtsc._update_solution()
  snap = vtsc.snapshot_debug_state()
  assert bool(snap['low_speed_margin_override']) is True
  assert bool(snap['occl_positive_margin']) is True
  assert float(snap['a_cmd']) >= -1e-6  # no enforced decel in relaxed case


def test_low_speed_occlusion_tight_curve_still_blocks(monkeypatch):
  vtsc = mk_vtsc_with_params(bool_overrides={'VTSCFailOpen': False})
  simulate_sequence(steps=_steps_constant(curvature=0.0, confidence=0.9, n=10), vtsc=vtsc, v0_mps=5.0, v_cruise_mps=8.0, dt=0.05)

  vtsc._fov_occluded = True
  vtsc._fov_on_cnt = 12
  vtsc._occl_lead_bypass_active = False
  vtsc._freeway_failopen_active = False
  vtsc._vis_horizon_s = 1.4
  vtsc._vis_margin_m = 10.0
  vtsc._v_ego = 4.5
  vtsc._prev_target_speed = 4.3
  vtsc._v_cruise_setpoint = 9.0
  vtsc._filtered_curvature = 0.009
  vtsc._dbg_target_raw = 8.0
  vtsc._dbg_target_final = 6.0
  vtsc._plan_advanced_speed_trajectory = lambda: 6.0
  occ = vtsc._occlusion_state
  occ.vision_good = False
  occ.est_curvature = 0.009
  occ.last_valid_curvature = 0.009
  occ.distance_since_m = 6.0
  occ.occluded_since_time = 0.0

  t0 = 200.0
  monkeypatch.setattr('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: t0)
  monkeypatch.setattr('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: t0)
  vtsc._update_solution()
  snap = vtsc.snapshot_debug_state()
  assert bool(snap['low_speed_margin_override']) is False
  assert bool(snap['occl_positive_margin']) is False


def test_rlog_loader_smoke():
  rlog_path = Path('docs/chauffeur/vtsc/fullTrace/cases/overslow_2025-09-05/00000085--f247b281ca--80/rlog_00000085--f247b281ca--80.zst')
  assert rlog_path.exists()
  steps = load_steps_from_rlog(str(rlog_path), limit=20)
  assert steps, "Expected at least one step from rlog"
  assert isinstance(steps[0].curvature, float)
  assert isinstance(steps[0].confidence, float)
  vtsc = mk_vtsc_with_params()
  snap = simulate_sequence(steps=steps, vtsc=vtsc, v0_mps=20.0, v_cruise_mps=25.0, dt=0.05)
  assert snap
