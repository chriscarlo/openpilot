#!/usr/bin/env python3
import math

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
  curvature_to_speed,
  SEVERE_OVERSHOOT_SPEED_SCALE_MIN,
  VTURN_HOLD_S,
)
from sunnypilot.selfdrive.controls.lib.vtsc_map_strategy import (
  MapCapCandidate,
  MapStrategyState,
  compute_map_cap_candidate,
  evaluate_map_strategy,
)
from openpilot.selfdrive.controls.lib.longitudinal_response_model import build_cruise_response_model
from pathlib import Path

from .harness import Step, simulate_sequence, simulate_sequence_trace, mk_vtsc_with_params, load_steps_from_rlog


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


def test_freeway_cap_hold_prevents_single_frame_flicker():
  # Regression (freeway): we observed many short (<0.5s) VTSC cap dips in rlogs, which the
  # longitudinal planner/MPC often cannot respond to quickly. VTSC should hold a material cap
  # reduction briefly so the planner sees a stable target and begins braking sooner.
  v0 = 32.0
  v_cruise = 33.0
  dt = 0.05

  # One-frame "curve ahead" pulse (horizon only), then straight. Without the hold, v_turn would
  # immediately jump back to cruise on the next frame.
  steps = [Step(curvature=0.0, curvature_ahead=0.012, confidence=0.95)]
  steps += [Step(curvature=0.0, curvature_ahead=0.0, confidence=0.95) for _ in range(int(VTURN_HOLD_S / dt) + 12)]

  trace = simulate_sequence_trace(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=dt, integrate_ego=False)
  assert trace and len(trace) >= 3

  # First frame should produce a meaningful cap reduction.
  assert float(trace[0]['v_turn']) <= v_cruise - 1.0
  # Second frame must remain held low even though horizon is straight.
  assert float(trace[1]['v_turn']) <= v_cruise - 1.0
  # After the hold expires, cap should recover to cruise promptly.
  assert float(trace[-1]['v_turn']) >= v_cruise - 1e-3


def test_mountain_cap_hold_prevents_single_frame_flicker_under_occlusion():
  # Regression (mountain): at 30-50 mph we observed "cap flapping" where VTSC briefly recommends a
  # much lower speed for <0.5s, then returns to cruise. The longitudinal planner often cannot
  # react within that window, so braking begins late and the driver intervenes.
  #
  # Hold material cap reductions briefly under degraded vision so the planner sees a stable target.
  v0 = 15.0
  v_cruise = 24.0
  dt = 0.05
  conf = 0.52  # below CONF_BAD_TH to emulate severe/occluded frames in real events

  # Warm-in low confidence so the controller enters occlusion mode (smoothed confidence hysteresis).
  steps = [Step(curvature=0.0, curvature_ahead=0.0, confidence=conf) for _ in range(10)]
  # One-frame "curve ahead" pulse (horizon only), then straight.
  steps += [Step(curvature=0.0, curvature_ahead=0.012, confidence=conf)]
  steps += [Step(curvature=0.0, curvature_ahead=0.0, confidence=conf) for _ in range(int(VTURN_HOLD_S / dt) + 12)]

  trace = simulate_sequence_trace(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=dt, integrate_ego=False)
  assert trace and len(trace) >= 3

  # First frame should produce a meaningful cap reduction.
  first = 10
  assert float(trace[first]['v_turn']) <= v_cruise - 1.0
  # Second frame must remain held low even though horizon is straight.
  assert float(trace[first + 1]['v_turn']) <= v_cruise - 1.0
  # After the hold expires, cap should recover to cruise promptly.
  assert float(trace[-1]['v_turn']) >= v_cruise - 1e-3


def test_lead_bypass_active_allows_raise_with_margin():
  # Occluded vision with a lead at ~2.4 s headway should enable lead-bypass
  v0 = 20.0
  v_cruise = 22.0
  # Choose a small curvature where base speed allows gentle raise
  k = 0.001
  # 2.4 s headway => d_rel ~ v * t
  headway_s = 2.4
  d_rel = v0 * headway_s
  # Build controller with defaults
  vtsc = mk_vtsc_with_params()
  # Run: first ensure occlusion, then present lead
  # Warm-in occlusion by low confidence, then keep occluded with the lead present
  steps = list(_steps_constant(curvature=k, confidence=0.4, n=40, lead_d=None)) 
  steps += list(_steps_constant(curvature=k, confidence=0.4, n=40, lead_d=d_rel))
  snap = simulate_sequence(steps=steps, vtsc=vtsc, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  assert snap, "Snapshot missing"
  # Lead-bypass active under occlusion
  assert not snap['conf'] >= 0.7  # still occluded scenario
  assert bool(snap['occl_lead_bypass_active']) is True
  # With lead bypass active, no undue slow-down under benign curvature
  assert float(snap['final']) >= v0 - 0.5


def test_fov_occlusion_clears_on_straight_even_with_mediocre_confidence():
  # Regression-style invariant (human-like recovery):
  # - Enter a curve (FOV exit condition trips, confidence drops)
  # - Exit to a straight but confidence stays mediocre (paint wear / glare / lead artifacts)
  #
  # Once geometry says the path is safely within FoV again, VTSC should clear the FOV-occlusion latch
  # and allow acceleration back toward cruise. This should be true with *or without* a lead vehicle.
  v0 = 25.0
  v_cruise = 30.0
  dt = 0.05

  # Tight-ish curve to ensure FOV occlusion gate activates at highway speeds.
  k_curve = 0.012
  # Keep predicted lat acc above the UI-disable threshold so the VTSC state machine doesn't call _reset().
  k_ahead = 0.004
  conf = 0.55
  lead_d = 30.0  # close lead; headway stays <= 3s over this sequence
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

  def saw_positive_accel_soon(trace, start_idx: int) -> bool:
    # Allow a little time for jerk-limited recovery back to positive accel.
    # Include the boundary sample at exactly 1.5s (30 frames at 20 Hz).
    return any(s['a_target'] > 0.05 for s in trace[start_idx:start_idx + int(1.5 / dt) + 1])

  trace_no_lead = run(None)
  assert any(s['fov_occluded'] for s in trace_no_lead[:n_curve]), "Expected FOV occlusion latch during curve phase"
  clear_idx = first_clear_idx(trace_no_lead)
  assert clear_idx is not None
  assert (clear_idx - n_curve) * dt <= 1.0
  assert saw_positive_accel_soon(trace_no_lead, clear_idx) is True

  trace_lead = run(lead_d)
  assert any(s['fov_occluded'] for s in trace_lead[:n_curve]), "Expected FOV occlusion latch during curve phase"
  clear_idx_lead = first_clear_idx(trace_lead)
  assert clear_idx_lead is not None
  assert (clear_idx_lead - n_curve) * dt <= 1.0
  assert saw_positive_accel_soon(trace_lead, clear_idx_lead) is True

  # Lead presence should not materially delay clearing.
  assert abs(clear_idx_lead - clear_idx) * dt <= 0.25


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


def test_partial_occlusion_no_lead_blocks_raise_when_no_margin():
  # Occluded, no lead, relatively tight curvature -> margin negative; block positive accel
  v0 = 20.0
  v_cruise = 22.0
  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.008, confidence=0.4, n=80),
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert snap['vision_status'] != 'FULL'
  assert bool(snap['occl_lead_bypass_active']) is False
  # Negative margin expected for tight curvature
  assert bool(snap['occl_positive_margin']) is False
  # No positive accel while occluded without margin
  assert float(snap['a_last']) <= 1e-6
  # Decel command bounded by comfort cap while occluded
  assert float(snap['decel_cmd']) >= float(snap['comfort_decel']) - 1e-6


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
  # Guard against over-reaching occlusion logic:
  #
  # Very low lane-line confidence can occur on straight roads for reasons that are not
  # "can't see around a bend" (e.g., worn paint, glare). VTSC should not block acceleration
  # purely due to confidence when curvature is ~0.
  v0 = 20.0
  v_cruise = 25.0
  snap = simulate_sequence(
    steps=_steps_constant(curvature=0.0, confidence=0.10, n=15),
    v0_mps=v0,
    v_cruise_mps=v_cruise,
    dt=0.05,
  )
  assert snap
  assert snap['vision_status'] in ('SEVERE', 'LOST')
  assert bool(snap['occluded']) is True
  assert float(snap['a_last']) >= 0.05


def test_severe_confidence_low_speed_lead_does_not_freeze():
  # Regression (real-world stop-and-go):
  #
  # When crawling behind a lead, lane-line confidence can drop to SEVERE/LOST (lead covers lines),
  # but VTSC must not "stick" in a no-raise clamp that prevents resuming motion when the lead moves.
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
  assert snap['vision_status'] in ('SEVERE', 'LOST')
  assert bool(snap['lead']) is True
  assert bool(snap['occl_lead_bypass_active']) is True
  # Cap must be above current speed (otherwise planner will not accelerate)
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


def test_hidden_turn_early_decel_with_caps():
  # During early occlusion phase, with tightening curvature behind FoV, ensure decel engages and respects caps
  v0 = 22.0
  v_cruise = 24.0
  # Build increasing curvature steps to mimic tail growth
  ks = [0.002 + i * (0.006 - 0.002) / 20 for i in range(20)]
  steps = [Step(curvature=k, confidence=0.4) for k in ks]
  steps += [Step(curvature=0.006, confidence=0.4) for _ in range(20)]
  snap = simulate_sequence(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  assert snap
  # Decel engaged (negative) and jerk bounded; comfort cap enforced while occluded
  assert float(snap['decel_cmd']) <= 0.0
  assert float(snap['decel_cmd']) >= float(snap['comfort_decel']) - 1e-6
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
  vtsc._curve_phase_offset_s = float(curve_phase_s)
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
  assert float(snap['map_strategic_cap']) >= float(snap['map_advisory_cap']) - 1e-6


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


def test_partial_occlusion_blends_map_tail_start_for_short_hidden_curve(monkeypatch):
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
  assert snap['vision_status'] == 'PARTIAL'
  assert bool(snap['map_tail_active']) is True
  assert float(snap['map_tail_cap']) < v_cruise - 1e-3
  assert float(snap['map_tail_start_m']) > 10.0
  assert float(snap['map_tail_start_m']) < (v0 * float(snap['vis_horizon_s']) + 10.0) - 1e-3
  assert snap['active_cap'] == 'map'


def test_partial_occlusion_keeps_tighter_map_cap_when_visible_curve_is_looser(monkeypatch):
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
  assert snap['vision_status'] == 'PARTIAL'
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
    curve_phase_offset_s=0.0,
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
    curve_phase_offset_s=0.0,
    overshoot_phase_offset_s=2.0,
    reference_speed_mps=vsafe + 0.5,
  )

  assert float(candidate_early.cap_mps) == pytest.approx(float(candidate_late.cap_mps), abs=1e-6)


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
    now_s=0.95,
    apex_exit_ready=False,
  )
  assert d1.apply_map_cap is False
  assert d1.vision_relax_allowed is True
  assert d1.vision_relax_reason == 'counterevidence_dwell'
  assert d1.strategy_state == 'vision_owns'


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


def test_occlusion_dwell_hysteresis_stability():
  # Confidence bouncing around thresholds should respect dwell timers and not oscillate rapidly
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

  # Brief dips below bad threshold may mark PARTIAL, but should not escalate to SEVERE/LOST
  snap = run(0.0, 0.64, 2)   # 0.10 s < 0.20 s dwell
  assert snap['vision_status'] in ('FULL', 'PARTIAL')

  # Sustain poor vision long enough to enter occlusion
  snap = run(0.002, 0.40, 10)  # 0.50 s > 0.20 s dwell
  assert snap['vision_status'] != 'FULL'

  # A brief borderline phase shouldn't be relied on for exit; avoid oscillation checks on status string here
  snap = run(0.002, 0.72, 2)
  # Exit dwell is short by design; don't enforce FULL/non-FULL here.
  # We only care that we don't oscillate rapidly into severe states during borderline segments.
  assert snap['vision_status'] in ('FULL', 'PARTIAL')

  # Now sustain strong vision long enough to exit occlusion
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
