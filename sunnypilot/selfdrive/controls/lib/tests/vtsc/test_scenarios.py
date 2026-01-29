#!/usr/bin/env python3
import math

import pytest

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed, VTURN_HOLD_S
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
    return any(s['a_target'] > 0.05 for s in trace[start_idx:start_idx + int(1.5 / dt)])

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
  k = 0.0035

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


def test_map_lookahead_cap_applies_when_available(monkeypatch):
  # Patch controller to provide GPS + synthetic map tail; assert map tail active and cap < cruise
  v0 = 25.0
  v_cruise = 30.0
  vtsc = mk_vtsc_with_params()
  # Enable map lookahead explicitly for this test (mk_vtsc_with_params defaults most toggles off).
  orig_get_bool = vtsc._get_bool_param
  def _get_bool(key: str, default: bool = False) -> bool:
    if key == 'MTSCLookaheadEnabled':
      return True
    return bool(orig_get_bool(key, default))
  monkeypatch.setattr(vtsc, "_get_bool_param", _get_bool, raising=True)
  # Enable map lookahead via get_bool and patch data providers
  import types
  def _gps():
    return (37.0, -122.0)
  def _map_pts():
    # Construct ~10 points with increasing curvature ahead (~0.0 near, then 0.008 farther)
    base_lat, base_lon = 37.0, -122.0
    pts = []
    for i in range(10):
      lat = base_lat + 0.0001 * i
      lon = base_lon
      k = 0.0 if i < 3 else 0.008
      pts.append((lat, lon, k))
    return pts
  monkeypatch.setattr(vtsc, "_get_last_gps", _gps, raising=True)
  monkeypatch.setattr(vtsc, "_load_map_curvatures", _map_pts, raising=True)
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

  # Enable map lookahead explicitly for this test (mk_vtsc_with_params defaults most toggles off).
  orig_get_bool = vtsc._get_bool_param
  def _get_bool(key: str, default: bool = False) -> bool:
    if key == 'MTSCLookaheadEnabled':
      return True
    return bool(orig_get_bool(key, default))
  monkeypatch.setattr(vtsc, "_get_bool_param", _get_bool, raising=True)

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
  monkeypatch.setattr(vtsc, "_get_last_gps", lambda: (lat0, lon0), raising=True)
  monkeypatch.setattr(vtsc, "_load_map_curvatures", lambda: pts, raising=True)

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
  v_expected = math.sqrt(max(0.0, v_safe * v_safe + 2.0 * a_comf * 10.0))
  assert trace[0]['v_turn'] < v_cruise - 1e-3
  assert trace[0]['v_turn'] == pytest.approx(v_expected, abs=0.75)


def test_map_lookahead_absent_no_cap(monkeypatch):
  # When GPS or map points are absent, map cap stays inactive
  v0 = 25.0
  v_cruise = 30.0
  vtsc = mk_vtsc_with_params()
  # Disable map by returning no GPS or empty points
  def _gps_none():
    return None
  def _map_empty():
    return []
  monkeypatch.setattr(vtsc, "_get_last_gps", _gps_none, raising=True)
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
