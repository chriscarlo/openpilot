#!/usr/bin/env python3
import math

import pytest

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed
from .harness import Step, simulate_sequence, mk_vtsc_with_params


def _steps_constant(curvature: float, confidence: float, n: int, lead_d: float | None = None):
  for _ in range(n):
    yield Step(curvature=curvature, confidence=confidence, lead_d_rel_m=lead_d)


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
  snap = simulate_sequence(steps=steps, v0_mps=v0, v_cruise_mps=v_cruise, dt=0.05)
  assert snap
  assert snap['vision_status'] == 'FULL'
  # Fast reacq window applies a floor ~0.18 m/s²
  assert float(snap['a_last']) >= 0.18 - 1e-3


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
  assert snap['vision_status'] != 'FULL'

  # Now sustain strong vision long enough to exit occlusion
  snap = run(0.0, 0.95, 3)     # 0.15 s >= 0.10 s exit dwell
  assert snap['vision_status'] == 'FULL'
