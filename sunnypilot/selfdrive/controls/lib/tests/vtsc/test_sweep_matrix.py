#!/usr/bin/env python3
from __future__ import annotations

import pytest

from .harness import Step, mk_vtsc_with_params, simulate_sequence_trace


def _steps_curve_then_straight(*,
                               k_curve: float,
                               k_straight: float,
                               confidence: float,
                               n_curve: int,
                               n_straight: int,
                               lead_d_rel_m: float | None,
                               curvature_ahead: float | None) -> list[Step]:
  steps: list[Step] = []
  steps += [Step(curvature=k_curve, curvature_ahead=None, confidence=confidence, lead_d_rel_m=lead_d_rel_m) for _ in range(int(n_curve))]
  steps += [Step(curvature=k_straight, curvature_ahead=curvature_ahead, confidence=confidence, lead_d_rel_m=lead_d_rel_m) for _ in range(int(n_straight))]
  return steps


@pytest.mark.parametrize("v0,v_cruise,k_curve", [
  (25.0, 30.0, 0.008),
  (30.0, 33.0, 0.010),
  (30.0, 33.0, 0.012),
])
def test_v_turn_releases_after_curve_sweep(v0: float, v_cruise: float, k_curve: float):
  # Sweep invariant: with good vision, VTSC must:
  # - impose a cap during the curve (v_turn < cruise), and
  # - then wind the cap back up smoothly after the curve instead of jumping straight to cruise.
  dt = 0.05
  n_curve = 40     # 2.0 s
  n_straight = 60  # 3.0 s (enough for recovery timing checks)

  vtsc = mk_vtsc_with_params()
  steps = _steps_curve_then_straight(
    k_curve=k_curve,
    k_straight=0.0,
    confidence=0.95,
    n_curve=n_curve,
    n_straight=n_straight,
    lead_d_rel_m=None,
    curvature_ahead=None,
  )
  # This sweep is about VTSC's *cap output* (v_turn), not its internal accel plan.
  trace = simulate_sequence_trace(steps=steps, vtsc=vtsc, v0_mps=v0, v_cruise_mps=v_cruise, dt=dt, integrate_ego=True)
  assert len(trace) == (n_curve + n_straight)

  min_curve_v_turn = min(s['v_turn'] for s in trace[:n_curve])
  assert min_curve_v_turn <= v_cruise - 0.20, "Scenario did not exercise a meaningful VTSC cap during the curve"

  post = trace[n_curve:]
  assert post, "Missing post-curve recovery window"
  peak_step = max(max(0.0, nxt['v_turn'] - cur['v_turn']) for cur, nxt in zip(post, post[1:], strict=False))
  assert peak_step <= 0.25
  assert float(post[-1]['v_turn']) >= float(post[0]['v_turn']) + 3.0


@pytest.mark.parametrize("headway_s", [
  2.0,
  2.9,
  3.0,
  3.1,
  4.0,
])
def test_lead_headway_does_not_activate_dead_occlusion_paths(headway_s: float):
  # With occlusion removed, close-vs-far lead headway should not activate any FOV latch or
  # lead-bypass path. The cap should still unwind smoothly after the curve.
  dt = 0.05
  v0 = 25.0
  v_cruise = 30.0
  conf = 0.55
  k_curve = 0.012
  # Keep predicted curvature slightly non-zero on the straight so the UI state machine doesn't reset,
  # but also keep it small enough that it doesn't impose a genuine physics cap below v_cruise.
  k_ahead = 0.0025
  n_curve = 10
  n_straight = 60

  # Approximate constant headway by using distance = headway * v0; v_ego stays near v0 in this scenario.
  lead_d = float(headway_s * v0)

  vtsc = mk_vtsc_with_params()
  steps = _steps_curve_then_straight(
    k_curve=k_curve,
    k_straight=0.0,
    confidence=conf,
    n_curve=n_curve,
    n_straight=n_straight,
    lead_d_rel_m=lead_d,
    curvature_ahead=k_ahead,
  )
  trace = simulate_sequence_trace(steps=steps, vtsc=vtsc, v0_mps=v0, v_cruise_mps=v_cruise, dt=dt, integrate_ego=False)
  assert len(trace) == (n_curve + n_straight)

  assert all(not s['fov_occluded'] for s in trace)
  assert all(not s['occl_lead_bypass_active'] for s in trace)

  post = trace[n_curve:]
  assert post, "Missing post-curve recovery window"
  peak_step = max(max(0.0, nxt['v_turn'] - cur['v_turn']) for cur, nxt in zip(post, post[1:], strict=False))
  assert peak_step <= 0.25
  assert float(post[-1]['v_turn']) >= float(post[0]['v_turn']) + 3.0
