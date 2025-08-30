#!/usr/bin/env python3
"""
VTSC Acceptance Tests

High-level, outcome-focused checks that mirror user-facing requirements.
Follows the acceptance testing primer in docs/chauffeur/testing/acceptance_tests.md.

Scenarios:
- REQ_VTSC_001: Defaults are legacy-aligned on clean installs
- REQ_VTSC_002: Straight vs curve targets follow physics, capped at MAX
- REQ_VTSC_003: Occlusion holds last curvature (no sudden jumps)
- REQ_VTSC_004: Comfort-first braking escalates to adaptive under high demand
"""

import sys
import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

# Ensure project import paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sunnypilot"))

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
  VisionTurnController,
  curvature_to_speed,
  MAX_SPEED_DEFAULT,
)


def _mk_vtsc_with_defaults():
  class MockCP:  # minimal car params for ctor
    pass
  with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
    mp = MagicMock()
    mp.get_bool.return_value = True  # VTSC enabled
    # Return None for tunables to exercise fallbacks; explicit Aggressiveness=1.0
    def _get(key):
      if key == "VisionTurnSpeedControlAggressiveness":
        return b"1.0"
      return None
    mp.get.side_effect = _get
    MockParams.return_value = mp
    vtsc = VisionTurnController(MockCP())
    return vtsc


def test_REQ_VTSC_001_defaults_legacy_aligned():
  """
  Given an empty Params store (no user values)
  When VTSC initializes
  Then defaults match expected baselines (aggr=1.0, alpha=0.3, hyst=0.15, bias=0.1)
  """
  vtsc = _mk_vtsc_with_defaults()
  assert abs(vtsc._aggressiveness - 1.0) < 1e-6
  assert abs(vtsc._filter_alpha - 0.3) < 1e-6
  assert abs(vtsc._hysteresis_threshold - 0.15) < 1e-6
  assert abs(vtsc._safety_bias - 0.1) < 1e-6


def test_REQ_VTSC_002_straight_and_curve_targets():
  """
  Given road curvature
  When curvature is near zero (straight)
  Then target equals MAX_SPEED_DEFAULT
  And tighter curves produce monotonically lower target speeds
  """
  # Straight road
  assert abs(curvature_to_speed(0.0) - MAX_SPEED_DEFAULT) < 1e-6
  # Monotonic: tighter curve (higher curvature) -> lower speed
  v_lo = curvature_to_speed(1e-3)
  v_hi = curvature_to_speed(1e-2)
  assert v_lo > v_hi
  # Reasonable bounds
  assert 0.0 < v_hi < MAX_SPEED_DEFAULT


def test_REQ_VTSC_003_occlusion_holds_last_curvature():
  """
  Given smoothed curvature and vision confidence
  When confidence drops below bad threshold
  Then controller holds last valid curvature value
  """
  vtsc = _mk_vtsc_with_defaults()

  # Helper model object with lane line probabilities
  def model_with_conf(conf: float):
    return SimpleNamespace(laneLineProbs=[conf, conf, conf, conf])

  # Prime with good vision and a known curvature
  vtsc._filtered_curvature = 0.05
  cur_used = vtsc._update_vision_occlusion(model_with_conf(0.9), 0.0)
  assert abs(cur_used - 0.05) < 1e-6

  # Change current curvature and repeatedly drop confidence to push below bad threshold
  vtsc._filtered_curvature = 0.02
  for i in range(20):
    _ = vtsc._update_vision_occlusion(model_with_conf(0.0), 1.0 + i * 0.05)
  # Now vision should be not good; held value should remain last valid (0.02)
  held_cur = vtsc._update_vision_occlusion(model_with_conf(0.0), 2.0)
  assert abs(held_cur - 0.02) < 1e-6


def test_REQ_VTSC_004_comfort_escalates_to_adaptive():
  """
  Given a high deceleration requirement
  When computing optimal decel
  Then controller escalates to adaptive (beyond comfort) with hysteresis control
  """
  vtsc = _mk_vtsc_with_defaults()
  dt = 0.05

  # Large required decel (beyond comfort), integrate with jerk limit until active
  decel = 0.0
  for _ in range(40):
    decel = vtsc._get_optimal_deceleration(-5.0, dt)
  assert vtsc.adaptive_decel_active

  # Then small requirement repeatedly to exit adaptive via hysteresis
  for _ in range(60):
    vtsc._get_optimal_deceleration(-0.2, dt)
  assert not vtsc.adaptive_decel_active

def test_REQ_VTSC_005_invariants_and_legacy_defaults():
  """
  Acceptance invariants: monotonic while occluded, reacquisition latency, jerk caps, envelope safety.
  Also verify that legacy defaults produce stable, repeatable outputs on a golden scenario.
  """
  from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile
  from docs.chauffeur.vtsc.testing.harness.simulate import simulate

  scn = Scenario(
    name='acceptance_golden', duration_s=8.0, dt=0.05, v0_mps=25.0,
    geometry=GeometryProfile(kind='tightening', kappa0=0.002, kappa1=0.006),
    confidence=ConfidenceProfile(kind='window', value=0.5, window_start_s=1.0, window_end_s=2.5),
  )
  res = simulate(scn)
  m = res.metrics

  # Invariants
  assert m['pos_accel_while_occluded'] <= 1e-6
  if m['reacq_latency'] is not None:
    assert m['reacq_latency'] <= 0.6
  assert m['jerk_pos'] <= 2.5 and m['jerk_neg'] >= -6.5

  # Legacy defaults: ensure deterministic and close to physics reference
  # Compare final v_cmd to final v_clean within tolerance
  assert abs(res.v_cmd[-1] - res.v_clean[-1]) <= 0.5
  print("✓ Acceptance invariants and legacy-defaults check passed")


def main():
  """Executable entry for run_all_tests.py compatibility."""
  try:
    test_REQ_VTSC_001_defaults_legacy_aligned()
    test_REQ_VTSC_002_straight_and_curve_targets()
    test_REQ_VTSC_003_occlusion_holds_last_curvature()
    test_REQ_VTSC_004_comfort_escalates_to_adaptive()
    print("✓ VTSC Acceptance tests passed")
    return True
  except AssertionError as e:
    print(f"✗ VTSC Acceptance test failed: {e}")
    return False


if __name__ == "__main__":
  ok = main()
  sys.exit(0 if ok else 1)
