#!/usr/bin/env python3
"""
Scenario: 50 mph straight into an abrupt hidden turn (≈25 mph safe speed).

The bend veers outside the front camera's ~60° FOV quickly (modeled as persistent
low confidence/occlusion). We verify occlusion invariants and observe that the
controller decelerates conservatively toward the physics bound without positive
acceleration under occlusion.
"""

import sys
import os
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile
from docs.chauffeur.vtsc.testing.harness.simulate import simulate
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed
from opendbc.car.common.conversions import Conversions as CV


def _kappa_for_target_speed_mps(v_target_mps: float) -> float:
    """Find curvature k such that curvature_to_speed(k) ≈ v_target_mps via bisection."""
    # Reasonable bounds for road curvature in 1/m
    lo, hi = 1e-5, 0.05
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        v_mid = curvature_to_speed(mid)
        if v_mid > v_target_mps:
            # Need tighter curve (higher k) to reduce speed
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class TestAbruptHiddenTurn(unittest.TestCase):
    def test_abrupt_hidden_turn_outside_fov(self):
        v0_mph = 50.0
        v0_mps = v0_mph * CV.MPH_TO_MS
        v_safe_mph = 25.0
        v_safe_mps = v_safe_mph * CV.MPH_TO_MS

        # Compute curvature for ~25 mph safe speed using current physics mapping
        k_tight = _kappa_for_target_speed_mps(v_safe_mps)
        # Convert to radius for geometry segments
        r_tight = 1.0 / max(1e-6, k_tight)

        # Build piecewise geometry: straight-in, then abrupt tight bend
        segments = []
        segments.append({"duration_s": 2.0, "radius_m": 1e9, "sign": 1.0})  # straight
        segments.append({"duration_s": 6.0, "radius_m": r_tight, "sign": 1.0})  # tight bend

        scn = Scenario(
            name='abrupt_hidden_turn',
            duration_s=sum(s['duration_s'] for s in segments) + 1.0,
            dt=0.05,
            v0_mps=v0_mps,
            geometry=GeometryProfile(kind='multi_curve', segments=segments),
            # Persistently low confidence to model turn outside FoV
            confidence=ConfidenceProfile(kind='stable', value=0.3),
        )
        # Production-like barrier parameters, but make visibility conservative
        scn.vis_horizon_s = 1.0
        scn.vis_margin_m = 12.0
        scn.gamma_per_meter = 0.00020
        scn.lat_jerk_cap = 2.0

        res = simulate(scn)
        m = res.metrics

        # Diagnostics
        min_cmd_mps = float(np.min(res.v_cmd))
        min_clean_mps = float(np.min(res.v_clean))
        print(f"Abrupt-hidden-turn: v0={v0_mph:.1f}mph, v_safe≈{v_safe_mph:.1f}mph, "
              f"min_cmd={min_cmd_mps*CV.MS_TO_MPH:.1f}mph, min_clean={min_clean_mps*CV.MS_TO_MPH:.1f}mph")

        # Jerk caps under occlusion
        self.assertLessEqual(m['jerk_pos'], 2.6)
        self.assertGreaterEqual(m['jerk_neg'], -6.5)

        # Expect meaningful deceleration within 4s after bend start (2.0→6.0s)
        bend_start_idx = int(round(2.0 / scn.dt))
        check_idx = min(len(res.t) - 1, bend_start_idx + int(round(4.0 / scn.dt)))
        self.assertLessEqual(res.v_cmd[check_idx], 20.0, "Should reduce below ~45 mph within ~4s of bend")

        # By end of scenario, commanded speed should be close to physics bound
        self.assertLessEqual(res.v_cmd[-1] - res.v_clean[-1], 2.0, "Final commanded near physics bound (≤ ~4.5 mph)")

        # Around bend entry (2.0→4.0s window), ensure no positive accel while occluded
        w0 = bend_start_idx
        w1 = min(len(res.t) - 1, bend_start_idx + int(round(2.0 / scn.dt)))
        a_win = np.array(res.a_cmd[w0:w1])
        occ_win = np.array(res.occluded[w0:w1])
        if a_win.size > 0:
            max_pos_accel = float(np.max(np.maximum(a_win[occ_win], 0.0))) if np.any(occ_win) else 0.0
            self.assertLessEqual(max_pos_accel, 0.02, "No positive accel near bend while occluded")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAbruptHiddenTurn)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
