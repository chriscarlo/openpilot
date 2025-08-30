#!/usr/bin/env python3
"""
Realistic hidden turn: 50 mph straight, then constant-radius bend requiring ~25 mph
safe speed. Confidence stays high until the bend rotates near the 60° FOV edge,
then drops low. This gives VTSC a fair, real-world lead-in before occlusion.
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
    lo, hi = 1e-5, 0.05
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v_mid = curvature_to_speed(mid)
        if v_mid > v_target_mps:
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

        # Curvature corresponding to ~25 mph safe physics speed
        k_tight = _kappa_for_target_speed_mps(v_safe_mps)
        r_tight = 1.0 / max(1e-6, k_tight)

        # Geometry: straight-in 2.0 s, then constant-radius bend
        straight_s = 2.0
        bend_s = 6.0
        segments = [
            {"duration_s": straight_s, "radius_m": 1e9, "sign": 1.0},
            {"duration_s": bend_s,     "radius_m": r_tight, "sign": 1.0},
        ]

        # Time to rotate ~30° (=0.524 rad) inside the bend
        delta_psi = 0.524
        s_to_30deg = delta_psi / max(1e-6, k_tight)
        t_to_30deg = s_to_30deg / max(1e-6, v0_mps)
        t_bend = straight_s
        # Begin occlusion when tangent nears the FOV edge
        t_occ_start = t_bend + t_to_30deg
        # Window the low confidence to persist from that point onward
        t_occ_end = straight_s + bend_s + 1.0

        scn = Scenario(
            name='abrupt_hidden_turn_realistic',
            duration_s=t_occ_end,
            dt=0.05,
            v0_mps=v0_mps,
            geometry=GeometryProfile(kind='multi_curve', segments=segments),
            # High confidence initially; drop to low once bend reaches the FOV edge
            confidence=ConfidenceProfile(kind='window', value=0.3, window_start_s=t_occ_start, window_end_s=t_occ_end),
        )
        # Barrier parameters
        scn.vis_horizon_s = 1.4
        scn.vis_margin_m = 12.0
        scn.gamma_per_meter = 0.00020
        scn.lat_jerk_cap = 2.0

        res = simulate(scn)
        m = res.metrics

        # Diagnostics
        min_cmd_mps = float(np.min(res.v_cmd))
        min_clean_mps = float(np.min(res.v_clean))
        print(f"Hidden-turn realistic: v0={v0_mph:.1f}mph, v_safe≈{v_safe_mph:.1f}mph, "
              f"min_cmd={min_cmd_mps*CV.MS_TO_MPH:.1f}mph, min_clean={min_clean_mps*CV.MS_TO_MPH:.1f}mph, "
              f"t_occ_start={t_occ_start:.2f}s")

        # Jerk caps within bounds
        self.assertLessEqual(m['jerk_pos'], 2.6)
        self.assertGreaterEqual(m['jerk_neg'], -6.5)

        # Fair deceleration expectation: within ~3.0 s after occlusion onset, we should see
        # a meaningful reduction (>= 2.0 m/s). This avoids penalizing the pre-occlusion lead-in.
        occ_idx = int(round(t_occ_start / scn.dt))
        chk_idx = min(len(res.t) - 1, occ_idx + int(round(3.0 / scn.dt)))
        self.assertLessEqual(res.v_cmd[chk_idx], v0_mps - 2.0, "Reduce ≥ 2 m/s within ~3 s after occlusion start")

        # End-of-scenario closeness to physics bound (≤ ~3 m/s)
        self.assertLessEqual(res.v_cmd[-1] - res.v_clean[-1], 3.0, "Final commanded near physics bound (≤ ~6.7 mph)")

        # No positive accel while occluded near the onset (first ~1.5 s after occlusion)
        w0 = occ_idx
        w1 = min(len(res.t) - 1, occ_idx + int(round(1.5 / scn.dt)))
        a_win = np.array(res.a_cmd[w0:w1])
        occ_win = np.array(res.occluded[w0:w1])
        if a_win.size > 0 and np.any(occ_win):
            max_pos_accel = float(np.max(np.maximum(a_win[occ_win], 0.0)))
            self.assertLessEqual(max_pos_accel, 0.02, "No positive accel just after occlusion onset")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAbruptHiddenTurn)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
