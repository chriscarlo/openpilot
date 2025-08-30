#!/usr/bin/env python3
"""
FOV-based fairness readout for realistic hidden turn.

Assumptions:
- Flat road; camera aligned to tangent
- 60° FOV (±30°)
- Prints 500ms physics vs controller speeds.
"""

import sys
import os
import unittest
import numpy as np
from opendbc.car.common.conversions import Conversions as CV

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, SpeedLimitProfile
from docs.chauffeur.vtsc.testing.harness.visibility import visibility_fov, visibility_walls, pos_margin_mask
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import COMFORT_DECEL_LIMIT
from docs.chauffeur.vtsc.testing.harness.simulate import simulate
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed


def _kappa_for_target_speed_mps(v_target_mps: float) -> float:
    lo, hi = 1e-5, 0.05
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v_mid = curvature_to_speed(mid)
        if v_mid > v_target_mps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class TestAbruptHiddenTurnFOV(unittest.TestCase):
    def test_hidden_turn_readout_fov(self):
        v0_mph = 50.0
        v0_mps = v0_mph * CV.MPH_TO_MS
        v_safe_mps = 25.0 * CV.MPH_TO_MS
        k_tight = _kappa_for_target_speed_mps(v_safe_mps)
        r_tight = 1.0 / max(1e-6, k_tight)

        straight_s = 2.0
        bend_s = 6.0
        segments = [
            {"duration_s": straight_s, "radius_m": 1e9, "sign": 1.0},
            {"duration_s": bend_s,     "radius_m": r_tight, "sign": 1.0},
        ]
        scn = Scenario(
            name='abrupt_hidden_turn_fov',
            duration_s=straight_s + bend_s + 1.0,
            dt=0.05,
            v0_mps=v0_mps,
            geometry=GeometryProfile(kind='multi_curve', segments=segments),
            confidence=ConfidenceProfile(kind='window', value=0.3, window_start_s=3.0, window_end_s=straight_s + bend_s + 1.0),
            speed_limit=SpeedLimitProfile(kind='none', start_mps=v0_mps),
        )
        res = simulate(scn)

        conv = 2.2369362920544
        # FOV-only coverage
        s_fov = visibility_fov(res.kappa)
        pm = pos_margin_mask(res.v_cmd, res.v_clean, s_fov, abs(float(COMFORT_DECEL_LIMIT)), 8.0)
        print(f"[hidden_turn/FOV] pos_margin_coverage={float(np.mean(pm))*100:.1f}% | 500ms physics vs cmd:")
        print("time_s, physics_mph, controller_mph")
        step = int(round(0.5 / scn.dt))
        for i in range(0, len(res.t), step):
            ts = float(res.t[i])
            vp = float(res.v_clean[i]) * conv
            vc = float(res.v_cmd[i]) * conv
            print(f"{ts:5.2f}, {vp:7.2f}, {vc:7.2f}")
        # FOV+WALL coverage
        s_vis = np.minimum(visibility_fov(res.kappa), visibility_walls(res.kappa))
        pm2 = pos_margin_mask(res.v_cmd, res.v_clean, s_vis, abs(float(COMFORT_DECEL_LIMIT)), 8.0)
        print(f"[hidden_turn/FOV+WALL] pos_margin_coverage={float(np.mean(pm2))*100:.1f}% | 500ms physics vs cmd:")
        print("time_s, physics_mph, controller_mph")
        for i in range(0, len(res.t), step):
            ts = float(res.t[i])
            vp = float(res.v_clean[i]) * conv
            vc = float(res.v_cmd[i]) * conv
            print(f"{ts:5.2f}, {vp:7.2f}, {vc:7.2f}")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAbruptHiddenTurnFOV)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
