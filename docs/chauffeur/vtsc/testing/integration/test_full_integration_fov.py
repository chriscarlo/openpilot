#!/usr/bin/env python3
"""
FOV-based fairness version of selected full-integration scenarios.

Assumptions:
- Flat road; camera aligned to tangent
- 60° FOV (±30° half-angle)
- Visibility distance s_vis_fov = (π/6)/|kappa|

Prints 500ms physics vs controller speed for each scenario.
"""

import sys
import os
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile
from docs.chauffeur.vtsc.testing.harness.visibility import visibility_fov, visibility_walls, pos_margin_mask
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import COMFORT_DECEL_LIMIT
from docs.chauffeur.vtsc.testing.harness.simulate import simulate


class TestFullIntegrationFOV(unittest.TestCase):
    FOV_HALF_RAD = np.pi / 6.0
    LANE_WIDTH_M = 11.0 * 0.3048
    SHOULDER_WIDTH_M = 8.0 * 0.3048

    def _print_readout(self, name: str, scn: Scenario):
        res = simulate(scn)
        conv = 2.2369362920544
        # FOV-only margin coverage summary
        s_fov = visibility_fov(res.kappa, self.FOV_HALF_RAD)
        pm = pos_margin_mask(res.v_cmd, res.v_clean, s_fov, abs(float(COMFORT_DECEL_LIMIT)), 8.0)
        print(f"[{name}/FOV] pos_margin_coverage={float(np.mean(pm))*100:.1f}% | 500ms physics vs cmd:")
        print("time_s, physics_mph, controller_mph")
        step = int(round(0.5 / scn.dt))
        for i in range(0, len(res.t), step):
            ts = float(res.t[i])
            vp = float(res.v_clean[i]) * conv
            vc = float(res.v_cmd[i]) * conv
            print(f"{ts:5.2f}, {vp:7.2f}, {vc:7.2f}")

    def _print_readout_with_walls(self, name: str, scn: Scenario):
        # Shared readout; horizon calc used for reporting only
        res = simulate(scn)
        conv = 2.2369362920544
        # FOV+WALL margin coverage summary
        s_vis = np.minimum(visibility_fov(res.kappa, self.FOV_HALF_RAD),
                           visibility_walls(res.kappa, self.LANE_WIDTH_M, self.SHOULDER_WIDTH_M, True))
        pm = pos_margin_mask(res.v_cmd, res.v_clean, s_vis, abs(float(COMFORT_DECEL_LIMIT)), 8.0)
        print(f"[{name}/FOV+WALL] pos_margin_coverage={float(np.mean(pm))*100:.1f}% | 500ms physics vs cmd:")
        print("time_s, physics_mph, controller_mph")
        step = int(round(0.5 / scn.dt))
        for i in range(0, len(res.t), step):
            ts = float(res.t[i])
            vp = float(res.v_clean[i]) * conv
            vc = float(res.v_cmd[i]) * conv
            print(f"{ts:5.2f}, {vp:7.2f}, {vc:7.2f}")

    def test_tightening_curve_fov(self):
        scn = Scenario(
            name='tightening_fov', duration_s=10.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='tightening', kappa0=0.002, kappa1=0.006),
            confidence=ConfidenceProfile(kind='stable', value=0.3),
        )
        self._print_readout('tightening', scn)
        self._print_readout_with_walls('tightening', scn)

    def test_easing_curve_fov(self):
        scn = Scenario(
            name='easing_fov', duration_s=10.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='easing', kappa0=0.002, kappa1=0.006),
            confidence=ConfidenceProfile(kind='stable', value=0.3),
        )
        self._print_readout('easing', scn)
        self._print_readout_with_walls('easing', scn)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFullIntegrationFOV)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
