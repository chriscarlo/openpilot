#!/usr/bin/env python3
"""
FOV-based fairness readouts for high-value VTSC scenarios.

Assumptions:
- Flat road; camera aligned to tangent
- 60° FOV (±30° half-angle)
- Visibility s_vis_fov = (π/6)/|kappa|

Prints 500ms physics vs controller speeds for each scenario.
"""

import sys
import os
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, SpeedLimitProfile
from docs.chauffeur.vtsc.testing.harness.visibility import visibility_fov, visibility_walls, pos_margin_mask
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import COMFORT_DECEL_LIMIT
from docs.chauffeur.vtsc.testing.harness.simulate import simulate


class TestHighValueScenariosFOV(unittest.TestCase):
    def _readout(self, label: str, scn: Scenario):
        res = simulate(scn)
        conv = 2.2369362920544
        # FOV-only
        s_fov = visibility_fov(res.kappa)
        pm = pos_margin_mask(res.v_cmd, res.v_clean, s_fov, abs(float(COMFORT_DECEL_LIMIT)), 8.0)
        print(f"[{label}/FOV] pos_margin_coverage={float(np.mean(pm))*100:.1f}% | 500ms physics vs cmd:")
        print("time_s, physics_mph, controller_mph")
        step = int(round(0.5 / scn.dt))
        for i in range(0, len(res.t), step):
            ts = float(res.t[i])
            vp = float(res.v_clean[i]) * conv
            vc = float(res.v_cmd[i]) * conv
            print(f"{ts:5.2f}, {vp:7.2f}, {vc:7.2f}")
        # FOV+WALL
        s_vis = np.minimum(visibility_fov(res.kappa), visibility_walls(res.kappa))
        pm2 = pos_margin_mask(res.v_cmd, res.v_clean, s_vis, abs(float(COMFORT_DECEL_LIMIT)), 8.0)
        print(f"[{label}/FOV+WALL] pos_margin_coverage={float(np.mean(pm2))*100:.1f}% | 500ms physics vs cmd:")
        print("time_s, physics_mph, controller_mph")
        for i in range(0, len(res.t), step):
            ts = float(res.t[i])
            vp = float(res.v_clean[i]) * conv
            vc = float(res.v_cmd[i]) * conv
            print(f"{ts:5.2f}, {vp:7.2f}, {vc:7.2f}")

    def test_borderline_confidence_chatter_fov(self):
        scn = Scenario(
            name='borderline_chatter_fov', duration_s=8.0, dt=0.05, v0_mps=22.0,
            geometry=GeometryProfile(kind='constant', kappa0=0.004),
            confidence=ConfidenceProfile(kind='borderline_lpf', low=0.68, high=0.76, freq_hz=2.5),
        )
        self._readout('borderline_chatter', scn)
        self._readout('borderline_chatter+walls', scn)

    def test_late_apex_easing_exit_fov(self):
        scn = Scenario(
            name='late_apex_fov', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='easing', kappa0=0.002, kappa1=0.006),
            confidence=ConfidenceProfile(kind='window', value=0.6, window_start_s=3.0, window_end_s=5.0),
        )
        self._readout('late_apex', scn)
        self._readout('late_apex+walls', scn)

    def test_s_curve_inflection_fov(self):
        scn = Scenario(
            name='s_curve_fov', duration_s=9.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='s_curve', kappa0=0.004, kappa1=0.004, mid_straight_s=8.0),
            confidence=ConfidenceProfile(kind='borderline_lpf', low=0.68, high=0.76, freq_hz=2.5),
        )
        self._readout('s_curve', scn)
        self._readout('s_curve+walls', scn)

    def test_speed_limit_step_under_occlusion_fov(self):
        scn = Scenario(
            name='limit_step_fov', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='constant', kappa0=0.004),
            confidence=ConfidenceProfile(kind='window', value=0.5, window_start_s=1.0, window_end_s=3.0),
            speed_limit=SpeedLimitProfile(kind='step', start_mps=29.0, step_time_s=2.0, step_to_mps=20.0),
        )
        self._readout('limit_step', scn)
        self._readout('limit_step+walls', scn)

    def test_pipeline_latency_injection_fov(self):
        scn = Scenario(
            name='latency_fov', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='tightening', kappa0=0.002, kappa1=0.006),
            confidence=ConfidenceProfile(kind='stable', value=0.9),
        )
        self._readout('latency', scn)
        self._readout('latency+walls', scn)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHighValueScenariosFOV)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
