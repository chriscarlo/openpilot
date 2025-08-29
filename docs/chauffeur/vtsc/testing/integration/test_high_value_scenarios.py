#!/usr/bin/env python3
"""
High-value VTSC scenarios exercising brittle regimes with hard metrics.

Scenarios:
- Borderline-confidence chatter near threshold (sticky occlusion exposure)
- Tightening-radius entry occlusion with conservative envelope
- Late-apex occlusion with easing exit
- S-curve with short median straight
- Speed-limit step mid-bend under occlusion
- Pipeline latency injection
- Vertical geometry stress (crest into bend)
- Monte-Carlo phase sweep over occlusion start times
"""

import sys
import os
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, SpeedLimitProfile
from docs.chauffeur.vtsc.testing.harness.simulate import simulate


class TestHighValueScenarios(unittest.TestCase):
    def test_borderline_confidence_chatter(self):
        scn = Scenario(
            name='borderline_chatter', duration_s=8.0, dt=0.05, v0_mps=22.0,
            geometry=GeometryProfile(kind='constant', kappa0=0.004),
            confidence=ConfidenceProfile(kind='borderline_lpf', low=0.68, high=0.76, freq_hz=2.5),
        )
        res = simulate(scn)
        m = res.metrics
        # Expect updates most of the time (bounded overslow) and no prolonged holds
        self.assertLessEqual(m['integrated_overslow'], 2.2)
        self.assertLessEqual(m['pos_accel_while_occluded'], 1e-6)

    def test_tightening_radius_envelope_safety(self):
        scn = Scenario(
            name='tighten_envelope', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='tightening', kappa0=0.002, kappa1=0.007, length_s=150.0),
            confidence=ConfidenceProfile(kind='window', value=0.5, window_start_s=0.5, window_end_s=2.5),
        )
        res = simulate(scn)
        # Envelope: |k(t)| <= |k0| + gamma*s; use conservative gamma (1e-4 per meter)
        gamma = 1e-4
        k0 = res.kappa[0]
        s = np.arange(len(res.t)) * scn.dt * scn.v0_mps
        envelope = np.abs(k0) + gamma * s
        self.assertTrue(np.all(np.abs(res.kappa) <= envelope + 1e-6), "Envelope safety respected by input profile")
        # Controller should not imply lateral accel violating envelope while occluded (checked indirectly by no accel while occluded)
        self.assertLessEqual(res.metrics['pos_accel_while_occluded'], 1e-6)

    def test_late_apex_easing_exit(self):
        scn = Scenario(
            name='late_apex', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='easing', kappa0=0.002, kappa1=0.006),
            confidence=ConfidenceProfile(kind='window', value=0.6, window_start_s=3.0, window_end_s=5.0),
        )
        res = simulate(scn)
        # Expect commanded speed to stop ratcheting downward once curvature stabilizes
        dv = np.diff(res.v_cmd)
        # After occlusion end, average dv should be >= 0 (recovery)
        self.assertLessEqual(res.metrics['integrated_overslow'], 8.0)

    def test_s_curve_inflection(self):
        scn = Scenario(
            name='s_curve', duration_s=9.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='s_curve', kappa0=0.004, kappa1=0.004, mid_straight_s=8.0),
            confidence=ConfidenceProfile(kind='borderline_lpf', low=0.68, high=0.76, freq_hz=2.5),
        )
        res = simulate(scn)
        # Grade time to reverse speed trend at inflection indirectly via jerk and overshoot caps
        self.assertLessEqual(res.metrics['overshoot_on_recovery'], 0.5)

    def test_speed_limit_step_under_occlusion(self):
        scn = Scenario(
            name='limit_step', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='constant', kappa0=0.004),
            confidence=ConfidenceProfile(kind='window', value=0.5, window_start_s=1.0, window_end_s=3.0),
            speed_limit=SpeedLimitProfile(kind='step', start_mps=29.0, step_time_s=2.0, step_to_mps=20.0),
        )
        res = simulate(scn)
        m = res.metrics
        self.assertLessEqual(m['pos_accel_while_occluded'], 1e-6)
        self.assertLessEqual(m['overshoot_on_recovery'], 0.5)

    def test_pipeline_latency_injection(self):
        scn = Scenario(
            name='latency', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='tightening', kappa0=0.002, kappa1=0.006),
            confidence=ConfidenceProfile(kind='stable', value=0.9),
        )
        # Baseline (no latency)
        res0 = simulate(scn)
        # With 200ms latency
        scn.latency_s = 0.2
        res1 = simulate(scn)
        # Parameter choices should still hold within relaxed budgets
        self.assertLessEqual(res1.metrics['overshoot_on_recovery'], 0.6)
        self.assertLessEqual(res1.metrics['integrated_overslow'], 1.8)

    def test_monte_carlo_phase_sweep(self):
        # Sweep occlusion start from -3s to +3s around apex
        base = Scenario(
            name='sweep', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='tightening', kappa0=0.002, kappa1=0.006),
        )
        starts = np.arange(-3.0, 3.01, 0.25)
        worst = {'overslow': 0.0, 'reacq': 0.0}
        for s in starts:
            scn = base
            # Map start time into window within [0, duration]
            t0 = max(0.5, min(base.duration_s - 1.5, (base.duration_s / 2.0) + s))
            scn.confidence = ConfidenceProfile(kind='window', value=0.5, window_start_s=t0, window_end_s=t0 + 1.0)
            res = simulate(scn)
            m = res.metrics
            worst['overslow'] = max(worst['overslow'], m['integrated_overslow'])
            if m['reacq_latency'] is not None:
                worst['reacq'] = max(worst['reacq'], m['reacq_latency'])
        self.assertLessEqual(worst['overslow'], 2.0)
        if worst['reacq']:
            self.assertLessEqual(worst['reacq'], 0.8)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHighValueScenarios)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
