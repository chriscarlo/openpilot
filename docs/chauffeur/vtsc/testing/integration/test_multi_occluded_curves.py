#!/usr/bin/env python3
"""
Mountain highway style: three gentle curves with sustained occlusion.

Goal:
- Reproduce pronounced overslow when vision is mostly occluded across successive curves.
- Each curve’s physics allows ~65 mph; start at 70–75 mph.
- Confidence: first 2s visible, then sustained low confidence across the multi-curve sequence.
"""

import sys
import os
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, VTSCParams
from docs.chauffeur.vtsc.testing.harness.simulate import simulate
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed


class TestMultiOccludedCurves(unittest.TestCase):
    def test_three_occluded_curves(self):
        # Compute curvature for ~65 mph physics target
        v65 = 65.0 * 0.44704
        kappa = 3.0 / max(1e-6, v65**2)
        vt = curvature_to_speed(kappa)
        print(f"Physics target from curvature: {vt:.2f} m/s (~{vt*2.237:.1f} mph)")

        # Build three curve segments with short straights between them
        segments = [
            {"duration_s": 1.0, "kappa": 0.0},            # straight
            {"duration_s": 3.0, "kappa": kappa},          # curve 1
            {"duration_s": 0.8, "kappa": 0.0},            # short straight
            {"duration_s": 3.0, "kappa": kappa},          # curve 2
            {"duration_s": 0.6, "kappa": 0.0},            # short straight
            {"duration_s": 3.0, "kappa": kappa},          # curve 3
        ]

        scn = Scenario(
            name='three_occluded_curves',
            duration_s=12.0,
            dt=0.05,
            v0_mps=70.0*0.44704,  # start ~70 mph
            geometry=GeometryProfile(kind='multi_curve', segments=segments),
            # First 2s visible, then mostly occluded until near end
            confidence=ConfidenceProfile(kind='window', value=0.3, window_start_s=2.0, window_end_s=11.0),
        )
        scn.params = VTSCParams(aggressiveness=1.0, alpha=0.3, hysteresis=0.2, safety_bias=0.1)

        res = simulate(scn)
        m = res.metrics
        print("Metrics:", m)
        print("Occluded fraction:", float(np.mean(res.occluded)))

        # mph helpers
        v_cmd_mph = res.v_cmd * 2.237
        v_clean_mph = res.v_clean * 2.237

        # Min commanded during occlusion
        min_cmd_mph_occ = float(np.min(v_cmd_mph[res.occluded])) if np.any(res.occluded) else float(np.min(v_cmd_mph))
        print(f"Min commanded (occluded): {min_cmd_mph_occ:.1f} mph")
        print(f"Physics target (mph): {vt*2.237:.1f}")

        # No assertions beyond invariants here; we want to see reproduction details
        self.assertLessEqual(m['pos_accel_while_occluded'], 1e-6)
        # Require that we actually experienced occlusion for a substantial portion
        self.assertGreater(float(np.mean(res.occluded)), 0.3)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiOccludedCurves)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
