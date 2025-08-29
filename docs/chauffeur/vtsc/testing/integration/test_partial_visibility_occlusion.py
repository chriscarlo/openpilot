#!/usr/bin/env python3
"""
Partial-visibility occlusion test

Scenario:
- Gentle constant-radius curve whose physics target is ~65 mph (~29 m/s)
- Vision confidence is good for most of the approach, then goes bad (occluded)
  near the end due to mountainside/retaining wall blocking horizon.

Purpose:
- Verify no positive acceleration while occluded (monotonic invariant)
- Observe overslow budget on a gentle curve with late occlusion
- Confirm jerk caps
- Provide practical metrics to compare against road feel
"""

import sys
import os
import math
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, VTSCParams
from docs.chauffeur.vtsc.testing.harness.simulate import simulate
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed


class TestPartialVisibilityOcclusion(unittest.TestCase):
    def test_gentle_curve_late_occlusion(self):
        # Target ~65 mph (~29.06 m/s). Using allowed lat accel ~3.0 m/s^2 gives k ≈ 3 / v^2
        v_target_ms = 65.0 * 0.44704
        kappa = 3.0 / max(1e-6, v_target_ms ** 2)

        # Verify curvature_to_speed is near our target (sanity)
        vt = curvature_to_speed(kappa)
        print(f"Curvature {kappa:.6f} -> physics target {vt:.2f} m/s (~{vt*2.237:.1f} mph)")

        scn = Scenario(
            name='gentle_curve_late_occlusion',
            duration_s=10.0,
            dt=0.05,
            v0_mps=30.0,  # start just above 65 mph to let VTSC decel toward physics target
            geometry=GeometryProfile(kind='constant', kappa0=kappa),
            # Good vision initially; occluded in the final ~3.5s
            confidence=ConfidenceProfile(kind='window', value=0.5, window_start_s=6.5, window_end_s=10.0),
        )
        # Use standard VTSC params (defaults already set)
        scn.params = VTSCParams(aggressiveness=1.0, alpha=0.3, hysteresis=0.2, safety_bias=0.1)

        res = simulate(scn)
        m = res.metrics

        # Invariants
        self.assertLessEqual(m['pos_accel_while_occluded'], 1e-6, "No positive accel while occluded")
        self.assertLessEqual(m['jerk_pos'], 2.5, "Positive jerk within comfort cap")
        self.assertGreaterEqual(m['jerk_neg'], -6.5, "Negative jerk within system cap")

        # On a gentle curve with late occlusion, overslow should remain modest
        # (Expect small integral since target is high and occlusion is short)
        print("Metrics:", m)
        self.assertLessEqual(m['integrated_overslow'], 2.5, "Overslow reasonable for a gentle curve with late occlusion")

        # Helpful summary for visual inspection
        tail = slice(int(0.6 * len(res.t)), None)
        v_cmd_tail = (float(res.v_cmd[tail.start]), float(res.v_cmd[-1]))
        v_clean_tail = (float(res.v_clean[tail.start]), float(res.v_clean[-1]))
        print(f"v_cmd(tail): {v_cmd_tail[0]:.2f} -> {v_cmd_tail[1]:.2f} m/s,  v_clean(tail): {v_clean_tail[0]:.2f} -> {v_clean_tail[1]:.2f} m/s")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPartialVisibilityOcclusion)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)

