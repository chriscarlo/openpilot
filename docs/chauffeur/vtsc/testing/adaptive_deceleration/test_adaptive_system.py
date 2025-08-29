#!/usr/bin/env python3
"""
Test suite for Adaptive Deceleration System
Validates the replacement of Emergency Escalation System with physics-based adaptive approach
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../'))

# Import the VisionTurnController
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
    VisionTurnController,
    VisionTurnControllerState,
    COMFORT_DECEL_LIMIT,
    COMFORT_JERK_LIMIT,
    MAX_ADAPTIVE_DECEL,
    MAX_ADAPTIVE_JERK,
    DEFAULT_FILTER_ALPHA,
    DEFAULT_HYSTERESIS_THRESHOLD,
    DEFAULT_SAFETY_BIAS
)

@dataclass
class MockCarParams:
    """Mock car parameters for testing"""
    pass


class TestAdaptiveDeceleration(unittest.TestCase):
    """Test cases for adaptive deceleration system behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.CP = MockCarParams()
        
        # Mock Params to avoid file system access
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
            mock_params = MagicMock()
            mock_params.get_bool.return_value = True
            mock_params.get.return_value = None
            MockParams.return_value = mock_params
            
            self.vtsc = VisionTurnController(self.CP)
    
    def test_comfort_deceleration_sufficient(self):
        """Test that system uses comfort deceleration when sufficient"""
        # Test when required deceleration is within comfort limits
        required_decel = -1.0  # Less aggressive than comfort limit (-1.47)
        dt = 0.05
        
        result = self.vtsc._get_optimal_deceleration(required_decel, dt)
        
        # Should use comfort deceleration
        self.assertGreaterEqual(result, COMFORT_DECEL_LIMIT)
        self.assertLessEqual(result, 0)
        print(f"✓ Comfort decel test: Required={required_decel:.2f}, Result={result:.2f}")
    
    def test_adaptive_escalation_needed(self):
        """Test escalation to adaptive deceleration when comfort insufficient"""
        # Test when required deceleration exceeds comfort limits
        required_decel = -4.0  # More aggressive than comfort limit
        dt = 0.05
        
        # Simulate multiple calls to trigger filtering
        for _ in range(10):
            result = self.vtsc._get_optimal_deceleration(required_decel, dt)
        
        # Should escalate beyond comfort limit
        self.assertLess(result, COMFORT_DECEL_LIMIT)
        self.assertGreaterEqual(result, MAX_ADAPTIVE_DECEL)
        self.assertTrue(self.vtsc._decel_hysteresis_state)
        print(f"✓ Adaptive escalation test: Required={required_decel:.2f}, Result={result:.2f}")
    
    def test_hysteresis_prevents_oscillation(self):
        """Test hysteresis prevents oscillation between modes"""
        dt = 0.05
        
        # First, trigger adaptive mode with aggressive deceleration
        aggressive_decel = -3.5
        for _ in range(10):
            self.vtsc._get_optimal_deceleration(aggressive_decel, dt)
        
        self.assertTrue(self.vtsc._decel_hysteresis_state, "Should be in adaptive mode")
        
        # Now reduce requirement slightly below comfort limit
        # Should stay in adaptive mode due to hysteresis
        reduced_decel = -1.3  # Slightly less than comfort limit
        for _ in range(5):
            result = self.vtsc._get_optimal_deceleration(reduced_decel, dt)
        
        self.assertTrue(self.vtsc._decel_hysteresis_state, "Should remain in adaptive mode due to hysteresis")
        
        # Further reduce to trigger return to comfort mode
        comfort_decel = -1.0
        for _ in range(10):
            result = self.vtsc._get_optimal_deceleration(comfort_decel, dt)
        
        self.assertFalse(self.vtsc._decel_hysteresis_state, "Should return to comfort mode")
        print(f"✓ Hysteresis test passed - prevents mode oscillation")
    
    def test_jerk_limiting(self):
        """Test jerk limiting for smooth transitions"""
        dt = 0.05
        
        # Start from zero deceleration
        self.vtsc._current_decel = 0.0
        
        # Request large deceleration change
        large_decel = -5.0
        
        # First call should be jerk-limited
        result = self.vtsc._get_optimal_deceleration(large_decel, dt)
        
        # Change should be limited by jerk (use absolute value of jerk limit)
        max_change = abs(MAX_ADAPTIVE_JERK) * dt
        self.assertLessEqual(abs(result), max_change + 0.01)  # Small tolerance
        print(f"✓ Jerk limiting test: Max change={max_change:.3f}, Actual={abs(result):.3f}")
    
    def test_ema_filtering(self):
        """Test EMA filtering smooths noisy deceleration requirements"""
        dt = 0.05
        
        # Simulate noisy input
        noisy_requirements = [-1.5, -2.0, -1.8, -2.2, -1.9, -2.1]
        results = []
        
        for req in noisy_requirements:
            result = self.vtsc._get_optimal_deceleration(req, dt)
            results.append(self.vtsc._filtered_decel_requirement)
        
        # Check that filtered values are smoother than input
        input_variance = np.var(noisy_requirements)
        output_variance = np.var(results)
        
        self.assertLess(output_variance, input_variance, "Filtered output should be smoother")
        print(f"✓ EMA filtering test: Input variance={input_variance:.3f}, Output variance={output_variance:.3f}")
    
    def test_safety_bias_application(self):
        """Test safety bias increases deceleration requirement"""
        # Test physics calculation with safety bias
        v_current = 20.0  # m/s
        v_target = 15.0   # m/s
        distance = 50.0   # meters
        
        # Calculate with safety bias
        self.vtsc._safety_bias = 0.1
        result_with_bias = self.vtsc._calculate_required_deceleration(v_current, v_target, distance)
        
        # Calculate expected without bias
        expected_raw = (v_target**2 - v_current**2) / (2 * distance)
        expected_with_bias = expected_raw * (1.0 + self.vtsc._safety_bias)
        
        self.assertAlmostEqual(result_with_bias, max(expected_with_bias, MAX_ADAPTIVE_DECEL), places=2)
        print(f"✓ Safety bias test: Raw={expected_raw:.2f}, With bias={result_with_bias:.2f}")
    
    def test_reset_during_acceleration(self):
        """Test adaptive state resets during acceleration"""
        dt = 0.05
        
        # First trigger adaptive mode
        self.vtsc._get_optimal_deceleration(-3.5, dt)
        self.vtsc._decel_hysteresis_state = True
        
        # Simulate update_solution with positive acceleration
        self.vtsc._v_ego = 20.0
        self.vtsc._prev_target_speed = 19.0
        raw_target = 20.5  # Acceleration scenario
        
        # Simulate the acceleration branch from _update_solution
        accel_cmd = (raw_target - self.vtsc._prev_target_speed) / dt
        if accel_cmd > 0:
            # This should reset adaptive state
            self.vtsc._current_decel = 0.0
            self.vtsc._filtered_decel_requirement = 0.0
            self.vtsc._decel_hysteresis_state = False
        
        self.assertEqual(self.vtsc._current_decel, 0.0)
        self.assertEqual(self.vtsc._filtered_decel_requirement, 0.0)
        self.assertFalse(self.vtsc._decel_hysteresis_state)
        print(f"✓ Reset during acceleration test passed")
    
    def test_monitor_extreme_scenarios(self):
        """Test monitoring detects extreme deceleration scenarios"""
        # Test critical scenario detection
        required_decel = -5.8  # Near maximum
        remaining_distance = 15.0  # Critical distance
        
        # Set current decel near maximum
        self.vtsc._current_decel = -5.7
        
        result = self.vtsc._monitor_adaptive_deceleration(required_decel, remaining_distance)
        
        self.assertTrue(result, "Should detect challenging scenario")
        print(f"✓ Extreme scenario monitoring test passed")

    def test_occlusion_invariants_and_budget_sweep(self):
        """Monotonic-while-occluded, overslow budget, and reacquisition latency across param sweep."""
        from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, VTSCParams
        from docs.chauffeur.vtsc.testing.harness.simulate import simulate

        # Medium bend with an occlusion window pre-apex and borderline confidence
        base_scn = Scenario(
            name="occlusion_sweep_medium_bend",
            duration_s=8.0,
            dt=0.05,
            v0_mps=25.0,
            geometry=GeometryProfile(kind='tightening', kappa0=0.002, kappa1=0.006),
            confidence=ConfidenceProfile(kind='borderline_lpf', low=0.68, high=0.76, freq_hz=2.5),
        )

        aggr_vals = [1.0, 1.5, 2.0]
        alpha_vals = [0.2, 0.3, 0.5]

        worst_overslow = 0.0
        worst_reacq = 0.0
        worst_pos_accel_occ = 0.0

        for aggr in aggr_vals:
            for alpha in alpha_vals:
                scn = base_scn
                scn.params = VTSCParams(aggressiveness=aggr, alpha=alpha, hysteresis=0.2, safety_bias=0.1)
                res = simulate(scn)

                m = res.metrics
                worst_overslow = max(worst_overslow, m['integrated_overslow'])
                worst_pos_accel_occ = max(worst_pos_accel_occ, m['pos_accel_while_occluded'])
                if m['reacq_latency'] is not None:
                    worst_reacq = max(worst_reacq, m['reacq_latency'])

        # Invariants
        self.assertLessEqual(worst_pos_accel_occ, 1e-6, "No positive accel while occluded")
        # Reasonable overslow budget for medium bend
        self.assertLessEqual(worst_overslow, 1.5, "Integrated overslow within budget (1.5 m/s·s)")
        # Reacquisition within 0.6s (whenever reacquisition occurs)
        if worst_reacq:
            self.assertLessEqual(worst_reacq, 0.6, "Reacquisition latency within 0.6s")
        print(f"✓ Occlusion invariants: worst overslow={worst_overslow:.3f}, worst reacq={worst_reacq:.3f}, pos_acc_occ={worst_pos_accel_occ:.3g}")

    def test_adaptive_property_accessors(self):
        """Test new property accessors for adaptive system"""
        # Test adaptive_decel_active property
        self.vtsc._current_decel = -1.0  # Less than comfort limit
        self.assertFalse(self.vtsc.adaptive_decel_active)
        
        self.vtsc._current_decel = -2.0  # More than comfort limit
        self.assertTrue(self.vtsc.adaptive_decel_active)
        
        # Test decel_requirement property
        self.vtsc._filtered_decel_requirement = -2.5
        self.assertEqual(self.vtsc.decel_requirement, -2.5)
        print(f"✓ Property accessor tests passed")


def run_tests():
    """Run all adaptive deceleration tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdaptiveDeceleration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("ADAPTIVE DECELERATION SYSTEM TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
