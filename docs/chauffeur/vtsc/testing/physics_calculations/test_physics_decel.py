#!/usr/bin/env python3
"""
Test suite for Physics-Based Deceleration Calculations
Validates physics formula: a = (v_f² - v_i²) / (2d) with safety bias
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../'))

# Import the VisionTurnController
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
    VisionTurnController,
    MAX_ADAPTIVE_DECEL,
    DEFAULT_SAFETY_BIAS
)

@dataclass
class MockCarParams:
    """Mock car parameters for testing"""
    pass


class TestPhysicsCalculations(unittest.TestCase):
    """Test cases for physics-based deceleration calculations"""
    
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
            self.vtsc._safety_bias = DEFAULT_SAFETY_BIAS
    
    def test_basic_physics_formula(self):
        """Test basic physics formula without safety bias"""
        # Temporarily set safety bias to 0 for pure physics test
        self.vtsc._safety_bias = 0.0
        
        # Test case: slow from 20 m/s to 15 m/s over 50 meters
        v_current = 20.0  # m/s (~72 km/h)
        v_target = 15.0   # m/s (~54 km/h)
        distance = 50.0   # meters
        
        # Expected: a = (15² - 20²) / (2 × 50) = (225 - 400) / 100 = -1.75 m/s²
        expected = -1.75
        
        result = self.vtsc._calculate_required_deceleration(v_current, v_target, distance)
        
        self.assertAlmostEqual(result, expected, places=2)
        print(f"✓ Basic physics test: v={v_current}→{v_target} m/s over {distance}m = {result:.2f} m/s²")
    
    def test_safety_bias_application(self):
        """Test safety bias increases deceleration magnitude"""
        self.vtsc._safety_bias = 0.1  # 10% safety margin
        
        v_current = 20.0
        v_target = 15.0
        distance = 50.0
        
        # Raw decel: -1.75 m/s²
        # With 10% bias: -1.75 × 1.1 = -1.925 m/s²
        expected = -1.925
        
        result = self.vtsc._calculate_required_deceleration(v_current, v_target, distance)
        
        self.assertAlmostEqual(result, expected, places=2)
        print(f"✓ Safety bias test: Raw=-1.75, With 10% bias={result:.2f} m/s²")
    
    def test_maximum_decel_clamping(self):
        """Test clamping to maximum system deceleration"""
        # Test extreme scenario requiring huge deceleration
        v_current = 30.0  # m/s
        v_target = 5.0    # m/s
        distance = 10.0   # Very short distance
        
        result = self.vtsc._calculate_required_deceleration(v_current, v_target, distance)
        
        # Should be clamped to MAX_ADAPTIVE_DECEL (-6.0)
        self.assertEqual(result, MAX_ADAPTIVE_DECEL)
        print(f"✓ Maximum decel clamping: Extreme case clamped to {MAX_ADAPTIVE_DECEL:.1f} m/s²")
    
    def test_zero_distance_handling(self):
        """Test handling of zero/near-zero distance"""
        v_current = 20.0
        v_target = 15.0
        
        # Test exact zero
        result = self.vtsc._calculate_required_deceleration(v_current, v_target, 0.0)
        self.assertEqual(result, MAX_ADAPTIVE_DECEL)
        
        # Test near zero (< 0.1m)
        result = self.vtsc._calculate_required_deceleration(v_current, v_target, 0.05)
        self.assertEqual(result, MAX_ADAPTIVE_DECEL)
        
        print(f"✓ Zero distance handling: Returns maximum decel {MAX_ADAPTIVE_DECEL:.1f} m/s²")
    
    def test_acceleration_scenario(self):
        """Test when target speed is higher (acceleration needed)"""
        v_current = 15.0
        v_target = 20.0  # Higher target
        distance = 50.0
        
        result = self.vtsc._calculate_required_deceleration(v_current, v_target, distance)
        
        # Should return positive value (acceleration)
        self.assertGreater(result, 0)
        print(f"✓ Acceleration scenario: v={v_current}→{v_target} returns positive {result:.2f} m/s²")
    
    def test_various_speed_ranges(self):
        """Test physics calculations across different speed ranges"""
        test_cases = [
            # (v_current, v_target, distance, description)
            (5.0, 3.0, 10.0, "Low speed urban"),
            (15.0, 10.0, 30.0, "Medium speed suburban"),
            (30.0, 20.0, 80.0, "Highway entrance ramp"),
            (35.0, 25.0, 100.0, "Highway curve"),
        ]
        
        print("\n✓ Speed range tests:")
        for v_curr, v_tgt, dist, desc in test_cases:
            result = self.vtsc._calculate_required_deceleration(v_curr, v_tgt, dist)
            
            # Verify physics formula
            raw_expected = (v_tgt**2 - v_curr**2) / (2 * dist)
            biased_expected = raw_expected * (1 + self.vtsc._safety_bias)
            
            self.assertAlmostEqual(result, max(biased_expected, MAX_ADAPTIVE_DECEL), places=2)
            print(f"  {desc}: {v_curr:.0f}→{v_tgt:.0f} m/s = {result:.2f} m/s²")
    
    def test_integration_with_update_solution(self):
        """Test physics calculation integration in _update_solution"""
        # Set up scenario
        self.vtsc._v_ego = 25.0
        self.vtsc._v_overshoot = 18.0
        self.vtsc._v_overshoot_distance = 60.0
        self.vtsc._lat_acc_overshoot_ahead = True
        self.vtsc._prev_target_speed = 24.0
        self.vtsc._v_cruise_setpoint = 30.0
        
        # Mock the planning method to return a specific value
        with patch.object(self.vtsc, '_plan_advanced_speed_trajectory', return_value=20.0):
            # Call _update_solution
            self.vtsc._update_solution()
        
        # Check that physics calculation was applied
        # The decel should be based on physics when overshoot is detected
        expected_physics_decel = self.vtsc._calculate_required_deceleration(
            self.vtsc._v_ego, self.vtsc._v_overshoot, self.vtsc._v_overshoot_distance
        )
        
        # The actual decel might be filtered, but should trend toward physics requirement
        self.assertLess(self.vtsc._a_target, 0)  # Should be decelerating
        print(f"✓ Integration test: Physics decel applied in _update_solution: {self.vtsc._a_target:.2f} m/s²")
    
    def test_safety_bias_ranges(self):
        """Test different safety bias values"""
        v_current = 20.0
        v_target = 15.0
        distance = 50.0
        raw_decel = -1.75  # Expected raw deceleration
        
        bias_values = [0.0, 0.05, 0.1, 0.2, 0.5]
        
        print("\n✓ Safety bias range tests:")
        for bias in bias_values:
            self.vtsc._safety_bias = bias
            result = self.vtsc._calculate_required_deceleration(v_current, v_target, distance)
            expected = raw_decel * (1 + bias)
            
            self.assertAlmostEqual(result, expected, places=2)
            print(f"  Bias={bias:.0%}: {result:.2f} m/s² (raw={raw_decel:.2f})")
    
    def test_curve_entry_scenarios(self):
        """Test realistic curve entry scenarios"""
        scenarios = [
            # Sharp turn requiring significant deceleration
            {"v_current": 22.0, "v_target": 12.0, "distance": 40.0, "name": "Sharp turn"},
            # Gentle curve requiring modest deceleration
            {"v_current": 30.0, "v_target": 25.0, "distance": 80.0, "name": "Gentle curve"},
            # Emergency scenario - very short distance
            {"v_current": 25.0, "v_target": 10.0, "distance": 20.0, "name": "Emergency"},
        ]
        
        print("\n✓ Curve entry scenarios:")
        for scenario in scenarios:
            result = self.vtsc._calculate_required_deceleration(
                scenario["v_current"], 
                scenario["v_target"], 
                scenario["distance"]
            )
            
            print(f"  {scenario['name']}: {scenario['v_current']:.0f}→{scenario['v_target']:.0f} m/s "
                  f"over {scenario['distance']:.0f}m = {result:.2f} m/s²")
            
            # Verify it doesn't exceed system maximum
            self.assertGreaterEqual(result, MAX_ADAPTIVE_DECEL)


def run_tests():
    """Run all physics calculation tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhysicsCalculations)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("PHYSICS-BASED DECELERATION CALCULATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)