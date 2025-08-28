#!/usr/bin/env python3
"""
Test suite for Adaptive Deceleration Parameter Loading and Validation
Validates configurable parameters: filter alpha, hysteresis threshold, safety bias
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../'))

# Import the VisionTurnController
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
    VisionTurnController,
    DEFAULT_FILTER_ALPHA,
    DEFAULT_HYSTERESIS_THRESHOLD,
    DEFAULT_SAFETY_BIAS
)

@dataclass
class MockCarParams:
    """Mock car parameters for testing"""
    pass


class TestParameterLoading(unittest.TestCase):
    """Test cases for parameter loading and validation"""
    
    def create_vtsc_with_params(self, filter_alpha=None, hysteresis=None, safety_bias=None):
        """Helper to create VTSC with specific parameter values"""
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
            mock_params = MagicMock()
            mock_params.get_bool.return_value = True
            
            # Set up parameter returns
            def get_param(key):
                if key == "VisionTurnSpeedControlAggressiveness":
                    return b"1.0"
                elif key == "VisionTurnSpeedControlFixedLeadTimeSeconds":
                    return b"0.0"
                elif key == "VisionTurnSpeedControlFilterAlpha":
                    return filter_alpha.encode() if filter_alpha else None
                elif key == "VisionTurnSpeedControlHysteresisThreshold":
                    return hysteresis.encode() if hysteresis else None
                elif key == "VisionTurnSpeedControlSafetyBias":
                    return safety_bias.encode() if safety_bias else None
                return None
            
            mock_params.get.side_effect = get_param
            MockParams.return_value = mock_params
            
            return VisionTurnController(MockCarParams())
    
    def test_default_parameters(self):
        """Test default parameter values when not configured"""
        vtsc = self.create_vtsc_with_params()
        
        self.assertAlmostEqual(vtsc._filter_alpha, DEFAULT_FILTER_ALPHA, places=3)
        self.assertAlmostEqual(vtsc._hysteresis_threshold, DEFAULT_HYSTERESIS_THRESHOLD, places=3)
        self.assertAlmostEqual(vtsc._safety_bias, DEFAULT_SAFETY_BIAS, places=3)
        
        print(f"✓ Default parameters: alpha={vtsc._filter_alpha:.2f}, "
              f"hysteresis={vtsc._hysteresis_threshold:.2f}, bias={vtsc._safety_bias:.2f}")
    
    def test_filter_alpha_validation(self):
        """Test filter alpha parameter validation and clamping"""
        # Test valid value
        vtsc = self.create_vtsc_with_params(filter_alpha="0.5")
        self.assertAlmostEqual(vtsc._filter_alpha, 0.5, places=3)
        
        # Test clamping low
        vtsc = self.create_vtsc_with_params(filter_alpha="0.05")
        self.assertAlmostEqual(vtsc._filter_alpha, 0.1, places=3)
        
        # Test clamping high
        vtsc = self.create_vtsc_with_params(filter_alpha="1.5")
        self.assertAlmostEqual(vtsc._filter_alpha, 0.9, places=3)
        
        # Test invalid string
        vtsc = self.create_vtsc_with_params(filter_alpha="invalid")
        self.assertAlmostEqual(vtsc._filter_alpha, DEFAULT_FILTER_ALPHA, places=3)
        
        print(f"✓ Filter alpha validation: Valid range [0.1, 0.9], defaults to {DEFAULT_FILTER_ALPHA}")
    
    def test_hysteresis_threshold_validation(self):
        """Test hysteresis threshold parameter validation"""
        # Test valid value
        vtsc = self.create_vtsc_with_params(hysteresis="0.3")
        self.assertAlmostEqual(vtsc._hysteresis_threshold, 0.3, places=3)
        
        # Test clamping low
        vtsc = self.create_vtsc_with_params(hysteresis="0.05")
        self.assertAlmostEqual(vtsc._hysteresis_threshold, 0.1, places=3)
        
        # Test clamping high
        vtsc = self.create_vtsc_with_params(hysteresis="0.8")
        self.assertAlmostEqual(vtsc._hysteresis_threshold, 0.5, places=3)
        
        print(f"✓ Hysteresis threshold validation: Valid range [0.1, 0.5], defaults to {DEFAULT_HYSTERESIS_THRESHOLD}")
    
    def test_safety_bias_validation(self):
        """Test safety bias parameter validation"""
        # Test valid value
        vtsc = self.create_vtsc_with_params(safety_bias="0.2")
        self.assertAlmostEqual(vtsc._safety_bias, 0.2, places=3)
        
        # Test clamping low
        vtsc = self.create_vtsc_with_params(safety_bias="-0.1")
        self.assertAlmostEqual(vtsc._safety_bias, 0.0, places=3)
        
        # Test clamping high
        vtsc = self.create_vtsc_with_params(safety_bias="0.8")
        self.assertAlmostEqual(vtsc._safety_bias, 0.5, places=3)
        
        print(f"✓ Safety bias validation: Valid range [0.0, 0.5], defaults to {DEFAULT_SAFETY_BIAS}")
    
    def test_parameter_update_cycle(self):
        """Test parameter updates during runtime via _update_params"""
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
            with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic') as mock_time:
                mock_params = MagicMock()
                mock_params.get_bool.return_value = True
                
                # Initial values
                param_values = {
                    "VisionTurnSpeedControlFilterAlpha": b"0.3",
                    "VisionTurnSpeedControlHysteresisThreshold": b"0.2",
                    "VisionTurnSpeedControlSafetyBias": b"0.1"
                }
                
                def get_param(key):
                    return param_values.get(key, b"1.0" if "Aggressiveness" in key else b"0.0")
                
                mock_params.get.side_effect = get_param
                MockParams.return_value = mock_params
                
                vtsc = VisionTurnController(MockCarParams())
                
                # Initial values
                self.assertAlmostEqual(vtsc._filter_alpha, 0.3, places=3)
                
                # Update parameters
                param_values["VisionTurnSpeedControlFilterAlpha"] = b"0.6"
                
                # Simulate time passing (>5 seconds)
                mock_time.return_value = vtsc._last_params_update + 6.0
                
                vtsc._update_params()
                
                # Check updated value
                self.assertAlmostEqual(vtsc._filter_alpha, 0.6, places=3)
                
                print(f"✓ Parameter update cycle test: Parameters reload after 5 seconds")
    
    def test_parameter_edge_cases(self):
        """Test edge cases in parameter parsing"""
        # Test empty string
        vtsc = self.create_vtsc_with_params(filter_alpha="")
        self.assertAlmostEqual(vtsc._filter_alpha, DEFAULT_FILTER_ALPHA, places=3)
        
        # Test whitespace
        vtsc = self.create_vtsc_with_params(filter_alpha=" 0.4 ")
        self.assertAlmostEqual(vtsc._filter_alpha, 0.4, places=3)
        
        # Test scientific notation
        vtsc = self.create_vtsc_with_params(filter_alpha="3e-1")
        self.assertAlmostEqual(vtsc._filter_alpha, 0.3, places=3)
        
        # Test negative zero
        vtsc = self.create_vtsc_with_params(safety_bias="-0.0")
        self.assertAlmostEqual(vtsc._safety_bias, 0.0, places=3)
        
        print(f"✓ Parameter edge cases handled correctly")
    
    def test_all_parameters_combined(self):
        """Test loading all parameters together"""
        vtsc = self.create_vtsc_with_params(
            filter_alpha="0.45",
            hysteresis="0.25",
            safety_bias="0.15"
        )
        
        self.assertAlmostEqual(vtsc._filter_alpha, 0.45, places=3)
        self.assertAlmostEqual(vtsc._hysteresis_threshold, 0.25, places=3)
        self.assertAlmostEqual(vtsc._safety_bias, 0.15, places=3)
        
        print(f"✓ All parameters loaded: alpha={vtsc._filter_alpha:.2f}, "
              f"hysteresis={vtsc._hysteresis_threshold:.2f}, bias={vtsc._safety_bias:.2f}")
    
    def test_parameter_persistence(self):
        """Test that parameters persist across multiple get_optimal_deceleration calls"""
        vtsc = self.create_vtsc_with_params(
            filter_alpha="0.7",
            hysteresis="0.35",
            safety_bias="0.2"
        )
        
        initial_alpha = vtsc._filter_alpha
        initial_hysteresis = vtsc._hysteresis_threshold
        initial_bias = vtsc._safety_bias
        
        # Run multiple deceleration calculations
        dt = 0.05
        for _ in range(20):
            vtsc._get_optimal_deceleration(-2.0, dt)
        
        # Parameters should remain unchanged
        self.assertAlmostEqual(vtsc._filter_alpha, initial_alpha, places=3)
        self.assertAlmostEqual(vtsc._hysteresis_threshold, initial_hysteresis, places=3)
        self.assertAlmostEqual(vtsc._safety_bias, initial_bias, places=3)
        
        print(f"✓ Parameters persist correctly during operation")


def run_tests():
    """Run all parameter validation tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParameterLoading)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("PARAMETER LOADING AND VALIDATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)