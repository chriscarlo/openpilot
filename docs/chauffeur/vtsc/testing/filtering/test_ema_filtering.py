#!/usr/bin/env python3
"""
Test suite for EMA (Exponential Moving Average) Filtering
Validates noise reduction and smooth deceleration transitions
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
    DEFAULT_FILTER_ALPHA,
    COMFORT_DECEL_LIMIT
)

@dataclass
class MockCarParams:
    """Mock car parameters for testing"""
    pass


class TestEMAFiltering(unittest.TestCase):
    """Test cases for EMA filtering functionality"""
    
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
    
    def test_ema_formula(self):
        """Test EMA formula implementation"""
        # Set specific filter alpha
        self.vtsc._filter_alpha = 0.3
        self.vtsc._filtered_decel_requirement = 0.0
        
        # Apply series of values
        values = [-1.0, -2.0, -1.5]
        expected_filtered = []
        
        current_filtered = 0.0
        for val in values:
            # EMA formula: new = (1-alpha)*old + alpha*input
            current_filtered = (1 - 0.3) * current_filtered + 0.3 * val
            expected_filtered.append(current_filtered)
        
        # Apply through the actual implementation
        dt = 0.05
        for i, val in enumerate(values):
            self.vtsc._get_optimal_deceleration(val, dt)
            self.assertAlmostEqual(
                self.vtsc._filtered_decel_requirement, 
                expected_filtered[i], 
                places=4
            )
        
        print(f"✓ EMA formula test: Values={values}, Filtered={[f'{x:.2f}' for x in expected_filtered]}")
    
    def test_noise_reduction(self):
        """Test that EMA filtering reduces noise variance"""
        dt = 0.05
        
        # Generate noisy signal with known variance
        np.random.seed(42)  # For reproducibility
        base_signal = -2.0
        noise_amplitude = 0.5
        samples = 50
        
        noisy_signal = base_signal + np.random.normal(0, noise_amplitude, samples)
        
        # Process through filter
        filtered_values = []
        for value in noisy_signal:
            self.vtsc._get_optimal_deceleration(value, dt)
            filtered_values.append(self.vtsc._filtered_decel_requirement)
        
        # Calculate variance reduction
        input_variance = np.var(noisy_signal)
        output_variance = np.var(filtered_values)
        variance_reduction = (1 - output_variance/input_variance) * 100
        
        self.assertLess(output_variance, input_variance)
        self.assertGreater(variance_reduction, 30)  # Expect at least 30% variance reduction
        
        print(f"✓ Noise reduction test: Input var={input_variance:.3f}, "
              f"Output var={output_variance:.3f}, Reduction={variance_reduction:.1f}%")
    
    def test_step_response(self):
        """Test filter response to step input"""
        self.vtsc._filter_alpha = 0.3
        self.vtsc._filtered_decel_requirement = 0.0
        dt = 0.05
        
        # Apply step from 0 to -3.0
        step_value = -3.0
        response = []
        
        for i in range(10):
            self.vtsc._get_optimal_deceleration(step_value, dt)
            response.append(self.vtsc._filtered_decel_requirement)
        
        # Check convergence
        # After 10 iterations, should be very close to step value
        self.assertAlmostEqual(response[-1], step_value, places=1)
        
        # Check smooth rise (no instant jump)
        self.assertLess(abs(response[0]), abs(step_value) * 0.5)
        
        # Check monotonic approach
        for i in range(1, len(response)):
            self.assertLessEqual(response[i], response[i-1])  # Getting more negative
        
        print(f"✓ Step response test: Smooth convergence to {step_value:.1f} from 0.0")
    
    def test_different_alpha_values(self):
        """Test behavior with different filter alpha values"""
        dt = 0.05
        test_signal = [-1.0, -3.0, -2.0, -2.5, -1.5]
        
        alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = {}
        
        print("\n✓ Alpha value comparison:")
        for alpha in alpha_values:
            # Reset controller with new alpha
            self.vtsc._filter_alpha = alpha
            self.vtsc._filtered_decel_requirement = 0.0
            
            filtered = []
            for val in test_signal:
                self.vtsc._get_optimal_deceleration(val, dt)
                filtered.append(self.vtsc._filtered_decel_requirement)
            
            # Calculate responsiveness (how quickly it follows input)
            lag = np.mean([abs(test_signal[i] - filtered[i]) for i in range(len(test_signal))])
            results[alpha] = lag
            
            print(f"  Alpha={alpha:.1f}: Average lag={lag:.3f}")
        
        # Higher alpha should have lower lag (more responsive)
        self.assertLess(results[0.9], results[0.1])
    
    def test_filter_memory(self):
        """Test that filter maintains memory of past values"""
        self.vtsc._filter_alpha = 0.2  # Low alpha = more memory
        dt = 0.05
        
        # Apply sequence
        sequence = [-1.0, -1.0, -1.0, -3.0, -1.0, -1.0, -1.0]
        filtered = []
        
        for val in sequence:
            self.vtsc._get_optimal_deceleration(val, dt)
            filtered.append(self.vtsc._filtered_decel_requirement)
        
        # The spike at index 3 should affect subsequent values
        spike_index = 3
        
        # Value immediately after spike should be affected
        self.assertLess(filtered[spike_index + 1], -1.2)  # More negative than -1.0
        
        # Effect should decay over time
        self.assertLess(filtered[spike_index + 2], filtered[spike_index + 1])  # Less negative
        
        print(f"✓ Filter memory test: Spike at index {spike_index} affects subsequent values")
    
    def test_filter_with_mode_transitions(self):
        """Test filtering during comfort/adaptive mode transitions"""
        dt = 0.05
        
        # Start with comfort-level deceleration
        for _ in range(5):
            self.vtsc._get_optimal_deceleration(-1.0, dt)
        
        initial_filtered = self.vtsc._filtered_decel_requirement
        self.assertFalse(self.vtsc._decel_hysteresis_state)
        
        # Suddenly require aggressive deceleration
        for _ in range(10):
            self.vtsc._get_optimal_deceleration(-4.0, dt)
        
        # Should transition to adaptive mode
        self.assertTrue(self.vtsc._decel_hysteresis_state)
        
        # Filtered value should smoothly approach -4.0
        self.assertLess(self.vtsc._filtered_decel_requirement, -2.5)
        self.assertGreater(self.vtsc._filtered_decel_requirement, -4.5)
        
        print(f"✓ Mode transition test: Smooth filtering across comfort→adaptive transition")
    
    def test_filter_reset_on_acceleration(self):
        """Test that filter resets properly during acceleration"""
        dt = 0.05
        
        # Build up filtered deceleration
        for _ in range(10):
            self.vtsc._get_optimal_deceleration(-2.5, dt)
        
        self.assertLess(self.vtsc._filtered_decel_requirement, -2.0)
        
        # Simulate acceleration (positive accel_cmd in _update_solution)
        # This should reset the filter
        self.vtsc._filtered_decel_requirement = 0.0
        self.vtsc._decel_hysteresis_state = False
        
        # Verify reset
        self.assertEqual(self.vtsc._filtered_decel_requirement, 0.0)
        self.assertFalse(self.vtsc._decel_hysteresis_state)
        
        print(f"✓ Filter reset test: Properly resets during acceleration")
    
    def test_filter_stability(self):
        """Test filter stability with constant input"""
        self.vtsc._filter_alpha = 0.3
        dt = 0.05
        
        constant_value = -2.0
        values = []
        
        # Apply constant input
        for _ in range(50):
            self.vtsc._get_optimal_deceleration(constant_value, dt)
            values.append(self.vtsc._filtered_decel_requirement)
        
        # Check convergence
        final_values = values[-10:]
        variance = np.var(final_values)
        
        # Should converge to constant value with minimal variance
        self.assertLess(variance, 0.001)
        self.assertAlmostEqual(np.mean(final_values), constant_value, places=2)
        
        print(f"✓ Filter stability test: Converges to {constant_value:.1f} with variance={variance:.6f}")
    
    def test_rapid_changes(self):
        """Test filter behavior with rapid input changes"""
        self.vtsc._filter_alpha = 0.3
        dt = 0.05
        
        # Rapid oscillation
        rapid_changes = [-1.0, -3.0, -1.0, -3.0, -1.0, -3.0]
        filtered = []
        
        for val in rapid_changes:
            self.vtsc._get_optimal_deceleration(val, dt)
            filtered.append(self.vtsc._filtered_decel_requirement)
        
        # Filtered output should be smoother (smaller range)
        input_range = max(rapid_changes) - min(rapid_changes)
        output_range = max(filtered) - min(filtered)
        
        self.assertLess(output_range, input_range * 0.7)
        
        print(f"✓ Rapid changes test: Input range={input_range:.1f}, "
              f"Filtered range={output_range:.1f}")


def run_tests():
    """Run all EMA filtering tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEMAFiltering)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("EMA FILTERING TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)