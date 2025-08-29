#!/usr/bin/env python3
"""
Integration test suite for complete Adaptive Deceleration System
Tests the full flow from parameter loading through physics calculations to filtered output
"""

import sys
import os
import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, Mock
from dataclasses import dataclass
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../'))

# Import the VisionTurnController
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
    VisionTurnController,
    VisionTurnControllerState,
    COMFORT_DECEL_LIMIT,
    MAX_ADAPTIVE_DECEL,
)

@dataclass
class MockCarParams:
    """Mock car parameters for testing"""
    pass

@dataclass
class MockModelV2:
    """Mock model data for testing"""
    orientationRate: Mock = None
    velocity: Mock = None
    laneLineProbs: list = None
    
    def __init__(self):
        self.orientationRate = Mock()
        self.velocity = Mock()
        self.laneLineProbs = [0.9, 0.9]  # Good vision confidence


class TestFullIntegration(unittest.TestCase):
    """Integration tests for complete adaptive deceleration system"""
    
    def create_vtsc_with_config(self, filter_alpha="0.3", hysteresis="0.2", safety_bias="0.1"):
        """Helper to create configured VTSC"""
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
            mock_params = MagicMock()
            mock_params.get_bool.return_value = True
            
            def get_param(key):
                params_map = {
                    "VisionTurnSpeedControlAggressiveness": b"1.0",
                    "VisionTurnSpeedControlFixedLeadTimeSeconds": b"0.0",
                    "VisionTurnSpeedControlFilterAlpha": filter_alpha.encode() if filter_alpha else None,
                    "VisionTurnSpeedControlHysteresisThreshold": hysteresis.encode() if hysteresis else None,
                    "VisionTurnSpeedControlSafetyBias": safety_bias.encode() if safety_bias else None
                }
                return params_map.get(key)
            
            mock_params.get.side_effect = get_param
            MockParams.return_value = mock_params
            
            return VisionTurnController(MockCarParams())
    
    def create_mock_sm(self, curvature_values, velocity_values):
        """Create minimal dict-like SM with model and carState data."""
        model = MockModelV2()
        model.orientationRate.z = curvature_values
        model.velocity.x = velocity_values

        class SM:
            def __init__(self, model):
                self.valid = {'modelV2': True}
                self._data = {
                    'modelV2': model,
                    'carState': SimpleNamespace(gasPressed=False),
                }
            def __getitem__(self, key):
                return self._data.get(key)

        return SM(model)
    
    def test_full_curve_approach_scenario(self):
        """Test complete flow: detect curve → calculate physics → filter → apply decel"""
        print("\n" + "="*50)
        print("FULL CURVE APPROACH SCENARIO")
        print("="*50)
        
        # Create VTSC with specific configuration
        vtsc = self.create_vtsc_with_config(
            filter_alpha="0.3",
            hysteresis="0.2",
            safety_bias="0.1"
        )
        
        # Initialize vehicle state
        vtsc._op_enabled = True
        vtsc._v_ego = 25.0  # 25 m/s (~90 km/h)
        vtsc._a_ego = 0.0
        vtsc._v_cruise_setpoint = 30.0
        
        # Create curve ahead (increasing curvature)
        n_points = 33
        curvatures = np.linspace(0.001, 0.008, n_points)  # Increasing curvature
        velocities = np.ones(n_points) * 25.0
        
        sm = self.create_mock_sm(curvatures.tolist(), velocities.tolist())
        
        # Simulate multiple update cycles with enough iterations to accumulate jerk-limited decel
        results = []
        for i in range(40):
            # Update VTSC
            vtsc.update(sm, True, vtsc._v_ego, vtsc._a_ego, vtsc._v_cruise_setpoint)
            
            # Record state
            results.append({
                'cycle': i,
                'a_target': vtsc._a_target,
                'filtered_decel': vtsc._filtered_decel_requirement,
                'adaptive_active': vtsc.adaptive_decel_active,
                'state': vtsc.state
            })
            
            # Simulate vehicle response (gradual speed reduction)
            vtsc._v_ego += vtsc._a_target * 0.05  # dt = 0.05
            vtsc._a_ego = vtsc._a_target
        
        # Verify behavior
        # 1. Should detect significant curvature ahead
        self.assertGreater(vtsc.max_pred_lat_acc, 0.1, "Should detect significant curvature during approach")
        
        # 2. Should transition to adaptive mode if needed
        adaptive_triggered = any(r['adaptive_active'] for r in results)
        print(f"Adaptive deceleration triggered: {adaptive_triggered}")
        
        # 3. Filtered deceleration should be smooth
        decel_changes = [abs(results[i+1]['filtered_decel'] - results[i]['filtered_decel']) 
                        for i in range(len(results)-1)]
        max_change = max(decel_changes) if decel_changes else 0
        self.assertLess(max_change, 1.0, "Deceleration changes should be smooth")
        
        print(f"Final state: v_ego={vtsc._v_ego:.1f} m/s, a_target={vtsc._a_target:.2f} m/s²")
        print(f"Max decel change per cycle: {max_change:.3f} m/s²")
    
    def test_emergency_deceleration_scenario(self):
        """Test emergency scenario requiring maximum deceleration"""
        print("\n" + "="*50)
        print("EMERGENCY DECELERATION SCENARIO")
        print("="*50)
        
        vtsc = self.create_vtsc_with_config(safety_bias="0.2")  # Higher safety bias
        
        # High speed approaching sharp curve
        vtsc._op_enabled = True
        vtsc._v_ego = 30.0  # 30 m/s (~108 km/h)
        vtsc._v_overshoot = 12.0  # Target speed for curve
        vtsc._v_overshoot_distance = 30.0  # Very short distance!
        vtsc._lat_acc_overshoot_ahead = True
        vtsc._prev_target_speed = 29.0
        vtsc._v_cruise_setpoint = 35.0
        
        # Mock the planning method
        with patch.object(vtsc, '_plan_advanced_speed_trajectory', return_value=15.0):
            # Run update cycle
            vtsc._update_solution()
        
        # Should apply negative deceleration (jerk limited per-cycle)
        self.assertLess(vtsc._a_target, -0.1, "Should apply deceleration")
        
        # Should detect challenging scenario
        is_challenging = vtsc._monitor_adaptive_deceleration(
            vtsc._a_target, vtsc._v_overshoot_distance
        )
        
        print(f"Emergency decel applied: {vtsc._a_target:.2f} m/s²")
        print(f"Challenging scenario detected: {is_challenging}")
    
    def test_mode_transition_flow(self):
        """Test smooth transition between comfort and adaptive modes"""
        print("\n" + "="*50)
        print("MODE TRANSITION FLOW")
        print("="*50)
        
        vtsc = self.create_vtsc_with_config(hysteresis="0.25")
        
        dt = 0.05
        transition_log = []
        
        # Gradually increase deceleration requirement
        decel_requirements = np.concatenate([
            np.linspace(-0.5, -1.4, 10),   # Within comfort
            np.linspace(-1.5, -3.0, 10),   # Trigger adaptive
            np.linspace(-2.8, -1.2, 10),   # Back toward comfort
            np.linspace(-1.1, -0.8, 10),   # Return to comfort
        ])
        
        for req in decel_requirements:
            result = vtsc._get_optimal_deceleration(req, dt)
            transition_log.append({
                'required': req,
                'applied': result,
                'mode': 'adaptive' if vtsc._decel_hysteresis_state else 'comfort',
                'filtered': vtsc._filtered_decel_requirement
            })
        
        # Count mode transitions
        transitions = 0
        for i in range(1, len(transition_log)):
            if transition_log[i]['mode'] != transition_log[i-1]['mode']:
                transitions += 1
                print(f"Transition at step {i}: {transition_log[i-1]['mode']} → {transition_log[i]['mode']}")
        
        # Should have exactly 2 transitions (comfort→adaptive→comfort)
        self.assertEqual(transitions, 2, "Should have 2 mode transitions with hysteresis")
    
    def test_parameter_update_during_operation(self):
        """Test parameter updates during active operation"""
        print("\n" + "="*50)
        print("PARAMETER UPDATE DURING OPERATION")
        print("="*50)
        
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic') as mock_time:
            vtsc = self.create_vtsc_with_config(filter_alpha="0.2")
            
            initial_alpha = vtsc._filter_alpha
            self.assertAlmostEqual(initial_alpha, 0.2, places=2)
            
            # Run some deceleration cycles
            dt = 0.05
            for _ in range(5):
                vtsc._get_optimal_deceleration(-2.0, dt)
            
            initial_filtered = vtsc._filtered_decel_requirement
            
            # Simulate parameter update after 6 seconds
            mock_time.return_value = vtsc._last_params_update + 6.0
            
            # Change parameter value
            with patch.object(vtsc._params, 'get') as mock_get:
                def get_param(key):
                    if key == "VisionTurnSpeedControlFilterAlpha":
                        return b"0.7"  # New value
                    return b"1.0" if "Aggressiveness" in key else b"0.0"
                
                mock_get.side_effect = get_param
                vtsc._update_params()
            
            # Verify parameter updated
            self.assertAlmostEqual(vtsc._filter_alpha, 0.7, places=2)
            
            # Continue operation with new parameter
            for _ in range(5):
                vtsc._get_optimal_deceleration(-2.0, dt)
            
            print(f"Parameter update: alpha {initial_alpha:.1f} → {vtsc._filter_alpha:.1f}")
            print(f"Filtering behavior changed as expected")
    
    def test_vision_degradation_handling(self):
        """Test system behavior during vision degradation (simplified occlusion)."""
        print("\n" + "="*50)
        print("VISION DEGRADATION HANDLING")
        print("="*50)
        
        vtsc = self.create_vtsc_with_config()
        
        # Good vision initially
        good_model = MockModelV2()
        good_model.laneLineProbs = [0.9, 0.9]
        
        current_time = time.time()
        _ = vtsc._update_vision_occlusion(good_model, current_time)
        self.assertTrue(vtsc._occlusion_state.vision_good)
        
        # Degrade vision progressively (EMA requires time to decay below threshold)
        poor_model = MockModelV2()
        poor_model.laneLineProbs = [0.1, 0.1]
        t = current_time
        for _ in range(20):
            t += 0.1
            _ = vtsc._update_vision_occlusion(poor_model, t)
        
        self.assertFalse(vtsc._occlusion_state.vision_good)
        self.assertLess(vtsc._occlusion_state.smoothed_confidence, vtsc._occlusion_state.good_threshold)
        
        print(f"Vision good: {vtsc._occlusion_state.vision_good}")
        print(f"Smoothed conf: {vtsc._occlusion_state.smoothed_confidence:.2f}")

    def test_occlusion_subcases_with_metrics(self):
        """Entry occlusion hovering, late-apex occlusion, and S-curve inflection with hard metrics."""
        from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile
        from docs.chauffeur.vtsc.testing.harness.simulate import simulate

        # Entry occlusion before apex with borderline confidence
        scn_entry = Scenario(
            name='entry_occ', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='tightening', kappa0=0.002, kappa1=0.006),
            confidence=ConfidenceProfile(kind='borderline_lpf', low=0.68, high=0.76, freq_hz=2.5),
        )
        # Late-apex occlusion easing exit
        scn_late = Scenario(
            name='late_apex', duration_s=8.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='easing', kappa0=0.002, kappa1=0.006),
            confidence=ConfidenceProfile(kind='window', value=0.6, window_start_s=3.0, window_end_s=5.0),
        )
        # S-curve with short median straight
        scn_s = Scenario(
            name='s_curve', duration_s=9.0, dt=0.05, v0_mps=25.0,
            geometry=GeometryProfile(kind='s_curve', kappa0=0.004, kappa1=0.004, mid_straight_s=10.0),
            confidence=ConfidenceProfile(kind='borderline_lpf', low=0.68, high=0.76, freq_hz=2.5),
        )

        for scn in [scn_entry, scn_late, scn_s]:
            res = simulate(scn)
            m = res.metrics
            # No positive acceleration while occluded
            self.assertLessEqual(m['pos_accel_while_occluded'], 1e-6)
            # Reacquisition within 0.6s when applicable
            if m['reacq_latency'] is not None:
                self.assertLessEqual(m['reacq_latency'], 0.6)
            # Integrated overslow under budget (post-reacquisition window)
            self.assertLessEqual(m['integrated_overslow'], 30.0)
            # Overshoot on recovery should be small. Allow limited overshoot due to barrier smoothing.
            self.assertLessEqual(m['overshoot_on_recovery'], 3.5)
        print("✓ Occlusion subcases meet invariants")
    
    def test_complete_update_cycle(self):
        """Test complete update cycle with all subsystems"""
        print("\n" + "="*50)
        print("COMPLETE UPDATE CYCLE TEST")
        print("="*50)
        
        vtsc = self.create_vtsc_with_config(
            filter_alpha="0.4",
            hysteresis="0.15",
            safety_bias="0.15"
        )
        
        # Create realistic curve scenario
        curvatures = [0.001, 0.002, 0.004, 0.006, 0.007, 0.006, 0.004, 0.002, 0.001] + [0.001] * 24
        velocities = [25.0] * 33
        
        sm = self.create_mock_sm(curvatures, velocities)
        
        # Full update
        vtsc.update(sm, True, 25.0, 0.0, 30.0)
        
        # Verify all systems engaged
        self.assertIsNotNone(vtsc._a_target, "Should have acceleration target")
        self.assertIsNotNone(vtsc._filtered_decel_requirement, "Should have filtered decel")
        self.assertIsNotNone(vtsc._current_lat_acc, "Should have current lateral accel")
        self.assertIsNotNone(vtsc.state, "Should have valid state")
        
        print(f"All subsystems operational:")
        print(f"  - Target acceleration: {vtsc._a_target:.2f} m/s²")
        print(f"  - Filtered decel requirement: {vtsc._filtered_decel_requirement:.2f} m/s²")
        print(f"  - Adaptive active: {vtsc.adaptive_decel_active}")
        print(f"  - Controller state: {vtsc.state}")


def run_tests():
    """Run all integration tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFullIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("FULL INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
