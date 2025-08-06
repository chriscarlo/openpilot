"""
Emergency Escalation Logic Unit Test

Tests the emergency escalation system in the integrated VTSC.
Validates emergency level determination, deceleration limiting, jerk limiting,
and state transition management.

CRITICAL: This test must validate the 5-level escalation system works correctly.
"""

import sys
import os
import time

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from vtsc_test_framework import VTSCTestBase, EmergencyLevel, VTSCTestRunner

class TestEmergencyEscalationLogic(VTSCTestBase):
    """Test suite for Emergency Escalation Logic"""

    def test_emergency_level_determination_thresholds(self):
        """Test that emergency levels are determined at correct deceleration thresholds"""

        # Test NORMAL level (≤ 1.47 m/s²)
        result_level = self.vtsc._determine_emergency_level(-1.0, time.time())
        if result_level != EmergencyLevel.NORMAL:
            raise AssertionError(f"Decel -1.0 should be NORMAL level. Got: {result_level.name}")

        result_level = self.vtsc._determine_emergency_level(-1.47, time.time())  # Exact boundary
        if result_level != EmergencyLevel.NORMAL:
            raise AssertionError(f"Decel -1.47 (boundary) should be NORMAL level. Got: {result_level.name}")

        # Test CAUTION level (≤ 2.45 m/s²)
        result_level = self.vtsc._determine_emergency_level(-2.0, time.time())
        if result_level != EmergencyLevel.CAUTION:
            raise AssertionError(f"Decel -2.0 should be CAUTION level. Got: {result_level.name}")

        result_level = self.vtsc._determine_emergency_level(-2.45, time.time())  # Exact boundary
        if result_level != EmergencyLevel.CAUTION:
            raise AssertionError(f"Decel -2.45 (boundary) should be CAUTION level. Got: {result_level.name}")

        # Test WARNING level (≤ 3.92 m/s²)
        result_level = self.vtsc._determine_emergency_level(-3.5, time.time())
        if result_level != EmergencyLevel.WARNING:
            raise AssertionError(f"Decel -3.5 should be WARNING level. Got: {result_level.name}")

        result_level = self.vtsc._determine_emergency_level(-3.92, time.time())  # Exact boundary
        if result_level != EmergencyLevel.WARNING:
            raise AssertionError(f"Decel -3.92 (boundary) should be WARNING level. Got: {result_level.name}")

        # Test CRITICAL level (≤ 5.50 m/s²)
        result_level = self.vtsc._determine_emergency_level(-5.0, time.time())
        if result_level != EmergencyLevel.CRITICAL:
            raise AssertionError(f"Decel -5.0 should be CRITICAL level. Got: {result_level.name}")

        result_level = self.vtsc._determine_emergency_level(-5.50, time.time())  # Exact boundary
        if result_level != EmergencyLevel.CRITICAL:
            raise AssertionError(f"Decel -5.50 (boundary) should be CRITICAL level. Got: {result_level.name}")

        # Test INTERVENTION level (> 5.50 m/s²)
        result_level = self.vtsc._determine_emergency_level(-6.0, time.time())
        if result_level != EmergencyLevel.INTERVENTION:
            raise AssertionError(f"Decel -6.0 should be INTERVENTION level. Got: {result_level.name}")

        result_level = self.vtsc._determine_emergency_level(-8.0, time.time())  # Well above
        if result_level != EmergencyLevel.INTERVENTION:
            raise AssertionError(f"Decel -8.0 should be INTERVENTION level. Got: {result_level.name}")

    def test_deceleration_limiting_by_emergency_level(self):
        """Test that deceleration is properly limited by emergency level"""

        # Test NORMAL level limiting (-1.47 m/s²)
        # Note: When requesting -3.0, level changes to WARNING and uses WARNING jerk limit
        self.vtsc._emergency_level = EmergencyLevel.NORMAL
        limited_decel = self.vtsc._get_optimal_deceleration(-3.0, 0.1)  # Raw: -3.0, level becomes WARNING
        # WARNING jerk limit is -4.0 m/s³, so first step: 4.0 * 0.1 = 0.4 m/s² change
        self.assert_approximately_equal(limited_decel, -0.4, tolerance=0.05,
                                      message="First step with WARNING jerk should be -0.4 m/s²")

        # Test CAUTION level limiting (-2.45 m/s²)
        self.vtsc._emergency_level = EmergencyLevel.CAUTION
        self.vtsc._current_decel = -1.47  # Start from NORMAL level
        limited_decel = self.vtsc._get_optimal_deceleration(-4.0, 0.1)  # Raw: -4.0, should limit to -2.45
        # Due to jerk limiting, might not reach -2.45 immediately, but should move toward it
        if limited_decel > -1.47:  # Should be moving more negative (stronger decel)
            raise AssertionError(f"CAUTION level should move toward -2.45 m/s². Got: {limited_decel:.3f}")

        # Test WARNING level limiting (-3.92 m/s²)
        self.vtsc._emergency_level = EmergencyLevel.WARNING
        self.vtsc._current_decel = -2.45  # Start from CAUTION level
        limited_decel = self.vtsc._get_optimal_deceleration(-6.0, 0.1)  # Raw: -6.0, should limit to -3.92
        if limited_decel > -2.45:  # Should be moving more negative
            raise AssertionError(f"WARNING level should move toward -3.92 m/s². Got: {limited_decel:.3f}")

        # Test CRITICAL level limiting (-5.50 m/s²)
        self.vtsc._emergency_level = EmergencyLevel.CRITICAL
        self.vtsc._current_decel = -3.92  # Start from WARNING level
        limited_decel = self.vtsc._get_optimal_deceleration(-8.0, 0.1)  # Raw: -8.0, should limit to -5.50
        if limited_decel > -3.92:  # Should be moving more negative
            raise AssertionError(f"CRITICAL level should move toward -5.50 m/s². Got: {limited_decel:.3f}")

        # Test INTERVENTION level limiting (-6.00 m/s²)
        self.vtsc._emergency_level = EmergencyLevel.INTERVENTION
        self.vtsc._current_decel = -5.50  # Start from CRITICAL level
        limited_decel = self.vtsc._get_optimal_deceleration(-10.0, 0.1)  # Raw: -10.0, should limit to -6.00
        if limited_decel > -5.50:  # Should be moving more negative
            raise AssertionError(f"INTERVENTION level should move toward -6.00 m/s². Got: {limited_decel:.3f}")

    def test_jerk_limiting_smooth_transitions(self):
        """Test that jerk limiting provides smooth transitions between emergency levels"""

        # Start at NORMAL level with 0 decel
        self.vtsc._emergency_level = EmergencyLevel.NORMAL
        self.vtsc._current_decel = 0.0

        # Request moderate decel that should be CAUTION level
        dt = 0.1  # 100ms timestep
        limited_decel = self.vtsc._get_optimal_deceleration(-2.0, dt)

        # Request triggers CAUTION level, so uses CAUTION jerk limit: -3.0 m/s³
        max_change = 3.0 * dt  # 0.3 m/s² change allowed
        expected_decel = 0.0 - max_change  # -0.3 m/s²
        self.assert_approximately_equal(limited_decel, expected_decel, tolerance=0.01,
                                      message="First step should be limited by CAUTION jerk (-3.0 m/s³)")

        # Continue applying same request - should gradually approach target
        for i in range(5):
            limited_decel = self.vtsc._get_optimal_deceleration(-2.0, dt)

        # After several steps, should be closer to -2.0 (but may still be jerk-limited)
        if limited_decel > -0.5:  # Should have made significant progress
            raise AssertionError(f"After jerk-limited steps, should approach target. Got: {limited_decel:.3f}")

    def test_emergency_level_transitions_and_timing(self):
        """Test emergency level transitions and timing state management"""

        # Start at NORMAL level
        initial_time = time.time()
        self.assert_emergency_level(EmergencyLevel.NORMAL, "Should start at NORMAL level")

        # Verify initial timing state
        if self.vtsc._time_at_current_level != 0.0:
            raise AssertionError(f"Initial time_at_current_level should be 0.0. Got: {self.vtsc._time_at_current_level}")

        # Trigger transition to CAUTION level
        self.vtsc._get_optimal_deceleration(-2.0, 0.1)  # Should trigger CAUTION
        self.assert_emergency_level(EmergencyLevel.CAUTION, "Should transition to CAUTION level")

        # Check that timing was reset on transition
        if self.vtsc._time_at_current_level != 0.0:
            raise AssertionError(f"time_at_current_level should reset on transition. Got: {self.vtsc._time_at_current_level}")

        # Stay at same level - time should accumulate
        dt = 0.1
        self.vtsc._get_optimal_deceleration(-2.0, dt)  # Same level
        self.assert_approximately_equal(self.vtsc._time_at_current_level, dt, tolerance=0.01,
                                      message="time_at_current_level should accumulate when staying at same level")

        # Another step at same level
        self.vtsc._get_optimal_deceleration(-2.0, dt)
        self.assert_approximately_equal(self.vtsc._time_at_current_level, 2*dt, tolerance=0.01,
                                      message="time_at_current_level should continue accumulating")

        # Trigger transition to WARNING level
        self.vtsc._get_optimal_deceleration(-3.5, dt)  # Should trigger WARNING
        self.assert_emergency_level(EmergencyLevel.WARNING, "Should transition to WARNING level")

        # Time should reset again
        if self.vtsc._time_at_current_level != 0.0:
            raise AssertionError(f"time_at_current_level should reset on WARNING transition. Got: {self.vtsc._time_at_current_level}")

    def test_emergency_level_state_variables(self):
        """Test that emergency level state variables are managed correctly"""

        # Check initial state
        self.assert_emergency_level(EmergencyLevel.NORMAL, "Should start at NORMAL")
        self.assert_approximately_equal(self.vtsc._current_decel, 0.0, tolerance=0.001,
                                      message="Should start with 0 current decel")
        self.assert_approximately_equal(self.vtsc._time_at_current_level, 0.0, tolerance=0.001,
                                      message="Should start with 0 time at level")

        # Trigger escalation and check state updates
        dt = 0.1
        result_decel = self.vtsc._get_optimal_deceleration(-4.0, dt)  # Should trigger CRITICAL

        # Emergency level should update
        self.assert_emergency_level(EmergencyLevel.CRITICAL, "Should escalate to CRITICAL for -4.0 m/s²")

        # Current decel should be updated (jerk-limited)
        if self.vtsc._current_decel == 0.0:
            raise AssertionError("_current_decel should be updated from initial 0.0")

        # Time should be managed
        if self.vtsc._last_emergency_update_time == 0.0:
            raise AssertionError("_last_emergency_update_time should be set")

        # Return result should match current_decel state
        self.assert_approximately_equal(result_decel, self.vtsc._current_decel, tolerance=0.001,
                                      message="Returned decel should match internal _current_decel state")

    def test_emergency_reset_behavior(self):
        """Test that emergency system resets properly when VTSC is reset"""

        # Escalate to high emergency level
        self.vtsc._get_optimal_deceleration(-6.0, 0.1)  # INTERVENTION level
        self.assert_emergency_level(EmergencyLevel.INTERVENTION, "Should escalate to INTERVENTION")

        # Take another step to accumulate time
        self.vtsc._get_optimal_deceleration(-6.0, 0.1)  # Second step

        # Verify we have non-zero state
        if self.vtsc._current_decel == 0.0:
            raise AssertionError("Should have non-zero _current_decel before reset")
        if self.vtsc._time_at_current_level == 0.0:
            raise AssertionError("Should have non-zero _time_at_current_level before reset")

        # Reset VTSC
        self.vtsc._reset()

        # Check that emergency state is reset
        self.assert_emergency_level(EmergencyLevel.NORMAL, "Should reset to NORMAL level")
        self.assert_approximately_equal(self.vtsc._current_decel, 0.0, tolerance=0.001,
                                      message="Should reset _current_decel to 0.0")

    def test_extreme_deceleration_requests(self):
        """Test system behavior with extreme deceleration requests"""

        # Test very high deceleration request
        extreme_decel = -15.0  # Much higher than INTERVENTION limit
        result_decel = self.vtsc._get_optimal_deceleration(extreme_decel, 0.1)

        # Should escalate to INTERVENTION level
        self.assert_emergency_level(EmergencyLevel.INTERVENTION, "Extreme decel should trigger INTERVENTION")

        # Should be limited and jerk-controlled
        if result_decel < -7.0:  # Should not exceed reasonable bounds even with jerk limiting
            raise AssertionError(f"Extreme decel should be bounded. Got: {result_decel:.3f}")

        # Test positive deceleration (acceleration)
        accel_request = 2.0  # Positive = acceleration
        result_decel = self.vtsc._get_optimal_deceleration(accel_request, 0.1)

        # Should trigger CAUTION level (same as -2.0 m/s²)
        self.assert_emergency_level(EmergencyLevel.CAUTION, "2.0 m/s² acceleration should trigger CAUTION level")

    def test_boundary_conditions_and_edge_cases(self):
        """Test boundary conditions and edge cases in emergency escalation"""

        # Test rapid oscillation between levels
        dt = 0.05  # 50ms timestep

        # Oscillate between NORMAL and CAUTION thresholds
        oscillation_sequence = [-1.4, -1.5, -1.45, -1.48, -1.46]  # Around -1.47 threshold

        for decel_request in oscillation_sequence:
            self.vtsc._get_optimal_deceleration(decel_request, dt)
            # Should handle rapid changes without issues
            if self.vtsc._emergency_level not in [EmergencyLevel.NORMAL, EmergencyLevel.CAUTION]:
                raise AssertionError(f"Oscillation around threshold caused unexpected level: {self.vtsc._emergency_level.name}")

        # Test zero deceleration request
        zero_result = self.vtsc._get_optimal_deceleration(0.0, dt)
        self.assert_emergency_level(EmergencyLevel.NORMAL, "Zero decel should be NORMAL level")

        # Test very small timestep
        tiny_dt = 0.001  # 1ms
        self.vtsc._get_optimal_deceleration(-3.0, tiny_dt)
        # Should handle tiny timesteps without issues (jerk limiting should still work)
        if self.vtsc._current_decel < -1.0:  # With tiny dt, change should be very small
            raise AssertionError(f"Tiny timestep should limit decel change. Got: {self.vtsc._current_decel:.6f}")

def run_emergency_escalation_tests():
    """Run all emergency escalation logic tests"""
    runner = VTSCTestRunner()

    # Define all test methods
    test_methods = [
        'test_emergency_level_determination_thresholds',
        'test_deceleration_limiting_by_emergency_level',
        'test_jerk_limiting_smooth_transitions',
        'test_emergency_level_transitions_and_timing',
        'test_emergency_level_state_variables',
        'test_emergency_reset_behavior',
        'test_extreme_deceleration_requests',
        'test_boundary_conditions_and_edge_cases'
    ]

    print("Running Emergency Escalation Logic Tests...")
    print("="*80)

    # Run each test
    for test_method in test_methods:
        result = runner.run_test(TestEmergencyEscalationLogic, test_method)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:4} | {test_method:45} | {result.execution_time:.3f}s")
        if not result.passed:
            print(f"     | Error: {result.error_message}")

    # Print summary
    runner.print_results()
    return runner.results

if __name__ == "__main__":
    results = run_emergency_escalation_tests()

    # Exit with error code if any tests failed
    failed_tests = [r for r in results if not r.passed]
    if failed_tests:
        print(f"\n{len(failed_tests)} test(s) failed!")
        exit(1)
    else:
        print(f"\nAll {len(results)} tests passed!")
        exit(0)
