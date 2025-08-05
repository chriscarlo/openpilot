"""
Vision Occlusion State Transitions Unit Test

Tests the critical VisionOcclusionState logic that was fixed in the integrated VTSC.
Validates state transitions, timing logic, confidence decay, and curvature extrapolation.

CRITICAL: This test must catch the occlusion timing bug if reintroduced.
"""

import sys
import os
import time

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from vtsc_test_framework import VTSCTestBase, VisionStatus, VTSCTestRunner, calculate_expected_confidence_decay

class TestVisionOcclusionStateTransitions(VTSCTestBase):
    """Test suite for VisionOcclusionState state transitions and logic"""

    def test_confidence_threshold_state_transitions(self):
        """Test that vision status changes at correct confidence thresholds"""

        # Test FULL_VISIBILITY (confidence > 0.8)
        self.update_vtsc(vision_confidence=0.9)
        self.assert_vision_status(VisionStatus.FULL_VISIBILITY, "High confidence should give full visibility")

        # Test PARTIAL_OCCLUSION (0.5 < confidence <= 0.8)
        self.update_vtsc(vision_confidence=0.65)
        self.assert_vision_status(VisionStatus.PARTIAL_OCCLUSION, "Medium confidence should give partial occlusion")

        # Test SEVERE_OCCLUSION (0.2 < confidence <= 0.5)
        self.update_vtsc(vision_confidence=0.35)
        self.assert_vision_status(VisionStatus.SEVERE_OCCLUSION, "Low confidence should give severe occlusion")

        # Test VISION_LOST (confidence <= 0.2)
        self.update_vtsc(vision_confidence=0.1)
        self.assert_vision_status(VisionStatus.VISION_LOST, "Very low confidence should result in vision lost")

        # Test exact threshold boundaries
        self.update_vtsc(vision_confidence=0.8)
        self.assert_vision_status(VisionStatus.PARTIAL_OCCLUSION, "Confidence of 0.8 (boundary) should be partial occlusion")

        self.update_vtsc(vision_confidence=0.5)
        self.assert_vision_status(VisionStatus.SEVERE_OCCLUSION, "Confidence of 0.5 (boundary) should be severe occlusion")

        self.update_vtsc(vision_confidence=0.2)
        self.assert_vision_status(VisionStatus.VISION_LOST, "Confidence of 0.2 (boundary) should be vision lost")

    def test_occlusion_start_time_logic(self):
        """Test that occlusion_start_time is set correctly on transitions FROM full visibility"""

        # Start with full visibility
        self.update_vtsc(vision_confidence=0.9)
        initial_occlusion_time = self.vtsc._occlusion_state.occlusion_start_time

        # Transition to partial occlusion - should update occlusion_start_time
        self.update_vtsc(vision_confidence=0.65)
        first_occlusion_time = self.vtsc._occlusion_state.occlusion_start_time
        self.assert_occlusion_start_time_updated(initial_occlusion_time,
                                               "Occlusion start time should update on transition from full visibility")

        # Transition from partial to severe occlusion - should NOT update occlusion_start_time
        time.sleep(0.01)  # Small delay to ensure time would change if updated
        self.update_vtsc(vision_confidence=0.3)
        second_occlusion_time = self.vtsc._occlusion_state.occlusion_start_time
        if second_occlusion_time != first_occlusion_time:
            raise AssertionError(f"Occlusion start time should NOT update when transitioning between occlusion states. "
                               f"Expected: {first_occlusion_time}, Actual: {second_occlusion_time}")

        # Transition from severe to vision lost - should NOT update occlusion_start_time
        time.sleep(0.01)
        self.update_vtsc(vision_confidence=0.1)
        third_occlusion_time = self.vtsc._occlusion_state.occlusion_start_time
        if third_occlusion_time != first_occlusion_time:
            raise AssertionError(f"Occlusion start time should NOT update when transitioning between occlusion states. "
                               f"Expected: {first_occlusion_time}, Actual: {third_occlusion_time}")

        # Return to full visibility - should reset, ready for next occlusion
        self.update_vtsc(vision_confidence=0.95)
        self.assert_vision_status(VisionStatus.FULL_VISIBILITY, "Should return to full visibility")

        # New occlusion should update start time again
        time.sleep(0.01)
        self.update_vtsc(vision_confidence=0.4)
        fourth_occlusion_time = self.vtsc._occlusion_state.occlusion_start_time
        self.assert_occlusion_start_time_updated(first_occlusion_time,
                                               "New occlusion after returning to full visibility should update start time")

    def test_confidence_decay_calculation(self):
        """Test that confidence decay calculation works correctly over time"""

        # Start with full visibility to establish baseline
        self.update_vtsc(vision_confidence=0.9, curvature=0.08)
        self.assert_vision_status(VisionStatus.FULL_VISIBILITY)
        initial_curvature = self.vtsc._occlusion_state.last_valid_curvature

        # Transition to occlusion
        start_time = time.time()
        self.update_vtsc(vision_confidence=0.4)  # SEVERE_OCCLUSION
        self.assert_vision_status(VisionStatus.SEVERE_OCCLUSION)

        # Simulate passage of time and check decay calculation
        test_durations = [0.5, 1.0, 2.0, 3.0, 5.0]

        for duration in test_durations:
            # Mock the occlusion start time to simulate elapsed time
            self.vtsc._occlusion_state.occlusion_start_time = start_time - duration

            # Update to trigger decay calculation
            self.update_vtsc(vision_confidence=0.4)

            # Calculate expected decay
            expected_decay = calculate_expected_confidence_decay(duration)
            actual_decay = self.vtsc._occlusion_state.confidence_decay_factor

            self.assert_approximately_equal(actual_decay, expected_decay, tolerance=0.001,
                                          message=f"Confidence decay incorrect after {duration}s")

            # Check extrapolated curvature uses decay factor
            expected_extrapolated = initial_curvature * expected_decay
            actual_extrapolated = self.vtsc._occlusion_state.extrapolated_curvature

            self.assert_approximately_equal(actual_extrapolated, expected_extrapolated, tolerance=0.001,
                                          message=f"Extrapolated curvature incorrect after {duration}s")

    def test_curvature_extrapolation_logic(self):
        """Test that curvature extrapolation uses correct formula and updates properly"""

        # Start with full visibility and specific curvature
        test_curvature = 0.12
        # Multiple updates needed for EMA filter to converge
        for _ in range(10):  # Allow EMA to converge to target curvature
            self.update_vtsc(vision_confidence=0.9, curvature=test_curvature)
        self.assert_vision_status(VisionStatus.FULL_VISIBILITY)

        # Verify last_valid_curvature is captured
        last_valid = self.vtsc._occlusion_state.last_valid_curvature
        self.assert_approximately_equal(last_valid, test_curvature, tolerance=0.02,
                                      message="Last valid curvature should converge near input during full visibility")

        # Transition to occlusion
        self.update_vtsc(vision_confidence=0.25)  # SEVERE_OCCLUSION

        # Check that extrapolated curvature equals last_valid * decay_factor
        decay_factor = self.vtsc._occlusion_state.confidence_decay_factor
        expected_extrapolated = last_valid * decay_factor
        actual_extrapolated = self.vtsc._occlusion_state.extrapolated_curvature

        self.assert_approximately_equal(actual_extrapolated, expected_extrapolated, tolerance=0.001,
                                      message="Extrapolated curvature should equal last_valid * decay_factor")

        # Test with different curvature values during occlusion (should use last valid, not current)
        self.update_vtsc(vision_confidence=0.25, curvature=0.20)  # Different curvature during occlusion

        # last_valid_curvature should NOT change during occlusion
        current_last_valid = self.vtsc._occlusion_state.last_valid_curvature
        self.assert_approximately_equal(current_last_valid, last_valid, tolerance=0.001,
                                      message="last_valid_curvature should not change during occlusion")

    def test_rapid_state_changes_edge_case(self):
        """Test rapid state changes around confidence thresholds"""

        # Start with full visibility
        self.update_vtsc(vision_confidence=0.9)
        initial_occlusion_time = self.vtsc._occlusion_state.occlusion_start_time

        # Rapidly oscillate around the 0.8 threshold
        confidences = [0.82, 0.78, 0.81, 0.79, 0.83, 0.77]
        expected_states = [VisionStatus.FULL_VISIBILITY, VisionStatus.PARTIAL_OCCLUSION,
                          VisionStatus.FULL_VISIBILITY, VisionStatus.PARTIAL_OCCLUSION,
                          VisionStatus.FULL_VISIBILITY, VisionStatus.PARTIAL_OCCLUSION]

        for i, (confidence, expected_state) in enumerate(zip(confidences, expected_states, strict=False)):
            self.update_vtsc(vision_confidence=confidence)
            self.assert_vision_status(expected_state, f"Rapid change {i+1}: confidence {confidence}")

        # Test oscillation around 0.5 threshold
        confidences = [0.52, 0.48, 0.51, 0.49]
        expected_states = [VisionStatus.PARTIAL_OCCLUSION, VisionStatus.SEVERE_OCCLUSION,
                          VisionStatus.PARTIAL_OCCLUSION, VisionStatus.SEVERE_OCCLUSION]

        for i, (confidence, expected_state) in enumerate(zip(confidences, expected_states, strict=False)):
            self.update_vtsc(vision_confidence=confidence)
            self.assert_vision_status(expected_state, f"Threshold oscillation {i+1}: confidence {confidence}")

    def test_confidence_decay_minimum_floor(self):
        """Test that confidence decay respects the 0.3 minimum floor"""

        # Start with full visibility
        self.update_vtsc(vision_confidence=0.9, curvature=0.1)
        self.assert_vision_status(VisionStatus.FULL_VISIBILITY)

        # Transition to occlusion
        start_time = time.time()
        self.update_vtsc(vision_confidence=0.1)  # VISION_LOST

        # Simulate very long occlusion (should hit the 0.3 floor)
        very_long_duration = 15.0  # Long enough to decay below 0.3 without floor
        self.vtsc._occlusion_state.occlusion_start_time = start_time - very_long_duration

        # Update to trigger decay calculation
        self.update_vtsc(vision_confidence=0.1)

        # Should be at the 0.3 floor, not lower
        actual_decay = self.vtsc._occlusion_state.confidence_decay_factor
        self.assert_approximately_equal(actual_decay, 0.3, tolerance=0.001,
                                      message="Confidence decay should not drop below 0.3 floor")

        # Verify extrapolated curvature uses the floor value
        last_valid = self.vtsc._occlusion_state.last_valid_curvature
        expected_extrapolated = last_valid * 0.3
        actual_extrapolated = self.vtsc._occlusion_state.extrapolated_curvature

        self.assert_approximately_equal(actual_extrapolated, expected_extrapolated, tolerance=0.001,
                                      message="Extrapolated curvature should use 0.3 floor in decay calculation")

    def test_full_visibility_resets(self):
        """Test that returning to full visibility properly resets the state"""

        # Start with occlusion
        self.update_vtsc(vision_confidence=0.3, curvature=0.06)
        self.assert_vision_status(VisionStatus.SEVERE_OCCLUSION)

        # Verify we're in occlusion with decay
        self.assert_confidence_decay_in_range(0.9, 1.0, "Should have near-maximum decay initially")

        # Return to full visibility with new curvature
        new_curvature = 0.15
        # Multiple updates needed for EMA filter to converge
        for _ in range(10):  # Allow EMA to converge to new curvature
            self.update_vtsc(vision_confidence=0.95, curvature=new_curvature)

        # Check that full visibility state is restored
        self.assert_vision_status(VisionStatus.FULL_VISIBILITY)

        # Check that confidence decay is reset to 1.0
        actual_decay = self.vtsc._occlusion_state.confidence_decay_factor
        self.assert_approximately_equal(actual_decay, 1.0, tolerance=0.001,
                                      message="Confidence decay should reset to 1.0 in full visibility")

        # Check that last_valid_curvature is updated with new value
        last_valid = self.vtsc._occlusion_state.last_valid_curvature
        self.assert_approximately_equal(last_valid, new_curvature, tolerance=0.02,
                                      message="last_valid_curvature should converge to new value during full visibility")

def run_vision_occlusion_tests():
    """Run all vision occlusion state transition tests"""
    runner = VTSCTestRunner()

    # Define all test methods
    test_methods = [
        'test_confidence_threshold_state_transitions',
        'test_occlusion_start_time_logic',
        'test_confidence_decay_calculation',
        'test_curvature_extrapolation_logic',
        'test_rapid_state_changes_edge_case',
        'test_confidence_decay_minimum_floor',
        'test_full_visibility_resets'
    ]

    print("Running Vision Occlusion State Transition Tests...")
    print("="*80)

    # Run each test
    for test_method in test_methods:
        result = runner.run_test(TestVisionOcclusionStateTransitions, test_method)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:4} | {test_method:45} | {result.execution_time:.3f}s")
        if not result.passed:
            print(f"     | Error: {result.error_message}")

    # Print summary
    runner.print_results()
    return runner.results

if __name__ == "__main__":
    results = run_vision_occlusion_tests()

    # Exit with error code if any tests failed
    failed_tests = [r for r in results if not r.passed]
    if failed_tests:
        print(f"\n{len(failed_tests)} test(s) failed!")
        exit(1)
    else:
        print(f"\nAll {len(results)} tests passed!")
        exit(0)
