#!/usr/bin/env python3
"""
Unit tests for Vision Occlusion Handling in VisionTurnController
Tests the VisionOcclusionState class and related functionality
"""

import pytest
import time
import math
from dataclasses import dataclass
from enum import IntEnum

# Mock the VisionStatus enum and VisionOcclusionState class
# These would normally be imported from vision_turn_controller.py
class VisionStatus(IntEnum):
    """Vision quality status for occlusion handling."""
    FULL_VISIBILITY = 0
    PARTIAL_OCCLUSION = 1
    SEVERE_OCCLUSION = 2
    VISION_LOST = 3

@dataclass
class VisionOcclusionState:
    """State tracking for vision occlusion scenarios."""
    last_valid_curvature: float = 0.0
    vision_status: VisionStatus = VisionStatus.FULL_VISIBILITY
    confidence_decay_factor: float = 1.0
    extrapolated_curvature: float = 0.0
    occlusion_start_time: float = 0.0

    def update(self, current_curvature: float, vision_confidence: float, current_time: float):
        """Update occlusion state based on current vision conditions."""
        # Store previous status before updating
        previous_status = self.vision_status

        # Determine vision status from confidence
        if vision_confidence > 0.8:
            self.vision_status = VisionStatus.FULL_VISIBILITY
            self.last_valid_curvature = current_curvature
            self.confidence_decay_factor = 1.0
        elif vision_confidence > 0.5:
            self.vision_status = VisionStatus.PARTIAL_OCCLUSION
        elif vision_confidence > 0.2:
            self.vision_status = VisionStatus.SEVERE_OCCLUSION
        else:
            self.vision_status = VisionStatus.VISION_LOST

        # Set occlusion start time when transitioning from full visibility to any occlusion
        if previous_status == VisionStatus.FULL_VISIBILITY and self.vision_status != VisionStatus.FULL_VISIBILITY:
            self.occlusion_start_time = current_time

        # Update confidence decay factor based on occlusion duration
        if self.vision_status != VisionStatus.FULL_VISIBILITY:
            occlusion_duration = current_time - self.occlusion_start_time
            # Exponential decay: starts at 1.0, decays to 0.3 over 5 seconds
            self.confidence_decay_factor = max(0.3, math.exp(-occlusion_duration / 3.0))

            # Extrapolate curvature during occlusion
            self.extrapolated_curvature = self.last_valid_curvature * self.confidence_decay_factor


class TestVisionOcclusionState:
    """Test suite for VisionOcclusionState class"""

    def test_initial_state(self):
        """Test that initial state is properly set"""
        state = VisionOcclusionState()
        assert state.last_valid_curvature == 0.0
        assert state.vision_status == VisionStatus.FULL_VISIBILITY
        assert state.confidence_decay_factor == 1.0
        assert state.extrapolated_curvature == 0.0
        assert state.occlusion_start_time == 0.0

    def test_full_visibility_update(self):
        """Test update with full visibility (confidence > 0.8)"""
        state = VisionOcclusionState()
        curvature = 0.05
        confidence = 0.9
        current_time = 100.0
        
        state.update(curvature, confidence, current_time)
        
        assert state.vision_status == VisionStatus.FULL_VISIBILITY
        assert state.last_valid_curvature == curvature
        assert state.confidence_decay_factor == 1.0
        assert state.occlusion_start_time == 0.0  # Should remain 0 in full visibility

    def test_transition_to_partial_occlusion(self):
        """Test transition from full visibility to partial occlusion"""
        state = VisionOcclusionState()
        
        # Start with full visibility
        state.update(0.05, 0.9, 100.0)
        assert state.vision_status == VisionStatus.FULL_VISIBILITY
        
        # Transition to partial occlusion
        state.update(0.05, 0.6, 101.0)
        assert state.vision_status == VisionStatus.PARTIAL_OCCLUSION
        assert state.occlusion_start_time == 101.0
        assert state.confidence_decay_factor < 1.0  # Should start decaying

    def test_confidence_decay_over_time(self):
        """Test that confidence decay follows exponential decay pattern"""
        state = VisionOcclusionState()
        
        # Start with full visibility
        state.update(0.05, 0.9, 100.0)
        
        # Enter occlusion
        state.update(0.05, 0.3, 100.0)
        assert state.occlusion_start_time == 100.0
        
        # Check decay at various time points
        state.update(0.05, 0.3, 101.0)  # 1 second
        decay_1s = state.confidence_decay_factor
        assert 0.6 < decay_1s < 0.8  # Approximate expected value
        
        state.update(0.05, 0.3, 103.0)  # 3 seconds
        decay_3s = state.confidence_decay_factor
        assert 0.3 < decay_3s < 0.5
        
        state.update(0.05, 0.3, 110.0)  # 10 seconds
        decay_10s = state.confidence_decay_factor
        assert decay_10s == 0.3  # Should hit minimum

    def test_extrapolated_curvature_calculation(self):
        """Test that extrapolated curvature is calculated correctly"""
        state = VisionOcclusionState()
        initial_curvature = 0.1
        
        # Set initial good curvature
        state.update(initial_curvature, 0.9, 100.0)
        assert state.last_valid_curvature == initial_curvature
        
        # Enter occlusion
        state.update(0.05, 0.3, 101.0)
        
        # Extrapolated curvature should be last_valid * decay_factor
        expected = initial_curvature * state.confidence_decay_factor
        assert abs(state.extrapolated_curvature - expected) < 0.001

    @pytest.mark.parametrize("confidence,expected_status", [
        (0.9, VisionStatus.FULL_VISIBILITY),
        (0.81, VisionStatus.FULL_VISIBILITY),
        (0.8, VisionStatus.PARTIAL_OCCLUSION),  # Edge case
        (0.6, VisionStatus.PARTIAL_OCCLUSION),
        (0.51, VisionStatus.PARTIAL_OCCLUSION),
        (0.5, VisionStatus.SEVERE_OCCLUSION),  # Edge case
        (0.3, VisionStatus.SEVERE_OCCLUSION),
        (0.21, VisionStatus.SEVERE_OCCLUSION),
        (0.2, VisionStatus.VISION_LOST),  # Edge case
        (0.1, VisionStatus.VISION_LOST),
        (0.0, VisionStatus.VISION_LOST),
    ])
    def test_confidence_thresholds(self, confidence, expected_status):
        """Test that confidence thresholds map to correct vision status"""
        state = VisionOcclusionState()
        state.update(0.05, confidence, 100.0)
        assert state.vision_status == expected_status

    def test_bug_initial_occlusion_state(self):
        """
        BUG TEST: System starting in occluded state has incorrect occlusion_start_time
        This test demonstrates the bug where occlusion_start_time remains 0.0
        """
        state = VisionOcclusionState()
        
        # Start directly in occlusion (no prior FULL_VISIBILITY state)
        current_time = 1000.0
        state.update(0.05, 0.3, current_time)
        
        assert state.vision_status == VisionStatus.SEVERE_OCCLUSION
        # BUG: occlusion_start_time should be set to current_time but remains 0.0
        assert state.occlusion_start_time == 0.0, "BUG: occlusion_start_time not initialized"
        
        # This causes incorrect duration calculation
        occlusion_duration = current_time - state.occlusion_start_time
        assert occlusion_duration == 1000.0, "BUG: Huge incorrect duration"
        
        # Which causes confidence to immediately drop to minimum
        assert state.confidence_decay_factor == 0.3, "BUG: Confidence immediately at minimum"

    def test_bug_missing_reset_on_recovery(self):
        """
        BUG TEST: occlusion_start_time not reset when returning to FULL_VISIBILITY
        """
        state = VisionOcclusionState()
        
        # Start with full visibility
        state.update(0.05, 0.9, 100.0)
        
        # Enter occlusion
        state.update(0.05, 0.3, 101.0)
        assert state.occlusion_start_time == 101.0
        
        # Return to full visibility
        state.update(0.05, 0.9, 105.0)
        assert state.vision_status == VisionStatus.FULL_VISIBILITY
        
        # BUG: occlusion_start_time should be reset but isn't
        assert state.occlusion_start_time == 101.0, "BUG: occlusion_start_time not reset"
        
        # Enter occlusion again
        state.update(0.05, 0.3, 106.0)
        
        # BUG: occlusion_start_time won't be updated since transition is not from FULL_VISIBILITY
        assert state.occlusion_start_time == 101.0, "BUG: Old occlusion_start_time retained"

    def test_bug_no_hysteresis(self):
        """
        BUG TEST: No hysteresis causes rapid state changes
        """
        state = VisionOcclusionState()
        
        # Confidence hovering around 0.8 threshold
        confidences = [0.81, 0.79, 0.81, 0.79, 0.81]
        statuses = []
        
        for i, conf in enumerate(confidences):
            state.update(0.05, conf, 100.0 + i)
            statuses.append(state.vision_status)
        
        # BUG: Status oscillates between FULL_VISIBILITY and PARTIAL_OCCLUSION
        assert statuses == [
            VisionStatus.FULL_VISIBILITY,
            VisionStatus.PARTIAL_OCCLUSION,
            VisionStatus.FULL_VISIBILITY,
            VisionStatus.PARTIAL_OCCLUSION,
            VisionStatus.FULL_VISIBILITY
        ], "BUG: No hysteresis causes oscillation"

    def test_bug_uniform_decay_for_all_occlusions(self):
        """
        BUG TEST: Same decay rate for partial occlusion and complete vision loss
        """
        # Test partial occlusion
        state_partial = VisionOcclusionState()
        state_partial.update(0.05, 0.9, 100.0)  # Full visibility
        state_partial.update(0.05, 0.6, 100.0)  # Partial occlusion
        state_partial.update(0.05, 0.6, 102.0)  # 2 seconds later
        partial_decay = state_partial.confidence_decay_factor
        
        # Test vision lost
        state_lost = VisionOcclusionState()
        state_lost.update(0.05, 0.9, 100.0)  # Full visibility
        state_lost.update(0.05, 0.1, 100.0)  # Vision lost
        state_lost.update(0.05, 0.1, 102.0)  # 2 seconds later
        lost_decay = state_lost.confidence_decay_factor
        
        # BUG: Both use same decay rate despite different severity
        assert partial_decay == lost_decay, "BUG: Same decay for different occlusion levels"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])