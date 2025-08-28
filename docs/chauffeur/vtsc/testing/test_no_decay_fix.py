#!/usr/bin/env python3
"""
Test to verify the vision occlusion fix maintains last known curvature without decay
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../sunnypilot'))

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionOcclusionState, VisionStatus


def test_no_curvature_decay_during_occlusion():
    """Verify that curvature is maintained exactly during occlusion (no decay)"""
    state = VisionOcclusionState()
    
    # Establish a known curvature with good vision
    initial_curvature = 0.1
    state.update(initial_curvature, 0.9, 100.0)
    assert state.vision_status == VisionStatus.FULL_VISIBILITY
    assert state.last_valid_curvature == 0.0  # Not updated until 3 good frames
    
    # Need multiple good frames to establish trust
    state.update(initial_curvature, 0.9, 100.1)
    state.update(initial_curvature, 0.9, 100.2)
    assert state.good_vision_frames == 3
    assert state.last_valid_curvature == initial_curvature
    
    # Enter occlusion
    state.update(0.05, 0.3, 101.0)  # Different curvature but low confidence
    assert state.vision_status == VisionStatus.SEVERE_OCCLUSION
    assert state.good_vision_frames == 0
    
    # Check that extrapolated curvature equals last valid (no decay)
    assert state.extrapolated_curvature == initial_curvature
    assert state.confidence_decay_factor == 1.0  # No decay applied
    
    # After 5 seconds, curvature should still be maintained
    state.update(0.05, 0.3, 106.0)
    assert state.extrapolated_curvature == initial_curvature  # Still exact same value
    assert state.confidence_decay_factor == 1.0  # Still no decay
    
    # After 10 seconds, curvature should STILL be maintained
    state.update(0.05, 0.3, 111.0)
    assert state.extrapolated_curvature == initial_curvature  # No change!
    assert state.confidence_decay_factor == 1.0
    
    print("✅ PASS: Curvature correctly maintained without decay during occlusion")
    return True


def test_frame_validation_requirement():
    """Verify that multiple good frames are required before trusting new curvature"""
    state = VisionOcclusionState()
    
    # First good frame - shouldn't update curvature
    state.update(0.1, 0.9, 100.0)
    assert state.good_vision_frames == 1
    assert state.last_valid_curvature == 0.0  # Not updated yet
    
    # Second good frame - still shouldn't update
    state.update(0.1, 0.9, 100.1)
    assert state.good_vision_frames == 2
    assert state.last_valid_curvature == 0.0  # Not updated yet
    
    # Third good frame - NOW it updates
    state.update(0.1, 0.9, 100.2)
    assert state.good_vision_frames == 3
    assert state.last_valid_curvature == 0.1  # Updated!
    
    # Occlusion resets counter
    state.update(0.05, 0.3, 101.0)
    assert state.good_vision_frames == 0
    
    # Recovery requires 3 frames again
    state.update(0.2, 0.9, 102.0)  # Frame 1
    state.update(0.2, 0.9, 102.1)  # Frame 2
    assert state.last_valid_curvature == 0.1  # Still old value
    state.update(0.2, 0.9, 102.2)  # Frame 3
    assert state.last_valid_curvature == 0.2  # Now updated
    
    print("✅ PASS: Frame validation correctly requires 3 good frames")
    return True


def main():
    """Run all tests"""
    print("\n=== Testing Vision Occlusion Fix ===\n")
    
    tests = [
        test_no_curvature_decay_during_occlusion,
        test_frame_validation_requirement,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR in {test.__name__}: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)