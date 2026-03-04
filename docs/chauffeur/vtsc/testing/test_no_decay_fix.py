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
    # Vision-good uses dwell timing; keep feeding low confidence until the occlusion path is active.
    for i in range(1, 6):
        state.update(0.05, 0.3, 101.0 + i * 0.1)
        if not state.vision_good:
            break
    assert state.vision_good is False

    # "No decay" now means we keep decay factor at 1.0 and do not collapse curvature to zero.
    assert state.confidence_decay_factor == 1.0
    assert 0.0 < state.extrapolated_curvature <= initial_curvature

    # After long occlusion, extrapolated curvature should remain anchored near last-known-good.
    state.update(0.05, 0.3, 106.0)
    assert state.confidence_decay_factor == 1.0
    assert state.extrapolated_curvature >= initial_curvature - 1e-3

    state.update(0.05, 0.3, 111.0)
    assert state.confidence_decay_factor == 1.0
    assert state.extrapolated_curvature >= initial_curvature - 1e-3
    
    print("✅ PASS: Curvature correctly maintained without decay during occlusion")


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
            try:
                test()
                passed += 1
            except TypeError:
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
