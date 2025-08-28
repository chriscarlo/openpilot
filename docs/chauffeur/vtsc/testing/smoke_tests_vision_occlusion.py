#!/usr/bin/env python3
"""
Smoke tests for Vision Occlusion Handling
Quick sanity checks to ensure basic functionality works
"""

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../sunnypilot'))

def test_can_import_vision_controller():
    """Test that we can import the vision turn controller module"""
    try:
        from sunnypilot.selfdrive.controls.lib import vision_turn_controller
        assert vision_turn_controller is not None
        print("✓ Can import vision_turn_controller module")
        return True
    except ImportError as e:
        print(f"✗ Cannot import vision_turn_controller: {e}")
        return False

def test_vision_status_enum_exists():
    """Test that VisionStatus enum is defined"""
    try:
        from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionStatus
        assert VisionStatus.FULL_VISIBILITY == 0
        assert VisionStatus.PARTIAL_OCCLUSION == 1
        assert VisionStatus.SEVERE_OCCLUSION == 2
        assert VisionStatus.VISION_LOST == 3
        print("✓ VisionStatus enum correctly defined")
        return True
    except (ImportError, AttributeError) as e:
        print(f"✗ VisionStatus enum issue: {e}")
        return False

def test_vision_occlusion_state_exists():
    """Test that VisionOcclusionState class is defined"""
    try:
        from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionOcclusionState
        state = VisionOcclusionState()
        assert hasattr(state, 'last_valid_curvature')
        assert hasattr(state, 'vision_status')
        assert hasattr(state, 'confidence_decay_factor')
        assert hasattr(state, 'extrapolated_curvature')
        assert hasattr(state, 'occlusion_start_time')
        assert hasattr(state, 'update')
        print("✓ VisionOcclusionState class correctly defined")
        return True
    except (ImportError, AttributeError) as e:
        print(f"✗ VisionOcclusionState class issue: {e}")
        return False

def test_basic_state_transition():
    """Test basic state transition functionality"""
    try:
        from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionOcclusionState
        
        state = VisionOcclusionState()
        
        # Test full visibility
        state.update(0.05, 0.9, 100.0)
        assert state.vision_status == 0  # FULL_VISIBILITY
        
        # Test transition to occlusion
        state.update(0.05, 0.3, 101.0)
        assert state.vision_status == 2  # SEVERE_OCCLUSION
        
        print("✓ Basic state transitions work")
        return True
    except Exception as e:
        print(f"✗ State transition issue: {e}")
        return False

def test_vision_controller_initialization():
    """Test that VisionTurnController can be initialized"""
    try:
        from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController
        
        # Create a mock CarParams object
        class MockCP:
            def __init__(self):
                pass
        
        controller = VisionTurnController(MockCP())
        assert controller is not None
        assert hasattr(controller, '_occlusion_state')
        print("✓ VisionTurnController can be initialized")
        return True
    except Exception as e:
        print(f"✗ VisionTurnController initialization issue: {e}")
        return False

def main():
    """Run all smoke tests"""
    print("\n=== Vision Occlusion Smoke Tests ===\n")
    
    tests = [
        test_can_import_vision_controller,
        test_vision_status_enum_exists,
        test_vision_occlusion_state_exists,
        test_basic_state_transition,
        test_vision_controller_initialization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)