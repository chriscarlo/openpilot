#!/usr/bin/env python3
"""
Minimal test to verify VisionTurnController can be imported and instantiated.
Start from scratch with empirical verification.
"""

import sys
import os
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, '/projects/chauffeur/data/openpilot')
sys.path.insert(0, '/projects/chauffeur/data/openpilot/sunnypilot')

class MockParams:
    """Minimal mock for OpenPilot Params."""
    
    def get_bool(self, key):
        return True  # VTSC enabled for testing
        
    def get(self, key):
        # Return default values for VTSC parameters
        defaults = {
            b'VisionTurnSpeedControlAggressiveness': b'1.0',
            b'VisionTurnSpeedControlFixedLeadTimeSeconds': b'0.0', 
            b'VisionTurnSpeedControlFilterAlpha': b'0.3',
            b'VisionTurnSpeedControlHysteresisThreshold': b'0.2',
            b'VisionTurnSpeedControlSafetyBias': b'0.1',
        }
        return defaults.get(key.encode() if isinstance(key, str) else key, b'1.0')

def test_basic_import():
    """Test that we can import VisionTurnController without crashing."""
    from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController  # noqa: F401
    import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vmod
    print("SUCCESS: Import successful from", getattr(vmod, '__file__', '?'))

def test_basic_instantiation():
    """Test that we can instantiate VisionTurnController."""
    from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController
    # Mock the Params class to avoid parameter system dependencies
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params', MockParams):
        vtsc = VisionTurnController(None)  # CP can be None for basic test
        assert vtsc is not None
        print("SUCCESS: Instantiation with mocked Params successful")

def test_basic_assertion():
    """Test basic properties of VisionTurnController."""
    from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController, VisionTurnControllerState
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params', MockParams):
        vtsc = VisionTurnController(None)
        # Test that initial state is disabled
        assert vtsc.state == VisionTurnControllerState.disabled, f"Expected disabled state, got {vtsc.state}"
        # Test that aggressiveness parameter was loaded correctly
        assert vtsc._aggressiveness == 1.0, f"Expected aggressiveness=1.0, got {vtsc._aggressiveness}"
        print("SUCCESS: Basic assertions passed")

def test_physics_bounds_and_monotonicity():
    """Check lateral-accel bounds and curvature_to_speed monotonicity."""
    from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
        VisionTurnController, curvature_to_speed,
        _physics_based_lateral_acceleration as lat_accel,
        PHYSICS_MIN_LAT_ACCEL, PHYSICS_MAX_LAT_ACCEL,
    )
    # Instantiate once with mocked Params (ensure no side-effects)
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params', MockParams):
        _ = VisionTurnController(None)
    # Lat accel stays within configured bounds across a range
    for k in [1e-8, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
        a = lat_accel(k)
        assert PHYSICS_MIN_LAT_ACCEL - 1e-6 <= a <= PHYSICS_MAX_LAT_ACCEL + 1e-6
    # curvature_to_speed decreases with increasing curvature (choose values below ceiling)
    v1 = curvature_to_speed(1e-3)
    v2 = curvature_to_speed(2e-3)
    v3 = curvature_to_speed(5e-3)
    assert v1 >= v2 >= v3 and v1 > 0.0
    print("SUCCESS: Physics bounds and monotonicity passed")

if __name__ == '__main__':
    print("Testing basic import of VisionTurnController...")
    import_success = test_basic_import()
    print(f"Import result: {'PASS' if import_success else 'FAIL'}")
    
    instantiate_success = False
    assertion_success = False
    
    if import_success:
        print("\nTesting basic instantiation...")
        instantiate_success = test_basic_instantiation()
        print(f"Instantiation result: {'PASS' if instantiate_success else 'FAIL'}")
        
        if instantiate_success:
            print("\nTesting basic assertions...")
            assertion_success = test_basic_assertion()
            print(f"Assertion result: {'PASS' if assertion_success else 'FAIL'}")
    
    all_success = import_success and instantiate_success and assertion_success
    print(f"\nOverall result: {'PASS' if all_success else 'FAIL'}")
