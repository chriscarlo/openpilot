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
    try:
        from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController
        print("SUCCESS: Import successful")
        return True
    except Exception as e:
        print(f"FAILED: Import failed with error: {e}")
        return False

def test_basic_instantiation():
    """Test that we can instantiate VisionTurnController."""
    try:
        from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController
        
        # Mock the Params class to avoid parameter system dependencies
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params', MockParams):
            vtsc = VisionTurnController(None)  # CP can be None for basic test
            print("SUCCESS: Instantiation with mocked Params successful")
            return True
    except Exception as e:
        print(f"FAILED: Instantiation failed with error: {e}")
        return False

def test_basic_assertion():
    """Test basic properties of VisionTurnController."""
    try:
        from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController
        
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params', MockParams):
            vtsc = VisionTurnController(None)
            
            # Test that initial state is disabled
            from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnControllerState
            assert vtsc.state == VisionTurnControllerState.disabled, f"Expected disabled state, got {vtsc.state}"
            
            # Test that aggressiveness parameter was loaded correctly
            assert vtsc._aggressiveness == 1.0, f"Expected aggressiveness=1.0, got {vtsc._aggressiveness}"
            
            print("SUCCESS: Basic assertions passed")
            return True
    except Exception as e:
        print(f"FAILED: Assertion test failed with error: {e}")
        return False

def test_emergency_levels():
    """Test the emergency level determination system."""
    try:
        from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController, EmergencyLevel, DECEL_LIMITS
        
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params', MockParams):
            vtsc = VisionTurnController(None)
            
            # Test that emergency level constants exist and are correct
            assert EmergencyLevel.NORMAL == 0, f"Expected NORMAL=0, got {EmergencyLevel.NORMAL}"
            assert EmergencyLevel.INTERVENTION == 4, f"Expected INTERVENTION=4, got {EmergencyLevel.INTERVENTION}"
            
            # Test decel limits are properly defined
            expected_limits = {
                EmergencyLevel.NORMAL: -1.47,
                EmergencyLevel.CAUTION: -2.45,
                EmergencyLevel.WARNING: -3.92,
                EmergencyLevel.CRITICAL: -5.50,
                EmergencyLevel.INTERVENTION: -6.00
            }
            
            for level, expected_limit in expected_limits.items():
                actual_limit = DECEL_LIMITS[level]
                assert abs(actual_limit - expected_limit) < 0.01, f"Expected {level.name}: {expected_limit}, got {actual_limit}"
            
            print("SUCCESS: Emergency levels test passed")
            return True
    except Exception as e:
        print(f"FAILED: Emergency levels test failed with error: {e}")
        return False

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