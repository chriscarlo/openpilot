#!/usr/bin/env python3
"""
Simple test to verify post-apex acceleration logic is functional in production VTSC.
This test specifically addresses the issue where post-apex acceleration was calculated 
but never returned by the a_target property.
"""

import sys
from unittest.mock import patch

sys.path.insert(0, '/data/openpilot')

# Mock the Params import before importing VisionTurnController
class MockParams:
    def get_bool(self, key):
        return True if key == "VisionTurnSpeedControl" else False

with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params', MockParams):
    from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController

def test_post_apex_acceleration_fix():
    """Test that post-apex acceleration is actually returned by a_target property"""

    # Create mock car params
    class MockCP:
        def __init__(self):
            self.steerRatio = 15.0
            self.wheelbase = 2.7

    # Initialize controller
    controller = VisionTurnController(MockCP())

    # Test 1: Verify that when NOT in post-apex mode, a_target returns _current_decel
    controller._state = 1  # entering state (active)
    controller._current_decel = -2.5
    controller._apex_acceleration_active = False
    controller._apex_acceleration_value = 0.0

    assert controller.a_target == -2.5, f"Expected -2.5, got {controller.a_target}"
    print("PASS Test 1: When not in post-apex mode, a_target returns _current_decel")

    # Test 2: Verify that when in post-apex mode, a_target returns _apex_acceleration_value
    controller._apex_acceleration_active = True
    controller._apex_acceleration_value = 1.8  # Positive acceleration for post-apex

    assert controller.a_target == 1.8, f"Expected 1.8, got {controller.a_target}"
    print("PASS Test 2: When in post-apex mode, a_target returns _apex_acceleration_value")

    # Test 3: Verify disabled state returns ego acceleration
    controller._state = 0  # disabled
    controller._a_ego = -0.5

    assert controller.a_target == -0.5, f"Expected -0.5, got {controller.a_target}"
    print("PASS Test 3: When disabled, a_target returns _a_ego")

    # Test 4: Verify the post-apex logic can be triggered
    # Set up conditions that should trigger post-apex acceleration
    controller._state = 1  # active
    controller._past_apex = True
    controller._acceleration_embargo_lifted = True
    controller.filtered_curvature = 0.01  # Small curvature (gentle curve)
    controller._v_ego = 15.0  # 54 km/h
    controller._v_cruise_setpoint = 25.0  # 90 km/h cruise

    # Mock the curvature-to-speed conversion to return a reasonable target
    original_method = controller._curvature_to_speed
    controller._curvature_to_speed = lambda c: 20.0  # 72 km/h safe speed

    # Make sure current_decel satisfies the condition
    controller._current_decel = 0.0  # Start with no deceleration

    # Call _update_solution to trigger the post-apex logic
    controller._update_solution()

    # Verify that post-apex acceleration was activated
    assert controller._apex_acceleration_active, "Post-apex acceleration should be active"
    assert controller._apex_acceleration_value > 0, f"Expected positive acceleration, got {controller._apex_acceleration_value}"
    assert controller.a_target > 0, f"a_target should return positive acceleration, got {controller.a_target}"

    print(f"PASS Test 4: Post-apex acceleration activated with value {controller._apex_acceleration_value:.2f} m/s^2")

    # Restore original method
    controller._curvature_to_speed = original_method

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
    print("VERIFIED: Post-apex acceleration logic is now FUNCTIONAL")
    print("VERIFIED: The a_target property correctly returns acceleration values")
    print("VERIFIED: The core issue identified in your analysis has been FIXED")

    return True

if __name__ == "__main__":
    print("Testing Post-Apex Acceleration Fix")
    print("="*60)
    success = test_post_apex_acceleration_fix()
    if success:
        print("\nSUCCESS: VTSC POST-APEX ACCELERATION FIX VERIFIED")
    else:
        print("\nFAILED: Tests failed")
        sys.exit(1)
