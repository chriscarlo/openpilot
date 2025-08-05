#!/usr/bin/env python3
"""
Debug Update Calculations

Test the _update_calculations method directly with proper mock data setup.
This should identify why overshoot detection isn't working.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from vtsc_test_framework import VTSCTestBase, MockSubMaster, MockModelData

def debug_update_calculations_with_proper_mock():
    """Debug _update_calculations with properly configured mock data"""

    print("=== DEBUGGING _update_calculations WITH PROPER MOCK ===")

    test = VTSCTestBase()
    test.setUp()

    # Set test conditions
    test_speed = 25.0  # 90 kph
    test.vtsc._v_ego = test_speed
    test.vtsc._v_cruise_setpoint = test_speed

    # Create properly configured mock submasters
    mock_sm = MockSubMaster()

    # Create mock model data with sharp curve ahead
    sharp_curvatures = [0.0, 0.0, 0.08, 0.10, 0.12]  # Sharp curve starting at position 2

    # Create new model data
    mock_model_data = MockModelData()

    # Set orientation rates (curvature * velocity)
    mock_model_data.orientationRate.z = [curv * test_speed for curv in sharp_curvatures]
    # Pad to 33 elements with decreasing curvature
    for i in range(len(sharp_curvatures), 33):
        decay_factor = max(0.0, 1.0 - (i - len(sharp_curvatures)) * 0.1)
        mock_model_data.orientationRate.z.append(sharp_curvatures[-1] * test_speed * decay_factor)

    # Set velocities to constant speed
    mock_model_data.velocity.x = [test_speed] * 33

    # Set good vision confidence
    mock_model_data.laneLineProbs = [0.9] * 4
    mock_model_data.laneLineStds = [0.1] * 4

    # Assign to submasters
    mock_sm.data['modelV2'] = mock_model_data
    mock_sm.valid['modelV2'] = True

    print("Mock setup:")
    print(f"  Test speed: {test_speed:.1f} m/s ({test_speed*3.6:.0f} kph)")
    print(f"  Sharp curvatures: {sharp_curvatures}")
    print(f"  Orientation rates (first 5): {mock_model_data.orientationRate.z[:5]}")
    print(f"  Velocities (first 5): {mock_model_data.velocity.x[:5]}")
    print(f"  modelV2 valid: {mock_sm.valid['modelV2']}")

    # Reset detection flags before test
    test.vtsc._lat_acc_overshoot_ahead = False
    test.vtsc._v_overshoot_distance = 0.0
    test.vtsc._v_overshoot = 0.0

    print("\nBefore _update_calculations:")
    print(f"  _lat_acc_overshoot_ahead: {test.vtsc._lat_acc_overshoot_ahead}")
    print(f"  _v_overshoot_distance: {test.vtsc._v_overshoot_distance}")
    print(f"  _v_overshoot: {test.vtsc._v_overshoot}")

    # Call _update_calculations
    test.vtsc._update_calculations(mock_sm)

    print("\nAfter _update_calculations:")
    print(f"  _lat_acc_overshoot_ahead: {test.vtsc._lat_acc_overshoot_ahead}")
    print(f"  _v_overshoot_distance: {test.vtsc._v_overshoot_distance:.1f} m")
    print(f"  _v_overshoot: {test.vtsc._v_overshoot:.2f} m/s")
    print(f"  _max_pred_lat_acc: {test.vtsc._max_pred_lat_acc:.2f} m/s²")
    print(f"  _filtered_curvature: {test.vtsc._filtered_curvature:.4f}")

    # Manual verification of what should happen
    print("\nManual verification:")
    import numpy as np
    sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
    from vision_turn_controller import curvature_to_speed

    # Calculate what the curvature array should be
    orientation_rates = np.array(mock_model_data.orientationRate.z[:5])
    velocities = np.array(mock_model_data.velocity.x[:5])
    curvature_array = np.abs(orientation_rates) / velocities

    print(f"  Manual curvature calculation: {curvature_array}")

    safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in curvature_array])
    print(f"  Manual safe speeds: {safe_speeds}")

    overshoot_mask = safe_speeds < test_speed
    print(f"  Manual overshoot mask: {overshoot_mask}")
    print(f"  Manual should detect overshoot: {np.any(overshoot_mask)}")

    if np.any(overshoot_mask):
        overshoot_idx = np.where(overshoot_mask)[0][0]
        print(f"  Manual overshoot index: {overshoot_idx}")
        print(f"  Manual target speed: {safe_speeds[overshoot_idx]:.2f} m/s")

def debug_anticipatory_calculation_flow():
    """Debug the complete anticipatory flow including target speed calculation"""

    print("\n=== DEBUGGING COMPLETE ANTICIPATORY FLOW ===")

    test = VTSCTestBase()
    test.setUp()

    # Set test conditions
    test_speed = 25.0
    test.vtsc._v_ego = test_speed
    test.vtsc._v_cruise_setpoint = test_speed

    # Manually set detection flags to simulate working overshoot detection
    test.vtsc._lat_acc_overshoot_ahead = True
    test.vtsc._v_overshoot = 4.0  # Target speed for sharp curve (realistic from curvature 0.08)
    test.vtsc._v_overshoot_distance = 50.0  # 50m to curve
    test.vtsc._is_decelerating_for_curve = False

    print("Manually set conditions:")
    print(f"  _lat_acc_overshoot_ahead: {test.vtsc._lat_acc_overshoot_ahead}")
    print(f"  _v_overshoot: {test.vtsc._v_overshoot:.1f} m/s")
    print(f"  _v_overshoot_distance: {test.vtsc._v_overshoot_distance:.1f} m")
    print(f"  _is_decelerating_for_curve: {test.vtsc._is_decelerating_for_curve}")

    # Test target speed calculation with the anticipatory logic
    target_speed = test.vtsc.v_turn

    print("\nTarget speed result:")
    print(f"  v_turn: {target_speed:.2f} m/s ({target_speed*3.6:.1f} kph)")
    print(f"  _is_decelerating_for_curve after: {test.vtsc._is_decelerating_for_curve}")
    print(f"  _curve_detection_distance: {test.vtsc._curve_detection_distance:.1f} m")

    # Test progression as we approach the curve
    print("\nProgression as we approach curve:")
    for i in range(6):
        distance = max(50.0 - i * 10.0, 5.0)
        test.vtsc._v_overshoot_distance = distance
        target_speed = test.vtsc.v_turn
        print(f"  Distance: {distance:4.1f}m -> Target: {target_speed:5.2f} m/s -> Decelerating: {test.vtsc._is_decelerating_for_curve}")

if __name__ == "__main__":
    debug_update_calculations_with_proper_mock()
    debug_anticipatory_calculation_flow()
