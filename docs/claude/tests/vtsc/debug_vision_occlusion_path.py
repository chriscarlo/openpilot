#!/usr/bin/env python3
"""
Debug Vision Occlusion Execution Path

Trace the exact execution path through _update_vision_occlusion to find where
the overshoot detection is failing.
"""

import sys
import os
import numpy as np

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from vtsc_test_framework import VTSCTestBase

# Import directly from production VTSC
sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
from vision_turn_controller import curvature_to_speed

def debug_vision_occlusion_execution():
    """Debug the exact execution path in _update_vision_occlusion"""

    print("=== DEBUGGING VISION OCCLUSION EXECUTION PATH ===")

    test = VTSCTestBase()
    test.setUp()

    # Set test conditions
    test_speed = 25.0
    test.vtsc._v_ego = test_speed

    # Create mock data with sharp curve
    mock_data = test.create_mock_model_data()
    sharp_curvatures = [0.0, 0.0, 0.08, 0.10, 0.12]
    mock_data.orientationRate.z = [curv * test_speed for curv in sharp_curvatures]
    mock_data.velocity.x = [test_speed] * len(sharp_curvatures)

    print("Mock data setup:")
    print(f"  orientationRate.z length: {len(mock_data.orientationRate.z)}")
    print(f"  velocity.x length: {len(mock_data.velocity.x)}")
    print(f"  orientationRate.z: {mock_data.orientationRate.z[:5]}")
    print(f"  velocity.x: {mock_data.velocity.x[:5]}")

    # Check if model data passes initial conditions
    print("\nChecking _update_vision_occlusion entry conditions:")
    print(f"  model_data is not None: {mock_data is not None}")
    print(f"  hasattr orientationRate: {hasattr(mock_data, 'orientationRate')}")
    print(f"  hasattr velocity: {hasattr(mock_data, 'velocity')}")
    print(f"  orientationRate.z is not None: {mock_data.orientationRate.z is not None}")
    print(f"  velocity.x is not None: {mock_data.velocity.x is not None}")

    # Check the condition for entering advanced method
    entry_condition = (mock_data is not None and
                      hasattr(mock_data, 'orientationRate') and hasattr(mock_data, 'velocity') and
                      mock_data.orientationRate.z is not None and mock_data.velocity.x is not None)
    print(f"  Entry condition met: {entry_condition}")

    if entry_condition:
        orientation_rate_raw = mock_data.orientationRate.z
        velocity_pred_raw = mock_data.velocity.x

        MIN_POINTS = 3
        min_points_condition = (len(orientation_rate_raw) >= MIN_POINTS and len(velocity_pred_raw) >= MIN_POINTS)
        print(f"  MIN_POINTS condition: {min_points_condition} (need {MIN_POINTS}, have {len(orientation_rate_raw)}, {len(velocity_pred_raw)})")

        if min_points_condition:
            # Import N_POINTS constant
            try:
                from vision_turn_controller import N_POINTS
                print(f"  N_POINTS: {N_POINTS}")
            except ImportError:
                N_POINTS = 33  # Fallback
                print(f"  N_POINTS (fallback): {N_POINTS}")

            n_points = int(min(len(orientation_rate_raw), len(velocity_pred_raw), N_POINTS))
            print(f"  n_points calculated: {n_points}")

            orientation_rate = np.abs(np.array(list(orientation_rate_raw)[:n_points], dtype=float))
            velocity_pred = np.array(list(velocity_pred_raw)[:n_points], dtype=float)

            print(f"  orientation_rate: {orientation_rate[:5]}")
            print(f"  velocity_pred: {velocity_pred[:5]}")

            # Compute curvature array
            eps = 1e-9
            curvature_array = orientation_rate / np.clip(velocity_pred, eps, None)
            print(f"  curvature_array: {curvature_array[:5]}")

            max_pred_curvature = float(np.max(curvature_array))
            print(f"  max_pred_curvature: {max_pred_curvature}")

            # This is where the overshoot detection should happen
            print("\n  Now testing overshoot detection:")
            safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in curvature_array])
            print(f"  safe_speeds: {safe_speeds[:5]}")

            overshoot_mask = safe_speeds < test_speed
            print(f"  overshoot_mask: {overshoot_mask[:5]}")

            should_detect_overshoot = np.any(overshoot_mask)
            print(f"  should_detect_overshoot: {should_detect_overshoot}")

    # Now call the actual method and see what happens
    print("\n=== CALLING ACTUAL _update_vision_occlusion ===")

    test.vtsc._lat_acc_overshoot_ahead = False  # Reset
    test.vtsc._v_overshoot_distance = 0.0

    result = test.vtsc._update_vision_occlusion(mock_data, 0.0)

    print("After _update_vision_occlusion:")
    print(f"  _lat_acc_overshoot_ahead: {test.vtsc._lat_acc_overshoot_ahead}")
    print(f"  _v_overshoot_distance: {test.vtsc._v_overshoot_distance}")
    print(f"  _v_overshoot: {test.vtsc._v_overshoot}")
    print(f"  Return value: {result}")

def debug_update_calculations_path():
    """Debug the _update_calculations method that calls _update_vision_occlusion"""

    print("\n=== DEBUGGING _update_calculations PATH ===")

    test = VTSCTestBase()
    test.setUp()

    # Set test conditions
    test_speed = 25.0
    test.vtsc._v_ego = test_speed

    # Create mock submasters
    sm = test.create_mock_submasters()

    # Create mock model data and add to submasters
    mock_model_data = test.create_mock_model_data()
    sharp_curvatures = [0.0, 0.0, 0.08, 0.10, 0.12]
    mock_model_data.orientationRate.z = [curv * test_speed for curv in sharp_curvatures]
    mock_model_data.velocity.x = [test_speed] * len(sharp_curvatures)

    # Set the model data in submasters
    sm['modelV2'] = mock_model_data
    sm.valid['modelV2'] = True

    print("Submasters setup:")
    print(f"  sm.valid['modelV2']: {sm.valid.get('modelV2', False)}")
    print(f"  sm['modelV2'] exists: {'modelV2' in sm}")

    # Reset detection flags
    test.vtsc._lat_acc_overshoot_ahead = False
    test.vtsc._v_overshoot_distance = 0.0

    print("\nBefore _update_calculations:")
    print(f"  _lat_acc_overshoot_ahead: {test.vtsc._lat_acc_overshoot_ahead}")

    # Call _update_calculations (which calls _update_vision_occlusion)
    test.vtsc._update_calculations(sm)

    print("\nAfter _update_calculations:")
    print(f"  _lat_acc_overshoot_ahead: {test.vtsc._lat_acc_overshoot_ahead}")
    print(f"  _v_overshoot_distance: {test.vtsc._v_overshoot_distance}")
    print(f"  _v_overshoot: {test.vtsc._v_overshoot}")

if __name__ == "__main__":
    debug_vision_occlusion_execution()
    debug_update_calculations_path()
