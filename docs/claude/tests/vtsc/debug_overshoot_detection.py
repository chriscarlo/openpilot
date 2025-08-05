#!/usr/bin/env python3
"""
Debug Overshoot Detection

Focus on why the _lat_acc_overshoot_ahead flag is never being set to True.
This is the critical issue preventing anticipatory slowing.
"""

import sys
import os
import numpy as np

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from vtsc_test_framework import VTSCTestBase

# Import the curvature_to_speed function directly
sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
from vision_turn_controller import curvature_to_speed

def test_curvature_to_speed_function():
    """Test the curvature_to_speed function with realistic values"""

    print("=== TESTING CURVATURE_TO_SPEED FUNCTION ===")

    curvatures = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

    for curv in curvatures:
        safe_speed = curvature_to_speed(curv)
        print(f"Curvature: {curv:6.3f} /m -> Safe speed: {safe_speed:6.2f} m/s ({safe_speed*3.6:6.1f} kph)")

def debug_overshoot_logic():
    """Debug the overshoot detection logic step by step"""

    print("\n=== DEBUGGING OVERSHOOT DETECTION LOGIC ===")

    # Test conditions that should trigger overshoot
    test_speed = 25.0  # 90 kph
    curvature_array = np.array([0.0, 0.0, 0.08, 0.10, 0.12])  # Sharp curve ahead

    print(f"Test speed: {test_speed:.1f} m/s ({test_speed*3.6:.0f} kph)")
    print(f"Curvature array: {curvature_array}")

    # Calculate safe speeds for each curvature
    safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in curvature_array])
    print(f"Safe speeds: {safe_speeds}")

    # Check overshoot mask
    overshoot_mask = safe_speeds < test_speed
    print(f"Overshoot mask: {overshoot_mask}")
    print(f"Any overshoot: {np.any(overshoot_mask)}")

    if np.any(overshoot_mask):
        overshoot_idx = np.where(overshoot_mask)[0][0]
        print(f"First overshoot at index: {overshoot_idx}")
        print(f"Safe speed at that point: {safe_speeds[overshoot_idx]:.2f} m/s")

def debug_mock_data_setup():
    """Debug how mock data is being set up in the test framework"""

    print("\n=== DEBUGGING MOCK DATA SETUP ===")

    # Create test instance
    test = VTSCTestBase()
    test.setUp()

    # Create mock model data
    mock_data = test.create_mock_model_data()

    print("Mock data structure:")
    print(f"  orientationRate: {hasattr(mock_data, 'orientationRate')}")
    print(f"  velocity: {hasattr(mock_data, 'velocity')}")

    if hasattr(mock_data, 'orientationRate'):
        print(f"  orientationRate.z: {hasattr(mock_data.orientationRate, 'z')}")
        if hasattr(mock_data.orientationRate, 'z'):
            print(f"  orientationRate.z value: {mock_data.orientationRate.z}")

    if hasattr(mock_data, 'velocity'):
        print(f"  velocity.x: {hasattr(mock_data.velocity, 'x')}")
        if hasattr(mock_data.velocity, 'x'):
            print(f"  velocity.x value: {mock_data.velocity.x}")

def debug_full_overshoot_detection():
    """Debug the complete overshoot detection process"""

    print("\n=== DEBUGGING FULL OVERSHOOT DETECTION PROCESS ===")

    test = VTSCTestBase()
    test.setUp()

    # Set test conditions
    test_speed = 25.0  # 90 kph
    test.vtsc._v_ego = test_speed

    # Create mock data with sharp curve ahead
    mock_data = test.create_mock_model_data()

    # Set orientation rates that correspond to curvatures
    # orientation_rate = curvature * velocity
    sharp_curvatures = [0.0, 0.0, 0.08, 0.10, 0.12]
    mock_data.orientationRate.z = [curv * test_speed for curv in sharp_curvatures]
    mock_data.velocity.x = [test_speed] * len(sharp_curvatures)

    print("Test setup:")
    print(f"  Speed: {test_speed:.1f} m/s")
    print(f"  Curvatures: {sharp_curvatures}")
    print(f"  Orientation rates: {mock_data.orientationRate.z}")
    print(f"  Velocities: {mock_data.velocity.x}")

    # Reset detection flags
    test.vtsc._lat_acc_overshoot_ahead = False
    test.vtsc._v_overshoot_distance = 0.0

    print("\nBefore detection:")
    print(f"  _lat_acc_overshoot_ahead: {test.vtsc._lat_acc_overshoot_ahead}")
    print(f"  _v_overshoot_distance: {test.vtsc._v_overshoot_distance}")

    # Run the vision occlusion update (which includes overshoot detection)
    test.vtsc._update_vision_occlusion(mock_data, 0.0)

    print("\nAfter detection:")
    print(f"  _lat_acc_overshoot_ahead: {test.vtsc._lat_acc_overshoot_ahead}")
    print(f"  _v_overshoot_distance: {test.vtsc._v_overshoot_distance}")
    print(f"  _v_overshoot: {test.vtsc._v_overshoot}")

    # Check if the issue is in the calculation logic itself
    print("\nManual calculation check:")
    curvature_array = np.array([abs(rate / test_speed) for rate in mock_data.orientationRate.z])
    print(f"  Calculated curvatures: {curvature_array}")

    safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in curvature_array])
    print(f"  Safe speeds: {safe_speeds}")

    overshoot_mask = safe_speeds < test_speed
    print(f"  Overshoot mask: {overshoot_mask}")
    print(f"  Should detect overshoot: {np.any(overshoot_mask)}")

if __name__ == "__main__":
    test_curvature_to_speed_function()
    debug_overshoot_logic()
    debug_mock_data_setup()
    debug_full_overshoot_detection()
