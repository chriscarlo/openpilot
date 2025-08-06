#!/usr/bin/env python3
"""
Debug Distance Calculation

Investigate the distance calculation issue in anticipatory slowing.
Focus on ModelConstants.T_IDXS and the anticipation distance adjustment.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../shared'))
from vtsc_test_framework import VTSCTestBase

sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
from vision_turn_controller import curvature_to_speed, calculate_anticipation_time

def debug_model_constants():
    """Debug ModelConstants.T_IDXS values"""

    print("=== DEBUGGING MODEL CONSTANTS ===")

    try:
        sys.path.append('/data/openpilot')
        from selfdrive.modeld.constants import ModelConstants
        print(f"ModelConstants.T_IDXS (first 10): {ModelConstants.T_IDXS[:10]}")
        print(f"ModelConstants.T_IDXS (full length): {len(ModelConstants.T_IDXS)}")
        print(f"ModelConstants.T_IDXS max value: {ModelConstants.T_IDXS[-1]:.2f}s")
    except ImportError as e:
        print(f"Could not import ModelConstants: {e}")
        # Use approximation
        T_IDXS = np.linspace(0.0, 10.0, 33)  # 0 to 10 seconds
        print(f"Using approximation T_IDXS (first 10): {T_IDXS[:10]}")
        return T_IDXS

    return ModelConstants.T_IDXS

def debug_distance_calculation_details():
    """Debug the detailed distance calculation process"""

    print("\n=== DEBUGGING DISTANCE CALCULATION DETAILS ===")

    test = VTSCTestBase()
    test.setUp()

    # Test conditions
    test_speed = 25.0  # 90 kph
    sharp_curvatures = [0.0, 0.0, 0.08, 0.10, 0.12]

    print("Test setup:")
    print(f"  Speed: {test_speed:.1f} m/s ({test_speed*3.6:.0f} kph)")
    print(f"  Curvatures: {sharp_curvatures}")

    # Calculate what happens in the distance calculation
    safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in sharp_curvatures])
    overshoot_mask = safe_speeds < test_speed
    overshoot_idx = np.where(overshoot_mask)[0][0]  # Should be 2
    target_speed = safe_speeds[overshoot_idx]

    print("\nOvershoot calculation:")
    print(f"  Safe speeds: {safe_speeds}")
    print(f"  Overshoot mask: {overshoot_mask}")
    print(f"  Overshoot index: {overshoot_idx}")
    print(f"  Target speed: {target_speed:.2f} m/s")

    # Get model time constants
    T_IDXS = debug_model_constants()

    # Calculate original distance using time indices
    times = np.array(T_IDXS[:len(sharp_curvatures)])
    original_distance = times[overshoot_idx] * test_speed

    print("\nOriginal distance calculation:")
    print(f"  Time indices: {times}")
    print(f"  Time at overshoot index {overshoot_idx}: {times[overshoot_idx]:.2f}s")
    print(f"  Original distance: {original_distance:.1f}m")

    # Calculate anticipation time and distance
    max_pred_curvature = max(sharp_curvatures)  # 0.12
    max_pred_lat_acc = max_pred_curvature * test_speed**2

    anticipation_time = calculate_anticipation_time(
        test_speed,
        target_speed,
        max_pred_lat_acc
    )
    anticipation_distance = anticipation_time * test_speed

    print("\nAnticipation calculation:")
    print(f"  Max predicted curvature: {max_pred_curvature:.3f}")
    print(f"  Max predicted lat acc: {max_pred_lat_acc:.2f} m/s²")
    print(f"  Anticipation time: {anticipation_time:.2f}s")
    print(f"  Anticipation distance: {anticipation_distance:.1f}m")

    # Calculate final adjusted distance
    adjusted_distance = max(original_distance - anticipation_distance, 10.0)

    print("\nDistance adjustment:")
    print(f"  Original distance: {original_distance:.1f}m")
    print(f"  Minus anticipation: {original_distance:.1f}m - {anticipation_distance:.1f}m = {original_distance - anticipation_distance:.1f}m")
    print(f"  Final distance (clipped): {adjusted_distance:.1f}m")

    print("\nProblem analysis:")
    if adjusted_distance == 10.0:
        print("  ❌ PROBLEM: Distance clipped to minimum 10.0m")
        print(f"  ❌ This means deceleration starts only {adjusted_distance/test_speed:.1f}s before curve")
        print(f"  ❌ Instead of intended {anticipation_time:.1f}s early deceleration")
    else:
        print("  ✅ Distance adjustment working correctly")

def debug_realistic_scenario():
    """Debug with realistic highway scenario"""

    print("\n=== DEBUGGING REALISTIC HIGHWAY SCENARIO ===")

    # Highway scenario: 30 m/s (108 kph), curve ahead
    speeds = [30.0, 25.0, 20.0]  # Different highway speeds
    curvature = 0.05  # Moderate highway curve

    T_IDXS = debug_model_constants()

    for speed in speeds:
        print(f"\nSpeed: {speed:.0f} m/s ({speed*3.6:.0f} kph)")

        # Calculate target speed and anticipation
        target_speed = curvature_to_speed(curvature)
        max_pred_lat_acc = curvature * speed**2
        anticipation_time = calculate_anticipation_time(speed, target_speed, max_pred_lat_acc)
        anticipation_distance = anticipation_time * speed

        print(f"  Target speed: {target_speed:.1f} m/s")
        print(f"  Anticipation time: {anticipation_time:.1f}s")
        print(f"  Anticipation distance: {anticipation_distance:.1f}m")

        # Test different original distances (based on when curve is detected)
        detection_times = [0.5, 1.0, 2.0, 3.0, 4.0]  # Seconds ahead

        for det_time in detection_times:
            original_distance = det_time * speed
            adjusted_distance = max(original_distance - anticipation_distance, 10.0)

            effective_anticipation = original_distance / speed if adjusted_distance > 10.0 else 10.0 / speed

            result = "✅ OK" if adjusted_distance > 10.0 else "❌ CLIPPED"
            print(f"    Detect at {det_time:.1f}s ({original_distance:.0f}m) -> Final: {adjusted_distance:.0f}m -> Effective: {effective_anticipation:.1f}s {result}")

if __name__ == "__main__":
    debug_model_constants()
    debug_distance_calculation_details()
    debug_realistic_scenario()
