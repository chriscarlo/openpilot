#!/usr/bin/env python3
"""
Fix for Anticipatory Slowing Distance Calculation

This script demonstrates the proper fix for the anticipatory slowing malfunction.
The issue is that the current algorithm subtracts anticipation_distance from tiny
original distances based on early model time indices, causing clipping to 10.0m.

The solution is to find the model index that naturally provides the desired
anticipation distance, ensuring proper early deceleration timing.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
from vision_turn_controller import curvature_to_speed, calculate_anticipation_time

def demonstrate_current_bug():
    """Demonstrate the current bug in the distance calculation"""

    print("=== DEMONSTRATING CURRENT BUG ===")

    # Get model constants
    sys.path.append('/data/openpilot')
    from selfdrive.modeld.constants import ModelConstants

    # Test scenario
    test_speed = 25.0  # 90 kph
    sharp_curvatures = [0.0, 0.0, 0.08, 0.10, 0.12]

    # Current algorithm (BUGGY)
    safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in sharp_curvatures])
    overshoot_mask = safe_speeds < test_speed
    overshoot_idx = np.where(overshoot_mask)[0][0]  # Index 2

    # Current distance calculation (BROKEN)
    times = np.array(ModelConstants.T_IDXS[:len(sharp_curvatures)])
    original_distance = times[overshoot_idx] * test_speed  # 0.04s * 25 m/s = 1.0m

    # Current anticipation calculation
    target_speed = safe_speeds[overshoot_idx]
    max_pred_curvature = max(sharp_curvatures)
    anticipation_time = calculate_anticipation_time(test_speed, target_speed, max_pred_curvature * test_speed**2)
    anticipation_distance = anticipation_time * test_speed  # 3.0s * 25 m/s = 75.0m

    # Current adjustment (FAILS)
    adjusted_distance = max(original_distance - anticipation_distance, 10.0)  # 1.0 - 75.0 = -74.0 → 10.0

    print("Current Algorithm (BUGGY):")
    print(f"  Overshoot detected at index: {overshoot_idx}")
    print(f"  Original distance: {original_distance:.1f}m (time: {times[overshoot_idx]:.3f}s)")
    print(f"  Anticipation time: {anticipation_time:.1f}s")
    print(f"  Anticipation distance: {anticipation_distance:.1f}m")
    print(f"  Adjusted distance: {adjusted_distance:.1f}m ❌ CLIPPED TO MINIMUM")
    print(f"  Effective deceleration start: {adjusted_distance/test_speed:.1f}s before curve ❌ TOO LATE")

def demonstrate_correct_fix():
    """Demonstrate the correct fix for the distance calculation"""

    print("\n=== DEMONSTRATING CORRECT FIX ===")

    # Get model constants
    sys.path.append('/data/openpilot')
    from selfdrive.modeld.constants import ModelConstants

    # Test scenario
    test_speed = 25.0  # 90 kph
    sharp_curvatures = [0.0, 0.0, 0.08, 0.10, 0.12]

    # Step 1: Find overshoot (same as before)
    safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in sharp_curvatures])
    overshoot_mask = safe_speeds < test_speed
    overshoot_idx = np.where(overshoot_mask)[0][0]
    target_speed = safe_speeds[overshoot_idx]
    max_pred_curvature = max(sharp_curvatures)

    # Step 2: Calculate desired anticipation time
    anticipation_time = calculate_anticipation_time(test_speed, target_speed, max_pred_curvature * test_speed**2)

    print("Fixed Algorithm:")
    print(f"  Overshoot detected at index: {overshoot_idx}")
    print(f"  Target speed: {target_speed:.2f} m/s")
    print(f"  Desired anticipation time: {anticipation_time:.1f}s")

    # Step 3: FIXED APPROACH - Find the model index that provides the desired anticipation distance
    times = np.array(ModelConstants.T_IDXS)

    # Find the index where the time matches our desired anticipation time
    # We want to find the latest index where time >= anticipation_time
    anticipation_indices = np.where(times >= anticipation_time)[0]

    if len(anticipation_indices) > 0:
        # Use the first index that provides sufficient anticipation time
        anticipation_idx = anticipation_indices[0]
        anticipation_distance = times[anticipation_idx] * test_speed

        print(f"  Found anticipation index: {anticipation_idx}")
        print(f"  Anticipation time at index: {times[anticipation_idx]:.2f}s")
        print(f"  Anticipation distance: {anticipation_distance:.1f}m")
        print(f"  ✅ PROPER EARLY DECELERATION: {times[anticipation_idx]:.1f}s before curve")

        return anticipation_distance, anticipation_idx
    else:
        # Fallback to maximum available time
        max_idx = len(times) - 1
        max_distance = times[max_idx] * test_speed
        print(f"  Using maximum available time: {times[max_idx]:.1f}s")
        print(f"  Maximum distance: {max_distance:.1f}m")
        print("  ⚠️  LIMITED BY MODEL PREDICTION RANGE")

        return max_distance, max_idx

def test_fix_with_various_scenarios():
    """Test the fix with various speed and curvature scenarios"""

    print("\n=== TESTING FIX WITH VARIOUS SCENARIOS ===")

    sys.path.append('/data/openpilot')
    from selfdrive.modeld.constants import ModelConstants
    times = np.array(ModelConstants.T_IDXS)

    scenarios = [
        {"name": "City curve", "speed": 15.0, "curvature": 0.06},
        {"name": "Highway curve", "speed": 25.0, "curvature": 0.05},
        {"name": "Highway exit", "speed": 30.0, "curvature": 0.08},
        {"name": "Sharp turn", "speed": 20.0, "curvature": 0.12},
    ]

    for scenario in scenarios:
        speed = scenario["speed"]
        curvature = scenario["curvature"]

        # Calculate target speed and anticipation time
        target_speed = curvature_to_speed(curvature)
        anticipation_time = calculate_anticipation_time(speed, target_speed, curvature * speed**2)

        # Apply the fix - find appropriate index
        anticipation_indices = np.where(times >= anticipation_time)[0]

        if len(anticipation_indices) > 0:
            anticipation_idx = anticipation_indices[0]
            actual_time = times[anticipation_idx]
            distance = actual_time * speed
            result = f"✅ {actual_time:.1f}s ({distance:.0f}m)"
        else:
            max_time = times[-1]
            max_distance = max_time * speed
            result = f"⚠️  {max_time:.1f}s ({max_distance:.0f}m) LIMITED"

        print(f"  {scenario['name']:12} ({speed:.0f} m/s, curv {curvature:.3f}): Want {anticipation_time:.1f}s → Got {result}")

def create_fixed_algorithm_code():
    """Generate the actual code fix for the production VTSC"""

    print("\n=== GENERATED CODE FIX ===")

    fix_code = '''
# FIXED ANTICIPATORY SLOWING DISTANCE CALCULATION
# Replace lines 610-623 in _update_calculations method

if self._lat_acc_overshoot_ahead:
    overshoot_idx = np.where(overshoot_mask)[0][0]
    self._v_overshoot = min(safe_speeds[overshoot_idx], self._v_cruise_setpoint)
    
    # Calculate desired anticipation time for early deceleration
    anticipation_time = calculate_anticipation_time(
        self._v_ego,
        self._v_overshoot,
        max_pred_curvature * self._v_ego**2
    )
    
    # FIXED APPROACH: Find model index that provides desired anticipation time
    # Instead of subtracting from tiny original distance, find appropriate index
    times = np.array(ModelConstants.T_IDXS[:n_points])
    anticipation_indices = np.where(times >= anticipation_time)[0]
    
    if len(anticipation_indices) > 0:
        # Use the first index that provides sufficient anticipation time
        anticipation_idx = anticipation_indices[0]
        self._v_overshoot_distance = max(times[anticipation_idx] * self._v_ego, 10.0)
        actual_anticipation_time = times[anticipation_idx]
    else:
        # Fallback to maximum available prediction time
        max_idx = len(times) - 1
        self._v_overshoot_distance = max(times[max_idx] * self._v_ego, 10.0)
        actual_anticipation_time = times[max_idx]
    
    _debug(f'TVC: Fixed Anticipatory Slowing. Dist: {self._v_overshoot_distance:.2f}m, '
           f'v: {self._v_overshoot * CV.MS_TO_KPH:.2f}kph, '
           f'anticipation: {actual_anticipation_time:.1f}s')
'''

    print(fix_code)

if __name__ == "__main__":
    demonstrate_current_bug()
    demonstrate_correct_fix()
    test_fix_with_various_scenarios()
    create_fixed_algorithm_code()
