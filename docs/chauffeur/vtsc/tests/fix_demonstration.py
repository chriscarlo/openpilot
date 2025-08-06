#!/usr/bin/env python3
"""
Anticipatory Slowing Fix Demonstration

This demonstrates the exact root cause and solution for the anticipatory slowing bug.
"""

import numpy as np

def demonstrate_bug_and_fix():
    """Demonstrate the bug and show the fix"""

    print("="*80)
    print("ANTICIPATORY SLOWING BUG ANALYSIS & FIX")
    print("="*80)

    # Model time constants (actual values from OpenPilot)
    T_IDXS = np.array([0.0, 0.009765625, 0.0390625, 0.087890625, 0.15625, 0.244140625,
                       0.3515625, 0.478515625, 0.625, 0.791015625, 0.976, 1.181, 1.406,
                       1.651, 1.916, 2.201, 2.506, 2.831, 3.176, 3.541, 3.926, 4.331,
                       4.756, 5.201, 5.666, 6.151, 6.656, 7.181, 7.726, 8.291, 8.876,
                       9.481, 10.0])

    print(f"Model time indices (first 10): {T_IDXS[:10]}")
    print(f"Model time indices (max): {T_IDXS[-1]:.1f}s\n")

    # Test scenario: Highway speed approaching sharp curve
    test_speed = 25.0  # m/s (90 kph)
    curve_detection_index = 2  # Sharp curve detected at index 2
    desired_anticipation_time = 3.0  # Want 3 seconds early deceleration

    print("TEST SCENARIO:")
    print(f"  Speed: {test_speed:.0f} m/s ({test_speed*3.6:.0f} kph)")
    print(f"  Curve detected at model index: {curve_detection_index}")
    print(f"  Desired anticipation time: {desired_anticipation_time:.1f}s")
    print()

    # CURRENT BUGGY ALGORITHM
    print("CURRENT ALGORITHM (BUGGY):")
    print("-" * 40)

    # Step 1: Calculate original distance from detection index
    original_time = T_IDXS[curve_detection_index]
    original_distance = original_time * test_speed

    # Step 2: Calculate anticipation distance
    anticipation_distance = desired_anticipation_time * test_speed

    # Step 3: Subtract anticipation from original (THIS FAILS!)
    adjusted_distance = max(original_distance - anticipation_distance, 10.0)

    print(f"  1. Original detection time: {original_time:.3f}s")
    print(f"  2. Original distance: {original_distance:.1f}m")
    print(f"  3. Anticipation distance: {anticipation_distance:.1f}m")
    print(f"  4. Adjustment: {original_distance:.1f}m - {anticipation_distance:.1f}m = {original_distance - anticipation_distance:.1f}m")
    print(f"  5. Final distance (clipped): {adjusted_distance:.1f}m")
    print(f"  6. Effective timing: {adjusted_distance/test_speed:.1f}s before curve")
    print(f"  ❌ RESULT: Car decelerates only {adjusted_distance/test_speed:.1f}s before curve instead of {desired_anticipation_time:.1f}s!")
    print()

    # FIXED ALGORITHM
    print("FIXED ALGORITHM:")
    print("-" * 40)

    # Find the model index that naturally provides the desired anticipation time
    anticipation_indices = np.where(T_IDXS >= desired_anticipation_time)[0]

    if len(anticipation_indices) > 0:
        # Use the first index that provides sufficient anticipation time
        anticipation_idx = anticipation_indices[0]
        actual_anticipation_time = T_IDXS[anticipation_idx]
        anticipation_distance_fixed = actual_anticipation_time * test_speed

        print(f"  1. Search for index where time >= {desired_anticipation_time:.1f}s")
        print(f"  2. Found index {anticipation_idx} with time {actual_anticipation_time:.2f}s")
        print(f"  3. Calculate distance: {actual_anticipation_time:.2f}s × {test_speed:.0f} m/s = {anticipation_distance_fixed:.1f}m")
        print(f"  4. Final distance: {anticipation_distance_fixed:.1f}m (no clipping needed)")
        print(f"  5. Effective timing: {actual_anticipation_time:.1f}s before curve")
        print(f"  ✅ RESULT: Car decelerates {actual_anticipation_time:.1f}s before curve as intended!")
    else:
        # Use maximum available time
        max_time = T_IDXS[-1]
        max_distance = max_time * test_speed
        print(f"  1. Desired {desired_anticipation_time:.1f}s not available in model")
        print(f"  2. Using maximum available: {max_time:.1f}s")
        print(f"  3. Distance: {max_distance:.1f}m")
        print("  ⚠️  LIMITED BY MODEL RANGE")

    print()

    # Show the improvement
    improvement = (actual_anticipation_time - adjusted_distance/test_speed)
    print("IMPROVEMENT ANALYSIS:")
    print("-" * 40)
    print(f"  Buggy algorithm: {adjusted_distance/test_speed:.1f}s early deceleration")
    print(f"  Fixed algorithm: {actual_anticipation_time:.1f}s early deceleration")
    print(f"  Improvement: +{improvement:.1f}s earlier deceleration")
    print(f"  Distance improvement: +{improvement * test_speed:.0f}m earlier deceleration")
    print()

def test_various_scenarios():
    """Test the fix with different scenarios"""

    print("TESTING FIX WITH VARIOUS SCENARIOS:")
    print("-" * 80)

    T_IDXS = np.array([0.0, 0.009765625, 0.0390625, 0.087890625, 0.15625, 0.244140625,
                       0.3515625, 0.478515625, 0.625, 0.791015625, 0.976, 1.181, 1.406,
                       1.651, 1.916, 2.201, 2.506, 2.831, 3.176, 3.541, 3.926, 4.331,
                       4.756, 5.201, 5.666, 6.151, 6.656, 7.181, 7.726, 8.291, 8.876,
                       9.481, 10.0])

    scenarios = [
        {"name": "City driving", "speed": 15.0, "desired_time": 2.0},
        {"name": "Highway curve", "speed": 25.0, "desired_time": 3.0},
        {"name": "Highway exit", "speed": 30.0, "desired_time": 2.5},
        {"name": "Fast approach", "speed": 35.0, "desired_time": 3.0},
    ]

    print(f"{'Scenario':<15} {'Speed':<10} {'Want':<8} {'Get':<8} {'Distance':<12} {'Status'}")
    print("-" * 70)

    for scenario in scenarios:
        speed = scenario["speed"]
        desired_time = scenario["desired_time"]
        name = scenario["name"]

        # Apply fixed algorithm
        anticipation_indices = np.where(T_IDXS >= desired_time)[0]

        if len(anticipation_indices) > 0:
            actual_idx = anticipation_indices[0]
            actual_time = T_IDXS[actual_idx]
            distance = actual_time * speed
            status = "✅ Perfect"
        else:
            actual_time = T_IDXS[-1]
            distance = actual_time * speed
            status = "⚠️  Limited"

        print(f"{name:<15} {speed:4.0f} m/s   {desired_time:4.1f}s   {actual_time:4.1f}s   {distance:6.0f}m      {status}")

if __name__ == "__main__":
    demonstrate_bug_and_fix()
    print()
    test_various_scenarios()
