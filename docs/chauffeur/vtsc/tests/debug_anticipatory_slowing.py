#!/usr/bin/env python3
"""
Debug Anticipatory Slowing Logic

Test script to investigate why anticipatory slowing isn't working in real-world scenarios.
This script simulates realistic curve approach scenarios to identify potential issues.
"""

import sys
import os

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../shared'))

from vtsc_test_framework import VTSCTestBase

def debug_anticipation_time_calculation():
    """Test the anticipation time calculation with realistic values"""

    print("=== DEBUGGING ANTICIPATION TIME CALCULATION ===")

    # Import the calculate_anticipation_time function directly
    sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
    from vision_turn_controller import calculate_anticipation_time

    # Test with realistic scenarios
    scenarios = [
        {"name": "Highway exit (30m/s → 20m/s)", "v_ego": 30.0, "target": 20.0, "lat_acc": 2.0},
        {"name": "City curve (15m/s → 10m/s)", "v_ego": 15.0, "target": 10.0, "lat_acc": 1.5},
        {"name": "Sharp turn (20m/s → 8m/s)", "v_ego": 20.0, "target": 8.0, "lat_acc": 3.0},
        {"name": "Gentle curve (25m/s → 22m/s)", "v_ego": 25.0, "target": 22.0, "lat_acc": 1.0},
    ]

    for scenario in scenarios:
        anticipation_time = calculate_anticipation_time(
            scenario["v_ego"],
            scenario["target"],
            scenario["lat_acc"]
        )
        anticipation_distance = anticipation_time * scenario["v_ego"]
        print(f"{scenario['name']:30} -> {anticipation_time:.2f}s, {anticipation_distance:.1f}m")

def debug_overshoot_detection():
    """Test the overshoot detection logic with mock curvature data"""

    print("\n=== DEBUGGING OVERSHOOT DETECTION ===")

    # Create test instance
    test = VTSCTestBase()
    test.setUp()

    # Test different curvature scenarios
    curvature_scenarios = [
        {"name": "Straight road", "curvatures": [0.0, 0.0, 0.0, 0.0, 0.0]},
        {"name": "Gradual curve", "curvatures": [0.0, 0.01, 0.02, 0.03, 0.04]},
        {"name": "Sharp turn ahead", "curvatures": [0.0, 0.0, 0.05, 0.08, 0.10]},
        {"name": "Immediate sharp turn", "curvatures": [0.08, 0.10, 0.12, 0.10, 0.08]},
    ]

    # Test at different speeds
    test_speeds = [15.0, 25.0, 35.0]  # m/s (54, 90, 126 kph)

    for speed in test_speeds:
        print(f"\nTesting at {speed:.0f} m/s ({speed*3.6:.0f} kph):")
        print("-" * 50)

        for scenario in curvature_scenarios:
            test.setUp()  # Reset

            # Set test conditions
            test.vtsc._v_ego = speed

            # Create mock model data with curvature array
            mock_data = test.create_mock_model_data()
            mock_data.orientationRate.z = [curv * speed for curv in scenario["curvatures"]]
            mock_data.velocity.x = [speed] * len(scenario["curvatures"])

            # Clear previous flags
            test.vtsc._lat_acc_overshoot_ahead = False
            test.vtsc._v_overshoot_distance = 0.0
            test.vtsc._v_overshoot = speed

            # Process the data
            test.vtsc._update_vision_occlusion(mock_data, 0.0)

            print(f"  {scenario['name']:20} -> Overshoot: {test.vtsc._lat_acc_overshoot_ahead}, "
                  f"Distance: {test.vtsc._v_overshoot_distance:.1f}m, "
                  f"Target: {test.vtsc._v_overshoot:.1f} m/s")

def debug_anticipatory_deceleration_logic():
    """Test the anticipatory deceleration trigger logic"""

    print("\n=== DEBUGGING ANTICIPATORY DECELERATION LOGIC ===")

    test = VTSCTestBase()
    test.setUp()

    # Simulate a curve approach scenario
    test.vtsc._v_ego = 25.0  # 90 kph
    test.vtsc._v_cruise_setpoint = 25.0

    # Simulate curve detection
    test.vtsc._lat_acc_overshoot_ahead = True  # Manually set detection
    test.vtsc._v_overshoot = 15.0  # Target speed for curve
    test.vtsc._v_overshoot_distance = 50.0  # 50m to curve start

    print("Initial state:")
    print(f"  Speed: {test.vtsc._v_ego:.1f} m/s")
    print(f"  Overshoot ahead: {test.vtsc._lat_acc_overshoot_ahead}")
    print(f"  Target speed: {test.vtsc._v_overshoot:.1f} m/s")
    print(f"  Distance to curve: {test.vtsc._v_overshoot_distance:.1f} m")
    print(f"  Decelerating flag: {test.vtsc._is_decelerating_for_curve}")

    # Test the target speed calculation
    target_speed = test.vtsc.v_turn

    print("\nAfter target speed calculation:")
    print(f"  Target speed: {target_speed:.2f} m/s")
    print(f"  Decelerating flag: {test.vtsc._is_decelerating_for_curve}")
    print(f"  Distance recorded: {test.vtsc._curve_detection_distance:.1f} m")

    # Test multiple iterations to see progression
    print("\nProgression over time:")
    for i in range(5):
        # Simulate moving closer to curve
        test.vtsc._v_overshoot_distance = max(50.0 - i * 10.0, 5.0)
        target_speed = test.vtsc.v_turn
        print(f"  Step {i+1}: Distance: {test.vtsc._v_overshoot_distance:.1f}m, "
              f"Target: {target_speed:.2f} m/s, "
              f"Decelerating: {test.vtsc._is_decelerating_for_curve}")

def debug_distance_calculation():
    """Test distance calculation accuracy"""

    print("\n=== DEBUGGING DISTANCE CALCULATION ===")

    # Test the distance calculation logic from model time indices
    sys.path.append('/data/openpilot')
    try:
        from selfdrive.modeld.constants import ModelConstants
        print(f"ModelConstants.T_IDXS: {ModelConstants.T_IDXS[:10]}")  # First 10 values
    except ImportError:
        print("Could not import ModelConstants - using approximation")
        # Approximate time indices based on typical model prediction times
        T_IDXS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        print(f"Approximate T_IDXS: {T_IDXS}")

    # Test distance calculation at different speeds
    speeds = [15.0, 25.0, 35.0]  # m/s
    overshoot_indices = [2, 5, 8]  # Different positions in prediction array

    for speed in speeds:
        print(f"\nSpeed: {speed:.0f} m/s ({speed*3.6:.0f} kph)")
        for idx in overshoot_indices:
            try:
                time_to_overshoot = ModelConstants.T_IDXS[idx]
            except:
                time_to_overshoot = idx * 0.2  # Approximate

            distance = time_to_overshoot * speed
            print(f"  Index {idx} (t={time_to_overshoot:.1f}s) -> Distance: {distance:.1f}m")

if __name__ == "__main__":
    debug_anticipation_time_calculation()
    debug_overshoot_detection()
    debug_anticipatory_deceleration_logic()
    debug_distance_calculation()
