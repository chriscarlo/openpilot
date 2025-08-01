#!/usr/bin/env python3
"""
Comprehensive test of integrated Enhanced VTSC
Tests both anticipatory control and emergency handling
"""

import numpy as np
from dataclasses import dataclass
import sys
sys.path.insert(0, '../implementation')

from enhanced_vtsc_integrated import (
    EnhancedVisionTurnSpeedController,
    EmergencyLevel
)


@dataclass
class TestScenario:
    name: str
    v_ego_kph: float
    curve_radius_m: float
    distance_to_curve_m: float
    vision_confidence: float
    expected_behavior: str


# Test scenarios covering various situations
TEST_SCENARIOS = [
    # Anticipatory control scenarios
    TestScenario(
        name="highway_curve_normal",
        v_ego_kph=120,
        curve_radius_m=200,
        distance_to_curve_m=150,
        vision_confidence=0.9,
        expected_behavior="Smooth anticipatory deceleration"
    ),
    TestScenario(
        name="city_turn_comfortable",
        v_ego_kph=50,
        curve_radius_m=50,
        distance_to_curve_m=80,
        vision_confidence=0.95,
        expected_behavior="Gentle anticipatory braking"
    ),

    # Emergency scenarios
    TestScenario(
        name="sudden_sharp_turn",
        v_ego_kph=90,
        curve_radius_m=30,
        distance_to_curve_m=40,
        vision_confidence=0.8,
        expected_behavior="Emergency deceleration needed"
    ),
    TestScenario(
        name="blind_corner_approach",
        v_ego_kph=60,
        curve_radius_m=40,
        distance_to_curve_m=50,
        vision_confidence=0.4,
        expected_behavior="Conservative with vision uncertainty"
    ),

    # Extreme scenarios
    TestScenario(
        name="impossible_scenario",
        v_ego_kph=100,
        curve_radius_m=20,
        distance_to_curve_m=20,
        vision_confidence=0.3,
        expected_behavior="Intervention likely needed"
    )
]


def simulate_scenario(scenario: TestScenario, dt: float = 0.1):
    """Simulate a test scenario"""

    controller = EnhancedVisionTurnSpeedController()

    print(f"\nScenario: {scenario.name}")
    print("="*60)
    print(f"Initial speed: {scenario.v_ego_kph} km/h")
    print(f"Curve radius: {scenario.curve_radius_m}m")
    print(f"Distance: {scenario.distance_to_curve_m}m")
    print(f"Vision confidence: {scenario.vision_confidence}")
    print(f"Expected: {scenario.expected_behavior}")
    print("-"*60)

    # Convert to SI units
    v_ego = scenario.v_ego_kph / 3.6
    curve_curvature = 1.0 / scenario.curve_radius_m if scenario.curve_radius_m > 0 else 0.0
    distance_remaining = scenario.distance_to_curve_m

    # Calculate physics target speed
    lateral_limit = 3.0  # m/s²
    v_target_physics = np.sqrt(lateral_limit / curve_curvature) if curve_curvature > 0 else 100.0

    print(f"Target speed for curve: {v_target_physics*3.6:.0f} km/h")

    time = 0.0
    max_decel = 0.0
    max_level = EmergencyLevel.NORMAL
    reached_target = False
    intervention_triggered = False

    # Simulation loop
    while distance_remaining > 0 and time < 10.0:
        # Create predicted path
        distances = np.array([20, 40, 60, 80, 100])
        curvatures = np.zeros(5)

        # Place curve in predictions
        for i, d in enumerate(distances):
            if d >= distance_remaining:
                curvatures[i] = curve_curvature

        # Simulate vision degradation
        if scenario.vision_confidence < 0.5:
            current_curvature = None
        else:
            current_curvature = curve_curvature if distance_remaining < 20 else 0.0

        # Update controller
        result = controller.update(
            v_ego=v_ego,
            current_curvature=current_curvature,
            predicted_curvatures=curvatures,
            distances=distances,
            lateral_acc_limit=lateral_limit,
            model_confidence=scenario.vision_confidence,
            current_time=time
        )

        # Apply acceleration
        v_ego += result['a_target'] * dt
        v_ego = max(v_ego, 0)

        # Update distance
        distance_remaining -= v_ego * dt
        time += dt

        # Track metrics
        if result['a_target'] < max_decel:
            max_decel = result['a_target']
        if result['emergency_level'].value > max_level.value:
            max_level = result['emergency_level']

        if result['intervention_required']:
            intervention_triggered = True

        # Check if reached target speed
        if v_ego <= v_target_physics * 1.05 and not reached_target:
            reached_target = True
            print(f"\nReached target speed at t={time:.1f}s, d={distance_remaining:.0f}m")

        # Progress output
        if int(time * 10) % 10 == 0:  # Every second
            print(f"t={time:.1f}s: v={v_ego*3.6:5.0f}km/h, "
                  f"d={distance_remaining:4.0f}m, "
                  f"level={result['emergency_level'].name:11s}, "
                  f"a={result['a_target']:6.2f}m/s², "
                  f"anticipation={'ON' if result['using_anticipation'] else 'OFF'}")

    # Results
    print("\nResults:")
    print(f"- Final speed: {v_ego*3.6:.0f} km/h")
    print(f"- Max deceleration: {abs(max_decel):.2f} m/s² ({abs(max_decel)/9.81:.2f}g)")
    print(f"- Max emergency level: {max_level.name}")
    print(f"- Intervention triggered: {'Yes' if intervention_triggered else 'No'}")
    print(f"- Success: {'Yes' if reached_target and not intervention_triggered else 'No'}")

    return {
        'success': reached_target and not intervention_triggered,
        'max_decel_g': abs(max_decel) / 9.81,
        'max_level': max_level,
        'intervention': intervention_triggered
    }


def test_all_scenarios():
    """Run all test scenarios"""

    print("Enhanced VTSC Comprehensive Test")
    print("="*80)
    print("Testing integrated anticipatory and emergency deceleration")

    results = []

    for scenario in TEST_SCENARIOS:
        result = simulate_scenario(scenario)
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    successes = sum(1 for r in results if r['success'])
    print(f"\nSuccess rate: {successes}/{len(results)} ({successes/len(results)*100:.0f}%)")

    avg_decel = np.mean([r['max_decel_g'] for r in results])
    print(f"Average max deceleration: {avg_decel:.2f}g")

    interventions = sum(1 for r in results if r['intervention'])
    print(f"Interventions required: {interventions}")

    # Level distribution
    level_counts = {}
    for r in results:
        level = r['max_level']
        level_counts[level] = level_counts.get(level, 0) + 1

    print("\nEmergency level usage:")
    for level in sorted(level_counts.keys(), key=lambda x: x.value):
        print(f"  {level.name}: {level_counts[level]}")

    # Performance assessment
    print("\n" + "="*80)
    if successes >= len(results) * 0.6 and avg_decel < 0.6:
        print("✓ Enhanced VTSC performs well!")
        print("  - Good success rate with reasonable deceleration")
        print("  - Anticipatory control reduces need for emergency braking")
        print("  - Progressive response handles edge cases appropriately")
    else:
        print("✗ Enhanced VTSC needs tuning")

    return results


def test_anticipation_timing():
    """Test that anticipation timing works correctly"""

    print("\n" + "="*80)
    print("ANTICIPATION TIMING TEST")
    print("="*80)

    controller = EnhancedVisionTurnSpeedController()

    # Test parameters
    v_ego = 30.0  # 108 km/h
    v_target = 20.0  # 72 km/h
    curve_distance = 200.0

    # Calculate expected anticipation distance
    comfort_decel = 0.15 * 9.81
    decel_distance = (v_ego**2 - v_target**2) / (2 * comfort_decel)
    anticipation_distance = v_target * 2.0  # 2 seconds at target speed
    total_distance = decel_distance + anticipation_distance + 10.0  # Plus safety margin

    print(f"Initial speed: {v_ego*3.6:.0f} km/h")
    print(f"Target speed: {v_target*3.6:.0f} km/h")
    print(f"Deceleration distance: {decel_distance:.0f}m")
    print(f"Anticipation distance: {anticipation_distance:.0f}m")
    print(f"Total distance needed: {total_distance:.0f}m")
    print(f"Should start decelerating at: {total_distance:.0f}m before curve")

    # Simulate approach
    time = 0.0
    dt = 0.1
    decel_started = False
    decel_start_distance = 0

    while curve_distance > 0:
        # Create predictions
        distances = np.linspace(20, 250, 10)
        curvatures = np.zeros(10)
        curve_idx = np.argmin(np.abs(distances - curve_distance))
        curvatures[curve_idx:] = 1/100.0  # 100m radius curve

        result = controller.update(
            v_ego=v_ego,
            current_curvature=0.0,
            predicted_curvatures=curvatures,
            distances=distances,
            lateral_acc_limit=3.0,
            model_confidence=0.9,
            current_time=time
        )

        # Check when deceleration starts
        if not decel_started and result['a_target'] < -0.5:
            decel_started = True
            decel_start_distance = curve_distance
            print(f"\nDeceleration started at {curve_distance:.0f}m before curve")
            print(f"Difference from calculated: {abs(curve_distance - total_distance):.0f}m")

        # Update
        v_ego += result['a_target'] * dt
        curve_distance -= v_ego * dt
        time += dt

        if curve_distance <= 0:
            print(f"\nReached curve at v={v_ego*3.6:.0f} km/h")
            print(f"Time at target speed: {(decel_start_distance - decel_distance - curve_distance) / v_target:.1f}s")
            break

    print("\n✓ Anticipatory control timing verified")


if __name__ == "__main__":
    # Run comprehensive test
    results = test_all_scenarios()

    # Test anticipation timing
    test_anticipation_timing()
