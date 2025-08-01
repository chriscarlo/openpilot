#!/usr/bin/env python3
"""
Expanded integrated test with 10 scenarios covering normal, challenging, and edge cases
Tests the fully integrated Enhanced VTSC with both features
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
    category: str  # normal, challenging, edge
    v_ego_kph: float
    curve_radius_m: float
    distance_to_curve_m: float
    vision_confidence: float
    expected_behavior: str


# Expanded test scenarios
TEST_SCENARIOS = [
    # Normal scenarios (3) - Must pass 100%
    TestScenario(
        name="highway_gentle_curve",
        category="normal",
        v_ego_kph=110,
        curve_radius_m=250,  # Reduced from 400m to require deceleration
        distance_to_curve_m=150,
        vision_confidence=0.95,
        expected_behavior="Smooth anticipatory deceleration"
    ),
    TestScenario(
        name="city_normal_turn",
        category="normal",
        v_ego_kph=50,
        curve_radius_m=50,
        distance_to_curve_m=80,
        vision_confidence=0.9,
        expected_behavior="Gentle anticipatory braking"
    ),
    TestScenario(
        name="suburban_curve",
        category="normal",
        v_ego_kph=70,
        curve_radius_m=100,
        distance_to_curve_m=100,
        vision_confidence=0.85,
        expected_behavior="Moderate anticipatory deceleration"
    ),

    # Challenging scenarios (5) - Must pass 80%
    TestScenario(
        name="highway_sharp_curve",
        category="challenging",
        v_ego_kph=120,
        curve_radius_m=150,
        distance_to_curve_m=120,
        vision_confidence=0.8,
        expected_behavior="Strong anticipatory deceleration"
    ),
    TestScenario(
        name="late_detection_curve",
        category="challenging",
        v_ego_kph=90,
        curve_radius_m=60,
        distance_to_curve_m=70,
        vision_confidence=0.7,
        expected_behavior="Emergency deceleration with anticipation"
    ),
    TestScenario(
        name="blind_corner_moderate",
        category="challenging",
        v_ego_kph=60,
        curve_radius_m=40,
        distance_to_curve_m=50,
        vision_confidence=0.4,
        expected_behavior="Conservative with vision uncertainty"
    ),
    TestScenario(
        name="decreasing_radius_entry",
        category="challenging",
        v_ego_kph=80,
        curve_radius_m=80,
        distance_to_curve_m=90,
        vision_confidence=0.75,
        expected_behavior="Progressive deceleration"
    ),
    TestScenario(
        name="fog_reduced_visibility",
        category="challenging",
        v_ego_kph=70,
        curve_radius_m=70,
        distance_to_curve_m=60,
        vision_confidence=0.5,
        expected_behavior="Extra caution with degraded vision"
    ),

    # Edge cases (2) - Must handle gracefully
    TestScenario(
        name="very_late_sharp_turn",
        category="edge",
        v_ego_kph=80,
        curve_radius_m=25,
        distance_to_curve_m=35,
        vision_confidence=0.6,
        expected_behavior="Maximum system deceleration"
    ),
    TestScenario(
        name="impossible_scenario",
        category="edge",
        v_ego_kph=100,
        curve_radius_m=20,
        distance_to_curve_m=30,
        vision_confidence=0.3,
        expected_behavior="Appropriate intervention request"
    )
]


def simulate_scenario(scenario: TestScenario, dt: float = 0.1, debug: bool = True):
    """Simulate a test scenario with detailed output"""

    controller = EnhancedVisionTurnSpeedController()

    if debug:
        print(f"\nScenario: {scenario.name} ({scenario.category})")
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

    if debug:
        print(f"Physics target speed: {v_target_physics*3.6:.0f} km/h")
        print(f"Required decel: {(v_ego**2 - v_target_physics**2)/(2*distance_remaining):.2f} m/s² "
              f"({abs((v_ego**2 - v_target_physics**2)/(2*distance_remaining))/9.81:.2f}g)")

    time = 0.0
    max_decel = 0.0
    max_level = EmergencyLevel.NORMAL
    reached_target = False
    intervention_triggered = False
    started_anticipation = False
    anticipation_distance = 0.0

    # Track trajectory
    trajectory = []

    # Simulation loop
    while distance_remaining > -10 and time < 10.0:  # Allow slight overshoot
        # Create predicted path (33 points as per v2 model)
        t_idxs = [10, 20, 30, 40, 50]  # ~0.5s, 1s, 1.5s, 2s, 2.5s lookahead
        distances = np.array([v_ego * t * 0.05 for t in t_idxs])
        curvatures = np.zeros(len(t_idxs))

        # Place curve in predictions
        for i, d in enumerate(distances):
            if d >= distance_remaining:
                curvatures[i] = curve_curvature

        # Update controller
        result = controller.update(
            v_ego=v_ego,
            current_curvature=curve_curvature if distance_remaining <= 0 else 0.0,
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

        # Track anticipation
        if result['using_anticipation'] and not started_anticipation:
            started_anticipation = True
            anticipation_distance = distance_remaining

        # Check if reached target speed
        if v_ego <= v_target_physics * 1.05 and not reached_target:
            reached_target = True
            if debug:
                print(f"\nReached target speed at t={time:.1f}s, d={distance_remaining:.0f}m")

        # Store trajectory point
        trajectory.append({
            'time': time,
            'speed': v_ego,
            'distance': distance_remaining,
            'level': result['emergency_level'],
            'decel': result['a_target']
        })

        # Progress output
        if debug and int(time * 10) % 10 == 0:  # Every second
            print(f"t={time:.1f}s: v={v_ego*3.6:5.0f}km/h, "
                  f"d={distance_remaining:4.0f}m, "
                  f"level={result['emergency_level'].name:11s}, "
                  f"a={result['a_target']:6.2f}m/s², "
                  f"anticipation={'ON' if result['using_anticipation'] else 'OFF'}")

    # Determine success based on category
    success = False
    if scenario.category == "normal":
        # Must reach target smoothly with minimal deceleration
        success = (reached_target and
                  abs(max_decel) <= 2.45 and  # ≤0.25g
                  not intervention_triggered and
                  started_anticipation)
    elif scenario.category == "challenging":
        # Must reach safe speed without intervention
        success = (v_ego <= v_target_physics * 1.1 and
                  abs(max_decel) <= 6.0 and
                  not intervention_triggered)
    else:  # edge
        # Either reach safe speed OR trigger appropriate intervention
        if abs((v_ego**2 - v_target_physics**2)/(2*scenario.distance_to_curve_m)) > 6.0:
            # Physics demands more than system can provide
            success = intervention_triggered or v_ego <= v_target_physics * 1.2
        else:
            # Should handle without intervention
            success = v_ego <= v_target_physics * 1.1

    # Results
    if debug:
        print("\nResults:")
        print(f"- Final speed: {v_ego*3.6:.0f} km/h (target: {v_target_physics*3.6:.0f})")
        print(f"- Max deceleration: {abs(max_decel):.2f} m/s² ({abs(max_decel)/9.81:.2f}g)")
        print(f"- Max emergency level: {max_level.name}")
        print(f"- Intervention triggered: {'Yes' if intervention_triggered else 'No'}")
        print(f"- Anticipation started: {'Yes' if started_anticipation else 'No'}")
        if started_anticipation:
            print(f"- Anticipation distance: {anticipation_distance:.0f}m")
        print(f"- Success: {'✓ PASS' if success else '✗ FAIL'}")

    return {
        'success': success,
        'category': scenario.category,
        'max_decel_g': abs(max_decel) / 9.81,
        'max_level': max_level,
        'intervention': intervention_triggered,
        'anticipation': started_anticipation,
        'final_speed_error': (v_ego - v_target_physics) / v_target_physics if v_target_physics > 0 else 0
    }


def test_all_scenarios():
    """Run all test scenarios and evaluate"""

    print("Enhanced VTSC Expanded Integration Test")
    print("="*80)
    print("Testing integrated anticipatory and emergency deceleration")
    print("System constraint: -6.0 m/s² maximum deceleration")

    results = []
    category_results = {'normal': [], 'challenging': [], 'edge': []}

    for scenario in TEST_SCENARIOS:
        result = simulate_scenario(scenario)
        results.append(result)
        category_results[scenario.category].append(result)

    # Analysis by category
    print("\n" + "="*80)
    print("RESULTS BY CATEGORY")
    print("="*80)

    for category, cat_results in category_results.items():
        successes = sum(1 for r in cat_results if r['success'])
        total = len(cat_results)
        print(f"\n{category.upper()} scenarios: {successes}/{total} ({successes/total*100:.0f}%)")

        if category == "normal" and successes < total:
            print("  WARNING: Normal scenarios must pass 100%!")

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    total = len(results)
    successful = sum(1 for r in results if r['success'])

    print(f"\nTotal Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")

    # Check against pass criteria
    normal_pass = sum(1 for r in category_results['normal'] if r['success']) == len(category_results['normal'])
    challenging_pass = sum(1 for r in category_results['challenging'] if r['success']) / len(category_results['challenging']) >= 0.8
    edge_handled = sum(1 for r in category_results['edge'] if r['success']) / len(category_results['edge']) >= 0.5
    overall_pass = successful / total >= 0.8

    print("\nPass Criteria Check:")
    print(f"- Normal 100%: {'✓ PASS' if normal_pass else '✗ FAIL'}")
    print(f"- Challenging ≥80%: {'✓ PASS' if challenging_pass else '✗ FAIL'}")
    print(f"- Edge ≥50%: {'✓ PASS' if edge_handled else '✗ FAIL'}")
    print(f"- Overall ≥80%: {'✓ PASS' if overall_pass else '✗ FAIL'}")

    # Detailed metrics
    avg_decel = np.mean([r['max_decel_g'] for r in results])
    interventions = sum(1 for r in results if r['intervention'])
    anticipations = sum(1 for r in results if r['anticipation'])

    print("\nDetailed Metrics:")
    print(f"- Average max deceleration: {avg_decel:.2f}g")
    print(f"- Interventions triggered: {interventions}")
    print(f"- Anticipation activated: {anticipations}/{total}")

    # Failure analysis
    failures = [(TEST_SCENARIOS[i], r) for i, r in enumerate(results) if not r['success']]
    if failures:
        print("\n" + "="*80)
        print("FAILURE ANALYSIS")
        print("="*80)
        for scenario, result in failures[:5]:  # Show up to 5 failures
            print(f"\n{scenario.name}:")
            print(f"  Category: {scenario.category}")
            print(f"  Max decel: {result['max_decel_g']:.2f}g")
            print(f"  Emergency level: {result['max_level'].name}")
            print(f"  Speed error: {result['final_speed_error']*100:.0f}%")

    meets_criteria = normal_pass and challenging_pass and edge_handled and overall_pass

    print("\n" + "="*80)
    if meets_criteria:
        print("✓ SUCCESS! Enhanced VTSC meets all pass criteria.")
    else:
        print("✗ Enhanced VTSC needs refinement to meet pass criteria.")
    print("="*80)

    return results, meets_criteria


if __name__ == "__main__":
    results, meets_criteria = test_all_scenarios()
