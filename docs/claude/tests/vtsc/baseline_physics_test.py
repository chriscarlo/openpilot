#!/usr/bin/env python3
"""
Baseline test runner for physics-based VTSC
Runs our existing test scenarios against the original physics version
to establish performance baseline before enhancements
"""

import numpy as np
import sys
from dataclasses import dataclass

sys.path.insert(0, '.')

# Import the physics-based VTSC
from enhanced_physics_vtsc import VisionTurnController

# Mock objects needed for testing
class MockCP:
    def __init__(self):
        self.steerRatio = 15.0
        self.wheelbase = 2.7

class MockModelData:
    def __init__(self):
        self.orientationRate = MockOrientation()
        self.velocity = MockVelocity()
        self.laneLineProbs = [0.9, 0.9, 0.9]
        self.laneLineStds = [0.1, 0.1, 0.1]

class MockOrientation:
    def __init__(self):
        self.z = []

class MockVelocity:
    def __init__(self):
        self.x = []

class MockCarState:
    def __init__(self):
        self.gasPressed = False
        self.steeringAngleDeg = 0.0

@dataclass
class TestResult:
    scenario_name: str
    success: bool
    max_decel_g: float
    final_speed_error: float
    reached_target: bool
    details: dict

def simulate_curve_scenario(controller, scenario_name: str,
                          v_ego_kph: float, curve_radius_m: float,
                          distance_to_curve_m: float,
                          vision_confidence: float = 0.9,
                          dt: float = 0.05) -> TestResult:
    """Simulate a curve scenario with the physics-based controller"""

    print(f"\nTesting: {scenario_name}")
    print(f"  Speed: {v_ego_kph} km/h")
    print(f"  Curve radius: {curve_radius_m}m")
    print(f"  Distance: {distance_to_curve_m}m")

    # Convert to SI units
    v_ego = v_ego_kph / 3.6
    curvature = 1.0 / curve_radius_m if curve_radius_m > 0 else 0.0

    # Physics target speed
    lateral_limit = 3.0  # m/s²
    v_target_physics = np.sqrt(lateral_limit / curvature) if curvature > 0 else 100.0

    # Initialize controller state
    class MockSM:
        def __init__(self):
            self.valid = {'modelV2': True}

    sm = MockSM()
    sm.modelV2 = MockModelData()
    sm.carState = MockCarState()
    sm['modelV2'] = sm.modelV2
    sm['carState'] = sm.carState

    # Make dict-like access work
    sm.__getitem__ = lambda self, key: getattr(self, key)

    # Tracking variables
    time_sim = 0.0
    distance_remaining = distance_to_curve_m
    max_decel = 0.0
    reached_target = False
    speeds = []
    decels = []

    # Run simulation
    while distance_remaining > -10 and time_sim < 10.0:
        # Simulate model predictions (33 points, ~2.5 seconds ahead)
        n_points = 33
        times = np.linspace(0, 2.5, n_points)
        distances = v_ego * times

        # Create curvature predictions
        orientation_rates = []
        velocities = []

        for i, d in enumerate(distances):
            if distance_remaining - d <= 0:  # In the curve
                # Model would detect turn rate
                turn_rate = curvature * v_ego  # rad/s
                orientation_rates.append(abs(turn_rate))
            else:
                orientation_rates.append(0.0)
            velocities.append(max(v_ego, 1.0))  # Predicted velocity

        # Update model data
        sm['modelV2'].orientationRate.z = orientation_rates
        sm['modelV2'].velocity.x = velocities

        # Update controller
        controller.update(sm, enabled=True, v_ego=v_ego, a_ego=0.0,
                         v_cruise_setpoint=v_ego_kph/3.6 + 5.0)  # Cruise slightly above current

        # Get acceleration command
        a_target = controller.a_target

        # Apply physics
        v_ego += a_target * dt
        v_ego = max(v_ego, 0)
        distance_remaining -= v_ego * dt
        time_sim += dt

        # Track metrics
        speeds.append(v_ego)
        decels.append(a_target)
        if a_target < max_decel:
            max_decel = a_target

        # Check if reached target
        if v_ego <= v_target_physics * 1.05 and not reached_target:
            reached_target = True
            print(f"  Reached target speed at t={time_sim:.1f}s, d={distance_remaining:.0f}m")

    # Calculate results
    final_speed = v_ego
    speed_error = (final_speed - v_target_physics) / v_target_physics if v_target_physics > 0 else 0

    # Determine success based on original criteria
    success = False
    if "normal" in scenario_name or "gentle" in scenario_name:
        # Normal scenarios need smooth deceleration
        success = reached_target and abs(max_decel) <= 2.45  # 0.25g
    elif "sharp" in scenario_name or "late" in scenario_name:
        # Challenging scenarios need to reach safe speed
        success = final_speed <= v_target_physics * 1.1 and abs(max_decel) <= 6.0
    else:
        # Edge cases
        success = final_speed <= v_target_physics * 1.2

    print(f"  Final speed: {final_speed*3.6:.0f} km/h (target: {v_target_physics*3.6:.0f})")
    print(f"  Max decel: {abs(max_decel)/9.81:.2f}g")
    print(f"  Success: {'✓' if success else '✗'}")

    return TestResult(
        scenario_name=scenario_name,
        success=success,
        max_decel_g=abs(max_decel)/9.81,
        final_speed_error=speed_error,
        reached_target=reached_target,
        details={
            'final_speed_kph': final_speed * 3.6,
            'target_speed_kph': v_target_physics * 3.6,
            'time_to_target': time_sim if reached_target else None,
            'min_distance': min(distance_remaining, 0)
        }
    )

def run_baseline_tests():
    """Run our standard test scenarios on the physics-based controller"""

    print("="*80)
    print("BASELINE PHYSICS VTSC TESTING")
    print("="*80)
    print("Testing original physics-based implementation")

    # Initialize controller
    CP = MockCP()
    controller = VisionTurnController(CP)

    # Define test scenarios (same as our enhanced version tests)
    test_scenarios = [
        # Normal scenarios
        ("highway_gentle_curve", 110, 250, 150, 0.95),
        ("city_normal_turn", 50, 50, 80, 0.9),
        ("suburban_curve", 70, 100, 100, 0.85),

        # Challenging scenarios
        ("highway_sharp_curve", 120, 150, 120, 0.8),
        ("late_detection_curve", 90, 60, 70, 0.7),
        ("blind_corner_moderate", 60, 40, 50, 0.4),

        # Edge cases
        ("very_late_sharp_turn", 80, 25, 35, 0.6),
        ("impossible_scenario", 100, 20, 30, 0.3),
    ]

    results = []

    for scenario_name, v_kph, radius, distance, confidence in test_scenarios:
        result = simulate_curve_scenario(
            controller, scenario_name, v_kph, radius, distance, confidence
        )
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("BASELINE RESULTS SUMMARY")
    print("="*80)

    successes = sum(1 for r in results if r.success)
    print(f"\nOverall Success Rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")

    # Categorized results
    normal_results = [r for r in results if "normal" in r.scenario_name or "gentle" in r.scenario_name]
    challenging_results = [r for r in results if "sharp" in r.scenario_name or "late" in r.scenario_name or "blind" in r.scenario_name]
    edge_results = [r for r in results if "very_late" in r.scenario_name or "impossible" in r.scenario_name]

    print(f"\nNormal scenarios: {sum(1 for r in normal_results if r.success)}/{len(normal_results)}")
    print(f"Challenging scenarios: {sum(1 for r in challenging_results if r.success)}/{len(challenging_results)}")
    print(f"Edge scenarios: {sum(1 for r in edge_results if r.success)}/{len(edge_results)}")

    # Performance metrics
    avg_decel = np.mean([r.max_decel_g for r in results])
    max_decel_overall = max(r.max_decel_g for r in results)

    print("\nPerformance Metrics:")
    print(f"- Average max deceleration: {avg_decel:.2f}g")
    print(f"- Maximum deceleration used: {max_decel_overall:.2f}g")
    print(f"- Scenarios exceeding 0.6g: {sum(1 for r in results if r.max_decel_g > 0.6)}")

    # Areas needing improvement
    print("\n" + "="*80)
    print("AREAS FOR ENHANCEMENT")
    print("="*80)

    failures = [r for r in results if not r.success]
    if failures:
        print("\nFailed scenarios:")
        for f in failures:
            print(f"- {f.scenario_name}: {f.max_decel_g:.2f}g decel, {f.final_speed_error*100:.0f}% speed error")

    # Missing features vs our enhanced version
    print("\nMissing features from enhanced version:")
    print("- No explicit emergency level system")
    print("- No vision occlusion handling")
    print("- No system constraint enforcement (-6.0 m/s²)")
    print("- No intervention triggering for impossible scenarios")
    print("- Limited jerk control")

    return results

if __name__ == "__main__":
    results = run_baseline_tests()
