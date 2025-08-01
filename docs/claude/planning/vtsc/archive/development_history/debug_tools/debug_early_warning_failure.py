#!/usr/bin/env python3
"""
Debug why early_warning_manageable is failing
"""

import numpy as np
from emergency_scenarios_definition import EMERGENCY_SCENARIOS
from progressive_deceleration_iteration_2 import ProgressiveDecelerationV2


def debug_early_warning():
    """Debug the early_warning_manageable scenario"""

    scenario = next(s for s in EMERGENCY_SCENARIOS if s.name == 'early_warning_manageable')

    print("Debugging early_warning_manageable")
    print("="*60)
    print(f"Speed: {scenario.v_ego_kph:.0f} → {scenario.v_target_kph:.0f} km/h")
    print(f"Distance: {scenario.distance_to_curve_m:.0f}m")
    print(f"Required: {scenario.required_decel_g:.2f}g")
    print(f"Curvature: {scenario.max_curvature:.3f}")

    # Calculate physics target
    lateral_limit = 3.05
    physics_target = np.sqrt(lateral_limit / scenario.max_curvature)
    print(f"\nPhysics target speed: {physics_target*3.6:.0f} km/h")
    print(f"Scenario target: {scenario.v_target_ms*3.6:.0f} km/h")

    controller = ProgressiveDecelerationV2()

    v_ego = scenario.v_ego_ms
    distance_traveled = 0.0
    time = 0.0
    dt = 0.05

    print("\nSimulation:")
    print("time | speed | remain | level | target_level | decel | crit_dist | req_decel")
    print("-"*80)

    for i in range(20):  # First 1 second
        remaining = scenario.distance_to_curve_m - distance_traveled

        # Predicted distances
        pred_distances = np.linspace(
            min(10, remaining),
            min(remaining + 30, 100),
            5
        )
        pred_curvatures = np.ones(5) * scenario.max_curvature

        result = controller.update(
            v_ego=v_ego,
            current_curvature=scenario.max_curvature,  # Full visibility
            predicted_curvatures=pred_curvatures,
            distances=pred_distances,
            vision_status=scenario.vision_status,
            model_confidence=scenario.confidence,
            lateral_acc_limit=lateral_limit,
            current_time=time
        )

        print(f"{time:4.2f} | {v_ego*3.6:5.0f} | {remaining:6.0f} | "
              f"{result['emergency_level'].name:11s} | "
              f"{result['target_level'].name:11s} | "
              f"{result['decel_limit']/9.81:5.2f}g | "
              f"{result['critical_distance']:9.0f} | "
              f"{abs(result['required_decel'])/9.81:5.2f}g")

        # Update state
        v_ego += result['decel_limit'] * dt
        distance_traveled += v_ego * dt
        time += dt

        if result['intervention_required']:
            print(f"\nINTERVENTION at t={time:.2f}s")
            break

        if v_ego <= result['target_speed'] * 1.05:
            print(f"\nSUCCESS at t={time:.2f}s")
            break

    # Analyze the issue
    print("\nAnalysis:")
    print(f"Target speed from controller: {result['target_speed']*3.6:.0f} km/h")
    print(f"Required decel: {abs(result['required_decel'])/9.81:.2f}g")
    print(f"Critical distance: {result['critical_distance']:.0f}m")


if __name__ == "__main__":
    debug_early_warning()
