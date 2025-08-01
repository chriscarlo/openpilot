#!/usr/bin/env python3
"""
Debug a single scenario to understand controller behavior
"""

import numpy as np
from emergency_scenarios_definition import (
    EMERGENCY_SCENARIOS, VisionStatus
)
from progressive_deceleration_system import (
    ProgressiveDecelerationController
)
from progressive_deceleration_balanced import (
    BalancedProgressiveDecelerationController
)


def debug_scenario(controller, scenario_name: str):
    """Debug a specific scenario with detailed output"""

    # Find the scenario
    scenario = None
    for s in EMERGENCY_SCENARIOS:
        if s.name == scenario_name:
            scenario = s
            break

    if not scenario:
        print(f"Scenario '{scenario_name}' not found")
        return

    print(f"Debugging: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Initial speed: {scenario.v_ego_kph:.0f} km/h")
    print(f"Target speed: {scenario.v_target_kph:.0f} km/h")
    print(f"Distance: {scenario.distance_to_curve_m:.0f}m")
    print(f"Required decel: {scenario.required_decel_g:.2f}g")
    print(f"Vision status: {scenario.vision_status.name}")
    print("=" * 80)

    # Initial conditions
    v_ego = scenario.v_ego_ms
    v_target = scenario.v_target_ms
    distance_traveled = 0.0
    time_elapsed = 0.0
    dt = 0.05

    # Run simulation
    for i in range(100):  # Max 5 seconds
        # Calculate remaining distance
        remaining_distance = scenario.distance_to_curve_m - distance_traveled

        if remaining_distance <= 0:
            print(f"\nReached curve location at t={time_elapsed:.2f}s")
            break

        # Create mock curvature data
        if scenario.vision_status == VisionStatus.FULL_VISIBILITY:
            current_curvature = scenario.max_curvature
        else:
            current_curvature = None

        # Mock predicted curvatures
        pred_distances = np.linspace(10, max(remaining_distance, 20), 5)
        pred_curvatures = np.ones(5) * scenario.max_curvature

        # Update controller
        result = controller.update(
            v_ego=v_ego,
            current_curvature=current_curvature,
            predicted_curvatures=pred_curvatures,
            distances=pred_distances,
            vision_status=scenario.vision_status,
            model_confidence=scenario.confidence,
            lateral_acc_limit=3.05,
            current_time=time_elapsed
        )

        # Extract results
        decel_limit = result['decel_limit']
        emergency_level = result['emergency_level']
        target_level = result['target_level']
        target_speed = result['target_speed']
        required_decel = result['required_decel']

        # Apply deceleration
        actual_decel = max(decel_limit, scenario.required_decel_ms2)

        # Update vehicle state
        v_ego += actual_decel * dt
        v_ego = max(v_ego, 0)

        distance_traveled += v_ego * dt
        time_elapsed += dt

        # Print status every 0.2s
        if i % 4 == 0:
            print(f"t={time_elapsed:.2f}s: v={v_ego*3.6:5.1f}km/h, "
                  f"d_remain={remaining_distance:5.1f}m, "
                  f"a={actual_decel/9.81:5.2f}g, "
                  f"level={emergency_level.name:12s}, "
                  f"target={target_level.name:12s}")

            if abs(time_elapsed - 0.2) < 0.01:  # At 0.2s, print more details
                print(f"  Details: target_v={target_speed*3.6:.1f}km/h, "
                      f"required_a={required_decel/9.81:.2f}g, "
                      f"decel_limit={decel_limit/9.81:.2f}g")
                print(f"  Vision: degraded={result['vision_degraded']}, "
                      f"confidence={result['vision_confidence']:.2f}")

        # Check for intervention
        if result['intervention_required']:
            print(f"\nINTERVENTION REQUIRED at t={time_elapsed:.2f}s")
            print(f"  Current speed: {v_ego*3.6:.1f} km/h")
            print(f"  Target speed: {v_target*3.6:.1f} km/h")
            print(f"  Speed error: {(v_ego - v_target)/v_target*100:.1f}%")
            break

        # Check if we reached target speed
        if v_ego <= v_target * 1.05:
            print(f"\nReached target speed at t={time_elapsed:.2f}s")
            print(f"  Distance used: {distance_traveled:.1f}m")
            print(f"  Final speed: {v_ego*3.6:.1f} km/h")
            break

    # Final status
    print(f"\nFinal state at t={time_elapsed:.2f}s:")
    print(f"  Speed: {v_ego*3.6:.1f} km/h (target: {v_target*3.6:.1f} km/h)")
    print(f"  Distance traveled: {distance_traveled:.1f}m")
    print(f"  Success: {v_ego <= v_target * 1.1}")


def main():
    """Debug specific scenarios"""

    # Test the blind hairpin scenario which Original succeeded at
    print("\n" + "="*80)
    print("ORIGINAL CONTROLLER - blind_hairpin_exceeds_fov")
    print("="*80)
    controller = ProgressiveDecelerationController()
    debug_scenario(controller, 'blind_hairpin_exceeds_fov')

    print("\n" + "="*80)
    print("BALANCED CONTROLLER - blind_hairpin_exceeds_fov")
    print("="*80)
    controller = BalancedProgressiveDecelerationController()
    debug_scenario(controller, 'blind_hairpin_exceeds_fov')

    # Test a scenario that should be manageable
    print("\n" + "="*80)
    print("ORIGINAL CONTROLLER - early_warning_manageable")
    print("="*80)
    controller = ProgressiveDecelerationController()
    debug_scenario(controller, 'early_warning_manageable')

    print("\n" + "="*80)
    print("BALANCED CONTROLLER - early_warning_manageable")
    print("="*80)
    controller = BalancedProgressiveDecelerationController()
    debug_scenario(controller, 'early_warning_manageable')


if __name__ == "__main__":
    main()
