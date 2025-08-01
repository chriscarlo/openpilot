#!/usr/bin/env python3
"""
Debug emergency level determination
"""

import numpy as np
from emergency_scenarios_definition import (
    EMERGENCY_SCENARIOS, EmergencyLevel, VisionStatus
)
from progressive_deceleration_balanced import (
    BalancedProgressiveDecelerationController
)


def debug_level_determination():
    """Debug how emergency levels are determined"""

    controller = BalancedProgressiveDecelerationController()

    # Test scenarios
    test_cases = [
        # (name, v_ego_ms, v_target_ms, distance_m, vision_status)
        ("Normal highway", 28.0, 22.0, 70.0, VisionStatus.FULL_VISIBILITY),
        ("Blind hairpin", 15.0, 7.0, 40.0, VisionStatus.CURVE_EXCEEDS_FOV),
        ("Late detection", 25.0, 15.0, 30.0, VisionStatus.PARTIAL_OCCLUSION),
        ("Emergency stop", 30.0, 10.0, 40.0, VisionStatus.LOST_ROAD),
    ]

    print("Emergency Level Determination Debug")
    print("=" * 80)

    for name, v_ego, v_target, distance, vision in test_cases:
        print(f"\n{name}:")
        print(f"  v_ego: {v_ego*3.6:.0f} km/h")
        print(f"  v_target: {v_target*3.6:.0f} km/h")
        print(f"  distance: {distance:.0f}m")
        print(f"  vision: {vision.name}")

        # Calculate basic deceleration
        if distance > 0 and v_ego > v_target:
            required_decel = -(v_ego**2 - v_target**2) / (2 * distance)
            required_g = abs(required_decel) / 9.81
        else:
            required_decel = 0.0
            required_g = 0.0

        print(f"  required_decel: {required_decel:.2f} m/s² ({required_g:.2f}g)")

        # Determine base level
        if required_g <= 0.15:
            base_level = EmergencyLevel.NORMAL
        elif required_g <= 0.25:
            base_level = EmergencyLevel.CAUTION
        elif required_g <= 0.40:
            base_level = EmergencyLevel.WARNING
        elif required_g <= 0.60:
            base_level = EmergencyLevel.CRITICAL
        else:
            base_level = EmergencyLevel.INTERVENTION

        print(f"  base_level: {base_level.name}")

        # Check distance escalation
        if distance < 30 and required_g > 0.5:
            print(f"  Distance escalation triggered (d={distance}m, g={required_g:.2f})")
            escalated_level = EmergencyLevel(min(base_level.value + 1, EmergencyLevel.INTERVENTION.value))
            print(f"  After distance escalation: {escalated_level.name}")
        else:
            escalated_level = base_level

        # Vision boost
        final_level = escalated_level
        if vision == VisionStatus.LOST_ROAD:
            final_level = EmergencyLevel(min(
                escalated_level.value + 1,
                EmergencyLevel.CRITICAL.value
            ))
            print(f"  Vision boost for LOST_ROAD: {escalated_level.name} -> {final_level.name}")

        print(f"  FINAL LEVEL: {final_level.name}")

    # Now test with actual scenarios
    print("\n" + "="*80)
    print("Testing Actual Scenarios")
    print("="*80)

    for scenario in EMERGENCY_SCENARIOS[:3]:  # First 3 scenarios
        print(f"\n{scenario.name}:")

        # Calculate what the controller would determine
        controller = BalancedProgressiveDecelerationController()

        # Mock some predicted curvatures
        pred_curvatures = np.array([scenario.max_curvature] * 5)
        pred_distances = np.array([10, 20, 30, 40, 50])

        # Get safe speed for curve
        v_safe = controller.calculate_safe_speed_for_curve(
            scenario.max_curvature, 3.05
        )

        print(f"  Max curvature: {scenario.max_curvature:.3f} (radius: {1/scenario.max_curvature:.0f}m)")
        print(f"  Safe speed for curve: {v_safe*3.6:.0f} km/h")
        print(f"  Current speed: {scenario.v_ego_kph:.0f} km/h")
        print(f"  Expected target: {scenario.v_target_kph:.0f} km/h")

        # Determine required deceleration
        distance = scenario.distance_to_curve_m
        required_decel, level = controller.determine_required_deceleration(
            scenario.v_ego_ms, v_safe, distance
        )

        print(f"  Required decel to v_safe: {abs(required_decel)/9.81:.2f}g")
        print(f"  Base emergency level: {level.name}")

        # Check if vision would boost it
        if scenario.vision_status == VisionStatus.LOST_ROAD:
            boosted_level = EmergencyLevel(min(level.value + 1, EmergencyLevel.CRITICAL.value))
            print(f"  After vision boost: {boosted_level.name}")


if __name__ == "__main__":
    debug_level_determination()
