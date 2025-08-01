#!/usr/bin/env python3
"""
Debug why controllers are triggering intervention immediately
"""

import numpy as np
from emergency_scenarios_definition import (
    EMERGENCY_SCENARIOS
)
from progressive_deceleration_final import (
    FinalProgressiveDecelerationController
)


def debug_first_update():
    """Debug what happens in the very first controller update"""

    # Use blind_hairpin_exceeds_fov scenario
    scenario = next(s for s in EMERGENCY_SCENARIOS if s.name == 'blind_hairpin_exceeds_fov')

    print("Debugging First Update - blind_hairpin_exceeds_fov")
    print("="*80)
    print("Initial conditions:")
    print(f"  v_ego: {scenario.v_ego_ms*3.6:.1f} km/h")
    print(f"  Distance: {scenario.distance_to_curve_m:.1f}m")
    print(f"  Max curvature: {scenario.max_curvature:.3f} (r={1/scenario.max_curvature:.0f}m)")
    print(f"  Vision: {scenario.vision_status.name}")

    controller = FinalProgressiveDecelerationController()

    # Mock data for first update
    v_ego = scenario.v_ego_ms
    current_curvature = None  # Can't see it (CURVE_EXCEEDS_FOV)
    pred_curvatures = np.array([scenario.max_curvature] * 5)
    pred_distances = np.array([10, 20, 30, 40, 50])

    print("\nFirst update inputs:")
    print(f"  v_ego: {v_ego*3.6:.1f} km/h")
    print(f"  current_curvature: {current_curvature}")
    print(f"  predicted_curvatures: {pred_curvatures}")
    print(f"  distances: {pred_distances}")

    # Calculate what should happen
    print("\nManual calculations:")

    # Safe speed for curve
    lateral_limit = 3.05  # m/s²
    v_safe = np.sqrt(lateral_limit / scenario.max_curvature)
    print(f"  v_safe for curve: {v_safe*3.6:.1f} km/h")

    # Required deceleration to v_safe
    distance = pred_distances[0]  # First predicted distance
    if v_ego > v_safe:
        required_decel = -(v_ego**2 - v_safe**2) / (2 * distance)
        required_g = abs(required_decel) / 9.81
    else:
        required_decel = 0
        required_g = 0

    print(f"  Required decel to v_safe at {distance}m: {required_decel:.2f} m/s² ({required_g:.2f}g)")

    # What emergency level should this be?
    if required_g <= 0.15:
        expected_level = "NORMAL"
    elif required_g <= 0.25:
        expected_level = "CAUTION"
    elif required_g <= 0.40:
        expected_level = "WARNING"
    elif required_g <= 0.60:
        expected_level = "CRITICAL"
    else:
        expected_level = "INTERVENTION"

    print(f"  Expected emergency level: {expected_level}")

    # Now run actual update
    print("\nActual controller update:")
    result = controller.update(
        v_ego=v_ego,
        current_curvature=current_curvature,
        predicted_curvatures=pred_curvatures,
        distances=pred_distances,
        vision_status=scenario.vision_status,
        model_confidence=scenario.confidence,
        lateral_acc_limit=lateral_limit,
        current_time=0.0
    )

    print(f"  target_speed: {result['target_speed']*3.6:.1f} km/h")
    print(f"  required_decel: {result['required_decel']:.2f} m/s² ({abs(result['required_decel'])/9.81:.2f}g)")
    print(f"  emergency_level: {result['emergency_level'].name}")
    print(f"  target_level: {result['target_level'].name}")
    print(f"  decel_limit: {result['decel_limit']:.2f} m/s² ({abs(result['decel_limit'])/9.81:.2f}g)")
    print(f"  intervention_required: {result['intervention_required']}")

    # Check occlusion handling
    print("\nOcclusion state:")
    print(f"  extrapolated_curvature: {result['extrapolated_curvature']:.3f}")
    print(f"  vision_degraded: {result['vision_degraded']}")
    print(f"  vision_confidence: {result['vision_confidence']:.2f}")


def trace_emergency_level_calc():
    """Trace through emergency level calculation step by step"""

    print("\n" + "="*80)
    print("Tracing Emergency Level Calculation")
    print("="*80)

    # Test case: 54 km/h to 20 km/h over 10m
    v_ego = 15.0  # m/s (54 km/h)
    v_target = 5.53  # m/s (20 km/h)
    distance = 10.0  # m

    print(f"Test case: {v_ego*3.6:.0f} km/h → {v_target*3.6:.0f} km/h over {distance}m")

    # Calculate deceleration
    required_decel = -(v_ego**2 - v_target**2) / (2 * distance)
    required_g = abs(required_decel) / 9.81

    print(f"Required decel: {required_decel:.2f} m/s² ({required_g:.2f}g)")

    # Check each threshold
    print("\nThreshold checks:")
    print(f"  <= 0.15g (NORMAL)? {required_g <= 0.15}")
    print(f"  <= 0.25g (CAUTION)? {required_g <= 0.25}")
    print(f"  <= 0.40g (WARNING)? {required_g <= 0.40}")
    print(f"  <= 0.60g (CRITICAL)? {required_g <= 0.60}")
    print(f"  > 0.60g (INTERVENTION)? {required_g > 0.60}")

    # Different distances
    print("\nEffect of distance on emergency level:")
    for d in [10, 20, 30, 40, 50]:
        req_decel = -(v_ego**2 - v_target**2) / (2 * d)
        req_g = abs(req_decel) / 9.81

        if req_g <= 0.15:
            level = "NORMAL"
        elif req_g <= 0.25:
            level = "CAUTION"
        elif req_g <= 0.40:
            level = "WARNING"
        elif req_g <= 0.60:
            level = "CRITICAL"
        else:
            level = "INTERVENTION"

        print(f"  {d}m: {req_g:.2f}g → {level}")


if __name__ == "__main__":
    debug_first_update()
    trace_emergency_level_calc()
