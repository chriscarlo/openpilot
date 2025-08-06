#!/usr/bin/env python3
"""Compare sigmoid tuning between reference (a_min=1.2) and production (a_min=1.8)"""

import math

def physics_based_lateral_acceleration(curvature: float, a_min: float) -> float:
    """Calculate lateral acceleration for given curvature and a_min"""
    a_max = 3.12
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

def curvature_to_speed(curvature: float, a_min: float) -> float:
    """Calculate target speed for given curvature and a_min"""
    if curvature < 1e-7:
        return 70.0  # MAX_SPEED_DEFAULT

    safe_lat_accel = physics_based_lateral_acceleration(curvature, a_min)
    try:
        base_speed_mps = math.sqrt(safe_lat_accel / curvature)
    except (ValueError, ZeroDivisionError):
        base_speed_mps = 0.0

    return min(base_speed_mps, 70.0)

print("=== Sigmoid Tuning Comparison: a_min impact on curve speeds ===")
print("\nReference tuning: a_min = 1.2 m/s²")
print("Production tuning: a_min = 1.8 m/s²")
print(f"Speed increase factor: {(1.8/1.2)**0.5 - 1:.1%}\n")

# Test different curve severities
curvatures = [
    (0.200, "Hairpin turn (R=5m)"),
    (0.150, "Very tight turn (R=6.7m)"),
    (0.100, "Tight turn (R=10m)"),
    (0.050, "Moderate turn (R=20m)"),
    (0.020, "Gentle curve (R=50m)"),
    (0.010, "Highway curve (R=100m)"),
    (0.005, "Highway sweeper (R=200m)")
]

print("Curvature | Description          | Ref Speed | Prod Speed | Difference | % Increase")
print("-" * 85)

for curv, desc in curvatures:
    ref_speed = curvature_to_speed(curv, a_min=1.2)
    prod_speed = curvature_to_speed(curv, a_min=1.8)
    diff = prod_speed - ref_speed
    pct = (prod_speed / ref_speed - 1) * 100 if ref_speed > 0 else 0

    print(f"{curv:8.3f} | {desc:20s} | {ref_speed*2.237:5.1f} mph | {prod_speed*2.237:6.1f} mph | {diff*2.237:+6.1f} mph | {pct:+5.1f}%")

print("\n=== Speed comparison across the range 0-50 mph ===")
print("\nFinding curves where production speed exceeds various thresholds...")

# Find curves where production tuning gives specific speeds
target_speeds_mph = [10, 15, 20, 25, 30, 35, 40, 45, 50]
for target_mph in target_speeds_mph:
    target_mps = target_mph / 2.237
    # Binary search for curvature that gives this speed
    low, high = 0.001, 0.5
    for _ in range(20):
        mid = (low + high) / 2
        speed = curvature_to_speed(mid, a_min=1.8)
        if speed < target_mps:
            high = mid
        else:
            low = mid

    ref_speed = curvature_to_speed(mid, a_min=1.2)
    print(f"  {target_mph:2d} mph: curvature={mid:.4f}, ref would be {ref_speed*2.237:.1f} mph (diff: {(target_mps-ref_speed)*2.237:+.1f} mph)")

print("\n=== Analysis for speeds below 50 mph ===")
print("\nFor tight curves (high curvature), the production tuning with a_min=1.8")
print("results in speeds that are ~22.5% higher than the reference tuning with a_min=1.2")
print("\nThis explains why the car 'barrels FAR too aggressively' into curves below 50mph.")
print("The physics formula v = sqrt(a/k) means speed increases with sqrt(a_min).")
print("So increasing a_min from 1.2 to 1.8 (50% increase) gives sqrt(1.5) = 22.5% speed increase.")
