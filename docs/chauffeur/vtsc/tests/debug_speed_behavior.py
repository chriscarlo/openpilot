#!/usr/bin/env python3
"""Debug why VTSC behaves differently above and below 50mph"""

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

print("=== Investigation: Why does VTSC work above 50mph but not below? ===\n")

# Key constants
MIN_V = 5.6  # m/s = 12.5 mph - minimum operating speed
a_min_ref = 1.2   # Reference tuning
a_min_prod = 1.8  # Production tuning

print(f"Minimum operating speed: {MIN_V:.1f} m/s = {MIN_V*2.237:.1f} mph")
print(f"Reference a_min: {a_min_ref:.1f} m/s²")
print(f"Production a_min: {a_min_prod:.1f} m/s²\n")

# Test curves that would naturally require different speeds
test_curves = [
    (0.200, "Hairpin (5mph ideal)"),
    (0.150, "Very tight (7mph ideal)"),
    (0.100, "Tight (10mph ideal)"),
    (0.050, "Moderate (15mph ideal)"),
    (0.030, "Gentle (20mph ideal)"),
    (0.015, "Highway entry (30mph ideal)"),
    (0.008, "Highway curve (40mph ideal)"),
    (0.005, "Highway sweeper (50mph ideal)"),
    (0.003, "Gentle highway (60mph ideal)")
]

print("Curve Type            | Ideal | Ref Calc | Prod Calc | Prod Cmd | Error")
print("-" * 75)

for curv, desc in test_curves:
    # Calculate raw physics speeds
    ref_speed = curvature_to_speed(curv, a_min_ref)
    prod_speed = curvature_to_speed(curv, a_min_prod)

    # Apply minimum speed floor (what actually gets commanded)
    prod_commanded = max(prod_speed, MIN_V)

    # Estimate "ideal" speed (roughly what feels safe)
    ideal_speed = curvature_to_speed(curv, 1.0)  # Conservative a_min=1.0

    # Calculate error
    error_pct = ((prod_commanded - ideal_speed) / ideal_speed * 100) if ideal_speed > 0 else 0

    print(f"{desc:20s} | {ideal_speed*2.237:4.0f} | {ref_speed*2.237:7.1f} | {prod_speed*2.237:8.1f} | {prod_commanded*2.237:7.1f} | {error_pct:+5.0f}%")

print("\n=== Analysis ===")
print("\nBelow 50mph problems:")
print("1. Production a_min=1.8 causes 22.5% higher calculated speeds than reference")
print("2. MIN_V=12.5mph floor prevents slowing below this speed")
print("3. Combined effect: curves needing 5-10mph are taken at 12.5mph minimum")
print("\nAbove 50mph it works because:")
print("1. Highway curves have lower curvature")
print("2. Calculated speeds are naturally above MIN_V floor")
print("3. The 22.5% speed increase is less noticeable at highway speeds")
print("4. Error percentage decreases as speeds increase")

print("\n=== The smoking gun ===")
print("For curves below 50mph, especially tight ones:")
print("- Reference (a_min=1.2) would command reasonable speeds")
print("- Production (a_min=1.8) calculates 22.5% higher speeds")
print("- MIN_V floor makes it even worse for very tight curves")
print("- Result: Car barrels into curves way too fast!")
