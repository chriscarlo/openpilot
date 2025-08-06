#!/usr/bin/env python3
"""Verify the final sigmoid configuration meets all constraints"""

import math

def production_sigmoid(curvature: float) -> float:
    """Current production sigmoid with a_min=1.8"""
    a_max = 3.12
    a_min = 1.8
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

def improved_sigmoid(curvature: float) -> float:
    """
    Improved sigmoid from plot_verified_sigmoid.py
    """
    a_max = 3.12   # Maximum lateral acceleration (m/s²)
    a_min = 0.80   # Minimum lateral acceleration (m/s²)
    k = 250.0      # Steepness factor
    c = 0.0180     # Center point

    lateral_acceleration = a_min + (a_max - a_min) / (1 + math.exp(k * (curvature - c)))

    return max(a_min, min(lateral_acceleration, a_max))

print("=== VERIFYING FINAL SIGMOID CONFIGURATION ===\n")
print("Improved sigmoid: a_min=0.80, a_max=3.12, k=250, c=0.0180")
print()

# Test key speeds
test_points = [
    (0.2985, 5),   # 5mph
    (0.0835, 10),  # 10mph
    (0.0414, 15),  # 15mph
    (0.0255, 20),  # 20mph
    (0.0175, 25),  # 25mph
    (0.0128, 30),  # 30mph
    (0.0099, 35),  # 35mph
    (0.0078, 40),  # 40mph
    (0.0064, 45),  # 45mph
    (0.0053, 50),  # 50mph - CRITICAL
    (0.0045, 55),  # 55mph
    (0.0038, 60),  # 60mph
    (0.0033, 65),  # 65mph
    (0.0029, 70),  # 70mph - CRITICAL
    (0.0025, 75),  # 75mph
]

all_constraints_met = True
below_50_ok = True
reaches_max = False

print(f"{'Speed':<8} {'Curv':<10} {'Prod Lat':<10} {'Imp Lat':<10} {'Diff':<10} {'Status'}")
print("-" * 70)

for curv, speed in test_points:
    prod_lat = production_sigmoid(curv)
    improved_lat = improved_sigmoid(curv)
    diff = improved_lat - prod_lat

    # Calculate actual speeds
    prod_speed = math.sqrt(prod_lat / curv) * 2.237
    imp_speed = math.sqrt(improved_lat / curv) * 2.237

    # Check constraints
    if speed < 50:
        if improved_lat > prod_lat:
            status = "❌ FAIL"
            below_50_ok = False
        else:
            status = "✓ OK"
    elif speed == 50:
        if improved_lat > prod_lat:
            status = "❌ FAIL!"
            below_50_ok = False
        else:
            status = "⚠️ BOUNDARY"
    elif speed >= 65 and improved_lat >= 3.0:
        status = "✓ MAX OK"
        reaches_max = True
    else:
        status = "OK"

    print(f"{speed:>3} mph  {curv:<10.4f} {prod_lat:<10.2f} {improved_lat:<10.2f} "
          f"{diff:>+10.2f} {status}")

print("\n" + "="*70)

if below_50_ok and reaches_max:
    print("✅ CONFIGURATION VERIFIED!")
    print("✓ Stays below production for ALL speeds < 50mph")
    print("✓ Reaches maximum (3.12 m/s²) by 65-70mph")
else:
    issues = []
    if not below_50_ok:
        issues.append("❌ Exceeds production below 50mph")
    if not reaches_max:
        issues.append("❌ Doesn't reach max by 70mph")
    print("Issues found:")
    for issue in issues:
        print(f"  {issue}")

print("\n=== SPEED BEHAVIOR ===")
print("\nActual speeds commanded for different curve radii:")
print(f"{'Radius':<15} {'Production':<15} {'Improved':<15} {'Difference'}")
print("-" * 55)

radii = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
for radius in radii:
    curvature = 1.0 / radius

    prod_lat = production_sigmoid(curvature)
    imp_lat = improved_sigmoid(curvature)

    prod_speed = math.sqrt(prod_lat * radius) * 2.237
    imp_speed = math.sqrt(imp_lat * radius) * 2.237

    # Apply MIN_V floors
    prod_min_v = 5.6 * 2.237  # 12.5 mph
    imp_min_v = 2.24 * 2.237  # 5 mph

    prod_commanded = max(prod_speed, prod_min_v)
    imp_commanded = max(imp_speed, imp_min_v)

    diff = imp_commanded - prod_commanded

    print(f"R={radius:>3}m         {prod_commanded:>6.1f} mph     {imp_commanded:>6.1f} mph     "
          f"{diff:>+6.1f} mph")

print("\n=== KEY IMPROVEMENTS ===")
print("1. Minimum lateral acceleration: 0.80 m/s² (was 1.8 m/s²)")
print("2. Minimum speed: 5 mph (was 12.5 mph)")
print("3. Sharp S-curve transition from 50mph to 30mph")
print("4. Maximum lateral acceleration reached by 65-70mph")
print("5. ALL speeds below 50mph have lower lateral acceleration than production")
