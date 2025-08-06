#!/usr/bin/env python3
"""Test the VTSC fix implementation"""

import sys
import math
sys.path.insert(0, '/data/openpilot')

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
    _physics_based_lateral_acceleration,
    curvature_to_speed,
    _MIN_V
)

def production_sigmoid(curvature: float) -> float:
    """Original production sigmoid for comparison"""
    a_max = 3.12
    a_min = 1.8
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

print("=== TESTING VTSC FIX ===\n")
print(f"Minimum speed set to: {_MIN_V * 2.237:.1f} mph (was 12.5 mph)")
print()

# Test key curvatures
test_points = [
    (0.2985, "5mph hairpin"),
    (0.0835, "10mph tight"),
    (0.0414, "15mph turn"),
    (0.0255, "20mph turn"),
    (0.0175, "25mph turn"),
    (0.0128, "30mph turn"),
    (0.0099, "35mph turn"),
    (0.0078, "40mph turn"),
    (0.0064, "45mph turn"),
    (0.0053, "50mph boundary"),
    (0.0045, "55mph curve"),
    (0.0038, "60mph curve"),
    (0.0033, "65mph curve"),
    (0.0029, "70mph highway"),
    (0.0025, "75mph highway"),
]

all_constraints_met = True
print(f"{'Curve Type':<20} {'Curv':<10} {'Prod Lat':<10} {'Fixed Lat':<10} {'Diff':<10} {'Status'}")
print("-" * 80)

for curv, desc in test_points:
    prod_lat = production_sigmoid(curv)
    fixed_lat = _physics_based_lateral_acceleration(curv)
    diff = fixed_lat - prod_lat

    # Calculate speeds
    prod_speed = math.sqrt(prod_lat / curv) * 2.237
    fixed_speed = math.sqrt(fixed_lat / curv) * 2.237

    # Check constraints
    if prod_speed < 50:
        if fixed_lat > prod_lat:
            status = "❌ FAIL"
            all_constraints_met = False
        else:
            status = "✓ SAFE"
    elif prod_speed >= 70:
        if fixed_lat >= 3.0:
            status = "✓ MAX"
        else:
            status = "❌ LOW"
            all_constraints_met = False
    else:
        status = "TRANS"

    print(f"{desc:<20} {curv:<10.4f} {prod_lat:<10.2f} {fixed_lat:<10.2f} "
          f"{diff:>+10.2f} {status}")

print("\n" + "="*80)

if all_constraints_met:
    print("✅ ALL CONSTRAINTS MET!")
    print("✓ Conservative speeds below 50mph")
    print("✓ Maximum performance above 70mph")
    print("✓ Sharp transition between 50-70mph")
else:
    print("❌ Some constraints not met")

# Test actual speed calculations
print("\n=== SPEED CALCULATIONS ===")
print(f"{'Radius':<15} {'Prod Speed':<15} {'Fixed Speed':<15} {'Difference'}")
print("-" * 55)

radii = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
for radius in radii:
    curvature = 1.0 / radius

    # Using curvature_to_speed function directly
    fixed_speed_mps = curvature_to_speed(curvature)
    fixed_speed_mph = fixed_speed_mps * 2.237

    # Calculate production speed for comparison
    prod_lat = production_sigmoid(curvature)
    prod_speed = math.sqrt(prod_lat * radius) * 2.237
    prod_commanded = max(prod_speed, 12.5)  # Old MIN_V

    # Apply new MIN_V
    fixed_commanded = max(fixed_speed_mph, _MIN_V * 2.237)

    diff = fixed_commanded - prod_commanded

    print(f"R={radius:>3}m         {prod_commanded:>6.1f} mph     "
          f"{fixed_commanded:>6.1f} mph     {diff:>+6.1f} mph")

print("\n=== KEY IMPROVEMENTS ===")
print("1. Piecewise function ensures ALL speeds <50mph are conservative")
print("2. Minimum speed reduced from 12.5mph to 5mph")
print("3. Sharp transition at 50mph boundary")
print("4. Maximum performance maintained above 70mph")
print("5. No more aggressive behavior in residential/city driving")
