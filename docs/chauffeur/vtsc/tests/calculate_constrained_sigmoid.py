#!/usr/bin/env python3
"""Calculate sigmoid that stays BELOW production for all speeds under 50mph"""

import math
import numpy as np

def production_sigmoid(curvature: float) -> float:
    """Current production sigmoid with a_min=1.8"""
    a_max = 3.12
    a_min = 1.8
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

def test_improved_sigmoid(a_min, a_max, k, c, power=1.0):
    """Test if sigmoid stays below production below 50mph"""

    # Test curvatures corresponding to different speeds
    test_points = [
        # (curvature, approx_speed_at_prod, description)
        (0.200, 7, "Hairpin"),
        (0.150, 8, "Very tight"),
        (0.100, 10, "Tight turn"),
        (0.050, 14, "Residential"),
        (0.030, 18, "City turn"),
        (0.020, 22, "Normal turn"),
        (0.015, 26, "Gentle turn"),
        (0.010, 31, "Fast turn"),
        (0.008, 35, "Highway entry"),
        (0.006, 40, "Fast curve"),
        (0.005, 44, "Highway curve"),
        (0.004, 50, "Highway sweeper"),
        (0.003, 57, "Fast highway"),
        (0.002, 70, "Gentle highway"),
    ]

    all_below = True
    results = []

    for curv, approx_speed, desc in test_points:
        # Production sigmoid
        prod_lat = production_sigmoid(curv)
        prod_speed = math.sqrt(prod_lat / curv) * 2.237

        # Improved sigmoid
        improved_lat = a_min + (a_max - a_min) / (1 + math.exp(k * (curv**power - c)))
        improved_lat = max(a_min, min(improved_lat, a_max))
        improved_speed = math.sqrt(improved_lat / curv) * 2.237

        # Check if improved is below production for speeds under 50mph
        if prod_speed < 50:
            if improved_lat > prod_lat:
                all_below = False
                status = "FAIL"
            else:
                status = "OK"
        else:
            status = "-"

        results.append({
            'curv': curv,
            'desc': desc,
            'prod_lat': prod_lat,
            'prod_speed': prod_speed,
            'improved_lat': improved_lat,
            'improved_speed': improved_speed,
            'status': status
        })

    return all_below, results

print("=== FINDING SIGMOID THAT STAYS BELOW PRODUCTION FOR ALL SPEEDS < 50MPH ===\n")

# First, understand what production gives us at key speeds
print("Production sigmoid behavior (a_min=1.8):")
print(f"{'Speed':<10} {'Curvature':<12} {'Lat Accel':<12} {'Description'}")
print("-" * 50)

key_speeds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]
prod_at_speeds = {}

for target_speed in key_speeds:
    # Find what curvature gives this speed with production sigmoid
    for test_curv in np.logspace(-3, -0.5, 1000):
        prod_lat = production_sigmoid(test_curv)
        calc_speed = math.sqrt(prod_lat / test_curv) * 2.237
        if abs(calc_speed - target_speed) < 0.5:
            prod_at_speeds[target_speed] = (test_curv, prod_lat)
            print(f"{target_speed:>3} mph    {test_curv:<12.4f} {prod_lat:<12.2f}")
            break

print("\n=== CONSTRAINT ===")
print("Improved sigmoid must have LOWER lateral acceleration than production")
print("for ALL curvatures that produce speeds below 50mph")
print("But must still reach 3.12 m/s² by 65-70mph\n")

# Try different configurations
# Key insight: we need the transition to happen at LOWER curvature (higher speed)
# so that by the time we get to 50mph speeds, we're already well below production

configurations = [
    # (a_min, a_max, k, c, power)
    # Moving midpoint to lower curvature (larger radius)
    (0.6, 3.12, 300, 0.0045, 1.0),  # Midpoint at ~222m radius
    (0.6, 3.12, 250, 0.0050, 1.0),  # Midpoint at 200m radius
    (0.6, 3.12, 200, 0.0055, 1.0),  # Midpoint at 182m radius
    (0.5, 3.12, 280, 0.0048, 1.0),  # Lower min, different midpoint
    (0.7, 3.12, 350, 0.0042, 1.0),  # Very steep, early transition
    (0.65, 3.12, 320, 0.0046, 1.0), # Balanced approach
]

print("Testing configurations...\n")

best_config = None
for config in configurations:
    a_min, a_max, k, c, power = config
    all_below, results = test_improved_sigmoid(a_min, a_max, k, c, power)

    print(f"Config: a_min={a_min:.2f}, k={k:.0f}, c={c:.4f}")

    # Check if it meets our constraints
    meets_50mph_constraint = True
    reaches_max_by_70 = False

    for r in results:
        # Check < 50mph constraint
        if r['prod_speed'] < 50 and r['improved_lat'] > r['prod_lat']:
            meets_50mph_constraint = False

        # Check if we reach near-max by 70mph
        if r['prod_speed'] >= 65 and r['improved_lat'] >= 3.0:
            reaches_max_by_70 = True

    if meets_50mph_constraint and reaches_max_by_70:
        print("  ✓ Meets all constraints!")
        best_config = config

        # Print details
        print("\n  Details:")
        print(f"  {'Curv':<8} {'Desc':<12} {'Prod mph':<10} {'Prod lat':<10} {'Imp lat':<10} {'Imp mph':<10} {'Check'}")
        print("  " + "-"*75)

        for r in results:
            if r['prod_speed'] < 55 or r['prod_speed'] > 65:  # Show key ranges
                check = "✓" if r['status'] == 'OK' or r['status'] == '-' else "✗"
                print(f"  {r['curv']:<8.4f} {r['desc']:<12} {r['prod_speed']:<10.1f} "
                      f"{r['prod_lat']:<10.2f} {r['improved_lat']:<10.2f} "
                      f"{r['improved_speed']:<10.1f} {check}")
        break
    else:
        issues = []
        if not meets_50mph_constraint:
            issues.append("exceeds production below 50mph")
        if not reaches_max_by_70:
            issues.append("doesn't reach max by 70mph")
        print(f"  ✗ Issues: {', '.join(issues)}")
    print()

if best_config:
    a_min, a_max, k, c, power = best_config
    print("\n" + "="*80)
    print("BEST CONFIGURATION THAT MEETS ALL CONSTRAINTS")
    print("="*80)
    print(f"""
def improved_sigmoid_lateral_acceleration(curvature: float) -> float:
    '''
    Improved sigmoid that:
    - Stays BELOW production for all speeds < 50mph
    - Reaches max (3.12 m/s²) by 65-70mph
    - Sharp S-curve transition
    - Safe minimum for tight curves
    '''
    a_max = {a_max:.2f}   # Maximum lateral acceleration (m/s²)
    a_min = {a_min:.2f}   # Minimum lateral acceleration (m/s²)
    k = {k:.1f}      # Steepness factor
    c = {c:.4f}     # Center point (midpoint at {1/c:.1f}m radius)
    
    lateral_acceleration = a_min + (a_max - a_min) / (1 + math.exp(k * (curvature - c)))
    
    return max(a_min, min(lateral_acceleration, a_max))

# Also need to change MIN_V
_MIN_V = 2.24  # 5 mph minimum operating speed (was 5.6 m/s = 12.5 mph)
""")

    # Show speed comparison
    print("\nSpeed comparison at key points:")
    print(f"{'Speed':<10} {'Production':<15} {'Improved':<15} {'Difference'}")
    print("-" * 50)

    for speed in [5, 10, 20, 30, 40, 45, 50, 60, 70]:
        # Find curvature that gives this speed with production
        for test_curv in np.logspace(-3, -0.5, 1000):
            prod_lat = production_sigmoid(test_curv)
            calc_speed = math.sqrt(prod_lat / test_curv) * 2.237
            if abs(calc_speed - speed) < 0.5:
                improved_lat = a_min + (a_max - a_min) / (1 + math.exp(k * (test_curv - c)))
                improved_lat = max(a_min, min(improved_lat, a_max))
                diff = improved_lat - prod_lat
                symbol = "✓" if diff <= 0 or speed >= 50 else "✗"
                print(f"{speed:>3} mph    {prod_lat:>6.2f} m/s²     {improved_lat:>6.2f} m/s²     "
                      f"{diff:>+6.2f} m/s² {symbol}")
                break
else:
    print("\nNo configuration found that meets all constraints!")
    print("Need to adjust search range or relax constraints.")
