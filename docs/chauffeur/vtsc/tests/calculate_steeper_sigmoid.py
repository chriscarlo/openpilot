#!/usr/bin/env python3
"""Calculate sigmoid with very steep transition right at 50mph"""

import math

def production_sigmoid(curvature: float) -> float:
    """Current production sigmoid with a_min=1.8"""
    a_max = 3.12
    a_min = 1.8
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

print("=== DESIGNING SIGMOID WITH STEEP TRANSITION AT 50MPH ===\n")

# Key constraint: at 50mph (curvature ~0.0053), production gives 2.68 m/s²
# We need to be below this but reach 3.12 by 70mph (curvature ~0.0029)

print("Critical points:")
print("50mph: curvature = 0.0053, production lat = 2.68 m/s²")
print("70mph: curvature = 0.0029, need lat = 3.12 m/s²")
print()

# Try configurations with very steep k and midpoint right around 50-60mph transition
configurations = [
    # (a_min, a_max, k, c)
    # Very steep transitions centered right at the 50mph curvature
    (1.0, 3.12, 500, 0.0053, 1.0),   # Centered exactly at 50mph
    (1.0, 3.12, 600, 0.0051, 1.0),   # Slightly above 50mph
    (1.0, 3.12, 700, 0.0049, 1.0),   # Slightly above 50mph
    (1.2, 3.12, 800, 0.0048, 1.0),   # Higher min, very steep
    (1.4, 3.12, 900, 0.0047, 1.0),   # Even higher min, extremely steep
    (1.5, 3.12, 1000, 0.0046, 1.0),  # Very high min, ultra steep
    (1.3, 3.12, 1200, 0.0045, 1.0),  # Extreme steepness
]

print("Testing configurations...\n")

best_config = None
best_score = float('inf')

for a_min, a_max, k, c, power in configurations:
    print(f"\nConfig: a_min={a_min:.1f}, k={k:.0f}, c={c:.4f}")

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
        (0.0038, 60),  # 60mph
        (0.0029, 70),  # 70mph - CRITICAL
    ]

    all_constraints_met = True
    below_50_ok = True
    reaches_max = False

    print(f"{'Speed':<8} {'Curv':<10} {'Prod Lat':<10} {'Imp Lat':<10} {'Diff':<10} {'Status'}")
    print("-" * 60)

    for curv, speed in test_points:
        prod_lat = production_sigmoid(curv)

        # Calculate improved sigmoid
        improved_lat = a_min + (a_max - a_min) / (1 + math.exp(k * (curv**power - c)))
        improved_lat = max(a_min, min(improved_lat, a_max))

        diff = improved_lat - prod_lat

        # Check constraints
        if speed < 50 and improved_lat > prod_lat:
            status = "FAIL"
            below_50_ok = False
        elif speed == 50:
            if improved_lat > prod_lat:
                status = "FAIL!"
                below_50_ok = False
            else:
                status = "CRITICAL"
        elif speed >= 70 and improved_lat >= 3.0:
            status = "MAX OK"
            reaches_max = True
        else:
            status = "OK"

        print(f"{speed:>3} mph  {curv:<10.4f} {prod_lat:<10.2f} {improved_lat:<10.2f} "
              f"{diff:>+10.2f} {status}")

    if below_50_ok and reaches_max:
        print("\n✓✓✓ FOUND VALID CONFIGURATION! ✓✓✓")
        best_config = (a_min, a_max, k, c, power)
        break
    else:
        issues = []
        if not below_50_ok:
            issues.append("exceeds production below 50mph")
        if not reaches_max:
            issues.append("doesn't reach max by 70mph")
        print(f"\n✗ Issues: {', '.join(issues)}")

if best_config:
    a_min, a_max, k, c, power = best_config
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION")
    print("="*80)

    print(f"""
def improved_sigmoid_lateral_acceleration(curvature: float) -> float:
    '''
    Improved sigmoid with constraints:
    - ALWAYS below production for speeds < 50mph
    - Reaches max (3.12 m/s²) by 70mph
    - Very steep transition at 50mph boundary
    '''
    a_max = {a_max:.2f}   # Maximum lateral acceleration (m/s²)
    a_min = {a_min:.2f}   # Minimum lateral acceleration (m/s²)
    k = {k:.1f}     # Very steep transition
    c = {c:.4f}    # Midpoint at {1/c:.1f}m radius (~50mph curves)
    
    lateral_acceleration = a_min + (a_max - a_min) / (1 + math.exp(k * (curvature - c)))
    
    return max(a_min, min(lateral_acceleration, a_max))

_MIN_V = 2.24  # 5 mph minimum (was 5.6 m/s = 12.5 mph)
""")

    # Verify the behavior
    print("\nBehavior verification:")
    print(f"- Below 50mph: Lateral accel stays at/near minimum ({a_min:.1f} m/s²)")
    print("- At 50mph: Sharp transition begins")
    print("- 50-70mph: Rapid climb to maximum")
    print("- Above 70mph: At maximum (3.12 m/s²)")

    print("\nThis creates a 'step function' effect:")
    print("- Conservative for all city/residential driving")
    print("- Aggressive for highway driving")
    print("- Sharp boundary at 50mph")
else:
    print("\n✗ Could not find configuration meeting all constraints")
    print("May need to relax the requirement slightly or use a different sigmoid form")
