#!/usr/bin/env python3
"""Find sigmoid that ACTUALLY stays below production for all speeds < 50mph"""

import math

def production_sigmoid(curvature: float) -> float:
    """Current production sigmoid with a_min=1.8"""
    a_max = 3.12
    a_min = 1.8
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

print("=== FINDING CORRECT SIGMOID CONFIGURATION ===\n")

# The key insight: We need the sigmoid transition to happen at MUCH lower curvature
# (higher speed curves) so that by 50mph we're already in the low part

# Test configurations with transition point well above 50mph
configurations = [
    # (a_min, a_max, k, c)
    # Very low minimum, transition at low curvature (high speed)
    (0.5, 3.12, 400, 0.0035, 1.0),   # Transition around 60-70mph curves
    (0.6, 3.12, 450, 0.0032, 1.0),   # Slightly higher min
    (0.7, 3.12, 500, 0.0030, 1.0),   # Higher min, steeper
    (0.8, 3.12, 600, 0.0028, 1.0),   # Even steeper
    (0.9, 3.12, 700, 0.0026, 1.0),   # Very steep
    (1.0, 3.12, 800, 0.0024, 1.0),   # Extremely steep
    (1.1, 3.12, 900, 0.0023, 1.0),   # Ultra steep
    (1.2, 3.12, 1000, 0.0022, 1.0),  # Maximum steepness
]

# Key test points
test_points = [
    (0.2985, 5),   # 5mph - hairpin
    (0.0835, 10),  # 10mph
    (0.0414, 15),  # 15mph
    (0.0255, 20),  # 20mph
    (0.0175, 25),  # 25mph
    (0.0128, 30),  # 30mph
    (0.0099, 35),  # 35mph
    (0.0078, 40),  # 40mph
    (0.0064, 45),  # 45mph
    (0.0053, 50),  # 50mph - CRITICAL BOUNDARY
    (0.0045, 55),  # 55mph
    (0.0038, 60),  # 60mph
    (0.0033, 65),  # 65mph
    (0.0029, 70),  # 70mph - MUST BE NEAR MAX
]

best_config = None
best_score = float('inf')

for a_min, a_max, k, c, power in configurations:
    print(f"\nTesting: a_min={a_min:.1f}, k={k:.0f}, c={c:.4f}")

    all_below_50 = True
    reaches_max_by_70 = False
    max_violation = 0

    results_to_print = []

    for curv, speed in test_points:
        prod_lat = production_sigmoid(curv)

        # Calculate improved sigmoid
        improved_lat = a_min + (a_max - a_min) / (1 + math.exp(k * (curv**power - c)))
        improved_lat = max(a_min, min(improved_lat, a_max))

        diff = improved_lat - prod_lat

        # Check constraints
        if speed < 50:
            if improved_lat > prod_lat:
                all_below_50 = False
                max_violation = max(max_violation, diff)
                status = f"FAIL ({diff:+.2f})"
            else:
                status = f"OK ({diff:+.2f})"
        elif speed == 50:
            if improved_lat > prod_lat:
                all_below_50 = False
                max_violation = max(max_violation, diff)
                status = f"BOUNDARY FAIL ({diff:+.2f})"
            else:
                status = f"BOUNDARY OK ({diff:+.2f})"
        elif speed >= 65:
            if improved_lat >= 3.0:
                reaches_max_by_70 = True
                status = f"MAX OK ({improved_lat:.2f})"
            else:
                status = f"Not max ({improved_lat:.2f})"
        else:
            status = f"({diff:+.2f})"

        if speed in [5, 20, 30, 40, 50, 60, 70]:
            results_to_print.append(f"  {speed:>2}mph: prod={prod_lat:.2f}, imp={improved_lat:.2f} {status}")

    # Print key speeds
    for result in results_to_print:
        print(result)

    if all_below_50 and reaches_max_by_70:
        print("  ✅ SUCCESS! All constraints met!")
        best_config = (a_min, a_max, k, c, power)
        break
    else:
        issues = []
        if not all_below_50:
            issues.append(f"exceeds prod <50mph (max violation: {max_violation:.2f})")
        if not reaches_max_by_70:
            issues.append("doesn't reach max by 70mph")
        print(f"  ❌ Issues: {', '.join(issues)}")

if best_config:
    a_min, a_max, k, c, power = best_config
    print("\n" + "="*80)
    print("FOUND VALID CONFIGURATION!")
    print("="*80)

    print(f"""
def corrected_sigmoid_lateral_acceleration(curvature: float) -> float:
    '''
    Corrected sigmoid that ACTUALLY meets all constraints:
    - ALWAYS below production for speeds < 50mph
    - Reaches max (3.12 m/s²) by 65-70mph
    - Smooth S-curve transition
    '''
    a_max = {a_max:.2f}   # Maximum lateral acceleration (m/s²)
    a_min = {a_min:.2f}   # Minimum lateral acceleration (m/s²)
    k = {k:.1f}      # Steepness factor
    c = {c:.4f}     # Transition point (curvature)
    
    lateral_acceleration = a_min + (a_max - a_min) / (1 + math.exp(k * (curvature - c)))
    
    return max(a_min, min(lateral_acceleration, a_max))

# Also need to change MIN_V
_MIN_V = 2.24  # 5 mph minimum (was 5.6 m/s = 12.5 mph)
""")

    # Show behavior at different curve radii
    print("\nBehavior for different curve radii:")
    print(f"{'Radius':<10} {'Prod Speed':<12} {'Imp Speed':<12} {'Difference'}")
    print("-" * 45)

    for radius in [5, 10, 20, 30, 50, 75, 100, 150, 200]:
        curvature = 1.0 / radius

        prod_lat = production_sigmoid(curvature)
        improved_lat = a_min + (a_max - a_min) / (1 + math.exp(k * (curvature - c)))
        improved_lat = max(a_min, min(improved_lat, a_max))

        prod_speed = math.sqrt(prod_lat * radius) * 2.237
        imp_speed = math.sqrt(improved_lat * radius) * 2.237

        # Apply MIN_V floors
        prod_commanded = max(prod_speed, 12.5)
        imp_commanded = max(imp_speed, 5.0)

        diff = imp_commanded - prod_commanded
        symbol = "✓" if diff <= 0 or prod_commanded >= 50 else "✗"

        print(f"R={radius:>3}m      {prod_commanded:>6.1f} mph   {imp_commanded:>6.1f} mph   "
              f"{diff:>+6.1f} mph {symbol}")

    print("\nThis configuration ensures:")
    print("1. Conservative speeds for ALL city/residential driving (<50mph)")
    print("2. Appropriate performance for highway driving (>50mph)")
    print("3. No aggressive behavior in the problem zone")
else:
    print("\n❌ Could not find a configuration that meets all constraints")
    print("The requirements may be physically incompatible with a standard sigmoid")
    print("Consider using a piecewise function or modified sigmoid form")
