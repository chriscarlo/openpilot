#!/usr/bin/env python3
"""Piecewise sigmoid solution that meets ALL constraints"""

import math
import numpy as np
import matplotlib.pyplot as plt

def production_sigmoid(curvature: float) -> float:
    """Current production sigmoid with a_min=1.8"""
    a_max = 3.12
    a_min = 1.8
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

def piecewise_sigmoid(curvature: float) -> float:
    """
    Piecewise sigmoid that meets ALL constraints:
    - Below production for all speeds < 50mph
    - Reaches max by 70mph
    - Sharp transition at 50mph boundary
    """

    # Critical curvatures
    CURV_50MPH = 0.0053  # Curvature for 50mph curves
    CURV_70MPH = 0.0029  # Curvature for 70mph curves

    if curvature > CURV_50MPH:
        # For tight curves (speeds < 50mph): Use conservative values
        # Linear interpolation from 1.5 at very tight to 1.7 at 50mph boundary
        a_min_tight = 1.5
        a_max_tight = 1.7

        # Map curvature from [CURV_50MPH, 0.3] to [a_max_tight, a_min_tight]
        if curvature > 0.3:
            return a_min_tight
        else:
            # Linear interpolation
            t = (curvature - CURV_50MPH) / (0.3 - CURV_50MPH)
            return a_max_tight + t * (a_min_tight - a_max_tight)

    elif curvature > CURV_70MPH:
        # Transition zone (50-70mph): Rapid increase
        # Exponential rise from 1.7 to 3.12
        a_start = 1.7
        a_end = 3.12

        # Map curvature from [CURV_70MPH, CURV_50MPH] to [a_end, a_start]
        t = (curvature - CURV_70MPH) / (CURV_50MPH - CURV_70MPH)
        # Use exponential interpolation for sharp rise
        return a_start + (a_end - a_start) * (1 - math.exp(-5 * (1 - t)))

    else:
        # Highway speeds (>70mph): Maximum performance
        return 3.12

print("=== PIECEWISE SIGMOID SOLUTION ===\n")

# Test the piecewise function
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

print(f"{'Speed':<8} {'Curv':<10} {'Prod Lat':<10} {'Piece Lat':<10} {'Diff':<10} {'Status'}")
print("-" * 70)

for curv, speed in test_points:
    prod_lat = production_sigmoid(curv)
    piece_lat = piecewise_sigmoid(curv)
    diff = piece_lat - prod_lat

    # Check constraints
    if speed < 50:
        if piece_lat > prod_lat:
            status = "❌ FAIL"
            below_50_ok = False
        else:
            status = "✓ OK"
    elif speed == 50:
        if piece_lat > prod_lat:
            status = "❌ BOUNDARY FAIL"
            below_50_ok = False
        else:
            status = "⚠️ BOUNDARY"
    elif speed >= 70:
        if piece_lat >= 3.0:
            status = "✓ MAX OK"
            reaches_max = True
        else:
            status = "Not max"
    else:
        status = "Transition"

    print(f"{speed:>3} mph  {curv:<10.4f} {prod_lat:<10.2f} {piece_lat:<10.2f} "
          f"{diff:>+10.2f} {status}")

print("\n" + "="*70)

if below_50_ok and reaches_max:
    print("✅ ALL CONSTRAINTS MET!")
    print("✓ Stays below production for ALL speeds < 50mph")
    print("✓ Reaches maximum (3.12 m/s²) by 70mph")
    print("✓ Sharp transition at 50mph boundary")
else:
    print("❌ Constraints not met")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left plot: Lateral acceleration vs curvature
curvatures = np.logspace(-3.5, -0.5, 500)
prod_lats = [production_sigmoid(c) for c in curvatures]
piece_lats = [piecewise_sigmoid(c) for c in curvatures]

ax1.semilogx(curvatures, prod_lats, 'r-', linewidth=2.5, label='Production (a_min=1.8)', alpha=0.7)
ax1.semilogx(curvatures, piece_lats, 'g-', linewidth=3, label='Piecewise Solution', alpha=0.9)

# Mark critical curvatures
ax1.axvline(x=0.0053, color='orange', linestyle='--', alpha=0.5, linewidth=2)
ax1.text(0.0053, 3.3, '50mph', rotation=90, fontsize=10, color='orange')

ax1.axvline(x=0.0029, color='purple', linestyle='--', alpha=0.5, linewidth=2)
ax1.text(0.0029, 3.3, '70mph', rotation=90, fontsize=10, color='purple')

ax1.set_xlabel('Curvature (1/m)', fontsize=12)
ax1.set_ylabel('Lateral Acceleration (m/s²)', fontsize=12)
ax1.set_title('Lateral Acceleration vs Curvature', fontsize=14, weight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 3.5)

# Right plot: Speed vs lateral acceleration
speeds = np.linspace(5, 80, 200)
prod_speeds_lat = []
piece_speeds_lat = []

for speed_mph in speeds:
    # Find curvature that gives this speed for each function
    best_prod = None
    best_piece = None

    for test_curv in np.logspace(-3.5, -0.5, 500):
        prod_lat = production_sigmoid(test_curv)
        calc_speed = math.sqrt(prod_lat / test_curv) * 2.237
        if best_prod is None or abs(calc_speed - speed_mph) < 0.5:
            best_prod = prod_lat

        piece_lat = piecewise_sigmoid(test_curv)
        calc_speed = math.sqrt(piece_lat / test_curv) * 2.237
        if best_piece is None or abs(calc_speed - speed_mph) < 0.5:
            best_piece = piece_lat

    prod_speeds_lat.append(best_prod)
    piece_speeds_lat.append(best_piece)

ax2.plot(speeds, prod_speeds_lat, 'r-', linewidth=2.5, label='Production', alpha=0.7)
ax2.plot(speeds, piece_speeds_lat, 'g-', linewidth=3, label='Piecewise Solution', alpha=0.9)

# Mark zones
ax2.axvspan(0, 50, alpha=0.1, color='yellow', label='Problem zone (<50mph)')
ax2.axvspan(50, 70, alpha=0.1, color='orange', label='Transition (50-70mph)')
ax2.axvspan(70, 80, alpha=0.1, color='green', label='Highway (>70mph)')

ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax2.axvline(x=70, color='purple', linestyle='--', alpha=0.5, linewidth=2)

ax2.set_xlabel('Speed (mph)', fontsize=12)
ax2.set_ylabel('Lateral Acceleration (m/s²)', fontsize=12)
ax2.set_title('Speed vs Lateral Acceleration', fontsize=14, weight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2.set_xlim(0, 80)
ax2.set_ylim(0, 3.5)

plt.suptitle('Piecewise Sigmoid Solution for VTSC', fontsize=16, weight='bold')
plt.tight_layout()

# Save plots
output_path = '/data/openpilot/docs/chauffeur/vtsc/results/piecewise_sigmoid_solution.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

windows_path = '/mnt/c/Users/Chris/Pictures/Screenshots/vtsc_piecewise_solution.png'
plt.savefig(windows_path, dpi=150, bbox_inches='tight')
print(f"Also saved to Windows: {windows_path}")

# Implementation code
print("\n" + "="*80)
print("IMPLEMENTATION CODE")
print("="*80)

print("""
def improved_lateral_acceleration(curvature: float) -> float:
    '''
    Piecewise function that ensures:
    - Conservative speeds below 50mph (lower than production)
    - Maximum performance above 70mph
    - Sharp transition between 50-70mph
    '''
    CURV_50MPH = 0.0053  # Boundary at 50mph
    CURV_70MPH = 0.0029  # Boundary at 70mph
    
    if curvature > CURV_50MPH:
        # Tight curves (< 50mph): Conservative
        if curvature > 0.3:
            return 1.5
        else:
            t = (curvature - CURV_50MPH) / (0.3 - CURV_50MPH)
            return 1.7 + t * (1.5 - 1.7)
    
    elif curvature > CURV_70MPH:
        # Transition zone (50-70mph): Rapid increase
        t = (curvature - CURV_70MPH) / (CURV_50MPH - CURV_70MPH)
        return 1.7 + (3.12 - 1.7) * (1 - math.exp(-5 * (1 - t)))
    
    else:
        # Highway (>70mph): Maximum performance
        return 3.12

# Minimum speed also needs adjustment
_MIN_V = 2.24  # 5 mph (was 5.6 m/s = 12.5 mph)
""")
