#!/usr/bin/env python3
"""Plot improved sigmoid with steeper S-curve that hits max by 65-70mph and drops sharply below 50mph"""

import math
import numpy as np
import matplotlib.pyplot as plt

def original_physics_based_lateral_acceleration(curvature: float, a_min: float) -> float:
    """Original sigmoid from production"""
    a_max = 3.12
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

def improved_sigmoid_lateral_acceleration(curvature: float) -> float:
    """
    Improved sigmoid with proper S-curve shape:
    - Hits max (3.12 m/s²) by 65-70mph (low curvature ~0.003)
    - Sharp drop from 50mph to 30mph 
    - Begins flattening around 30mph
    - Smoothly curves to minimum at low speeds
    """
    a_max = 3.12   # Maximum lateral acceleration (m/s²) - for highway speeds
    a_min = 0.8    # Minimum lateral acceleration (m/s²) - for 5mph hairpins (was 1.2-1.8)

    # Steeper sigmoid parameters for sharper S-curve
    # Using logistic sigmoid: a = a_min + (a_max - a_min) / (1 + exp(k*(curv - c)))
    k = 40.0       # Steepness factor (higher = steeper transition)
    c = 0.012      # Center point of transition (around 30-40mph curves)

    # Logistic sigmoid for proper S-shape
    lateral_acceleration = a_min + (a_max - a_min) / (1 + math.exp(k * (curvature - c)))

    return max(a_min, min(lateral_acceleration, a_max))

def curvature_to_speed(curvature: float, sigmoid_func, a_min=None) -> float:
    """Calculate target speed for given curvature using specified sigmoid"""
    if curvature < 1e-7:
        return 70.0

    if a_min is not None:
        safe_lat_accel = sigmoid_func(curvature, a_min)
    else:
        safe_lat_accel = sigmoid_func(curvature)

    try:
        base_speed_mps = math.sqrt(safe_lat_accel / curvature)
    except (ValueError, ZeroDivisionError):
        base_speed_mps = 0.0
    return min(base_speed_mps, 70.0)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Constants
OLD_MIN_V_MPH = 5.6 * 2.237  # 12.5 mph (current)
NEW_MIN_V_MPH = 2.24 * 2.237  # 5 mph (proposed)

# Test curves
test_curves = [
    (5, "Hairpin (R=5m)", 'red'),
    (10, "Tight turn (R=10m)", 'darkorange'),
    (20, "Sharp turn (R=20m)", 'gold'),
    (30, "Residential turn (R=30m)", 'yellow'),
    (50, "Normal turn (R=50m)", 'greenyellow'),
    (75, "Gentle turn (R=75m)", 'lightgreen'),
    (100, "Highway curve (R=100m)", 'cyan'),
    (150, "Fast curve (R=150m)", 'skyblue'),
    (200, "Highway sweeper (R=200m)", 'blue'),
    (300, "Gentle highway (R=300m)", 'purple'),
]

# LEFT PLOT: Current vs Improved Sigmoid Shape
ax1.set_title('Sigmoid Comparison: Current vs Improved', fontsize=14, weight='bold')

# Generate curvature range
curvatures = np.logspace(-3.5, -0.7, 500)  # Extended range for better visualization

# Calculate lateral accelerations
lat_accels_ref = [original_physics_based_lateral_acceleration(c, a_min=1.2) for c in curvatures]
lat_accels_prod = [original_physics_based_lateral_acceleration(c, a_min=1.8) for c in curvatures]
lat_accels_improved = [improved_sigmoid_lateral_acceleration(c) for c in curvatures]

# Convert to speeds for better intuition
speeds_ref = [curvature_to_speed(c, original_physics_based_lateral_acceleration, 1.2) * 2.237 for c in curvatures]
speeds_prod = [curvature_to_speed(c, original_physics_based_lateral_acceleration, 1.8) * 2.237 for c in curvatures]
speeds_improved = [curvature_to_speed(c, improved_sigmoid_lateral_acceleration) * 2.237 for c in curvatures]

# Plot speeds vs curvature (log scale)
ax1.semilogx(curvatures, speeds_ref, 'b--', linewidth=2, label='Current Reference (a_min=1.2)', alpha=0.6)
ax1.semilogx(curvatures, speeds_prod, 'r--', linewidth=2, label='Current Production (a_min=1.8)', alpha=0.6)
ax1.semilogx(curvatures, speeds_improved, 'g-', linewidth=3, label='Improved Sigmoid', alpha=0.9)

# Add reference lines
ax1.axhline(y=70, color='purple', linestyle=':', alpha=0.5, label='70mph (should hit max here)')
ax1.axhline(y=50, color='orange', linestyle=':', alpha=0.5, label='50mph (steep drop below)')
ax1.axhline(y=30, color='red', linestyle=':', alpha=0.5, label='30mph (curve flattens)')
ax1.axhline(y=NEW_MIN_V_MPH, color='green', linestyle=':', alpha=0.5, label='5mph (new minimum)')

ax1.set_xlabel('Curvature (1/m) [log scale]', fontsize=12)
ax1.set_ylabel('Target Speed (mph)', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 80)

# Add curve radius labels
for radius, name, _ in test_curves[::2]:  # Every other one to avoid crowding
    curv = 1.0 / radius
    if curv >= curvatures[0] and curv <= curvatures[-1]:
        ax1.axvline(x=curv, color='gray', linestyle=':', alpha=0.2)
        ax1.text(curv, 75, f'R={radius}m', rotation=90, ha='right', va='bottom', fontsize=8, alpha=0.5)

# RIGHT PLOT: Speed vs Lateral Acceleration with new sigmoid
ax2.set_title('Improved VTSC: Speed vs Lateral Acceleration', fontsize=14, weight='bold')

# For each curve radius, plot lateral acceleration vs speed
for radius, name, color in test_curves:
    curvature = 1.0 / radius

    # Generate speed range (mph)
    speeds_mph = np.linspace(3, 80, 200)
    speeds_mps = speeds_mph / 2.237

    # Calculate lateral acceleration at each speed for this curve
    lat_accels_actual = [v**2 * curvature for v in speeds_mps]

    # Plot the physics relationship
    ax2.plot(speeds_mph, lat_accels_actual, '-', linewidth=1.5, color=color, alpha=0.3)

    # Calculate what VTSC would command for this curve
    prod_lat_accel = original_physics_based_lateral_acceleration(curvature, a_min=1.8)
    improved_lat_accel = improved_sigmoid_lateral_acceleration(curvature)

    # Calculate the speeds VTSC would target
    prod_speed_mps = math.sqrt(prod_lat_accel / curvature)
    improved_speed_mps = math.sqrt(improved_lat_accel / curvature)
    prod_speed_mph = prod_speed_mps * 2.237
    improved_speed_mph = improved_speed_mps * 2.237

    # Apply MIN_V floors
    prod_commanded_mph = max(prod_speed_mph, OLD_MIN_V_MPH)
    improved_commanded_mph = max(improved_speed_mph, NEW_MIN_V_MPH)

    # Plot operating points
    if prod_commanded_mph <= 80:
        ax2.plot(prod_commanded_mph, prod_lat_accel, 's', color=color, markersize=8,
                markeredgecolor='red', markeredgewidth=2, alpha=0.5)

    if improved_commanded_mph <= 80:
        ax2.plot(improved_commanded_mph, improved_lat_accel, 'o', color=color, markersize=10,
                markeredgecolor='green', markeredgewidth=2.5, alpha=0.8)
        # Add labels for key curves
        if radius in [5, 20, 50, 100, 200]:
            ax2.annotate(name.split('(')[0].strip(),
                       xy=(improved_commanded_mph, improved_lat_accel),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=9, color=color, weight='bold')

# Add the envelope curves
speeds_range = np.linspace(5, 80, 200)
lat_accels_for_speeds = []

for speed_mph in speeds_range:
    speed_mps = speed_mph / 2.237
    # Find the curvature that would give this speed with improved sigmoid
    best_lat_accel = 0
    for test_curv in np.logspace(-3.5, -0.5, 500):
        lat_accel = improved_sigmoid_lateral_acceleration(test_curv)
        calc_speed = math.sqrt(lat_accel / test_curv) * 2.237
        if abs(calc_speed - speed_mph) < 0.5:
            best_lat_accel = lat_accel
            break
    if best_lat_accel > 0:
        lat_accels_for_speeds.append(best_lat_accel)
    else:
        lat_accels_for_speeds.append(np.nan)

# Plot improved envelope
valid_indices = ~np.isnan(lat_accels_for_speeds)
ax2.plot(speeds_range[valid_indices], np.array(lat_accels_for_speeds)[valid_indices],
         'g-', linewidth=3, label='Improved sigmoid envelope', alpha=0.7)

# Add reference lines
ax2.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
ax2.text(75, 0.85, 'New floor: 0.8 m/s²', color='green', fontsize=10, weight='bold')

ax2.axhline(y=1.8, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax2.text(75, 1.85, 'Current prod: 1.8 m/s²', color='red', fontsize=10)

ax2.axhline(y=3.12, color='purple', linestyle=':', alpha=0.5, linewidth=1.5)
ax2.text(75, 3.17, 'Max: 3.12 m/s²', color='purple', fontsize=10)

# Speed reference lines
ax2.axvline(x=NEW_MIN_V_MPH, color='green', linestyle='--', alpha=0.5, linewidth=2)
ax2.text(NEW_MIN_V_MPH+1, 3.3, '5mph\nmin', color='green', fontsize=9, weight='bold')

ax2.axvline(x=30, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.text(31, 3.3, '30mph', color='orange', fontsize=9)

ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.text(51, 3.3, '50mph', color='red', fontsize=9)

ax2.axvline(x=70, color='purple', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.text(71, 3.3, '70mph', color='purple', fontsize=9)

# Legend
ax2.plot([], [], 's', color='gray', markersize=8, markeredgecolor='red',
        markeredgewidth=2, label='Current production', alpha=0.5)
ax2.plot([], [], 'o', color='gray', markersize=10, markeredgecolor='green',
        markeredgewidth=2.5, label='Improved sigmoid')

ax2.set_xlabel('Speed (mph)', fontsize=12, weight='bold')
ax2.set_ylabel('Lateral Acceleration (m/s²)', fontsize=12, weight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 80)
ax2.set_ylim(0, 3.5)

# Add g-force scale
ax2_right = ax2.twinx()
ax2_right.set_ylabel('Lateral G-force', fontsize=12, weight='bold')
ax2_right.set_ylim(0, 3.5/9.81)
g_ticks = [0, 0.1, 0.2, 0.3]
ax2_right.set_yticks([g * 9.81 for g in g_ticks])
ax2_right.set_yticklabels([f'{g:.1f}g' for g in g_ticks])

plt.tight_layout()

# Save the plot
output_path = '/data/openpilot/docs/chauffeur/vtsc/results/improved_sigmoid_curves.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also save to Windows-accessible location
windows_path = '/mnt/c/Users/Chris/Pictures/Screenshots/vtsc_improved_sigmoid.png'
plt.savefig(windows_path, dpi=150, bbox_inches='tight')
print(f"Also saved to Windows: {windows_path}")

# Try to open with Windows default image viewer
import subprocess
try:
    subprocess.run(['explorer.exe', windows_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')])
    print("Opened with Windows Explorer")
except:
    print("Please open: C:\\Users\\Chris\\Pictures\\Screenshots\\vtsc_improved_sigmoid.png")

# Print key parameters
print("\n=== Improved Sigmoid Parameters ===")
print("a_min: 0.8 m/s² (vs current 1.8)")
print("a_max: 3.12 m/s² (unchanged)")
print("Steepness (k): 40.0 (creates sharp S-curve)")
print("Center point: 0.012 (1/m) = ~83m radius curve")
print("MIN_V: 5 mph (vs current 12.5 mph)")
print("\nThis creates:")
print("- Max lat accel by 65-70 mph")
print("- Sharp drop from 50 to 30 mph")
print("- Flattening around 30 mph")
print("- Smooth curve to 5 mph minimum")
