#!/usr/bin/env python3
"""Plot lateral acceleration vs speed to show how VTSC behaves at different speeds"""

import math
import numpy as np
import matplotlib.pyplot as plt

def physics_based_lateral_acceleration(curvature: float, a_min: float) -> float:
    """Calculate lateral acceleration for given curvature and a_min"""
    a_max = 3.12
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Constants
MIN_V_MPH = 5.6 * 2.237  # 12.5 mph

# Define test curves by their radius
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

# For each curve radius, plot lateral acceleration vs speed
for radius, name, color in test_curves:
    curvature = 1.0 / radius

    # Generate speed range (mph)
    speeds_mph = np.linspace(5, 80, 200)
    speeds_mps = speeds_mph / 2.237

    # Calculate lateral acceleration at each speed for this curve
    # a_lat = v^2 * curvature
    lat_accels_actual = [v**2 * curvature for v in speeds_mps]

    # Plot the physics relationship
    ax.plot(speeds_mph, lat_accels_actual, '-', linewidth=1.5, color=color, alpha=0.3)

    # Calculate what VTSC would command for this curve
    ref_lat_accel = physics_based_lateral_acceleration(curvature, a_min=1.2)
    prod_lat_accel = physics_based_lateral_acceleration(curvature, a_min=1.8)

    # Calculate the speed VTSC would target
    ref_speed_mps = math.sqrt(ref_lat_accel / curvature)
    prod_speed_mps = math.sqrt(prod_lat_accel / curvature)
    ref_speed_mph = ref_speed_mps * 2.237
    prod_speed_mph = prod_speed_mps * 2.237
    prod_commanded_mph = max(prod_speed_mph, MIN_V_MPH)

    # Plot operating points
    if ref_speed_mph <= 80:
        ax.plot(ref_speed_mph, ref_lat_accel, 'o', color=color, markersize=10,
                markeredgecolor='blue', markeredgewidth=2.5, alpha=0.8)
        # Add curve label at reference point
        if radius in [5, 10, 20, 50, 100, 200]:
            ax.annotate(name.split('(')[0].strip(),
                       xy=(ref_speed_mph, ref_lat_accel),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=color, weight='bold')

    if prod_commanded_mph <= 80:
        ax.plot(prod_commanded_mph, prod_lat_accel, 's', color=color, markersize=10,
                markeredgecolor='red', markeredgewidth=2.5, alpha=0.8)

# Add the sigmoid curves themselves
speeds_for_sigmoid = np.linspace(5, 80, 100)
curvatures_for_speeds_ref = []
curvatures_for_speeds_prod = []
lat_accels_ref_sigmoid = []
lat_accels_prod_sigmoid = []

for speed_mph in speeds_for_sigmoid:
    speed_mps = speed_mph / 2.237

    # Find what curvature would give this speed for each tuning
    # We need to solve: speed = sqrt(lat_accel / curvature)
    # Try different curvatures and find the one that matches
    for test_curv in np.logspace(-3, -0.5, 1000):
        ref_lat = physics_based_lateral_acceleration(test_curv, a_min=1.2)
        calc_speed = math.sqrt(ref_lat / test_curv) * 2.237
        if abs(calc_speed - speed_mph) < 0.5:
            curvatures_for_speeds_ref.append(test_curv)
            lat_accels_ref_sigmoid.append(ref_lat)
            break

    for test_curv in np.logspace(-3, -0.5, 1000):
        prod_lat = physics_based_lateral_acceleration(test_curv, a_min=1.8)
        calc_speed = math.sqrt(prod_lat / test_curv) * 2.237
        if abs(calc_speed - speed_mph) < 0.5:
            curvatures_for_speeds_prod.append(test_curv)
            lat_accels_prod_sigmoid.append(prod_lat)
            break

# Plot sigmoid envelopes (if we have enough points)
if len(lat_accels_ref_sigmoid) > 10:
    ax.plot(speeds_for_sigmoid[:len(lat_accels_ref_sigmoid)], lat_accels_ref_sigmoid,
            'b-', linewidth=3, label='Reference envelope (a_min=1.2)', alpha=0.7)
if len(lat_accels_prod_sigmoid) > 10:
    ax.plot(speeds_for_sigmoid[:len(lat_accels_prod_sigmoid)], lat_accels_prod_sigmoid,
            'r-', linewidth=3, label='Production envelope (a_min=1.8)', alpha=0.7)

# Add horizontal lines for key lateral acceleration values
ax.axhline(y=1.2, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(75, 1.25, 'Ref floor: 1.2 m/s²', color='blue', fontsize=10)

ax.axhline(y=1.8, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(75, 1.85, 'Prod floor: 1.8 m/s²', color='red', fontsize=10)

# Add vertical lines for key speeds
ax.axvline(x=MIN_V_MPH, color='orange', linestyle='--', alpha=0.5, linewidth=2)
ax.text(MIN_V_MPH+1, 3.0, f'MIN_V\n{MIN_V_MPH:.1f}mph', color='orange', fontsize=10, weight='bold')

ax.axvline(x=45, color='purple', linestyle='--', alpha=0.5, linewidth=2)
ax.text(46, 3.0, '45mph', color='purple', fontsize=10, weight='bold')

ax.axvline(x=50, color='green', linestyle='--', alpha=0.5, linewidth=2)
ax.text(51, 3.0, '50mph', color='green', fontsize=10, weight='bold')

# Shade problem zones
ax.axvspan(0, 50, alpha=0.05, color='red', label='Problem zone (<50mph)')
ax.axvspan(50, 80, alpha=0.05, color='green', label='Working zone (>50mph)')

# Add legend items for markers
ax.plot([], [], 'o', color='gray', markersize=10, markeredgecolor='blue',
        markeredgewidth=2.5, label='Reference operating points')
ax.plot([], [], 's', color='gray', markersize=10, markeredgecolor='red',
        markeredgewidth=2.5, label='Production operating points')

# Labels and formatting
ax.set_xlabel('Speed (mph)', fontsize=14, weight='bold')
ax.set_ylabel('Lateral Acceleration (m/s²)', fontsize=14, weight='bold')
ax.set_title('VTSC Lateral Acceleration vs Speed for Different Curve Radii', fontsize=16, weight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(0, 80)
ax.set_ylim(0, 3.5)

# Add secondary y-axis for g-force
ax2 = ax.twinx()
ax2.set_ylabel('Lateral G-force', fontsize=14, weight='bold')
ax2.set_ylim(0, 3.5/9.81)
g_ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
ax2.set_yticks([g * 9.81 for g in g_ticks])
ax2.set_yticklabels([f'{g:.2f}g' for g in g_ticks])

# Add explanatory text
ax.text(30, 0.5,
        "Each curve shows the physics relationship: lat_accel = v²/R\n" +
        "Blue circles: Reference targets (conservative)\n" +
        "Red squares: Production targets (aggressive)\n" +
        "Production uses 50% higher minimum lat accel (1.8 vs 1.2)",
        fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3))

plt.tight_layout()

# Save the plot
output_path = '/data/openpilot/docs/chauffeur/vtsc/results/speed_vs_lateral_accel.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also save to Windows-accessible location
windows_path = '/mnt/c/Users/Chris/Pictures/Screenshots/vtsc_speed_vs_lateral_accel.png'
plt.savefig(windows_path, dpi=150, bbox_inches='tight')
print(f"Also saved to Windows: {windows_path}")

# Try to open with Windows default image viewer
import subprocess
try:
    subprocess.run(['explorer.exe', windows_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')])
    print("Opened with Windows Explorer")
except:
    print("Please open: C:\\Users\\Chris\\Pictures\\Screenshots\\vtsc_speed_vs_lateral_accel.png")
