#!/usr/bin/env python3
"""Plot speed vs lateral acceleration to show how the sigmoid affects real-world driving"""

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

# Define test curves by their radius (more intuitive than curvature)
test_curves = [
    (5, "Hairpin (R=5m)"),
    (10, "Tight turn (R=10m)"),
    (20, "Sharp turn (R=20m)"),
    (50, "Normal turn (R=50m)"),
    (100, "Highway curve (R=100m)"),
    (200, "Highway sweeper (R=200m)"),
]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Constants
MIN_V_MPH = 5.6 * 2.237  # 12.5 mph
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(test_curves)))

# Left plot: How sigmoid maps curvature to lateral acceleration
curvatures = np.logspace(-3, -0.7, 200)  # 0.001 to 0.2
lat_accels_ref = [physics_based_lateral_acceleration(c, a_min=1.2) for c in curvatures]
lat_accels_prod = [physics_based_lateral_acceleration(c, a_min=1.8) for c in curvatures]

ax1.semilogx(curvatures, lat_accels_ref, 'b-', linewidth=2.5, label='Reference (a_min=1.2 m/s²)')
ax1.semilogx(curvatures, lat_accels_prod, 'r-', linewidth=2.5, label='Production (a_min=1.8 m/s²)')
ax1.axhline(y=1.2, color='blue', linestyle=':', alpha=0.5, label='Reference floor')
ax1.axhline(y=1.8, color='red', linestyle=':', alpha=0.5, label='Production floor')

# Add curve radius labels
for radius, name in test_curves:
    curv = 1.0 / radius
    if curv >= curvatures[0] and curv <= curvatures[-1]:
        ax1.axvline(x=curv, color='gray', linestyle=':', alpha=0.3)
        ax1.text(curv, 3.0, name.split('(')[0].strip(), rotation=90, ha='right', va='bottom', fontsize=9, alpha=0.7)

ax1.set_xlabel('Curvature (1/m)', fontsize=12)
ax1.set_ylabel('Lateral Acceleration Limit (m/s²)', fontsize=12)
ax1.set_title('Sigmoid Function: How Curvature Maps to Lat Accel', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')
ax1.set_ylim(1.0, 3.3)

# Add g-force reference
ax1_right = ax1.twinx()
ax1_right.set_ylabel('Lateral G-force', fontsize=12)
ax1_right.set_ylim(1.0/9.81, 3.3/9.81)

# Right plot: Speed vs Lateral Acceleration for different curve radii
ax2.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50 mph threshold')
ax2.axhline(y=45, color='purple', linestyle='--', alpha=0.5, label='45 mph')
ax2.axhline(y=MIN_V_MPH, color='orange', linestyle='--', alpha=0.5, label=f'MIN_V floor ({MIN_V_MPH:.1f} mph)')

# For each curve radius, plot how speed varies with lateral acceleration
lat_accel_range = np.linspace(1.0, 3.2, 100)

for i, (radius, name) in enumerate(test_curves):
    curvature = 1.0 / radius
    # Speed = sqrt(a_lat / curvature)
    speeds_ideal = [math.sqrt(a / curvature) * 2.237 for a in lat_accel_range]  # Convert to mph
    ax2.plot(lat_accel_range, speeds_ideal, '-', linewidth=2, color=colors[i], label=name)

    # Mark where reference and production sigmoids would put this curve
    ref_lat_accel = physics_based_lateral_acceleration(curvature, a_min=1.2)
    prod_lat_accel = physics_based_lateral_acceleration(curvature, a_min=1.8)

    ref_speed = math.sqrt(ref_lat_accel / curvature) * 2.237
    prod_speed = math.sqrt(prod_lat_accel / curvature) * 2.237
    prod_commanded = max(prod_speed, MIN_V_MPH)

    # Plot reference point
    ax2.plot(ref_lat_accel, ref_speed, 'o', color=colors[i], markersize=8,
             markeredgecolor='blue', markeredgewidth=2)

    # Plot production point
    ax2.plot(prod_lat_accel, prod_commanded, 's', color=colors[i], markersize=8,
             markeredgecolor='red', markeredgewidth=2)

# Add legend entries for markers
ax2.plot([], [], 'o', color='gray', markersize=8, markeredgecolor='blue',
         markeredgewidth=2, label='Reference operating point')
ax2.plot([], [], 's', color='gray', markersize=8, markeredgecolor='red',
         markeredgewidth=2, label='Production operating point')

# Shade the regions
ax2.axvspan(1.0, 1.2, alpha=0.15, color='blue', label='Ref never goes below')
ax2.axvspan(1.0, 1.8, alpha=0.15, color='red', label='Prod never goes below')

ax2.set_xlabel('Lateral Acceleration (m/s²)', fontsize=12)
ax2.set_ylabel('Speed (mph)', fontsize=12)
ax2.set_title('Speed vs Lateral Acceleration for Different Curves', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', ncol=2, fontsize=9)
ax2.set_xlim(1.0, 3.3)
ax2.set_ylim(0, 80)

# Add g-force scale on top
ax2_top = ax2.twiny()
ax2_top.set_xlabel('Lateral G-force', fontsize=12)
g_ticks = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
ax2_top.set_xticks(g_ticks * 9.81)
ax2_top.set_xticklabels([f'{g:.2f}g' for g in g_ticks])
ax2_top.set_xlim(1.0, 3.3)

# Add text annotation explaining the problem
ax2.text(2.5, 20, "Production (red squares) uses\nhigher lat accel than Reference\n(blue circles) for same curves,\nresulting in higher speeds",
         fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3))

plt.tight_layout()

# Save the plot
output_path = '/data/openpilot/docs/chauffeur/vtsc/results/lateral_accel_curves.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also save to Windows-accessible location
windows_path = '/mnt/c/Users/Chris/Pictures/Screenshots/vtsc_lateral_accel_curves.png'
plt.savefig(windows_path, dpi=150, bbox_inches='tight')
print(f"Also saved to Windows: {windows_path}")

# Try to open with Windows default image viewer
import subprocess
try:
    subprocess.run(['explorer.exe', windows_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')])
    print("Opened with Windows Explorer")
except:
    print("Please open: C:\\Users\\Chris\\Pictures\\Screenshots\\vtsc_lateral_accel_curves.png")
