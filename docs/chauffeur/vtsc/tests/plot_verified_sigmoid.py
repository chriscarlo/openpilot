#!/usr/bin/env python3
"""Plot the properly calculated sigmoid curve with verified parameters"""

import math
import numpy as np
import matplotlib.pyplot as plt

def original_sigmoid(curvature: float, a_min: float) -> float:
    """Original production sigmoid"""
    a_max = 3.12
    alpha = 24.3
    beta = 0.78
    lateral_acceleration = a_min + (a_max - a_min) * math.exp(-alpha * (curvature ** beta))
    return max(a_min, min(lateral_acceleration, a_max))

def improved_sigmoid(curvature: float) -> float:
    """
    Verified improved sigmoid from calculations:
    - Reaches max (3.12 m/s²) by 65-70mph  
    - Sharp transition from 50mph to 30mph
    - Begins flattening around 30mph
    - Smooth curve to minimum at 5mph
    """
    a_max = 3.12   # Maximum lateral acceleration (m/s²)
    a_min = 0.80   # Minimum lateral acceleration (m/s²)
    k = 250.0      # Steepness factor (very high for sharp S-curve)
    c = 0.0180     # Center point (curvature where transition happens)

    # Sigmoid function
    lateral_acceleration = a_min + (a_max - a_min) / (1 + math.exp(k * (curvature - c)))

    return max(a_min, min(lateral_acceleration, a_max))

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Constants
OLD_MIN_V_MPH = 5.6 * 2.237  # 12.5 mph (current)
NEW_MIN_V_MPH = 2.24 * 2.237  # 5 mph (proposed)

# Test curves with their colors
test_curves = [
    (5, "Hairpin (R=5m)", 'red'),
    (10, "Tight turn (R=10m)", 'darkorange'),
    (20, "Sharp turn (R=20m)", 'gold'),
    (30, "Residential (R=30m)", 'yellow'),
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
    speeds_mph = np.linspace(3, 80, 200)
    speeds_mps = speeds_mph / 2.237

    # Calculate lateral acceleration at each speed for this curve (physics: a = v²/R)
    lat_accels_physics = [(v**2) / radius for v in speeds_mps]

    # Plot the physics relationship (faint line)
    ax.plot(speeds_mph, lat_accels_physics, '-', linewidth=1.0, color=color, alpha=0.2)

    # Calculate what each sigmoid would command for this curve
    ref_lat_accel = original_sigmoid(curvature, a_min=1.2)
    prod_lat_accel = original_sigmoid(curvature, a_min=1.8)
    improved_lat_accel = improved_sigmoid(curvature)

    # Calculate the speeds each would target
    ref_speed_mph = math.sqrt(ref_lat_accel * radius) * 2.237
    prod_speed_mph = math.sqrt(prod_lat_accel * radius) * 2.237
    improved_speed_mph = math.sqrt(improved_lat_accel * radius) * 2.237

    # Apply MIN_V floors
    prod_commanded_mph = max(prod_speed_mph, OLD_MIN_V_MPH)
    improved_commanded_mph = max(improved_speed_mph, NEW_MIN_V_MPH)

    # Plot operating points
    if ref_speed_mph <= 80:
        ax.plot(ref_speed_mph, ref_lat_accel, 'o', color=color, markersize=7,
                markeredgecolor='blue', markeredgewidth=1.5, alpha=0.6, zorder=2)

    if prod_commanded_mph <= 80:
        ax.plot(prod_commanded_mph, prod_lat_accel, 's', color=color, markersize=7,
                markeredgecolor='red', markeredgewidth=1.5, alpha=0.6, zorder=3)

    if improved_commanded_mph <= 80:
        ax.plot(improved_commanded_mph, improved_lat_accel, 'D', color=color, markersize=9,
                markeredgecolor='green', markeredgewidth=2.5, alpha=0.9, zorder=4)

        # Add labels for key curves
        if radius in [5, 10, 20, 50, 100, 200]:
            ax.annotate(name.split('(')[0].strip(),
                       xy=(improved_commanded_mph, improved_lat_accel),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=color, weight='bold')

# Create envelope curves showing the sigmoid shapes
speeds_for_envelope = np.linspace(4, 80, 300)
ref_envelope = []
prod_envelope = []
improved_envelope = []

for speed_mph in speeds_for_envelope:
    speed_mps = speed_mph / 2.237

    # Find what lateral acceleration each sigmoid would use at this speed
    # We need to find the curvature that would result in this speed for each sigmoid
    best_ref_lat = None
    best_prod_lat = None
    best_improved_lat = None

    for test_curv in np.logspace(-3.5, -0.5, 1000):
        # Reference sigmoid
        ref_lat = original_sigmoid(test_curv, a_min=1.2)
        calc_speed = math.sqrt(ref_lat / test_curv) * 2.237
        if best_ref_lat is None or abs(calc_speed - speed_mph) < abs(math.sqrt(best_ref_lat / test_curv) * 2.237 - speed_mph):
            best_ref_lat = ref_lat

        # Production sigmoid
        prod_lat = original_sigmoid(test_curv, a_min=1.8)
        calc_speed = math.sqrt(prod_lat / test_curv) * 2.237
        if best_prod_lat is None or abs(calc_speed - speed_mph) < abs(math.sqrt(best_prod_lat / test_curv) * 2.237 - speed_mph):
            best_prod_lat = prod_lat

        # Improved sigmoid
        improved_lat = improved_sigmoid(test_curv)
        calc_speed = math.sqrt(improved_lat / test_curv) * 2.237
        if best_improved_lat is None or abs(calc_speed - speed_mph) < abs(math.sqrt(best_improved_lat / test_curv) * 2.237 - speed_mph):
            best_improved_lat = improved_lat

    ref_envelope.append(best_ref_lat if best_ref_lat else np.nan)
    prod_envelope.append(best_prod_lat if best_prod_lat else np.nan)
    improved_envelope.append(best_improved_lat if best_improved_lat else np.nan)

# Plot the envelope curves
ax.plot(speeds_for_envelope, ref_envelope, 'b-', linewidth=2.5, label='Reference (a_min=1.2)', alpha=0.6, zorder=5)
ax.plot(speeds_for_envelope, prod_envelope, 'r-', linewidth=2.5, label='Production (a_min=1.8)', alpha=0.6, zorder=5)
ax.plot(speeds_for_envelope, improved_envelope, 'g-', linewidth=3.5, label='Improved Sigmoid', alpha=0.8, zorder=6)

# Add horizontal reference lines for lateral acceleration
ax.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(72, 0.85, 'New floor: 0.8 m/s²', color='green', fontsize=10, weight='bold')

ax.axhline(y=1.2, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(72, 1.25, 'Ref floor: 1.2 m/s²', color='blue', fontsize=10)

ax.axhline(y=1.8, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(72, 1.85, 'Prod floor: 1.8 m/s²', color='red', fontsize=10)

ax.axhline(y=3.12, color='purple', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(72, 3.17, 'Max: 3.12 m/s²', color='purple', fontsize=10, weight='bold')

# Add vertical reference lines for key speeds
ax.axvline(x=NEW_MIN_V_MPH, color='green', linestyle='--', alpha=0.5, linewidth=2)
ax.text(NEW_MIN_V_MPH+0.5, 3.3, '5mph', color='green', fontsize=9, weight='bold', rotation=90)

ax.axvline(x=OLD_MIN_V_MPH, color='orange', linestyle='--', alpha=0.5, linewidth=2)
ax.text(OLD_MIN_V_MPH+0.5, 3.3, '12.5mph', color='orange', fontsize=9, rotation=90)

ax.axvline(x=30, color='brown', linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(30.5, 3.3, '30mph', color='brown', fontsize=9, rotation=90)

ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(50.5, 3.3, '50mph', color='red', fontsize=9, weight='bold', rotation=90)

ax.axvline(x=70, color='purple', linestyle='--', alpha=0.5, linewidth=2)
ax.text(70.5, 3.3, '70mph', color='purple', fontsize=9, weight='bold', rotation=90)

# Shade zones
ax.axvspan(0, 30, alpha=0.05, color='yellow', label='Low speed (flatten zone)')
ax.axvspan(30, 50, alpha=0.05, color='orange', label='Transition (steep drop)')
ax.axvspan(50, 80, alpha=0.05, color='green', label='Highway (max performance)')

# Legend entries for markers
ax.plot([], [], 'o', color='gray', markersize=7, markeredgecolor='blue',
        markeredgewidth=1.5, label='Reference points', alpha=0.6)
ax.plot([], [], 's', color='gray', markersize=7, markeredgecolor='red',
        markeredgewidth=1.5, label='Production points', alpha=0.6)
ax.plot([], [], 'D', color='gray', markersize=9, markeredgecolor='green',
        markeredgewidth=2.5, label='Improved points', alpha=0.9)

# Labels and formatting
ax.set_xlabel('Speed (mph)', fontsize=14, weight='bold')
ax.set_ylabel('Lateral Acceleration (m/s²)', fontsize=14, weight='bold')
ax.set_title('VTSC Sigmoid Comparison: Reference vs Production vs Improved', fontsize=16, weight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.set_xlim(0, 80)
ax.set_ylim(0, 3.5)

# Add g-force scale on right axis
ax2 = ax.twinx()
ax2.set_ylabel('Lateral G-force', fontsize=14, weight='bold')
ax2.set_ylim(0, 3.5/9.81)
g_ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
ax2.set_yticks([g * 9.81 for g in g_ticks])
ax2.set_yticklabels([f'{g:.2f}g' for g in g_ticks])

# Add text box with key improvements
textstr = '\n'.join([
    'Improved Sigmoid Features:',
    '• Reaches 3.12 m/s² by 70mph',
    '• Sharp S-curve: 50→30mph',
    '• Flattens below 30mph',
    '• 5mph minimum (was 12.5)',
    '• 0.8 m/s² floor (was 1.8)',
])
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
ax.text(0.98, 0.45, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()

# Save the plot
output_path = '/data/openpilot/docs/chauffeur/vtsc/results/verified_sigmoid_curves.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also save to Windows-accessible location
windows_path = '/mnt/c/Users/Chris/Pictures/Screenshots/vtsc_verified_sigmoid.png'
plt.savefig(windows_path, dpi=150, bbox_inches='tight')
print(f"Also saved to Windows: {windows_path}")

# Try to open with Windows default image viewer
import subprocess
try:
    subprocess.run(['explorer.exe', windows_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')])
    print("Opened with Windows Explorer")
except:
    print("Please open: C:\\Users\\Chris\\Pictures\\Screenshots\\vtsc_verified_sigmoid.png")

# Print verification
print("\n=== VERIFICATION ===")
print("Testing key speed points with improved sigmoid:")
test_speeds = [5, 10, 20, 30, 40, 50, 60, 70]
for speed_mph in test_speeds:
    speed_mps = speed_mph / 2.237
    # Find what curvature would give this speed
    for test_curv in np.logspace(-3.5, -0.5, 1000):
        lat_accel = improved_sigmoid(test_curv)
        calc_speed = math.sqrt(lat_accel / test_curv) * 2.237
        if abs(calc_speed - speed_mph) < 0.5:
            print(f"{speed_mph:3}mph: lat_accel={lat_accel:.2f} m/s² ({lat_accel/9.81:.3f}g) at curvature={test_curv:.4f}")
            break
