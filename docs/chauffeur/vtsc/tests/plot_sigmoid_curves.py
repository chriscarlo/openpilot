#!/usr/bin/env python3
"""Plot sigmoid curves to visualize the difference between reference and production tuning"""

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

def curvature_to_speed(curvature: float, a_min: float) -> float:
    """Calculate target speed for given curvature and a_min"""
    if curvature < 1e-7:
        return 70.0
    safe_lat_accel = physics_based_lateral_acceleration(curvature, a_min)
    try:
        base_speed_mps = math.sqrt(safe_lat_accel / curvature)
    except (ValueError, ZeroDivisionError):
        base_speed_mps = 0.0
    return min(base_speed_mps, 70.0)

# Generate curvature range
curvatures = np.logspace(-3, -0.7, 200)  # 0.001 to 0.2 (log scale for better visualization)

# Calculate speeds for both tunings
speeds_ref = [curvature_to_speed(c, a_min=1.2) * 2.237 for c in curvatures]  # Convert to mph
speeds_prod = [curvature_to_speed(c, a_min=1.8) * 2.237 for c in curvatures]

# Apply MIN_V floor to production speeds (what actually gets commanded)
MIN_V_MPH = 5.6 * 2.237  # 12.5 mph
speeds_prod_commanded = [max(s, MIN_V_MPH) for s in speeds_prod]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top plot: Speed vs Curvature
ax1.semilogx(curvatures, speeds_ref, 'b-', linewidth=2, label='Reference (a_min=1.2)')
ax1.semilogx(curvatures, speeds_prod, 'r--', linewidth=2, label='Production calc (a_min=1.8)')
ax1.semilogx(curvatures, speeds_prod_commanded, 'r-', linewidth=2, label='Production commanded (with MIN_V floor)')
ax1.axhline(y=MIN_V_MPH, color='orange', linestyle=':', alpha=0.7, label=f'MIN_V floor ({MIN_V_MPH:.1f} mph)')
ax1.axhline(y=50, color='green', linestyle=':', alpha=0.5, label='50 mph threshold')
ax1.axhline(y=45, color='purple', linestyle=':', alpha=0.5, label='45 mph')

# Add shaded regions
ax1.axhspan(0, 50, alpha=0.1, color='red', label='Problem zone (<50mph)')
ax1.axhspan(50, 80, alpha=0.1, color='green', label='Working zone (>50mph)')

ax1.set_xlabel('Curvature (1/m)', fontsize=12)
ax1.set_ylabel('Speed (mph)', fontsize=12)
ax1.set_title('VTSC Sigmoid Curves: Reference vs Production', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 80)

# Add annotations for key points
# Find curvature for 45mph in production
for i, speed in enumerate(speeds_prod_commanded):
    if speed >= 45:
        curv_45 = curvatures[i]
        ax1.annotate(f'45mph @ curv={curv_45:.4f}',
                    xy=(curv_45, 45), xytext=(curv_45*2, 35),
                    arrowprops=dict(arrowstyle='->', color='purple'),
                    fontsize=10, color='purple')
        break

# Bottom plot: Speed difference (error)
speed_diff = [p - r for p, r in zip(speeds_prod_commanded, speeds_ref, strict=False)]
ax2.semilogx(curvatures, speed_diff, 'k-', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax2.fill_between(curvatures, 0, speed_diff, where=[d > 0 for d in speed_diff],
                  alpha=0.3, color='red', label='Production faster than reference')

ax2.set_xlabel('Curvature (1/m)', fontsize=12)
ax2.set_ylabel('Speed Difference (mph)', fontsize=12)
ax2.set_title('Speed Error: Production - Reference', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add radius annotations
radius_labels = [(0.2, "5m"), (0.1, "10m"), (0.05, "20m"), (0.02, "50m"), (0.01, "100m"), (0.005, "200m")]
for curv, label in radius_labels:
    if curv >= curvatures[0] and curv <= curvatures[-1]:
        ax1.axvline(x=curv, color='gray', linestyle=':', alpha=0.3)
        ax1.text(curv, 75, f'R={label}', rotation=90, ha='right', va='bottom', fontsize=8, alpha=0.7)

plt.tight_layout()

# Save the plot
output_path = '/data/openpilot/docs/chauffeur/vtsc/results/sigmoid_curves_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also save to Windows-accessible location for easy viewing
windows_path = '/mnt/c/Users/Chris/Pictures/Screenshots/vtsc_sigmoid_curves.png'
plt.savefig(windows_path, dpi=150, bbox_inches='tight')
print(f"Also saved to Windows: {windows_path}")

# Try to open with Windows default image viewer
import subprocess
try:
    # Method 1: Using explorer.exe
    subprocess.run(['explorer.exe', windows_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')])
    print("Attempted to open with Windows Explorer")
except:
    try:
        # Method 2: Using wslview if available
        subprocess.run(['wslview', windows_path])
        print("Attempted to open with wslview")
    except:
        try:
            # Method 3: Using xdg-open
            subprocess.run(['xdg-open', output_path])
            print("Attempted to open with xdg-open")
        except:
            print("\nCould not auto-open. Please open one of these files manually:")
            print(f"  WSL path: {output_path}")
            print("  Windows path: C:\\Users\\Chris\\Pictures\\Screenshots\\vtsc_sigmoid_curves.png")
