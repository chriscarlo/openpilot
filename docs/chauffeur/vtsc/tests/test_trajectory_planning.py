#!/usr/bin/env python3
"""Test script to understand proper trajectory planning for curves"""

import numpy as np
import matplotlib.pyplot as plt

# Hairpin scenario parameters
v_ego = 11.2  # 25 mph
max_curvature = 0.15
max_decel = -3.5  # m/s²

# Generate hairpin curvature profile (Gaussian)
points = 33
x = np.linspace(-3, 3, points)
curvature_profile = max_curvature * np.exp(-x**2 / 0.5)

# Time stamps for trajectory points (from ModelConstants.T_IDXS)
times = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
                  1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                  2.2, 2.4])[:points]

# Approximate distances (integrate velocity over time)
distances = times * v_ego

# Calculate safe speeds for each point
# Using simplified physics: v = sqrt(a_lat_max / curvature)
a_lat_max = 1.8  # m/s² (from sigmoid a_min)
safe_speeds = np.array([np.sqrt(a_lat_max / curv) if curv > 1e-7 else 70.0
                        for curv in curvature_profile])

print("=== Trajectory Planning Analysis ===")
print(f"v_ego = {v_ego:.1f} m/s ({v_ego*2.237:.1f} mph)")
print(f"max_decel = {max_decel:.1f} m/s²")

# For each point, calculate when we need to start slowing
print("\n=== Backward Pass Planning ===")
print("For each point, when do we need to start slowing?")

anticipation_times = []
for i in range(points):
    if safe_speeds[i] < v_ego:
        # Need to slow down for this point
        # Using kinematic equation: v² = u² + 2as
        # Solving for distance: s = (v² - u²) / (2a)
        speed_diff = safe_speeds[i]**2 - v_ego**2
        decel_distance = speed_diff / (2 * max_decel)

        # When should we start slowing?
        # Point is at distance[i], we need decel_distance to slow down
        start_slowing_distance = distances[i] - decel_distance
        start_slowing_time = start_slowing_distance / v_ego if start_slowing_distance > 0 else 0

        anticipation_times.append(start_slowing_time)

        if i in [11, 12, 16, 20, 21]:  # Key points
            print(f"  Point {i:2d} ({distances[i]:5.1f}m, t={times[i]:.2f}s):")
            print(f"    Target speed: {safe_speeds[i]:.1f} m/s ({safe_speeds[i]*2.237:.1f} mph)")
            print(f"    Need {decel_distance:.1f}m to decelerate")
            print(f"    Start slowing at: {start_slowing_distance:.1f}m (t={start_slowing_time:.2f}s)")
    else:
        anticipation_times.append(float('inf'))

# Find the most restrictive requirement at t=0
print("\n=== Current Speed Requirement ===")
current_requirements = []
for i in range(points):
    if anticipation_times[i] <= 0:  # Need to be slowing NOW for this point
        current_requirements.append((i, safe_speeds[i]))

if current_requirements:
    print("Points requiring immediate action:")
    for idx, speed in current_requirements:
        print(f"  Point {idx}: need {speed:.1f} m/s for curvature {curvature_profile[idx]:.6f}")

    # The current target should be based on the NEAREST point that needs action
    # But also considering the TIGHTEST point ahead
    nearest_idx = min([idx for idx, _ in current_requirements])
    tightest_speed = min([speed for _, speed in current_requirements])

    print(f"\nCurrent approach (first): targets point {nearest_idx} at {safe_speeds[nearest_idx]:.1f} m/s")
    print(f"Better approach (tightest): targets {tightest_speed:.1f} m/s for apex")

print("\n=== The Problem ===")
print("Current VTSC picks ONE point and ONE speed (_v_overshoot)")
print("But it SHOULD create a speed profile for the ENTIRE trajectory:")
print("- Each point has a target speed")
print("- Each point has an anticipation time (when to start slowing)")
print("- At any moment, use the most restrictive requirement")
print("- For apex boost, apply at ACTUAL position, not anticipated")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Curvature profile
axes[0].plot(distances, curvature_profile, 'b-', label='Curvature')
axes[0].axhline(y=0.026, color='r', linestyle='--', alpha=0.5, label='First overshoot (0.026)')
axes[0].axhline(y=0.15, color='g', linestyle='--', alpha=0.5, label='Peak (0.15)')
axes[0].set_ylabel('Curvature (1/m)')
axes[0].legend()
axes[0].grid(True)

# Safe speeds
axes[1].plot(distances, safe_speeds, 'g-', label='Safe speeds')
axes[1].axhline(y=v_ego, color='b', linestyle='--', label=f'v_ego ({v_ego:.1f} m/s)')
axes[1].axhline(y=9.1, color='r', linestyle='--', alpha=0.5, label='Current _v_overshoot (9.1 m/s)')
axes[1].axhline(y=3.5, color='g', linestyle='--', alpha=0.5, label='Apex speed (3.5 m/s)')
axes[1].set_ylabel('Speed (m/s)')
axes[1].legend()
axes[1].grid(True)

# Anticipation requirements
anticipation_distances = [times[i] * v_ego - (v_ego**2 - safe_speeds[i]**2)/(2*max_decel)
                          if safe_speeds[i] < v_ego else float('inf')
                          for i in range(points)]
valid_anticipation = [d if d != float('inf') else None for d in anticipation_distances]
axes[2].plot(distances, valid_anticipation, 'r-', label='Start slowing distance')
axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[2].set_xlabel('Distance (m)')
axes[2].set_ylabel('Start slowing point (m)')
axes[2].legend()
axes[2].grid(True)

plt.suptitle('Trajectory Planning for Hairpin Curve')
plt.tight_layout()
plt.savefig('/data/openpilot/docs/chauffeur/vtsc/results/trajectory_planning.png')
print("\nVisualization saved to trajectory_planning.png")
