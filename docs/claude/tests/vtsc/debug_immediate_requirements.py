#!/usr/bin/env python3
"""Debug script to understand if immediate requirements logic is working"""

import sys
import numpy as np

# Setup paths
sys.path.insert(0, '.')
from stub_cereal import custom
from stub_openpilot import openpilot

# Install stubs
sys.modules['cereal'] = type('cereal', (), {'custom': custom})
sys.modules['openpilot'] = openpilot
sys.modules['openpilot.common'] = openpilot.common
sys.modules['openpilot.common.params'] = openpilot.common.params
sys.modules['openpilot.common.conversions'] = openpilot.common.conversions
sys.modules['openpilot.common.numpy_fast'] = openpilot.common.numpy_fast
sys.modules['openpilot.selfdrive'] = openpilot.selfdrive
sys.modules['openpilot.selfdrive.car'] = openpilot.selfdrive.car
sys.modules['openpilot.selfdrive.car.cruise'] = openpilot.selfdrive.car.cruise
sys.modules['openpilot.selfdrive.modeld'] = openpilot.selfdrive.modeld
sys.modules['openpilot.selfdrive.modeld.constants'] = openpilot.selfdrive.modeld.constants
sys.modules['openpilot.selfdrive.controls'] = openpilot.selfdrive.controls
sys.modules['openpilot.selfdrive.controls.lib'] = openpilot.selfdrive.controls.lib
sys.modules['openpilot.selfdrive.controls.lib.drive_helpers'] = openpilot.selfdrive.controls.lib.drive_helpers

# Import VTSC and ModelConstants
sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
from vision_turn_controller import curvature_to_speed
from openpilot.selfdrive.modeld.constants import ModelConstants

# Hairpin scenario parameters
v_ego = 11.2  # 25 mph
max_curvature = 0.15

# Generate hairpin curvature profile (Gaussian)
points = 33
x = np.linspace(-3, 3, points)
curvature_profile = max_curvature * np.exp(-x**2 / 0.5)

# Calculate safe speeds
safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in curvature_profile])
overshoot_mask = safe_speeds < v_ego
overshoot_indices = np.where(overshoot_mask)[0]

print("=== Simulating VTSC Immediate Requirements Logic ===")
print(f"v_ego = {v_ego:.1f} m/s ({v_ego*2.237:.1f} mph)")
print(f"\nOvershoot indices: {overshoot_indices}")

# Get time indices
times = np.array(ModelConstants.T_IDXS[:points])

# For each overshoot point, calculate immediate requirements
max_decel = 3.5  # m/s²
immediate_requirements = []

print("\n=== Checking Each Overshoot Point ===")
for idx in overshoot_indices:
    # How much distance do we need to slow down to this point's safe speed?
    speed_diff_sq = safe_speeds[idx]**2 - v_ego**2
    decel_distance_needed = abs(speed_diff_sq) / (2 * max_decel)

    # How far away is this point?
    point_distance = times[idx] * v_ego

    # Do we need to start slowing NOW for this point?
    needs_immediate = point_distance <= decel_distance_needed * 1.2  # 20% safety margin

    if idx in [11, 12, 16]:  # Key points
        print(f"\nIndex {idx}:")
        print(f"  Safe speed: {safe_speeds[idx]:.1f} m/s ({safe_speeds[idx]*2.237:.1f} mph)")
        print(f"  Curvature: {curvature_profile[idx]:.6f}")
        print(f"  Distance to point: {point_distance:.1f} m")
        print(f"  Decel distance needed: {decel_distance_needed:.1f} m")
        print(f"  With 20% margin: {decel_distance_needed * 1.2:.1f} m")
        print(f"  Needs immediate action: {needs_immediate}")

    if needs_immediate:
        immediate_requirements.append((idx, safe_speeds[idx], point_distance))

print("\n=== Immediate Requirements Summary ===")
if immediate_requirements:
    print(f"Found {len(immediate_requirements)} points needing immediate action:")
    for idx, speed, dist in immediate_requirements:
        print(f"  Index {idx}: {speed:.1f} m/s at {dist:.1f}m")

    # Find minimum required speed
    min_required_speed = min([speed for _, speed, _ in immediate_requirements])
    print(f"\nMinimum required speed: {min_required_speed:.1f} m/s ({min_required_speed*2.237:.1f} mph)")

    # Find which index would be selected
    for idx, speed, dist in immediate_requirements:
        if speed == min_required_speed:
            print(f"Selected index: {idx} (curvature={curvature_profile[idx]:.6f})")
            print(f"This should be _v_overshoot = {speed:.1f} m/s ({speed*2.237:.1f} mph)")
            break
else:
    print("No immediate requirements found!")
    print(f"Would use first overshoot: index {overshoot_indices[0]}")
    print(f"Speed: {safe_speeds[overshoot_indices[0]]:.1f} m/s")
