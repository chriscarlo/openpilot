#!/usr/bin/env python3
"""Debug script to understand exact safe_speeds calculation"""

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

# Import VTSC
sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
from vision_turn_controller import curvature_to_speed

# Hairpin scenario parameters
v_ego = 11.2  # 25 mph
v_cruise = 15.6  # 35 mph
max_curvature = 0.15

# Generate hairpin curvature profile (Gaussian)
points = 33
x = np.linspace(-3, 3, points)
curvature_profile = max_curvature * np.exp(-x**2 / 0.5)

print("=== Exact VTSC Logic Simulation ===")
print(f"v_ego = {v_ego:.1f} m/s ({v_ego*2.237:.1f} mph)")
print(f"v_cruise = {v_cruise:.1f} m/s ({v_cruise*2.237:.1f} mph)")

# This is what VTSC does:
safe_speeds = np.array([curvature_to_speed(abs(curv)) for curv in curvature_profile])
overshoot_mask = safe_speeds < v_ego
lat_acc_overshoot_ahead = np.any(overshoot_mask)

print(f"\nlat_acc_overshoot_ahead = {lat_acc_overshoot_ahead}")

if lat_acc_overshoot_ahead:
    overshoot_idx = np.where(overshoot_mask)[0][0]
    v_overshoot_raw = safe_speeds[overshoot_idx]
    v_overshoot = min(v_overshoot_raw, v_cruise)

    print(f"\nFirst overshoot point: index {overshoot_idx}")
    print(f"  Curvature at index {overshoot_idx}: {curvature_profile[overshoot_idx]:.6f}")
    print(f"  Safe speed at index {overshoot_idx}: {v_overshoot_raw:.1f} m/s ({v_overshoot_raw*2.237:.1f} mph)")
    print(f"  v_overshoot (after min with v_cruise): {v_overshoot:.1f} m/s ({v_overshoot*2.237:.1f} mph)")

    print("\n=== Checking specific indices ===")
    for i in range(max(0, overshoot_idx-2), min(points, overshoot_idx+3)):
        curv = curvature_profile[i]
        safe = safe_speeds[i]
        overshoot = safe < v_ego
        print(f"  Index {i:2d}: curv={curv:.6f}, safe_speed={safe:.1f} m/s ({safe*2.237:.1f} mph), overshoot={overshoot}")

    # Now check if 9.1 m/s appears anywhere
    print("\n=== Looking for 9.1 m/s in safe_speeds ===")
    for i, speed in enumerate(safe_speeds):
        if abs(speed - 9.1) < 0.2:
            print(f"  Index {i}: {speed:.1f} m/s (curvature={curvature_profile[i]:.6f})")

    # Check if any transformation could give 9.1
    print("\n=== Checking possible transformations ===")
    print(f"  v_overshoot_raw * 1.2 = {v_overshoot_raw * 1.2:.1f} m/s")
    print(f"  v_overshoot_raw * 1.1 = {v_overshoot_raw * 1.1:.1f} m/s")
    print(f"  sqrt(v_overshoot_raw * v_ego) = {np.sqrt(v_overshoot_raw * v_ego):.1f} m/s")
