#!/usr/bin/env python3
"""Debug script to understand why _v_overshoot is 9.1 m/s instead of 3.5 m/s"""

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
max_curvature = 0.15

# Generate hairpin curvature profile (Gaussian)
points = 33
x = np.linspace(-3, 3, points)
curvature_profile = max_curvature * np.exp(-x**2 / 0.5)

print("=== Curvature Profile Analysis ===")
print(f"Peak curvature: {max(curvature_profile):.3f} at index {np.argmax(curvature_profile)}")
print("\nCurvature at different trajectory points:")
for i in [0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32]:
    curv = curvature_profile[i]
    speed = curvature_to_speed(curv) if curv > 1e-7 else 70.0
    dist = i * 2  # Approx 2m per index
    print(f"  Index {i:2d} ({dist:3d}m ahead): curvature={curv:.6f}, physics_speed={speed:.1f} m/s = {speed*2.237:.1f} mph")

print("\n=== Understanding the Problem ===")
print("The test generates a Gaussian curve profile where:")
print("- Indices 0-4: Nearly straight (curvature ≈ 0)")
print("- Index 16: Peak curvature (0.15)")
print("- Indices 28-32: Nearly straight again")

print("\nWhen VTSC looks for overshoot, it finds the FIRST point that exceeds threshold.")
print("The threshold is based on lateral acceleration, not just curvature.")

# Calculate lateral accelerations
print("\n=== Lateral Acceleration Analysis ===")
lateral_accs = [curv * v_ego**2 for curv in curvature_profile]
threshold = 0.5  # Example threshold
print(f"With v_ego = {v_ego:.1f} m/s ({v_ego*2.237:.1f} mph):")
for i in [0, 1, 2, 3, 4, 8, 12, 16]:
    lat_acc = lateral_accs[i]
    exceeds = "YES" if lat_acc > threshold else "NO"
    print(f"  Index {i:2d}: lat_acc={lat_acc:.3f} m/s², exceeds threshold: {exceeds}")

# Find first index that exceeds threshold
first_exceed_idx = next((i for i, acc in enumerate(lateral_accs) if acc > threshold), None)
if first_exceed_idx is not None:
    curv_at_exceed = curvature_profile[first_exceed_idx]
    speed_at_exceed = curvature_to_speed(curv_at_exceed) if curv_at_exceed > 1e-7 else 70.0
    print(f"\nFirst point exceeding threshold: index {first_exceed_idx}")
    print(f"  Curvature: {curv_at_exceed:.6f}")
    print(f"  Physics speed: {speed_at_exceed:.1f} m/s = {speed_at_exceed*2.237:.1f} mph")
    print("  THIS is likely what _v_overshoot is set to!")

    # Compare to what we see
    print("\n  Observed _v_overshoot: 9.1 m/s")
    print(f"  Calculated for index {first_exceed_idx}: {speed_at_exceed:.1f} m/s")
    if abs(speed_at_exceed - 9.1) < 0.5:
        print("  ✓ MATCH! This explains the 20.3 mph speed!")
