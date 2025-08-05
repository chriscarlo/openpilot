#!/usr/bin/env python3
"""Debug script to understand why hairpin speed is 20 mph instead of expected ~8 mph"""

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
from vision_turn_controller import VisionTurnController, curvature_to_speed

# Test the physics function directly
print("=== Direct Physics Calculation ===")
for curv in [0.15, 0.2, 0.05, 0.01]:
    speed = curvature_to_speed(curv)
    print(f"Curvature {curv:.2f}: {speed:.1f} m/s = {speed*2.237:.1f} mph")

print("\n=== Simulating Hairpin Test Scenario ===")

# Mock CP
class MockCP:
    steerRatio = 15.0
    wheelbase = 2.7

# Create VTSC
vtsc = VisionTurnController(MockCP())

# Hairpin scenario parameters
v_ego = 11.2  # 25 mph
v_cruise = 15.6  # 35 mph
max_curvature = 0.15

# Generate hairpin curvature profile (Gaussian)
points = 33
x = np.linspace(-3, 3, points)
curvature_profile = max_curvature * np.exp(-x**2 / 0.5)

print("\nCurvature profile stats:")
print(f"  Peak curvature: {max(curvature_profile):.3f}")
print(f"  Peak at index: {np.argmax(curvature_profile)}")
print(f"  First 5 values: {curvature_profile[:5]}")

# Create mock SM data
class MockSM:
    def __init__(self):
        self.data = {}
        self.valid = {'modelV2': True, 'carState': True, 'lateralPlan': True}

    def __getitem__(self, key):
        if key == 'modelV2':
            return self.modelV2
        elif key == 'carState':
            return self.carState
        elif key == 'lateralPlan':
            return self.lateralPlan
        return None

sm = MockSM()

# Set up modelV2
class ModelV2:
    def __init__(self):
        self.orientationRate = type('', (), {'z': []})()
        self.velocity = type('', (), {'x': []})()

sm.modelV2 = ModelV2()
sm.modelV2.orientationRate.z = [curv * v_ego for curv in curvature_profile]
sm.modelV2.velocity.x = [v_ego] * 33

# Set up carState
sm.carState = type('', (), {
    'gasPressed': False,
    'steeringAngleDeg': 0.0
})()

# Set up lateralPlan (empty for now)
sm.lateralPlan = type('', (), {
    'psis': [0.0] * 50,
    'dPathPoints': [0.0] * 50
})()

print("\nInitial conditions:")
print(f"  v_ego: {v_ego:.1f} m/s = {v_ego*2.237:.1f} mph")
print(f"  v_cruise: {v_cruise:.1f} m/s = {v_cruise*2.237:.1f} mph")

# Manually trace through the key calculations
print("\n=== Manual Calculation Trace ===")

# What VTSC would calculate
orientation_rates = sm.modelV2.orientationRate.z
velocities = sm.modelV2.velocity.x

# Find max curvature
curvatures = [abs(rate) / max(vel, 1.0) for rate, vel in zip(orientation_rates, velocities, strict=False)]
max_pred_curvature = max(curvatures)
print(f"Max predicted curvature: {max_pred_curvature:.6f}")

# EMA filter simulation
filtered_curvature = 0.0
ema_ratio = 0.3
for i in range(30):  # 30 updates like the test
    filtered_curvature = (1 - ema_ratio) * filtered_curvature + ema_ratio * max_pred_curvature
    if i < 5 or i == 29:
        print(f"  After update {i+1}: filtered_curvature = {filtered_curvature:.6f}")

# Calculate physics speed
physics_speed = curvature_to_speed(filtered_curvature)
print(f"\nPhysics speed for filtered curvature: {physics_speed:.1f} m/s = {physics_speed*2.237:.1f} mph")

# Now run actual VTSC update
print("\n=== Running VTSC Updates ===")
vtsc._is_enabled = True  # Enable VTSC

# Initialize and run updates
for i in range(5):
    vtsc.update(sm, True, v_ego, 0.0, v_cruise)

    print(f"\nAfter update {i+1}:")
    print(f"  _filtered_curvature: {vtsc._filtered_curvature:.6f}")
    print(f"  _prev_target_speed: {vtsc._prev_target_speed:.1f} m/s = {vtsc._prev_target_speed*2.237:.1f} mph")
    print(f"  v_turn property: {vtsc.v_turn:.1f} m/s = {vtsc.v_turn*2.237:.1f} mph")
    print(f"  _lat_acc_overshoot_ahead: {vtsc._lat_acc_overshoot_ahead}")
    if vtsc._lat_acc_overshoot_ahead:
        print(f"  _v_overshoot: {vtsc._v_overshoot:.1f} m/s = {vtsc._v_overshoot*2.237:.1f} mph")
        print(f"  _v_overshoot_distance: {vtsc._v_overshoot_distance:.1f} m")
