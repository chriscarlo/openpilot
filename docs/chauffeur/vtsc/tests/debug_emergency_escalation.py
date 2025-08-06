#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../shared'))

from vtsc_test_framework import VTSCTestBase, EmergencyLevel

# Create test instance
test = VTSCTestBase()
test.setUp()

print("=== DEBUGGING EMERGENCY ESCALATION ===")

# Test 1: Emergency level determination
print("\n1. Emergency level determination:")
levels = [
    (-1.0, "should be NORMAL"),
    (-1.47, "should be NORMAL (boundary)"),
    (-2.0, "should be CAUTION"),
    (-2.45, "should be CAUTION (boundary)"),
    (-3.5, "should be WARNING"),
    (-3.92, "should be WARNING (boundary)"),
    (-4.0, "should be WARNING or CRITICAL?"),
    (-5.0, "should be CRITICAL"),
    (-5.50, "should be CRITICAL (boundary)"),
    (-6.0, "should be INTERVENTION")
]

for decel, expected in levels:
    result = test.vtsc._determine_emergency_level(decel, 0.0)
    print(f"  {decel:5.2f} m/s² -> {result.name:12} ({expected})")

# Test 2: Jerk limiting behavior
print("\n2. Jerk limiting behavior:")
print("Starting state:")
print(f"  Emergency level: {test.vtsc._emergency_level.name}")
print(f"  Current decel: {test.vtsc._current_decel:.3f}")

# Request -3.0 decel (should trigger emergency level change)
dt = 0.1
result_decel = test.vtsc._get_optimal_deceleration(-3.0, dt)
print("\nAfter requesting -3.0 m/s² with dt=0.1:")
print(f"  Emergency level: {test.vtsc._emergency_level.name}")
print(f"  Current decel: {test.vtsc._current_decel:.3f}")
print(f"  Returned decel: {result_decel:.3f}")
print(f"  Time at level: {test.vtsc._time_at_current_level:.3f}")

# Test 3: Positive deceleration (acceleration)
print("\n3. Positive deceleration (acceleration):")
test.setUp()  # Reset
result_decel = test.vtsc._get_optimal_deceleration(2.0, 0.1)
print("Requested +2.0 m/s² (acceleration):")
print(f"  Emergency level: {test.vtsc._emergency_level.name}")
print(f"  Returned decel: {result_decel:.3f}")

# Test 4: Time accumulation
print("\n4. Time accumulation:")
test.setUp()  # Reset
print("Initial state:")
print(f"  Time at level: {test.vtsc._time_at_current_level:.3f}")

# Stay at same level for multiple steps
for i in range(3):
    test.vtsc._get_optimal_deceleration(-2.0, 0.1)
    print(f"  Step {i+1}: time_at_level = {test.vtsc._time_at_current_level:.3f}")

# Test 5: Understanding the actual jerk limits
print("\n5. Jerk limits from constants:")
# Import directly from production VTSC
import sys
sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
from vision_turn_controller import JERK_LIMITS
for level in EmergencyLevel:
    print(f"  {level.name:12}: {JERK_LIMITS[level]:6.1f} m/s³")

# Test 6: Calculate expected jerk limited change
print("\n6. Expected jerk-limited decel from 0 to -3.0:")
dt = 0.1
# From NORMAL level (jerk limit -2.0 m/s³)
max_change = 2.0 * dt  # 0.2 m/s²
expected = 0.0 - max_change  # -0.2 m/s²
print(f"  Expected first step: {expected:.3f} m/s²")
print("  But got: -0.400 m/s²")
print("  This suggests NORMAL jerk limit is -4.0 m/s³, not -2.0")
