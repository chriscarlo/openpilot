#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from vtsc_test_framework import VTSCTestBase

# Create test instance
test = VTSCTestBase()
test.setUp()

print("=== DEBUGGING RESET BEHAVIOR ===")

print("Initial state:")
print(f"  Emergency level: {test.vtsc._emergency_level.name}")
print(f"  Current decel: {test.vtsc._current_decel:.3f}")
print(f"  Time at level: {test.vtsc._time_at_current_level:.3f}")

# Try to escalate to INTERVENTION
print("\nEscalating to INTERVENTION with -6.0 m/s²:")
result_decel = test.vtsc._get_optimal_deceleration(-6.0, 0.1)

print("After escalation:")
print(f"  Emergency level: {test.vtsc._emergency_level.name}")
print(f"  Current decel: {test.vtsc._current_decel:.3f}")
print(f"  Time at level: {test.vtsc._time_at_current_level:.3f}")
print(f"  Result decel: {result_decel:.3f}")

# Try another step to accumulate time
print("\nAnother step at INTERVENTION:")
result_decel2 = test.vtsc._get_optimal_deceleration(-6.0, 0.1)
print("After second step:")
print(f"  Emergency level: {test.vtsc._emergency_level.name}")
print(f"  Current decel: {test.vtsc._current_decel:.3f}")
print(f"  Time at level: {test.vtsc._time_at_current_level:.3f}")
print(f"  Result decel: {result_decel2:.3f}")

print("\nBefore reset - checking non-zero state:")
print(f"  Has non-zero current_decel: {test.vtsc._current_decel != 0.0}")
print(f"  Has non-zero time_at_level: {test.vtsc._time_at_current_level != 0.0}")

# Reset
print("\nResetting VTSC...")
test.vtsc.reset()

print("After reset:")
print(f"  Emergency level: {test.vtsc._emergency_level.name}")
print(f"  Current decel: {test.vtsc._current_decel:.3f}")
print(f"  Time at level: {test.vtsc._time_at_current_level:.3f}")
