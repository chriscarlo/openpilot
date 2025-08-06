#!/usr/bin/env python3
"""Debug script to understand why 2 emergency tests are failing"""

import sys
import os
import time
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), '../shared'))
from vtsc_test_framework import VTSCTestBase

sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
from vision_turn_controller import EmergencyLevel, DECEL_LIMITS

class TimeSequencer:
    """Deterministic time.time() replacement for testing"""
    def __init__(self, start: float = 1000.0, step: float = 0.05):
        self._current = start
        self._step = step
        self._call_count = 0

    def __call__(self):
        now = self._current
        self._current += self._step
        self._call_count += 1
        return now

# Test 1: Debug intervention activation
print("=" * 80)
print("TEST 1: INTERVENTION ACTIVATION CRITERIA")
print("=" * 80)

test_obj = VTSCTestBase()
test_obj.setUp()
vtsc = test_obj.vtsc

time_seq = TimeSequencer()
with patch('time.time', time_seq):
    required_decel = -6.0  # Beyond CRITICAL threshold
    remaining_distance = 20.0  # Less than 25m threshold

    # Reset intervention state
    vtsc._critical_situation_time = 0.0
    vtsc._intervention_required = False

    print("Initial state:")
    print(f"  Required decel: {required_decel} m/s²")
    print(f"  Remaining distance: {remaining_distance} m")
    print(f"  CRITICAL threshold: {DECEL_LIMITS[EmergencyLevel.CRITICAL]} m/s²")
    print(f"  Critical decel check: {abs(required_decel)} > {abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]) * 1.05} = {abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]) * 1.05}")
    print(f"  Close distance check: {remaining_distance} < 25.0 = {remaining_distance < 25.0}")
    print()

    # Call multiple times and track what happens
    for i in range(10):
        result = vtsc._check_intervention_required(required_decel, remaining_distance)
        current_time = time.time()
        duration = current_time - vtsc._critical_situation_time if vtsc._critical_situation_time != 0 else 0
        print(f"  Call {i+1}: time={current_time:.3f}, critical_time={vtsc._critical_situation_time:.3f}, duration={duration:.3f}s, result={result}, intervention_flag={vtsc._intervention_required}")

        if result:
            print(f"  ✓ Intervention triggered after {i+1} calls ({duration:.3f}s)")
            break
    else:
        print("  ✗ Intervention never triggered after 10 calls")

# Test 2: Debug emergency reset
print()
print("=" * 80)
print("TEST 2: EMERGENCY RESET ON CURVE EXIT")
print("=" * 80)

test_obj2 = VTSCTestBase()
test_obj2.setUp()
vtsc2 = test_obj2.vtsc

time_seq2 = TimeSequencer()
with patch('time.time', time_seq2):
    dt = 0.05

    # First, escalate to emergency level
    print("Phase 1: Escalate to INTERVENTION level")
    vtsc2._current_decel = 0.0
    vtsc2._emergency_level = EmergencyLevel.NORMAL

    result1 = vtsc2._get_optimal_deceleration(-6.0, dt)
    print(f"  After escalation: emergency_level={vtsc2.emergency_level.name}, current_decel={vtsc2._current_decel:.3f}")

    # Now simulate curve exit - acceleration phase
    print("\nPhase 2: Simulate curve exit with positive acceleration")
    positive_accel = 1.0

    # Check what _get_optimal_deceleration does with positive values
    print(f"  Calling _get_optimal_deceleration({positive_accel}, {dt})")

    # Look at the actual implementation
    raw_decel = positive_accel
    print(f"  Input raw_decel: {raw_decel}")

    # The function expects negative values for deceleration
    # Positive values should just pass through or be handled differently
    result2 = vtsc2._get_optimal_deceleration(positive_accel, dt)

    print(f"  After curve exit: emergency_level={vtsc2.emergency_level.name}, current_decel={vtsc2._current_decel:.3f}")
    print(f"  Result from _get_optimal_deceleration: {result2}")

    # Check if the issue is that positive acceleration goes through the emergency system
    if vtsc2._current_decel != 0.0:
        print(f"  ✗ Current decel not reset to 0.0, got {vtsc2._current_decel}")
    else:
        print("  ✓ Current decel correctly reset to 0.0")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
