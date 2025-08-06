#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../shared'))

from vtsc_test_framework import VTSCTestBase

# Create test instance
test = VTSCTestBase()
test.setUp()

print("=== DEBUGGING POSITIVE DECELERATION (ACCELERATION) ===")

# Test positive deceleration requests
positive_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

for pos_decel in positive_values:
    test.setUp()  # Reset for each test
    result_decel = test.vtsc._get_optimal_deceleration(pos_decel, 0.1)
    level = test.vtsc._emergency_level
    print(f"Request: +{pos_decel:3.1f} -> Level: {level.name:12} -> Result: {result_decel:6.3f}")

print("\n=== CHECKING EMERGENCY LEVEL DETERMINATION ===")
import time
for pos_decel in positive_values:
    level = test.vtsc._determine_emergency_level(pos_decel, time.time())
    print(f"Determine level for +{pos_decel:3.1f}: {level.name}")
