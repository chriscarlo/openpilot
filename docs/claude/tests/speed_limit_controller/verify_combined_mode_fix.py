#!/usr/bin/env python3
"""Verification script for combined mode speed limit fix"""

import sys
import time
import numpy as np
from unittest.mock import MagicMock

# Add sunnypilot to path
sys.path.insert(0, '/projects/chauffeur/data/openpilot')

from sunnypilot.selfdrive.controls.lib.speed_limit_controller.speed_limit_resolver import SpeedLimitResolver
from sunnypilot.selfdrive.controls.lib.speed_limit_controller.common import Source, Policy


def create_realistic_sm(car_limit_kmh, map_limit_kmh):
    """Create a realistic mock SubMaster with speed limits in km/h"""
    sm_mock = MagicMock()
    
    # Convert km/h to m/s 
    car_state_sp = MagicMock()
    car_state_sp.speedLimit = car_limit_kmh / 3.6 if car_limit_kmh > 0 else 0
    
    live_map_data = MagicMock()
    live_map_data.speedLimit = map_limit_kmh / 3.6 if map_limit_kmh > 0 else 0
    live_map_data.speedLimitValid = map_limit_kmh > 0
    live_map_data.speedLimitAhead = 0.
    live_map_data.speedLimitAheadValid = False
    live_map_data.speedLimitAheadDistance = 0.
    
    gps_data = MagicMock()
    gps_data.unixTimestampMillis = time.time() * 1000
    
    sm_mock.__getitem__.side_effect = lambda key: {
        'carStateSP': car_state_sp,
        'liveMapDataSP': live_map_data,
        'gpsLocation': gps_data,
    }[key]
    
    return sm_mock


def test_realistic_scenarios():
    """Test realistic speed limit scenarios"""
    resolver = SpeedLimitResolver(Policy.combined)
    
    print("REALISTIC COMBINED MODE VERIFICATION")
    print("=" * 60)
    print("Requirements:")
    print("1. Both same: use map value")
    print("2. Both different: use HIGHER value")
    print("3. Only one available: use what we have")
    print("4. None available: use none")
    print("=" * 60)
    
    test_cases = [
        # (car_kmh, map_kmh, expected_kmh, expected_source, description)
        (80, 80, 80, Source.map_data, "Both 80 km/h - should prefer map"),
        (100, 80, 100, Source.car_state, "Car 100, Map 80 - should use higher (car)"),
        (60, 80, 80, Source.map_data, "Car 60, Map 80 - should use higher (map)"),
        (70, 0, 70, Source.car_state, "Only car available (70 km/h)"),
        (0, 90, 90, Source.map_data, "Only map available (90 km/h)"),
        (0, 0, 0, Source.none, "No data available"),
        (50, 50, 50, Source.map_data, "Both 50 km/h - should prefer map"),
        (120, 100, 120, Source.car_state, "Car 120, Map 100 - should use higher (car)"),
    ]
    
    passed = 0
    failed = 0
    
    for car_kmh, map_kmh, expected_kmh, expected_source, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Input: Car={car_kmh} km/h, Map={map_kmh} km/h")
        
        sm = create_realistic_sm(car_kmh, map_kmh)
        speed_limit_ms, _, source = resolver.resolve(20.0, 0, sm)
        
        # Convert result back to km/h for display
        result_kmh = speed_limit_ms * 3.6 if speed_limit_ms > 0 else 0
        
        print(f"  Result: {result_kmh:.1f} km/h from {source.name}")
        print(f"  Expected: {expected_kmh} km/h from {expected_source.name}")
        
        # Check if result matches expected (with small tolerance for float comparison)
        speed_matches = abs(result_kmh - expected_kmh) < 0.1
        source_matches = source == expected_source
        
        if speed_matches and source_matches:
            print("  ✓ PASS")
            passed += 1
        else:
            print("  ✗ FAIL")
            failed += 1
            if not speed_matches:
                print(f"    Speed mismatch: got {result_kmh:.1f}, expected {expected_kmh}")
            if not source_matches:
                print(f"    Source mismatch: got {source.name}, expected {expected_source.name}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
    
    if failed == 0:
        print("✓ ALL TESTS PASSED - Combined mode fix is working correctly!")
        return True
    else:
        print("✗ SOME TESTS FAILED - Fix needs adjustment")
        return False


def test_edge_cases():
    """Test edge cases and floating point precision"""
    resolver = SpeedLimitResolver(Policy.combined)
    
    print("\n\nEDGE CASE VERIFICATION")
    print("=" * 60)
    
    # Test very close but not exactly equal values
    print("\nTest: Nearly equal values (should still use higher)")
    sm = create_realistic_sm(80.01, 80.00)  # Very slight difference
    speed_limit_ms, _, source = resolver.resolve(20.0, 0, sm)
    result_kmh = speed_limit_ms * 3.6
    
    # With epsilon of 0.01 m/s in the code, 80.01 vs 80.00 km/h 
    # is 22.225 vs 22.222 m/s, difference is 0.003 m/s < 0.01
    # So should be treated as equal and prefer map_data
    print(f"  Car=80.01 km/h, Map=80.00 km/h")
    print(f"  Result: {result_kmh:.2f} km/h from {source.name}")
    print(f"  Expected: ~80 km/h from map_data (treated as equal)")
    
    if source == Source.map_data:
        print("  ✓ PASS - Correctly treated as equal")
    else:
        print("  ✗ FAIL - Should treat as equal with epsilon")
    
    # Test clearly different values
    print("\nTest: Clearly different values")
    sm = create_realistic_sm(80.5, 80.0)  # Clear difference
    speed_limit_ms, _, source = resolver.resolve(20.0, 0, sm)
    result_kmh = speed_limit_ms * 3.6
    
    print(f"  Car=80.5 km/h, Map=80.0 km/h")
    print(f"  Result: {result_kmh:.2f} km/h from {source.name}")
    print(f"  Expected: 80.5 km/h from car_state (higher value)")
    
    if source == Source.car_state and abs(result_kmh - 80.5) < 0.1:
        print("  ✓ PASS - Correctly selected higher value")
    else:
        print("  ✗ FAIL - Should select higher value")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    success = test_realistic_scenarios()
    test_edge_cases()
    
    if success:
        print("\n✅ VERIFICATION COMPLETE - Fix is working as specified!")
        sys.exit(0)
    else:
        print("\n❌ VERIFICATION FAILED - Fix needs adjustment")
        sys.exit(1)