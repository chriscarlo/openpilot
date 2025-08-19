#!/usr/bin/env python3
"""
Test RTI speed recommendation with v_cruise_cluster (driver's set maximum).

Verifies:
1. No 90% reduction for close threats
2. Uses 20% of v_cruise_cluster when no posted limit exists
3. Falls back appropriately when v_cruise_cluster not available
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from dataclasses import dataclass
from sunnypilot.rtid.threat_detector import SpeedRecommendationEngine, ProcessedThreat


def test_speed_recommendations():
    """Test various speed recommendation scenarios."""
    engine = SpeedRecommendationEngine()
    
    print("Testing RTI Speed Recommendations with v_cruise_cluster")
    print("=" * 60)
    
    # Test 1: Close threat WITH posted speed limit - NO 90% reduction
    print("\nTest 1: Close threat (200m) with posted speed limit")
    threats = [
        ProcessedThreat(
            id="test1",
            type="police",
            latitude=37.123,
            longitude=-122.456,
            distance=200,  # Close threat
            direction="ahead",
            confidence=1.0,
            speed_limit_ms=25.0,  # Posted limit: 25 m/s (55 mph)
            on_same_road=True
        )
    ]
    
    current_speed = 30.0  # 30 m/s (67 mph)
    v_cruise = 33.0  # Driver set 33 m/s (74 mph)
    
    target, has_threat = engine.calculate_recommendation(
        threats, current_speed, (37.0, -122.0), v_cruise
    )
    
    print(f"  Current speed: {current_speed:.1f} m/s ({current_speed*2.237:.1f} mph)")
    print(f"  Driver set max (v_cruise_cluster): {v_cruise:.1f} m/s ({v_cruise*2.237:.1f} mph)")
    print(f"  Posted limit: 25.0 m/s (55.0 mph)")
    print(f"  Recommended speed: {target:.1f} m/s ({target*2.237:.1f} mph)")
    print(f"  Expected: 25.0 m/s (posted limit, NO 90% reduction)")
    assert abs(target - 25.0) < 0.01, f"Expected 25.0 m/s, got {target}"
    print("  ✓ PASSED - No 90% reduction applied")
    
    # Test 2: Close threat WITHOUT posted speed limit - use 20% of v_cruise_cluster
    print("\nTest 2: Close threat (200m) WITHOUT posted speed limit")
    threats = [
        ProcessedThreat(
            id="test2",
            type="hazard",
            latitude=37.123,
            longitude=-122.456,
            distance=200,
            direction="ahead",
            confidence=1.0,
            speed_limit_ms=0.0,  # No posted limit
            on_same_road=True
        )
    ]
    
    current_speed = 30.0  # Current: 30 m/s
    v_cruise = 35.0  # Driver set: 35 m/s (78 mph)
    
    target, has_threat = engine.calculate_recommendation(
        threats, current_speed, (37.0, -122.0), v_cruise
    )
    
    expected = v_cruise * 0.8  # 20% reduction from driver's set max
    print(f"  Current speed: {current_speed:.1f} m/s ({current_speed*2.237:.1f} mph)")
    print(f"  Driver set max (v_cruise_cluster): {v_cruise:.1f} m/s ({v_cruise*2.237:.1f} mph)")
    print(f"  No posted limit available")
    print(f"  Recommended speed: {target:.1f} m/s ({target*2.237:.1f} mph)")
    print(f"  Expected: {expected:.1f} m/s (80% of v_cruise_cluster)")
    assert abs(target - expected) < 0.01, f"Expected {expected} m/s, got {target}"
    print("  ✓ PASSED - Uses 20% reduction from v_cruise_cluster")
    
    # Test 3: No v_cruise_cluster available - fallback to default
    print("\nTest 3: No v_cruise_cluster available (cruise not set)")
    threats = [
        ProcessedThreat(
            id="test3",
            type="police",
            latitude=37.123,
            longitude=-122.456,
            distance=500,
            direction="ahead",
            confidence=1.0,
            speed_limit_ms=0.0,  # No posted limit
            on_same_road=True
        )
    ]
    
    current_speed = 30.0
    v_cruise = None  # Cruise not set
    
    target, has_threat = engine.calculate_recommendation(
        threats, current_speed, (37.0, -122.0), v_cruise
    )
    
    print(f"  Current speed: {current_speed:.1f} m/s ({current_speed*2.237:.1f} mph)")
    print(f"  Driver set max: Not available (cruise not set)")
    print(f"  No posted limit available")
    print(f"  Recommended speed: {target:.1f} m/s ({target*2.237:.1f} mph)")
    print(f"  Expected: 25.0 m/s (default fallback)")
    assert abs(target - 25.0) < 0.01, f"Expected 25.0 m/s fallback, got {target}"
    print("  ✓ PASSED - Falls back to default when no cruise set")
    
    # Test 4: Custom mode with v_cruise_cluster
    print("\nTest 4: Custom reduction mode with v_cruise_cluster")
    engine.speed_reduction_mode = "custom"
    engine.speed_reduction_ms = 4.47  # 10 mph reduction
    
    threats = [
        ProcessedThreat(
            id="test4",
            type="police",
            latitude=37.123,
            longitude=-122.456,
            distance=600,
            direction="ahead",
            confidence=1.0,
            speed_limit_ms=0.0,
            on_same_road=True
        )
    ]
    
    current_speed = 30.0
    v_cruise = 33.0  # Driver set: 33 m/s (74 mph)
    
    target, has_threat = engine.calculate_recommendation(
        threats, current_speed, (37.0, -122.0), v_cruise
    )
    
    expected = v_cruise - engine.speed_reduction_ms  # Fixed reduction from cruise
    expected = min(expected, current_speed)  # Safety cap at current speed
    
    print(f"  Current speed: {current_speed:.1f} m/s ({current_speed*2.237:.1f} mph)")
    print(f"  Driver set max (v_cruise_cluster): {v_cruise:.1f} m/s ({v_cruise*2.237:.1f} mph)")
    print(f"  Custom reduction: {engine.speed_reduction_ms:.1f} m/s ({engine.speed_reduction_ms*2.237:.1f} mph)")
    print(f"  Recommended speed: {target:.1f} m/s ({target*2.237:.1f} mph)")
    print(f"  Expected: {expected:.1f} m/s (v_cruise - reduction, capped at current)")
    assert abs(target - expected) < 0.01, f"Expected {expected} m/s, got {target}"
    print("  ✓ PASSED - Custom mode uses v_cruise_cluster correctly")
    
    # Test 5: Safety validation - never exceed current speed
    print("\nTest 5: Safety - never exceed current speed")
    engine.speed_reduction_mode = "posted"
    
    threats = [
        ProcessedThreat(
            id="test5",
            type="police",
            latitude=37.123,
            longitude=-122.456,
            distance=800,
            direction="ahead",
            confidence=1.0,
            speed_limit_ms=35.0,  # Posted: 35 m/s (78 mph)
            on_same_road=True
        )
    ]
    
    current_speed = 25.0  # Already below posted limit
    v_cruise = 40.0  # Driver set higher
    
    target, has_threat = engine.calculate_recommendation(
        threats, current_speed, (37.0, -122.0), v_cruise
    )
    
    print(f"  Current speed: {current_speed:.1f} m/s ({current_speed*2.237:.1f} mph)")
    print(f"  Posted limit: 35.0 m/s (78.0 mph)")
    print(f"  Recommended speed: {target:.1f} m/s ({target*2.237:.1f} mph)")
    print(f"  Expected: {current_speed:.1f} m/s (capped at current, no acceleration)")
    assert abs(target - current_speed) < 0.01, f"Expected {current_speed} m/s, got {target}"
    print("  ✓ PASSED - Never recommends acceleration")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nKey changes verified:")
    print("1. ✓ No 90% reduction for close threats")
    print("2. ✓ Uses 20% of v_cruise_cluster when no posted limit")
    print("3. ✓ Falls back appropriately when cruise not set")
    print("4. ✓ Custom mode uses v_cruise_cluster, not current speed")
    print("5. ✓ Safety: never exceeds current speed")


if __name__ == "__main__":
    test_speed_recommendations()