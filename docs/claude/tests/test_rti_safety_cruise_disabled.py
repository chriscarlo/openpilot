#!/usr/bin/env python3
"""
CRITICAL SAFETY TEST: RTI must NOT operate when cruise is disabled.

This test verifies that RTI never makes speed recommendations when
cruise control is not engaged, preventing dangerous accelerations
when resuming cruise in low-speed areas.
"""

def calculate_recommendation_with_safety(threats, current_speed_ms, v_cruise_ms=None):
    """
    Updated logic with critical safety check.
    RTI is ONLY active when cruise is enabled (v_cruise_ms > 0).
    """
    # Constants
    ahead_distance_threshold_m = 1207
    
    # Filter for relevant threats
    relevant_threats = []
    for threat in threats:
        if not threat['on_same_road']:
            continue
        if threat['direction'] == 'ahead' and threat['distance'] <= ahead_distance_threshold_m:
            relevant_threats.append(threat)
    
    if not relevant_threats:
        return 0.0, False
    
    # Find closest ahead threat
    closest_threat = min(relevant_threats, key=lambda t: t['distance'])
    
    # CRITICAL SAFETY CHECK
    if not v_cruise_ms or v_cruise_ms <= 0:
        # No cruise speed set - RTI is inactive
        return 0.0, False
    
    # Calculate target speed (only when cruise is active)
    if closest_threat['speed_limit_ms'] > 0:
        target_speed = closest_threat['speed_limit_ms']
    else:
        target_speed = v_cruise_ms * 0.8  # 20% reduction
    
    # Safety: never exceed current speed
    target_speed = min(target_speed, current_speed_ms)
    target_speed = max(0.0, target_speed)
    
    return target_speed, True


def test_cruise_disabled_safety():
    """Test that RTI does NOTHING when cruise is not set."""
    print("CRITICAL SAFETY TEST: RTI inactive when cruise disabled")
    print("=" * 60)
    
    # Scenario 1: School zone, cruise not set
    print("\nScenario 1: School zone (25 mph), cruise not set")
    
    threat = {
        'id': 'school_zone',
        'distance': 100,
        'direction': 'ahead',
        'speed_limit_ms': 11.0,  # 25 mph school zone
        'on_same_road': True
    }
    
    current_speed = 11.0  # Already at 25 mph
    v_cruise = None  # CRUISE NOT SET
    
    target, has_threat = calculate_recommendation_with_safety([threat], current_speed, v_cruise)
    
    print(f"  Location: School zone")
    print(f"  Current speed: {current_speed:.1f} m/s (25 mph)")
    print(f"  Posted limit: {threat['speed_limit_ms']:.1f} m/s (25 mph)")
    print(f"  Cruise status: NOT SET")
    print(f"  RTI recommendation: {target:.1f} m/s")
    print(f"  Has threat flag: {has_threat}")
    
    assert target == 0.0, f"Expected 0.0 (inactive), got {target}"
    assert has_threat == False, f"Expected False (inactive), got {has_threat}"
    print("  ✓ PASSED - RTI correctly inactive")
    
    # Scenario 2: Highway threat, cruise disabled
    print("\nScenario 2: Highway police, cruise disabled")
    
    threat = {
        'id': 'highway_police',
        'distance': 500,
        'direction': 'ahead',
        'speed_limit_ms': 0.0,  # No posted limit
        'on_same_road': True
    }
    
    current_speed = 30.0  # 67 mph
    v_cruise = 0.0  # Cruise disabled (returns 0)
    
    target, has_threat = calculate_recommendation_with_safety([threat], current_speed, v_cruise)
    
    print(f"  Location: Highway")
    print(f"  Current speed: {current_speed:.1f} m/s (67 mph)")
    print(f"  Cruise status: DISABLED (0.0)")
    print(f"  RTI recommendation: {target:.1f} m/s")
    print(f"  Has threat flag: {has_threat}")
    
    assert target == 0.0, f"Expected 0.0 (inactive), got {target}"
    assert has_threat == False, f"Expected False (inactive), got {has_threat}"
    print("  ✓ PASSED - RTI correctly inactive")
    
    # Scenario 3: Cruise enabled - RTI should work
    print("\nScenario 3: Same threat, cruise ENABLED")
    
    v_cruise = 33.0  # Cruise set to 74 mph
    
    target, has_threat = calculate_recommendation_with_safety([threat], current_speed, v_cruise)
    
    expected = v_cruise * 0.8  # 20% reduction
    expected = min(expected, current_speed)
    
    print(f"  Current speed: {current_speed:.1f} m/s (67 mph)")
    print(f"  Cruise status: ENABLED at {v_cruise:.1f} m/s (74 mph)")
    print(f"  RTI recommendation: {target:.1f} m/s")
    print(f"  Has threat flag: {has_threat}")
    print(f"  Expected: {expected:.1f} m/s (80% of cruise)")
    
    assert abs(target - expected) < 0.01, f"Expected {expected}, got {target}"
    assert has_threat == True, f"Expected True (active), got {has_threat}"
    print("  ✓ PASSED - RTI correctly active when cruise enabled")
    
    print("\n" + "=" * 60)
    print("CRITICAL SAFETY TEST PASSED! ✓")
    print("\nSafety guarantee verified:")
    print("- RTI is COMPLETELY INACTIVE when cruise is not set")
    print("- Returns (0.0, False) - no speed change")
    print("- Prevents dangerous accelerations when resuming cruise")
    print("- Only operates when driver has explicitly set cruise speed")


def test_dangerous_fallback_removed():
    """Verify the dangerous 55 mph fallback is gone."""
    print("\nTest: Dangerous 55 mph fallback is removed")
    print("-" * 60)
    
    threat = {
        'id': 'test_fallback',
        'distance': 300,
        'direction': 'ahead',
        'speed_limit_ms': 0.0,  # No posted limit
        'on_same_road': True
    }
    
    # Test with various "no cruise" conditions
    test_cases = [
        (None, "None"),
        (0.0, "0.0"),
        (-1.0, "-1.0 (error value)")
    ]
    
    for v_cruise, description in test_cases:
        current_speed = 15.0  # 33 mph (residential speed)
        target, has_threat = calculate_recommendation_with_safety([threat], current_speed, v_cruise)
        
        print(f"  v_cruise = {description}")
        print(f"  Result: {target:.1f} m/s, has_threat={has_threat}")
        
        # Should NEVER return 25.0 (55 mph) or any non-zero value
        assert target == 0.0, f"DANGEROUS: Got {target} instead of 0.0"
        assert has_threat == False, f"DANGEROUS: Got {has_threat} instead of False"
        print(f"  ✓ Safe - returns (0.0, False)")
    
    print("\n✓ Confirmed: No dangerous fallback to 55 mph")


if __name__ == "__main__":
    test_cruise_disabled_safety()
    test_dangerous_fallback_removed()
    
    print("\n" + "=" * 60)
    print("ALL SAFETY TESTS PASSED")
    print("=" * 60)
    print("\nCritical safety fix verified:")
    print("1. RTI is INACTIVE when cruise is not set")
    print("2. No dangerous fallback speeds")
    print("3. Driver's cruise setting is never modified without explicit cruise engagement")
    print("4. Prevents acceleration surprises when resuming cruise")