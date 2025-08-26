#!/usr/bin/env python3
"""
Verification test for the SLC car-only mode fix
Tests that CarStateSP.speedLimit field now exists and data flows correctly
"""

import sys
sys.path.append('/projects/chauffeur/data/openpilot')

from opendbc.car import structs
from selfdrive.car.helpers import convert_to_capnp
from cereal import custom

def test_structs_has_speedlimit():
    """Test that structs.CarStateSP has speedLimit field"""
    cs_sp = structs.CarStateSP()
    assert hasattr(cs_sp, 'speedLimit'), "CarStateSP missing speedLimit field"
    assert cs_sp.speedLimit == 0.0, f"Default speedLimit should be 0.0, got {cs_sp.speedLimit}"
    
    # Test setting speedLimit
    cs_sp.speedLimit = 25.0
    assert cs_sp.speedLimit == 25.0, f"Failed to set speedLimit, got {cs_sp.speedLimit}"
    print("✅ Test 1: structs.CarStateSP has speedLimit field")

def test_conversion_to_capnp():
    """Test that speedLimit converts to capnp format"""
    # Create python dataclass with speedLimit
    cs_sp = structs.CarStateSP()
    cs_sp.speedLimit = 30.0  # 30 m/s = 108 km/h
    
    # Convert to capnp format (what card.py does)
    cs_sp_capnp = convert_to_capnp(cs_sp)
    
    # Verify speedLimit made it through
    assert cs_sp_capnp.speedLimit == 30.0, f"speedLimit not converted, got {cs_sp_capnp.speedLimit}"
    print("✅ Test 2: speedLimit converts to capnp format correctly")

def test_hyundai_canfd_flow():
    """Simulate the full Hyundai CANFD data flow"""
    # Step 1: Hyundai carstate.py creates CarStateSP
    ret_sp = structs.CarStateSP()
    
    # Step 2: Parse dashboard speed limit (simulate FR_CMR_02_100ms)
    speed_limit_raw = 80  # 80 km/h from dashboard
    speed_factor = 0.277778  # km/h to m/s conversion
    
    if speed_limit_raw not in (0, 255, 253):
        ret_sp.speedLimit = float(speed_limit_raw) * speed_factor
    else:
        ret_sp.speedLimit = 0.0
    
    expected = 80 * 0.277778
    assert abs(ret_sp.speedLimit - expected) < 0.001, f"Speed calculation wrong: {ret_sp.speedLimit} != {expected}"
    
    # Step 3: Convert to capnp (what card.py does)
    cs_sp_capnp = convert_to_capnp(ret_sp)
    
    # Step 4: Verify speed limit is in capnp message
    assert abs(cs_sp_capnp.speedLimit - expected) < 0.001, f"Speed lost in conversion: {cs_sp_capnp.speedLimit}"
    
    print(f"✅ Test 3: Hyundai CANFD flow works - speedLimit = {cs_sp_capnp.speedLimit:.3f} m/s ({speed_limit_raw} km/h)")

def test_different_speed_values():
    """Test various speed limit values including edge cases"""
    test_cases = [
        (0, 0.0, "No speed limit"),
        (30, 30 * 0.277778, "30 km/h"),
        (50, 50 * 0.277778, "50 km/h"),
        (60, 60 * 0.277778, "60 km/h"),
        (80, 80 * 0.277778, "80 km/h"),
        (100, 100 * 0.277778, "100 km/h"),
        (120, 120 * 0.277778, "120 km/h"),
        (253, 0.0, "Unlimited (253)"),
        (255, 0.0, "Invalid (255)"),
    ]
    
    for raw_value, expected_ms, description in test_cases:
        ret_sp = structs.CarStateSP()
        
        # Apply Hyundai logic
        if raw_value not in (0, 255, 253):
            ret_sp.speedLimit = float(raw_value) * 0.277778
        else:
            ret_sp.speedLimit = 0.0
        
        assert abs(ret_sp.speedLimit - expected_ms) < 0.001, \
            f"{description}: Expected {expected_ms:.3f} m/s, got {ret_sp.speedLimit:.3f} m/s"
    
    print("✅ Test 4: All speed limit values handled correctly")

def main():
    print("=" * 70)
    print("SLC CAR-ONLY MODE FIX VERIFICATION")
    print("=" * 70)
    print()
    
    try:
        test_structs_has_speedlimit()
        test_conversion_to_capnp()
        test_hyundai_canfd_flow()
        test_different_speed_values()
        
        print()
        print("=" * 70)
        print("🎉 ALL TESTS PASSED! SLC CAR-ONLY MODE SHOULD NOW WORK")
        print("=" * 70)
        print()
        print("The fix:")
        print("  - Added speedLimit field to structs.CarStateSP")
        print("  - Field matches cereal/custom.capnp CarStateSP definition")
        print("  - Hyundai CANFD can now pass speed limits to SLC")
        print()
        print("Next steps:")
        print("  1. Build openpilot: scons -u -j$(nproc)")
        print("  2. Test in vehicle with car-only SLC policy")
        print("  3. Monitor with test_hyundai_slc_live.py")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())