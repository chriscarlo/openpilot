#!/usr/bin/env python3
"""
Test script for LiveSteerRatio functionality
Tests:
1. KIA EV6 default steer ratio (13.43)
2. LiveSteerRatio parameter handling in paramsd
3. Bounds calculation with live values
"""

import os
import sys

# Add openpilot to path (we're in docs/claude/tests/live_steer_ratio)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from common.params import Params
from opendbc.car.hyundai.values import CAR
from selfdrive.locationd.paramsd import retrieve_initial_vehicle_params


def test_ev6_default_steer_ratio():
    """Test that KIA EV6 has the correct default steer ratio"""
    print("\n=== Testing KIA EV6 Default Steer Ratio ===")

    # Get the EV6 car specs from config

    # KIA_EV6 is defined as a HyundaiCanFDPlatformConfig
    if hasattr(CAR.KIA_EV6, 'config'):
        ev6_config = CAR.KIA_EV6.config
    else:
        # It's the enum value, we need to get the actual config
        # Let's directly import and check the KIA_EV6 definition
        ev6_config = CAR.KIA_EV6

    # Get specs from the config
    ev6_specs = ev6_config.specs
    print(f"KIA EV6 steerRatio: {ev6_specs.steerRatio}")

    # Verify it's 13.43
    assert abs(ev6_specs.steerRatio - 13.43) < 0.001, f"Expected 13.43, got {ev6_specs.steerRatio}"
    print("✓ KIA EV6 steer ratio is correctly set to 13.43")

    return True


def test_live_steer_ratio_parameter():
    """Test LiveSteerRatio parameter handling"""
    print("\n=== Testing LiveSteerRatio Parameter ===")

    params = Params()

    # Test 1: No LiveSteerRatio set (should use car default)
    print("\nTest 1: No LiveSteerRatio parameter")
    try:
        params.remove("LiveSteerRatio")
    except:
        pass  # OK if it doesn't exist

    # Create a mock CarParams for EV6
    class MockCarParams:
        carFingerprint = "KIA_EV6"
        steerRatio = 13.43
        mass = 2055
        wheelbase = 2.9
        centerToFront = 1.45
        tireStiffnessFront = 200000
        tireStiffnessRear = 200000
        rotationalInertia = 3000

    CP = MockCarParams()

    steer_ratio, stiffness_factor, angle_offset_deg, p_initial, base_steer_ratio = retrieve_initial_vehicle_params(
        params, CP, replay=True, debug=True
    )

    print(f"Retrieved steer_ratio: {steer_ratio}")
    print(f"Retrieved base_steer_ratio: {base_steer_ratio}")
    assert abs(steer_ratio - 13.43) < 0.001, f"Expected 13.43, got {steer_ratio}"
    assert abs(base_steer_ratio - 13.43) < 0.001, f"Expected base 13.43, got {base_steer_ratio}"
    print("✓ Correctly uses car default when LiveSteerRatio not set")

    # Test 2: LiveSteerRatio set to 0 (should use car default)
    print("\nTest 2: LiveSteerRatio set to 0")
    params.put("LiveSteerRatio", "0.0")

    steer_ratio, stiffness_factor, angle_offset_deg, p_initial, base_steer_ratio = retrieve_initial_vehicle_params(
        params, CP, replay=True, debug=True
    )

    print(f"Retrieved steer_ratio: {steer_ratio}")
    print(f"Retrieved base_steer_ratio: {base_steer_ratio}")
    assert abs(steer_ratio - 13.43) < 0.001, f"Expected 13.43, got {steer_ratio}"
    assert abs(base_steer_ratio - 13.43) < 0.001, f"Expected base 13.43, got {base_steer_ratio}"
    print("✓ Correctly uses car default when LiveSteerRatio is 0")

    # Test 3: LiveSteerRatio set to custom value
    print("\nTest 3: LiveSteerRatio set to 15.0")
    params.put("LiveSteerRatio", "15.0")

    steer_ratio, stiffness_factor, angle_offset_deg, p_initial, base_steer_ratio = retrieve_initial_vehicle_params(
        params, CP, replay=True, debug=True
    )

    print(f"Retrieved steer_ratio: {steer_ratio}")
    print(f"Retrieved base_steer_ratio: {base_steer_ratio}")
    assert abs(steer_ratio - 15.0) < 0.001, f"Expected 15.0, got {steer_ratio}"
    assert abs(base_steer_ratio - 15.0) < 0.001, f"Expected base 15.0, got {base_steer_ratio}"
    print("✓ Correctly uses LiveSteerRatio when set to custom value")

    # Clean up
    try:
        params.remove("LiveSteerRatio")
    except:
        pass  # OK if it doesn't exist

    return True


def test_bounds_calculation():
    """Test that bounds are calculated correctly with live steer ratio"""
    print("\n=== Testing Bounds Calculation ===")

    from selfdrive.locationd.paramsd import VehicleParamsLearner

    # Create mock CarParams
    class MockCarParams:
        carFingerprint = "KIA_EV6"
        steerRatio = 13.43
        mass = 2055
        wheelbase = 2.9
        centerToFront = 1.45
        tireStiffnessFront = 200000
        tireStiffnessRear = 200000
        rotationalInertia = 3000

    CP = MockCarParams()

    # Test with default steer ratio
    print("\nTest with default steer ratio (13.43)")
    learner = VehicleParamsLearner(CP, 13.43, 1.0, 0.0, None, 13.43)
    print(f"Min SR: {learner.min_sr:.2f}, Max SR: {learner.max_sr:.2f}")
    expected_min = 0.5 * 13.43
    expected_max = 2.0 * 13.43
    assert abs(learner.min_sr - expected_min) < 0.01, f"Expected min {expected_min}, got {learner.min_sr}"
    assert abs(learner.max_sr - expected_max) < 0.01, f"Expected max {expected_max}, got {learner.max_sr}"
    print(f"✓ Bounds correctly calculated: {expected_min:.2f} - {expected_max:.2f}")

    # Test with custom live steer ratio
    print("\nTest with custom live steer ratio (15.0)")
    learner = VehicleParamsLearner(CP, 15.0, 1.0, 0.0, None, 15.0)
    print(f"Min SR: {learner.min_sr:.2f}, Max SR: {learner.max_sr:.2f}")
    expected_min = 0.5 * 15.0
    expected_max = 2.0 * 15.0
    assert abs(learner.min_sr - expected_min) < 0.01, f"Expected min {expected_min}, got {learner.min_sr}"
    assert abs(learner.max_sr - expected_max) < 0.01, f"Expected max {expected_max}, got {learner.max_sr}"
    print(f"✓ Bounds correctly calculated with live value: {expected_min:.2f} - {expected_max:.2f}")

    return True


def test_gui_parameter_storage():
    """Test that GUI correctly stores parameter values"""
    print("\n=== Testing GUI Parameter Storage ===")

    params = Params()

    # Test storing different values
    test_values = ["0.0", "13.43", "15.5", "20.0"]

    for value in test_values:
        params.put("LiveSteerRatio", value)
        retrieved = params.get("LiveSteerRatio")
        assert retrieved is not None, f"Failed to retrieve LiveSteerRatio after setting to {value}"
        retrieved_str = retrieved.decode('utf-8')
        assert retrieved_str == value, f"Expected {value}, got {retrieved_str}"
        print(f"✓ Successfully stored and retrieved: {value}")

    # Clean up
    try:
        params.remove("LiveSteerRatio")
    except:
        pass  # OK if it doesn't exist

    return True


def main():
    """Run all tests"""
    print("Starting LiveSteerRatio tests...")

    tests = [
        ("EV6 Default Steer Ratio", test_ev6_default_steer_ratio),
        ("Live Steer Ratio Parameter", test_live_steer_ratio_parameter),
        ("Bounds Calculation", test_bounds_calculation),
        ("GUI Parameter Storage", test_gui_parameter_storage),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n✓ All tests passed! LiveSteerRatio functionality is working correctly.")
        print("\nHow it works:")
        print("1. KIA EV6 now has default steer ratio of 13.43")
        print("2. Users can override with LiveSteerRatio parameter in GUI")
        print("3. Setting to 0 uses vehicle default (13.43 for EV6)")
        print("4. Parameter bounds adjust based on live value")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
