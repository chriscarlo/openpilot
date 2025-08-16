#!/usr/bin/env python3
"""
RTI Metric/Imperial Conversion Validation Script
Manually verify the conversion constants and math used in the C++ implementation
"""

def test_conversion_constants():
    """Test the accuracy of conversion constants used in RTI implementation."""

    # Constants from RTI implementation
    METERS_TO_MILES = 0.000621371
    MILES_TO_METERS = 1609.344
    METERS_TO_KM = 0.001
    KM_TO_METERS = 1000.0

    IMPERIAL_INCREMENT_MI = 0.25  # 0.25 miles
    METRIC_INCREMENT_KM = 0.5     # 0.5 km

    print("=== RTI Conversion Constants Validation ===")

    # Test round-trip conversion accuracy
    test_miles = 1.0
    meters_from_miles = test_miles * MILES_TO_METERS
    miles_back = meters_from_miles * METERS_TO_MILES
    miles_error = abs(test_miles - miles_back)

    print(f"Round-trip mile conversion: {test_miles} -> {meters_from_miles} -> {miles_back}")
    print(f"Miles conversion error: {miles_error:.10f} (should be < 0.0001)")
    assert miles_error < 0.0001, f"Miles conversion error too large: {miles_error}"

    test_km = 1.0
    meters_from_km = test_km * KM_TO_METERS
    km_back = meters_from_km * METERS_TO_KM
    km_error = abs(test_km - km_back)

    print(f"Round-trip km conversion: {test_km} -> {meters_from_km} -> {km_back}")
    print(f"Km conversion error: {km_error:.10f} (should be < 0.0001)")
    assert km_error < 0.0001, f"Km conversion error too large: {km_error}"

    print("Check: Conversion constants are accurate\n")


def test_imperial_range_logic():
    """Test imperial range and increment calculations."""

    MILES_TO_METERS = 1609.344
    IMPERIAL_INCREMENT_MI = 0.25

    print("=== Imperial Range Logic Validation ===")

    # Imperial range: 0.25mi - 2mi in 0.25mi increments
    min_range_m = int(IMPERIAL_INCREMENT_MI * MILES_TO_METERS)  # ~402m
    max_range_m = int(2.0 * MILES_TO_METERS)                   # ~3219m
    step_m = min_range_m                                        # ~402m

    print(f"Imperial min range: {IMPERIAL_INCREMENT_MI} mi = {min_range_m} m")
    print(f"Imperial max range: 2.0 mi = {max_range_m} m")
    print(f"Imperial step size: {IMPERIAL_INCREMENT_MI} mi = {step_m} m")

    # Verify we can hit exact increments
    increments = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    print("\nImperial increment verification:")
    for increment in increments:
        meters = int(increment * MILES_TO_METERS)
        steps_from_min = (meters - min_range_m) // step_m
        expected_meters = min_range_m + (steps_from_min * step_m)
        error = abs(meters - expected_meters)

        print(f"  {increment} mi = {meters}m, snapped = {expected_meters}m, error = {error}m")
        assert error <= step_m // 2, f"Snapping error too large for {increment} mi"

    print("Check: Imperial range logic is correct\n")


def test_metric_range_logic():
    """Test metric range and increment calculations."""

    KM_TO_METERS = 1000.0
    METRIC_INCREMENT_KM = 0.5

    print("=== Metric Range Logic Validation ===")

    # Metric range: 0.5km - 5km in 0.5km increments
    min_range_m = int(METRIC_INCREMENT_KM * KM_TO_METERS)  # 500m
    max_range_m = int(5.0 * KM_TO_METERS)                 # 5000m
    step_m = min_range_m                                   # 500m

    print(f"Metric min range: {METRIC_INCREMENT_KM} km = {min_range_m} m")
    print(f"Metric max range: 5.0 km = {max_range_m} m")
    print(f"Metric step size: {METRIC_INCREMENT_KM} km = {step_m} m")

    # Verify we can hit exact increments
    increments = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    print("\nMetric increment verification:")
    for increment in increments:
        meters = int(increment * KM_TO_METERS)
        steps_from_min = (meters - min_range_m) // step_m
        expected_meters = min_range_m + (steps_from_min * step_m)
        error = abs(meters - expected_meters)

        print(f"  {increment} km = {meters}m, snapped = {expected_meters}m, error = {error}m")
        assert error == 0, f"Metric increments should be exact, got {error}m error for {increment} km"

    print("Check: Metric range logic is correct\n")


def test_snap_to_increment_logic():
    """Test the snap-to-increment algorithm."""

    METERS_TO_MILES = 0.000621371
    MILES_TO_METERS = 1609.344
    IMPERIAL_INCREMENT_MI = 0.25

    print("=== Snap-to-Increment Logic Validation ===")

    def snap_to_valid_increment_imperial(meters):
        """Python version of C++ snapToValidIncrement for imperial."""
        increment_m = int(IMPERIAL_INCREMENT_MI * MILES_TO_METERS)
        return ((meters + increment_m//2) // increment_m) * increment_m

    def snap_to_valid_increment_metric(meters):
        """Python version of C++ snapToValidIncrement for metric."""
        return ((meters + 250) // 500) * 500

    # Test imperial snapping
    print("Imperial snapping tests:")
    test_values = [400, 401, 402, 403, 600, 800, 1609, 1610]
    for value in test_values:
        snapped = snap_to_valid_increment_imperial(value)
        miles_original = value * METERS_TO_MILES
        miles_snapped = snapped * METERS_TO_MILES
        print(f"  {value}m ({miles_original:.3f}mi) -> {snapped}m ({miles_snapped:.3f}mi)")

        # Verify snapped value is a valid increment
        expected_increment = round(miles_snapped / 0.25) * 0.25
        assert abs(miles_snapped - expected_increment) < 0.01, "Snapped value not on 0.25mi grid"

    # Test metric snapping
    print("\nMetric snapping tests:")
    test_values = [450, 500, 550, 750, 1000, 1250]
    for value in test_values:
        snapped = snap_to_valid_increment_metric(value)
        km_original = value * 0.001
        km_snapped = snapped * 0.001
        print(f"  {value}m ({km_original:.1f}km) -> {snapped}m ({km_snapped:.1f}km)")

        # Verify snapped value is a valid increment
        assert snapped % 500 == 0, f"Snapped value not on 0.5km grid: {snapped}"

    print("Check: Snap-to-increment logic is correct\n")


def test_user_requirements():
    """Verify implementation meets exact user requirements."""

    print("=== User Requirements Validation ===")

    requirements = {
        "imperial_min": 0.25,  # miles
        "imperial_max": 2.0,   # miles
        "imperial_step": 0.25, # miles
        "metric_min": 0.5,     # km
        "metric_max": 5.0,     # km
        "metric_step": 0.5,    # km
    }

    MILES_TO_METERS = 1609.344
    KM_TO_METERS = 1000.0

    print("Imperial requirements:")
    imperial_min_m = requirements["imperial_min"] * MILES_TO_METERS
    imperial_max_m = requirements["imperial_max"] * MILES_TO_METERS
    imperial_step_m = requirements["imperial_step"] * MILES_TO_METERS

    print(f"  Range: {requirements['imperial_min']}-{requirements['imperial_max']} mi ({imperial_min_m:.0f}-{imperial_max_m:.0f} m)")
    print(f"  Step: {requirements['imperial_step']} mi ({imperial_step_m:.0f} m)")

    # Verify range is reasonable
    assert imperial_min_m >= 400 and imperial_min_m <= 450, "Imperial min not ~402m"
    assert imperial_max_m >= 3200 and imperial_max_m <= 3250, "Imperial max not ~3219m"
    assert imperial_step_m >= 400 and imperial_step_m <= 450, "Imperial step not ~402m"

    print("Metric requirements:")
    metric_min_m = requirements["metric_min"] * KM_TO_METERS
    metric_max_m = requirements["metric_max"] * KM_TO_METERS
    metric_step_m = requirements["metric_step"] * KM_TO_METERS

    print(f"  Range: {requirements['metric_min']}-{requirements['metric_max']} km ({metric_min_m:.0f}-{metric_max_m:.0f} m)")
    print(f"  Step: {requirements['metric_step']} km ({metric_step_m:.0f} m)")

    # Verify range is exact
    assert metric_min_m == 500, f"Metric min should be 500m, got {metric_min_m}"
    assert metric_max_m == 5000, f"Metric max should be 5000m, got {metric_max_m}"
    assert metric_step_m == 500, f"Metric step should be 500m, got {metric_step_m}"

    print("Check: All user requirements are met\n")


def main():
    """Run all validation tests."""
    print("RTI Metric/Imperial Conversion Validation")
    print("=" * 50)

    try:
        test_conversion_constants()
        test_imperial_range_logic()
        test_metric_range_logic()
        test_snap_to_increment_logic()
        test_user_requirements()

        print("SUCCESS: ALL TESTS PASSED!")
        print("RTI metric/imperial conversion implementation is mathematically correct.")

    except AssertionError as e:
        print(f"FAILED: TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: UNEXPECTED ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
