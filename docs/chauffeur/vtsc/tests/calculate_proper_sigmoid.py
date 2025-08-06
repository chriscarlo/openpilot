#!/usr/bin/env python3
"""Calculate and verify proper sigmoid parameters before plotting"""

import math

def test_sigmoid(a_min, a_max, k, c, power=1.0):
    """Test a sigmoid configuration"""
    print(f"\nTesting sigmoid: a_min={a_min}, a_max={a_max}, k={k}, c={c}, power={power}")
    print("-" * 80)

    # Test specific curvatures and see what speeds they produce
    test_points = [
        (0.001, "Highway straight (~1000m radius)"),
        (0.003, "Highway curve (~333m radius)"),
        (0.005, "Highway sweeper (~200m radius)"),
        (0.010, "Fast curve (~100m radius)"),
        (0.020, "Normal turn (~50m radius)"),
        (0.030, "City turn (~33m radius)"),
        (0.050, "Residential turn (~20m radius)"),
        (0.100, "Tight turn (~10m radius)"),
        (0.150, "Very tight (~6.7m radius)"),
        (0.200, "Hairpin (~5m radius)"),
    ]

    print(f"{'Curvature':<12} {'Description':<30} {'Lat Accel':<12} {'Speed (mph)':<12} {'Speed (m/s)':<12}")
    print("-" * 80)

    for curv, desc in test_points:
        # Calculate lateral acceleration using sigmoid
        # Using form: a = a_min + (a_max - a_min) / (1 + exp(k*(curv^power - c)))
        lat_accel = a_min + (a_max - a_min) / (1 + math.exp(k * (curv**power - c)))
        lat_accel = max(a_min, min(lat_accel, a_max))

        # Calculate resulting speed
        speed_mps = math.sqrt(lat_accel / curv)
        speed_mph = speed_mps * 2.237

        print(f"{curv:<12.4f} {desc:<30} {lat_accel:<12.3f} {speed_mph:<12.1f} {speed_mps:<12.1f}")

    return True

# First, let's understand what speeds we want at different curvatures
print("=== TARGET BEHAVIOR ===")
print("We want:")
print("1. Hit max lateral accel (3.12 m/s²) by 65-70 mph")
print("2. Sharp drop from 50mph to 30mph")
print("3. Begin flattening around 30mph")
print("4. Smooth curve to 5mph minimum")
print("")

# Calculate what curvatures correspond to these key speeds
print("=== KEY SPEED POINTS ===")
key_speeds = [
    (70, "Max highway speed"),
    (65, "Should be at max lat accel"),
    (50, "Start of sharp drop"),
    (40, "Middle of transition"),
    (30, "Start flattening"),
    (20, "City speeds"),
    (10, "Neighborhood"),
    (5, "Minimum speed"),
]

print(f"{'Speed (mph)':<12} {'Description':<25} {'Curvature for a=3.12':<20} {'Curvature for a=1.5':<20} {'Curvature for a=0.8':<20}")
print("-" * 100)

for speed_mph, desc in key_speeds:
    speed_mps = speed_mph / 2.237
    # For different lateral accelerations, what curvature would give this speed?
    curv_at_max = 3.12 / (speed_mps ** 2)
    curv_at_mid = 1.5 / (speed_mps ** 2)
    curv_at_min = 0.8 / (speed_mps ** 2)
    print(f"{speed_mph:<12} {desc:<25} {curv_at_max:<20.4f} {curv_at_mid:<20.4f} {curv_at_min:<20.4f}")

print("\n=== ANALYSIS ===")
print("For 65-70mph to use max lat accel (3.12), curvature should be ~0.0035-0.0041")
print("For 50mph transition, curvature is ~0.0056 at max, ~0.0027 at mid")
print("For 30mph flattening, curvature is ~0.0155 at max, ~0.0075 at mid")
print("For 5mph minimum, curvature is ~0.159 at min lat accel")

# Now let's design a sigmoid that achieves this
print("\n" + "="*80)
print("DESIGNING PROPER SIGMOID")
print("="*80)

# Requirements:
# - At curvature 0.004 and below: lat_accel = 3.12 (max)
# - At curvature 0.006: still near max (for 50mph at high lat accel)
# - At curvature 0.010-0.020: sharp transition
# - At curvature 0.030: around 1.5 m/s² (flattening)
# - At curvature 0.16 and above: lat_accel = 0.8 (min)

# Try different sigmoid configurations
configurations = [
    # (a_min, a_max, k, c, power)
    (0.8, 3.12, 200, 0.015, 1.0),  # Simple logistic, high k
    (0.8, 3.12, 150, 0.012, 1.0),  # Slightly less steep
    (0.8, 3.12, 250, 0.018, 1.0),  # Very steep
    (0.7, 3.12, 180, 0.014, 0.9),  # Power modification
    (0.6, 3.12, 160, 0.013, 0.95), # Lower minimum
]

print("\nTesting different sigmoid configurations...")

best_config = None
best_score = float('inf')

for a_min, a_max, k, c, power in configurations:
    print(f"\n{'='*80}")
    print(f"Configuration: a_min={a_min}, a_max={a_max}, k={k}, c={c}, power={power}")

    # Test key points
    test_curvatures = [0.003, 0.006, 0.010, 0.020, 0.030, 0.050, 0.100, 0.200]
    target_speeds = [70, 50, 40, 28, 23, 17, 11, 5]  # Approximate target speeds

    total_error = 0
    for curv, target_speed in zip(test_curvatures, target_speeds, strict=False):
        lat_accel = a_min + (a_max - a_min) / (1 + math.exp(k * (curv**power - c)))
        lat_accel = max(a_min, min(lat_accel, a_max))
        speed_mph = math.sqrt(lat_accel / curv) * 2.237
        error = abs(speed_mph - target_speed)
        total_error += error
        print(f"  Curv={curv:.4f}: {speed_mph:.1f} mph (target: {target_speed}, error: {error:.1f})")

    print(f"  Total error: {total_error:.1f}")

    if total_error < best_score:
        best_score = total_error
        best_config = (a_min, a_max, k, c, power)

print("\n" + "="*80)
print("BEST CONFIGURATION")
print("="*80)
a_min, a_max, k, c, power = best_config
print(f"a_min={a_min}, a_max={a_max}, k={k}, c={c}, power={power}")
print(f"Total error from targets: {best_score:.1f}")

# Test the best configuration in detail
test_sigmoid(a_min, a_max, k, c, power)

# Calculate exact parameters for production code
print("\n" + "="*80)
print("FINAL SIGMOID PARAMETERS FOR IMPLEMENTATION")
print("="*80)
print(f"""
def improved_sigmoid_lateral_acceleration(curvature: float) -> float:
    \"\"\"
    Improved sigmoid with proper S-curve:
    - Reaches max (3.12 m/s²) by 65-70mph
    - Sharp transition from 50mph to 30mph
    - Begins flattening around 30mph
    - Smooth curve to minimum at 5mph
    \"\"\"
    a_max = {a_max:.2f}   # Maximum lateral acceleration (m/s²)
    a_min = {a_min:.2f}   # Minimum lateral acceleration (m/s²)
    k = {k:.1f}        # Steepness factor
    c = {c:.4f}      # Center point (curvature)
    power = {power:.2f}  # Power for curvature transform
    
    # Sigmoid function
    lateral_acceleration = a_min + (a_max - a_min) / (1 + math.exp(k * (curvature**power - c)))
    
    return max(a_min, min(lateral_acceleration, a_max))

# Also need to change MIN_V from 5.6 m/s (12.5 mph) to 2.24 m/s (5 mph)
_MIN_V = 2.24  # 5 mph minimum operating speed
""")
