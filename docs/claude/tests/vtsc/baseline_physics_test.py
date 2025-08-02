#!/usr/bin/env python3
"""
Baseline test runner for physics-based VTSC
Runs our existing test scenarios against the original physics version
to establish performance baseline before enhancements
"""

import numpy as np
import sys
from dataclasses import dataclass
try:
    from scipy import signal, optimize, interpolate
    from scipy.integrate import odeint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using basic analysis")
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

sys.path.insert(0, '.')

# Import the physics-based VTSC
from enhanced_physics_vtsc import VisionTurnController

# Mock objects needed for testing
class MockCP:
    def __init__(self):
        self.steerRatio = 15.0
        self.wheelbase = 2.7

class MockModelData:
    def __init__(self):
        self.orientationRate = MockOrientation()
        self.velocity = MockVelocity()
        self.laneLineProbs = [0.9, 0.9, 0.9]
        self.laneLineStds = [0.1, 0.1, 0.1]

class MockOrientation:
    def __init__(self):
        self.z = []

class MockVelocity:
    def __init__(self):
        self.x = []

class MockCarState:
    def __init__(self):
        self.gasPressed = False
        self.steeringAngleDeg = 0.0

@dataclass
class TestResult:
    scenario_name: str
    success: bool
    max_decel_g: float
    final_speed_error: float
    reached_target: bool
    details: dict
    trajectory: dict = None  # Store full trajectory for analysis

def calculate_jerk(accelerations: list[float], dt: float) -> list[float]:
    """Calculate jerk (derivative of acceleration) using signal processing"""
    if len(accelerations) < 2:
        return [0.0]

    if SCIPY_AVAILABLE and len(accelerations) >= 5:
        # Use Savitzky-Golay filter for smooth differentiation
        window_length = min(5, len(accelerations) if len(accelerations) % 2 == 1 else len(accelerations)-1)
        jerks = signal.savgol_filter(accelerations,
                                     window_length=window_length,
                                     polyorder=min(2, window_length-1),
                                     deriv=1) / dt
    else:
        jerks = np.gradient(accelerations) / dt

    return jerks.tolist()

def analyze_trajectory_comfort(speeds: list[float], accels: list[float], dt: float) -> dict:
    """Analyze trajectory comfort metrics using advanced signal processing"""
    speeds_np = np.array(speeds)
    accels_np = np.array(accels)

    # Calculate jerk
    jerks = calculate_jerk(accels, dt)
    jerks_np = np.array(jerks)

    # Calculate ride comfort metrics
    rms_accel = np.sqrt(np.mean(accels_np**2))
    rms_jerk = np.sqrt(np.mean(jerks_np**2))

    # Frequency analysis if scipy available
    if SCIPY_AVAILABLE and len(accels) > 10:
        freqs, psd = signal.welch(accels_np, fs=1/dt, nperseg=min(len(accels)//4, 256))
        dominant_freq = freqs[np.argmax(psd)]

        # ISO 2631 weighted acceleration (simplified)
        # For 20Hz sampling (dt=0.05), nyquist freq is 10Hz
        # Use a lowpass filter at 8Hz to avoid human discomfort frequencies
        b, a = signal.butter(4, 8.0, btype='low', fs=1/dt)
        weighted_accel = signal.filtfilt(b, a, accels_np)
        comfort_index = np.sqrt(np.mean(weighted_accel**2))
    else:
        dominant_freq = 0.0
        comfort_index = rms_accel

    return {
        'rms_accel': rms_accel,
        'rms_jerk': rms_jerk,
        'max_jerk': np.max(np.abs(jerks_np)),
        'dominant_freq': dominant_freq,
        'comfort_index': comfort_index,
        'smoothness': 1.0 / (1.0 + rms_jerk)  # Higher is smoother
    }

def simulate_curve_scenario(controller, scenario_name: str,
                          v_ego_kph: float, curve_radius_m: float,
                          distance_to_curve_m: float,
                          vision_confidence: float = 0.9,
                          dt: float = 0.05) -> TestResult:
    """Simulate a curve scenario with the physics-based controller"""

    print(f"\nTesting: {scenario_name}")
    print(f"  Speed: {v_ego_kph} km/h")
    print(f"  Curve radius: {curve_radius_m}m")
    print(f"  Distance: {distance_to_curve_m}m")

    # Convert to SI units
    v_ego = v_ego_kph / 3.6
    curvature = 1.0 / curve_radius_m if curve_radius_m > 0 else 0.0

    # Physics target speed
    lateral_limit = 3.0  # m/s²
    v_target_physics = np.sqrt(lateral_limit / curvature) if curvature > 0 else 100.0

    # Initialize controller state
    class MockSM(dict):
        """Mock SubMaster that acts like both dict and object"""
        def __init__(self):
            super().__init__()
            self.valid = {'modelV2': True}
            self['modelV2'] = MockModelData()
            self['carState'] = MockCarState()

    sm = MockSM()

    # Tracking variables
    time_sim = 0.0
    distance_remaining = distance_to_curve_m
    max_decel = 0.0
    reached_target = False
    # Trajectory storage
    times = []
    speeds = []
    accels = []
    distances = []

    # Run simulation
    while distance_remaining > -10 and time_sim < 10.0:
        # Simulate model predictions (33 points, ~2.5 seconds ahead)
        n_points = 33
        t_pred = np.linspace(0, 2.5, n_points)
        d_pred = v_ego * t_pred

        # Create curvature predictions with realistic model behavior
        orientation_rates = []
        velocities = []

        for i, d in enumerate(d_pred):
            if distance_remaining - d <= 0:  # In the curve
                # Model would detect turn rate with some prediction accuracy
                turn_rate = curvature * v_ego  # rad/s
                # Add slight noise to simulate real model uncertainty
                noise_factor = 1.0 + (0.02 * np.sin(2 * np.pi * i / n_points))
                orientation_rates.append(abs(turn_rate * noise_factor))
            else:
                # Approaching curve - model starts detecting it early
                if distance_remaining - d < 20:  # Within 20m of curve
                    early_detection = 0.1 * curvature * v_ego * (1 - (distance_remaining - d) / 20)
                    orientation_rates.append(max(0, early_detection))
                else:
                    orientation_rates.append(0.0)
            velocities.append(max(v_ego * (1 - 0.01 * i), 1.0))  # Predicted velocity with slight decay

        # Update model data
        sm['modelV2'].orientationRate.z = orientation_rates
        sm['modelV2'].velocity.x = velocities

        # Simulate varying vision confidence
        if vision_confidence < 0.7:
            # Low confidence - add more noise
            sm['modelV2'].laneLineProbs = [vision_confidence] * 3
            sm['modelV2'].laneLineStds = [0.3 - 0.2 * vision_confidence] * 3

        # Update controller
        controller.update(sm, enabled=True, v_ego=v_ego, a_ego=accels[-1] if accels else 0.0,
                         v_cruise_setpoint=v_ego_kph/3.6 + 5.0)  # Cruise slightly above current

        # Get acceleration command
        a_target = controller.a_target

        # Apply physics
        v_ego += a_target * dt
        v_ego = max(v_ego, 0)
        distance_remaining -= v_ego * dt
        time_sim += dt

        # Store trajectory
        times.append(time_sim)
        speeds.append(v_ego)
        accels.append(a_target)
        distances.append(distance_remaining)

        # Track metrics
        if a_target < max_decel:
            max_decel = a_target

        # Check if reached target
        if v_ego <= v_target_physics * 1.05 and not reached_target:
            reached_target = True
            print(f"  Reached target speed at t={time_sim:.1f}s, d={distance_remaining:.0f}m")

    # Calculate results
    final_speed = v_ego
    speed_error = (final_speed - v_target_physics) / v_target_physics if v_target_physics > 0 else 0

    # Analyze trajectory comfort
    comfort_metrics = analyze_trajectory_comfort(speeds, accels, dt)

    # Determine success based on multi-criteria evaluation
    success = False
    if "normal" in scenario_name or "gentle" in scenario_name:
        # Normal scenarios need smooth deceleration
        success = (reached_target and
                  abs(max_decel) <= 2.45 and  # 0.25g
                  comfort_metrics['max_jerk'] < 5.0 and  # Jerk limit
                  comfort_metrics['smoothness'] > 0.7)
    elif "sharp" in scenario_name or "late" in scenario_name:
        # Challenging scenarios need to reach safe speed
        success = (final_speed <= v_target_physics * 1.1 and
                  abs(max_decel) <= 6.0 and
                  comfort_metrics['max_jerk'] < 10.0)
    else:
        # Edge cases
        success = final_speed <= v_target_physics * 1.2

    print(f"  Final speed: {final_speed*3.6:.0f} km/h (target: {v_target_physics*3.6:.0f})")
    print(f"  Max decel: {abs(max_decel)/9.81:.2f}g")
    print(f"  Max jerk: {comfort_metrics['max_jerk']:.1f} m/s³")
    print(f"  Comfort index: {comfort_metrics['comfort_index']:.2f}")
    print(f"  Success: {'✓' if success else '✗'}")

    return TestResult(
        scenario_name=scenario_name,
        success=success,
        max_decel_g=abs(max_decel)/9.81,
        final_speed_error=speed_error,
        reached_target=reached_target,
        details={
            'final_speed_kph': final_speed * 3.6,
            'target_speed_kph': v_target_physics * 3.6,
            'time_to_target': time_sim if reached_target else None,
            'min_distance': min(distances),
            'comfort_metrics': analyze_trajectory_comfort(speeds, accels, dt)
        },
        trajectory={
            'times': times,
            'speeds': speeds,
            'accels': accels,
            'distances': distances
        }
    )

def run_baseline_tests():
    """Run our standard test scenarios on the physics-based controller"""

    print("="*80)
    print("BASELINE PHYSICS VTSC TESTING")
    print("="*80)
    print("Testing original physics-based implementation")

    # Initialize controller
    CP = MockCP()
    controller = VisionTurnController(CP)

    # Define test scenarios (same as our enhanced version tests)
    test_scenarios = [
        # Normal scenarios
        ("highway_gentle_curve", 110, 250, 150, 0.95),
        ("city_normal_turn", 50, 50, 80, 0.9),
        ("suburban_curve", 70, 100, 100, 0.85),

        # Challenging scenarios
        ("highway_sharp_curve", 120, 150, 120, 0.8),
        ("late_detection_curve", 90, 60, 70, 0.7),
        ("blind_corner_moderate", 60, 40, 50, 0.4),

        # Edge cases
        ("very_late_sharp_turn", 80, 25, 35, 0.6),
        ("impossible_scenario", 100, 20, 30, 0.3),
    ]

    results = []

    for scenario_name, v_kph, radius, distance, confidence in test_scenarios:
        result = simulate_curve_scenario(
            controller, scenario_name, v_kph, radius, distance, confidence
        )
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("BASELINE RESULTS SUMMARY")
    print("="*80)

    successes = sum(1 for r in results if r.success)
    print(f"\nOverall Success Rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")

    # Categorized results
    normal_results = [r for r in results if "normal" in r.scenario_name or "gentle" in r.scenario_name]
    challenging_results = [r for r in results if "sharp" in r.scenario_name or "late" in r.scenario_name or "blind" in r.scenario_name]
    edge_results = [r for r in results if "very_late" in r.scenario_name or "impossible" in r.scenario_name]

    print(f"\nNormal scenarios: {sum(1 for r in normal_results if r.success)}/{len(normal_results)}")
    print(f"Challenging scenarios: {sum(1 for r in challenging_results if r.success)}/{len(challenging_results)}")
    print(f"Edge scenarios: {sum(1 for r in edge_results if r.success)}/{len(edge_results)}")

    # Performance metrics
    avg_decel = np.mean([r.max_decel_g for r in results])
    max_decel_overall = max(r.max_decel_g for r in results)
    avg_comfort = np.mean([r.details['comfort_metrics']['comfort_index'] for r in results])
    avg_smoothness = np.mean([r.details['comfort_metrics']['smoothness'] for r in results])

    print("\nPerformance Metrics:")
    print(f"- Average max deceleration: {avg_decel:.2f}g")
    print(f"- Maximum deceleration used: {max_decel_overall:.2f}g")
    print(f"- Average comfort index: {avg_comfort:.2f}")
    print(f"- Average smoothness: {avg_smoothness:.2f}")
    print(f"- Scenarios exceeding 0.6g: {sum(1 for r in results if r.max_decel_g > 0.6)}")

    # Areas needing improvement
    print("\n" + "="*80)
    print("AREAS FOR ENHANCEMENT")
    print("="*80)

    failures = [r for r in results if not r.success]
    if failures:
        print("\nFailed scenarios:")
        for f in failures:
            print(f"- {f.scenario_name}: {f.max_decel_g:.2f}g decel, {f.final_speed_error*100:.0f}% speed error")

    # Missing features vs our enhanced version
    print("\nMissing features from enhanced version:")
    print("- No explicit emergency level system")
    print("- No vision occlusion handling")
    print("- No system constraint enforcement (-6.0 m/s²)")
    print("- No intervention triggering for impossible scenarios")
    print("- Limited jerk control per emergency level")
    print("- No explicit anticipation time calculation")

    # Save detailed results for analysis
    if PANDAS_AVAILABLE:
        df_results = pd.DataFrame([{
            'scenario': r.scenario_name,
            'success': r.success,
            'max_decel_g': r.max_decel_g,
            'speed_error_%': r.final_speed_error * 100,
            'max_jerk': r.details['comfort_metrics']['max_jerk'],
            'comfort_index': r.details['comfort_metrics']['comfort_index'],
            'smoothness': r.details['comfort_metrics']['smoothness']
        } for r in results])

        print("\n" + "="*80)
        print("DETAILED METRICS TABLE")
        print("="*80)
        print(df_results.to_string(index=False))

    return results

if __name__ == "__main__":
    results = run_baseline_tests()
