#!/usr/bin/env python3
"""
Baseline test runner for physics-based VTSC with advanced analysis
Uses scipy, matplotlib, and other tools for comprehensive testing
"""

import numpy as np
import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt
import scipy.signal as signal
import warnings
warnings.filterwarnings('ignore')

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

class MockSM(dict):
    """Mock SM that acts like both dict and object"""
    def __init__(self):
        super().__init__()
        self.valid = {'modelV2': True}
        self['modelV2'] = MockModelData()
        self['carState'] = MockCarState()

@dataclass
class TestResult:
    scenario_name: str
    success: bool
    max_decel_g: float
    final_speed_error: float
    reached_target: bool
    details: dict
    trajectory: dict

@dataclass
class PhysicsMetrics:
    """Advanced metrics for physics-based analysis"""
    jerk_rms: float
    jerk_max: float
    decel_smoothness: float  # Lower is smoother
    anticipation_quality: float  # 0-1, higher is better
    energy_efficiency: float  # Integral of acceleration squared
    comfort_score: float  # Combined metric

def calculate_physics_metrics(times: np.ndarray, speeds: np.ndarray,
                            accels: np.ndarray) -> PhysicsMetrics:
    """Calculate advanced physics metrics from trajectory"""
    dt = times[1] - times[0] if len(times) > 1 else 0.05

    # Calculate jerk
    jerk = np.diff(accels) / dt if len(accels) > 1 else np.array([0])
    jerk_rms = np.sqrt(np.mean(jerk**2))
    jerk_max = np.max(np.abs(jerk))

    # Deceleration smoothness (spectral flatness)
    if len(accels) > 10:
        freqs, psd = signal.welch(accels, fs=1/dt, nperseg=min(len(accels)//2, 64))
        # Geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)
        decel_smoothness = 1 - (geometric_mean / (arithmetic_mean + 1e-10))
    else:
        decel_smoothness = 0.5

    # Anticipation quality - how early and smoothly we reach target
    # Look for plateau in speed profile
    if len(speeds) > 20:
        speed_derivative = np.abs(np.diff(speeds))
        plateau_indices = np.where(speed_derivative < 0.1)[0]
        if len(plateau_indices) > 0:
            first_plateau = plateau_indices[0]
            anticipation_quality = 1.0 - (first_plateau / len(speeds))
        else:
            anticipation_quality = 0.0
    else:
        anticipation_quality = 0.5

    # Energy efficiency
    energy_efficiency = np.trapz(accels**2, times) if len(times) > 1 else 0

    # Comfort score (ISO 2631 inspired)
    # Weighted combination of acceleration and jerk
    accel_weight = 0.4
    jerk_weight = 0.6
    max_comfortable_accel = 2.0  # m/s²
    max_comfortable_jerk = 3.0   # m/s³

    accel_discomfort = np.mean(np.abs(accels)) / max_comfortable_accel
    jerk_discomfort = jerk_rms / max_comfortable_jerk
    comfort_score = 1.0 - (accel_weight * accel_discomfort + jerk_weight * jerk_discomfort)
    comfort_score = np.clip(comfort_score, 0, 1)

    return PhysicsMetrics(
        jerk_rms=jerk_rms,
        jerk_max=jerk_max,
        decel_smoothness=decel_smoothness,
        anticipation_quality=anticipation_quality,
        energy_efficiency=energy_efficiency,
        comfort_score=comfort_score
    )

def simulate_curve_scenario(controller, scenario_name: str,
                          v_ego_kph: float, curve_radius_m: float,
                          distance_to_curve_m: float,
                          vision_confidence: float = 0.9,
                          dt: float = 0.05,
                          plot: bool = False) -> TestResult:
    """Simulate a curve scenario with advanced metrics"""

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
    sm = MockSM()

    # Tracking variables
    time_sim = 0.0
    distance_remaining = distance_to_curve_m
    max_decel = 0.0
    reached_target = False

    # Trajectory storage
    times = []
    speeds = []
    decels = []
    distances = []
    states = []

    # Run simulation
    while distance_remaining > -50 and time_sim < 15.0:  # Extended for better analysis
        # Simulate model predictions (33 points, ~2.5 seconds ahead)
        n_points = 33
        time_points = np.linspace(0, 2.5, n_points)
        distance_points = v_ego * time_points

        # Create curvature predictions with realistic model behavior
        orientation_rates = []
        velocities = []

        for i, d in enumerate(distance_points):
            future_dist = distance_remaining - d
            if future_dist <= 0:  # In the curve
                # Model would detect turn rate with some noise
                turn_rate = curvature * v_ego
                noise = np.random.normal(0, 0.01 * turn_rate) if vision_confidence < 0.8 else 0
                orientation_rates.append(abs(turn_rate + noise))
            else:
                # Approaching curve - gradual detection
                if future_dist < 20:  # Within 20m, start detecting
                    partial_rate = curvature * v_ego * (1 - future_dist/20)
                    orientation_rates.append(abs(partial_rate))
                else:
                    orientation_rates.append(0.0)
            velocities.append(max(v_ego - 0.1 * i, 1.0))  # Predicted deceleration

        # Update model data
        sm['modelV2'].orientationRate.z = orientation_rates
        sm['modelV2'].velocity.x = velocities

        # Simulate vision confidence
        sm['modelV2'].laneLineProbs = [vision_confidence] * 3

        # Update controller
        controller.update(sm, enabled=True, v_ego=v_ego, a_ego=decels[-1] if decels else 0.0,
                         v_cruise_setpoint=v_ego_kph/3.6 + 5.0)

        # Get acceleration command
        a_target = controller.a_target

        # Apply physics with basic vehicle dynamics
        v_ego += a_target * dt
        v_ego = max(v_ego, 0)
        distance_remaining -= v_ego * dt
        time_sim += dt

        # Store trajectory
        times.append(time_sim)
        speeds.append(v_ego)
        decels.append(a_target)
        distances.append(distance_remaining)
        states.append(controller.state)

        # Track metrics
        if a_target < max_decel:
            max_decel = a_target

        # Check if reached target
        if v_ego <= v_target_physics * 1.02 and not reached_target:
            reached_target = True
            print(f"  Reached target speed at t={time_sim:.1f}s, d={distance_remaining:.0f}m")

    # Convert to arrays for analysis
    times = np.array(times)
    speeds = np.array(speeds)
    decels = np.array(decels)
    distances = np.array(distances)

    # Calculate physics metrics
    metrics = calculate_physics_metrics(times, speeds, decels)

    # Final results
    final_speed = speeds[-1] if len(speeds) > 0 else v_ego
    speed_error = (final_speed - v_target_physics) / v_target_physics if v_target_physics > 0 else 0

    # Determine success with refined criteria
    success = False
    if "normal" in scenario_name or "gentle" in scenario_name:
        # Normal scenarios need smooth deceleration AND good comfort
        success = (reached_target and
                  abs(max_decel) <= 2.45 and  # 0.25g
                  metrics.comfort_score > 0.7)
    elif "sharp" in scenario_name or "late" in scenario_name:
        # Challenging scenarios need to reach safe speed
        success = (final_speed <= v_target_physics * 1.1 and
                  abs(max_decel) <= 6.0)
    else:
        # Edge cases - just avoid crashes
        success = final_speed <= v_target_physics * 1.2

    print(f"  Final speed: {final_speed*3.6:.0f} km/h (target: {v_target_physics*3.6:.0f})")
    print(f"  Max decel: {abs(max_decel)/9.81:.2f}g")
    print(f"  Comfort score: {metrics.comfort_score:.2f}")
    print(f"  Anticipation quality: {metrics.anticipation_quality:.2f}")
    print(f"  Success: {'✓' if success else '✗'}")

    # Optional plotting
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        # Speed profile
        axes[0].plot(times, speeds * 3.6, 'b-', label='Speed')
        axes[0].axhline(y=v_target_physics * 3.6, color='r', linestyle='--', label='Target')
        axes[0].set_ylabel('Speed (km/h)')
        axes[0].legend()
        axes[0].grid(True)

        # Acceleration profile
        axes[1].plot(times, decels, 'g-', label='Acceleration')
        axes[1].axhline(y=-6.0, color='r', linestyle='--', label='System limit')
        axes[1].set_ylabel('Acceleration (m/s²)')
        axes[1].legend()
        axes[1].grid(True)

        # Distance to curve
        axes[2].plot(times, distances, 'm-', label='Distance to curve')
        axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[2].set_ylabel('Distance (m)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True)

        plt.suptitle(f'{scenario_name} - Comfort: {metrics.comfort_score:.2f}')
        plt.tight_layout()
        plt.savefig(f'/data/openpilot/docs/claude/tests/vtsc/plots/{scenario_name}_baseline.png')
        plt.close()

    return TestResult(
        scenario_name=scenario_name,
        success=success,
        max_decel_g=abs(max_decel)/9.81,
        final_speed_error=speed_error,
        reached_target=reached_target,
        details={
            'final_speed_kph': final_speed * 3.6,
            'target_speed_kph': v_target_physics * 3.6,
            'time_to_target': times[np.where(speeds <= v_target_physics * 1.02)[0][0]] if reached_target else None,
            'min_distance': np.min(distances),
            'metrics': metrics
        },
        trajectory={
            'times': times,
            'speeds': speeds,
            'accels': decels,
            'distances': distances
        }
    )

def optimize_sigmoid_parameters(test_results: list[TestResult]) -> dict:
    """Use scipy to optimize sigmoid parameters based on test results"""

    # Extract key metrics for optimization
    comfort_scores = [r.details['metrics'].comfort_score for r in test_results]
    success_rates = [1.0 if r.success else 0.0 for r in test_results]

    # Current sigmoid parameters
    current_params = {
        'high_accel': 3.12,
        'low_accel': 1.5,
        'center_curvature': 0.060,
        'k': 75
    }

    def objective(params):
        """Objective function to minimize"""
        # Would need to re-run tests with new params
        # For now, return a placeholder
        return -np.mean(comfort_scores) - 0.5 * np.mean(success_rates)

    # Bounds for parameters
    bounds = [
        (2.5, 4.0),    # high_accel
        (1.0, 2.0),    # low_accel
        (0.04, 0.08),  # center_curvature
        (50, 100)      # k
    ]

    # Optimize (would need full integration for real optimization)
    # result = opt.minimize(objective, list(current_params.values()), bounds=bounds)

    print("\nSigmoid Parameter Analysis:")
    print(f"Current high_accel: {current_params['high_accel']}")
    print(f"Current low_accel: {current_params['low_accel']}")
    print(f"Average comfort score: {np.mean(comfort_scores):.2f}")

    return current_params

def run_baseline_tests():
    """Run comprehensive baseline tests with advanced analysis"""

    print("="*80)
    print("BASELINE PHYSICS VTSC TESTING - ADVANCED ANALYSIS")
    print("="*80)
    print("Testing original physics-based implementation with detailed metrics")

    # Create plots directory
    import os
    os.makedirs('/data/openpilot/docs/claude/tests/vtsc/plots', exist_ok=True)

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

    for i, (scenario_name, v_kph, radius, distance, confidence) in enumerate(test_scenarios):
        # Plot first 3 scenarios
        plot = i < 3
        result = simulate_curve_scenario(
            controller, scenario_name, v_kph, radius, distance, confidence, plot=plot
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

    # Advanced metrics
    all_comfort = [r.details['metrics'].comfort_score for r in results]
    all_anticipation = [r.details['metrics'].anticipation_quality for r in results]
    all_smoothness = [r.details['metrics'].decel_smoothness for r in results]

    print("\nAdvanced Metrics:")
    print(f"- Average comfort score: {np.mean(all_comfort):.2f}")
    print(f"- Average anticipation quality: {np.mean(all_anticipation):.2f}")
    print(f"- Average decel smoothness: {np.mean(all_smoothness):.2f}")

    # Performance metrics
    avg_decel = np.mean([r.max_decel_g for r in results])
    max_decel_overall = max(r.max_decel_g for r in results)

    print("\nPerformance Metrics:")
    print(f"- Average max deceleration: {avg_decel:.2f}g")
    print(f"- Maximum deceleration used: {max_decel_overall:.2f}g")
    print(f"- Scenarios exceeding 0.6g: {sum(1 for r in results if r.max_decel_g > 0.6)}")

    # Optimize sigmoid parameters
    optimize_sigmoid_parameters(results)

    # Areas needing improvement
    print("\n" + "="*80)
    print("AREAS FOR ENHANCEMENT")
    print("="*80)

    failures = [r for r in results if not r.success]
    if failures:
        print("\nFailed scenarios:")
        for f in failures:
            print(f"- {f.scenario_name}: {f.max_decel_g:.2f}g decel, "
                  f"{f.final_speed_error*100:.0f}% speed error, "
                  f"comfort: {f.details['metrics'].comfort_score:.2f}")

    # Missing features vs our enhanced version
    print("\nKey enhancement opportunities:")
    print("1. Emergency level system integration with sigmoid curves")
    print("2. Vision occlusion handling with Kalman filtering")
    print("3. Adaptive sigmoid parameters based on emergency level")
    print("4. Enhanced jerk limiting per emergency state")
    print("5. System constraint enforcement with graceful degradation")
    print("6. Intervention triggering with driver handoff")

    # Statistical analysis
    print("\nStatistical Analysis:")
    from scipy import stats
    comfort_normal = [r.details['metrics'].comfort_score for r in normal_results]
    comfort_challenging = [r.details['metrics'].comfort_score for r in challenging_results]
    if len(comfort_normal) > 1 and len(comfort_challenging) > 1:
        t_stat, p_value = stats.ttest_ind(comfort_normal, comfort_challenging)
        print(f"Comfort score difference (normal vs challenging): p={p_value:.3f}")

    return results

if __name__ == "__main__":
    results = run_baseline_tests()
