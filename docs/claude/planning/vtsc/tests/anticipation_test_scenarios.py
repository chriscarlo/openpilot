#!/usr/bin/env python3
"""
Test scenarios for VTSC anticipatory deceleration

This module defines test cases that can be used to validate different
anticipation time calculation approaches before real-world testing.
"""

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Constants from VTSC
MS_TO_KPH = 3.6
KPH_TO_MS = 1 / 3.6
G_TO_MS2 = 9.81

@dataclass
class TestScenario:
    """Defines a test scenario for anticipatory deceleration"""
    name: str
    description: str
    v_ego_ms: float  # Current speed in m/s
    v_target_ms: float  # Target speed for curve in m/s
    max_lat_acc_ms2: float  # Maximum lateral acceleration in curve
    distance_to_apex_m: float  # Distance to apex in meters
    model_confidence: float  # Model confidence (0-1)
    expected_anticipation_s: float  # Expected anticipation time for validation
    road_type: str  # highway, mountain, city

    @property
    def v_ego_kph(self) -> float:
        return self.v_ego_ms * MS_TO_KPH

    @property
    def v_target_kph(self) -> float:
        return self.v_target_ms * MS_TO_KPH

    @property
    def speed_reduction_pct(self) -> float:
        if self.v_ego_ms > 0:
            return (self.v_ego_ms - self.v_target_ms) / self.v_ego_ms * 100
        return 0


# Define comprehensive test scenarios
TEST_SCENARIOS = [
    # Highway scenarios - High speed, gentle curves
    TestScenario(
        name="highway_gentle",
        description="Highway gentle curve at 120 kph",
        v_ego_ms=120 * KPH_TO_MS,
        v_target_ms=110 * KPH_TO_MS,
        max_lat_acc_ms2=0.8,
        distance_to_apex_m=200,
        model_confidence=0.9,
        expected_anticipation_s=2.0,
        road_type="highway"
    ),
    TestScenario(
        name="highway_medium",
        description="Highway medium curve at 130 kph",
        v_ego_ms=130 * KPH_TO_MS,
        v_target_ms=95 * KPH_TO_MS,
        max_lat_acc_ms2=1.5,
        distance_to_apex_m=180,
        model_confidence=0.85,
        expected_anticipation_s=2.8,
        road_type="highway"
    ),
    TestScenario(
        name="highway_sharp",
        description="Highway sharp curve at 140 kph",
        v_ego_ms=140 * KPH_TO_MS,
        v_target_ms=80 * KPH_TO_MS,
        max_lat_acc_ms2=2.2,
        distance_to_apex_m=250,
        model_confidence=0.8,
        expected_anticipation_s=3.5,
        road_type="highway"
    ),

    # Mountain road scenarios - Medium speed, sharp curves
    TestScenario(
        name="mountain_hairpin",
        description="Mountain hairpin at 60 kph",
        v_ego_ms=60 * KPH_TO_MS,
        v_target_ms=25 * KPH_TO_MS,
        max_lat_acc_ms2=2.8,
        distance_to_apex_m=80,
        model_confidence=0.75,
        expected_anticipation_s=2.5,
        road_type="mountain"
    ),
    TestScenario(
        name="mountain_medium",
        description="Mountain medium curve at 70 kph",
        v_ego_ms=70 * KPH_TO_MS,
        v_target_ms=45 * KPH_TO_MS,
        max_lat_acc_ms2=2.0,
        distance_to_apex_m=100,
        model_confidence=0.8,
        expected_anticipation_s=2.2,
        road_type="mountain"
    ),
    TestScenario(
        name="mountain_sweeper",
        description="Mountain sweeper at 80 kph",
        v_ego_ms=80 * KPH_TO_MS,
        v_target_ms=65 * KPH_TO_MS,
        max_lat_acc_ms2=1.2,
        distance_to_apex_m=120,
        model_confidence=0.85,
        expected_anticipation_s=1.8,
        road_type="mountain"
    ),

    # City scenarios - Low speed, tight turns
    TestScenario(
        name="city_intersection",
        description="City intersection turn at 50 kph",
        v_ego_ms=50 * KPH_TO_MS,
        v_target_ms=20 * KPH_TO_MS,
        max_lat_acc_ms2=2.5,
        distance_to_apex_m=40,
        model_confidence=0.7,
        expected_anticipation_s=1.5,
        road_type="city"
    ),
    TestScenario(
        name="city_roundabout",
        description="City roundabout at 40 kph",
        v_ego_ms=40 * KPH_TO_MS,
        v_target_ms=25 * KPH_TO_MS,
        max_lat_acc_ms2=1.8,
        distance_to_apex_m=35,
        model_confidence=0.75,
        expected_anticipation_s=1.2,
        road_type="city"
    ),

    # Edge cases
    TestScenario(
        name="edge_minimal_reduction",
        description="Very small speed reduction",
        v_ego_ms=80 * KPH_TO_MS,
        v_target_ms=75 * KPH_TO_MS,
        max_lat_acc_ms2=0.6,
        distance_to_apex_m=150,
        model_confidence=0.9,
        expected_anticipation_s=1.0,
        road_type="highway"
    ),
    TestScenario(
        name="edge_extreme_reduction",
        description="Extreme speed reduction",
        v_ego_ms=120 * KPH_TO_MS,
        v_target_ms=40 * KPH_TO_MS,
        max_lat_acc_ms2=3.0,
        distance_to_apex_m=200,
        model_confidence=0.6,
        expected_anticipation_s=3.8,
        road_type="mountain"
    ),
    TestScenario(
        name="edge_low_confidence",
        description="Low model confidence scenario",
        v_ego_ms=90 * KPH_TO_MS,
        v_target_ms=60 * KPH_TO_MS,
        max_lat_acc_ms2=1.5,
        distance_to_apex_m=150,
        model_confidence=0.4,
        expected_anticipation_s=2.5,
        road_type="highway"
    ),
]


class AnticipationCalculator:
    """Base class for different anticipation time calculation methods"""

    def calculate(self, scenario: TestScenario) -> float:
        """Calculate anticipation time for given scenario"""
        raise NotImplementedError

    def validate_comfort(self, scenario: TestScenario, anticipation_time: float) -> tuple[bool, float]:
        """
        Validate if the anticipation time allows comfortable deceleration
        Returns (is_comfortable, required_decel_g)
        """
        anticipation_distance = anticipation_time * scenario.v_ego_ms
        effective_distance = scenario.distance_to_apex_m - anticipation_distance

        if effective_distance <= 10:  # Minimum 10m
            return False, float('inf')

        # Calculate required deceleration
        required_decel_ms2 = (scenario.v_ego_ms**2 - scenario.v_target_ms**2) / (2 * effective_distance)
        required_decel_g = abs(required_decel_ms2) / G_TO_MS2

        # Comfortable is less than 0.2g
        is_comfortable = required_decel_g <= 0.2

        return is_comfortable, required_decel_g


class SimpleMultiplicativeCalculator(AnticipationCalculator):
    """Original simple multiplicative approach"""

    def calculate(self, scenario: TestScenario) -> float:
        base_time = 1.5

        # Speed factor
        speed_factor = np.clip(scenario.v_ego_ms / 20.0, 0.7, 1.5)

        # Delta factor
        if scenario.v_ego_ms > 0.1:
            delta_ratio = (scenario.v_ego_ms - scenario.v_target_ms) / scenario.v_ego_ms
            delta_factor = np.clip(1.0 + delta_ratio * 0.5, 1.0, 1.5)
        else:
            delta_factor = 1.0

        # Severity factor
        severity_factor = np.clip(scenario.max_lat_acc_ms2 / 1.5, 0.8, 1.3)

        anticipation_time = base_time * speed_factor * delta_factor * severity_factor

        return np.clip(anticipation_time, 1.0, 3.0)


class EnhancedSigmoidCalculator(AnticipationCalculator):
    """Enhanced approach using sigmoid functions for smoother scaling"""

    @staticmethod
    def sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
        """Sigmoid function for smooth transitions"""
        return 1 / (1 + np.exp(-k * (x - x0)))

    def calculate(self, scenario: TestScenario) -> float:
        # Base time varies with road type
        base_times = {
            "highway": 2.5,
            "mountain": 2.0,
            "city": 1.5
        }
        base_time = base_times.get(scenario.road_type, 2.0)

        # Speed factor: sigmoid curve centered at 25 m/s (~90 kph)
        # Maps 0-50 m/s to 0.5-1.5x multiplier
        speed_factor = 0.5 + self.sigmoid(scenario.v_ego_ms, k=0.06, x0=25)

        # Delta factor: based on percentage reduction
        speed_reduction_pct = scenario.speed_reduction_pct / 100
        delta_factor = 1.0 + 0.8 * self.sigmoid(speed_reduction_pct, k=10, x0=0.3)

        # Severity factor: sharper curves need more time
        severity_factor = 0.8 + 0.7 * self.sigmoid(scenario.max_lat_acc_ms2, k=2, x0=1.5)

        # Confidence factor: lower confidence = more conservative
        confidence_factor = 0.8 + 0.4 * scenario.model_confidence

        anticipation_time = base_time * speed_factor * delta_factor * severity_factor * confidence_factor

        # Apply comfort constraint
        min_comfort_time = self._calculate_min_comfort_time(scenario)
        anticipation_time = max(anticipation_time, min_comfort_time)

        return np.clip(anticipation_time, 0.8, 4.0)

    def _calculate_min_comfort_time(self, scenario: TestScenario) -> float:
        """Calculate minimum time needed for comfortable deceleration"""
        comfort_decel = 0.15 * G_TO_MS2  # 0.15g comfortable decel

        # Time needed to decelerate at comfort rate
        decel_time = (scenario.v_ego_ms - scenario.v_target_ms) / comfort_decel

        # Distance covered during deceleration
        decel_distance = (scenario.v_ego_ms**2 - scenario.v_target_ms**2) / (2 * comfort_decel)

        # Remaining distance at target speed
        remaining_distance = scenario.distance_to_apex_m - decel_distance

        if remaining_distance <= 0:
            # Can't achieve comfortable decel, return proportional time
            return min(2.0, scenario.distance_to_apex_m / scenario.v_ego_ms * 0.3)

        # Time at target speed (this is our anticipation time)
        time_at_target = remaining_distance / scenario.v_target_ms

        # We want at least 1 second at target speed
        return max(1.0, min(time_at_target, 3.0))


class AdaptiveComfortCalculator(AnticipationCalculator):
    """Adaptive approach that prioritizes comfort over fixed timing"""

    def calculate(self, scenario: TestScenario) -> float:
        # Define comfort zones based on deceleration rate
        comfort_zones = [
            (0.10, 1.0),  # Very comfortable: 0.1g → 1x time
            (0.15, 1.2),  # Comfortable: 0.15g → 1.2x time
            (0.20, 1.5),  # Noticeable: 0.2g → 1.5x time
            (0.25, 2.0),  # Firm: 0.25g → 2x time
        ]

        # Start with ideal anticipation based on speed
        ideal_anticipation = 1.0 + 2.5 * (scenario.v_ego_ms / 40.0)  # 1-3.5 seconds

        # Check what deceleration this would require
        for decel_g, time_multiplier in comfort_zones:
            test_time = ideal_anticipation * time_multiplier
            is_comfortable, required_g = self.validate_comfort(scenario, test_time)

            if required_g <= decel_g:
                # Factor in curve severity
                severity_bonus = 0.3 * (scenario.max_lat_acc_ms2 / 3.0)

                # Factor in speed reduction magnitude
                reduction_bonus = 0.2 * (scenario.speed_reduction_pct / 50.0)

                final_time = test_time + severity_bonus + reduction_bonus
                return np.clip(final_time, 1.0, 4.0)

        # If we can't find comfortable decel, return maximum
        return 4.0


def run_scenario_tests():
    """Run all test scenarios with different calculators"""
    calculators = {
        "Simple Multiplicative": SimpleMultiplicativeCalculator(),
        "Enhanced Sigmoid": EnhancedSigmoidCalculator(),
        "Adaptive Comfort": AdaptiveComfortCalculator(),
    }

    results = []

    print("=" * 100)
    print(f"{'Scenario':<25} {'Speed':<15} {'Reduction':<10} {'Calculator':<20} {'Anticipation':<12} {'Comfort':<10}")
    print("=" * 100)

    for scenario in TEST_SCENARIOS:
        for calc_name, calculator in calculators.items():
            anticipation_time = calculator.calculate(scenario)
            is_comfortable, decel_g = calculator.validate_comfort(scenario, anticipation_time)

            results.append({
                'scenario': scenario,
                'calculator': calc_name,
                'anticipation_time': anticipation_time,
                'is_comfortable': is_comfortable,
                'decel_g': decel_g
            })

            speed_str = f"{scenario.v_ego_kph:.0f}→{scenario.v_target_kph:.0f} kph"
            reduction_str = f"{scenario.speed_reduction_pct:.0f}%"
            comfort_str = f"{decel_g:.2f}g {'✓' if is_comfortable else '✗'}"

            print(f"{scenario.name:<25} {speed_str:<15} {reduction_str:<10} "
                  f"{calc_name:<20} {anticipation_time:>6.1f}s {comfort_str:<10}")

    print("=" * 100)

    return results


def plot_anticipation_curves():
    """Visualize anticipation time as a function of different parameters"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Anticipation Time Analysis', fontsize=16)

    calculators = {
        "Simple": SimpleMultiplicativeCalculator(),
        "Sigmoid": EnhancedSigmoidCalculator(),
        "Adaptive": AdaptiveComfortCalculator(),
    }

    # Plot 1: Anticipation vs Speed
    ax1 = axes[0, 0]
    speeds = np.linspace(30, 140, 50) * KPH_TO_MS
    for calc_name, calculator in calculators.items():
        anticipations = []
        for speed in speeds:
            scenario = TestScenario(
                name="test", description="", v_ego_ms=speed,
                v_target_ms=speed*0.7, max_lat_acc_ms2=1.5,
                distance_to_apex_m=150, model_confidence=0.8,
                expected_anticipation_s=2.0, road_type="highway"
            )
            anticipations.append(calculator.calculate(scenario))
        ax1.plot(speeds * MS_TO_KPH, anticipations, label=calc_name, linewidth=2)
    ax1.set_xlabel('Speed (km/h)')
    ax1.set_ylabel('Anticipation Time (s)')
    ax1.set_title('Anticipation vs Speed (30% reduction)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Anticipation vs Speed Reduction
    ax2 = axes[0, 1]
    reductions = np.linspace(5, 60, 50)
    for calc_name, calculator in calculators.items():
        anticipations = []
        for reduction_pct in reductions:
            v_ego = 90 * KPH_TO_MS
            v_target = v_ego * (1 - reduction_pct/100)
            scenario = TestScenario(
                name="test", description="", v_ego_ms=v_ego,
                v_target_ms=v_target, max_lat_acc_ms2=1.5,
                distance_to_apex_m=150, model_confidence=0.8,
                expected_anticipation_s=2.0, road_type="highway"
            )
            anticipations.append(calculator.calculate(scenario))
        ax2.plot(reductions, anticipations, label=calc_name, linewidth=2)
    ax2.set_xlabel('Speed Reduction (%)')
    ax2.set_ylabel('Anticipation Time (s)')
    ax2.set_title('Anticipation vs Speed Reduction (90 km/h)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Anticipation vs Lateral Acceleration
    ax3 = axes[1, 0]
    lat_accs = np.linspace(0.5, 3.0, 50)
    for calc_name, calculator in calculators.items():
        anticipations = []
        for lat_acc in lat_accs:
            scenario = TestScenario(
                name="test", description="", v_ego_ms=80*KPH_TO_MS,
                v_target_ms=60*KPH_TO_MS, max_lat_acc_ms2=lat_acc,
                distance_to_apex_m=150, model_confidence=0.8,
                expected_anticipation_s=2.0, road_type="highway"
            )
            anticipations.append(calculator.calculate(scenario))
        ax3.plot(lat_accs, anticipations, label=calc_name, linewidth=2)
    ax3.set_xlabel('Max Lateral Acceleration (m/s²)')
    ax3.set_ylabel('Anticipation Time (s)')
    ax3.set_title('Anticipation vs Curve Severity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Deceleration Profile
    ax4 = axes[1, 1]
    scenario = TEST_SCENARIOS[2]  # Highway sharp curve
    time_points = np.linspace(0, 8, 100)

    for calc_name, calculator in calculators.items():
        anticipation = calculator.calculate(scenario)
        speeds = []
        positions = []

        for t in time_points:
            if t < (scenario.distance_to_apex_m / scenario.v_ego_ms - anticipation):
                # Before deceleration
                speed = scenario.v_ego_ms
                position = scenario.v_ego_ms * t
            else:
                # During/after deceleration
                # Simplified constant decel model
                decel_start_time = scenario.distance_to_apex_m / scenario.v_ego_ms - anticipation
                time_since_decel = t - decel_start_time

                # Calculate position and speed during deceleration
                _, decel_g = calculator.validate_comfort(scenario, anticipation)
                decel = decel_g * G_TO_MS2

                if scenario.v_ego_ms - decel * time_since_decel > scenario.v_target_ms:
                    speed = scenario.v_ego_ms - decel * time_since_decel
                else:
                    speed = scenario.v_target_ms

                position = scenario.v_ego_ms * decel_start_time + \
                          scenario.v_ego_ms * time_since_decel - 0.5 * decel * time_since_decel**2

            speeds.append(speed * MS_TO_KPH)
            positions.append(position)

        ax4.plot(time_points, speeds, label=f"{calc_name} (ant={anticipation:.1f}s)", linewidth=2)

    ax4.axhline(y=scenario.v_target_kph, color='red', linestyle='--', alpha=0.5, label='Target Speed')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (km/h)')
    ax4.set_title(f'Speed Profile: {scenario.description}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("VTSC Anticipatory Deceleration Test Suite\n")

    # Run scenario tests
    results = run_scenario_tests()

    # Summary statistics
    print("\nSummary by Calculator:")
    print("-" * 50)

    for calc_name in ["Simple Multiplicative", "Enhanced Sigmoid", "Adaptive Comfort"]:
        calc_results = [r for r in results if r['calculator'] == calc_name]
        comfortable = sum(1 for r in calc_results if r['is_comfortable'])
        avg_anticipation = np.mean([r['anticipation_time'] for r in calc_results])
        avg_decel = np.mean([r['decel_g'] for r in calc_results])

        print(f"{calc_name}:")
        print(f"  Comfortable scenarios: {comfortable}/{len(calc_results)}")
        print(f"  Average anticipation: {avg_anticipation:.1f}s")
        print(f"  Average deceleration: {avg_decel:.2f}g")

    # Note: Plotting code is included but won't run in test environment
    # Uncomment below to generate plots when matplotlib is available
    # fig = plot_anticipation_curves()
    # plt.savefig('/data/openpilot/docs/claude/planning/vtsc/anticipation_analysis.png')
    # plt.show()
