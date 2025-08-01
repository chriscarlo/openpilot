#!/usr/bin/env python3
"""
Comprehensive test of refined progressive deceleration controller
"""

import numpy as np
from dataclasses import dataclass, asdict

from emergency_scenarios_definition import (
    EMERGENCY_SCENARIOS, EmergencyScenario, EmergencyLevel, VisionStatus
)
from progressive_deceleration_refined import (
    RefinedProgressiveDecelerationController
)


@dataclass
class TestResult:
    scenario_name: str
    success: bool
    intervention_avoided: bool
    max_decel_used: float
    max_decel_g: float
    time_to_target_speed: float
    distance_used: float
    emergency_level_reached: EmergencyLevel
    comfort_violations: int
    final_speed_error: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d['emergency_level_reached'] = self.emergency_level_reached.name
        return d


def simulate_scenario(controller, scenario: EmergencyScenario,
                     lateral_acc_limit: float = 3.05,
                     dt: float = 0.05, debug: bool = False) -> TestResult:
    """Simulate scenario with actual distance parameter"""

    # Calculate actual target speed based on physics
    if scenario.max_curvature > 0:
        physics_target = np.sqrt(lateral_acc_limit / scenario.max_curvature)
    else:
        physics_target = 100.0

    effective_target = min(scenario.v_target_ms, physics_target)

    # Initial conditions
    v_ego = scenario.v_ego_ms
    distance_traveled = 0.0
    time_elapsed = 0.0

    # Tracking
    max_decel_used = 0.0
    max_emergency_level = EmergencyLevel.NORMAL
    comfort_violations = 0
    intervention_triggered = False

    # Run simulation
    max_iterations = int(10.0 / dt)  # 10 seconds max

    for _ in range(max_iterations):
        remaining_distance = scenario.distance_to_curve_m - distance_traveled

        if remaining_distance <= 0 or v_ego <= effective_target * 1.05:
            break

        # Create realistic predicted distances
        pred_distances = np.linspace(10, 100, 5)
        pred_curvatures = np.ones(5) * scenario.max_curvature

        # Get current curvature based on vision
        if scenario.vision_status == VisionStatus.FULL_VISIBILITY:
            current_curvature = scenario.max_curvature
        else:
            current_curvature = None

        # Update controller with actual distance
        result = controller.update(
            v_ego=v_ego,
            current_curvature=current_curvature,
            predicted_curvatures=pred_curvatures,
            distances=pred_distances,
            vision_status=scenario.vision_status,
            model_confidence=scenario.confidence,
            lateral_acc_limit=lateral_acc_limit,
            current_time=time_elapsed,
            actual_distance_to_curve=remaining_distance
        )

        # Apply deceleration
        actual_decel = result['decel_limit']
        v_ego += actual_decel * dt
        v_ego = max(v_ego, 0)

        distance_traveled += v_ego * dt
        time_elapsed += dt

        # Track metrics
        max_decel_used = min(max_decel_used, actual_decel)
        if result['emergency_level'].value > max_emergency_level.value:
            max_emergency_level = result['emergency_level']

        if abs(actual_decel) > 1.47:
            comfort_violations += 1

        # Check for intervention
        if result['intervention_required']:
            intervention_triggered = True
            break

    # Results
    success = v_ego <= effective_target * 1.1
    intervention_avoided = not intervention_triggered

    return TestResult(
        scenario_name=scenario.name,
        success=success,
        intervention_avoided=intervention_avoided,
        max_decel_used=max_decel_used,
        max_decel_g=abs(max_decel_used) / 9.81,
        time_to_target_speed=time_elapsed,
        distance_used=distance_traveled,
        emergency_level_reached=max_emergency_level,
        comfort_violations=comfort_violations,
        final_speed_error=(v_ego - effective_target) / effective_target if effective_target > 0 else 0
    )


def test_all_scenarios():
    """Test against all emergency scenarios"""

    results = {}

    print("Testing Refined Progressive Deceleration Controller")
    print("="*80)
    print("\nSuccess Criteria:")
    print("1. Success Rate ≥ 60%")
    print("2. Use minimal deceleration (comfort)")
    print("3. Progressive response")
    print("4. Handle blind corners")
    print("5. No premature intervention")
    print("\n" + "="*80)

    for scenario in EMERGENCY_SCENARIOS:
        controller = RefinedProgressiveDecelerationController()  # Fresh controller
        result = simulate_scenario(controller, scenario)
        results[scenario.name] = result

        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        print(f"{status} {scenario.name}: "
              f"{result.max_decel_g:.2f}g, {result.emergency_level_reached.name}, "
              f"t={result.time_to_target_speed:.1f}s")

    # Analysis
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)

    total = len(results)
    successful = sum(1 for r in results.values() if r.success)
    intervention_avoided = sum(1 for r in results.values() if r.intervention_avoided)

    print(f"\nSuccess Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"Avoided Intervention: {intervention_avoided}/{total} ({intervention_avoided/total*100:.1f}%)")

    # Emergency level distribution
    level_counts = dict.fromkeys(EmergencyLevel, 0)
    for result in results.values():
        level_counts[result.emergency_level_reached] += 1

    print("\nEmergency Level Distribution:")
    for level, count in level_counts.items():
        if count > 0:
            print(f"  {level.name}: {count}")

    # Average metrics
    avg_g = np.mean([r.max_decel_g for r in results.values()])
    avg_comfort_violations = np.mean([r.comfort_violations for r in results.values()])

    print(f"\nAverage max deceleration: {avg_g:.2f}g")
    print(f"Average comfort violations: {avg_comfort_violations:.1f}")

    # Success criteria evaluation
    print("\n" + "="*80)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*80)

    criteria_met = []
    criteria_not_met = []

    # 1. Success rate ≥ 60%
    if successful/total >= 0.6:
        criteria_met.append(f"✓ Success rate: {successful/total*100:.1f}% ≥ 60%")
    else:
        criteria_not_met.append(f"✗ Success rate: {successful/total*100:.1f}% < 60%")

    # 2. Comfort (average g < 0.4)
    if avg_g < 0.4:
        criteria_met.append(f"✓ Comfort: avg {avg_g:.2f}g < 0.4g")
    else:
        criteria_not_met.append(f"✗ Comfort: avg {avg_g:.2f}g ≥ 0.4g")

    # 3. Progressive response (uses multiple levels)
    levels_used = sum(1 for count in level_counts.values() if count > 0)
    if levels_used >= 3:
        criteria_met.append(f"✓ Progressive: uses {levels_used} emergency levels")
    else:
        criteria_not_met.append(f"✗ Progressive: only uses {levels_used} levels")

    # 4. Blind corner handling
    blind_scenarios = [r for name, r in results.items() if 'blind' in name or 'fov' in name]
    blind_success = sum(1 for r in blind_scenarios if r.success) / len(blind_scenarios) if blind_scenarios else 0
    if blind_success >= 0.5:
        criteria_met.append(f"✓ Blind corners: {blind_success*100:.0f}% success")
    else:
        criteria_not_met.append(f"✗ Blind corners: {blind_success*100:.0f}% success")

    # 5. No premature intervention
    intervention_rate = (total - intervention_avoided) / total
    if intervention_rate < 0.3:
        criteria_met.append(f"✓ Intervention rate: {intervention_rate*100:.0f}% < 30%")
    else:
        criteria_not_met.append(f"✗ Intervention rate: {intervention_rate*100:.0f}% ≥ 30%")

    print("\nCriteria Met:")
    for c in criteria_met:
        print(f"  {c}")

    if criteria_not_met:
        print("\nCriteria Not Met:")
        for c in criteria_not_met:
            print(f"  {c}")

    print(f"\nOVERALL: {len(criteria_met)}/{5} criteria met")

    # Summary
    success_threshold = 4  # Need 4/5 criteria
    if len(criteria_met) >= success_threshold:
        print("\n" + "="*80)
        print("✓ SUCCESS! Refined controller meets the acceptance criteria.")
        print("="*80)

        # Performance summary
        print("\nPerformance Summary:")
        print(f"- Successfully handles {successful}/{total} scenarios")
        print(f"- Average deceleration: {avg_g:.2f}g")
        print(f"- Intervention rate: {intervention_rate*100:.0f}%")

        # Show intervention scenarios
        intervention_scenarios = [(name, r) for name, r in results.items()
                                 if not r.intervention_avoided]
        if intervention_scenarios:
            print("\nScenarios requiring driver intervention:")
            for name, _ in intervention_scenarios:
                scenario = next(s for s in EMERGENCY_SCENARIOS if s.name == name)
                print(f"  - {name} (needs {scenario.required_decel_g:.2f}g)")
    else:
        print("\n" + "="*80)
        print("✗ Refined controller still needs improvement.")
        print("="*80)

    return results, len(criteria_met) >= success_threshold


if __name__ == "__main__":
    results, meets_criteria = test_all_scenarios()
