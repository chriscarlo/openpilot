#!/usr/bin/env python3
"""
Iteration 4: Progressive Deceleration with refined emergency level determination

Key improvements:
1. More nuanced emergency level thresholds
2. Slower escalation for manageable scenarios
3. Better balance between safety and comfort
"""

import numpy as np
from dataclasses import dataclass

from emergency_scenarios_definition import (
    EmergencyLevel, VisionStatus
)


# Refined deceleration limits (m/s²)
DECEL_LIMITS = {
    EmergencyLevel.NORMAL: -1.47,      # 0.15g
    EmergencyLevel.CAUTION: -2.45,     # 0.25g
    EmergencyLevel.WARNING: -3.92,     # 0.40g
    EmergencyLevel.CRITICAL: -5.89,    # 0.60g
    EmergencyLevel.INTERVENTION: -7.85  # 0.80g
}

# Jerk limits for smooth transitions
JERK_LIMITS = {
    EmergencyLevel.NORMAL: 2.0,
    EmergencyLevel.CAUTION: 3.0,
    EmergencyLevel.WARNING: 4.0,
    EmergencyLevel.CRITICAL: 6.0,
    EmergencyLevel.INTERVENTION: 10.0
}

# Refined transition times - balance between urgency and stability
TRANSITION_TIMES = {
    EmergencyLevel.NORMAL: {EmergencyLevel.CAUTION: 0.5},
    EmergencyLevel.CAUTION: {EmergencyLevel.WARNING: 0.3},
    EmergencyLevel.WARNING: {EmergencyLevel.CRITICAL: 0.3},
    EmergencyLevel.CRITICAL: {EmergencyLevel.INTERVENTION: 0.5},
    'de_escalate': 2.0
}


@dataclass
class VisionOcclusionStateV4:
    """Vision occlusion tracking"""
    last_valid_curvature: float = 0.0
    last_valid_timestamp: float = 0.0
    occlusion_start_time: float | None = None
    extrapolated_curvature: float = 0.0
    confidence_decay_factor: float = 1.0
    vision_status: VisionStatus = VisionStatus.FULL_VISIBILITY
    initialized: bool = False

    def update(self, current_curvature: float | None,
               predicted_curvatures: np.ndarray | None,
               vision_status: VisionStatus,
               current_time: float) -> float:
        """Update with proper initialization"""
        self.vision_status = vision_status

        # Initialize from predictions if needed
        if not self.initialized and predicted_curvatures is not None and len(predicted_curvatures) > 0:
            self.last_valid_curvature = np.max(predicted_curvatures)
            self.last_valid_timestamp = current_time
            self.initialized = True

        if vision_status == VisionStatus.FULL_VISIBILITY and current_curvature is not None:
            self.last_valid_curvature = current_curvature
            self.last_valid_timestamp = current_time
            self.occlusion_start_time = None
            self.confidence_decay_factor = 1.0
            self.extrapolated_curvature = current_curvature
            self.initialized = True
            return current_curvature

        # Handle occlusion
        if self.occlusion_start_time is None:
            self.occlusion_start_time = current_time

        occlusion_duration = current_time - self.occlusion_start_time
        self.confidence_decay_factor = 0.5 ** (occlusion_duration / 2.0)

        if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
            # More conservative safety factor for blind corners
            safety_factor = 1.0 + 0.15 * min(occlusion_duration, 3.0)
            self.extrapolated_curvature = self.last_valid_curvature * safety_factor
        elif vision_status == VisionStatus.PARTIAL_OCCLUSION:
            if current_curvature is not None:
                blend_factor = self.confidence_decay_factor * 0.5
                self.extrapolated_curvature = (
                    blend_factor * current_curvature +
                    (1 - blend_factor) * self.last_valid_curvature * 1.1
                )
            else:
                self.extrapolated_curvature = self.last_valid_curvature * 1.15
        else:  # LOST_ROAD
            self.extrapolated_curvature = self.last_valid_curvature * 1.3

        return self.extrapolated_curvature


class ProgressiveDecelerationV4:
    """Iteration 4 with refined emergency level determination"""

    def __init__(self):
        self.current_level = EmergencyLevel.NORMAL
        self.target_level = EmergencyLevel.NORMAL
        self.level_transition_start = 0.0
        self.current_decel_limit = DECEL_LIMITS[EmergencyLevel.NORMAL]

        self.occlusion_state = VisionOcclusionStateV4()

        self.last_update_time = 0.0
        self.emergency_active_time = 0.0
        self.critical_level_time = 0.0
        self.intervention_level_time = 0.0

    def calculate_safe_speed_for_curve(self, curvature: float,
                                     lateral_acc_limit: float) -> float:
        """Calculate safe speed for given curvature"""
        if curvature <= 0:
            return 100.0
        return np.sqrt(lateral_acc_limit / curvature)

    def determine_required_deceleration(self, v_ego: float, v_target: float,
                                      distance: float) -> tuple[float, EmergencyLevel]:
        """Refined emergency level determination with better thresholds"""
        if distance <= 0 or v_ego <= v_target:
            return 0.0, EmergencyLevel.NORMAL

        # Basic kinematics
        required_decel = -(v_ego**2 - v_target**2) / (2 * distance)

        # Calculate required g-force
        required_g = abs(required_decel) / 9.81

        # Refined thresholds with consideration for distance
        # More lenient for longer distances
        distance_factor = min(distance / 50.0, 1.0)  # Normalize to 50m

        if required_g <= 0.15:
            level = EmergencyLevel.NORMAL
        elif required_g <= 0.22:  # Slightly tighter than 0.25
            level = EmergencyLevel.CAUTION
        elif required_g <= 0.35:  # Tighter than 0.40
            level = EmergencyLevel.WARNING
        elif required_g <= 0.55:  # Tighter than 0.60
            level = EmergencyLevel.CRITICAL
        else:
            # Only go to INTERVENTION if truly necessary
            # Consider distance - if we have more than 30m, try CRITICAL first
            if distance > 30 and required_g < 0.70:
                level = EmergencyLevel.CRITICAL
            else:
                level = EmergencyLevel.INTERVENTION

        return required_decel, level

    def get_transition_time(self, current: EmergencyLevel, target: EmergencyLevel) -> float:
        """Get appropriate transition time"""
        if target.value > current.value:
            # Escalating
            if current in TRANSITION_TIMES and target in TRANSITION_TIMES[current]:
                return TRANSITION_TIMES[current][target]
            # Default escalation time based on urgency
            level_diff = target.value - current.value
            return 0.3 + 0.1 * level_diff
        else:
            # De-escalating
            return TRANSITION_TIMES['de_escalate']

    def update(self,
               v_ego: float,
               current_curvature: float | None,
               predicted_curvatures: np.ndarray | None,
               distances: np.ndarray | None,
               vision_status: VisionStatus,
               model_confidence: float,
               lateral_acc_limit: float,
               current_time: float,
               actual_distance_to_curve: float = None) -> dict:
        """
        Main update with refined logic
        """

        dt = current_time - self.last_update_time if self.last_update_time > 0 else 0.05
        self.last_update_time = current_time

        # Update occlusion state
        extrapolated_curvature = self.occlusion_state.update(
            current_curvature, predicted_curvatures, vision_status, current_time
        )

        # Use extrapolated curvature if vision is degraded
        if vision_status != VisionStatus.FULL_VISIBILITY:
            planning_curvature = extrapolated_curvature
        else:
            planning_curvature = current_curvature if current_curvature is not None else 0.0

        # Calculate safe speed for current curvature
        v_safe_current = self.calculate_safe_speed_for_curve(
            planning_curvature, lateral_acc_limit
        )

        # Analyze predicted path
        critical_distance = actual_distance_to_curve if actual_distance_to_curve else 50.0
        v_safe_ahead = v_safe_current

        if predicted_curvatures is not None and distances is not None and len(predicted_curvatures) > 0:
            # Apply safety factor for blind corners
            if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
                # More conservative for blind corners
                safety_factor = 1.0 + 0.2 * (1 - self.occlusion_state.confidence_decay_factor)
                adjusted_curvatures = predicted_curvatures * safety_factor
            else:
                adjusted_curvatures = predicted_curvatures

            # Find the most critical point
            max_curve_idx = np.argmax(adjusted_curvatures)
            max_curvature = adjusted_curvatures[max_curve_idx]

            # If we have actual distance, use it
            if actual_distance_to_curve is not None:
                critical_distance = actual_distance_to_curve
            else:
                # Use the distance from prediction
                if max_curve_idx < len(distances):
                    critical_distance = distances[max_curve_idx]
                else:
                    critical_distance = distances[-1]

            # Target speed for critical curvature
            v_safe_ahead = self.calculate_safe_speed_for_curve(
                max_curvature, lateral_acc_limit * 0.9
            )

        # Determine target speed
        v_target = min(v_safe_current, v_safe_ahead)

        # Refined vision safety margin - less aggressive
        if vision_status == VisionStatus.LOST_ROAD:
            v_target *= 0.92
        elif vision_status == VisionStatus.CURVE_EXCEEDS_FOV and model_confidence < 0.3:
            v_target *= 0.96

        # Calculate required deceleration
        required_decel, required_level = self.determine_required_deceleration(
            v_ego, v_target, critical_distance
        )

        # Vision boost only for severe cases
        if vision_status == VisionStatus.LOST_ROAD and required_level == EmergencyLevel.CRITICAL:
            # Only boost if we're already at critical
            required_level = EmergencyLevel.CRITICAL

        # Update target level
        self.target_level = required_level

        # Handle level transitions with dynamic timing
        if self.target_level != self.current_level:
            if self.level_transition_start == 0:
                self.level_transition_start = current_time

            transition_time = self.get_transition_time(self.current_level, self.target_level)
            transition_progress = min(
                (current_time - self.level_transition_start) / transition_time,
                1.0
            )

            if transition_progress >= 1.0:
                self.current_level = self.target_level
                self.level_transition_start = 0.0
        else:
            self.level_transition_start = 0.0

        # Get deceleration limit
        current_limit = DECEL_LIMITS[self.current_level]

        # Apply jerk limiting
        max_jerk = JERK_LIMITS[self.current_level]
        max_change = max_jerk * dt

        if current_limit < self.current_decel_limit:
            self.current_decel_limit = max(
                current_limit,
                self.current_decel_limit - max_change
            )
        else:
            self.current_decel_limit = min(
                current_limit,
                self.current_decel_limit + max_change
            )

        # Track emergency duration
        if self.current_level.value >= EmergencyLevel.WARNING.value:
            self.emergency_active_time += dt
        else:
            self.emergency_active_time = 0.0

        if self.current_level == EmergencyLevel.CRITICAL:
            self.critical_level_time += dt
        else:
            self.critical_level_time = 0.0

        if self.current_level == EmergencyLevel.INTERVENTION:
            self.intervention_level_time += dt
        else:
            self.intervention_level_time = 0.0

        # Refined intervention logic - more conservative
        intervention_required = False

        if self.current_level == EmergencyLevel.INTERVENTION:
            # Only trigger intervention if we've been at this level for a bit
            # AND we truly can't handle it
            if self.intervention_level_time > 0.5 and required_decel < DECEL_LIMITS[EmergencyLevel.CRITICAL]:
                intervention_required = True
        elif self.current_level == EmergencyLevel.CRITICAL:
            # Only escalate to intervention if we've tried critical for a while
            # AND the required deceleration exceeds our critical capability
            if self.critical_level_time > 1.5 and required_decel < DECEL_LIMITS[EmergencyLevel.CRITICAL]:
                intervention_required = True

        return {
            'decel_limit': self.current_decel_limit,
            'emergency_level': self.current_level,
            'target_level': self.target_level,
            'target_speed': v_target,
            'required_decel': required_decel,
            'intervention_required': intervention_required,
            'extrapolated_curvature': extrapolated_curvature,
            'vision_degraded': vision_status != VisionStatus.FULL_VISIBILITY,
            'vision_confidence': self.occlusion_state.confidence_decay_factor,
            'emergency_duration': self.emergency_active_time,
            'critical_distance': critical_distance
        }


def test_iteration_4():
    """Test iteration 4 on problematic scenarios"""

    from emergency_scenarios_definition import EMERGENCY_SCENARIOS

    # Test the scenarios that failed in iteration 3
    test_scenarios = ['very_late_mountain_hairpin', 'early_warning_manageable']

    for scenario_name in test_scenarios:
        scenario = next(s for s in EMERGENCY_SCENARIOS if s.name == scenario_name)

        print(f"\nTesting Iteration 4 - {scenario_name}")
        print("="*60)
        print(f"Speed: {scenario.v_ego_kph:.0f} → {scenario.v_target_kph:.0f} km/h")
        print(f"Distance: {scenario.distance_to_curve_m:.0f}m")
        print(f"Required: {scenario.required_decel_g:.2f}g")

        controller = ProgressiveDecelerationV4()

        v_ego = scenario.v_ego_ms
        distance_traveled = 0.0
        time = 0.0
        dt = 0.05

        for i in range(100):  # 5 seconds max
            remaining_distance = scenario.distance_to_curve_m - distance_traveled

            if remaining_distance <= 0:
                print(f"Reached curve at t={time:.1f}s")
                break

            # Create predicted distances
            pred_distances = np.linspace(10, 100, 5)
            pred_curvatures = np.ones(5) * scenario.max_curvature

            result = controller.update(
                v_ego=v_ego,
                current_curvature=scenario.max_curvature if scenario.vision_status == VisionStatus.FULL_VISIBILITY else None,
                predicted_curvatures=pred_curvatures,
                distances=pred_distances,
                vision_status=scenario.vision_status,
                model_confidence=scenario.confidence,
                lateral_acc_limit=3.05,
                current_time=time,
                actual_distance_to_curve=remaining_distance
            )

            # Apply deceleration
            v_ego += result['decel_limit'] * dt
            v_ego = max(v_ego, 0)
            distance_traveled += v_ego * dt
            time += dt

            if i % 10 == 0:  # Every 0.5s
                print(f"t={time:.1f}s: v={v_ego*3.6:.0f}km/h, "
                      f"d_remain={remaining_distance:.0f}m, "
                      f"level={result['emergency_level'].name}, "
                      f"decel={result['decel_limit']/9.81:.2f}g")

            if result['intervention_required']:
                print(f"INTERVENTION at t={time:.1f}s")
                break

            if v_ego <= result['target_speed'] * 1.05:
                print(f"SUCCESS at t={time:.1f}s, v={v_ego*3.6:.0f}km/h")
                break


if __name__ == "__main__":
    test_iteration_4()
