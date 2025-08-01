#!/usr/bin/env python3
"""
Iteration 5: Progressive Deceleration with demand-based approach

Key improvements:
1. Use exact deceleration needed (not just the level's max)
2. Only escalate when current level insufficient
3. Smoother transitions
4. Better intervention logic based on actual capability
"""

import numpy as np
from dataclasses import dataclass

from emergency_scenarios_definition import (
    EmergencyLevel, VisionStatus
)


# Deceleration limits (m/s²) - maximum for each level
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


@dataclass
class VisionOcclusionStateV5:
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
            # Conservative for blind corners
            safety_factor = 1.0 + 0.1 * min(occlusion_duration, 3.0)
            self.extrapolated_curvature = self.last_valid_curvature * safety_factor
        elif vision_status == VisionStatus.PARTIAL_OCCLUSION:
            if current_curvature is not None:
                blend_factor = self.confidence_decay_factor * 0.5
                self.extrapolated_curvature = (
                    blend_factor * current_curvature +
                    (1 - blend_factor) * self.last_valid_curvature * 1.05
                )
            else:
                self.extrapolated_curvature = self.last_valid_curvature * 1.1
        else:  # LOST_ROAD
            self.extrapolated_curvature = self.last_valid_curvature * 1.2

        return self.extrapolated_curvature


class ProgressiveDecelerationV5:
    """Iteration 5 with demand-based deceleration"""

    def __init__(self):
        self.current_level = EmergencyLevel.NORMAL
        self.current_decel = 0.0  # Actual deceleration being used
        self.occlusion_state = VisionOcclusionStateV5()

        self.last_update_time = 0.0
        self.emergency_active_time = 0.0
        self.critical_active_time = 0.0

        # Smooth level transitions
        self.level_transition_progress = 0.0
        self.transitioning_to = None
        self.transition_start_time = 0.0

    def calculate_safe_speed_for_curve(self, curvature: float,
                                     lateral_acc_limit: float) -> float:
        """Calculate safe speed for given curvature"""
        if curvature <= 0:
            return 100.0
        return np.sqrt(lateral_acc_limit / curvature)

    def calculate_required_deceleration(self, v_ego: float, v_target: float,
                                      distance: float) -> float:
        """Calculate required deceleration"""
        if distance <= 0 or v_ego <= v_target:
            return 0.0

        # Basic kinematics with small safety margin
        safety_distance = max(distance - 5.0, distance * 0.9)  # 5m or 10% margin
        return -(v_ego**2 - v_target**2) / (2 * safety_distance)

    def determine_required_level(self, required_decel: float) -> EmergencyLevel:
        """Determine minimum level needed for required deceleration"""
        required_abs = abs(required_decel)

        # Find minimum level that can provide this deceleration
        for level in EmergencyLevel:
            if required_abs <= abs(DECEL_LIMITS[level]):
                return level

        return EmergencyLevel.INTERVENTION

    def get_effective_decel_limit(self, dt: float) -> float:
        """Get current effective deceleration limit with smooth transitions"""
        base_limit = DECEL_LIMITS[self.current_level]

        if self.transitioning_to is not None:
            target_limit = DECEL_LIMITS[self.transitioning_to]
            # Smooth interpolation
            effective_limit = (
                base_limit * (1 - self.level_transition_progress) +
                target_limit * self.level_transition_progress
            )
            return effective_limit

        return base_limit

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
        """Main update with demand-based approach"""

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
            # Apply minimal safety factor for blind corners
            if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
                safety_factor = 1.0 + 0.1 * (1 - self.occlusion_state.confidence_decay_factor)
                adjusted_curvatures = predicted_curvatures * safety_factor
            else:
                adjusted_curvatures = predicted_curvatures

            # Find the most critical point
            max_curve_idx = np.argmax(adjusted_curvatures)
            max_curvature = adjusted_curvatures[max_curve_idx]

            # Use actual distance if provided
            if actual_distance_to_curve is not None:
                critical_distance = actual_distance_to_curve

            # Target speed for critical curvature
            v_safe_ahead = self.calculate_safe_speed_for_curve(
                max_curvature, lateral_acc_limit * 0.95
            )

        # Determine target speed
        v_target = min(v_safe_current, v_safe_ahead)

        # Minimal vision safety margin
        if vision_status == VisionStatus.LOST_ROAD:
            v_target *= 0.95
        elif vision_status == VisionStatus.CURVE_EXCEEDS_FOV and model_confidence < 0.3:
            v_target *= 0.97

        # Calculate required deceleration
        required_decel = self.calculate_required_deceleration(
            v_ego, v_target, critical_distance
        )

        # Determine required emergency level
        required_level = self.determine_required_level(required_decel)

        # Handle level transitions
        if required_level != self.current_level:
            if self.transitioning_to != required_level:
                # Start new transition
                self.transitioning_to = required_level
                self.transition_start_time = current_time
                self.level_transition_progress = 0.0

            # Update transition progress
            if required_level.value > self.current_level.value:
                # Escalating - faster transition
                transition_time = 0.3
            else:
                # De-escalating - slower transition
                transition_time = 1.5

            self.level_transition_progress = min(
                (current_time - self.transition_start_time) / transition_time,
                1.0
            )

            if self.level_transition_progress >= 1.0:
                # Complete transition
                self.current_level = self.transitioning_to
                self.transitioning_to = None
                self.level_transition_progress = 0.0
        else:
            # No transition needed
            self.transitioning_to = None
            self.level_transition_progress = 0.0

        # Get effective deceleration limit
        effective_limit = self.get_effective_decel_limit(dt)

        # Use only the deceleration we need (up to the limit)
        if required_decel < effective_limit:
            # Need more than current limit provides
            target_decel = effective_limit
        else:
            # Use exactly what we need
            target_decel = max(required_decel, -1.0)  # At least -1.0 m/s² when decelerating

        # Apply jerk limiting for smooth changes
        max_jerk = JERK_LIMITS[self.current_level]
        max_change = max_jerk * dt

        if target_decel < self.current_decel:
            self.current_decel = max(
                target_decel,
                self.current_decel - max_change
            )
        else:
            self.current_decel = min(
                target_decel,
                self.current_decel + max_change
            )

        # Track emergency duration
        if self.current_level.value >= EmergencyLevel.WARNING.value:
            self.emergency_active_time += dt
        else:
            self.emergency_active_time = 0.0

        if self.current_level == EmergencyLevel.CRITICAL:
            self.critical_active_time += dt
        else:
            self.critical_active_time = 0.0

        # Intervention logic - only when physics demands it
        intervention_required = False

        # Check if we truly need intervention
        if required_level == EmergencyLevel.INTERVENTION:
            # We need more than 0.60g deceleration
            if self.current_level == EmergencyLevel.INTERVENTION:
                # Already at intervention level
                if abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]):
                    # Still need more than critical can provide
                    intervention_required = True
            elif self.current_level == EmergencyLevel.CRITICAL and self.critical_active_time > 1.0:
                # Been at critical for a while and still need more
                if abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]):
                    intervention_required = True

        return {
            'decel_limit': self.current_decel,
            'emergency_level': self.current_level,
            'target_level': required_level,
            'target_speed': v_target,
            'required_decel': required_decel,
            'intervention_required': intervention_required,
            'extrapolated_curvature': extrapolated_curvature,
            'vision_degraded': vision_status != VisionStatus.FULL_VISIBILITY,
            'vision_confidence': self.occlusion_state.confidence_decay_factor,
            'emergency_duration': self.emergency_active_time,
            'critical_distance': critical_distance,
            'transition_progress': self.level_transition_progress
        }


def test_iteration_5():
    """Test iteration 5 on key scenarios"""

    from emergency_scenarios_definition import EMERGENCY_SCENARIOS

    # Test scenarios that have been problematic
    test_scenarios = ['early_warning_manageable', 'very_late_mountain_hairpin', 'blind_hairpin_exceeds_fov']

    for scenario_name in test_scenarios:
        scenario = next(s for s in EMERGENCY_SCENARIOS if s.name == scenario_name)

        print(f"\nTesting Iteration 5 - {scenario_name}")
        print("="*60)
        print(f"Speed: {scenario.v_ego_kph:.0f} → {scenario.v_target_kph:.0f} km/h")
        print(f"Distance: {scenario.distance_to_curve_m:.0f}m")
        print(f"Required: {scenario.required_decel_g:.2f}g")

        controller = ProgressiveDecelerationV5()

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
                      f"decel={result['decel_limit']/9.81:.2f}g, "
                      f"req={abs(result['required_decel'])/9.81:.2f}g")

            if result['intervention_required']:
                print(f"INTERVENTION at t={time:.1f}s")
                break

            if v_ego <= result['target_speed'] * 1.05:
                print(f"SUCCESS at t={time:.1f}s, v={v_ego*3.6:.0f}km/h")
                break


if __name__ == "__main__":
    test_iteration_5()
