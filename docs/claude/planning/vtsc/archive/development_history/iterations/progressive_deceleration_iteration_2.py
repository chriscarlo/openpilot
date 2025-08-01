#!/usr/bin/env python3
"""
Iteration 2: Progressive Deceleration System with bug fixes

Fixed issues:
1. Use actual critical distance, not just first predicted distance
2. Properly initialize occlusion state when vision is degraded
3. Better distance handling for emergency determination
"""

import numpy as np
from dataclasses import dataclass

from emergency_scenarios_definition import (
    EmergencyLevel, VisionStatus
)


# Proven deceleration limits (m/s²)
DECEL_LIMITS = {
    EmergencyLevel.NORMAL: -1.47,      # 0.15g - Comfortable
    EmergencyLevel.CAUTION: -2.45,     # 0.25g - Slightly uncomfortable
    EmergencyLevel.WARNING: -3.92,     # 0.40g - Noticeably uncomfortable
    EmergencyLevel.CRITICAL: -5.89,    # 0.60g - Emergency braking
    EmergencyLevel.INTERVENTION: -7.85  # 0.80g - Maximum system capability
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
class VisionOcclusionStateV2:
    """Fixed vision occlusion tracking"""
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

        # Initialize from predictions if we haven't seen good data yet
        if not self.initialized and predicted_curvatures is not None and len(predicted_curvatures) > 0:
            # Use the maximum predicted curvature as initial estimate
            self.last_valid_curvature = np.max(predicted_curvatures)
            self.last_valid_timestamp = current_time
            self.initialized = True

        if vision_status == VisionStatus.FULL_VISIBILITY and current_curvature is not None:
            # Good visibility - update last known values
            self.last_valid_curvature = current_curvature
            self.last_valid_timestamp = current_time
            self.occlusion_start_time = None
            self.confidence_decay_factor = 1.0
            self.extrapolated_curvature = current_curvature
            self.initialized = True
            return current_curvature

        # Handle occlusion cases
        if self.occlusion_start_time is None:
            self.occlusion_start_time = current_time

        occlusion_duration = current_time - self.occlusion_start_time

        # Confidence decay
        self.confidence_decay_factor = 0.5 ** (occlusion_duration / 2.0)

        if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
            # For blind corners, assume moderate worsening
            safety_factor = 1.0 + 0.2 * min(occlusion_duration, 3.0)
            self.extrapolated_curvature = self.last_valid_curvature * safety_factor

        elif vision_status == VisionStatus.PARTIAL_OCCLUSION:
            if current_curvature is not None:
                blend_factor = self.confidence_decay_factor * 0.5
                self.extrapolated_curvature = (
                    blend_factor * current_curvature +
                    (1 - blend_factor) * self.last_valid_curvature * 1.1
                )
            else:
                self.extrapolated_curvature = self.last_valid_curvature * 1.2

        else:  # LOST_ROAD
            self.extrapolated_curvature = self.last_valid_curvature * 1.5

        return self.extrapolated_curvature


class ProgressiveDecelerationV2:
    """Fixed progressive deceleration controller"""

    def __init__(self):
        self.current_level = EmergencyLevel.NORMAL
        self.target_level = EmergencyLevel.NORMAL
        self.level_transition_start = 0.0
        self.current_decel_limit = DECEL_LIMITS[EmergencyLevel.NORMAL]

        self.occlusion_state = VisionOcclusionStateV2()

        self.last_update_time = 0.0
        self.emergency_active_time = 0.0
        self.critical_level_time = 0.0

    def calculate_safe_speed_for_curve(self, curvature: float,
                                     lateral_acc_limit: float) -> float:
        """Calculate safe speed for given curvature"""
        if curvature <= 0:
            return 100.0  # High speed for straight road
        return np.sqrt(lateral_acc_limit / curvature)

    def determine_required_deceleration(self, v_ego: float, v_target: float,
                                      distance: float) -> tuple[float, EmergencyLevel]:
        """Determine required deceleration and emergency level"""
        if distance <= 0 or v_ego <= v_target:
            return 0.0, EmergencyLevel.NORMAL

        # Basic kinematics
        required_decel = -(v_ego**2 - v_target**2) / (2 * distance)

        # Determine emergency level
        required_g = abs(required_decel) / 9.81

        if required_g <= 0.15:
            level = EmergencyLevel.NORMAL
        elif required_g <= 0.25:
            level = EmergencyLevel.CAUTION
        elif required_g <= 0.40:
            level = EmergencyLevel.WARNING
        elif required_g <= 0.60:
            level = EmergencyLevel.CRITICAL
        else:
            level = EmergencyLevel.INTERVENTION

        return required_decel, level

    def update(self,
               v_ego: float,
               current_curvature: float | None,
               predicted_curvatures: np.ndarray | None,
               distances: np.ndarray | None,
               vision_status: VisionStatus,
               model_confidence: float,
               lateral_acc_limit: float,
               current_time: float) -> dict:
        """Main update with fixed distance handling"""

        dt = current_time - self.last_update_time if self.last_update_time > 0 else 0.05
        self.last_update_time = current_time

        # Update occlusion state with predicted curvatures for initialization
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
        critical_distance = 50.0  # Default
        v_safe_ahead = v_safe_current

        if predicted_curvatures is not None and distances is not None and len(predicted_curvatures) > 0:
            # Apply safety factor for blind corners
            if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
                safety_factor = 1.0 + 0.3 * (1 - self.occlusion_state.confidence_decay_factor)
                adjusted_curvatures = predicted_curvatures * safety_factor
            else:
                adjusted_curvatures = predicted_curvatures

            # Find the most critical point (highest curvature)
            max_curve_idx = np.argmax(adjusted_curvatures)
            max_curvature = adjusted_curvatures[max_curve_idx]

            # Use the actual distance to that critical point
            if max_curve_idx < len(distances):
                critical_distance = distances[max_curve_idx]
            else:
                critical_distance = distances[-1]  # Use furthest distance

            # Ensure we have reasonable minimum distance
            critical_distance = max(critical_distance, 20.0)

            # Target speed for critical curvature
            v_safe_ahead = self.calculate_safe_speed_for_curve(
                max_curvature, lateral_acc_limit * 0.9
            )

        # Determine target speed
        v_target = min(v_safe_current, v_safe_ahead)

        # Apply minimal vision safety margin
        if vision_status == VisionStatus.LOST_ROAD:
            v_target *= 0.9
        elif vision_status == VisionStatus.CURVE_EXCEEDS_FOV and model_confidence < 0.3:
            v_target *= 0.95

        # Calculate required deceleration using the actual critical distance
        required_decel, required_level = self.determine_required_deceleration(
            v_ego, v_target, critical_distance
        )

        # Vision boost only when necessary
        if vision_status == VisionStatus.LOST_ROAD and required_level.value >= EmergencyLevel.WARNING.value:
            required_level = EmergencyLevel(min(
                required_level.value + 1,
                EmergencyLevel.CRITICAL.value
            ))

        # Update target level
        self.target_level = required_level

        # Handle level transitions
        if self.target_level != self.current_level:
            if self.level_transition_start == 0:
                self.level_transition_start = current_time

            # Transition timing
            if self.target_level.value > self.current_level.value:
                transition_time = 0.5
            else:
                transition_time = 2.0

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

        # Intervention logic
        intervention_required = False

        if self.current_level == EmergencyLevel.INTERVENTION:
            intervention_required = True
        elif self.current_level == EmergencyLevel.CRITICAL:
            # Only if we truly can't handle it
            if required_decel < DECEL_LIMITS[EmergencyLevel.CRITICAL] and self.critical_level_time > 1.0:
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


def test_iteration_2():
    """Test the fixed controller"""

    from emergency_scenarios_definition import EMERGENCY_SCENARIOS

    # Test blind hairpin scenario
    scenario = next(s for s in EMERGENCY_SCENARIOS if s.name == 'blind_hairpin_exceeds_fov')

    print("Testing Iteration 2 - blind_hairpin_exceeds_fov")
    print("="*60)

    controller = ProgressiveDecelerationV2()

    v_ego = scenario.v_ego_ms
    distance_traveled = 0.0
    time = 0.0
    dt = 0.05

    for i in range(80):  # 4 seconds
        remaining_distance = scenario.distance_to_curve_m - distance_traveled

        if remaining_distance <= 0:
            print(f"Reached curve at t={time:.1f}s")
            break

        # Create realistic predicted distances
        pred_distances = np.linspace(
            min(10, remaining_distance),
            min(remaining_distance + 20, 100),
            5
        )
        pred_curvatures = np.ones(5) * scenario.max_curvature

        result = controller.update(
            v_ego=v_ego,
            current_curvature=None,  # Can't see (blind corner)
            predicted_curvatures=pred_curvatures,
            distances=pred_distances,
            vision_status=scenario.vision_status,
            model_confidence=scenario.confidence,
            lateral_acc_limit=3.05,
            current_time=time
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
                  f"crit_d={result['critical_distance']:.0f}m")

        if result['intervention_required']:
            print(f"INTERVENTION at t={time:.1f}s")
            break

        if v_ego <= result['target_speed'] * 1.05:
            print(f"SUCCESS at t={time:.1f}s, v={v_ego*3.6:.0f}km/h")
            break


if __name__ == "__main__":
    test_iteration_2()
