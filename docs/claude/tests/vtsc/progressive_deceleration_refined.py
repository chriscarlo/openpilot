#!/usr/bin/env python3
"""
Refined Progressive Deceleration Controller

Final refinements for optimal balance:
1. More efficient deceleration usage
2. Better anticipatory control
3. Refined emergency thresholds
"""

import numpy as np
from dataclasses import dataclass

from emergency_scenarios_definition import (
    EmergencyLevel, VisionStatus
)


# Deceleration limits (m/s²)
# System constraint: Maximum -6.0 m/s²
DECEL_LIMITS = {
    EmergencyLevel.NORMAL: -1.47,      # 0.15g
    EmergencyLevel.CAUTION: -2.45,     # 0.25g
    EmergencyLevel.WARNING: -3.92,     # 0.40g
    EmergencyLevel.CRITICAL: -5.50,    # 0.56g
    EmergencyLevel.INTERVENTION: -6.00  # 0.61g - System maximum
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
class VisionOcclusionState:
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
        """Update occlusion state"""
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

        # Conservative safety factors
        if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
            safety_factor = 1.0 + 0.1 * min(occlusion_duration, 2.0)
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


class RefinedProgressiveDecelerationController:
    """Refined controller with optimal balance"""

    def __init__(self):
        self.current_level = EmergencyLevel.NORMAL
        self.current_decel = 0.0
        self.occlusion_state = VisionOcclusionState()

        self.last_update_time = 0.0
        self.time_at_current_level = 0.0
        self.critical_situation_time = 0.0

        # Smooth deceleration tracking
        self.decel_rate_of_change = 0.0

    def calculate_safe_speed_for_curve(self, curvature: float,
                                     lateral_acc_limit: float) -> float:
        """Calculate safe speed for given curvature"""
        if curvature <= 0:
            return 100.0
        return np.sqrt(lateral_acc_limit / curvature)

    def calculate_required_deceleration(self, v_ego: float, v_target: float,
                                      distance: float) -> float:
        """Calculate required deceleration with appropriate safety margin"""
        if distance <= 0 or v_ego <= v_target:
            return 0.0

        # Dynamic safety margin based on speed and distance
        if distance < 20:
            safety_margin = 0.9  # 10% margin for close distances
        elif distance < 40:
            safety_margin = 0.93  # 7% margin for medium distances
        else:
            safety_margin = 0.95  # 5% margin for far distances

        safety_distance = distance * safety_margin
        return -(v_ego**2 - v_target**2) / (2 * safety_distance)

    def determine_emergency_level(self, required_decel: float,
                                distance: float, v_ego: float) -> EmergencyLevel:
        """Refined emergency level determination"""
        required_g = abs(required_decel) / 9.81

        # Consider speed, distance, and required deceleration
        # High speeds need earlier intervention
        speed_factor = min(v_ego / 30.0, 1.0)  # Normalize to 30 m/s

        if distance > 60:
            # Far away - very relaxed thresholds
            if required_g <= 0.20:
                return EmergencyLevel.NORMAL
            elif required_g <= 0.30:
                return EmergencyLevel.CAUTION
            elif required_g <= 0.50:
                return EmergencyLevel.WARNING
            elif required_g <= 0.70:
                return EmergencyLevel.CRITICAL
            else:
                return EmergencyLevel.INTERVENTION
        elif distance > 40:
            # Medium distance - balanced thresholds
            if required_g <= 0.17:
                return EmergencyLevel.NORMAL
            elif required_g <= 0.27:
                return EmergencyLevel.CAUTION
            elif required_g <= 0.42:
                return EmergencyLevel.WARNING
            elif required_g <= 0.62:
                return EmergencyLevel.CRITICAL
            else:
                return EmergencyLevel.INTERVENTION
        else:
            # Close - standard thresholds with speed consideration
            base_thresholds = [0.15, 0.25, 0.40, 0.60]
            # Slightly lower thresholds at high speed
            adjusted_thresholds = [t * (1 - 0.1 * speed_factor) for t in base_thresholds]

            if required_g <= adjusted_thresholds[0]:
                return EmergencyLevel.NORMAL
            elif required_g <= adjusted_thresholds[1]:
                return EmergencyLevel.CAUTION
            elif required_g <= adjusted_thresholds[2]:
                return EmergencyLevel.WARNING
            elif required_g <= adjusted_thresholds[3]:
                return EmergencyLevel.CRITICAL
            else:
                return EmergencyLevel.INTERVENTION

    def get_optimal_deceleration(self, level: EmergencyLevel,
                               required_decel: float,
                               current_decel: float) -> float:
        """Get optimal deceleration - only what's needed"""
        level_limit = DECEL_LIMITS[level]

        # Use proportional control within the level
        if abs(required_decel) <= abs(level_limit):
            # Add small buffer for control stability
            buffer_factor = 1.1
            target = required_decel * buffer_factor
            # But don't exceed level limit
            if abs(target) > abs(level_limit):
                return level_limit
            return target
        else:
            # Need full capability of this level
            return level_limit

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
        """Main update function"""

        dt = current_time - self.last_update_time if self.last_update_time > 0 else 0.05
        self.last_update_time = current_time
        self.time_at_current_level += dt

        # Update occlusion state
        extrapolated_curvature = self.occlusion_state.update(
            current_curvature, predicted_curvatures, vision_status, current_time
        )

        # Determine planning curvature
        if vision_status != VisionStatus.FULL_VISIBILITY:
            planning_curvature = extrapolated_curvature
        else:
            planning_curvature = current_curvature if current_curvature is not None else 0.0

        # Calculate safe speeds
        v_safe_current = self.calculate_safe_speed_for_curve(
            planning_curvature, lateral_acc_limit
        )

        # Analyze predicted path
        critical_distance = actual_distance_to_curve if actual_distance_to_curve else 50.0
        v_safe_ahead = v_safe_current

        if predicted_curvatures is not None and distances is not None and len(predicted_curvatures) > 0:
            # Minimal safety factor for blind corners
            if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
                safety_factor = 1.0 + 0.1 * (1 - self.occlusion_state.confidence_decay_factor)
                adjusted_curvatures = predicted_curvatures * safety_factor
            else:
                adjusted_curvatures = predicted_curvatures

            # Find most critical curvature
            max_curve_idx = np.argmax(adjusted_curvatures)
            max_curvature = adjusted_curvatures[max_curve_idx]

            if actual_distance_to_curve is not None:
                critical_distance = actual_distance_to_curve

            # Target speed with small margin
            v_safe_ahead = self.calculate_safe_speed_for_curve(
                max_curvature, lateral_acc_limit * 0.97
            )

        # Determine target speed
        v_target = min(v_safe_current, v_safe_ahead)

        # Minimal vision safety margins
        if vision_status == VisionStatus.LOST_ROAD:
            v_target *= 0.95
        elif vision_status == VisionStatus.CURVE_EXCEEDS_FOV and model_confidence < 0.5:
            v_target *= 0.98

        # Calculate required deceleration
        required_decel = self.calculate_required_deceleration(
            v_ego, v_target, critical_distance
        )

        # Determine appropriate emergency level
        target_level = self.determine_emergency_level(
            required_decel, critical_distance, v_ego
        )

        # Smooth level transitions with appropriate hysteresis
        if target_level != self.current_level:
            should_transition = False

            if target_level.value > self.current_level.value:
                # Escalating - quick response
                if self.time_at_current_level > 0.15:
                    should_transition = True
            else:
                # De-escalating - wait for stability
                if self.time_at_current_level > 2.0:
                    should_transition = True

            if should_transition:
                self.current_level = target_level
                self.time_at_current_level = 0.0

        # Get optimal deceleration
        target_decel = self.get_optimal_deceleration(
            self.current_level, required_decel, self.current_decel
        )

        # Apply smooth jerk limiting
        max_jerk = JERK_LIMITS[self.current_level]
        max_change = max_jerk * dt

        # Track rate of change for smoother control
        decel_error = target_decel - self.current_decel

        if abs(decel_error) > max_change:
            # Apply jerk limit
            if decel_error < 0:
                self.current_decel -= max_change
            else:
                self.current_decel += max_change
        else:
            # Small changes - apply directly for responsiveness
            self.current_decel = target_decel

        # Track critical situations
        if abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]):
            self.critical_situation_time += dt
        else:
            self.critical_situation_time = 0.0

        # Refined intervention logic
        intervention_required = False

        # Only intervene when physics truly demands it
        if (abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]) * 1.05 and
            self.critical_situation_time > 0.3 and
            self.current_level == EmergencyLevel.CRITICAL and
            critical_distance < 25):  # Only if close
            intervention_required = True

        return {
            'decel_limit': self.current_decel,
            'emergency_level': self.current_level,
            'target_level': target_level,
            'target_speed': v_target,
            'required_decel': required_decel,
            'intervention_required': intervention_required,
            'extrapolated_curvature': extrapolated_curvature,
            'vision_degraded': vision_status != VisionStatus.FULL_VISIBILITY,
            'vision_confidence': self.occlusion_state.confidence_decay_factor,
            'critical_distance': critical_distance,
            'time_at_level': self.time_at_current_level
        }


def test_refined_controller():
    """Test refined controller"""

    from emergency_scenarios_definition import EMERGENCY_SCENARIOS

    test_scenarios = ['early_warning_manageable', 'late_highway_curve', 'very_late_mountain_hairpin']

    for scenario_name in test_scenarios:
        scenario = next(s for s in EMERGENCY_SCENARIOS if s.name == scenario_name)

        print(f"\nTesting Refined Controller - {scenario_name}")
        print("="*60)
        print(f"Speed: {scenario.v_ego_kph:.0f} → {scenario.v_target_kph:.0f} km/h")
        print(f"Distance: {scenario.distance_to_curve_m:.0f}m")
        print(f"Required: {scenario.required_decel_g:.2f}g")

        controller = RefinedProgressiveDecelerationController()

        v_ego = scenario.v_ego_ms
        distance_traveled = 0.0
        time = 0.0
        dt = 0.05
        max_decel = 0.0

        for i in range(100):  # 5 seconds max
            remaining_distance = scenario.distance_to_curve_m - distance_traveled

            if remaining_distance <= 0:
                print(f"Reached curve at t={time:.1f}s")
                break

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

            max_decel = min(max_decel, result['decel_limit'])

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
                print(f"Max decel used: {abs(max_decel)/9.81:.2f}g")
                break


if __name__ == "__main__":
    test_refined_controller()
