#!/usr/bin/env python3
"""
Enhanced Vision Turn Speed Controller - Integrated System

Combines:
1. Physics-based anticipatory deceleration
2. Progressive emergency deceleration
3. Vision occlusion handling
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum


# Emergency levels for progressive deceleration
class EmergencyLevel(IntEnum):
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
    INTERVENTION = 4


# Vision status types
class VisionStatus(IntEnum):
    FULL_VISIBILITY = 0
    PARTIAL_OCCLUSION = 1
    CURVE_EXCEEDS_FOV = 2
    LOST_ROAD = 3


# Deceleration limits for each emergency level (m/s²)
# Note: Longitudinal planner limits to 5.5-6.0 m/s², so INTERVENTION is capped
DECEL_LIMITS = {
    EmergencyLevel.NORMAL: -1.47,      # 0.15g - Comfortable
    EmergencyLevel.CAUTION: -2.45,     # 0.25g - Slightly uncomfortable
    EmergencyLevel.WARNING: -3.92,     # 0.40g - Noticeably uncomfortable
    EmergencyLevel.CRITICAL: -5.50,    # 0.56g - Near system limit
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

# Constants
MIN_SAFE_DISTANCE = 10.0  # meters
MIN_ANTICIPATION_TIME = 1.0  # seconds
MAX_ANTICIPATION_TIME = 3.0  # seconds


@dataclass
class VisionOcclusionState:
    """Tracks vision occlusion and extrapolates curvature"""
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
        """Update occlusion state and return extrapolated curvature"""
        self.vision_status = vision_status

        # Initialize from predictions if needed
        if not self.initialized and predicted_curvatures is not None and len(predicted_curvatures) > 0:
            self.last_valid_curvature = np.max(predicted_curvatures)
            self.last_valid_timestamp = current_time
            self.initialized = True

        if vision_status == VisionStatus.FULL_VISIBILITY and current_curvature is not None:
            # Good visibility - update state
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

        # Apply appropriate safety factors
        if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
            # 5% safety margin for blind corners (was 10-20%)
            safety_factor = 1.0 + 0.05 * min(occlusion_duration, 2.0)
            self.extrapolated_curvature = self.last_valid_curvature * safety_factor
        elif vision_status == VisionStatus.PARTIAL_OCCLUSION:
            if current_curvature is not None:
                blend_factor = self.confidence_decay_factor * 0.5
                self.extrapolated_curvature = (
                    blend_factor * current_curvature +
                    (1 - blend_factor) * self.last_valid_curvature * 1.05
                )
            else:
                self.extrapolated_curvature = self.last_valid_curvature * 1.05
        else:  # LOST_ROAD
            self.extrapolated_curvature = self.last_valid_curvature * 1.1

        return self.extrapolated_curvature


class EnhancedVisionTurnSpeedController:
    """Enhanced VTSC with anticipatory and emergency deceleration"""

    def __init__(self):
        # Emergency deceleration state
        self.current_level = EmergencyLevel.NORMAL
        self.current_decel = 0.0
        self.time_at_current_level = 0.0
        self.critical_situation_time = 0.0

        # Vision handling
        self.occlusion_state = VisionOcclusionState()

        # Timing
        self.last_update_time = 0.0

        # Anticipatory control parameters
        self.anticipation_time = 2.0  # seconds to reach target speed before physically necessary

    def calculate_safe_speed_for_curve(self, curvature: float,
                                     lateral_acc_limit: float) -> float:
        """Calculate safe speed for given curvature"""
        if curvature <= 0:
            return 100.0  # m/s (effectively no limit)
        return np.sqrt(lateral_acc_limit / curvature)

    def calculate_anticipation_distance(self, v_ego: float, v_target: float,
                                      distance_to_critical: float,
                                      comfort_decel_g: float = 0.15) -> float:
        """
        Calculate where to start deceleration for anticipatory control
        Returns the distance before the curve to begin deceleration
        """
        if v_ego <= v_target:
            return 0.0

        # Convert g to m/s²
        comfort_decel = comfort_decel_g * 9.81

        # Distance needed to decelerate comfortably
        decel_distance = (v_ego**2 - v_target**2) / (2 * comfort_decel)

        # Time at target speed before curve
        anticipation_time = np.clip(self.anticipation_time,
                                   MIN_ANTICIPATION_TIME,
                                   MAX_ANTICIPATION_TIME)
        anticipation_distance = v_target * anticipation_time

        # Total distance needed
        total_distance_needed = decel_distance + anticipation_distance + MIN_SAFE_DISTANCE

        # Start deceleration when we're this far from the curve
        return total_distance_needed

    def calculate_required_deceleration(self, v_ego: float, v_target: float,
                                      distance: float) -> float:
        """Calculate required deceleration with safety margin"""
        if distance <= 0 or v_ego <= v_target:
            return 0.0

        # Dynamic safety margin
        if distance < 20:
            safety_margin = 0.9
        elif distance < 40:
            safety_margin = 0.93
        else:
            safety_margin = 0.95

        safety_distance = distance * safety_margin
        return -(v_ego**2 - v_target**2) / (2 * safety_distance)

    def determine_emergency_level(self, required_decel: float,
                                distance: float, v_ego: float) -> EmergencyLevel:
        """Determine appropriate emergency level"""
        required_g = abs(required_decel) / 9.81

        # Consider speed factor
        speed_factor = min(v_ego / 30.0, 1.0)

        if distance > 60:
            # Far - relaxed thresholds
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
            # Medium - balanced thresholds
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
            # Close - standard thresholds
            base_thresholds = [0.15, 0.25, 0.40, 0.60]
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
                               required_decel: float) -> float:
        """Get optimal deceleration for level"""
        level_limit = DECEL_LIMITS[level]

        # Use only what's needed with small buffer
        if abs(required_decel) <= abs(level_limit):
            buffer_factor = 1.1
            target = required_decel * buffer_factor
            if abs(target) > abs(level_limit):
                return level_limit
            return target
        else:
            return level_limit

    def update(self,
               v_ego: float,
               current_curvature: float | None,
               predicted_curvatures: np.ndarray | None,
               distances: np.ndarray | None,
               lateral_acc_limit: float,
               model_confidence: float,
               current_time: float) -> dict:
        """
        Main update function
        
        Args:
            v_ego: Current vehicle speed (m/s)
            current_curvature: Current road curvature (1/m)
            predicted_curvatures: Array of predicted curvatures
            distances: Array of distances to predicted curvatures
            lateral_acc_limit: Maximum lateral acceleration (m/s²)
            model_confidence: Vision model confidence (0-1)
            current_time: Current time (seconds)
            
        Returns:
            dict with:
                - a_target: Target acceleration (m/s²)
                - v_target: Target speed (m/s)
                - emergency_level: Current emergency level
                - distance_to_curve: Distance to critical curve (m)
                - using_anticipation: Whether using anticipatory control
                - intervention_required: Whether driver intervention needed
        """

        dt = current_time - self.last_update_time if self.last_update_time > 0 else 0.05
        self.last_update_time = current_time
        self.time_at_current_level += dt

        # Determine vision status based on confidence and curvature
        if model_confidence < 0.3:
            vision_status = VisionStatus.LOST_ROAD
        elif model_confidence < 0.6 and current_curvature is None:
            vision_status = VisionStatus.PARTIAL_OCCLUSION
        elif current_curvature is None:
            vision_status = VisionStatus.CURVE_EXCEEDS_FOV
        else:
            vision_status = VisionStatus.FULL_VISIBILITY

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
        critical_distance = 100.0  # Default far distance
        v_safe_ahead = v_safe_current
        max_curvature = planning_curvature

        if predicted_curvatures is not None and distances is not None and len(predicted_curvatures) > 0:
            # Apply vision safety factors
            if vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
                safety_factor = 1.0 + 0.05 * (1 - self.occlusion_state.confidence_decay_factor)
                adjusted_curvatures = predicted_curvatures * safety_factor
            else:
                adjusted_curvatures = predicted_curvatures

            # Find most critical curvature
            max_curve_idx = np.argmax(adjusted_curvatures)
            max_curvature = adjusted_curvatures[max_curve_idx]

            if max_curve_idx < len(distances):
                critical_distance = distances[max_curve_idx]

            # Target speed for critical curvature
            v_safe_ahead = self.calculate_safe_speed_for_curve(
                max_curvature, lateral_acc_limit * 0.97
            )

        # Determine target speed
        v_target_physics = min(v_safe_current, v_safe_ahead)

        # Apply vision safety margins
        if vision_status == VisionStatus.LOST_ROAD:
            v_target_physics *= 0.95
        elif vision_status == VisionStatus.CURVE_EXCEEDS_FOV and model_confidence < 0.5:
            v_target_physics *= 0.98

        # Calculate anticipatory deceleration point
        anticipation_distance = self.calculate_anticipation_distance(
            v_ego, v_target_physics, critical_distance
        )

        # Check if we should use anticipatory control
        using_anticipation = critical_distance <= anticipation_distance

        # Calculate required deceleration
        if using_anticipation:
            # Use remaining distance for deceleration
            remaining_distance = critical_distance
        else:
            # Not yet time for anticipatory control
            remaining_distance = critical_distance

        required_decel = self.calculate_required_deceleration(
            v_ego, v_target_physics, remaining_distance
        )

        # Determine emergency level
        target_level = self.determine_emergency_level(
            required_decel, remaining_distance, v_ego
        )

        # Handle level transitions
        if target_level != self.current_level:
            should_transition = False

            if target_level.value > self.current_level.value:
                # Escalating
                if self.time_at_current_level > 0.15:
                    should_transition = True
            else:
                # De-escalating
                if self.time_at_current_level > 2.0:
                    should_transition = True

            if should_transition:
                self.current_level = target_level
                self.time_at_current_level = 0.0

        # Get optimal deceleration
        target_decel = self.get_optimal_deceleration(self.current_level, required_decel)

        # Apply jerk limiting
        max_jerk = JERK_LIMITS[self.current_level]
        max_change = max_jerk * dt

        decel_error = target_decel - self.current_decel

        if abs(decel_error) > max_change:
            if decel_error < 0:
                self.current_decel -= max_change
            else:
                self.current_decel += max_change
        else:
            self.current_decel = target_decel

        # Track critical situations
        if abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]):
            self.critical_situation_time += dt
        else:
            self.critical_situation_time = 0.0

        # Intervention logic
        intervention_required = False

        if (abs(required_decel) > abs(DECEL_LIMITS[EmergencyLevel.CRITICAL]) * 1.05 and
            self.critical_situation_time > 0.3 and
            self.current_level == EmergencyLevel.CRITICAL and
            remaining_distance < 25):
            intervention_required = True

        return {
            'a_target': self.current_decel,
            'v_target': v_target_physics,
            'emergency_level': self.current_level,
            'distance_to_curve': critical_distance,
            'using_anticipation': using_anticipation,
            'intervention_required': intervention_required,
            'vision_degraded': vision_status != VisionStatus.FULL_VISIBILITY,
            'max_curvature': max_curvature,
            'time_at_level': self.time_at_current_level
        }


def test_enhanced_vtsc():
    """Test the enhanced VTSC with sample scenarios"""

    controller = EnhancedVisionTurnSpeedController()

    # Test scenario: Approaching curve
    print("Testing Enhanced VTSC - Highway Curve Approach")
    print("="*60)

    # Initial conditions
    v_ego = 35.0  # 126 km/h
    time = 0.0
    dt = 0.1

    # Simulated curve ahead
    curve_distance = 100.0
    curve_curvature = 1/150.0  # 150m radius

    for i in range(50):  # 5 seconds
        # Create predicted path
        distances = np.array([20, 40, 60, 80, 100])
        curvatures = np.array([0, 0, curve_curvature/2, curve_curvature, curve_curvature])

        # Find which prediction contains our curve
        curve_idx = -1
        for j, d in enumerate(distances):
            if d >= curve_distance:
                curve_idx = j
                break

        if curve_idx >= 0:
            curvatures[curve_idx:] = curve_curvature

        result = controller.update(
            v_ego=v_ego,
            current_curvature=0.0,  # Currently on straight
            predicted_curvatures=curvatures,
            distances=distances,
            lateral_acc_limit=3.0,
            model_confidence=0.9,
            current_time=time
        )

        # Apply acceleration
        v_ego += result['a_target'] * dt
        v_ego = max(v_ego, 0)

        # Update distance
        curve_distance -= v_ego * dt
        time += dt

        if i % 10 == 0:  # Every second
            print(f"t={time:.1f}s: v={v_ego*3.6:.0f}km/h, "
                  f"d_curve={curve_distance:.0f}m, "
                  f"level={result['emergency_level'].name}, "
                  f"a={result['a_target']:.2f}m/s², "
                  f"anticipation={'ON' if result['using_anticipation'] else 'OFF'}")

        if curve_distance <= 0:
            print(f"\nReached curve at {time:.1f}s, v={v_ego*3.6:.0f}km/h")
            break


if __name__ == "__main__":
    test_enhanced_vtsc()
