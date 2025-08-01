#!/usr/bin/env python3
"""
VTSC Emergency Scenarios Definition and Classification

This module defines various "uh oh" scenarios where the VTSC needs to handle
situations beyond normal comfort operation, including vision limitations.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np

# Camera FOV specifications for comma 3X
CAMERA_FOV_NARROW = 30.0  # degrees - for distant objects
CAMERA_FOV_WIDE = 60.0    # degrees - for general road view

# Convert to radians for calculations
CAMERA_FOV_NARROW_RAD = np.radians(CAMERA_FOV_NARROW)
CAMERA_FOV_WIDE_RAD = np.radians(CAMERA_FOV_WIDE)


class EmergencyLevel(Enum):
    """Classification of emergency severity"""
    NORMAL = 0        # Comfort mode operation
    CAUTION = 1       # Slightly uncomfortable but safe
    WARNING = 2       # Uncomfortable deceleration needed
    CRITICAL = 3      # Emergency deceleration required
    INTERVENTION = 4  # Driver must take over


class VisionStatus(Enum):
    """Vision model status for road detection"""
    FULL_VISIBILITY = 0     # Road fully visible
    PARTIAL_OCCLUSION = 1   # Some road segments missing
    CURVE_EXCEEDS_FOV = 2   # Curve goes beyond camera view
    LOST_ROAD = 3           # Cannot detect road at all


@dataclass
class EmergencyScenario:
    """Defines an emergency scenario for testing"""
    name: str
    description: str

    # Initial conditions
    v_ego_ms: float              # Current speed (m/s)
    v_target_ms: float           # Target speed for curve (m/s)
    distance_to_curve_m: float   # Distance when curve detected
    max_curvature: float         # Maximum curvature (1/m)

    # Vision conditions
    vision_status: VisionStatus
    detection_delay_s: float     # How late the detection was
    confidence: float           # Model confidence (0-1)

    # Expected outcomes
    required_decel_ms2: float   # Physics-required deceleration
    emergency_level: EmergencyLevel
    intervention_required: bool

    @property
    def v_ego_kph(self) -> float:
        return self.v_ego_ms * 3.6

    @property
    def v_target_kph(self) -> float:
        return self.v_target_ms * 3.6

    @property
    def required_decel_g(self) -> float:
        return abs(self.required_decel_ms2) / 9.81

    @property
    def curve_radius_m(self) -> float:
        return 1.0 / self.max_curvature if self.max_curvature > 0 else float('inf')

    def can_curve_fit_in_fov(self, fov_rad: float) -> bool:
        """Check if curve fits within camera FOV"""
        if self.curve_radius_m == float('inf'):
            return True

        # Calculate arc angle for curve visibility
        # Assuming we need to see at least 50m ahead
        arc_length = 50.0
        arc_angle = arc_length / self.curve_radius_m

        return arc_angle <= fov_rad / 2  # Half FOV on each side


# Define comprehensive emergency scenarios
EMERGENCY_SCENARIOS = [
    # ========== Late Detection Scenarios ==========
    EmergencyScenario(
        name="late_highway_curve",
        description="Late detection of highway curve at high speed",
        v_ego_ms=35.0,  # 126 km/h
        v_target_ms=25.0,  # 90 km/h
        distance_to_curve_m=80.0,  # Only 80m to slow down
        max_curvature=0.005,  # 200m radius
        vision_status=VisionStatus.FULL_VISIBILITY,
        detection_delay_s=2.0,
        confidence=0.7,
        required_decel_ms2=-3.125,  # Need 0.32g
        emergency_level=EmergencyLevel.WARNING,
        intervention_required=False
    ),

    EmergencyScenario(
        name="very_late_mountain_hairpin",
        description="Very late detection of mountain hairpin",
        v_ego_ms=20.0,  # 72 km/h
        v_target_ms=8.0,   # 29 km/h
        distance_to_curve_m=30.0,  # Only 30m!
        max_curvature=0.05,  # 20m radius hairpin
        vision_status=VisionStatus.PARTIAL_OCCLUSION,
        detection_delay_s=3.0,
        confidence=0.5,
        required_decel_ms2=-5.6,  # Need 0.57g!
        emergency_level=EmergencyLevel.CRITICAL,
        intervention_required=False  # Can still make it with emergency braking
    ),

    # ========== Blind Corner Scenarios ==========
    EmergencyScenario(
        name="blind_hairpin_exceeds_fov",
        description="Hairpin curve that exceeds camera FOV",
        v_ego_ms=15.0,  # 54 km/h
        v_target_ms=7.0,   # 25 km/h
        distance_to_curve_m=40.0,
        max_curvature=0.1,  # 10m radius - very tight!
        vision_status=VisionStatus.CURVE_EXCEEDS_FOV,
        detection_delay_s=0.0,  # Not late, just can't see around corner
        confidence=0.4,  # Low confidence due to occlusion
        required_decel_ms2=-2.125,  # Need 0.22g
        emergency_level=EmergencyLevel.WARNING,
        intervention_required=False
    ),

    EmergencyScenario(
        name="blind_mountain_switchback",
        description="Switchback that goes completely out of view",
        v_ego_ms=18.0,  # 65 km/h
        v_target_ms=6.0,   # 22 km/h
        distance_to_curve_m=50.0,
        max_curvature=0.067,  # 15m radius
        vision_status=VisionStatus.CURVE_EXCEEDS_FOV,
        detection_delay_s=1.0,
        confidence=0.3,
        required_decel_ms2=-2.88,  # Need 0.29g
        emergency_level=EmergencyLevel.WARNING,
        intervention_required=False
    ),

    # ========== Critical Intervention Scenarios ==========
    EmergencyScenario(
        name="impossible_late_detection",
        description="Detection too late to avoid intervention",
        v_ego_ms=30.0,  # 108 km/h
        v_target_ms=15.0,  # 54 km/h
        distance_to_curve_m=40.0,  # Way too short!
        max_curvature=0.02,  # 50m radius
        vision_status=VisionStatus.PARTIAL_OCCLUSION,
        detection_delay_s=4.0,
        confidence=0.6,
        required_decel_ms2=-7.875,  # Need 0.80g - impossible!
        emergency_level=EmergencyLevel.INTERVENTION,
        intervention_required=True
    ),

    EmergencyScenario(
        name="lost_road_in_fog",
        description="Complete loss of road visibility",
        v_ego_ms=25.0,  # 90 km/h
        v_target_ms=15.0,  # 54 km/h
        distance_to_curve_m=60.0,
        max_curvature=0.01,  # 100m radius
        vision_status=VisionStatus.LOST_ROAD,
        detection_delay_s=2.0,
        confidence=0.1,  # Very low confidence
        required_decel_ms2=-3.33,  # Need 0.34g
        emergency_level=EmergencyLevel.CRITICAL,
        intervention_required=False  # Can still make it with max braking
    ),

    # ========== Edge Cases ==========
    EmergencyScenario(
        name="decreasing_radius_corner",
        description="Corner that tightens beyond FOV",
        v_ego_ms=22.0,  # 79 km/h
        v_target_ms=10.0,  # 36 km/h
        distance_to_curve_m=55.0,
        max_curvature=0.04,  # 25m initial radius, tightens further
        vision_status=VisionStatus.CURVE_EXCEEDS_FOV,
        detection_delay_s=1.5,
        confidence=0.5,
        required_decel_ms2=-2.95,  # Need 0.30g
        emergency_level=EmergencyLevel.WARNING,
        intervention_required=False
    ),

    EmergencyScenario(
        name="sudden_obstacle_in_curve",
        description="Obstacle detected mid-curve requiring harder braking",
        v_ego_ms=20.0,  # 72 km/h
        v_target_ms=5.0,   # 18 km/h - near stop!
        distance_to_curve_m=35.0,
        max_curvature=0.025,  # 40m radius
        vision_status=VisionStatus.PARTIAL_OCCLUSION,
        detection_delay_s=2.5,
        confidence=0.4,
        required_decel_ms2=-5.36,  # Need 0.55g
        emergency_level=EmergencyLevel.CRITICAL,
        intervention_required=False
    ),

    # ========== Comfortable Emergency Scenarios ==========
    EmergencyScenario(
        name="early_warning_manageable",
        description="Late but manageable with moderate discomfort",
        v_ego_ms=28.0,  # 101 km/h
        v_target_ms=22.0,  # 79 km/h
        distance_to_curve_m=70.0,
        max_curvature=0.008,  # 125m radius
        vision_status=VisionStatus.FULL_VISIBILITY,
        detection_delay_s=1.0,
        confidence=0.8,
        required_decel_ms2=-1.8,  # Need 0.18g - comfortable
        emergency_level=EmergencyLevel.CAUTION,
        intervention_required=False
    ),

    EmergencyScenario(
        name="moderate_blind_corner",
        description="Blind corner with adequate distance",
        v_ego_ms=18.0,  # 65 km/h
        v_target_ms=12.0,  # 43 km/h
        distance_to_curve_m=60.0,
        max_curvature=0.03,  # 33m radius
        vision_status=VisionStatus.CURVE_EXCEEDS_FOV,
        detection_delay_s=0.5,
        confidence=0.6,
        required_decel_ms2=-1.5,  # Need 0.15g - very comfortable
        emergency_level=EmergencyLevel.NORMAL,
        intervention_required=False
    ),
]


def calculate_stopping_distance(v_ms: float, decel_ms2: float) -> float:
    """Calculate distance needed to stop at given deceleration"""
    return v_ms**2 / (2 * abs(decel_ms2))


def calculate_curve_arc_angle(radius: float, distance: float) -> float:
    """Calculate the arc angle subtended by a curve segment"""
    return distance / radius  # In radians


def determine_emergency_level(required_decel_g: float,
                            vision_status: VisionStatus,
                            confidence: float) -> EmergencyLevel:
    """
    Determine emergency level based on multiple factors
    """
    # Base level on deceleration requirement
    if required_decel_g <= 0.15:
        level = EmergencyLevel.NORMAL
    elif required_decel_g <= 0.25:
        level = EmergencyLevel.CAUTION
    elif required_decel_g <= 0.40:
        level = EmergencyLevel.WARNING
    elif required_decel_g <= 0.60:
        level = EmergencyLevel.CRITICAL
    else:
        level = EmergencyLevel.INTERVENTION

    # Increase level for poor vision
    if vision_status == VisionStatus.LOST_ROAD:
        level = EmergencyLevel(min(level.value + 2, EmergencyLevel.INTERVENTION.value))
    elif vision_status == VisionStatus.CURVE_EXCEEDS_FOV:
        level = EmergencyLevel(min(level.value + 1, EmergencyLevel.INTERVENTION.value))

    # Increase level for low confidence
    if confidence < 0.3:
        level = EmergencyLevel(min(level.value + 1, EmergencyLevel.INTERVENTION.value))

    return level


def classify_vision_limitation(curvature: float, distance_visible: float,
                             fov_rad: float = CAMERA_FOV_WIDE_RAD) -> VisionStatus:
    """
    Classify the type of vision limitation based on curve geometry
    """
    if curvature == 0:
        return VisionStatus.FULL_VISIBILITY

    radius = 1.0 / curvature

    # Check if curve arc exceeds FOV
    arc_angle = distance_visible / radius
    if arc_angle > fov_rad / 2:
        return VisionStatus.CURVE_EXCEEDS_FOV

    # Other checks could be added here
    return VisionStatus.FULL_VISIBILITY


if __name__ == "__main__":
    print("VTSC Emergency Scenarios Analysis")
    print("=" * 80)

    # Analyze each scenario
    for scenario in EMERGENCY_SCENARIOS:
        print(f"\n{scenario.name}: {scenario.description}")
        print(f"  Speed: {scenario.v_ego_kph:.0f} → {scenario.v_target_kph:.0f} km/h")
        print(f"  Distance: {scenario.distance_to_curve_m:.0f}m")
        print(f"  Curve radius: {scenario.curve_radius_m:.0f}m")
        print(f"  Required decel: {scenario.required_decel_g:.2f}g")
        print(f"  Emergency level: {scenario.emergency_level.name}")
        print(f"  Vision status: {scenario.vision_status.name}")

        # Check if curve fits in FOV
        fits_narrow = scenario.can_curve_fit_in_fov(CAMERA_FOV_NARROW_RAD)
        fits_wide = scenario.can_curve_fit_in_fov(CAMERA_FOV_WIDE_RAD)
        print(f"  Fits in FOV: Narrow({fits_narrow}), Wide({fits_wide})")
        print(f"  Intervention required: {scenario.intervention_required}")

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"Total scenarios: {len(EMERGENCY_SCENARIOS)}")
    print(f"Requiring intervention: {sum(1 for s in EMERGENCY_SCENARIOS if s.intervention_required)}")
    print(f"Exceeding FOV: {sum(1 for s in EMERGENCY_SCENARIOS if s.vision_status == VisionStatus.CURVE_EXCEEDS_FOV)}")
