#!/usr/bin/env python3
"""
RTI Controller - Realtime Traffic Intelligence speed control integration.

Processes RTI threat data and provides speed recommendations to the
longitudinal planner for proactive speed management.
"""

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET, V_CRUISE_MAX
from openpilot.common.swaglog import cloudlog

# Speed safety constants
MIN_OPERATING_SPEED = 2.24  # 5 mph in m/s - minimum speed for RTI operation
MAX_DECEL_RATE = -2.0  # Maximum deceleration rate in m/s² for threat response
THREAT_ACTIVATION_DISTANCE = 1000  # Maximum distance to consider threats (meters)
THREAT_NEAR_DISTANCE = 300  # Distance threshold for near threats (meters)
THREAT_CRITICAL_DISTANCE = 100  # Distance threshold for critical threats (meters)

# Speed reduction factors based on threat distance
SPEED_REDUCTION_FACTORS = {
    'critical': 0.75,  # 75% of current speed for critical threats
    'near': 0.85,      # 85% of current speed for near threats
    'normal': 0.95     # 95% of current speed for normal threats
}


class RTIController:
    """RTI speed control following VTSC/SLC patterns."""

    def __init__(self, CP):
        """Initialize RTI controller with car parameters."""
        self.CP = CP
        self.params = Params()

        # State tracking
        self._enabled = False
        self._is_active = False
        self._speed_recommendation = V_CRUISE_UNSET
        self._threat_ahead = False
        self._threat_distance = 0.0
        self._threat_type = None
        self._confidence = 0.0

        # Vehicle state
        self._v_ego = 0.0
        self._a_ego = 0.0
        self._v_cruise = V_CRUISE_UNSET

        # Update counters for logging
        self._update_counter = 0

        cloudlog.info("RTI Controller initialized")

    def update(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float) -> None:
        """
        Update RTI controller state based on current conditions.

        Args:
            sm: SubMaster with message subscriptions
            v_ego: Current ego velocity in m/s
            a_ego: Current ego acceleration in m/s²
            v_cruise: Current cruise setpoint in m/s
        """
        # Store vehicle state
        self._v_ego = v_ego
        self._a_ego = a_ego
        self._v_cruise = v_cruise

        # Check if RTI is enabled
        self._enabled = self.params.get_bool("RTIEnabled")

        if not self._enabled:
            self._reset_state()
            return

        # Check for minimum operating speed
        if v_ego < MIN_OPERATING_SPEED:
            self._reset_state()
            return

        # Process RTI state if available
        if sm.valid.get('rtiStateSP', False):
            rti_state = sm['rtiStateSP']
            self._process_rti_state(rti_state)
        else:
            self._reset_state()

        # Log status periodically
        self._update_counter += 1
        if self._update_counter % 100 == 0:  # Every 100 updates (~10 seconds at 10Hz)
            if self._is_active:
                msg = f"RTI Active: threat at {self._threat_distance:.0f}m, recommending {self._speed_recommendation * CV.MS_TO_KPH:.1f} km/h"
                cloudlog.info(msg)

    def _process_rti_state(self, rti_state) -> None:
        """
        Process RTI state message and determine speed recommendation.

        Args:
            rti_state: RTI state message from rtiStateSP service
        """
        # Check if there's a valid threat ahead
        if not rti_state.threatAhead or rti_state.threatDistanceM <= 0:
            self._reset_state()
            return

        # Store threat information
        self._threat_ahead = True
        self._threat_distance = rti_state.threatDistanceM

        # Only activate if threat is within activation distance
        if self._threat_distance > THREAT_ACTIVATION_DISTANCE:
            self._reset_state()
            return

        # Get threat details if available
        if len(rti_state.threats) > 0:
            closest_threat = rti_state.threats[0]  # First threat is closest
            self._threat_type = closest_threat.type
            self._confidence = closest_threat.confidence

            # Use threat's speed limit if available
            if closest_threat.speedLimitMs > 0:
                target_speed = closest_threat.speedLimitMs
            else:
                # Fall back to RTI's recommended speed
                target_speed = rti_state.recommendedSpeed
        else:
            # Use general recommendation if no detailed threats
            target_speed = rti_state.recommendedSpeed
            self._confidence = 0.8  # Default confidence

        # Calculate safe speed based on threat distance
        safe_speed = self._calculate_safe_speed(target_speed)

        # Apply safety validations
        if safe_speed > 0 and safe_speed < self._v_cruise:
            self._speed_recommendation = safe_speed
            self._is_active = True
        else:
            self._reset_state()

    def _calculate_safe_speed(self, target_speed: float) -> float:
        """
        Calculate safe speed based on threat distance and type.

        Args:
            target_speed: Base target speed from threat data

        Returns:
            Safe speed recommendation in m/s
        """
        # Start with the target speed (typically posted speed limit)
        safe_speed = target_speed

        # Apply distance-based reduction factors
        if self._threat_distance < THREAT_CRITICAL_DISTANCE:
            # Critical distance - significant speed reduction
            reduction_factor = SPEED_REDUCTION_FACTORS['critical']
        elif self._threat_distance < THREAT_NEAR_DISTANCE:
            # Near distance - moderate speed reduction
            reduction_factor = SPEED_REDUCTION_FACTORS['near']
        else:
            # Normal distance - slight speed reduction
            reduction_factor = SPEED_REDUCTION_FACTORS['normal']

        # Apply reduction factor to current speed (not target)
        # This ensures we slow down gradually as we approach
        safe_speed = min(safe_speed, self._v_ego * reduction_factor)

        # Apply confidence scaling
        if self._confidence < 0.7:
            # Low confidence - be more conservative
            safe_speed = min(safe_speed, self._v_ego * 0.9)

        # Ensure we never recommend acceleration toward a threat
        safe_speed = min(safe_speed, self._v_ego)

        # Ensure minimum speed
        if safe_speed < MIN_OPERATING_SPEED:
            safe_speed = MIN_OPERATING_SPEED

        # Ensure maximum reasonable speed
        safe_speed = min(safe_speed, V_CRUISE_MAX)

        return safe_speed

    def _reset_state(self) -> None:
        """Reset controller state when inactive."""
        self._is_active = False
        self._speed_recommendation = V_CRUISE_UNSET
        self._threat_ahead = False
        self._threat_distance = 0.0
        self._threat_type = None
        self._confidence = 0.0

    @property
    def is_active(self) -> bool:
        """Return whether RTI speed control is active."""
        return self._is_active

    @property
    def speed_recommendation(self) -> float:
        """
        Return speed recommendation when active.

        Returns:
            Recommended speed in m/s, or V_CRUISE_UNSET if inactive
        """
        if self._is_active:
            return self._speed_recommendation
        return V_CRUISE_UNSET

    @property
    def threat_distance(self) -> float:
        """Return distance to closest threat in meters."""
        return self._threat_distance

    @property
    def threat_type(self) -> str | None:
        """Return type of closest threat."""
        return self._threat_type

    @property
    def enabled(self) -> bool:
        """Return whether RTI is enabled."""
        return self._enabled
