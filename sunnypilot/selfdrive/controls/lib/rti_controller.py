#!/usr/bin/env python3
"""
RTI Controller - Realtime Traffic Intelligence speed control integration.

Processes RTI threat data and provides speed recommendations to the
longitudinal planner for proactive speed management.
"""

from cereal import messaging
from openpilot.common.params import Params
from opendbc.car.common.conversions import Conversions as CV
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET, V_CRUISE_MAX
from openpilot.common.swaglog import cloudlog
import time

# Speed safety constants
MIN_OPERATING_SPEED = 2.24  # 5 mph in m/s - minimum speed for RTI operation
MAX_DECEL_RATE = -2.0  # Maximum deceleration rate in m/s² for threat response
# Note: THREAT_ACTIVATION_DISTANCE now uses RTIForwardSlowdownRange parameter
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
        self._threat_direction = None
        self._threat_type = None
        self._confidence = 0.0
        self._active_threat_id = None

        # Vehicle state
        self._v_ego = 0.0
        self._a_ego = 0.0
        self._v_cruise = V_CRUISE_UNSET

        # Ramped decel state for RTI-only speed target shaping
        self._ramped_speed = None  # m/s, internal smoothed recommendation
        self._last_update_ts = None  # monotonic seconds
        self._rti_decel_rate = 1.4  # m/s^2 default (gentle, slightly > coast)

        # Load user-configured parameters
        self._load_user_params()

        # Update counters for logging
        self._update_counter = 0

        cloudlog.info("RTI Controller initialized")

    def _load_user_params(self) -> None:
        """Load user-configured RTI parameters."""
        # Get forward slowdown range (when to activate RTI for threats ahead)
        forward_range = self.params.get("RTIForwardSlowdownRange")
        if forward_range:
            try:
                self._threat_activation_distance = float(forward_range)
            except (ValueError, TypeError):
                self._threat_activation_distance = 1609  # Default 1.0 miles
        else:
            self._threat_activation_distance = 1609  # Default 1.0 miles

        # Get resume speed distance (when to stop slowing after passing threat)
        resume_distance = self.params.get("RTIResumeSpeedDistance")
        if resume_distance:
            try:
                self._resume_speed_distance = float(resume_distance)
            except (ValueError, TypeError):
                self._resume_speed_distance = 1609  # Default 1.0 miles
        else:
            self._resume_speed_distance = 1609  # Default 1.0 miles

        # Get speed reduction settings
        speed_mode = self.params.get("RTISpeedReductionMode")
        if speed_mode:
            self._speed_reduction_mode = speed_mode.decode('utf-8') if isinstance(speed_mode, bytes) else str(speed_mode)
        else:
            self._speed_reduction_mode = "posted"

        # Get custom speed reduction amount
        speed_reduction = self.params.get("RTISpeedReduction")
        if speed_reduction:
            try:
                speed_reduction_kmh = float(speed_reduction)
                self._custom_speed_reduction_ms = speed_reduction_kmh / 3.6  # Convert km/h to m/s
            except (ValueError, TypeError):
                self._custom_speed_reduction_ms = 4.4  # Default 10 mph in m/s
        else:
            self._custom_speed_reduction_ms = 4.4  # Default 10 mph in m/s

        # Gentle RTI decel rate (m/s^2). Only affects RTI recommendation shaping.
        try:
            decel_rate = self.params.get("RTIDecelRate")
        except Exception:
            decel_rate = None
        if decel_rate:
            try:
                val = float(decel_rate)
                # clamp to safe range
                self._rti_decel_rate = max(0.5, min(3.0, val))
            except (ValueError, TypeError):
                self._rti_decel_rate = 1.4
        else:
            self._rti_decel_rate = 1.4

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

        # Compute dt for ramping
        now = time.monotonic()
        dt = 0.0
        if self._last_update_ts is not None:
            dt = max(0.0, min(1.0, now - self._last_update_ts))
        self._last_update_ts = now

        # Process RTI state if available
        if sm.valid.get('rtiStateSP', False):
            rti_state = sm['rtiStateSP']
            self._process_rti_state(rti_state, dt)
        else:
            self._reset_state()

        # Log status periodically
        self._update_counter += 1
        if self._update_counter % 100 == 0:  # Every 100 updates (~10 seconds at 10Hz)
            if self._is_active:
                msg = f"RTI Active: threat at {self._threat_distance:.0f}m, recommending {self._speed_recommendation * CV.MS_TO_KPH:.1f} km/h"
                cloudlog.info(msg)

    def _process_rti_state(self, rti_state, dt: float) -> None:
        """
        Process RTI state message and determine speed recommendation.

        Args:
            rti_state: RTI state message from rtiStateSP service
        """
        # Process all threats to handle both ahead and behind cases
        if len(rti_state.threats) == 0:
            self._reset_state()
            return

        # Find relevant threats based on direction and distance
        relevant_threat = None
        prev_active_threat_id = self._active_threat_id

        def _direction_name(direction) -> str:
            if isinstance(direction, str):
                return direction.lower()
            # capnp enums often stringify as "ahead"/"RtiStateSP.Direction.ahead"
            d = str(direction)
            if '.' in d:
                d = d.split('.')[-1]
            return d.lower()
        
        # Live-read user filter as extra defense (daemon also filters)
        def _filter_allows(threat_type) -> bool:
            try:
                tf_raw = self.params.get("RTIThreatFilter")
                filt = int(tf_raw) if tf_raw is not None else 0
            except Exception:
                filt = 0
            # Normalize type to lowercase string name for comparison
            tname = str(threat_type)
            tname = tname.split('.')[-1] if '.' in tname else tname
            tname = tname.lower()
            if filt == 0:  # all
                return True
            if filt == 1:  # police
                return tname in ("police", "policehiding")
            if filt == 2:  # cameras
                return tname in ("speedtrap", "speedcamera")
            if filt == 3:  # hazards
                return tname in ("hazard", "shoulderhazard", "roadhazard")
            # Custom (treat as all until per-type toggles exist)
            return True

        for threat in rti_state.threats:
            # Respect user-selected filter (belt-and-suspenders; rtid already applies it)
            if not _filter_allows(getattr(threat, 'type', None)):
                continue
            # Only let same-road threats affect longitudinal control.
            # HUD can still display off-road threats for awareness.
            if not bool(getattr(threat, 'onSameRoad', True)):
                continue
            threat_distance = threat.distance
            threat_direction = _direction_name(getattr(threat, 'direction', ''))
            
            # Check threats ahead within activation distance
            if threat_direction == 'ahead' and threat_distance <= self._threat_activation_distance:
                if relevant_threat is None or threat_distance < relevant_threat.distance:
                    relevant_threat = threat
            
            # Check threats behind within resume distance (continue slowing until past resume distance)
            elif threat_direction == 'behind' and threat_distance <= self._resume_speed_distance:
                # We recently passed this threat, continue speed control
                if relevant_threat is None or threat_distance < relevant_threat.distance:
                    relevant_threat = threat

        # Transition robustness: keep slowing for the last active threat if we briefly lose
        # its direction classification (common when ego is on top of the alert and bearing becomes unstable).
        if relevant_threat is None and self._is_active and prev_active_threat_id:
            for threat in rti_state.threats:
                if not _filter_allows(getattr(threat, 'type', None)):
                    continue
                if not bool(getattr(threat, 'onSameRoad', True)):
                    continue

                try:
                    tid = getattr(threat, 'id', None)
                    if tid is None:
                        continue
                    if str(tid) != str(prev_active_threat_id):
                        continue

                    threat_distance = float(getattr(threat, 'distance', 1e9))
                    if threat_distance <= self._resume_speed_distance:
                        relevant_threat = threat
                        break
                except Exception:
                    continue

        # No relevant threats found
        if relevant_threat is None:
            self._reset_state()
            return

        # Store threat information
        self._threat_direction = _direction_name(getattr(relevant_threat, 'direction', ''))
        self._threat_ahead = (self._threat_direction == 'ahead')
        self._threat_distance = relevant_threat.distance
        self._threat_type = relevant_threat.type
        self._confidence = relevant_threat.confidence
        self._active_threat_id = str(getattr(relevant_threat, 'id', '')) or None

        # Determine target speed based on threat and user settings
        if self._speed_reduction_mode == "posted" and relevant_threat.speedLimitMs > 0:
            # Use posted speed limit from threat
            target_speed = relevant_threat.speedLimitMs
        elif self._speed_reduction_mode == "custom":
            # Apply custom speed reduction from current cruise speed
            if self._v_cruise > 0:
                target_speed = self._v_cruise - self._custom_speed_reduction_ms
            else:
                target_speed = rti_state.recommendedSpeed
        else:
            # Fall back to RTI's recommended speed
            target_speed = rti_state.recommendedSpeed

        # Calculate safe speed based on threat distance
        safe_speed = self._calculate_safe_speed(target_speed)

        # RTI-only gentle ramp-down toward base target to limit decel aggressiveness.
        base_target = safe_speed

        # Initialize ramp from current cruise or ego on activation
        if self._ramped_speed is None or not self._is_active:
            start_from = self._v_cruise if self._v_cruise > 0 else max(self._v_ego, base_target)
            self._ramped_speed = max(base_target, min(start_from, V_CRUISE_MAX))
        else:
            # Decrease at most rti_decel_rate * dt; increase immediately
            if self._ramped_speed > base_target and dt > 0.0:
                max_drop = self._rti_decel_rate * dt
                self._ramped_speed = max(base_target, self._ramped_speed - max_drop)
            else:
                self._ramped_speed = min(V_CRUISE_MAX, base_target)

        # Safety: avoid encouraging acceleration while threat is ahead
        hysteresis = 0.45  # ~1 mph
        if self._threat_ahead and self._ramped_speed > (base_target + hysteresis):
            self._ramped_speed = min(self._ramped_speed, self._v_ego)

        # Final bounding vs. user's cruise
        final_reco = self._ramped_speed
        if self._v_cruise > 0:
            final_reco = min(final_reco, self._v_cruise)

        final_reco = min(final_reco, V_CRUISE_MAX)

        # Apply safety validations
        if final_reco > 0 and (self._v_cruise <= 0 or final_reco < self._v_cruise):
            self._speed_recommendation = final_reco
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

        # In "posted" mode, use the posted speed limit exactly (no extra reductions)
        # Keep within planner bounds and the user's cruise setpoint, but do not ratchet below the limit.
        if self._speed_reduction_mode == "posted":
            # Bound by user's current cruise setpoint if available
            if self._v_cruise > 0:
                safe_speed = min(safe_speed, self._v_cruise)
            # Bound by global max cruise
            safe_speed = min(safe_speed, V_CRUISE_MAX)
            # Ensure non-negative
            safe_speed = max(0.0, safe_speed)
            return safe_speed
        
        # For custom mode, apply distance-based reduction factors
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
        self._threat_direction = None
        self._threat_type = None
        self._confidence = 0.0
        self._ramped_speed = None
        self._active_threat_id = None

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
