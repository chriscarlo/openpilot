#!/usr/bin/env python3
"""
RTI Daemon - Realtime Traffic Intelligence

Fetches traffic alert data from external sources (Waze API), processes threats
for relevance to current route, and publishes RTI state for consumption by
the longitudinal planner.

Architecture: fetch → detect → publish at 1Hz
"""

import asyncio
import inspect
import json
import os
import time

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from .waze_api_client import WazeAPIClient
from .threat_detector import ThreatDetector


class RTIDaemon:
    """Main RTI daemon class following openpilot patterns."""

    def __init__(self):
        """Initialize RTI daemon with messaging and configuration."""
        self.params = Params()

        # Messaging setup - Now includes both map and dashboard speed limit data
        self.sm = messaging.SubMaster([
            'gpsLocationExternal',
            'gpsLocation',
            'carState',
            'liveMapDataSP',  # Map-based speed limit data
            'carStateSP'      # Dashboard-based speed limit data (from car's TSR camera)
        ], ignore_avg_freq=True)
        self.pm = messaging.PubMaster(['rtiStateSP'])

        # Load API configuration with environment awareness
        self.api_key = self._load_api_key()

        # Core components
        self.waze_client = WazeAPIClient(self.api_key) if self.api_key else None
        self.threat_detector = ThreatDetector()

        # State tracking
        self.enabled = False
        self.last_update_time = 0
        self.loop_count = 0

        # API fetch control - CRITICAL: Only fetch every 30-60 seconds!
        self.last_api_fetch_time = 0
        # 30 second interval = 120 calls/hour max
        self.api_fetch_interval = 30  # seconds between API calls
        self.cached_traffic_data = None
        self.cached_data_location = None
        self.cached_data_timestamp = 0
        self.max_data_age = 300  # 5 minutes max staleness
        self.dev_fallback_location = self._parse_dev_fallback_location()

        api_calls_per_hour = 3600 / self.api_fetch_interval
        cloudlog.info(f"RTI Daemon initialized - API interval: {self.api_fetch_interval}s ({api_calls_per_hour:.0f} calls/hr max)")

    @staticmethod
    def _parse_dev_fallback_location() -> tuple[float, float] | None:
        """Parse optional dev-only fallback GPS location from env.

        Set `RTID_DEV_LOCATION` as `lat,lon` (for example `37.7749,-122.4194`).
        """
        raw = os.getenv("RTID_DEV_LOCATION")
        if not raw:
            return None
        try:
            lat_str, lon_str = [x.strip() for x in raw.split(",", 1)]
            lat = float(lat_str)
            lon = float(lon_str)
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                cloudlog.warning(f"RTI dev fallback location enabled via RTID_DEV_LOCATION: {lat:.5f},{lon:.5f}")
                return (lat, lon)
        except Exception:
            pass
        cloudlog.error(f"RTI invalid RTID_DEV_LOCATION format: {raw!r} (expected 'lat,lon')")
        return None

    def _load_api_key(self) -> str | None:
        """Load Waze API key from params then environment/file manager.

        Supported sources (in priority order):
        - Params `RTIManualApiKey` (set by RTISettingsPanel on-device)
        - `api_key_manager.get_api_key()`:
          - Env vars (new + legacy compatibility)
          - `/persist/openwebninja_waze_api_key` (TICI)
          - `/projects/chauffeur/persist/openwebninja_waze_api_key` (dev)
        """

        # 1) Params (preferred on-device; avoids hard dependency on a specific file name)
        try:
            param_val = self.params.get("RTIManualApiKey")
            if isinstance(param_val, (bytes, bytearray)):
                param_val = param_val.decode("utf-8", errors="ignore")
            if isinstance(param_val, str) and param_val.strip():
                cloudlog.info("RTI API key loaded from Params (RTIManualApiKey)")
                return param_val.strip()
        except Exception as e:
            cloudlog.debug(f"RTI: Could not read RTIManualApiKey from Params: {e}")

        # 2) Environment and file-based key loading
        try:
            from .api_key_manager import get_api_key
            api_key_val = get_api_key()
            if isinstance(api_key_val, str) and api_key_val.strip():
                cloudlog.info("RTI API key loaded from env/persist key manager")
                return api_key_val.strip()
        except Exception as e:
            cloudlog.debug(f"RTI: api_key_manager.get_api_key() failed: {e}")

        cloudlog.warning("RTI API key not found - running in offline mode")
        return None

    @staticmethod
    def _map_health_to_api_status(health_status: str) -> str:
        """Map WazeAPIClient health strings to capnp ApiStatus values."""
        # capnp enum: connected, disconnected, rateLimited, error, offline
        if health_status in ("connected", "disconnected", "rateLimited", "error", "offline"):
            return health_status
        # Internal WazeAPIClient may use "degraded"; map to closest capnp value
        if health_status == "degraded":
            return "disconnected"
        return "disconnected"

    def _check_enabled(self) -> bool:
        """Check if RTI is enabled via params."""
        return bool(self.params.get_bool("RTIEnabled"))

    def _get_current_location(self) -> tuple[float, float] | None:
        """Get current GPS coordinates from location services."""
        # Update with small timeout to ensure we get fresh data
        self.sm.update(100)  # 100ms timeout to get fresh GPS

        # Prefer external GPS if available
        # NOTE: Don't check updated flag - we want current value even if not "new"
        gps_ext = self.sm['gpsLocationExternal']
        if gps_ext:
            lat = getattr(gps_ext, 'latitude', 0.0)
            lon = getattr(gps_ext, 'longitude', 0.0)
            # Check for valid coordinates (not 0,0) and reasonable accuracy
            if lat != 0.0 and lon != 0.0:
                accuracy = getattr(gps_ext, 'horizontalAccuracy', 0.0)
                # Accept if accuracy is reasonable (0 might mean unset or perfect)
                if accuracy <= 10.0:
                    return (lat, lon)

        # Fall back to regular GPS location
        # NOTE: Don't check updated flag - we want current value even if not "new"
        gps_loc = self.sm['gpsLocation']
        if gps_loc and getattr(gps_loc, 'hasFix', False):
            lat = getattr(gps_loc, 'latitude', 0.0)
            lon = getattr(gps_loc, 'longitude', 0.0)
            # Check for valid coordinates (not 0,0)
            if lat != 0.0 and lon != 0.0:
                return (lat, lon)

        # Optional dev-only fallback for testing in environments without live GPS.
        if self.dev_fallback_location is not None:
            return self.dev_fallback_location

        return None

    def _get_current_heading_deg(self) -> float | None:
        """Get current ego heading in degrees from GPS if available."""
        # Update with small timeout to ensure we get fresh data
        self.sm.update(100)  # 100ms timeout to get fresh GPS

        # Check internal GPS first (usually more reliable)
        # NOTE: Don't check updated flag - we want current value even if not "new"
        gps_loc = self.sm['gpsLocation']
        if gps_loc and getattr(gps_loc, 'hasFix', False):
            try:
                bearing = getattr(gps_loc, 'bearingDeg', None)
                if bearing is not None and 0.0 <= float(bearing) <= 360.0:
                    return float(bearing)
            except Exception:
                pass

        # Fall back to external GPS if internal doesn't have bearing
        gps_ext = self.sm['gpsLocationExternal']
        if gps_ext and getattr(gps_ext, 'hasFix', False):
            try:
                bearing = getattr(gps_ext, 'bearingDeg', None)
                # Validate numeric and finite
                if bearing is not None and 0.0 <= float(bearing) <= 360.0:
                    return float(bearing)
            except Exception:
                pass

        return None

    def _get_current_speed(self) -> float:
        """Get current vehicle speed in m/s."""
        self.sm.update(0)
        if self.sm.updated['carState']:
            # vEgo is in m/s
            try:
                return float(self.sm['carState'].vEgo)
            except Exception:
                return 0.0
        return 0.0

    def _get_cruise_cluster_speed(self) -> float:
        """Get driver's originally set cruise speed (as shown on cluster) in m/s."""
        self.sm.update(0)
        if self.sm.updated['carState']:
            cruise_state = self.sm['carState'].cruiseState
            # speedCluster is the driver's set speed shown on the instrument cluster
            # This is the original driver-set maximum, unmodified by controllers
            if cruise_state.enabled and cruise_state.speedCluster > 0:
                try:
                    return float(cruise_state.speedCluster)
                except Exception:
                    return 0.0
        return 0.0

    def _get_current_speed_limit(self) -> float:
        """Get current posted speed limit from both map and dashboard sources.

        Conservative combination: when both sources have data, use the LOWER value to
        avoid recommending speeds above a more restrictive posted limit (e.g. school zone).

        Returns:
            Speed limit in m/s, or 0.0 if not available
        """
        map_limit = 0.0
        dashboard_limit = 0.0

        # Get map-based speed limit
        try:
            map_data = self.sm['liveMapDataSP']
            if map_data.speedLimitValid:
                map_limit = float(map_data.speedLimit)
                cloudlog.debug(f"RTI: Map speed limit: {map_limit:.1f} m/s ({map_limit * 2.237:.0f} mph)")
        except Exception as e:
            cloudlog.debug(f"RTI: Could not get speed limit from map data: {e}")

        # Get dashboard-based speed limit (from car's traffic sign recognition)
        try:
            car_state_sp = self.sm['carStateSP']
            if car_state_sp.speedLimit > 0:
                dashboard_limit = float(car_state_sp.speedLimit)
                cloudlog.debug(f"RTI: Dashboard speed limit: {dashboard_limit:.1f} m/s ({dashboard_limit * 2.237:.0f} mph)")
        except Exception as e:
            cloudlog.debug(f"RTI: Could not get speed limit from dashboard: {e}")

        # CONSERVATIVE COMBINATION: Use MIN when both sources are available.
        # RTI errs on the side of safety to avoid recommending speeds above the
        # lower posted limit (e.g., construction or school zones).
        if map_limit > 0 and dashboard_limit > 0:
            combined_limit = min(map_limit, dashboard_limit)
            source = "dashboard" if dashboard_limit <= map_limit else "map"
            cloudlog.debug(f"RTI: Using {source} speed limit (conservative): {combined_limit:.1f} m/s")
            return combined_limit
        elif dashboard_limit > 0:
            # Only dashboard has data
            cloudlog.debug(f"RTI: Using dashboard-only speed limit: {dashboard_limit:.1f} m/s")
            return dashboard_limit
        elif map_limit > 0:
            # Only map has data
            cloudlog.debug(f"RTI: Using map-only speed limit: {map_limit:.1f} m/s")
            return map_limit
        else:
            # No speed limit data available from either source
            cloudlog.debug("RTI: No speed limit data available from map or dashboard")
            return 0.0

    async def _process_cycle_async(self):
        """Non-blocking version of process cycle that stores state for continuous publishing."""
        current_time = time.monotonic()

        try:
            # Get current vehicle state
            location = self._get_current_location()
            current_speed = self._get_current_speed()
            cruise_cluster_speed = self._get_cruise_cluster_speed()
            current_speed_limit = self._get_current_speed_limit()  # Get actual posted speed limit

            if location is None:
                # No valid GPS - store offline state
                self._last_processed_state = None
                return

            # Check if we need to fetch new API data (rate limited!)
            traffic_data = None
            data_age = current_time - self.cached_data_timestamp

            if self.waze_client:
                # Only fetch if:
                # 1. We haven't fetched recently (respect interval)
                # 2. We have no cached data OR data is stale
                should_fetch = (
                    (current_time - self.last_api_fetch_time) >= self.api_fetch_interval and
                    (self.cached_traffic_data is None or data_age > self.max_data_age)
                )

                if should_fetch:
                    try:
                        cloudlog.info(f"RTI fetching new API data (last fetch {current_time - self.last_api_fetch_time:.1f}s ago)")
                        # Use user-configured detection radius for API fetch
                        # Convert meters to km for API call
                        detection_radius = self.params.get("RTIDetectionRadius")
                        if detection_radius:
                            try:
                                radius_m = float(detection_radius)
                            except (ValueError, TypeError):
                                radius_m = 4828  # Default 3 miles in meters
                        else:
                            radius_m = 4828  # Default 3 miles in meters

                        radius_km = radius_m / 1000.0  # Convert to km

                        # Record the attempt immediately to avoid retry storms if the request fails.
                        # This daemon runs at 50Hz; without this, a failure could spam the API.
                        self.last_api_fetch_time = current_time

                        prev_success_time = getattr(self.waze_client, "last_success_time", 0)
                        traffic_data_candidate = await self.waze_client.get_traffic_alerts(
                            location[0], location[1], radius_km
                        )

                        # CAPTURE ALL POLICE DATA TO FILE
                        police_capture_file = "/data/openpilot/live_waze_police_capture.json"
                        police_alerts = []
                        all_alert_types = set()

                        for alert in (traffic_data_candidate or []):
                            # Track all alert types
                            all_alert_types.add(alert.type)

                            # Capture police alerts with ALL data
                            if alert.type in ['police', 'policeHiding', 'POLICE', 'POLICE_HIDING']:
                                police_data = {
                                    "timestamp": current_time,
                                    "location": location,
                                    "alert": {
                                        "id": alert.id,
                                        "type": alert.type,
                                        "latitude": alert.latitude,
                                        "longitude": alert.longitude,
                                        "confidence": alert.confidence,
                                        "speed_limit": alert.speed_limit,
                                        "street": alert.street,
                                        "country": alert.country,
                                        "raw_data": alert.raw_data
                                    }
                                }
                                police_alerts.append(police_data)

                        # Log and save if we found police
                        if police_alerts:
                            cloudlog.warning(f"RTI: POLICE ALERTS FOUND! Count: {len(police_alerts)}")
                            for pa in police_alerts:
                                msg = "RTI POLICE: {} at {} ({:.5f}, {:.5f})".format(
                                    pa['alert']['type'], pa['alert']['street'],
                                    pa['alert']['latitude'], pa['alert']['longitude']
                                )
                                cloudlog.warning(msg)
                                if pa['alert']['raw_data']:
                                    cloudlog.warning(f"RTI POLICE RAW: {json.dumps(pa['alert']['raw_data'])}")

                            # Append to capture file
                            try:
                                existing_data = []
                                if os.path.exists(police_capture_file):
                                    with open(police_capture_file) as f:
                                        existing_data = json.load(f)
                                existing_data.extend(police_alerts)
                                with open(police_capture_file, 'w') as f:
                                    json.dump(existing_data, f, indent=2)
                                cloudlog.warning(f"RTI: Saved {len(police_alerts)} police alerts to {police_capture_file}")
                            except Exception as e:
                                cloudlog.error(f"RTI: Failed to save police data: {e}")

                        # Only treat this fetch as "fresh" if the HTTP request actually succeeded.
                        # `get_traffic_alerts()` returns [] both for "no alerts" and for request errors,
                        # so we use `last_success_time` to differentiate.
                        request_succeeded = getattr(self.waze_client, "last_success_time", 0) > prev_success_time
                        if request_succeeded:
                            traffic_data = traffic_data_candidate
                            cloudlog.info(f"RTI: Found {len(traffic_data)} total alerts. Types: {all_alert_types}")

                            # Update cache on success (even if empty list)
                            self.cached_traffic_data = traffic_data
                            self.cached_data_location = location
                            self.cached_data_timestamp = current_time
                        else:
                            cloudlog.warning("RTI: API request failed; keeping cached traffic data")
                            traffic_data = self.cached_traffic_data
                    except Exception as e:
                        cloudlog.error(f"RTI API error: {e}")
                        # Keep using cached data if available
                        traffic_data = self.cached_traffic_data
                else:
                    # Use cached data
                    traffic_data = self.cached_traffic_data
                    if traffic_data is not None:  # Check for None, not truthiness (empty list is valid)
                        if data_age >= self.max_data_age:
                            cloudlog.warning(f"RTI data is stale ({data_age:.0f}s old)")

            # Derive API status from client health (matches capnp enum values).
            if self.waze_client:
                api_status = self._map_health_to_api_status(self.waze_client.get_health_status())
            else:
                api_status = 'offline'

            # Get current road name from map data
            current_road_name = None
            try:
                map_data = self.sm['liveMapDataSP']
                if map_data and hasattr(map_data, 'roadName') and map_data.roadName:
                    current_road_name = map_data.roadName
                    cloudlog.debug(f"RTI: Current road name from mapd: '{current_road_name}'")
            except Exception as e:
                cloudlog.debug(f"RTI: Could not get road name from map data: {e}")

            # Process threats and determine recommendations
            rti_state = self.threat_detector.process_threats(
                traffic_data=traffic_data,
                current_location=location,
                current_speed=current_speed,
                timestamp=time.monotonic_ns(),  # Monotonic nanoseconds since boot
                v_cruise=cruise_cluster_speed,  # Driver's original set speed from cluster
                current_heading_deg=self._get_current_heading_deg(),
                posted_speed_limit=current_speed_limit,  # Pass actual posted speed limit
                current_road_name=current_road_name,  # Pass current road name for street matching
            )

            # Update API status
            rti_state.api_status = api_status
            rti_state.source = 'waze'

            # Store the processed state for continuous republishing
            self._last_processed_state = rti_state

        except Exception as e:
            cloudlog.error(f"RTI cycle error: {e}")
            self._last_processed_state = None

    def _normalize_180(self, angle):
        """Normalize angle to [-180, 180] range."""
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def _calculate_relative_bearing(self, ego_lat, ego_lon, threat_lat, threat_lon, ego_heading_deg):
        """Calculate relative bearing from ego to threat.

        Args:
            ego_lat: Ego vehicle latitude in degrees
            ego_lon: Ego vehicle longitude in degrees
            threat_lat: Threat latitude in degrees
            threat_lon: Threat longitude in degrees
            ego_heading_deg: Ego vehicle heading in degrees (0=north, clockwise)

        Returns:
            Relative bearing in degrees [-180, 180] where 0 is ahead
        """
        import math

        # Check if threat is at same position as ego
        if abs(ego_lat - threat_lat) < 1e-9 and abs(ego_lon - threat_lon) < 1e-9:
            return 0.0  # Default to ahead

        # Convert to radians
        lat1 = math.radians(ego_lat)
        lat2 = math.radians(threat_lat)
        lon1 = math.radians(ego_lon)
        lon2 = math.radians(threat_lon)
        dLon = lon2 - lon1

        # Calculate bearing from ego to threat using forward azimuth formula
        y = math.sin(dLon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

        # Calculate absolute bearing in degrees (0° = north, clockwise positive)
        bearing_deg = math.degrees(math.atan2(y, x))

        # Normalize bearing to [0, 360)
        if bearing_deg < 0:
            bearing_deg += 360.0

        # Calculate relative bearing (threat bearing - ego heading)
        rel_bearing = bearing_deg - ego_heading_deg

        # Normalize to [-180, 180] for shortest rotation
        return self._normalize_180(rel_bearing)

    def _angle_for_direction(self, direction):
        """Convert discrete direction to angle for fallback display.

        Args:
            direction: RtiStateSP.Direction enum value

        Returns:
            Angle in degrees for arrow display
        """
        from cereal import custom
        D = custom.RtiStateSP.Direction

        direction_angles = {
            D.ahead: 0.0,     # forward
            D.right: 90.0,    # right
            D.behind: 180.0,  # behind
            D.left: -90.0,    # left
            D.unknown: 0.0,   # default to ahead
        }

        return direction_angles.get(direction, 0.0)

    def _is_valid_gps_pair(self, lat1, lon1, lat2, lon2):
        """Check if two GPS coordinate pairs are valid.

        Args:
            lat1, lon1: First GPS coordinate pair
            lat2, lon2: Second GPS coordinate pair

        Returns:
            True if both pairs are valid GPS coordinates
        """
        # Check latitude bounds (-90 to 90)
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
            return False

        # Check longitude bounds (-180 to 180)
        if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            return False

        # Check for invalid (0,0) coordinates
        if (abs(lat1) < 0.001 and abs(lon1) < 0.001) or \
           (abs(lat2) < 0.001 and abs(lon2) < 0.001):
            return False

        return True

    def _calculate_threat_display_data(self, threat, ego_location, ego_heading_deg):
        """Calculate display angle for a threat.

        Args:
            threat: Threat object from threat detector
            ego_location: Tuple of (latitude, longitude) or None
            ego_heading_deg: Ego vehicle heading in degrees

        Returns:
            Tuple of (display_arrow_angle, has_location)
        """
        # Check if we have valid GPS data for both ego and threat
        has_valid_location = False
        display_angle = 0.0

        if ego_location and ego_heading_deg is not None and hasattr(threat, 'latitude') and hasattr(threat, 'longitude'):
            ego_lat, ego_lon = ego_location
            if self._is_valid_gps_pair(ego_lat, ego_lon, threat.latitude, threat.longitude):
                has_valid_location = True
                # Calculate precise bearing
                display_angle = self._calculate_relative_bearing(
                    ego_lat, ego_lon,
                    threat.latitude, threat.longitude,
                    ego_heading_deg
                )

        if not has_valid_location:
            # Fallback to discrete direction with slight variation
            base_angle = self._angle_for_direction(threat.direction)
            # Add small variation based on threat ID for visual distinction
            import hashlib
            hash_val = int(hashlib.md5(threat.id.encode()).hexdigest()[:8], 16)
            variation = ((hash_val % 21) - 10) * 0.5  # ±5 degree variation
            display_angle = base_angle + variation

        return display_angle, has_valid_location

    def _publish_rti_state(self, rti_state):
        """Publish RTI state message."""
        msg = messaging.new_message('rtiStateSP', valid=True)

        # Copy state data to message
        msg.rtiStateSP.timeStamp = rti_state.timestamp
        msg.rtiStateSP.threatAhead = rti_state.threat_ahead
        msg.rtiStateSP.threatDistanceM = rti_state.threat_distance_m
        msg.rtiStateSP.recommendedSpeed = rti_state.recommended_speed
        msg.rtiStateSP.source = rti_state.source
        msg.rtiStateSP.apiStatus = rti_state.api_status

        # Get current ego location and heading for bearing calculations
        ego_location = self._get_current_location()
        ego_heading_deg = self._get_current_heading_deg()

        # Add threat details for HUD with pre-computed display angles
        threats_to_send = rti_state.threats[:5]  # Limit to 5 threats
        if threats_to_send:
            msg.rtiStateSP.init('threats', len(threats_to_send))
            for i, threat in enumerate(threats_to_send):
                threat_msg = msg.rtiStateSP.threats[i]
                threat_msg.id = threat.id
                threat_msg.type = threat.type
                threat_msg.latitude = threat.latitude
                threat_msg.longitude = threat.longitude
                threat_msg.distance = threat.distance
                threat_msg.direction = threat.direction
                threat_msg.confidence = threat.confidence
                threat_msg.speedLimitMs = threat.speed_limit_ms

                # Publish same-road determination explicitly (no inference)
                try:
                    threat_msg.onSameRoad = bool(threat.on_same_road)
                except Exception:
                    threat_msg.onSameRoad = False

                # Calculate and add display angle for HUD arrow
                display_angle, has_location = self._calculate_threat_display_data(
                    threat, ego_location, ego_heading_deg
                )
                threat_msg.displayArrowAngle = display_angle
                threat_msg.hasLocation = has_location
                
                # Mark if this threat is causing the speed recommendation
                threat_msg.isCausingRecommendation = (threat.id == rti_state.active_threat_id)

        self.pm.send('rtiStateSP', msg)

    def _publish_offline_state(self):
        """Publish offline/disabled RTI state."""
        msg = messaging.new_message('rtiStateSP', valid=True)
        msg.rtiStateSP.timeStamp = time.monotonic_ns()
        msg.rtiStateSP.threatAhead = False
        msg.rtiStateSP.threatDistanceM = 0.0
        msg.rtiStateSP.recommendedSpeed = 0.0
        msg.rtiStateSP.source = 'rti'
        msg.rtiStateSP.apiStatus = 'offline'

        self.pm.send('rtiStateSP', msg)

    async def cleanup(self):
        """Cleanup resources on daemon shutdown."""
        if self.waze_client:
            close_fn = getattr(self.waze_client, "close", None)
            if callable(close_fn):
                maybe_coro = close_fn()
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro
            cloudlog.info("RTI Daemon cleaned up resources")

    async def run(self):
        """Main daemon loop running at 50Hz to match test script."""
        cloudlog.info("RTI Daemon starting main loop")

        # Initialize last published state to ensure continuous publishing
        last_rti_state = None
        last_publish_time = 0

        try:
            while True:
                loop_start = time.monotonic()

                # Check if RTI is enabled
                self.enabled = self._check_enabled()

                if not self.enabled:
                    # Publish disabled state continuously at 50Hz
                    try:
                        self._publish_offline_state()
                    except Exception as msg_e:
                        cloudlog.error(f"RTI failed to publish disabled state: {msg_e}")
                    await asyncio.sleep(0.02)
                    continue

                # Process RTI cycle (non-blocking)
                current_time = time.monotonic()

                # CRITICAL FIX: Always publish something at 50Hz to keep updated() flag true
                # This matches the continuous_rti_test.py behavior that works 100%

                # If we have cached state and haven't processed recently, republish last state
                if last_rti_state and (current_time - last_publish_time) >= 0.019:  # ~50Hz
                    # Update timestamp to current time for freshness
                    last_rti_state.timestamp = time.monotonic_ns()
                    try:
                        self._publish_rti_state(last_rti_state)
                        last_publish_time = current_time
                    except Exception as msg_e:
                        cloudlog.error(f"RTI failed to republish state: {msg_e}")

                # Process new data if it's time (separate from publishing)
                await self._process_cycle_async()

                # If we got new state from processing, update our cache
                if hasattr(self, '_last_processed_state') and self._last_processed_state:
                    last_rti_state = self._last_processed_state
                    # Publish the new state immediately
                    try:
                        self._publish_rti_state(last_rti_state)
                        last_publish_time = time.monotonic()
                    except Exception as msg_e:
                        cloudlog.error(f"RTI failed to publish new state: {msg_e}")

                # Maintain 50Hz loop timing to match test script
                loop_duration = time.monotonic() - loop_start
                sleep_time = max(0.0, 0.02 - loop_duration)

                if loop_duration > 0.01:  # Warn if processing takes > 10ms (half of 20ms cycle)
                    cloudlog.warning(f"RTI cycle took {loop_duration:.3f}s")

                await asyncio.sleep(sleep_time)
        finally:
            await self.cleanup()


def main():
    """Entry point for RTI daemon - compatible with process manager."""
    asyncio.run(main_async())


async def main_async():
    """Main async entry point for RTI daemon."""
    daemon = RTIDaemon()

    try:
        await daemon.run()
    except KeyboardInterrupt:
        cloudlog.info("RTI Daemon stopped by user")
    except Exception as e:
        cloudlog.error(f"RTI Daemon crashed: {e}")
        raise
    finally:
        # Ensure cleanup even if run() doesn't complete normally
        await daemon.cleanup()


if __name__ == "__main__":
    main()
