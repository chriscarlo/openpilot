#!/usr/bin/env python3
"""
RTI Daemon - Realtime Traffic Intelligence

Fetches traffic alert data from external sources (Waze API), processes threats
for relevance to current route, and publishes RTI state for consumption by
the longitudinal planner.

Architecture: fetch → detect → publish at 1Hz
"""

import asyncio
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

        api_calls_per_hour = 3600 / self.api_fetch_interval
        cloudlog.info(f"RTI Daemon initialized - API interval: {self.api_fetch_interval}s ({api_calls_per_hour:.0f} calls/hr max)")

    def _load_api_key(self) -> str | None:
        """Load Waze API key from environment-aware location."""
        key_paths = [
            '/persist/waze/waze_rapidapi.json',  # Production TICI
            '/data/persist/waze/waze_rapidapi.json',  # Development fallback
        ]

        for key_path in key_paths:
            if os.path.exists(key_path):
                try:
                    with open(key_path) as f:
                        key_data = json.load(f)
                        api_key = key_data.get('api_key')
                        if api_key:
                            cloudlog.info(f"RTI API key loaded from {key_path}")
                            return api_key
                except (OSError, json.JSONDecodeError, KeyError) as e:
                    cloudlog.error(f"RTI failed to load API key from {key_path}: {e}")

        cloudlog.warning("RTI API key not found - running in offline mode")
        return None

    def _check_enabled(self) -> bool:
        """Check if RTI is enabled via params."""
        return self.params.get_bool("RTIEnabled")

    def _get_current_location(self) -> tuple[float, float] | None:
        """Get current GPS coordinates from location services."""
        self.sm.update(0)  # Non-blocking update

        # Prefer external GPS if available
        if self.sm.updated['gpsLocationExternal']:
            gps_ext = self.sm['gpsLocationExternal']
            if getattr(gps_ext, 'accuracy', 99.9) < 10.0:  # Only use if accuracy is reasonable
                return (gps_ext.latitude, gps_ext.longitude)

        # Fall back to regular GPS location
        if self.sm.updated['gpsLocation']:
            gps_loc = self.sm['gpsLocation']
            if gps_loc.hasFix:
                return (gps_loc.latitude, gps_loc.longitude)

        return None

    def _get_current_heading_deg(self) -> float | None:
        """Get current ego heading in degrees from GPS if available."""
        self.sm.update(0)
        # Prefer external GPS bearing if available and plausible
        if self.sm.updated['gpsLocationExternal']:
            gps_ext = self.sm['gpsLocationExternal']
            try:
                bearing = getattr(gps_ext, 'bearingDeg')
                # Validate numeric and finite
                if bearing is not None and 0.0 <= float(bearing) <= 360.0:
                    return float(bearing)
            except Exception:
                pass

        # Fall back to internal GPS if it exposes bearing
        if self.sm.updated['gpsLocation']:
            gps_loc = self.sm['gpsLocation']
            try:
                bearing = getattr(gps_loc, 'bearingDeg')
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
            return self.sm['carState'].vEgo
        return 0.0

    def _get_cruise_cluster_speed(self) -> float:
        """Get driver's originally set cruise speed (as shown on cluster) in m/s."""
        self.sm.update(0)
        if self.sm.updated['carState']:
            cruise_state = self.sm['carState'].cruiseState
            # speedCluster is the driver's set speed shown on the instrument cluster
            # This is the original driver-set maximum, unmodified by controllers
            if cruise_state.enabled and cruise_state.speedCluster > 0:
                return cruise_state.speedCluster
        return 0.0

    def _get_current_speed_limit(self) -> float:
        """Get current posted speed limit from both map and dashboard sources.
        
        Uses CONSERVATIVE combination: When both sources have data, uses the LOWER value
        for maximum safety in threat detection scenarios.
        
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
        
        # CONSERVATIVE COMBINATION: Use MIN instead of MAX for RTI safety
        # This differs from SLC which uses MAX (higher) value
        # RTI prefers the MORE CONSERVATIVE (lower) limit when sources disagree
        if map_limit > 0 and dashboard_limit > 0:
            # Both sources have data - use the LOWER value
            combined_limit = min(map_limit, dashboard_limit)
            source = "map" if map_limit <= dashboard_limit else "dashboard"
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
        current_time = time.time()

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
            api_status = 'offline'
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
                        
                        traffic_data = await self.waze_client.get_traffic_alerts(
                            location[0], location[1], radius_km
                        )
                        # Update cache
                        self.cached_traffic_data = traffic_data
                        self.cached_data_location = location
                        self.cached_data_timestamp = current_time
                        self.last_api_fetch_time = current_time
                        api_status = 'connected'
                    except Exception as e:
                        cloudlog.error(f"RTI API error: {e}")
                        api_status = 'error'
                        # Keep using cached data if available
                        traffic_data = self.cached_traffic_data
                else:
                    # Use cached data
                    traffic_data = self.cached_traffic_data
                    if traffic_data is not None:  # Check for None, not truthiness (empty list is valid)
                        if data_age < self.max_data_age:
                            # Cached data is still usable - keep reporting as connected
                            api_status = 'connected'
                        else:
                            # Data too old - effectively offline
                            api_status = 'offline'
                            cloudlog.warning(f"RTI data is stale ({data_age:.0f}s old)")

            # Process threats and determine recommendations
            rti_state = self.threat_detector.process_threats(
                traffic_data=traffic_data,
                current_location=location,
                current_speed=current_speed,
                timestamp=int(current_time * 1e9),  # Convert to nanoseconds
                v_cruise=cruise_cluster_speed,  # Driver's original set speed from cluster
                current_heading_deg=self._get_current_heading_deg(),
                posted_speed_limit=current_speed_limit,  # Pass actual posted speed limit
            )

            # Update API status
            rti_state.api_status = api_status
            rti_state.source = 'waze'

            # Store the processed state for continuous republishing
            self._last_processed_state = rti_state

        except Exception as e:
            cloudlog.error(f"RTI cycle error: {e}")
            self._last_processed_state = None

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

        # Add threat details for HUD
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

        self.pm.send('rtiStateSP', msg)

    def _publish_offline_state(self):
        """Publish offline/disabled RTI state."""
        msg = messaging.new_message('rtiStateSP', valid=True)
        msg.rtiStateSP.timeStamp = int(time.time() * 1e9)
        msg.rtiStateSP.threatAhead = False
        msg.rtiStateSP.threatDistanceM = 0.0
        msg.rtiStateSP.recommendedSpeed = 0.0
        msg.rtiStateSP.source = 'rti'
        msg.rtiStateSP.apiStatus = 'offline'

        self.pm.send('rtiStateSP', msg)

    async def cleanup(self):
        """Cleanup resources on daemon shutdown."""
        if self.waze_client:
            await self.waze_client.close()
            cloudlog.info("RTI Daemon cleaned up resources")

    async def run(self):
        """Main daemon loop running at 50Hz to match test script."""
        cloudlog.info("RTI Daemon starting main loop")
        
        # Initialize last published state to ensure continuous publishing
        last_rti_state = None
        last_publish_time = 0

        try:
            while True:
                loop_start = time.time()

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
                current_time = time.time()
                
                # CRITICAL FIX: Always publish something at 50Hz to keep updated() flag true
                # This matches the continuous_rti_test.py behavior that works 100%
                
                # If we have cached state and haven't processed recently, republish last state
                if last_rti_state and (current_time - last_publish_time) >= 0.019:  # ~50Hz
                    # Update timestamp to current time for freshness
                    last_rti_state.timestamp = int(current_time * 1e9)
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
                        last_publish_time = time.time()
                    except Exception as msg_e:
                        cloudlog.error(f"RTI failed to publish new state: {msg_e}")

                # Maintain 50Hz loop timing to match test script
                loop_duration = time.time() - loop_start
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
