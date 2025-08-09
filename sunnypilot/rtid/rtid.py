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

from cereal import log, messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from .waze_api_client import WazeAPIClient
from .threat_detector import ThreatDetector


class RTIDaemon:
    """Main RTI daemon class following openpilot patterns."""

    def __init__(self):
        """Initialize RTI daemon with messaging and configuration."""
        self.params = Params()

        # Messaging setup
        self.sm = messaging.SubMaster([
            'gpsLocationExternal',
            'liveLocationKalman',
            'carState'
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

        cloudlog.info("RTI Daemon initialized")

    def _load_api_key(self) -> str | None:
        """Load Waze API key from environment-aware location."""
        key_paths = [
            '/persist/waze_api_key.json',  # Production TICI
            '/data/persist/waze_api_key.json',  # Development
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
            if gps_ext.accuracy < 10.0:  # Only use if accuracy is reasonable
                return (gps_ext.latitude, gps_ext.longitude)

        # Fall back to Kalman filter location
        if self.sm.updated['liveLocationKalman']:
            live_loc = self.sm['liveLocationKalman']
            if live_loc.status == log.LiveLocationKalman.Status.valid:
                return (live_loc.lat, live_loc.lon)

        return None

    def _get_current_speed(self) -> float:
        """Get current vehicle speed in m/s."""
        self.sm.update(0)
        if self.sm.updated['carState']:
            # vEgo is in m/s
            return self.sm['carState'].vEgo
        return 0.0

    async def _process_cycle(self):
        """Main processing cycle: fetch → detect → publish."""
        current_time = time.time()

        try:
            # Get current vehicle state
            location = self._get_current_location()
            current_speed = self._get_current_speed()

            if location is None:
                # No valid GPS - publish offline state
                self._publish_offline_state()
                return

            # Fetch traffic data if API available
            traffic_data = None
            api_status = 'offline'

            if self.waze_client:
                try:
                    traffic_data = await self.waze_client.get_traffic_alerts(
                        location[0], location[1]
                    )
                    api_status = 'connected'
                except Exception as e:
                    cloudlog.error(f"RTI API error: {e}")
                    api_status = 'error'

            # Process threats and determine recommendations
            rti_state = self.threat_detector.process_threats(
                traffic_data=traffic_data,
                current_location=location,
                current_speed=current_speed,
                timestamp=int(current_time * 1e9)  # Convert to nanoseconds
            )

            # Update API status
            rti_state.api_status = api_status
            rti_state.source = 'waze'

            # Publish RTI state
            self._publish_rti_state(rti_state)

            self.loop_count += 1
            if self.loop_count % 60 == 0:  # Log status every minute
                # Get cache statistics if available
                cache_stats = self.waze_client.cache.get_stats() if self.waze_client else {}
                cache_info = f", Cache: {cache_stats.get('entries', 0)} items, {cache_stats.get('memory_mb', 0):.1f}MB" if cache_stats else ""
                
                cloudlog.info(f"RTI processed {self.loop_count} cycles, "
                             f"API: {api_status}, Location: {location}{cache_info}")

        except Exception as e:
            cloudlog.error(f"RTI cycle error: {e}")
            self._publish_offline_state()

    def _publish_rti_state(self, rti_state):
        """Publish RTI state message."""
        msg = messaging.new_message('rtiStateSP')

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

        self.pm.send('rtiStateSP', msg)

    def _publish_offline_state(self):
        """Publish offline/disabled RTI state."""
        msg = messaging.new_message('rtiStateSP')
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
        """Main daemon loop running at 1Hz."""
        cloudlog.info("RTI Daemon starting main loop")

        try:
            while True:
                loop_start = time.time()

                # Check if RTI is enabled
                self.enabled = self._check_enabled()

                if not self.enabled:
                    # Publish disabled state and sleep
                    self._publish_offline_state()
                    await asyncio.sleep(1.0)
                    continue

                # Process RTI cycle
                await self._process_cycle()

                # Maintain 1Hz loop timing
                loop_duration = time.time() - loop_start
                sleep_time = max(0.0, 1.0 - loop_duration)

                if loop_duration > 0.1:  # Warn if processing takes > 100ms
                    cloudlog.warning(f"RTI cycle took {loop_duration:.3f}s")

                await asyncio.sleep(sleep_time)
        finally:
            await self.cleanup()


async def main():
    """Main entry point for RTI daemon."""
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


def daemon_main():
    """Synchronous entry point that runs the async main."""
    asyncio.run(main())


if __name__ == "__main__":
    daemon_main()
