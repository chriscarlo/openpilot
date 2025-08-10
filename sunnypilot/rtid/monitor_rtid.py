#!/usr/bin/env python3
"""
RTI Daemon Monitor - Real-time monitoring of RTI threat data processing

This script monitors the rtid process to verify it's properly:
1. Ingesting threat data from the Waze API endpoint
2. Processing and filtering threats based on location
3. Publishing threat information to openpilot via rtiStateSP messages

Usage: python monitor_rtid.py
"""

import asyncio
import json
import os
import sys
import time
from collections import deque
from datetime import datetime

from cereal import messaging
from openpilot.common.params import Params

# Add rtid module to path for direct access to components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from waze_api_client import WazeAPIClient
    DIRECT_API_TEST = True
except ImportError:
    DIRECT_API_TEST = False
    print("Warning: Cannot import WazeAPIClient directly - API testing limited")


class RTIDMonitor:
    """Monitor for RTI daemon traffic data processing."""

    def __init__(self):
        """Initialize monitoring components."""
        self.params = Params()

        # Subscribe to relevant messages
        self.sm = messaging.SubMaster([
            'rtiStateSP',
            'gpsLocationExternal',
            'gpsLocation',
            'carState'
        ], ignore_avg_freq=True)

        # State tracking
        self.last_rti_msg = None
        self.message_count = 0
        self.api_hit_count = 0
        self.api_error_count = 0
        self.threat_count = 0
        self.last_location = None
        self.threats_history = deque(maxlen=50)  # Keep last 50 threats

        # Performance metrics
        self.message_timestamps = deque(maxlen=100)
        self.api_response_times = deque(maxlen=20)

        # Direct API client for testing
        self.api_client = None
        if DIRECT_API_TEST:
            api_key = self._load_api_key()
            if api_key:
                self.api_client = WazeAPIClient(api_key)
                print("[OK] Direct API client initialized for testing")
            else:
                print("[ERROR] No API key found - direct API testing disabled")

    def _load_api_key(self) -> str | None:
        """Load API key for direct testing."""
        key_paths = [
            '/persist/waze_api_key.json',
            '/data/persist/waze_api_key.json',
        ]

        for path in key_paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                        return data.get('api_key')
                except Exception:
                    pass
        return None

    def _format_timestamp(self, ns_timestamp: int) -> str:
        """Convert nanosecond timestamp to readable format."""
        if ns_timestamp == 0:
            return "N/A"
        seconds = ns_timestamp / 1e9
        dt = datetime.fromtimestamp(seconds)
        return dt.strftime("%H:%M:%S.%f")[:-3]

    def _format_location(self, lat: float, lon: float) -> str:
        """Format GPS coordinates."""
        return f"{lat:.6f}, {lon:.6f}"

    def _format_distance(self, meters: float) -> str:
        """Format distance in human-readable form."""
        if meters < 1000:
            return f"{meters:.0f}m"
        else:
            return f"{meters/1000:.1f}km"

    def _get_message_rate(self) -> float:
        """Calculate message reception rate."""
        if len(self.message_timestamps) < 2:
            return 0.0

        time_span = self.message_timestamps[-1] - self.message_timestamps[0]
        if time_span > 0:
            return len(self.message_timestamps) / time_span
        return 0.0

    async def test_direct_api(self, lat: float, lon: float):
        """Test direct API connection and data retrieval."""
        if not self.api_client:
            return None, "No API client available"

        print("\n[API TEST] Testing direct Waze API connection...")
        print(f"   Location: {self._format_location(lat, lon)}")

        try:
            start_time = time.time()

            # Create session and fetch data
            async with self.api_client:
                alerts = await self.api_client.get_traffic_alerts(lat, lon)

            elapsed = (time.time() - start_time) * 1000
            self.api_response_times.append(elapsed)

            if alerts:
                print(f"   [OK] API Response: {elapsed:.0f}ms")
                print(f"   [OK] Alerts received: {len(alerts)}")

                # Show first few alerts
                for i, alert in enumerate(alerts[:3]):
                    print(f"\n   Alert #{i+1}:")
                    print(f"     Type: {alert.get('type', 'unknown')}")
                    print(f"     Subtype: {alert.get('subtype', 'N/A')}")

                    if 'location' in alert:
                        loc = alert['location']
                        print(f"     Location: {loc.get('lat', 0):.6f}, {loc.get('lon', 0):.6f}")

                    if 'confidence' in alert:
                        print(f"     Confidence: {alert['confidence']}")

                    if 'street' in alert:
                        print(f"     Street: {alert['street']}")

                # Check cache stats
                cache_stats = self.api_client.cache.get_stats()
                print("\n   [CACHE STATS]:")
                print(f"     Entries: {cache_stats['entries']}")
                print(f"     Memory: {cache_stats['memory_mb']:.2f}MB")
                print(f"     Hit rate: {cache_stats['hit_rate']:.1%}")

                return alerts, None
            else:
                print("   [WARNING] No alerts in this area")
                return [], None

        except Exception as e:
            error_msg = str(e)
            print(f"   [ERROR] API Error: {error_msg}")
            return None, error_msg

    def print_status_header(self):
        """Print monitoring header."""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("=" * 80)
        print("                    RTI DAEMON MONITOR - LIVE STATUS")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"RTI Enabled: {'[YES]' if self.params.get_bool('RTIEnabled') else '[NO]'}")
        print("-" * 80)

    def print_connection_status(self):
        """Print API connection status."""
        print("\n[CONNECTION STATUS]")
        print("-" * 40)

        if self.last_rti_msg:
            api_status = self.last_rti_msg.rtiStateSP.apiStatus
            source = self.last_rti_msg.rtiStateSP.source

            status_icon = "[OK]" if api_status == "connected" else "[ERROR]"
            print(f"API Status: {status_icon} {api_status.upper()}")
            print(f"Data Source: {source}")
            print(f"Messages Received: {self.message_count}")
            print(f"Message Rate: {self._get_message_rate():.1f} Hz")

            if self.api_response_times:
                avg_response = sum(self.api_response_times) / len(self.api_response_times)
                print(f"Avg API Response: {avg_response:.0f}ms")
        else:
            print("[WARNING] No RTI messages received yet")

    def print_location_status(self):
        """Print current location information."""
        print("\n[LOCATION STATUS]")
        print("-" * 40)

        # Get GPS location
        if self.sm.updated['gpsLocationExternal']:
            gps = self.sm['gpsLocationExternal']
            self.last_location = (gps.latitude, gps.longitude)
            print("GPS Source: External")
            print(f"Location: {self._format_location(gps.latitude, gps.longitude)}")
            print(f"Accuracy: {gps.accuracy:.1f}m")
        elif self.sm.updated['gpsLocation']:
            gps = self.sm['gpsLocation']
            if gps.hasFix:
                self.last_location = (gps.latitude, gps.longitude)
                print("GPS Source: Internal")
                print(f"Location: {self._format_location(gps.latitude, gps.longitude)}")
        else:
            print("[WARNING] No GPS fix available")

        # Get vehicle speed
        if self.sm.updated['carState']:
            speed_ms = self.sm['carState'].vEgo
            speed_kph = speed_ms * 3.6
            speed_mph = speed_ms * 2.237
            print(f"Vehicle Speed: {speed_kph:.1f} km/h ({speed_mph:.1f} mph)")

    def print_threat_status(self):
        """Print current threat information."""
        print("\n[THREAT STATUS]")
        print("-" * 40)

        if not self.last_rti_msg:
            print("No threat data available")
            return

        rti = self.last_rti_msg.rtiStateSP

        if rti.threatAhead:
            print("[ALERT] THREAT DETECTED")
            print(f"Distance: {self._format_distance(rti.threatDistanceM)}")
            print(f"Recommended Speed: {rti.recommendedSpeed * 3.6:.1f} km/h")

            # Show threat details
            if len(rti.threats) > 0:
                print(f"\nActive Threats ({len(rti.threats)}):")
                for i, threat in enumerate(rti.threats[:5]):
                    print(f"\n  Threat #{i+1}:")
                    print(f"    Type: {threat.type}")
                    print(f"    Distance: {self._format_distance(threat.distance)}")
                    print(f"    Direction: {threat.direction} degrees")
                    print(f"    Confidence: {threat.confidence:.1%}")

                    if threat.speedLimitMs > 0:
                        speed_kph = threat.speedLimitMs * 3.6
                        print(f"    Speed Limit: {speed_kph:.0f} km/h")
        else:
            print("[OK] No threats detected")
            print(f"Last check: {self._format_timestamp(rti.timeStamp)}")

    def print_history(self):
        """Print threat history."""
        if self.threats_history:
            print("\n[THREAT HISTORY] (Last 10)")
            print("-" * 40)

            for i, (timestamp, threat_type, distance) in enumerate(list(self.threats_history)[-10:]):
                time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                print(f"{time_str} - {threat_type} @ {self._format_distance(distance)}")

    def print_diagnostics(self):
        """Print diagnostic information."""
        print("\n[DIAGNOSTICS]")
        print("-" * 40)

        # Check if rtid process is likely running
        rtid_running = self.message_count > 0 and (time.time() - self.message_timestamps[-1] < 5) if self.message_timestamps else False

        print(f"RTI Process: {'[RUNNING]' if rtid_running else '[NOT DETECTED]'}")
        print(f"Total Threats Detected: {self.threat_count}")

        # Check for issues
        issues = []
        if not self.params.get_bool('RTIEnabled'):
            issues.append("RTI is disabled in settings")
        if not rtid_running:
            issues.append("No recent RTI messages received")
        if self.last_rti_msg and self.last_rti_msg.rtiStateSP.apiStatus == "offline":
            issues.append("API connection offline")
        if not self.last_location:
            issues.append("No GPS location available")

        if issues:
            print("\n[ISSUES DETECTED]:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n[OK] All systems operational")

    async def monitor_loop(self):
        """Main monitoring loop."""
        print("Starting RTI monitor... Press Ctrl+C to exit")

        # Test direct API if available and we have location
        test_performed = False

        while True:
            try:
                # Update messages
                self.sm.update(0)

                # Track RTI messages
                if self.sm.updated['rtiStateSP']:
                    self.last_rti_msg = self.sm['rtiStateSP']
                    self.message_count += 1
                    self.message_timestamps.append(time.time())

                    # Track threats
                    rti = self.last_rti_msg.rtiStateSP
                    if rti.threatAhead:
                        self.threat_count += 1

                        # Add to history
                        if len(rti.threats) > 0:
                            threat = rti.threats[0]
                            self.threats_history.append((
                                time.time(),
                                threat.type,
                                threat.distance
                            ))

                    # Track API status
                    if rti.apiStatus == "connected":
                        self.api_hit_count += 1
                    elif rti.apiStatus == "error":
                        self.api_error_count += 1

                # Display status every second
                if self.message_count % 10 == 0 or self.message_count == 1:
                    self.print_status_header()
                    self.print_connection_status()
                    self.print_location_status()
                    self.print_threat_status()
                    self.print_history()
                    self.print_diagnostics()

                    # Run direct API test once when we have location
                    if not test_performed and self.last_location and DIRECT_API_TEST:
                        await self.test_direct_api(self.last_location[0], self.last_location[1])
                        test_performed = True

                await asyncio.sleep(0.1)  # 10Hz update rate

            except KeyboardInterrupt:
                print("\n\nMonitoring stopped.")
                break
            except Exception as e:
                print(f"\nError in monitor loop: {e}")
                await asyncio.sleep(1)


async def main():
    """Main entry point."""
    monitor = RTIDMonitor()
    await monitor.monitor_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting RTI monitor...")
