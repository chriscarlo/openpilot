#!/usr/bin/env python3
"""
RTI Live Validation Test Suite

Simulates real-world conditions with mock Waze API server and vehicle movement.
Tests RTI system response to dynamic traffic conditions.
"""

import asyncio
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import math

import cereal.messaging as messaging
from openpilot.common.params import Params


class MockWazeAPIServer:
    """Mock Waze API server for testing with dynamic threat data."""

    def __init__(self, port: int = 8765):
        self.port = port
        self.server = None
        self.thread = None
        self.threats = []
        self.vehicle_location = (37.4221, -122.0841)
        self.time_offset = 0

    def add_threat(self, threat_type: str, lat: float, lon: float,
                   speed_limit: float = None, confidence: float = 0.85):
        """Add a threat to the mock API response."""
        threat = {
            'id': f'threat_{len(self.threats)}_{int(time.time()*1000)}',
            'type': threat_type,
            'location': {'y': lat, 'x': lon},
            'confidence': confidence,
            'reliability': int(confidence * 10),
            'reportTimestamp': int((time.time() + self.time_offset) * 1000)
        }

        if speed_limit:
            threat['speedLimit'] = speed_limit

        self.threats.append(threat)

    def update_vehicle_location(self, lat: float, lon: float):
        """Update vehicle location for distance-based threat filtering."""
        self.vehicle_location = (lat, lon)

    def clear_threats(self):
        """Clear all threats."""
        self.threats = []

    def get_threats_near_vehicle(self, radius_m: float = 2000) -> list[dict]:
        """Get threats within radius of vehicle."""
        nearby_threats = []

        for threat in self.threats:
            # Calculate distance to threat
            threat_lat = threat['location']['y']
            threat_lon = threat['location']['x']
            distance = self._calculate_distance(
                self.vehicle_location[0], self.vehicle_location[1],
                threat_lat, threat_lon
            )

            if distance <= radius_m:
                # Add distance to threat data
                threat_copy = threat.copy()
                threat_copy['distance'] = distance
                nearby_threats.append(threat_copy)

        return sorted(nearby_threats, key=lambda x: x['distance'])

    def _calculate_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in meters."""
        R = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def start(self):
        """Start the mock API server."""

        class MockWazeHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress log messages

            def do_GET(handler_self):
                if '/api/alerts' in handler_self.path:
                    # Return mock threat data
                    response_data = {
                        'alerts': self.get_threats_near_vehicle(),
                        'timestamp': int(time.time() * 1000)
                    }

                    handler_self.send_response(200)
                    handler_self.send_header('Content-Type', 'application/json')
                    handler_self.end_headers()
                    handler_self.wfile.write(json.dumps(response_data).encode())
                else:
                    handler_self.send_error(404)

        self.server = HTTPServer(('localhost', self.port), MockWazeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the mock API server."""
        if self.server:
            self.server.shutdown()
            self.thread.join(timeout=1)


class VehicleSimulator:
    """Simulates vehicle movement and state."""

    def __init__(self):
        self.lat = 37.4221
        self.lon = -122.0841
        self.speed_ms = 20.0  # 45 mph default
        self.heading = 0.0  # North
        self.pm = messaging.PubMaster(['gpsLocationExternal', 'carState'])

    def update_position(self, delta_time: float):
        """Update vehicle position based on speed and heading."""
        # Calculate distance traveled
        distance_m = self.speed_ms * delta_time

        # Convert to lat/lon change (approximation)
        lat_change = (distance_m / 111000) * math.cos(math.radians(self.heading))
        lon_change = (distance_m / (111000 * math.cos(math.radians(self.lat)))) * \
                    math.sin(math.radians(self.heading))

        self.lat += lat_change
        self.lon += lon_change

    def publish_state(self):
        """Publish current vehicle state to messaging."""
        # Publish GPS location
        gps_msg = messaging.new_message('gpsLocationExternal')
        gps_msg.gpsLocationExternal.latitude = self.lat
        gps_msg.gpsLocationExternal.longitude = self.lon
        gps_msg.gpsLocationExternal.accuracy = 5.0
        gps_msg.gpsLocationExternal.timestamp = int(time.time() * 1e6)
        self.pm.send('gpsLocationExternal', gps_msg)

        # Publish car state
        car_msg = messaging.new_message('carState')
        car_msg.carState.vEgo = self.speed_ms
        car_msg.carState.aEgo = 0.0
        car_msg.carState.steeringAngleDeg = 0.0
        self.pm.send('carState', car_msg)


class RTILiveValidator:
    """Validates RTI system with simulated live conditions."""

    def __init__(self):
        self.api_server = MockWazeAPIServer()
        self.vehicle = VehicleSimulator()
        self.sm = messaging.SubMaster(['rtiStateSP'])
        self.params = Params()
        self.test_results = []

    async def test_approaching_threat(self):
        """Test RTI response when approaching a threat."""
        print("\nTest 1: Approaching Police Threat")
        print("-" * 40)

        # Place a police threat 1km ahead
        threat_lat = self.vehicle.lat + 0.009  # ~1km north
        self.api_server.add_threat('POLICE', threat_lat, self.vehicle.lon,
                                  speed_limit=25.0, confidence=0.9)

        # Simulate approaching the threat
        distances = []
        speeds = []

        for i in range(20):  # 20 seconds of approach
            # Update vehicle position (moving north)
            self.vehicle.update_position(1.0)
            self.vehicle.publish_state()

            # Update API server with vehicle location
            self.api_server.update_vehicle_location(self.vehicle.lat, self.vehicle.lon)

            # Wait for RTI response
            await asyncio.sleep(1.0)
            self.sm.update(0)

            if self.sm.updated['rtiStateSP']:
                rti_state = self.sm['rtiStateSP']
                if rti_state.threatAhead:
                    distances.append(rti_state.threatDistanceM)
                    speeds.append(rti_state.recommendedSpeed)
                    print(f"  {i}s: Threat at {rti_state.threatDistanceM:.0f}m, "
                          f"Speed: {rti_state.recommendedSpeed * 2.237:.1f} mph")

        # Validate response
        if distances:
            # Distance should decrease
            distance_decreased = all(distances[i] >= distances[i+1]
                                    for i in range(len(distances)-1))

            # Speed should reduce as we get closer
            speed_reduced = speeds[-1] < speeds[0] if len(speeds) > 1 else False

            print("\nResults:")
            print(f"  Distance decreased: {'✓' if distance_decreased else '✗'}")
            print(f"  Speed reduced: {'✓' if speed_reduced else '✗'}")

            return distance_decreased and speed_reduced
        else:
            print("  ✗ No RTI response received")
            return False

    async def test_multiple_threats(self):
        """Test RTI prioritization with multiple threats."""
        print("\nTest 2: Multiple Threat Prioritization")
        print("-" * 40)

        # Clear previous threats
        self.api_server.clear_threats()

        # Add multiple threats at different distances
        self.api_server.add_threat('ACCIDENT',
                                  self.vehicle.lat + 0.0045,  # ~500m
                                  self.vehicle.lon, confidence=0.8)
        self.api_server.add_threat('POLICE',
                                  self.vehicle.lat + 0.0027,  # ~300m
                                  self.vehicle.lon, speed_limit=25.0, confidence=0.9)
        self.api_server.add_threat('SPEED_CAMERA',
                                  self.vehicle.lat + 0.0072,  # ~800m
                                  self.vehicle.lon, speed_limit=35.0, confidence=0.95)

        # Update and wait for response
        self.vehicle.publish_state()
        self.api_server.update_vehicle_location(self.vehicle.lat, self.vehicle.lon)

        await asyncio.sleep(2.0)
        self.sm.update(0)

        if self.sm.updated['rtiStateSP']:
            rti_state = self.sm['rtiStateSP']

            print(f"  Threat detected: {rti_state.threatAhead}")
            print(f"  Closest threat: {rti_state.threatDistanceM:.0f}m")
            print(f"  Threats in response: {len(rti_state.threats)}")

            # Should prioritize closest threat (POLICE at 300m)
            closest_is_police = (rti_state.threatDistanceM > 250 and
                                rti_state.threatDistanceM < 350)

            print("\nResults:")
            print(f"  Correct prioritization: {'✓' if closest_is_police else '✗'}")

            return closest_is_police
        else:
            print("  ✗ No RTI response received")
            return False

    async def test_api_failure_recovery(self):
        """Test RTI behavior during API failures."""
        print("\nTest 3: API Failure and Recovery")
        print("-" * 40)

        # Stop API server to simulate failure
        print("  Stopping API server...")
        self.api_server.stop()

        # Publish vehicle state
        self.vehicle.publish_state()

        await asyncio.sleep(2.0)
        self.sm.update(0)

        offline_state = None
        if self.sm.updated['rtiStateSP']:
            offline_state = self.sm['rtiStateSP']
            print(f"  API Status during outage: {offline_state.apiStatus}")

        # Restart API server
        print("  Restarting API server...")
        self.api_server = MockWazeAPIServer()
        self.api_server.start()
        self.api_server.add_threat('POLICE',
                                  self.vehicle.lat + 0.0045,
                                  self.vehicle.lon, confidence=0.9)

        # Wait for recovery
        await asyncio.sleep(3.0)

        self.vehicle.publish_state()
        self.api_server.update_vehicle_location(self.vehicle.lat, self.vehicle.lon)

        await asyncio.sleep(2.0)
        self.sm.update(0)

        recovered_state = None
        if self.sm.updated['rtiStateSP']:
            recovered_state = self.sm['rtiStateSP']
            print(f"  API Status after recovery: {recovered_state.apiStatus}")

        # Validate failure handling and recovery
        handled_failure = (offline_state and
                          offline_state.apiStatus in ['offline', 'error'])
        recovered = (recovered_state and
                    recovered_state.apiStatus == 'connected')

        print("\nResults:")
        print(f"  Handled failure: {'✓' if handled_failure else '✗'}")
        print(f"  Recovered successfully: {'✓' if recovered else '✗'}")

        return handled_failure and recovered

    async def run_validation_suite(self):
        """Run complete validation suite."""
        print("\n" + "="*60)
        print("RTI LIVE VALIDATION TEST SUITE")
        print("="*60)

        # Setup
        print("\nSetup:")
        print("  Starting mock Waze API server...")
        self.api_server.start()

        print("  Configuring RTI parameters...")
        self.params.put_bool("RTIEnabled", True)
        self.params.put("RTIDataSource", "4")  # Manual API

        # Create API key file for daemon
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'api_key': 'test-key',
                'endpoint': f'http://localhost:{self.api_server.port}/api/alerts'
            }, f)
            api_key_file = f.name

        try:
            # Run tests
            results = []

            # Test 1: Approaching threat
            result1 = await self.test_approaching_threat()
            results.append(('Approaching Threat', result1))

            # Test 2: Multiple threats
            result2 = await self.test_multiple_threats()
            results.append(('Multiple Threats', result2))

            # Test 3: API failure
            result3 = await self.test_api_failure_recovery()
            results.append(('API Recovery', result3))

            # Summary
            print("\n" + "="*60)
            print("VALIDATION RESULTS")
            print("="*60)

            for test_name, passed in results:
                status = '✓ PASSED' if passed else '✗ FAILED'
                print(f"  {test_name}: {status}")

            all_passed = all(r[1] for r in results)

            if all_passed:
                print("\n✓ All live validation tests PASSED")
            else:
                print("\n✗ Some tests FAILED - review results above")

            return all_passed

        finally:
            # Cleanup
            self.api_server.stop()
            os.unlink(api_key_file)
            self.params.put_bool("RTIEnabled", False)


async def main():
    """Run RTI live validation tests."""
    validator = RTILiveValidator()

    # Note: This would require RTI daemon to be running
    # For testing purposes, we'll simulate the validation

    success = await validator.run_validation_suite()

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
