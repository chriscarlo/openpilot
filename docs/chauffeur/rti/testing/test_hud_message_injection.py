#!/usr/bin/env python3
"""
RTI HUD Widget Message Injection Test

This script injects cereal rtiStateSP messages into the live stream to test
the RTI HUD widget display without needing production code changes or builds.

Usage:
1. Run the UI with: FORCE_ONROAD_UI=1 ./selfdrive/ui/ui &
2. Run this script: python3 test_hud_message_injection.py
3. Watch the RTI widget in the HUD for threat messages

The script will publish various threat scenarios to test the widget.
"""

import time
import signal
import sys
from typing import Any

import cereal.messaging as messaging


class RTIHUDMessageInjector:
    """Injects RTI state messages to test HUD widget display."""

    def __init__(self):
        self.pm = messaging.PubMaster(['rtiStateSP'])
        self.running = False
        self.current_scenario = 0

        # Test scenarios to cycle through
        self.scenarios = [
            self.scenario_no_threats,
            self.scenario_single_police_close,
            self.scenario_single_camera_medium,
            self.scenario_single_accident_far,
            self.scenario_multiple_threats,
            self.scenario_construction_with_speed,
        ]

    def create_threat(self, threat_id: str, threat_type: str, distance: float,
                     latitude: float = 37.4231, longitude: float = -122.0841,
                     confidence: float = 0.85, speed_limit_ms: float = 15.0) -> dict[str, Any]:
        """Create a threat data structure for the message."""
        return {
            'id': threat_id,
            'type': threat_type,
            'latitude': latitude,
            'longitude': longitude,
            'distance': distance,
            'direction': 'ahead',  # Direction ahead for HUD testing
            'confidence': confidence,
            'speedLimitMs': speed_limit_ms
        }

    def publish_rti_state(self, threat_ahead: bool = False, threat_distance: float = 0.0,
                         recommended_speed: float = 0.0, threats: list[dict] = None,
                         api_status: str = 'connected', source: str = 'test'):
        """Publish an RTI state message."""
        if threats is None:
            threats = []

        # Create the message with valid=True for HUD display
        msg = messaging.new_message('rtiStateSP', valid=True)
        rti_state = msg.rtiStateSP

        # Set basic state
        rti_state.timeStamp = int(time.time() * 1e9)  # nanoseconds since boot
        rti_state.threatAhead = threat_ahead
        rti_state.threatDistanceM = threat_distance
        rti_state.recommendedSpeed = recommended_speed
        rti_state.source = source

        # Set API status
        if api_status == 'connected':
            rti_state.apiStatus = 'connected'
        elif api_status == 'offline':
            rti_state.apiStatus = 'offline'
        elif api_status == 'error':
            rti_state.apiStatus = 'error'
        else:
            rti_state.apiStatus = 'connected'

        # Add threats (use proper cereal message construction)
        if threats:
            rti_state.init('threats', len(threats))
            for i, threat_data in enumerate(threats):
                threat_msg = rti_state.threats[i]
                threat_msg.id = threat_data['id']
                threat_msg.type = threat_data['type']
                threat_msg.latitude = threat_data['latitude']
                threat_msg.longitude = threat_data['longitude']
                threat_msg.distance = threat_data['distance']
                threat_msg.direction = threat_data['direction']
                threat_msg.confidence = threat_data['confidence']
                threat_msg.speedLimitMs = threat_data['speedLimitMs']

        # Send the message
        self.pm.send('rtiStateSP', msg)

    def scenario_no_threats(self):
        """Test scenario: No threats detected."""
        print("Publishing: No threats detected")
        self.publish_rti_state(
            threat_ahead=False,
            threat_distance=0.0,
            recommended_speed=0.0,
            threats=[],
            api_status='connected'
        )

    def scenario_single_police_close(self):
        """Test scenario: Single police threat very close (should be red)."""
        print("Publishing: POLICE threat at 150m (CRITICAL - should show RED)")
        threat = self.create_threat(
            threat_id='police_close_001',
            threat_type='police',
            distance=150.0,  # Close = red color
            confidence=0.9,
            speed_limit_ms=11.2  # 25 mph
        )
        self.publish_rti_state(
            threat_ahead=True,
            threat_distance=150.0,
            recommended_speed=11.2,  # 25 mph in m/s
            threats=[threat],
            api_status='connected'
        )

    def scenario_single_camera_medium(self):
        """Test scenario: Single speed camera at medium distance (should be orange)."""
        print("Publishing: SPEED CAMERA at 250m (NEAR - should show ORANGE)")
        threat = self.create_threat(
            threat_id='camera_medium_001',
            threat_type='speedCamera',
            distance=250.0,  # Medium = orange color
            confidence=0.95,
            speed_limit_ms=13.4  # 30 mph
        )
        self.publish_rti_state(
            threat_ahead=True,
            threat_distance=250.0,
            recommended_speed=13.4,  # 30 mph in m/s
            threats=[threat],
            api_status='connected'
        )

    def scenario_single_accident_far(self):
        """Test scenario: Single accident at far distance (should be yellow)."""
        print("Publishing: ACCIDENT at 800m (NORMAL - should show YELLOW)")
        threat = self.create_threat(
            threat_id='accident_far_001',
            threat_type='accident',
            distance=800.0,  # Far = yellow color
            confidence=0.75,
            speed_limit_ms=8.9  # 20 mph
        )
        self.publish_rti_state(
            threat_ahead=True,
            threat_distance=800.0,
            recommended_speed=8.9,  # 20 mph in m/s
            threats=[threat],
            api_status='connected'
        )

    def scenario_multiple_threats(self):
        """Test scenario: Multiple threats to test multi-threat widget."""
        print("Publishing: MULTIPLE THREATS (testing multi-threat widget)")
        threats = [
            self.create_threat(
                threat_id='police_closest_001',
                threat_type='police',
                distance=200.0,  # Closest - orange
                latitude=37.4233,
                confidence=0.9,
                speed_limit_ms=11.2  # 25 mph
            ),
            self.create_threat(
                threat_id='camera_second_001',
                threat_type='speedCamera',
                distance=450.0,  # Medium - yellow
                latitude=37.4235,
                confidence=0.85,
                speed_limit_ms=13.4  # 30 mph
            ),
            self.create_threat(
                threat_id='construction_third_001',
                threat_type='construction',
                distance=750.0,  # Far - yellow
                latitude=37.4237,
                confidence=0.8,
                speed_limit_ms=8.9  # 20 mph
            ),
            self.create_threat(
                threat_id='accident_fourth_001',
                threat_type='accident',
                distance=1200.0,  # Very far - gray
                latitude=37.4240,
                confidence=0.7,
                speed_limit_ms=6.7  # 15 mph
            ),
        ]
        # Use closest threat for main state
        self.publish_rti_state(
            threat_ahead=True,
            threat_distance=200.0,  # Distance to closest threat
            recommended_speed=11.2,  # Speed for closest threat
            threats=threats,
            api_status='connected'
        )

    def scenario_construction_with_speed(self):
        """Test scenario: Construction zone with active speed recommendation."""
        print("Publishing: CONSTRUCTION with speed reduction (testing speed display)")
        threat = self.create_threat(
            threat_id='construction_speed_001',
            threat_type='construction',
            distance=300.0,  # Orange zone
            confidence=0.85,
            speed_limit_ms=8.9  # 20 mph - significant reduction
        )
        self.publish_rti_state(
            threat_ahead=True,
            threat_distance=300.0,
            recommended_speed=8.9,  # 20 mph - should show speed reduction
            threats=[threat],
            api_status='connected'
        )

    def run_test_sequence(self):
        """Run through all test scenarios."""
        print("Starting RTI HUD Message Injection Test")
        print("=" * 60)
        print("Make sure the UI is running with: FORCE_ONROAD_UI=1 ./selfdrive/ui/ui &")
        print("Watch the RTI widget in the bottom-left corner of the HUD")
        print("=" * 60)
        print()

        self.running = True
        cycle_count = 1

        try:
            while self.running:
                print(f"\nCYCLE {cycle_count} - Scenario {self.current_scenario + 1}/{len(self.scenarios)}")
                print("-" * 40)

                # Run current scenario
                self.scenarios[self.current_scenario]()

                # Wait for observation
                print("Holding for 8 seconds... (Check HUD widget now!)")
                for i in range(8):
                    if not self.running:
                        break
                    time.sleep(1)
                    if i < 7:  # Don't print on last iteration
                        print(f"   {7-i} seconds remaining...")

                # Move to next scenario
                self.current_scenario = (self.current_scenario + 1) % len(self.scenarios)

                # Start new cycle when we complete all scenarios
                if self.current_scenario == 0:
                    cycle_count += 1
                    print(f"\nCompleted cycle {cycle_count - 1}, starting cycle {cycle_count}")

        except KeyboardInterrupt:
            print("\n\nTest stopped by user")
        finally:
            self.cleanup()

    def run_single_threat_test(self, threat_type: str = 'police', distance: float = 300.0):
        """Run a simple single threat test for quick verification."""
        print(f"Running single threat test: {threat_type.upper()} at {distance}m")
        print("=" * 50)
        print("Make sure the UI is running with: FORCE_ONROAD_UI=1 ./selfdrive/ui/ui &")
        print("=" * 50)

        # Determine color based on distance
        if distance < 100:
            color = "RED (CRITICAL)"
        elif distance < 300:
            color = "ORANGE (NEAR)"
        elif distance < 1000:
            color = "YELLOW (NORMAL)"
        else:
            color = "GRAY (FAR)"

        threat = self.create_threat(
            threat_id=f'{threat_type}_test_001',
            threat_type=threat_type,
            distance=distance,
            confidence=0.85,
            speed_limit_ms=11.2  # 25 mph
        )

        try:
            self.running = True
            count = 0

            while self.running and count < 20:  # Run for ~40 seconds
                print(f"Publishing {threat_type.upper()} threat at {distance}m (should show {color})")

                self.publish_rti_state(
                    threat_ahead=True,
                    threat_distance=distance,
                    recommended_speed=11.2,
                    threats=[threat],
                    api_status='connected'
                )

                time.sleep(2)  # Publish every 2 seconds
                count += 1

        except KeyboardInterrupt:
            print("\nTest stopped by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up and send final 'no threats' message."""
        self.running = False
        print("\nCleaning up - sending 'no threats' message")
        self.publish_rti_state(
            threat_ahead=False,
            threat_distance=0.0,
            recommended_speed=0.0,
            threats=[],
            api_status='connected'
        )
        print("Cleanup complete")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nReceived interrupt signal, stopping test...")
    global injector
    if injector:
        injector.running = False


def main():
    global injector

    print("RTI HUD Widget Message Injection Test")
    print("=====================================")
    print()
    print("This script will inject RTI messages to test the HUD widget display.")
    print("Choose a test mode:")
    print()
    print("1. Full test sequence (cycles through all scenarios)")
    print("2. Single POLICE threat test (quick verification)")
    print("3. Single CAMERA threat test")
    print("4. Multiple threats test")
    print("5. Custom single threat test")
    print()

    # Check if running interactively or with arguments
    if len(sys.argv) > 1:
        # Command line mode - default to police test
        choice = '2'
        print("Non-interactive mode: Running POLICE threat test...")
    else:
        try:
            choice = input("Enter choice (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    injector = RTIHUDMessageInjector()

    try:
        if choice == '1':
            injector.run_test_sequence()
        elif choice == '2':
            injector.run_single_threat_test('police', 300.0)
        elif choice == '3':
            injector.run_single_threat_test('speedCamera', 400.0)
        elif choice == '4':
            injector.scenario_multiple_threats()
            print("Published multiple threats message. Check HUD widget!")
            time.sleep(10)
        elif choice == '5':
            print("\nCustom threat test:")
            threat_type = input("Threat type (police/speedCamera/accident/construction): ").strip().lower()
            if threat_type not in ['police', 'speedcamera', 'accident', 'construction']:
                threat_type = 'police'
            try:
                distance = float(input("Distance in meters (e.g., 300): ").strip())
            except ValueError:
                distance = 300.0
            injector.run_single_threat_test(threat_type, distance)
        else:
            print("Invalid choice, running single POLICE test...")
            injector.run_single_threat_test('police', 300.0)

    except Exception as e:
        print(f"\nError: {e}")
        injector.cleanup()


if __name__ == "__main__":
    main()
