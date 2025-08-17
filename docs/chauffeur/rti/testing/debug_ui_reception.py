#!/usr/bin/env python3

"""
Debug UI Message Reception

Monitors what the UI SubMaster actually receives vs what we send.
This will help identify if there's a barrier between message publishing and UI processing.

Usage: python3 debug_ui_reception.py
"""

import time
import threading
from cereal import messaging

class UIReceptionDebugger:
    def __init__(self):
        self.pm = messaging.PubMaster(['rtiStateSP'])
        self.sm = messaging.SubMaster(['rtiStateSP'])
        self.message_sent = False
        self.message_received = False
        self.last_received_data = None
        self.monitoring = True

    def monitor_reception(self):
        """Monitor incoming messages in separate thread."""
        print("Starting message reception monitor...")

        while self.monitoring:
            self.sm.update(0)  # Non-blocking

            if self.sm.updated['rtiStateSP']:
                self.message_received = True
                try:
                    msg = self.sm['rtiStateSP']

                    print(f"\n[RECEIVED] Message at {time.time():.3f}")
                    print(f"  Valid: {self.sm.valid['rtiStateSP']}")
                    print(f"  Updated: {self.sm.updated['rtiStateSP']}")
                    print(f"  Source: {msg.rtiStateSP.source}")
                    print(f"  ThreatAhead: {msg.rtiStateSP.threatAhead}")
                    print(f"  Threats count: {len(msg.rtiStateSP.threats)}")

                    if len(msg.rtiStateSP.threats) > 0:
                        threat = msg.rtiStateSP.threats[0]
                        print(f"  Threat[0]: {threat.type} at {threat.distance}m")

                    self.last_received_data = msg

                except Exception as e:
                    print(f"Error processing received message: {e}")

            time.sleep(0.05)  # 20Hz monitoring

    def send_test_message(self, test_name: str, config: dict):
        """Send a test message with specific configuration."""
        print(f"\n[SENDING] {test_name}")

        # Create message
        msg = messaging.new_message('rtiStateSP', valid=config.get('valid', True))
        rti_state = msg.rtiStateSP

        # Basic state
        rti_state.timeStamp = int(time.time() * 1e9)
        rti_state.threatAhead = config.get('threatAhead', False)
        rti_state.threatDistanceM = config.get('threatDistanceM', 0.0)
        rti_state.recommendedSpeed = config.get('recommendedSpeed', 0.0)
        rti_state.source = config.get('source', f'debug_{test_name}')
        rti_state.apiStatus = config.get('apiStatus', 'connected')

        # Add threats if specified
        if config.get('add_threats', False):
            rti_state.init('threats', 1)
            threat = rti_state.threats[0]
            threat.id = f'debug_{test_name}_threat'
            threat.type = config.get('threat_type', 'police')
            threat.latitude = 37.4231
            threat.longitude = -122.0841
            threat.distance = config.get('threat_distance', 250.0)
            threat.direction = 'ahead'
            threat.confidence = 0.9
            threat.speedLimitMs = 15.0

        # Send message
        print(f"  Message valid flag: {msg.valid}")
        print(f"  ThreatAhead: {rti_state.threatAhead}")
        print(f"  Threats: {len(rti_state.threats) if hasattr(rti_state, 'threats') else 0}")

        self.pm.send('rtiStateSP', msg)
        self.message_sent = True

        # Wait for reception
        start_time = time.time()
        received = False
        while time.time() - start_time < 2.0:  # Wait up to 2 seconds
            if self.last_received_data and self.last_received_data.rtiStateSP.source == rti_state.source:
                received = True
                break
            time.sleep(0.1)

        if received:
            print("  [SUCCESS] Message received by SubMaster")
        else:
            print("  [FAILURE] Message NOT received within 2 seconds")

        return received

    def run_debug_sequence(self):
        """Run comprehensive debug sequence."""
        print("=== UI Reception Debug Sequence ===")

        # Start monitoring in background
        monitor_thread = threading.Thread(target=self.monitor_reception, daemon=True)
        monitor_thread.start()

        time.sleep(1)  # Let monitor start

        # Test 1: Basic valid message
        test1_success = self.send_test_message("basic_valid", {
            'valid': True,
            'threatAhead': False,
            'add_threats': False
        })

        time.sleep(2)

        # Test 2: Valid message with threats (safe mode)
        test2_success = self.send_test_message("safe_with_threats", {
            'valid': True,
            'threatAhead': False,  # Safe for speed control
            'threatDistanceM': 0.0,  # Safe for speed control
            'add_threats': True,
            'threat_type': 'police',
            'threat_distance': 250.0
        })

        time.sleep(2)

        # Test 3: Invalid message (for comparison)
        test3_success = self.send_test_message("invalid_message", {
            'valid': False,
            'threatAhead': False,
            'add_threats': True
        })

        time.sleep(2)

        # Test 4: Message like real rtid would send
        test4_success = self.send_test_message("rtid_style", {
            'valid': True,
            'threatAhead': True,  # Like real rtid
            'threatDistanceM': 300.0,
            'recommendedSpeed': 15.0,
            'add_threats': True,
            'source': 'rtid',  # Mimic real rtid
            'apiStatus': 'connected'
        })

        # Stop monitoring
        self.monitoring = False
        time.sleep(0.5)

        # Summary
        print("\n=== Reception Test Summary ===")
        print(f"Test 1 (Basic valid): {'PASS' if test1_success else 'FAIL'}")
        print(f"Test 2 (Safe + threats): {'PASS' if test2_success else 'FAIL'}")
        print(f"Test 3 (Invalid): {'PASS' if test3_success else 'FAIL'}")
        print(f"Test 4 (RTI style): {'PASS' if test4_success else 'FAIL'}")

        if not any([test1_success, test2_success, test4_success]):
            print("\nCRITICAL: No valid messages received by SubMaster!")
            print("This indicates a fundamental messaging issue.")
        elif test2_success:
            print("\nRECEPTION: Messages are reaching SubMaster correctly")
            print("ISSUE: Must be in UI rendering logic or display code")

        print("\nNext step: Check UI process and HUD rendering logic")

def main():
    debugger = UIReceptionDebugger()
    debugger.run_debug_sequence()

if __name__ == "__main__":
    main()
