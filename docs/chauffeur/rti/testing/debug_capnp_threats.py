#!/usr/bin/env python3

"""
Debug Cap'n Proto Threats Array

Specifically tests if the threats array is properly created and accessible.
This will help determine why rti_has_threat stays false.

Usage: python3 debug_capnp_threats.py
"""

import time
from cereal import messaging

def test_capnp_threats():
    """Test Cap'n Proto threats array creation and access."""
    print("=== Cap'n Proto Threats Array Debug ===")

    pm = messaging.PubMaster(['rtiStateSP'])
    sm = messaging.SubMaster(['rtiStateSP'])

    print("Creating message with threats array...")

    # Create message with valid=True
    msg = messaging.new_message('rtiStateSP', valid=True)
    rti_state = msg.rtiStateSP

    # Basic message setup
    rti_state.timeStamp = int(time.time() * 1e9)
    rti_state.threatAhead = False  # Safe mode
    rti_state.threatDistanceM = 0.0  # Safe mode
    rti_state.recommendedSpeed = 0.0
    rti_state.source = 'capnp_debug'
    rti_state.apiStatus = 'connected'

    # Create threats array - THIS IS THE CRITICAL PART
    print("Initializing threats array with 1 element...")
    rti_state.init('threats', 1)

    # Set threat data
    print("Setting threat[0] data...")
    threat = rti_state.threats[0]
    threat.id = 'debug_police'
    threat.type = 'police'
    threat.latitude = 37.4231
    threat.longitude = -122.0841
    threat.distance = 250.0
    threat.direction = 'ahead'
    threat.confidence = 0.9
    threat.speedLimitMs = 15.0

    # Verify threats array before sending
    print(f"Before sending - threats array length: {len(rti_state.threats)}")
    if len(rti_state.threats) > 0:
        t = rti_state.threats[0]
        print(f"  Threat[0]: type='{t.type}', distance={t.distance}, id='{t.id}'")
    else:
        print("  ERROR: Threats array is empty!")

    # Send message
    print("\nSending message...")
    pm.send('rtiStateSP', msg)

    # Wait and receive
    print("Waiting for message reception...")
    for i in range(20):  # Wait up to 2 seconds
        sm.update(0)
        if sm.updated['rtiStateSP']:
            print(f"\nMessage received! Valid: {sm.valid['rtiStateSP']}")

            try:
                received_msg = sm['rtiStateSP']
                received_rti = received_msg.rtiStateSP

                print(f"Received message source: '{received_rti.source}'")
                print(f"Threats array length: {len(received_rti.threats)}")

                if len(received_rti.threats) > 0:
                    rt = received_rti.threats[0]
                    print(f"  Threat[0]: type='{rt.type}', distance={rt.distance}, id='{rt.id}'")
                    print("SUCCESS: Threats array properly created and accessible!")
                else:
                    print("FAILURE: Threats array is empty in received message!")

            except Exception as e:
                print(f"ERROR accessing received message: {e}")

            break

        time.sleep(0.1)
    else:
        print("TIMEOUT: No message received")

    print("\nThis should match what the HUD C++ code sees.")

if __name__ == "__main__":
    test_capnp_threats()
