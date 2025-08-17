#!/usr/bin/env python3
"""
Debug why threat messages aren't making it to rti_threats vector
"""

import time
import cereal.messaging as messaging


def inject_and_monitor():
    """Inject messages while monitoring reception."""
    print("RTI Message Reception Debug")
    print("=" * 40)

    # Create publisher and subscriber
    pm = messaging.PubMaster(['rtiStateSP'])
    sm = messaging.SubMaster(['rtiStateSP'])

    print("Starting injection and monitoring...")
    print("Publishing messages every 2 seconds...")
    print("Monitor will check for message reception...")
    print()

    for i in range(10):
        # Create and send message
        msg = messaging.new_message('rtiStateSP', valid=True)
        rti_state = msg.rtiStateSP

        rti_state.timeStamp = int(time.time() * 1e9)
        rti_state.threatAhead = True
        rti_state.threatDistanceM = 300.0
        rti_state.recommendedSpeed = 11.2
        rti_state.source = f'debug_test_{i}'
        rti_state.apiStatus = 'connected'

        # Add threat
        rti_state.init('threats', 1)
        threat_msg = rti_state.threats[0]
        threat_msg.id = f'debug_police_{i:03d}'
        threat_msg.type = 'police'
        threat_msg.latitude = 37.4231
        threat_msg.longitude = -122.0841
        threat_msg.distance = 300.0
        threat_msg.direction = 'ahead'
        threat_msg.confidence = 0.85
        threat_msg.speedLimitMs = 11.2

        # Send message
        pm.send('rtiStateSP', msg)
        print(f"SENT {i+1}: {threat_msg.id} at {rti_state.threatDistanceM}m")

        # Check if we can receive it back
        time.sleep(0.1)  # Small delay
        sm.update(0)

        if sm.updated['rtiStateSP']:
            received = sm['rtiStateSP']
            print(f"  RECEIVED: source={received.source}, threats={len(received.threats)}")
            if len(received.threats) > 0:
                print(f"    First threat: {received.threats[0].id}")
        else:
            print(f"  NO RECEPTION: valid={sm.valid['rtiStateSP']}")

        print()
        time.sleep(2)

    print("Test complete.")
    print()
    print("KEY QUESTIONS:")
    print("1. Are we seeing 'RECEIVED' messages?")
    print("2. If not, the UI SubMaster might not be receiving our injected messages")
    print("3. This could explain why rti_threats vector stays empty")


if __name__ == "__main__":
    inject_and_monitor()
