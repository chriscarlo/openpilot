#!/usr/bin/env python3
"""
Test with proper message format matching production exactly
"""

import time
import cereal.messaging as messaging


def test_production_format():
    """Test using exact production message format."""
    print("Testing Production Message Format")
    print("=" * 40)

    pm = messaging.PubMaster(['rtiStateSP'])
    sm = messaging.SubMaster(['rtiStateSP'])

    # Send at proper 1Hz frequency as defined in services.py
    for i in range(3):
        # Create message exactly like production rtid.py
        msg = messaging.new_message('rtiStateSP', valid=True)

        # Copy exact format from rtid.py
        msg.rtiStateSP.timeStamp = int(time.time() * 1e9)
        msg.rtiStateSP.threatAhead = True
        msg.rtiStateSP.threatDistanceM = 300.0
        msg.rtiStateSP.recommendedSpeed = 11.2
        msg.rtiStateSP.source = 'test'
        msg.rtiStateSP.apiStatus = 'connected'

        # Initialize threats exactly like production code
        threats_to_send = 1
        msg.rtiStateSP.init('threats', threats_to_send)

        threat_msg = msg.rtiStateSP.threats[0]
        threat_msg.id = f'test_police_{i:03d}'
        threat_msg.type = 'police'
        threat_msg.latitude = 37.4231
        threat_msg.longitude = -122.0841
        threat_msg.distance = 300.0
        threat_msg.direction = 'ahead'
        threat_msg.confidence = 0.85
        threat_msg.speedLimitMs = 11.2

        # Send message
        pm.send('rtiStateSP', msg)
        print(f"SENT {i+1}: {threat_msg.id} (1Hz timing)")

        # Check after 0.5 seconds
        time.sleep(0.5)
        sm.update(0)

        valid = sm.valid['rtiStateSP']
        updated = sm.updated['rtiStateSP']

        print(f"  valid={valid}, updated={updated}")

        if valid and updated:
            received = sm['rtiStateSP']
            print(f"  SUCCESS: {len(received.threats)} threats received")
            print("  HUD should now display threat!")
            break
        elif updated and not valid:
            print("  STILL INVALID - investigating...")
        else:
            print("  NO RECEPTION")

        # Wait remainder of 1 second (1Hz)
        time.sleep(0.5)
        print()

    print("If still invalid, the issue is not frequency or format.")


if __name__ == "__main__":
    test_production_format()
