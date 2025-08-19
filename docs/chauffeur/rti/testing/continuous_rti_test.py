#!/usr/bin/env python3

"""
Continuous RTI HUD Test

The KEY DIFFERENCE: RTI widget requires BOTH valid() AND updated() flags.
The updated() flag is only set when SubMaster receives a NEW message.
VTSC lateral accel only needs valid() so it works with old messages.

This sends messages continuously to ensure updated() flag is set when UI reads it.

Usage: python3 continuous_rti_test.py
"""

import time
from cereal import messaging

def send_continuous_rti_messages():
    """Send RTI messages continuously to trigger both valid() and updated() flags."""
    print("=== Continuous RTI Test (Fixed updated() Flag Issue) ===")
    print("The problem: RTI widget needs both valid() AND updated() flags")
    print("VTSC lateral accel only needs valid() flag - that's why it works!")
    print("Sending messages at 50Hz to prevent timing drift with UI refresh rate...")

    pm = messaging.PubMaster(['rtiStateSP'])

    message_count = 0

    try:
        while True:
            message_count += 1

            # Create message with valid=True
            msg = messaging.new_message('rtiStateSP', valid=True)

            # Settings for both visual and audio alerts
            msg.rtiStateSP.timeStamp = int(time.time() * 1e9)
            # Edge-triggered audio: start False, then True to trigger alert
            msg.rtiStateSP.threatAhead = (message_count > 50)  # False for first 50, then True
            msg.rtiStateSP.threatDistanceM = 250.0  # Real threat distance
            msg.rtiStateSP.recommendedSpeed = 0.0  # SAFE - no speed control
            msg.rtiStateSP.source = f'continuous_test_{message_count}'
            msg.rtiStateSP.apiStatus = 'connected'

            # Create threat
            msg.rtiStateSP.init('threats', 1)
            threat_msg = msg.rtiStateSP.threats[0]
            threat_msg.id = f'test_police_{message_count}'
            threat_msg.type = 'police'
            threat_msg.latitude = 37.4231
            threat_msg.longitude = -122.0841
            threat_msg.distance = 250.0
            threat_msg.direction = 'ahead'
            threat_msg.confidence = 0.95
            threat_msg.speedLimitMs = 15.0

            # Send message
            pm.send('rtiStateSP', msg)

            if message_count == 1 or message_count % 250 == 0:  # Print first and every 250th message (every 5 seconds)
                print(f"Message #{message_count}: Police threat sent (visual + audio)")
                print("Check RTI widget - should show police icon/text + hear audio alert!")

            # Send at 50Hz to prevent timing drift gaps (faster than UI 20Hz)
            time.sleep(0.02)  # 1/50 = 0.02 seconds

    except KeyboardInterrupt:
        print(f"\nStopped after {message_count} messages")
        print("If RTI widget showed threat, the issue is fixed!")
        print("If still blank, there may be other issues.")

if __name__ == "__main__":
    send_continuous_rti_messages()
