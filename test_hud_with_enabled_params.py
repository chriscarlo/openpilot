#!/usr/bin/env python3

"""
Test HUD Widget with Enabled Parameters
Now that RTI parameters are enabled, test if HUD widget appears
"""

import time
from cereal import messaging

def test_hud_with_enabled_params():
    """Test if HUD shows with parameters enabled"""
    print("=== Testing HUD with RTI Parameters Enabled ===")
    print("RTIEnabled=True, RTIHUDEnabled=True, RTIAudioAlerts=True")
    print()
    print("Sending test threat messages...")
    print("Check screen for RTI widget in bottom-left corner")
    print("Listen for audio alerts")
    print()

    pm = messaging.PubMaster(['rtiStateSP'])

    for i in range(100):  # Send 100 messages (5 seconds at 20Hz)
        # Create threat message
        msg = messaging.new_message('rtiStateSP', valid=True)
        msg.rtiStateSP.timeStamp = int(time.time() * 1e9)
        msg.rtiStateSP.threatAhead = True
        msg.rtiStateSP.threatDistanceM = 300.0
        msg.rtiStateSP.recommendedSpeed = 0.0
        msg.rtiStateSP.source = f'hud_test_{i}'
        msg.rtiStateSP.apiStatus = 'connected'

        # Create police threat
        msg.rtiStateSP.init('threats', 1)
        threat_msg = msg.rtiStateSP.threats[0]
        threat_msg.id = f'police_{i}'
        threat_msg.type = 'police'
        threat_msg.latitude = 37.4231
        threat_msg.longitude = -122.0841
        threat_msg.distance = 300.0
        threat_msg.direction = 'ahead'
        threat_msg.confidence = 0.95
        threat_msg.speedLimitMs = 15.0

        pm.send('rtiStateSP', msg)

        if i % 20 == 0:  # Print every second
            print(f"Sent {i+1} messages - police threat at 300m")

        time.sleep(0.05)  # 20Hz

    print()
    print("Test complete. Did you see:")
    print("1. RTI widget in bottom-left corner?")
    print("2. Police icon and text?")
    print("3. Distance (300m)?")
    print("4. Audio alert sounds?")

if __name__ == "__main__":
    test_hud_with_enabled_params()
