#!/usr/bin/env python3

"""
Safe HUD Test Script

Tests RTI HUD display without affecting speed control by:
- Setting threatAhead=False to prevent RTI controller activation
- Using valid=True for HUD display
- Only testing visual threat display functionality

This script is SAFE for testing while driving.

Usage: python3 test_hud_safe.py
"""

import time
from cereal import messaging

def test_hud_display_safe():
    """Test HUD threat display without affecting speed control."""
    print("=== SAFE RTI HUD Display Test ===")
    print("This test will NOT affect speed control")
    print("Testing visual threat display only")

    # Create PubMaster
    pm = messaging.PubMaster(['rtiStateSP'])

    for i in range(5):
        print(f"\nSending safe test message {i+1}/5...")

        # Create message with valid=True for HUD display
        msg = messaging.new_message('rtiStateSP', valid=True)
        rti_state = msg.rtiStateSP

        # SAFE SETTINGS: These prevent speed control activation
        rti_state.timeStamp = int(time.time() * 1e9)
        rti_state.threatAhead = False  # ← CRITICAL: Prevents speed control
        rti_state.threatDistanceM = 0.0  # ← CRITICAL: Prevents speed control
        rti_state.recommendedSpeed = 0.0  # ← SAFE: No speed recommendation
        rti_state.source = f'safe_hud_test_{i}'
        rti_state.apiStatus = 'connected'

        # Add threat for HUD display (should not affect driving)
        rti_state.init('threats', 1)
        threat = rti_state.threats[0]
        threat.id = f'safe_test_police_{i}'
        threat.type = 'police'
        threat.latitude = 37.4231
        threat.longitude = -122.0841
        threat.distance = 300.0  # Display distance
        threat.direction = 'ahead'
        threat.confidence = 0.9
        threat.speedLimitMs = 15.0

        # Send message
        pm.send('rtiStateSP', msg)
        print("Sent: Police threat at 300m (threatAhead=False for safety)")

        time.sleep(2)

    print("\n=== Test Complete ===")
    print("Check RTI widget for threat display")
    print("Speed control should NOT be affected")

if __name__ == "__main__":
    test_hud_display_safe()
