#!/usr/bin/env python3

"""
Test Valid Flag Override

This script tests the hypothesis that explicitly setting valid=True
in messaging.new_message() will make the message valid for HUD display.

Usage: python3 test_valid_override.py
"""

import time
from cereal import messaging

def test_valid_flag_override():
    """Test creating a message with valid=True override."""
    print("=== Testing Valid Flag Override ===")

    # Create PubMaster
    pm = messaging.PubMaster(['rtiStateSP'])

    print("\nTesting message creation with valid=True override...")

    # Create message with explicit valid=True
    msg = messaging.new_message('rtiStateSP', valid=True)
    rti_state = msg.rtiStateSP

    # Set basic state
    rti_state.timeStamp = int(time.time() * 1e9)
    rti_state.threatAhead = True
    rti_state.threatDistanceM = 200.0
    rti_state.recommendedSpeed = 12.0  # ~27 mph
    rti_state.source = 'valid_test'
    rti_state.apiStatus = 'connected'

    # Create a single threat
    rti_state.init('threats', 1)
    threat = rti_state.threats[0]
    threat.id = 'valid_test_police'
    threat.type = 'police'
    threat.latitude = 37.4231
    threat.longitude = -122.0841
    threat.distance = 200.0
    threat.direction = 'ahead'
    threat.confidence = 0.95
    threat.speedLimitMs = 15.0

    # Print message details
    print(f"Message valid flag: {msg.valid}")
    print(f"Threat type: {threat.type}")
    print(f"Distance: {threat.distance}m")

    # Send message
    print("\nSending message with valid=True...")
    pm.send('rtiStateSP', msg)
    print("Message sent!")

    print("\nIf HUD shows threat, the hypothesis is correct!")
    print("Check the bottom-left RTI widget for threat display.")

    # Wait a bit for message to be processed
    time.sleep(2)

if __name__ == "__main__":
    test_valid_flag_override()
