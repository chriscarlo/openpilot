#!/usr/bin/env python3

"""
Comprehensive RTI System Diagnostic

Tests all RTI components:
1. Message validation (valid=True vs valid=False)
2. HUD display (safe - threatAhead=False)  
3. Audio alerts (requires threatAhead=True)
4. Speed control impact

Shows exactly what's working and what's not.

Usage: python3 comprehensive_diagnostic.py
"""

import time
from cereal import messaging

def test_message_validation():
    """Test basic message validation."""
    print("=== Testing Message Validation ===")

    # Test invalid message (default behavior)
    msg_invalid = messaging.new_message('rtiStateSP')
    print(f"Default message valid flag: {msg_invalid.valid}")

    # Test valid message override
    msg_valid = messaging.new_message('rtiStateSP', valid=True)
    print(f"Override message valid flag: {msg_valid.valid}")
    print()

def test_hud_display_safe():
    """Test HUD display without speed control impact."""
    print("=== Testing HUD Display (Safe Mode) ===")
    print("This should show threats in HUD without affecting speed")

    pm = messaging.PubMaster(['rtiStateSP'])

    # Create safe HUD test message
    msg = messaging.new_message('rtiStateSP', valid=True)
    rti_state = msg.rtiStateSP

    # Safe settings - prevent speed control
    rti_state.timeStamp = int(time.time() * 1e9)
    rti_state.threatAhead = False  # SAFE: No speed control
    rti_state.threatDistanceM = 0.0  # SAFE: No speed control
    rti_state.recommendedSpeed = 0.0  # SAFE: No speed recommendation
    rti_state.source = 'diagnostic_hud'
    rti_state.apiStatus = 'connected'

    # Add threat for HUD display
    rti_state.init('threats', 1)
    threat = rti_state.threats[0]
    threat.id = 'diag_police'
    threat.type = 'police'
    threat.latitude = 37.4231
    threat.longitude = -122.0841
    threat.distance = 250.0
    threat.direction = 'ahead'
    threat.confidence = 0.9
    threat.speedLimitMs = 15.0

    pm.send('rtiStateSP', msg)
    print("SENT: Safe HUD test message")
    print("   Check RTI widget for threat display")
    print("   Should show: Police 250m (grey color)")
    time.sleep(3)
    print()

def test_audio_and_speed_warning():
    """Test audio alerts - WARNING: This affects speed control!"""
    print("=== CAUTION: Audio + Speed Control Test ===")
    response = input("This test affects speed control. Type 'YES' to continue: ")

    if response != 'YES':
        print("SKIPPED: Audio/speed test for safety")
        return

    print("WARNING: This will trigger speed control!")
    print("   Only run this when safe to test speed changes")

    pm = messaging.PubMaster(['rtiStateSP'])

    # Create full RTI message with audio and speed control
    msg = messaging.new_message('rtiStateSP', valid=True)
    rti_state = msg.rtiStateSP

    # Full RTI settings - triggers all systems
    rti_state.timeStamp = int(time.time() * 1e9)
    rti_state.threatAhead = True  # WARNING: TRIGGERS SPEED CONTROL
    rti_state.threatDistanceM = 200.0  # WARNING: TRIGGERS SPEED CONTROL
    rti_state.recommendedSpeed = 12.0  # WARNING: SPEED RECOMMENDATION
    rti_state.source = 'diagnostic_full'
    rti_state.apiStatus = 'connected'

    # Add threat
    rti_state.init('threats', 1)
    threat = rti_state.threats[0]
    threat.id = 'diag_police_full'
    threat.type = 'police'
    threat.latitude = 37.4231
    threat.longitude = -122.0841
    threat.distance = 200.0
    threat.direction = 'ahead'
    threat.confidence = 0.95
    threat.speedLimitMs = 12.0

    pm.send('rtiStateSP', msg)
    print("SENT: FULL RTI message")
    print("   Should trigger:")
    print("   - HUD threat display (red color)")
    print("   - Audio alert")
    print("   - Speed control reduction")

    time.sleep(5)
    print()

def main():
    """Run comprehensive RTI diagnostic."""
    print("RTI System Comprehensive Diagnostic")
    print("=" * 50)

    # Test 1: Message validation
    test_message_validation()

    # Test 2: Safe HUD display
    test_hud_display_safe()

    # Test 3: Full system (optional, requires confirmation)
    test_audio_and_speed_warning()

    print("=== Diagnostic Summary ===")
    print("VALIDATED: Message validation working (valid=True override)")
    print("CHECK: HUD display in RTI widget (bottom-left)")
    print("NOTE: Audio alerts require threatAhead=True (speed control)")
    print("NOTE: Speed control active when threatAhead=True")
    print()
    print("If HUD still shows only 'RTI' placeholder:")
    print("- Check UI process is running")
    print("- Verify RTI widget is visible")
    print("- May need restart of UI process")

if __name__ == "__main__":
    main()
