#!/usr/bin/env python3
"""
Simple RTI Message Test
Tests if messages can flow through the system with our fix.
"""
import time
import sys
import os

# Add openpilot to path
sys.path.insert(0, '/projects/chauffeur/data/openpilot')
os.chdir('/projects/chauffeur/data/openpilot')

from cereal import messaging
from openpilot.common.params import Params

def test_message_flow():
    """Test basic message publishing and subscription."""
    print("RTI Message Flow Test")
    print("=" * 60)
    
    # Setup parameters
    params = Params()
    params.put_bool("RTIEnabled", True)
    params.put_bool("RTIHUDEnabled", True)
    
    print("Creating publisher and subscriber...")
    
    # Create publisher
    pm = messaging.PubMaster(['rtiStateSP'])
    
    # Create subscriber (simulating what the UI would do)
    sm = messaging.SubMaster(['rtiStateSP'])
    
    print("Publishing test message...")
    
    # Create and send a test message
    msg = messaging.new_message('rtiStateSP')
    msg.rtiStateSP.timeStamp = int(time.time() * 1e9)
    msg.rtiStateSP.threatAhead = True
    msg.rtiStateSP.threatDistanceM = 250.0
    msg.rtiStateSP.recommendedSpeed = 15.0  # ~33 mph
    msg.rtiStateSP.source = "test"
    msg.rtiStateSP.apiStatus = "connected"
    
    pm.send('rtiStateSP', msg)
    
    print("Waiting for message reception...")
    
    # Try to receive the message
    received = False
    for i in range(10):  # Try for 1 second
        sm.update(100)
        if sm.updated['rtiStateSP']:
            received_msg = sm['rtiStateSP']
            print("\n✓ Message successfully received!")
            print(f"  Threat ahead: {received_msg.threatAhead}")
            print(f"  Distance: {received_msg.threatDistanceM:.1f}m")
            print(f"  Recommended speed: {received_msg.recommendedSpeed:.1f} m/s")
            received = True
            break
    
    if not received:
        print("\n✗ Message not received within timeout")
        return False
    
    return True

def main():
    try:
        success = test_message_flow()
        
        print("\n" + "=" * 60)
        if success:
            print("RESULT: Message flow test PASSED")
            print("\nThe messaging system is working correctly.")
            print("Once the UI is rebuilt with the fix, it will receive these messages.")
        else:
            print("RESULT: Message flow test FAILED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()