#!/usr/bin/env python3
"""
RTI Message Verification Script
Verifies that the RTI daemon is running and publishing messages correctly.
"""
import time
import sys
import os
import subprocess

# Add openpilot to path
sys.path.insert(0, '/projects/chauffeur/data/openpilot')
os.chdir('/projects/chauffeur/data/openpilot')

from cereal import messaging
from openpilot.common.params import Params

def check_rtid_process():
    """Check if rtid process is running."""
    result = subprocess.run(['pgrep', '-f', 'rtid.py'], capture_output=True, text=True)
    return result.returncode == 0

def monitor_rti_messages():
    """Monitor rtiStateSP messages."""
    print("Setting up RTI message monitoring...")
    
    # Check if RTI is enabled
    params = Params()
    rti_enabled = params.get_bool("RTIEnabled")
    rti_hud_enabled = params.get_bool("RTIHUDEnabled")
    
    print(f"RTIEnabled: {rti_enabled}")
    print(f"RTIHUDEnabled: {rti_hud_enabled}")
    
    if not rti_enabled:
        print("WARNING: RTI is not enabled. Setting RTIEnabled=True")
        params.put_bool("RTIEnabled", True)
        rti_enabled = True
        
    if not rti_hud_enabled:
        print("WARNING: RTI HUD is not enabled. Setting RTIHUDEnabled=True")
        params.put_bool("RTIHUDEnabled", True)
        rti_hud_enabled = True
    
    # Check if rtid is running
    if not check_rtid_process():
        print("WARNING: rtid process is not running!")
        print("You may need to restart openpilot for rtid to start.")
    else:
        print("rtid process is running.")
    
    # Subscribe to RTI messages
    print("\nSubscribing to rtiStateSP messages...")
    sm = messaging.SubMaster(['rtiStateSP'])
    
    print("Monitoring for RTI messages (press Ctrl+C to stop)...")
    print("-" * 60)
    
    message_count = 0
    last_update_time = 0
    
    try:
        while True:
            sm.update(1000)  # 1 second timeout
            
            if sm.updated['rtiStateSP']:
                message_count += 1
                msg = sm['rtiStateSP']
                
                print(f"\n[Message #{message_count}] RTI State Received:")
                print(f"  Timestamp: {msg.timeStamp}")
                print(f"  Threat Ahead: {msg.threatAhead}")
                print(f"  Threat Distance: {msg.threatDistanceM:.1f}m")
                print(f"  Recommended Speed: {msg.recommendedSpeed:.1f} m/s")
                print(f"  API Status: {msg.apiStatus}")
                print(f"  Source: {msg.source}")
                
                if len(msg.threats) > 0:
                    print(f"  Threats ({len(msg.threats)}):")
                    for i, threat in enumerate(msg.threats):
                        print(f"    [{i}] Type: {threat.type}, Distance: {threat.distance:.1f}m, Confidence: {threat.confidence:.2f}")
                else:
                    print("  No threats detected")
                    
                last_update_time = time.time()
            else:
                # Check for timeout
                if last_update_time > 0 and (time.time() - last_update_time) > 5:
                    print(f"\nNo RTI messages received for {time.time() - last_update_time:.1f} seconds")
                    last_update_time = 0
                    
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Received {message_count} RTI messages total.")

def check_ui_subscription():
    """Check if the UI is properly subscribed to RTI messages."""
    print("\nChecking UI subscription configuration...")
    
    ui_file = "/projects/chauffeur/data/openpilot/selfdrive/ui/sunnypilot/ui.cc"
    with open(ui_file, 'r') as f:
        content = f.read()
        
    if 'rtiStateSP' in content and 'SubMaster' in content:
        # Check if it's in the subscription list
        import re
        pattern = r'SubMaster.*?\{([^}]+)\}'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            subscription_list = match.group(1)
            if 'rtiStateSP' in subscription_list:
                print("✓ UI is subscribed to rtiStateSP")
                return True
            else:
                print("✗ UI is NOT subscribed to rtiStateSP")
                print("  The SubMaster initialization does not include 'rtiStateSP'")
                return False
    
    print("✗ Could not verify UI subscription")
    return False

if __name__ == "__main__":
    print("RTI Message Verification Script")
    print("=" * 60)
    
    # Check UI subscription
    ui_subscribed = check_ui_subscription()
    
    if not ui_subscribed:
        print("\n*** ROOT CAUSE IDENTIFIED ***")
        print("The UI is not subscribed to rtiStateSP messages!")
        print("This is why the RTI widget never displays threats.")
        print("\nTo fix this issue, 'rtiStateSP' needs to be added to the")
        print("SubMaster subscription list in selfdrive/ui/sunnypilot/ui.cc")
    
    print("\n" + "=" * 60)
    # Monitor messages anyway to verify backend is working
    monitor_rti_messages()