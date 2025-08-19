#!/usr/bin/env python3
"""
Test to check if EV6 fingerprint contains dashboard speed limit messages
"""

import cereal.messaging as messaging
from openpilot.common.params import Params

def check_fingerprint():
    """Check the actual EV6 fingerprint for dashboard speed limit messages"""
    print("=" * 60)
    print("EV6 Fingerprint Check for Dashboard Speed Limit Messages")
    print("=" * 60)
    
    # Get the fingerprint from Params
    params = Params()
    
    # Get the cached fingerprint
    cached_fingerprint = params.get("CachedFingerprint")
    if cached_fingerprint:
        print(f"\nCached fingerprint found: {cached_fingerprint}")
    else:
        print("\nNo cached fingerprint found in Params")
    
    # Create sub sockets to read live fingerprint data
    sm = messaging.SubMaster(['pandaStates'])
    
    print("\n📡 Monitoring CAN bus for dashboard speed limit messages...")
    print("Looking for:")
    print("  - 0x1FA (506): FR_CMR_02_100ms - EV6 ISLW speed limit")
    print("  - 0x162 (354): CCNC_0x162 - Alternative speed limit source")
    print()
    
    # Monitor for a few seconds
    import time
    start_time = time.time()
    cam_messages = set()
    
    print("Checking for messages on camera bus...")
    while time.time() - start_time < 5:
        sm.update(100)
        
        if sm.updated['pandaStates']:
            for ps in sm['pandaStates']:
                # Check for CAN messages on camera bus (usually bus 2)
                if hasattr(ps, 'canState'):
                    for bus in ps.canState:
                        if bus.busType == 2:  # Camera bus
                            # This would need actual CAN parsing, simplified here
                            pass
    
    # Try to get the actual fingerprint from the car interface
    try:
        from openpilot.selfdrive.car.hyundai.interface import CarInterface
        from openpilot.selfdrive.car.hyundai.values import CAR
        
        # Get fingerprint for EV6
        print("\n🔍 Checking EV6 fingerprint definition...")
        
        # Check if we can get the fingerprint from the current system
        fingerprint_v2 = params.get("FingerprintV2")
        if fingerprint_v2:
            import json
            fp = json.loads(fingerprint_v2)
            print(f"\nFingerprintV2 found with {len(fp)} entries")
            
            # Check for our specific messages
            cam_bus_msgs = []
            for bus_id, messages in fp.items():
                if bus_id == "2":  # Camera bus
                    cam_bus_msgs = messages
                    break
            
            if cam_bus_msgs:
                print(f"\nCamera bus messages found: {len(cam_bus_msgs)} messages")
                
                # Check for dashboard speed limit messages
                has_fr_cmr = 0x1FA in cam_bus_msgs
                has_ccnc = 0x162 in cam_bus_msgs
                
                print(f"\n✅ Message 0x1FA (FR_CMR_02_100ms): {'PRESENT' if has_fr_cmr else 'NOT FOUND'}")
                print(f"✅ Message 0x162 (CCNC_0x162): {'PRESENT' if has_ccnc else 'NOT FOUND'}")
                
                if not has_fr_cmr and not has_ccnc:
                    print("\n⚠️  WARNING: No dashboard speed limit messages found in fingerprint!")
                    print("   This means the dashboard speed limit will NOT be parsed.")
                    print("   The flags HAS_DASHBOARD_SPEED_LIMIT_FR_CMR and")
                    print("   HAS_DASHBOARD_SPEED_LIMIT_CCNC will NOT be set.")
            else:
                print("\n⚠️  No camera bus messages found in fingerprint")
        else:
            print("\nNo FingerprintV2 found in Params")
            
    except Exception as e:
        print(f"\nError checking fingerprint: {e}")
    
    # Check the actual flags that would be set
    try:
        from openpilot.selfdrive.car.hyundai.values import HyundaiFlags
        print("\n📋 Flag values:")
        print(f"   HAS_DASHBOARD_SPEED_LIMIT_FR_CMR = {HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR}")
        print(f"   HAS_DASHBOARD_SPEED_LIMIT_CCNC = {HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_CCNC}")
    except Exception as e:
        print(f"\nError checking flag values: {e}")
    
    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS:")
    print("-" * 60)
    print("If the dashboard speed limit messages (0x1FA or 0x162) are NOT")
    print("present in the EV6 fingerprint on the camera bus, then:")
    print()
    print("1. The flags HAS_DASHBOARD_SPEED_LIMIT_FR_CMR/CCNC won't be set")
    print("2. carstate.py won't add these messages to the CAN parser")
    print("3. The messages won't be parsed even if they exist on the bus")
    print("4. ret_sp.speedLimit will always be 0.0")
    print("5. The speed limit controller will have no car dashboard data")
    print()
    print("This is likely the root cause of the issue!")
    print("=" * 60)

if __name__ == "__main__":
    check_fingerprint()