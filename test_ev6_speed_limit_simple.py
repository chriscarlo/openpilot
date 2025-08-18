#!/usr/bin/env python3
"""
Simple test to check if dashboard speed limit messages exist on EV6 CAN bus
Run this on an actual EV6 to verify message presence
"""
import time
import cereal.messaging as messaging

def check_can_messages():
    """Monitor CAN bus for dashboard speed limit messages"""
    print("=" * 60)
    print("EV6 CAN Bus Monitor - Dashboard Speed Limit Messages")
    print("=" * 60)
    
    # Create subscriber for CAN messages
    sm = messaging.SubMaster(['can'])
    
    print("\n📡 Monitoring CAN bus for speed limit messages...")
    print("Looking for:")
    print("  - 0x1FA (506): FR_CMR_02_100ms - ISLW speed limit")
    print("  - 0x162 (354): CCNC_0x162 - Alternative speed limit")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    found_1fa = False
    found_162 = False
    msg_count = 0
    
    try:
        while True:
            sm.update(100)
            
            if sm.updated['can']:
                for msg in sm['can']:
                    msg_count += 1
                    
                    # Check camera bus (usually bus 2)
                    if msg.src == 2:  # Camera bus
                        if msg.address == 0x1FA and not found_1fa:
                            found_1fa = True
                            print(f"✅ Found 0x1FA on camera bus! Message data: {msg.dat.hex()}")
                            
                        if msg.address == 0x162 and not found_162:
                            found_162 = True
                            print(f"✅ Found 0x162 on camera bus! Message data: {msg.dat.hex()}")
                    
                    # Print progress every 1000 messages
                    if msg_count % 1000 == 0:
                        print(f"  Scanned {msg_count} messages... 0x1FA: {'✓' if found_1fa else '✗'}, 0x162: {'✓' if found_162 else '✗'}")
                    
                    if found_1fa or found_162:
                        print("\n" + "=" * 60)
                        print("🎯 DASHBOARD SPEED LIMIT MESSAGES DETECTED!")
                        print("The EV6 DOES transmit speed limit data on the CAN bus.")
                        print("These messages need to be added to the fingerprint.")
                        print("=" * 60)
                        return
                        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total messages scanned: {msg_count}")
    print(f"  0x1FA found: {'Yes ✅' if found_1fa else 'No ❌'}")
    print(f"  0x162 found: {'Yes ✅' if found_162 else 'No ❌'}")
    
    if not found_1fa and not found_162:
        print("\n⚠️  No dashboard speed limit messages detected.")
        print("  Either the messages don't exist or they use different IDs.")
    print("=" * 60)

if __name__ == "__main__":
    check_can_messages()
