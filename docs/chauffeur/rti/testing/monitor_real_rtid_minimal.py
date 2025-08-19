#!/usr/bin/env python3

"""
Monitor Real RTID Messages (Minimal)

Simple monitor to see if real rtid is actually publishing messages successfully.

Usage: python3 monitor_real_rtid_minimal.py
"""

import time
from cereal import messaging

def monitor_rtid():
    """Monitor actual rtid message publication."""
    print("=== Monitoring Real RTID Messages ===")

    sm = messaging.SubMaster(['rtiStateSP'])

    print("Listening for rtiStateSP messages from real rtid...")
    print("Press Ctrl+C to stop")

    last_message_time = 0
    message_count = 0

    try:
        while True:
            sm.update(0)

            if sm.updated['rtiStateSP']:
                message_count += 1
                current_time = time.time()

                print(f"\n[{message_count}] Message received at {current_time:.3f}")
                print(f"  Valid: {sm.valid['rtiStateSP']}")
                print(f"  Updated: {sm.updated['rtiStateSP']}")

                # Try to access the message
                try:
                    msg = sm['rtiStateSP']
                    print("  SUCCESS: Message accessed successfully")
                    print(f"  Message type: {type(msg)}")

                    # Try to access rtiStateSP field
                    try:
                        rti_data = msg.rtiStateSP
                        print("  SUCCESS: rtiStateSP field accessed")
                        print(f"  Source: {rti_data.source}")
                        print(f"  ThreatAhead: {rti_data.threatAhead}")

                    except Exception as e:
                        print(f"  ERROR accessing rtiStateSP field: {e}")

                except Exception as e:
                    print(f"  ERROR accessing message: {e}")

                last_message_time = current_time

            elif time.time() - last_message_time > 5 and message_count > 0:
                print(f"No messages for 5 seconds (last: {message_count})")
                last_message_time = time.time()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\nMonitoring stopped. Total messages: {message_count}")

if __name__ == "__main__":
    monitor_rtid()
