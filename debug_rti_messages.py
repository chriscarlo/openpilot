#!/usr/bin/env python3
"""
Debug RTI message reception to see if UI is getting the messages
"""

import time
import cereal.messaging as messaging


def monitor_rti_messages():
    """Monitor rtiStateSP messages being received by the system."""
    print("Monitoring rtiStateSP messages...")
    print("Press Ctrl+C to stop")
    print()

    sm = messaging.SubMaster(['rtiStateSP'])

    last_msg_time = 0
    msg_count = 0

    while True:
        sm.update(0)

        if sm.updated['rtiStateSP']:
            msg_count += 1
            current_time = time.time()

            rti_state = sm['rtiStateSP']

            print(f"Message {msg_count}:")
            print(f"  Time: {current_time:.3f}")
            print(f"  Threat ahead: {rti_state.threatAhead}")
            print(f"  Distance: {rti_state.threatDistanceM}")
            print(f"  Recommended speed: {rti_state.recommendedSpeed}")
            print(f"  Source: {rti_state.source}")
            print(f"  API status: {rti_state.apiStatus}")
            print(f"  Threats count: {len(rti_state.threats)}")
            if len(rti_state.threats) > 0:
                threat = rti_state.threats[0]
                print(f"    First threat: {threat.type} at {threat.distance}m")
            print()

            last_msg_time = current_time

        time.sleep(0.1)


if __name__ == "__main__":
    try:
        monitor_rti_messages()
    except KeyboardInterrupt:
        print("\nStopped monitoring")
