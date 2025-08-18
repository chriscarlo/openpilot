#!/usr/bin/env python3

"""
Quick RTI Content Checker
Check what actual threat data is in RTI messages
"""

import time
from cereal import messaging

def check_rti_content():
    """Check RTI message content for threats"""
    print("=== RTI Content Checker ===")
    print("Checking for threat data in RTI messages...")

    sm = messaging.SubMaster(['rtiStateSP'])

    for i in range(10):  # Check 10 messages
        sm.update(1000)  # 1 second timeout

        if sm.updated['rtiStateSP']:
            rti = sm['rtiStateSP']
            print(f"\nMessage #{i+1}:")
            print(f"  threatAhead: {rti.threatAhead}")
            print(f"  threatDistanceM: {rti.threatDistanceM}")
            print(f"  recommendedSpeed: {rti.recommendedSpeed}")
            print(f"  source: {rti.source}")
            print(f"  apiStatus: {rti.apiStatus}")
            print(f"  threats count: {len(rti.threats)}")

            if len(rti.threats) > 0:
                print("  Threat details:")
                for j, threat in enumerate(rti.threats):
                    print(f"    Threat {j+1}: {threat.type} at {threat.distance}m")
            else:
                print("  No threats detected")

        time.sleep(0.5)

    print("\nDone checking RTI content.")

if __name__ == "__main__":
    check_rti_content()
