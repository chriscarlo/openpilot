#!/usr/bin/env python3

"""
RTI Real Message Monitor

Monitors actual rtid messages to understand what makes them valid=True
vs our test messages that are valid=False.

Captures:
- Message content and format
- Validation status 
- Timing and metadata
- Raw message data for comparison

Usage: python3 monitor_real_rtid_messages.py
"""

import time
import json
from typing import Any
from cereal import messaging
from system.manager.process_config import managed_processes

def format_message_data(msg) -> dict[str, Any]:
    """Extract and format all message data for analysis"""
    if not hasattr(msg, 'rtiStateSP'):
        return {"error": "No rtiStateSP field"}

    rti_msg = msg.rtiStateSP

    # Convert to dict for JSON serialization
    data = {
        'logMonoTime': msg.logMonoTime,
        'source': getattr(msg, 'source', 'unknown'),
        'rtiStateSP': {
            'threats': [],
            'activeThreatTypes': list(rti_msg.activeThreatTypes) if hasattr(rti_msg, 'activeThreatTypes') else [],
            'mostSevereWarningLevel': rti_msg.mostSevereWarningLevel if hasattr(rti_msg, 'mostSevereWarningLevel') else 0,
            'timeSinceLastThreatMs': rti_msg.timeSinceLastThreatMs if hasattr(rti_msg, 'timeSinceLastThreatMs') else 0,
        }
    }

    # Extract threat details
    if hasattr(rti_msg, 'threats'):
        for i, threat in enumerate(rti_msg.threats):
            threat_data = {
                'id': threat.id if hasattr(threat, 'id') else f'threat_{i}',
                'type': threat.type if hasattr(threat, 'type') else 'unknown',
                'distanceMeters': threat.distanceMeters if hasattr(threat, 'distanceMeters') else 0,
                'bearingDeg': threat.bearingDeg if hasattr(threat, 'bearingDeg') else 0,
                'speedKmh': threat.speedKmh if hasattr(threat, 'speedKmh') else 0,
                'warningLevel': threat.warningLevel if hasattr(threat, 'warningLevel') else 0,
                'lastUpdatedMs': threat.lastUpdatedMs if hasattr(threat, 'lastUpdatedMs') else 0,
            }
            data['rtiStateSP']['threats'].append(threat_data)

    return data

def check_rtid_running():
    """Check if rtid process is running"""
    if 'rtid' in managed_processes:
        status = managed_processes['rtid']
        print(f"rtid process status: {status}")
    else:
        print("rtid not found in managed processes")

def main():
    print("=== RTI Real Message Monitor ===")
    print("Monitoring actual rtid messages to understand validation logic...")
    print("Press Ctrl+C to stop\n")

    # Check rtid status
    check_rtid_running()

    # Set up subscriber
    sm = messaging.SubMaster(['rtiStateSP'])

    message_count = 0
    valid_count = 0
    invalid_count = 0
    last_valid_msg = None
    last_invalid_msg = None

    print("Waiting for rtid messages...")
    print("=" * 80)

    try:
        while True:
            sm.update(0)  # Non-blocking update

            if sm.updated['rtiStateSP']:
                message_count += 1
                is_valid = sm.valid['rtiStateSP']

                # Track counts
                if is_valid:
                    valid_count += 1
                    print(f"\n[VALID] MESSAGE #{valid_count} (total: {message_count})")
                    last_valid_msg = sm['rtiStateSP']
                else:
                    invalid_count += 1
                    print(f"\n[INVALID] MESSAGE #{invalid_count} (total: {message_count})")
                    last_invalid_msg = sm['rtiStateSP']

                # Display message details
                print(f"Timestamp: {time.time()}")
                print(f"Valid: {is_valid}")
                print(f"Updated: {sm.updated['rtiStateSP']}")

                # Extract and display message content
                try:
                    msg_data = format_message_data(sm.msg('rtiStateSP'))
                    print(f"Message data: {json.dumps(msg_data, indent=2)}")

                    # Save valid message for comparison
                    if is_valid:
                        with open('/data/openpilot/docs/chauffeur/rti/testing/last_valid_rtid_message.json', 'w') as f:
                            json.dump(msg_data, f, indent=2)
                        print("SAVED: Valid message to last_valid_rtid_message.json")

                except Exception as e:
                    print(f"Error processing message: {e}")

                print("-" * 40)

                # Show summary every 10 messages
                if message_count % 10 == 0:
                    print(f"\nSUMMARY: {message_count} total | {valid_count} valid | {invalid_count} invalid")
                    if valid_count > 0:
                        validity_rate = (valid_count / message_count) * 100
                        print(f"Validity rate: {validity_rate:.1f}%")

            time.sleep(0.1)  # 10Hz monitoring

    except KeyboardInterrupt:
        print("\n\n=== FINAL SUMMARY ===")
        print(f"Total messages: {message_count}")
        print(f"Valid messages: {valid_count}")
        print(f"Invalid messages: {invalid_count}")

        if valid_count > 0:
            validity_rate = (valid_count / message_count) * 100
            print(f"Validity rate: {validity_rate:.1f}%")
            print("\nAt least one valid message captured for analysis!")
        else:
            print("\nNo valid messages captured. RTI daemon may not be active.")

        if last_valid_msg and last_invalid_msg:
            print("\nReady for comparison analysis between valid and invalid messages.")

if __name__ == "__main__":
    main()
