#!/usr/bin/env python3

"""
Debug RTI Daemon Live
Monitor RTI daemon behavior and API calls in real-time
"""

import asyncio
import time
import json
from cereal import messaging
from sunnypilot.rtid.waze_api_client import WazeAPIClient

async def debug_rtid_live():
    """Debug RTI daemon live behavior"""
    print("=== RTI Daemon Live Debug ===")

    # Monitor RTI messages
    sm = messaging.SubMaster(['rtiStateSP', 'gpsLocationExternal', 'gpsLocation'])

    # Test API directly with same location as daemon
    try:
        with open('/persist/waze/waze_rapidapi.json') as f:
            key_data = json.load(f)
            api_key = key_data.get('api_key')
    except Exception as e:
        print(f"Failed to load API key: {e}")
        return

    print("Monitoring for 30 seconds...")
    start_time = time.time()
    last_gps = None
    api_test_done = False

    while time.time() - start_time < 30:
        sm.update(1000)

        # Check GPS
        if sm.updated['gpsLocationExternal']:
            gps_ext = sm['gpsLocationExternal']
            if gps_ext.accuracy < 10.0:
                current_gps = (gps_ext.latitude, gps_ext.longitude)
                if current_gps != last_gps:
                    print(f"GPS: {current_gps[0]:.6f}, {current_gps[1]:.6f} (accuracy: {gps_ext.accuracy:.1f}m)")
                    last_gps = current_gps

                    # Test API with same location once
                    if not api_test_done:
                        print("Testing API with daemon's GPS location...")
                        try:
                            client = WazeAPIClient(api_key)
                            alerts = await client.get_traffic_alerts(current_gps[0], current_gps[1], 16.0)
                            print(f"Direct API test: {len(alerts)} alerts found with 16km radius")
                            if alerts:
                                for i, alert in enumerate(alerts[:3]):
                                    print(f"  Alert {i+1}: {alert.type} ({alert.confidence:.2f})")
                            await client.close()
                        except Exception as e:
                            print(f"Direct API test failed: {e}")
                        api_test_done = True

        # Check RTI messages
        if sm.updated['rtiStateSP']:
            rti = sm['rtiStateSP']
            timestamp = time.strftime('%H:%M:%S', time.localtime())
            print(f"[{timestamp}] RTI: status={rti.apiStatus}, threats={len(rti.threats)}, threatAhead={rti.threatAhead}")

        await asyncio.sleep(0.5)

    print("Debug complete.")

if __name__ == "__main__":
    asyncio.run(debug_rtid_live())
