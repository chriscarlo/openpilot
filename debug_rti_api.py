#!/usr/bin/env python3

"""
Debug RTI API Status
Check GPS and API connectivity issues
"""

import asyncio
import json
from cereal import messaging
from sunnypilot.rtid.waze_api_client import WazeAPIClient

async def debug_rti_api():
    """Debug RTI API connectivity and GPS"""
    print("=== RTI API Debug ===")

    # Check GPS location
    print("1. Checking GPS location...")
    sm = messaging.SubMaster(['gpsLocationExternal', 'gpsLocation'])
    sm.update(2000)  # 2 second timeout

    lat, lon = None, None

    if sm.updated['gpsLocationExternal']:
        gps_ext = sm['gpsLocationExternal']
        if gps_ext.accuracy < 10.0:
            lat, lon = gps_ext.latitude, gps_ext.longitude
            print(f"  GPS External: {lat:.6f}, {lon:.6f} (accuracy: {gps_ext.accuracy:.1f}m)")

    if not lat and sm.updated['gpsLocation']:
        gps_loc = sm['gpsLocation']
        if gps_loc.hasFix:
            lat, lon = gps_loc.latitude, gps_loc.longitude
            print(f"  GPS Internal: {lat:.6f}, {lon:.6f}")

    if not lat:
        print("  ❌ No GPS location available")
        # Use test coordinates
        lat, lon = 37.4221, -122.0841
        print(f"  Using test coordinates: {lat}, {lon}")
    else:
        print("  ✅ GPS location available")

    # Check API key
    print("\n2. Checking API key...")
    try:
        with open('/persist/waze/waze_rapidapi.json') as f:
            key_data = json.load(f)
            api_key = key_data.get('api_key')
            if api_key:
                print(f"  ✅ API key found: {api_key[:8]}...")
            else:
                print("  ❌ No API key in file")
                return
    except Exception as e:
        print(f"  ❌ Failed to load API key: {e}")
        return

    # Test API connectivity
    print("\n3. Testing API connectivity...")
    try:
        client = WazeAPIClient(api_key)

        print("  Making API call...")
        alerts = await client.get_traffic_alerts(lat, lon)

        print(f"  ✅ API call successful: {len(alerts)} alerts returned")
        if alerts:
            for i, alert in enumerate(alerts[:3]):  # Show first 3
                print(f"    Alert {i+1}: {alert.type} at {alert.distance:.0f}m")
        else:
            print("    No alerts in area")

        health = client.get_health_status()
        print(f"  API health: {health}")

        await client.close()

    except Exception as e:
        print(f"  ❌ API call failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_rti_api())
