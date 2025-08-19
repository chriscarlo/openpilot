#!/usr/bin/env python3

"""
Test Threat Detector Logic
Test actual threat processing like RTI daemon does
"""

import asyncio
import json
import time
from cereal import messaging
from sunnypilot.rtid.waze_api_client import WazeAPIClient
from sunnypilot.rtid.threat_detector import ThreatDetector

async def test_threat_detector():
    """Test threat detector with real API data"""
    print("=== Threat Detector Test ===")

    # Get GPS location
    sm = messaging.SubMaster(['gpsLocationExternal', 'gpsLocation'])
    sm.update(2000)

    lat, lon = None, None
    if sm.updated['gpsLocationExternal']:
        gps_ext = sm['gpsLocationExternal']
        if gps_ext.accuracy < 10.0:
            lat, lon = gps_ext.latitude, gps_ext.longitude

    if not lat and sm.updated['gpsLocation']:
        gps_loc = sm['gpsLocation']
        if gps_loc.hasFix:
            lat, lon = gps_loc.latitude, gps_loc.longitude

    if not lat:
        print("No GPS, using test coords")
        lat, lon = 37.4221, -122.0841

    print(f"Location: {lat:.6f}, {lon:.6f}")

    # Load API key and test
    with open('/persist/waze/waze_rapidapi.json') as f:
        key_data = json.load(f)
        api_key = key_data.get('api_key')

    print("1. Testing API with 16km radius...")
    client = WazeAPIClient(api_key)
    try:
        raw_alerts = await client.get_traffic_alerts(lat, lon, 16.0)
        print(f"Raw API alerts: {len(raw_alerts)}")
        for i, alert in enumerate(raw_alerts):
            print(f"  Alert {i+1}: {alert.type} at {alert.latitude:.6f}, {alert.longitude:.6f}")
    except Exception as e:
        print(f"API error: {e}")
        return
    finally:
        await client.close()

    if not raw_alerts:
        print("No raw alerts to process")
        return

    print("2. Processing alerts through threat detector...")
    threat_detector = ThreatDetector()

    # Process alerts like RTI daemon does
    rti_state = threat_detector.process_threats(
        traffic_data=raw_alerts,
        current_location=(lat, lon),
        current_speed=20.0,  # 45 mph in m/s
        timestamp=int(time.time() * 1e9),
        v_cruise=22.0  # 50 mph cruise speed
    )

    print(f"Processed threats: {len(rti_state.threats)}")
    print(f"Threat ahead: {rti_state.threat_ahead}")
    print(f"Threat distance: {rti_state.threat_distance_m:.1f}m")
    print(f"Recommended speed: {rti_state.recommended_speed:.1f} m/s")

    for i, threat in enumerate(rti_state.threats):
        print(f"  Threat {i+1}: {threat.type} at {threat.distance:.0f}m, {threat.direction}, same_road={threat.on_same_road}")

    if not rti_state.threats:
        print("All threats filtered out by threat detector!")
        print("This explains why HUD shows no threats despite API finding them.")

if __name__ == "__main__":
    asyncio.run(test_threat_detector())
