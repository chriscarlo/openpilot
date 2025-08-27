#!/usr/bin/env python3
"""
Capture ALL Waze API data about police threats and save to file
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

# Import the Waze API client and related classes
import sys
sys.path.append('/data/openpilot')
from sunnypilot.rtid.waze_api_client import WazeAPIClient


async def capture_police_data(latitude: float = 37.4419, longitude: float = -122.1430, radius_km: float = 16.0):
    """
    Capture all Waze API data about police threats
    
    Args:
        latitude: GPS latitude (default: Palo Alto, CA)
        longitude: GPS longitude
        radius_km: Search radius in kilometers (default 16km = ~10 miles)
    """

    # Initialize output file
    output_file = Path("/data/openpilot/waze_police_data_capture.json")

    # Comprehensive data structure to capture everything
    capture_data = {
        "timestamp": datetime.now().isoformat(),
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km
        },
        "police_alerts": [],
        "all_raw_alerts": [],
        "api_response_metadata": {},
        "data_fields_found": set()
    }

    print(f"Capturing Waze police data at {latitude}, {longitude} with radius {radius_km}km...")

    try:
        # Create API client
        client = WazeAPIClient.from_persistent_key()

        async with client:
            # Temporarily patch the client to capture raw response
            original_make_request = client._make_request
            raw_response_data = {}

            async def capturing_make_request(endpoint, params, center_lat, center_lon):
                result = await original_make_request(endpoint, params, center_lat, center_lon)
                if result:
                    raw_response_data['full_response'] = result
                    capture_data['api_response_metadata'] = {
                        'endpoint': endpoint,
                        'params': params,
                        'response_status': result.get('status'),
                        'timestamp': time.time()
                    }
                return result

            client._make_request = capturing_make_request

            # Fetch all alerts
            alerts = await client.get_traffic_alerts(latitude, longitude, radius_km)

            # Process each alert
            for alert in alerts:
                # Capture ALL raw data from the alert
                if alert.raw_data:
                    capture_data['all_raw_alerts'].append(alert.raw_data)

                    # Track all fields we find
                    for key in alert.raw_data.keys():
                        capture_data['data_fields_found'].add(key)

                # Filter for police-related alerts
                if alert.type in ['police', 'policeHiding', 'POLICE', 'POLICE_HIDING']:
                    police_data = {
                        # Standard fields
                        "id": alert.id,
                        "type": alert.type,
                        "latitude": alert.latitude,
                        "longitude": alert.longitude,
                        "confidence": alert.confidence,
                        "speed_limit": alert.speed_limit,
                        "street": alert.street,
                        "country": alert.country,

                        # ALL raw data fields (if available)
                        "raw_data": alert.raw_data if alert.raw_data else {}
                    }

                    # Extract additional fields from raw_data if present
                    if alert.raw_data:
                        # Common Waze fields that might be present
                        potential_fields = [
                            'alert_id', 'uuid', 'pubMillis', 'reportMillis',
                            'reportRating', 'reliability', 'magvar',
                            'reportBy', 'wazeData', 'reportDescription',
                            'nThumbsUp', 'inscale', 'comments', 'isJamUnifiedAlert',
                            'jamUuid', 'additionalInfo', 'nearBy', 'imageUrl',
                            'reportMood', 'roadType', 'speed', 'direction',
                            'updateMillis', 'city', 'state', 'zipCode',
                            'subtype', 'alertType', 'alertSubType',
                            'numberOfReports', 'numberOfThumbsUp',
                            'distanceFromRoute', 'timeToReach',
                            'createdDate', 'updatedDate', 'expirationDate'
                        ]

                        for field in potential_fields:
                            if field in alert.raw_data:
                                police_data[f"raw_{field}"] = alert.raw_data[field]

                    capture_data['police_alerts'].append(police_data)

            # Add full raw response if we captured it
            if raw_response_data:
                capture_data['full_api_response'] = raw_response_data.get('full_response', {})

                # Extract all alerts from raw response for complete picture
                if 'data' in raw_response_data.get('full_response', {}):
                    response_data = raw_response_data['full_response']['data']

                    # Capture raw alerts structure
                    if 'alerts' in response_data:
                        capture_data['raw_alerts_from_api'] = response_data['alerts']

                    # Capture raw jams structure (sometimes contains police info)
                    if 'jams' in response_data:
                        capture_data['raw_jams_from_api'] = response_data['jams']

    except Exception as e:
        capture_data['error'] = str(e)
        print(f"Error capturing data: {e}")

    # Convert set to list for JSON serialization
    capture_data['data_fields_found'] = list(capture_data['data_fields_found'])

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(capture_data, f, indent=2, default=str)

    print(f"\nData saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("WAZE POLICE DATA CAPTURE SUMMARY")
    print("="*60)
    print(f"Total alerts captured: {len(capture_data['all_raw_alerts'])}")
    print(f"Police alerts found: {len(capture_data['police_alerts'])}")
    print("\nData fields discovered in alerts:")
    for field in sorted(capture_data['data_fields_found']):
        print(f"  - {field}")

    # Print detailed police alert info
    if capture_data['police_alerts']:
        print("\n" + "-"*60)
        print("POLICE ALERTS DETAIL:")
        print("-"*60)
        for i, police_alert in enumerate(capture_data['police_alerts'], 1):
            print(f"\nPolice Alert #{i}:")
            print(f"  ID: {police_alert.get('id')}")
            print(f"  Type: {police_alert.get('type')}")
            print(f"  Location: {police_alert.get('latitude')}, {police_alert.get('longitude')}")
            print(f"  Street: {police_alert.get('street')}")
            print(f"  Confidence: {police_alert.get('confidence')}")

            # Print all raw fields
            raw_data = police_alert.get('raw_data', {})
            if raw_data:
                print("  Raw data fields:")
                for key, value in raw_data.items():
                    print(f"    {key}: {value}")

    return capture_data


async def continuous_capture(duration_minutes: int = 5, interval_seconds: int = 30):
    """
    Continuously capture police data for a specified duration
    
    Args:
        duration_minutes: How long to run capture
        interval_seconds: Seconds between captures
    """
    print(f"Starting continuous capture for {duration_minutes} minutes...")

    all_captures = []
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    while time.time() < end_time:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Capturing data...")

        # Capture data at current location (you can modify these coordinates)
        data = await capture_police_data()
        all_captures.append(data)

        # Wait before next capture
        remaining_time = end_time - time.time()
        if remaining_time > interval_seconds:
            print(f"Waiting {interval_seconds} seconds until next capture...")
            await asyncio.sleep(interval_seconds)
        else:
            break

    # Save all captures to a comprehensive file
    output_file = Path(f"/data/openpilot/waze_police_continuous_capture_{int(start_time)}.json")
    with open(output_file, 'w') as f:
        json.dump(all_captures, f, indent=2, default=str)

    print(f"\nContinuous capture complete. Data saved to: {output_file}")
    print(f"Total captures: {len(all_captures)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture Waze police threat data")
    parser.add_argument("--lat", type=float, default=37.4419, help="Latitude")
    parser.add_argument("--lon", type=float, default=-122.1430, help="Longitude")
    parser.add_argument("--radius", type=float, default=16.0, help="Search radius in km")
    parser.add_argument("--continuous", action="store_true", help="Run continuous capture")
    parser.add_argument("--duration", type=int, default=5, help="Duration in minutes for continuous capture")
    parser.add_argument("--interval", type=int, default=30, help="Interval in seconds between captures")

    args = parser.parse_args()

    if args.continuous:
        asyncio.run(continuous_capture(args.duration, args.interval))
    else:
        asyncio.run(capture_police_data(args.lat, args.lon, args.radius))
