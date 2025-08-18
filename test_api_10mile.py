#!/usr/bin/env python3

"""
Test RTI API for 10 mile radius
Check for threats in wider area around current location
"""

import asyncio
import json
import math
from cereal import messaging
from sunnypilot.rtid.waze_api_client import WazeAPIClient

def generate_grid_points(center_lat, center_lon, radius_miles=10, grid_size=5):
    """Generate grid of test points within radius"""
    # Convert miles to degrees (rough approximation)
    lat_degree_miles = 69.0  # 1 degree lat ≈ 69 miles
    lon_degree_miles = 69.0 * math.cos(math.radians(center_lat))  # Adjust for longitude

    lat_range = radius_miles / lat_degree_miles
    lon_range = radius_miles / lon_degree_miles

    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Create grid from -range to +range
            lat_offset = (i / (grid_size - 1) - 0.5) * 2 * lat_range
            lon_offset = (j / (grid_size - 1) - 0.5) * 2 * lon_range

            test_lat = center_lat + lat_offset
            test_lon = center_lon + lon_offset

            # Calculate distance to verify within radius
            distance_miles = math.sqrt(
                (lat_offset * lat_degree_miles) ** 2 +
                (lon_offset * lon_degree_miles) ** 2
            )

            if distance_miles <= radius_miles:
                points.append((test_lat, test_lon, distance_miles))

    return points

async def test_api_10mile():
    """Test API calls across 10 mile radius"""
    print("=== RTI API 10 Mile Radius Test ===")

    # Get current GPS location
    sm = messaging.SubMaster(['gpsLocationExternal', 'gpsLocation'])
    sm.update(2000)

    center_lat, center_lon = None, None

    if sm.updated['gpsLocationExternal']:
        gps_ext = sm['gpsLocationExternal']
        if gps_ext.accuracy < 10.0:
            center_lat, center_lon = gps_ext.latitude, gps_ext.longitude

    if not center_lat and sm.updated['gpsLocation']:
        gps_loc = sm['gpsLocation']
        if gps_loc.hasFix:
            center_lat, center_lon = gps_loc.latitude, gps_loc.longitude

    if not center_lat:
        print("No GPS available, using test coordinates")
        center_lat, center_lon = 37.4221, -122.0841

    print(f"Center location: {center_lat:.6f}, {center_lon:.6f}")

    # Load API key
    try:
        with open('/persist/waze/waze_rapidapi.json') as f:
            key_data = json.load(f)
            api_key = key_data.get('api_key')
    except Exception as e:
        print(f"Failed to load API key: {e}")
        return

    # Generate test points in 10 mile radius
    test_points = generate_grid_points(center_lat, center_lon, radius_miles=10, grid_size=5)
    print(f"Testing {len(test_points)} locations within 10 mile radius...")

    total_alerts = 0
    locations_with_alerts = 0

    try:
        client = WazeAPIClient(api_key)

        for i, (lat, lon, distance) in enumerate(test_points):
            print(f"\nPoint {i+1}/{len(test_points)}: {lat:.4f}, {lon:.4f} ({distance:.1f}mi from center)")

            try:
                alerts = await client.get_traffic_alerts(lat, lon)

                if alerts:
                    locations_with_alerts += 1
                    total_alerts += len(alerts)
                    print(f"  FOUND {len(alerts)} alerts:")
                    for j, alert in enumerate(alerts[:5]):  # Show up to 5 alerts
                        print(f"    {j+1}. {alert.type} at {alert.distance:.0f}m (confidence: {alert.confidence:.2f})")
                else:
                    print("  No alerts found")

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"  API call failed: {e}")

        await client.close()

        print("\n=== SUMMARY ===")
        print(f"Locations tested: {len(test_points)}")
        print(f"Locations with alerts: {locations_with_alerts}")
        print(f"Total alerts found: {total_alerts}")
        print(f"Average alerts per location: {total_alerts/len(test_points):.1f}")

        if total_alerts == 0:
            print("\nNo threats detected in 10 mile radius")
            print("This explains why RTI shows no alerts - the area is clean!")
        else:
            print("\nThreats detected in area - RTI should be showing alerts")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_10mile())
