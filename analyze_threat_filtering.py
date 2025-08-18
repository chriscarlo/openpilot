#!/usr/bin/env python3

"""
Analyze Threat Filtering
Debug why threat detector filters out threats
"""

import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth"""
    R = 6371000  # Earth radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def analyze_threat():
    """Analyze the police threat that's being filtered"""
    print("=== Threat Filtering Analysis ===")

    # Current location and threat location from our test
    ego_lat, ego_lon = 38.753421, -120.623170
    threat_lat, threat_lon = 38.760325, -120.521602

    print(f"Ego location: {ego_lat:.6f}, {ego_lon:.6f}")
    print(f"Threat location: {threat_lat:.6f}, {threat_lon:.6f}")

    # Calculate distance
    distance_m = haversine_distance(ego_lat, ego_lon, threat_lat, threat_lon)
    distance_km = distance_m / 1000
    distance_mi = distance_m / 1609.34

    print(f"Distance: {distance_m:.0f}m ({distance_km:.1f}km, {distance_mi:.1f}mi)")

    # Check common filtering thresholds
    print("\nFiltering Analysis:")
    print(f"  Distance > 1600m (1 mile): {distance_m > 1600}")
    print(f"  Distance > 5000m (5km original): {distance_m > 5000}")
    print(f"  Distance > 16000m (16km new): {distance_m > 16000}")

    # Calculate bearing
    lat1_rad = math.radians(ego_lat)
    lat2_rad = math.radians(threat_lat)
    delta_lon_rad = math.radians(threat_lon - ego_lon)

    y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad))

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    if bearing_deg < 0:
        bearing_deg += 360

    print(f"  Bearing: {bearing_deg:.1f}° (0°=N, 90°=E, 180°=S, 270°=W)")

    # Direction analysis
    if bearing_deg < 45 or bearing_deg > 315:
        direction = "North"
    elif bearing_deg < 135:
        direction = "East"
    elif bearing_deg < 225:
        direction = "South"
    else:
        direction = "West"

    print(f"  Direction: {direction}")
    print(f"  Threat is ~{distance_mi:.1f} miles {direction} of current position")

    print("\nPossible reasons for filtering:")
    print("1. Distance threshold - threat may be beyond max detection range")
    print("2. Road matching - threat may not be on same road/route")
    print("3. Direction filtering - threat may not be 'ahead' on route")
    print("4. Confidence threshold - threat confidence may be too low")

if __name__ == "__main__":
    analyze_threat()
