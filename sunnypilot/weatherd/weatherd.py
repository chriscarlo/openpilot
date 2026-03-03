#!/usr/bin/env python3
"""
Weather Daemon — polls Open-Meteo for precipitation data and writes
a WeatherCondition param consumed by the longitudinal planner.

Architecture: GPS → HTTP poll (5-min) → Params write
"""

import json
import math
import time

import requests

from cereal import messaging
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

# WMO weather code → rain severity mapping
# See https://open-meteo.com/en/docs — "WMO Weather interpretation codes"
WMO_SEVERITY: dict[int, str] = {
  # Drizzle
  51: "light",
  53: "light",
  55: "moderate",
  56: "light",      # freezing drizzle (light)
  57: "moderate",    # freezing drizzle (dense)
  # Rain
  61: "light",
  63: "moderate",
  65: "heavy",
  66: "moderate",    # freezing rain (light)
  67: "heavy",       # freezing rain (heavy)
  # Showers
  80: "light",
  81: "moderate",
  82: "heavy",
  # Thunderstorm
  95: "heavy",
  96: "heavy",       # thunderstorm with slight hail
  99: "heavy",       # thunderstorm with heavy hail
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
POLL_INTERVAL_S = 300        # 5 minutes between API calls
LOCATION_CHANGE_KM = 5.0     # re-poll if moved >5 km
STALE_TIMEOUT_S = 900        # clear state after 15 min without a successful fetch
LOOP_HZ = 0.2                # 1 iteration per 5 seconds (light main loop)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Great-circle distance in km between two lat/lon points."""
  R = 6371.0
  dlat = math.radians(lat2 - lat1)
  dlon = math.radians(lon2 - lon1)
  a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
  return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _classify_weather(data: dict) -> dict:
  """Parse Open-Meteo 'current' block into a severity dict."""
  current = data.get("current", {})
  weather_code = int(current.get("weather_code", 0))
  precipitation_mm = float(current.get("precipitation", 0.0))
  rain_mm = float(current.get("rain", 0.0))
  showers_mm = float(current.get("showers", 0.0))
  snowfall_mm = float(current.get("snowfall", 0.0))

  severity = WMO_SEVERITY.get(weather_code, "none")

  # Belt-and-suspenders: if the WMO code says none but we see actual
  # precipitation, bump to at least light.
  if severity == "none" and precipitation_mm > 0.1:
    severity = "light"

  return {
    "weather_code": weather_code,
    "precipitation_mm": precipitation_mm,
    "rain_mm": rain_mm,
    "showers_mm": showers_mm,
    "snowfall_mm": snowfall_mm,
    "severity": severity,
    "timestamp": time.time(),
  }


def _fetch_weather(lat: float, lon: float) -> dict | None:
  """Fetch current weather from Open-Meteo. Returns parsed dict or None."""
  try:
    resp = requests.get(OPEN_METEO_URL, params={
      "latitude": round(lat, 4),
      "longitude": round(lon, 4),
      "current": "precipitation,rain,showers,snowfall,weather_code",
      "timezone": "auto",
    }, timeout=10)
    resp.raise_for_status()
    return _classify_weather(resp.json())
  except Exception as e:
    cloudlog.warning(f"weatherd: fetch failed: {e}")
    return None


def main():
  params = Params()
  gps_service = get_gps_location_service(params)
  sm = messaging.SubMaster([gps_service])

  rk = Ratekeeper(LOOP_HZ, print_delay_threshold=None)

  last_fetch_time = 0.0
  last_fetch_lat = 0.0
  last_fetch_lon = 0.0
  last_success_time = 0.0

  cloudlog.info("weatherd: started")

  while True:
    sm.update(0)

    # Read GPS
    gps = sm[gps_service]
    lat = gps.latitude
    lon = gps.longitude

    # Need valid GPS fix
    if abs(lat) < 0.1 and abs(lon) < 0.1:
      rk.keep_time()
      continue

    now = time.monotonic()
    time_since_fetch = now - last_fetch_time
    distance_km = _haversine_km(lat, lon, last_fetch_lat, last_fetch_lon) if last_fetch_time > 0 else float('inf')

    # Poll if enough time has passed or location changed significantly
    if time_since_fetch >= POLL_INTERVAL_S or distance_km >= LOCATION_CHANGE_KM:
      result = _fetch_weather(lat, lon)
      last_fetch_time = now
      last_fetch_lat = lat
      last_fetch_lon = lon

      if result is not None:
        last_success_time = now
        params.put("WeatherCondition", json.dumps(result))
        if result["severity"] != "none":
          cloudlog.info(f"weatherd: {result['severity']} precipitation (code {result['weather_code']}, {result['precipitation_mm']:.1f} mm)")

    # Clear stale data if we haven't had a successful fetch in a while
    if last_success_time > 0 and (now - last_success_time) > STALE_TIMEOUT_S:
      params.remove("WeatherCondition")
      last_success_time = 0.0
      cloudlog.info("weatherd: cleared stale weather data")

    rk.keep_time()


if __name__ == "__main__":
  main()
