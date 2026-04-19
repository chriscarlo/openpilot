#!/usr/bin/env python3
"""
Weather Daemon — polls Pirate Weather's point-forecast API for the
precipitation intensity at the ego's current lat/lon, and writes a
WeatherCondition param consumed by the longitudinal planner (WeatherController).

Pirate Weather follows the Dark Sky-compatible schema. We fetch:

    https://api.pirateweather.net/forecast/{KEY}/{lat},{lon}?units=si

and parse the `currently` block for precipIntensity (mm/hr at units=si) and
time (unix seconds — the observation wall-clock the controller uses for its
freshness check).

Pipeline:
  1. Read ego GPS from the active location service.
  2. If the poll interval has elapsed or the car has moved more than
     LOCATION_CHANGE_KM, hit /forecast.
  3. Parse currently.precipIntensity → mm/hr + currently.time → timestamp.
  4. Classify severity via Marshall-Palmer-equivalent thresholds and write
     the WeatherCondition param as JSON.
  5. Clear the param after STALE_TIMEOUT_S without a successful fetch so a
     dropped network doesn't strand a stale reading in front of the planner.

The 10k/month free tier is the bottleneck. The poll cadence + location gate
are the only rate-limiter; at 5 min polling plus typical drive durations,
worst-case monthly usage stays well below 10k.
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
from openpilot.sunnypilot.weatherd.api_key_manager import get_api_key

PIRATE_WEATHER_URL = "https://api.pirateweather.net/forecast/{key}/{lat},{lon}"
PIRATE_WEATHER_UNITS = "si"      # precipIntensity in mm/hr, wind in m/s

POLL_INTERVAL_S = 300            # 5 minutes between Pirate Weather polls
LOCATION_CHANGE_KM = 5.0         # re-poll if moved >5 km
STALE_TIMEOUT_S = 900            # clear state after 15 min without a successful fetch
LOOP_HZ = 0.2                    # 1 iteration per 5 seconds (light main loop)
HTTP_TIMEOUT_S = 10

# Severity thresholds (mm/hr). Pinned to the palette discontinuities of the
# RainViewer Universal Blue scheme (dBZ 15 / 35 / 45) converted via Marshall-
# Palmer Z = 200·R^1.6, matching WeatherController.PRECIP_LIGHT/MODERATE/HEAVY.
# The severity label for a given mm/hr therefore corresponds to the color band
# the same mm/hr would paint onto the HUD overlay.
SEVERITY_LIGHT_MM = 0.32     # dBZ 15 — cyan appears
SEVERITY_MODERATE_MM = 5.62  # dBZ 35 — yellow appears
SEVERITY_HEAVY_MM = 23.67    # dBZ 45 — red appears


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Great-circle distance in km between two lat/lon points."""
  R = 6371.0
  dlat = math.radians(lat2 - lat1)
  dlon = math.radians(lon2 - lon1)
  a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
  return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def severity_from_mm_per_hr(mm_per_hr: float) -> str:
  """Map a mm/hr intensity to a coarse severity string for the WeatherCondition JSON."""
  if mm_per_hr < SEVERITY_LIGHT_MM:
    return "none"
  if mm_per_hr < SEVERITY_MODERATE_MM:
    return "light"
  if mm_per_hr < SEVERITY_HEAVY_MM:
    return "moderate"
  return "heavy"


def fetch_point_conditions(api_key: str, lat: float, lon: float) -> tuple[float, int] | None:
  """Return (mm_per_hr, unix_seconds) for the ego position, or None on failure.

  Pirate Weather's `currently.precipIntensity` is documented as mm/hr when
  `units=si`. `currently.time` is the observation wall-clock (unix seconds)
  which the controller's freshness check compares against time.time().
  """
  url = PIRATE_WEATHER_URL.format(key=api_key, lat=lat, lon=lon)
  try:
    resp = requests.get(url, params={"units": PIRATE_WEATHER_UNITS}, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    payload = resp.json()
  except Exception as e:
    cloudlog.warning(f"weatherd: pirate weather fetch failed: {e}")
    return None

  currently = payload.get("currently") or {}
  precip_raw = currently.get("precipIntensity", 0.0)
  try:
    precip_mm = max(0.0, float(precip_raw or 0.0))
  except (TypeError, ValueError):
    cloudlog.warning(f"weatherd: pirate weather precipIntensity not numeric: {precip_raw!r}")
    return None

  timestamp_raw = currently.get("time", 0)
  try:
    timestamp = int(timestamp_raw or 0)
  except (TypeError, ValueError):
    cloudlog.warning(f"weatherd: pirate weather currently.time not numeric: {timestamp_raw!r}")
    return None
  if timestamp <= 0:
    cloudlog.warning("weatherd: pirate weather response missing currently.time")
    return None
  return precip_mm, timestamp


def main():
  params = Params()
  gps_service = get_gps_location_service(params)
  sm = messaging.SubMaster([gps_service])

  rk = Ratekeeper(LOOP_HZ, print_delay_threshold=None)

  last_fetch_time = 0.0
  last_fetch_lat = 0.0
  last_fetch_lon = 0.0
  last_success_time = 0.0
  last_published_timestamp = 0
  missing_key_logged = False

  cloudlog.info("weatherd: started (Pirate Weather source)")

  while True:
    sm.update(0)

    gps = sm[gps_service]
    lat = gps.latitude
    lon = gps.longitude
    if abs(lat) < 0.1 and abs(lon) < 0.1:
      rk.keep_time()
      continue

    api_key = get_api_key()
    if not api_key:
      if not missing_key_logged:
        cloudlog.warning("weatherd: no Pirate Weather API key found — idling")
        missing_key_logged = True
      rk.keep_time()
      continue
    missing_key_logged = False

    now = time.monotonic()
    time_since_fetch = now - last_fetch_time
    distance_km = _haversine_km(lat, lon, last_fetch_lat, last_fetch_lon) if last_fetch_time > 0 else float('inf')

    if time_since_fetch >= POLL_INTERVAL_S or distance_km >= LOCATION_CHANGE_KM:
      result = fetch_point_conditions(api_key, lat, lon)
      last_fetch_time = now
      last_fetch_lat = lat
      last_fetch_lon = lon

      if result is not None:
        mm_per_hr, timestamp = result
        last_success_time = now
        # Only publish when the observation actually changed, or when the
        # poll cadence elapsed (so the controller's freshness check sees a
        # refreshed timestamp even if precip is unchanged).
        timestamp_changed = timestamp != last_published_timestamp
        if timestamp_changed or time_since_fetch >= POLL_INTERVAL_S:
          severity = severity_from_mm_per_hr(mm_per_hr)
          payload = {
            "precipitation_mm": float(mm_per_hr),
            "severity": severity,
            "timestamp": int(timestamp),
            "frame_time": int(timestamp),
          }
          params.put("WeatherCondition", json.dumps(payload))
          last_published_timestamp = timestamp
          if severity != "none":
            obs_age = int(time.time() - timestamp)
            cloudlog.info(f"weatherd: {severity} precipitation ({mm_per_hr:.2f} mm/hr, obs age {obs_age}s)")

    if last_success_time > 0 and (now - last_success_time) > STALE_TIMEOUT_S:
      params.remove("WeatherCondition")
      last_success_time = 0.0
      last_published_timestamp = 0
      cloudlog.info("weatherd: cleared stale weather data")

    rk.keep_time()


if __name__ == "__main__":
  main()
