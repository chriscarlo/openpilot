#!/usr/bin/env python3
"""
Weather Daemon — polls RainViewer's public doppler-radar API for the
precipitation intensity at the ego's current lat/lon, and writes a
WeatherCondition param consumed by the longitudinal planner.

Unlike an NWP forecast, doppler radar is an observation: we get "is it
raining at this exact point, right now". RainViewer's scans refresh every
10 minutes and the newest frame is typically 0-10 minutes old.

Pipeline:
  1. GET weather-maps.json — list of available radar frames (cached 4 min).
  2. GET a 256x256 radar tile CENTERED at the ego lat/lon.
  3. Decode the PNG using a pure-Python stdlib decoder (no Pillow dep).
  4. Sample a 3x3 pixel window at the tile center, take max intensity.
  5. Classify RGBA via the Universal Blue palette to get mm/hour.
  6. Write WeatherCondition with that intensity + the radar frame's wall-clock
     time so the controller's freshness check measures radar staleness.
"""

import json
import math
import struct
import time
import zlib

import requests

from cereal import messaging
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.weatherd.radar_palette import (
  sample_intensity_max,
  severity_from_mm_per_hr,
)

METADATA_URL = "https://api.rainviewer.com/public/weather-maps.json"
# Tile endpoint format: {host}{path}/{size}/{z}/{lat}/{lon}/{color}/{smooth}_{snow}.png
# color=2 → "Universal Blue" palette, smooth=0 → discrete pixels (no blur),
# snow=0 → snow rendered in same palette (no separate color scheme).
TILE_SIZE = 256
TILE_ZOOM = 7                 # ~1.2 km/pixel at 40°N — balances nearby-rain capture vs ego precision
TILE_COLOR_SCHEME = 2
TILE_OPTIONS = "0_0"          # smooth=0, snow=0
SAMPLE_WINDOW_PX = 3          # 3x3 max around the tile center

POLL_INTERVAL_S = 300         # 5 minutes between RainViewer polls
LOCATION_CHANGE_KM = 5.0      # re-poll if moved >5 km
STALE_TIMEOUT_S = 900         # clear state after 15 min without a successful fetch
METADATA_CACHE_S = 240        # reuse metadata for up to 4 min (just under the 5-min cadence)
LOOP_HZ = 0.2                 # 1 iteration per 5 seconds (light main loop)
HTTP_TIMEOUT_S = 10


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Great-circle distance in km between two lat/lon points."""
  R = 6371.0
  dlat = math.radians(lat2 - lat1)
  dlon = math.radians(lon2 - lon1)
  a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
  return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _decode_png_rgba(data: bytes) -> tuple[int, int, bytes] | None:
  """Decode a PNG into (width, height, rgba_bytes). Stdlib-only.

  Supports RGBA (color type 6) at 8 bits per channel — which is what
  RainViewer serves. Returns None on malformed input or unsupported format.
  """
  if len(data) < 8 or data[:8] != b'\x89PNG\r\n\x1a\n':
    return None
  i = 8
  idat_chunks: list[bytes] = []
  width = height = bit_depth = color_type = 0
  try:
    while i < len(data):
      length = struct.unpack('>I', data[i:i + 4])[0]
      i += 4
      chunk_type = data[i:i + 4]
      i += 4
      chunk_data = data[i:i + length]
      i += length
      i += 4  # skip CRC
      if chunk_type == b'IHDR':
        width, height, bit_depth, color_type = struct.unpack('>IIBB', chunk_data[:10])
      elif chunk_type == b'IDAT':
        idat_chunks.append(chunk_data)
      elif chunk_type == b'IEND':
        break
    if not idat_chunks or bit_depth != 8 or color_type != 6:
      return None
    raw = zlib.decompress(b''.join(idat_chunks))
  except Exception:
    return None

  bpp = 4  # RGBA
  stride = width * bpp
  pixels = bytearray(height * stride)
  prev_row = bytearray(stride)
  off = 0
  for y in range(height):
    filter_byte = raw[off]
    off += 1
    row = bytearray(raw[off:off + stride])
    off += stride
    if filter_byte == 1:        # Sub
      for x in range(bpp, stride):
        row[x] = (row[x] + row[x - bpp]) & 0xFF
    elif filter_byte == 2:      # Up
      for x in range(stride):
        row[x] = (row[x] + prev_row[x]) & 0xFF
    elif filter_byte == 3:      # Average
      for x in range(stride):
        left = row[x - bpp] if x >= bpp else 0
        row[x] = (row[x] + (left + prev_row[x]) // 2) & 0xFF
    elif filter_byte == 4:      # Paeth
      for x in range(stride):
        a = row[x - bpp] if x >= bpp else 0
        b = prev_row[x]
        c = prev_row[x - bpp] if x >= bpp else 0
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc:
          pred = a
        elif pb <= pc:
          pred = b
        else:
          pred = c
        row[x] = (row[x] + pred) & 0xFF
    # filter_byte == 0 → no filter, row already correct
    pixels[y * stride:y * stride + stride] = row
    prev_row = row
  return width, height, bytes(pixels)


class MetadataCache:
  """Memoizes the RainViewer metadata response across polls."""

  def __init__(self):
    self._host: str | None = None
    self._path: str | None = None
    self._frame_time: int = 0
    self._fetched_at: float = 0.0

  def get(self, now_monotonic: float) -> tuple[str, str, int] | None:
    """Return (host, path, frame_time) or None on failure."""
    if self._host and (now_monotonic - self._fetched_at) < METADATA_CACHE_S:
      return self._host, self._path, self._frame_time  # type: ignore[return-value]
    try:
      resp = requests.get(METADATA_URL, timeout=HTTP_TIMEOUT_S)
      resp.raise_for_status()
      meta = resp.json()
      past = meta.get("radar", {}).get("past") or []
      if not past:
        return None
      newest = past[-1]
      self._host = str(meta["host"])
      self._path = str(newest["path"])
      self._frame_time = int(newest["time"])
      self._fetched_at = now_monotonic
      return self._host, self._path, self._frame_time
    except Exception as e:
      cloudlog.warning(f"weatherd: metadata fetch failed: {e}")
      return None


def _fetch_tile(host: str, path: str, lat: float, lon: float) -> tuple[int, int, bytes] | None:
  """Fetch the radar tile centered on (lat, lon) and decode to RGBA bytes."""
  # Round to 4 decimals (~11 m precision — more than enough) to increase tile cache hits.
  url = f"{host}{path}/{TILE_SIZE}/{TILE_ZOOM}/{round(lat, 4)}/{round(lon, 4)}/{TILE_COLOR_SCHEME}/{TILE_OPTIONS}.png"
  try:
    resp = requests.get(url, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
  except Exception as e:
    cloudlog.warning(f"weatherd: tile fetch failed: {e}")
    return None
  return _decode_png_rgba(resp.content)


def _measure_precipitation(lat: float, lon: float, metadata: MetadataCache,
                           now_monotonic: float) -> tuple[float, int] | None:
  """Return (mm_per_hr, frame_time) for ego at (lat, lon), or None on failure."""
  meta = metadata.get(now_monotonic)
  if meta is None:
    return None
  host, path, frame_time = meta

  tile = _fetch_tile(host, path, lat, lon)
  if tile is None:
    return None
  width, height, pixels = tile

  center_x = width // 2
  center_y = height // 2
  mm = sample_intensity_max(pixels, width, height, center_x, center_y, SAMPLE_WINDOW_PX)
  return mm, frame_time


def main():
  params = Params()
  gps_service = get_gps_location_service(params)
  sm = messaging.SubMaster([gps_service])

  rk = Ratekeeper(LOOP_HZ, print_delay_threshold=None)
  metadata = MetadataCache()

  last_fetch_time = 0.0
  last_fetch_lat = 0.0
  last_fetch_lon = 0.0
  last_success_time = 0.0
  last_published_frame_time = 0

  cloudlog.info("weatherd: started (RainViewer doppler source)")

  while True:
    sm.update(0)

    gps = sm[gps_service]
    lat = gps.latitude
    lon = gps.longitude
    if abs(lat) < 0.1 and abs(lon) < 0.1:
      rk.keep_time()
      continue

    now = time.monotonic()
    time_since_fetch = now - last_fetch_time
    distance_km = _haversine_km(lat, lon, last_fetch_lat, last_fetch_lon) if last_fetch_time > 0 else float('inf')

    if time_since_fetch >= POLL_INTERVAL_S or distance_km >= LOCATION_CHANGE_KM:
      result = _measure_precipitation(lat, lon, metadata, now)
      last_fetch_time = now
      last_fetch_lat = lat
      last_fetch_lon = lon

      if result is not None:
        mm_per_hr, frame_time = result
        last_success_time = now
        # Only publish when the frame actually changed OR enough poll time has
        # passed to refresh staleness. This prevents thrashing on the params
        # write path when metadata is cached and rain is constant.
        frame_changed = frame_time != last_published_frame_time
        if frame_changed or time_since_fetch >= POLL_INTERVAL_S:
          severity = severity_from_mm_per_hr(mm_per_hr)
          payload = {
            "precipitation_mm": float(mm_per_hr),
            "severity": severity,
            "timestamp": int(frame_time),
            "frame_time": int(frame_time),
          }
          params.put("WeatherCondition", json.dumps(payload))
          last_published_frame_time = frame_time
          if severity != "none":
            frame_age = int(time.time() - frame_time)
            cloudlog.info(f"weatherd: {severity} precipitation ({mm_per_hr:.2f} mm/hr, frame age {frame_age}s)")

    if last_success_time > 0 and (now - last_success_time) > STALE_TIMEOUT_S:
      params.remove("WeatherCondition")
      last_success_time = 0.0
      last_published_frame_time = 0
      cloudlog.info("weatherd: cleared stale weather data")

    rk.keep_time()


if __name__ == "__main__":
  main()
