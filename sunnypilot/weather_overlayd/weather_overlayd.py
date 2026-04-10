#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from io import BytesIO

import requests
from PIL import Image

from cereal import messaging
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

from .overlay_core import (
  CANVAS_SIZE_PX,
  CAR_ANCHOR_X,
  CAR_ANCHOR_Y,
  OverlayConfig,
  Viewport,
  build_viewport,
  compose_layer,
  encode_png,
  image_has_precipitation,
  iter_tile_requests,
)

RAINVIEWER_METADATA_URL = "https://api.rainviewer.com/public/weather-maps.json"
RAINVIEWER_TILE_SIZE = 256
RAINVIEWER_COLOR_SCHEME = 2       # Universal Blue — free-tier only allows this scheme
RAINVIEWER_TILE_OPTIONS = "1_1"   # smooth=1 (anti-aliased wash), snow=1 (separate snow palette)
RAINVIEWER_ZOOM_MAX = 7           # Free-tier cap
RAINVIEWER_ZOOM_MIN = 6

METADATA_REFRESH_S = 300          # Refresh RainViewer frame index every ~5 min
LOOP_HZ = 0.5
API_TIMEOUT_S = 10
REFRESH_JITTER_S = 15
RECENTER_MIN_DISTANCE_M = 100.0


@dataclass
class CachedTile:
  image: Image.Image
  fetched_monotonic: float


@dataclass
class RadarFrame:
  host: str
  path: str
  time_unix: int


class TileLookup:
  def __init__(self, daemon: "WeatherOverlayDaemon", requests_):
    self.daemon = daemon
    self.requests = requests_

  def get_tile(self, request):
    return self.daemon.get_cached_tile(request)


class WeatherOverlayDaemon:
  def __init__(self):
    self.params = Params()
    self.session = requests.Session()
    self.pm = messaging.PubMaster(["weatherOverlaySP"])

    self.gps_service = get_gps_location_service(self.params)
    self.sm = messaging.SubMaster([self.gps_service])

    # Tile cache keyed by (frame_path, zoom, tile_x, tile_y). When the active
    # RainViewer frame rotates, non-matching entries are dropped so compose_layer
    # never mixes tiles from different scan times.
    self.cache: dict[tuple[str, int, int, int], CachedTile] = {}
    self.active_frame: RadarFrame | None = None
    self.metadata_fetched_monotonic: float = 0.0

    self.last_config: OverlayConfig | None = None
    self.last_publish_lat = math.nan
    self.last_publish_lon = math.nan
    self.last_publish_signature: tuple | None = None
    self.last_fetch_error = False

  def read_config(self) -> OverlayConfig:
    def clamp_int(key: str, default: int, lo: int, hi: int) -> int:
      raw = self.params.get(key)
      if not raw:
        return default
      try:
        return max(lo, min(hi, int(raw)))
      except ValueError:
        return default

    return OverlayConfig(
      range_km=clamp_int("WeatherOverlayRangeKm", 16, 3, 40),
      zoom=clamp_int("WeatherOverlayZoomLevel", RAINVIEWER_ZOOM_MAX, RAINVIEWER_ZOOM_MIN, RAINVIEWER_ZOOM_MAX),
      refresh_seconds=clamp_int("WeatherOverlayRefreshSeconds", 120, 30, 600),
    )

  def get_location(self) -> tuple[float, float] | None:
    gps = self.sm[self.gps_service]
    lat = getattr(gps, "latitude", 0.0)
    lon = getattr(gps, "longitude", 0.0)
    if abs(lat) < 0.1 and abs(lon) < 0.1:
      return None
    return lat, lon

  @staticmethod
  def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2.0) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2.0) ** 2
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

  def should_recenter(self, lat: float, lon: float, config: OverlayConfig) -> bool:
    if math.isnan(self.last_publish_lat) or math.isnan(self.last_publish_lon):
      return True
    distance_m = self.haversine_m(lat, lon, self.last_publish_lat, self.last_publish_lon)
    threshold_m = max(RECENTER_MIN_DISTANCE_M, config.range_km * 1000.0 * 0.03)
    return distance_m >= threshold_m

  def refresh_metadata(self, now_monotonic: float) -> RadarFrame | None:
    """Fetch the RainViewer weather-maps index every METADATA_REFRESH_S.

    Picks the newest entry from `radar.past` (nowcast frames are skipped per
    the free-tier constraint). When the frame path rotates, invalidates all
    cached tiles belonging to the old frame.
    """
    if self.active_frame is not None and (now_monotonic - self.metadata_fetched_monotonic) < METADATA_REFRESH_S:
      return self.active_frame

    try:
      resp = self.session.get(RAINVIEWER_METADATA_URL, timeout=API_TIMEOUT_S)
      resp.raise_for_status()
      meta = resp.json()
    except Exception as e:
      self.last_fetch_error = True
      cloudlog.warning(f"weather_overlayd: metadata fetch failed: {e}")
      return self.active_frame

    past = (meta.get("radar") or {}).get("past") or []
    if not past:
      self.last_fetch_error = True
      cloudlog.warning("weather_overlayd: RainViewer metadata has no past frames")
      return self.active_frame

    newest = past[-1]
    host = str(meta.get("host", ""))
    path = str(newest.get("path", ""))
    if not host or not path:
      self.last_fetch_error = True
      cloudlog.warning("weather_overlayd: RainViewer metadata missing host/path")
      return self.active_frame

    frame = RadarFrame(host=host, path=path, time_unix=int(newest.get("time", 0) or 0))
    self.metadata_fetched_monotonic = now_monotonic

    if self.active_frame is None or frame.path != self.active_frame.path:
      if self.active_frame is not None:
        self.cache = {k: v for k, v in self.cache.items() if k[0] == frame.path}
      cloudlog.info(f"weather_overlayd: active radar frame → {frame.path} (t={frame.time_unix})")

    self.active_frame = frame
    self.last_fetch_error = False
    return frame

  def build_tile_url(self, frame: RadarFrame, zoom: int, tile_x: int, tile_y: int) -> str:
    return (
      f"{frame.host}{frame.path}/{RAINVIEWER_TILE_SIZE}/{zoom}/{tile_x}/{tile_y}"
      f"/{RAINVIEWER_COLOR_SCHEME}/{RAINVIEWER_TILE_OPTIONS}.png"
    )

  def cache_key(self, frame_path: str, zoom: int, tile_x: int, tile_y: int) -> tuple[str, int, int, int]:
    return frame_path, zoom, tile_x, tile_y

  def fetch_tile(self, frame: RadarFrame, zoom: int, tile_x: int, tile_y: int) -> Image.Image | None:
    url = self.build_tile_url(frame, zoom, tile_x, tile_y)
    try:
      resp = self.session.get(url, timeout=API_TIMEOUT_S)
      resp.raise_for_status()
      image = Image.open(BytesIO(resp.content)).convert("RGBA")
      image.load()
      self.last_fetch_error = False
      return image
    except Exception as e:
      self.last_fetch_error = True
      cloudlog.warning(f"weather_overlayd: tile fetch failed for {zoom}/{tile_x}/{tile_y}: {e}")
      return None

  def ensure_tiles(self, frame: RadarFrame, config: OverlayConfig, viewport: Viewport, now_monotonic: float) -> list:
    tile_requests = iter_tile_requests(config.zoom, viewport)
    for request in tile_requests:
      key = self.cache_key(frame.path, config.zoom, request.tile_x, request.tile_y)
      cached = self.cache.get(key)
      tile_expired = cached is None or (now_monotonic - cached.fetched_monotonic) >= config.refresh_seconds
      if tile_expired:
        image = self.fetch_tile(frame, config.zoom, request.tile_x, request.tile_y)
        if image is not None:
          self.cache[key] = CachedTile(image=image, fetched_monotonic=now_monotonic)
    return tile_requests

  def get_cached_tile(self, request) -> Image.Image | None:
    if self.active_frame is None:
      return None
    cached = self.cache.get(self.cache_key(self.active_frame.path, request.zoom, request.tile_x, request.tile_y))
    return None if cached is None else cached.image

  def cache_age_seconds(self, frame: RadarFrame, requests_) -> float:
    ages = []
    now = time.monotonic()
    for request in requests_:
      cached = self.cache.get(self.cache_key(frame.path, request.zoom, request.tile_x, request.tile_y))
      if cached is not None:
        ages.append(now - cached.fetched_monotonic)
    return max(ages) if ages else float("inf")

  def publish_state(self, *, available: bool, precipitation_in_range: bool, stale: bool,
                    config: OverlayConfig, precip_png: bytes = b"",
                    provider_status: str = "offline", precip_tile_count: int = 0) -> None:
    msg = messaging.new_message("weatherOverlaySP", valid=available or provider_status != "ok")
    overlay = msg.weatherOverlaySP
    overlay.available = available
    overlay.precipitationInRange = precipitation_in_range
    overlay.stale = stale
    overlay.zoom = config.zoom
    overlay.rangeKm = float(config.range_km)
    overlay.carAnchorX = CAR_ANCHOR_X
    overlay.carAnchorY = CAR_ANCHOR_Y
    # RainViewer's Universal Blue scheme is a single combined precip palette
    # (rain + snow share tones). Route the composited PNG through rainLayerPng
    # so the existing HUD wash keeps working; snowLayerPng stays empty.
    overlay.rainLayerPng = precip_png
    overlay.snowLayerPng = b""
    overlay.generatedAtUnixSec = int(time.time())
    overlay.providerStatus = provider_status
    overlay.rainTileCount = precip_tile_count
    overlay.snowTileCount = 0
    self.pm.send("weatherOverlaySP", msg)

  def maybe_publish_unavailable(self, provider_status: str, config: OverlayConfig) -> None:
    signature = (provider_status, config.range_km, config.zoom, config.refresh_seconds)
    if signature == self.last_publish_signature:
      return
    self.publish_state(
      available=False,
      precipitation_in_range=False,
      stale=True,
      config=config,
      provider_status=provider_status,
    )
    self.last_publish_signature = signature

  def run(self):
    rk = Ratekeeper(LOOP_HZ, print_delay_threshold=None)
    cloudlog.info("weather_overlayd: started (RainViewer source)")

    while True:
      self.sm.update(0)
      config = self.read_config()

      if self.last_config != config:
        self.cache.clear()
        self.last_publish_signature = None
        self.last_config = config

      location = self.get_location()
      if location is None:
        self.maybe_publish_unavailable("offline", config)
        rk.keep_time()
        continue

      lat, lon = location
      now_monotonic = time.monotonic()

      frame = self.refresh_metadata(now_monotonic)
      if frame is None:
        self.maybe_publish_unavailable("fetchError", config)
        rk.keep_time()
        continue

      viewport = build_viewport(lat, lon, config.range_km, config.zoom)
      precip_requests = self.ensure_tiles(frame, config, viewport, now_monotonic)
      precip_lookup = TileLookup(self, precip_requests)
      precip_image, precip_tile_count = compose_layer(precip_lookup, viewport, CANVAS_SIZE_PX)

      available = precip_tile_count > 0
      if not available:
        self.maybe_publish_unavailable("fetchError" if self.last_fetch_error else "offline", config)
        rk.keep_time()
        continue

      precipitation_in_range = image_has_precipitation(precip_image)
      tile_age_s = self.cache_age_seconds(frame, precip_requests)
      stale = tile_age_s > (config.refresh_seconds + REFRESH_JITTER_S)
      signature = (
        config.range_km,
        config.zoom,
        config.refresh_seconds,
        precipitation_in_range,
        stale,
        precip_tile_count,
        frame.path,
        "ok" if not self.last_fetch_error else "fetchError",
      )

      should_publish = self.last_publish_signature is None or signature != self.last_publish_signature or self.should_recenter(lat, lon, config)
      if should_publish:
        self.publish_state(
          available=True,
          precipitation_in_range=precipitation_in_range,
          stale=stale,
          config=config,
          precip_png=encode_png(precip_image),
          provider_status="ok" if not self.last_fetch_error else "fetchError",
          precip_tile_count=precip_tile_count,
        )
        self.last_publish_lat = lat
        self.last_publish_lon = lon
        self.last_publish_signature = signature

      rk.keep_time()


def main():
  WeatherOverlayDaemon().run()


if __name__ == "__main__":
  main()
