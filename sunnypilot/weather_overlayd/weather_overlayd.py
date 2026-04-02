#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import requests
from PIL import Image
from io import BytesIO

from cereal import messaging
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

from .api_key_manager import get_api_key
from .overlay_core import (
  CANVAS_SIZE_PX,
  CAR_ANCHOR_X,
  CAR_ANCHOR_Y,
  OverlayConfig,
  RAIN_LAYER,
  RAIN_PALETTE,
  SNOW_LAYER,
  SNOW_PALETTE,
  Viewport,
  build_viewport,
  compose_layer,
  encode_png,
  image_has_precipitation,
  iter_tile_requests,
)

OPENWEATHER_TILE_URL = "https://maps.openweathermap.org/maps/2.0/weather/{layer}/{z}/{x}/{y}"
LOOP_HZ = 0.5
API_TIMEOUT_S = 10
REFRESH_JITTER_S = 15
RECENTER_MIN_DISTANCE_M = 100.0


@dataclass
class CachedTile:
  image: Image.Image
  fetched_monotonic: float


class TileLookup:
  def __init__(self, daemon: "WeatherOverlayDaemon", requests_):
    self.daemon = daemon
    self.requests = requests_

  def get_tile(self, request):
    return self.daemon.get_cached_tile(request)


class WeatherOverlayDaemon:
  def __init__(self):
    self.params = Params()
    self.api_key = None
    self.session = requests.Session()
    self.pm = messaging.PubMaster(["weatherOverlaySP"])

    self.gps_service = get_gps_location_service(self.params)
    self.sm = messaging.SubMaster([self.gps_service])

    self.cache: dict[tuple[str, int, int, int], CachedTile] = {}
    self.last_config: OverlayConfig | None = None
    self.last_publish_lat = math.nan
    self.last_publish_lon = math.nan
    self.last_publish_signature: tuple | None = None
    self.last_api_key_check_monotonic = 0.0
    self.last_fetch_error = False

  def load_api_key(self, force: bool = False) -> str | None:
    now = time.monotonic()
    if not force and self.api_key is not None and (now - self.last_api_key_check_monotonic) < 30.0:
      return self.api_key

    self.last_api_key_check_monotonic = now
    self.api_key = get_api_key()
    return self.api_key

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
      range_km=clamp_int("WeatherOverlayRangeKm", 12, 3, 40),
      zoom=clamp_int("WeatherOverlayZoomLevel", 10, 6, 13),
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

  def cache_key(self, layer: str, zoom: int, tile_x: int, tile_y: int) -> tuple[str, int, int, int]:
    return layer, zoom, tile_x, tile_y

  def build_tile_params(self, palette: str) -> dict[str, str]:
    return {
      "appid": self.api_key or "",
      "fill_bound": "true",
      "opacity": "1.0",
      "palette": palette,
    }

  def fetch_tile(self, layer: str, zoom: int, tile_x: int, tile_y: int, palette: str) -> Image.Image | None:
    if not self.api_key:
      return None

    url = OPENWEATHER_TILE_URL.format(layer=layer, z=zoom, x=tile_x, y=tile_y)
    try:
      resp = self.session.get(url, params=self.build_tile_params(palette), timeout=API_TIMEOUT_S)
      resp.raise_for_status()
      image = Image.open(BytesIO(resp.content)).convert("RGBA")
      image.load()
      self.last_fetch_error = False
      return image
    except Exception as e:
      self.last_fetch_error = True
      cloudlog.warning(f"weather_overlayd: tile fetch failed for {layer}/{zoom}/{tile_x}/{tile_y}: {e}")
      return None

  def ensure_tiles(self, layer: str, palette: str, config: OverlayConfig, viewport: Viewport, now_monotonic: float) -> list:
    tile_requests = iter_tile_requests(layer, config.zoom, viewport)
    for request in tile_requests:
      key = self.cache_key(layer, config.zoom, request.tile_x, request.tile_y)
      cached = self.cache.get(key)
      tile_expired = cached is None or (now_monotonic - cached.fetched_monotonic) >= config.refresh_seconds
      if tile_expired:
        image = self.fetch_tile(layer, config.zoom, request.tile_x, request.tile_y, palette)
        if image is not None:
          self.cache[key] = CachedTile(image=image, fetched_monotonic=now_monotonic)
    return tile_requests

  def get_cached_tile(self, request) -> Image.Image | None:
    cached = self.cache.get(self.cache_key(request.layer, request.zoom, request.tile_x, request.tile_y))
    return None if cached is None else cached.image

  def cache_age_seconds(self, requests_) -> float:
    ages = []
    now = time.monotonic()
    for request in requests_:
      cached = self.cache.get(self.cache_key(request.layer, request.zoom, request.tile_x, request.tile_y))
      if cached is not None:
        ages.append(now - cached.fetched_monotonic)
    return max(ages) if ages else float("inf")

  def publish_state(self, *, available: bool, precipitation_in_range: bool, stale: bool,
                    config: OverlayConfig, rain_png: bytes = b"", snow_png: bytes = b"",
                    provider_status: str = "offline", rain_tile_count: int = 0, snow_tile_count: int = 0) -> None:
    msg = messaging.new_message("weatherOverlaySP", valid=available or provider_status != "ok")
    overlay = msg.weatherOverlaySP
    overlay.available = available
    overlay.precipitationInRange = precipitation_in_range
    overlay.stale = stale
    overlay.zoom = config.zoom
    overlay.rangeKm = float(config.range_km)
    overlay.carAnchorX = CAR_ANCHOR_X
    overlay.carAnchorY = CAR_ANCHOR_Y
    overlay.rainLayerPng = rain_png
    overlay.snowLayerPng = snow_png
    overlay.generatedAtUnixSec = int(time.time())
    overlay.providerStatus = provider_status
    overlay.rainTileCount = rain_tile_count
    overlay.snowTileCount = snow_tile_count
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
    cloudlog.info("weather_overlayd: started")

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

      if not self.load_api_key():
        self.maybe_publish_unavailable("missingKey", config)
        rk.keep_time()
        continue

      lat, lon = location
      viewport = build_viewport(lat, lon, config.range_km, config.zoom)
      now_monotonic = time.monotonic()

      rain_requests = self.ensure_tiles(RAIN_LAYER, RAIN_PALETTE, config, viewport, now_monotonic)
      snow_requests = self.ensure_tiles(SNOW_LAYER, SNOW_PALETTE, config, viewport, now_monotonic)

      rain_lookup = TileLookup(self, rain_requests)
      snow_lookup = TileLookup(self, snow_requests)
      rain_image, rain_tile_count = compose_layer(rain_lookup, viewport, CANVAS_SIZE_PX)
      snow_image, snow_tile_count = compose_layer(snow_lookup, viewport, CANVAS_SIZE_PX)

      available = (rain_tile_count + snow_tile_count) > 0
      if not available:
        self.maybe_publish_unavailable("fetchError" if self.last_fetch_error else "offline", config)
        rk.keep_time()
        continue

      precipitation_in_range = image_has_precipitation(rain_image) or image_has_precipitation(snow_image)
      tile_age_s = max(self.cache_age_seconds(rain_requests), self.cache_age_seconds(snow_requests))
      stale = tile_age_s > (config.refresh_seconds + REFRESH_JITTER_S)
      signature = (
        config.range_km,
        config.zoom,
        config.refresh_seconds,
        precipitation_in_range,
        stale,
        rain_tile_count,
        snow_tile_count,
        "ok" if not self.last_fetch_error else "fetchError",
      )

      should_publish = self.last_publish_signature is None or signature != self.last_publish_signature or self.should_recenter(lat, lon, config)
      if should_publish:
        self.publish_state(
          available=True,
          precipitation_in_range=precipitation_in_range,
          stale=stale,
          config=config,
          rain_png=encode_png(rain_image),
          snow_png=encode_png(snow_image),
          provider_status="ok" if not self.last_fetch_error else "fetchError",
          rain_tile_count=rain_tile_count,
          snow_tile_count=snow_tile_count,
        )
        self.last_publish_lat = lat
        self.last_publish_lon = lon
        self.last_publish_signature = signature

      rk.keep_time()


def main():
  WeatherOverlayDaemon().run()


if __name__ == "__main__":
  main()
