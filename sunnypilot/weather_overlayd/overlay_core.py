from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math

import numpy as np
from PIL import Image

RESAMPLING_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

TILE_SIZE = 256
MERCATOR_MAX_LAT = 85.05112878
EARTH_CIRCUMFERENCE_M = 40075016.68557849
CANVAS_SIZE_PX = 1024
CAR_ANCHOR_X = 0.50
CAR_ANCHOR_Y = 0.68
ALPHA_THRESHOLD = 8

# The HUD draws the overlay into a square centered on the car anchor so the map
# can rotate heading-up without exposing camera corners. Model that square here
# using the tici's landscape aspect ratio, then size the world viewport so the
# visible behind-the-car distance is the configured range times the requested
# zoom-out factor.
HUD_TARGET_ASPECT_RATIO = 2.0
HUD_ZOOM_OUT_FACTOR = 1.5

_ANCHOR_X_ASPECT = CAR_ANCHOR_X * HUD_TARGET_ASPECT_RATIO
HUD_ROTATION_COVER_RADIUS = max(
  math.hypot(_ANCHOR_X_ASPECT, CAR_ANCHOR_Y),
  math.hypot(HUD_TARGET_ASPECT_RATIO - _ANCHOR_X_ASPECT, CAR_ANCHOR_Y),
  math.hypot(_ANCHOR_X_ASPECT, 1.0 - CAR_ANCHOR_Y),
  math.hypot(HUD_TARGET_ASPECT_RATIO - _ANCHOR_X_ASPECT, 1.0 - CAR_ANCHOR_Y),
)
HUD_ROTATION_COVER_SIDE = HUD_ROTATION_COVER_RADIUS * 2.0
HUD_VISIBLE_BEHIND_FRACTION = (1.0 - CAR_ANCHOR_Y) / HUD_ROTATION_COVER_SIDE


@dataclass(frozen=True)
class OverlayConfig:
  range_km: int
  zoom: int
  refresh_seconds: int


@dataclass(frozen=True)
class Viewport:
  logical_tile_x_min: int
  logical_tile_x_max: int
  tile_y_min: int
  tile_y_max: int
  src_box: tuple[float, float, float, float]
  world_size_px: float


@dataclass(frozen=True)
class TileRequest:
  zoom: int
  logical_x: int
  tile_x: int
  tile_y: int


def clamp_latitude(lat: float) -> float:
  return max(-MERCATOR_MAX_LAT, min(MERCATOR_MAX_LAT, lat))


def latlon_to_world_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
  lat = clamp_latitude(lat)
  world_scale = TILE_SIZE * (2 ** zoom)
  x = (lon + 180.0) / 360.0 * world_scale
  sin_lat = math.sin(math.radians(lat))
  y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * world_scale
  return x, y


def meters_per_world_px(lat: float, zoom: int) -> float:
  return math.cos(math.radians(clamp_latitude(lat))) * EARTH_CIRCUMFERENCE_M / (TILE_SIZE * (2 ** zoom))


def build_viewport(lat: float, lon: float, range_km: int, zoom: int) -> Viewport:
  world_x, world_y = latlon_to_world_px(lat, lon, zoom)
  range_world_px = (range_km * 1000.0) / meters_per_world_px(lat, zoom)
  # Fit the map to the full tici HUD instead of the old inscribed-circle mask.
  # The configured range now maps to the visible behind-the-car distance, then
  # expands by HUD_ZOOM_OUT_FACTOR to show roughly 50% more area.
  world_size_px = (range_world_px * HUD_ZOOM_OUT_FACTOR) / HUD_VISIBLE_BEHIND_FRACTION

  left = world_x - (CAR_ANCHOR_X * world_size_px)
  top = world_y - (CAR_ANCHOR_Y * world_size_px)
  right = left + world_size_px
  bottom = top + world_size_px

  logical_tile_x_min = math.floor(left / TILE_SIZE)
  logical_tile_x_max = math.floor((right - 1.0) / TILE_SIZE)
  tile_y_min = max(0, math.floor(top / TILE_SIZE))
  tile_y_max = min((2 ** zoom) - 1, math.floor((bottom - 1.0) / TILE_SIZE))

  src_left = left - (logical_tile_x_min * TILE_SIZE)
  src_top = top - (tile_y_min * TILE_SIZE)
  src_right = src_left + world_size_px
  src_bottom = src_top + world_size_px

  return Viewport(
    logical_tile_x_min=logical_tile_x_min,
    logical_tile_x_max=logical_tile_x_max,
    tile_y_min=tile_y_min,
    tile_y_max=tile_y_max,
    src_box=(src_left, src_top, src_right, src_bottom),
    world_size_px=world_size_px,
  )


def iter_tile_requests(zoom: int, viewport: Viewport) -> list[TileRequest]:
  tile_mod = 2 ** zoom
  return [
    TileRequest(
      zoom=zoom,
      logical_x=logical_x,
      tile_x=logical_x % tile_mod,
      tile_y=tile_y,
    )
    for tile_y in range(viewport.tile_y_min, viewport.tile_y_max + 1)
    for logical_x in range(viewport.logical_tile_x_min, viewport.logical_tile_x_max + 1)
  ]


def compose_layer(tile_lookup, viewport: Viewport, canvas_size_px: int = CANVAS_SIZE_PX) -> tuple[Image.Image, int]:
  atlas_width_tiles = viewport.logical_tile_x_max - viewport.logical_tile_x_min + 1
  atlas_height_tiles = viewport.tile_y_max - viewport.tile_y_min + 1
  atlas = Image.new("RGBA", (atlas_width_tiles * TILE_SIZE, atlas_height_tiles * TILE_SIZE), (0, 0, 0, 0))

  loaded_tiles = 0
  for request in tile_lookup.requests:
    tile_image = tile_lookup.get_tile(request)
    if tile_image is None:
      continue
    paste_x = (request.logical_x - viewport.logical_tile_x_min) * TILE_SIZE
    paste_y = (request.tile_y - viewport.tile_y_min) * TILE_SIZE
    atlas.alpha_composite(tile_image, (paste_x, paste_y))
    loaded_tiles += 1

  composed = atlas.transform(
    (canvas_size_px, canvas_size_px),
    Image.Transform.EXTENT,
    viewport.src_box,
    resample=RESAMPLING_BILINEAR,
  )
  return composed, loaded_tiles


def image_has_precipitation(image: Image.Image) -> bool:
  return bool(np.any(np.asarray(image.getchannel("A"), dtype=np.uint8) > ALPHA_THRESHOLD))


def encode_png(image: Image.Image) -> bytes:
  buffer = BytesIO()
  image.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()
