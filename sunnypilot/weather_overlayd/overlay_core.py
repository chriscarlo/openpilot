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
MASK_FEATHER = 0.10
ALPHA_THRESHOLD = 8

# The mask circle must fit entirely inside the square canvas so that heading-up
# rotation in the HUD never exposes a flat canvas edge. The largest such circle
# is bounded by the shortest distance from the car anchor to any canvas edge.
# With anchor_y > 0.5 (car biased forward), the behind-the-car distance is the
# binding constraint, and this min ratio is what we scale the world viewport by.
CAR_MIN_ANCHOR_DISTANCE = min(CAR_ANCHOR_X, 1.0 - CAR_ANCHOR_X, CAR_ANCHOR_Y, 1.0 - CAR_ANCHOR_Y)


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
  # range_km is the desired inscribed-circle radius around the car (the visible
  # disc the user sees on the HUD). Inflate the square world window so that the
  # largest circle centered on the offset anchor fits inside it — i.e. the
  # shortest anchor-to-edge distance equals range_km in real-world terms.
  range_world_px = (range_km * 1000.0) / meters_per_world_px(lat, zoom)
  world_size_px = range_world_px / CAR_MIN_ANCHOR_DISTANCE

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


def build_mask(size_px: int = CANVAS_SIZE_PX) -> Image.Image:
  y_idx, x_idx = np.ogrid[:size_px, :size_px]
  center_x = CAR_ANCHOR_X * (size_px - 1)
  center_y = CAR_ANCHOR_Y * (size_px - 1)
  # Largest circle around the car anchor that fits entirely inside the canvas.
  # This must match build_viewport's inflation factor so the circle's radius in
  # canvas pixels corresponds to range_km in real-world km.
  radius = min(center_x, center_y, (size_px - 1) - center_x, (size_px - 1) - center_y)
  inner_radius = radius * (1.0 - MASK_FEATHER)
  distance = np.sqrt((x_idx - center_x) ** 2 + (y_idx - center_y) ** 2)
  ramp = np.clip((radius - distance) / max(radius - inner_radius, 1.0), 0.0, 1.0)
  smooth = ramp * ramp * (3.0 - 2.0 * ramp)
  return Image.fromarray((smooth * 255.0).astype(np.uint8), mode="L")


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

  alpha_src = np.asarray(composed.getchannel("A"), dtype=np.float32)
  mask = np.asarray(build_mask(canvas_size_px), dtype=np.float32) / 255.0
  alpha = Image.fromarray(np.clip(alpha_src * mask, 0.0, 255.0).astype(np.uint8), mode="L")
  composed.putalpha(alpha)
  return composed, loaded_tiles


def image_has_precipitation(image: Image.Image) -> bool:
  return bool(np.any(np.asarray(image.getchannel("A"), dtype=np.uint8) > ALPHA_THRESHOLD))


def encode_png(image: Image.Image) -> bytes:
  buffer = BytesIO()
  image.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()
