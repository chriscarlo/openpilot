from io import BytesIO

import pytest
from PIL import Image

from openpilot.sunnypilot.weather_overlayd.overlay_core import (
  CAR_ANCHOR_X,
  CAR_ANCHOR_Y,
  CANVAS_SIZE_PX,
  HUD_VISIBLE_BEHIND_FRACTION,
  HUD_ZOOM_OUT_FACTOR,
  build_viewport,
  compose_layer,
  encode_png,
  image_has_precipitation,
  iter_tile_requests,
  meters_per_world_px,
)


class StaticLookup:
  def __init__(self, image):
    self.image = image
    self.requests = []

  def get_tile(self, request):
    return self.image


def test_viewport_anchor_bias_shows_more_ahead_than_behind():
  viewport = build_viewport(37.7749, -122.4194, 12, 7)
  assert viewport.world_size_px > 0.0
  ahead_px = CAR_ANCHOR_Y * viewport.world_size_px
  behind_px = (1.0 - CAR_ANCHOR_Y) * viewport.world_size_px
  assert ahead_px > behind_px


def test_compose_layer_masks_to_dry_when_tiles_are_empty():
  transparent = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
  viewport = build_viewport(37.7749, -122.4194, 12, 7)
  lookup = StaticLookup(transparent)
  lookup.requests = iter_tile_requests(7, viewport)
  image, tile_count = compose_layer(lookup, viewport, CANVAS_SIZE_PX)

  assert tile_count == len(lookup.requests)
  assert not image_has_precipitation(image)


def test_compose_layer_preserves_precipitation_inside_range():
  wet = Image.new("RGBA", (256, 256), (120, 220, 90, 255))
  viewport = build_viewport(37.7749, -122.4194, 12, 7)
  lookup = StaticLookup(wet)
  lookup.requests = iter_tile_requests(7, viewport)
  image, tile_count = compose_layer(lookup, viewport, CANVAS_SIZE_PX)
  png_bytes = encode_png(image)

  assert tile_count == len(lookup.requests)
  assert image_has_precipitation(image)
  assert image.getchannel("A").getpixel((0, 0)) == 255
  reloaded = Image.open(BytesIO(png_bytes)).convert("RGBA")
  assert reloaded.size == (CANVAS_SIZE_PX, CANVAS_SIZE_PX)


def test_build_viewport_visible_behind_distance_matches_zoomed_range():
  lat, lon, range_km, zoom = 37.7749, -122.4194, 16, 7
  viewport = build_viewport(lat, lon, range_km, zoom)
  mpp = meters_per_world_px(lat, zoom)
  visible_behind_km = viewport.world_size_px * HUD_VISIBLE_BEHIND_FRACTION * mpp / 1000.0
  assert visible_behind_km == pytest.approx(range_km * HUD_ZOOM_OUT_FACTOR, rel=1e-9)


def test_build_viewport_fullscreen_cover_preserves_anchor_bias():
  viewport = build_viewport(37.7749, -122.4194, 12, 7)
  ahead_px = CAR_ANCHOR_Y * viewport.world_size_px
  behind_px = (1.0 - CAR_ANCHOR_Y) * viewport.world_size_px
  assert ahead_px > behind_px
