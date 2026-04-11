from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from openpilot.sunnypilot.weather_overlayd.overlay_core import (
  CAR_ANCHOR_X,
  CAR_ANCHOR_Y,
  CAR_MIN_ANCHOR_DISTANCE,
  CANVAS_SIZE_PX,
  OverlayConfig,
  build_mask,
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
  assert image.getchannel("A").getpixel((0, 0)) == 0
  reloaded = Image.open(BytesIO(png_bytes)).convert("RGBA")
  assert reloaded.size == (CANVAS_SIZE_PX, CANVAS_SIZE_PX)


def test_build_mask_has_zero_alpha_on_all_canvas_borders():
  # The mask circle must fit entirely inside the canvas so heading-up rotation
  # in the HUD never exposes a flat canvas edge. Every pixel on every border
  # should be fully transparent after the fit-to-canvas shrink.
  mask = np.asarray(build_mask(CANVAS_SIZE_PX), dtype=np.uint8)
  assert (mask[0, :] == 0).all(), "top border has opaque pixels"
  assert (mask[-1, :] == 0).all(), "bottom border has opaque pixels"
  assert (mask[:, 0] == 0).all(), "left border has opaque pixels"
  assert (mask[:, -1] == 0).all(), "right border has opaque pixels"


def test_build_mask_is_opaque_at_car_anchor_center():
  mask = np.asarray(build_mask(CANVAS_SIZE_PX), dtype=np.uint8)
  cx = int(round(CAR_ANCHOR_X * (CANVAS_SIZE_PX - 1)))
  cy = int(round(CAR_ANCHOR_Y * (CANVAS_SIZE_PX - 1)))
  assert mask[cy, cx] == 255, "mask should be fully opaque at the car anchor"


def test_build_viewport_inscribed_circle_equals_range_km():
  # With the fit-to-canvas mask, the visible circle radius in canvas px is
  # CAR_MIN_ANCHOR_DISTANCE * CANVAS_SIZE_PX. Converting that back to km via
  # world_size_px should yield the user-requested range_km.
  lat, lon, range_km, zoom = 37.7749, -122.4194, 16, 7
  viewport = build_viewport(lat, lon, range_km, zoom)
  mpp = meters_per_world_px(lat, zoom)
  # world_size_px is in zoom-world-pixels; convert to km
  world_size_km = viewport.world_size_px * mpp / 1000.0
  inscribed_radius_km = world_size_km * CAR_MIN_ANCHOR_DISTANCE
  assert inscribed_radius_km == pytest.approx(range_km, rel=1e-9)


def test_build_viewport_inflation_preserves_anchor_bias():
  # Even after inflating the world window, the car anchor stays forward-biased:
  # the distance ahead of the car must still exceed the distance behind.
  viewport = build_viewport(37.7749, -122.4194, 12, 7)
  ahead_px = CAR_ANCHOR_Y * viewport.world_size_px
  behind_px = (1.0 - CAR_ANCHOR_Y) * viewport.world_size_px
  assert ahead_px > behind_px
