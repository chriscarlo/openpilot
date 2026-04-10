from io import BytesIO

from PIL import Image

from openpilot.sunnypilot.weather_overlayd.overlay_core import (
  CAR_ANCHOR_Y,
  CANVAS_SIZE_PX,
  OverlayConfig,
  build_viewport,
  compose_layer,
  encode_png,
  image_has_precipitation,
  iter_tile_requests,
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
