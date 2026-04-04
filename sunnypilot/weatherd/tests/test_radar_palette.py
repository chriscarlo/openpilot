#!/usr/bin/env python3

import pytest

from sunnypilot.weatherd.radar_palette import (
  MIN_ALPHA,
  _dbz_to_mm_per_hr,
  rgba_to_mm_per_hr,
  sample_intensity_max,
  severity_from_mm_per_hr,
)


def test_dbz_to_mm_per_hr_marshall_palmer_reference_values():
  # Marshall-Palmer reference: R(35 dBZ) ≈ 5.62 mm/hr, R(40) ≈ 11.53, R(20) ≈ 0.65
  assert _dbz_to_mm_per_hr(20) == pytest.approx(0.65, rel=0.02)
  assert _dbz_to_mm_per_hr(35) == pytest.approx(5.62, rel=0.02)
  assert _dbz_to_mm_per_hr(40) == pytest.approx(11.53, rel=0.02)
  assert _dbz_to_mm_per_hr(0) == 0.0
  assert _dbz_to_mm_per_hr(-5) == 0.0


def test_transparent_pixel_is_zero_regardless_of_rgb():
  assert rgba_to_mm_per_hr(0, 0, 0, 0) == 0.0
  assert rgba_to_mm_per_hr(255, 255, 255, 0) == 0.0
  # Low alpha is rendering anti-aliasing, not precipitation
  assert rgba_to_mm_per_hr(0, 163, 224, MIN_ALPHA - 1) == 0.0


def test_cyan_pixels_map_to_light_precipitation():
  # Observed palette: light cyan through dark cyan span 5-20 dBZ
  light_cyan = rgba_to_mm_per_hr(136, 221, 238, 255)
  dark_cyan = rgba_to_mm_per_hr(27, 174, 226, 255)
  # Light cyan is drizzle-level; dark cyan is light rain
  assert 0.05 <= light_cyan <= 0.3, f"light cyan: {light_cyan}"
  assert 0.3 <= dark_cyan <= 1.0, f"dark cyan: {dark_cyan}"
  # Within cyan family, darker cyan (lower R) = higher precip
  assert dark_cyan > light_cyan


def test_blue_pixels_span_light_to_moderate_precipitation():
  darkest_blue = rgba_to_mm_per_hr(0, 71, 104, 255)
  brightest_blue = rgba_to_mm_per_hr(0, 163, 224, 255)
  # Blue family spans ~20-35 dBZ = ~0.65-5.6 mm/hr
  assert 0.5 <= brightest_blue <= 1.5, f"bright blue: {brightest_blue}"
  assert 3.0 <= darkest_blue <= 7.0, f"dark blue: {darkest_blue}"
  # Darker blue = more intense precipitation in Universal Blue
  assert darkest_blue > brightest_blue


def test_yellow_pixels_map_to_heavy_precipitation():
  yellow = rgba_to_mm_per_hr(255, 224, 0, 255)
  mm = yellow
  # Yellow ≈ 37-39 dBZ, Marshall-Palmer ~8-10 mm/hr
  assert 5.0 <= mm <= 15.0, f"yellow: {mm}"


def test_orange_pixels_map_higher_than_yellow():
  yellow = rgba_to_mm_per_hr(255, 224, 0, 255)
  orange = rgba_to_mm_per_hr(255, 129, 0, 255)
  assert orange > yellow


def test_red_pixels_map_higher_than_orange():
  orange = rgba_to_mm_per_hr(255, 129, 0, 255)
  red = rgba_to_mm_per_hr(255, 68, 0, 255)
  assert red > orange


def test_dark_red_maps_higher_than_bright_red():
  red = rgba_to_mm_per_hr(255, 68, 0, 255)
  dark_red = rgba_to_mm_per_hr(93, 0, 0, 255)
  assert dark_red > red


def test_purple_white_maps_to_extreme():
  purple = rgba_to_mm_per_hr(255, 108, 255, 255)
  white = rgba_to_mm_per_hr(255, 255, 255, 255)
  # Both are extreme / wrap-around
  assert purple > 50.0
  assert white > 50.0


def test_monotonic_intensity_across_representative_palette():
  # A monotonic walk through the palette: cyan → blue → yellow → orange → red → dark red
  sequence = [
    (136, 221, 238, 255),   # light cyan
    (27, 174, 226, 255),    # dark cyan
    (0, 163, 224, 255),     # light blue
    (0, 71, 104, 255),      # dark blue
    (255, 238, 0, 255),     # yellow
    (255, 129, 0, 255),     # orange
    (255, 68, 0, 255),      # red
    (93, 0, 0, 255),        # dark red
  ]
  mm_values = [rgba_to_mm_per_hr(*rgba) for rgba in sequence]
  for i in range(1, len(mm_values)):
    msg = f"non-monotonic at {i}: {sequence[i-1]} → {mm_values[i-1]:.2f}, {sequence[i]} → {mm_values[i]:.2f}"
    assert mm_values[i] > mm_values[i - 1], msg


def _make_tile(colors: list[tuple[int, int, int, int]], width: int, height: int) -> bytes:
  """Build a flat RGBA byte array from a list of pixels (row-major)."""
  assert len(colors) == width * height
  out = bytearray()
  for r, g, b, a in colors:
    out.extend((r, g, b, a))
  return bytes(out)


def test_sample_window_returns_max_over_neighborhood():
  # 5x5 tile: heavy rain at center, clear edges
  transparent = (0, 0, 0, 0)
  yellow = (255, 224, 0, 255)
  pixels = [transparent] * 25
  pixels[12] = yellow   # center of 5x5
  data = _make_tile(pixels, 5, 5)
  mm = sample_intensity_max(data, 5, 5, center_x=2, center_y=2, window=3)
  assert mm == rgba_to_mm_per_hr(*yellow)
  assert mm > 5.0


def test_sample_window_finds_neighbor_pixel():
  # Put rain at (1, 1), sample with center at (2, 2), window=3 → should still find it
  transparent = (0, 0, 0, 0)
  red = (255, 68, 0, 255)
  pixels = [transparent] * 25
  pixels[1 * 5 + 1] = red
  data = _make_tile(pixels, 5, 5)
  mm = sample_intensity_max(data, 5, 5, center_x=2, center_y=2, window=3)
  assert mm == rgba_to_mm_per_hr(*red)


def test_sample_window_all_transparent_returns_zero():
  pixels = [(0, 0, 0, 0)] * 25
  data = _make_tile(pixels, 5, 5)
  mm = sample_intensity_max(data, 5, 5, center_x=2, center_y=2, window=3)
  assert mm == 0.0


def test_sample_window_clamps_to_image_bounds():
  # center near edge; should not IndexError
  pixels = [(0, 163, 224, 255)] * 25
  data = _make_tile(pixels, 5, 5)
  mm = sample_intensity_max(data, 5, 5, center_x=0, center_y=0, window=3)
  assert mm > 0.0


def test_severity_from_mm_per_hr_matches_controller_anchors():
  # Controller PRECIP anchors: LIGHT=0.5, MODERATE=2.5, HEAVY=7.5 mm/hr
  assert severity_from_mm_per_hr(0.0) == "none"
  assert severity_from_mm_per_hr(0.3) == "none"
  assert severity_from_mm_per_hr(0.5) == "light"
  assert severity_from_mm_per_hr(1.5) == "light"
  assert severity_from_mm_per_hr(2.5) == "moderate"
  assert severity_from_mm_per_hr(5.0) == "moderate"
  assert severity_from_mm_per_hr(7.5) == "heavy"
  assert severity_from_mm_per_hr(20.0) == "heavy"
