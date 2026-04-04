#!/usr/bin/env python3

import struct
import zlib

import pytest

from sunnypilot.weatherd import weatherd


def _encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
  """Encode a raw RGBA byte buffer to a PNG (for round-tripping through decoder)."""
  assert len(rgba) == width * height * 4
  ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
  raw = bytearray()
  stride = width * 4
  for y in range(height):
    raw.append(0)
    raw.extend(rgba[y * stride:(y + 1) * stride])
  idat = zlib.compress(bytes(raw), 9)

  def _chunk(t: bytes, d: bytes) -> bytes:
    crc = zlib.crc32(t + d)
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", crc)

  return (b"\x89PNG\r\n\x1a\n"
          + _chunk(b"IHDR", ihdr)
          + _chunk(b"IDAT", idat)
          + _chunk(b"IEND", b""))


class FakeResponse:
  def __init__(self, json_body=None, content=None):
    self._json = json_body
    self.content = content

  def json(self):
    return self._json

  def raise_for_status(self):
    pass


class RequestsStub:
  """Minimal stand-in for `requests` that routes URLs to configured responses."""

  def __init__(self):
    self.handlers: dict[str, object] = {}
    self.call_count = 0

  def register(self, url_substring: str, response_or_exc):
    self.handlers[url_substring] = response_or_exc

  def get(self, url: str, *args, **kwargs):
    self.call_count += 1
    for substr, resp in self.handlers.items():
      if substr in url:
        if isinstance(resp, Exception):
          raise resp
        return resp
    raise AssertionError(f"No handler registered for URL: {url}")


def test_decode_png_rgba_roundtrip_solid_color():
  yellow_px = bytes([255, 224, 0, 255])
  raw = yellow_px * 16
  png = _encode_png_rgba(4, 4, raw)
  result = weatherd._decode_png_rgba(png)
  assert result is not None
  w, h, pixels = result
  assert w == 4 and h == 4
  assert pixels == raw


def test_decode_png_rgba_rejects_garbage():
  assert weatherd._decode_png_rgba(b"") is None
  assert weatherd._decode_png_rgba(b"not a png") is None
  assert weatherd._decode_png_rgba(b"\x89PNG\r\n\x1a\n garbage after header") is None


def test_decode_png_rgba_handles_transparent_and_opaque_pixels():
  transparent = bytes([0, 0, 0, 0])
  yellow = bytes([255, 224, 0, 255])
  blue = bytes([0, 112, 163, 255])
  red = bytes([255, 68, 0, 255])
  raw = transparent + yellow + blue + red
  png = _encode_png_rgba(2, 2, raw)
  result = weatherd._decode_png_rgba(png)
  assert result is not None
  w, h, pixels = result
  assert (w, h) == (2, 2)
  assert pixels[0:4] == transparent
  assert pixels[4:8] == yellow
  assert pixels[8:12] == blue
  assert pixels[12:16] == red


def test_metadata_cache_stores_newest_past_frame(monkeypatch):
  stub = RequestsStub()
  stub.register("weather-maps.json", FakeResponse(json_body={
    "host": "https://tilecache.example.com",
    "radar": {
      "past": [
        {"time": 1000, "path": "/v2/radar/aaa"},
        {"time": 1600, "path": "/v2/radar/bbb"},
        {"time": 2200, "path": "/v2/radar/ccc"},
      ],
    },
  }))
  monkeypatch.setattr(weatherd, "requests", stub)

  cache = weatherd.MetadataCache()
  out = cache.get(now_monotonic=100.0)
  assert out == ("https://tilecache.example.com", "/v2/radar/ccc", 2200)
  assert stub.call_count == 1

  # Second call within cache window reuses cached data, no new HTTP
  out2 = cache.get(now_monotonic=100.0 + 30.0)
  assert out2 == out
  assert stub.call_count == 1

  # After cache expiry, refetches
  out3 = cache.get(now_monotonic=100.0 + weatherd.METADATA_CACHE_S + 1.0)
  assert out3 is not None
  assert stub.call_count == 2


def test_metadata_cache_returns_none_on_http_failure(monkeypatch):
  stub = RequestsStub()
  stub.register("weather-maps.json", Exception("boom"))
  monkeypatch.setattr(weatherd, "requests", stub)
  cache = weatherd.MetadataCache()
  assert cache.get(now_monotonic=100.0) is None


def test_metadata_cache_returns_none_when_past_is_empty(monkeypatch):
  stub = RequestsStub()
  stub.register("weather-maps.json", FakeResponse(json_body={
    "host": "h",
    "radar": {"past": []},
  }))
  monkeypatch.setattr(weatherd, "requests", stub)
  cache = weatherd.MetadataCache()
  assert cache.get(now_monotonic=100.0) is None


def test_fetch_tile_decodes_response(monkeypatch):
  size = 256
  transparent = bytes([0, 0, 0, 0])
  yellow = bytes([255, 224, 0, 255])
  raw = bytearray(transparent * (size * size))
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      idx = ((size // 2 + dy) * size + (size // 2 + dx)) * 4
      raw[idx:idx + 4] = yellow
  png = _encode_png_rgba(size, size, bytes(raw))

  stub = RequestsStub()
  stub.register(".png", FakeResponse(content=png))
  monkeypatch.setattr(weatherd, "requests", stub)
  result = weatherd._fetch_tile("host", "/path", 40.0, -95.0)
  assert result is not None
  w, h, _ = result
  assert (w, h) == (size, size)


def test_fetch_tile_returns_none_on_http_failure(monkeypatch):
  stub = RequestsStub()
  stub.register(".png", Exception("network down"))
  monkeypatch.setattr(weatherd, "requests", stub)
  assert weatherd._fetch_tile("host", "/path", 40.0, -95.0) is None


def _register_meta_and_tile(monkeypatch, tile_png: bytes):
  stub = RequestsStub()
  stub.register("weather-maps.json", FakeResponse(json_body={
    "host": "https://tile.example.com",
    "radar": {"past": [{"time": 1_700_000_000, "path": "/p"}]},
  }))
  stub.register(".png", FakeResponse(content=tile_png))
  monkeypatch.setattr(weatherd, "requests", stub)


def test_measure_precipitation_end_to_end_with_rainy_tile(monkeypatch):
  size = 256
  red = bytes([255, 68, 0, 255])
  transparent = bytes([0, 0, 0, 0])
  raw = bytearray(transparent * (size * size))
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      idx = ((size // 2 + dy) * size + (size // 2 + dx)) * 4
      raw[idx:idx + 4] = red
  _register_meta_and_tile(monkeypatch, _encode_png_rgba(size, size, bytes(raw)))

  metadata = weatherd.MetadataCache()
  out = weatherd._measure_precipitation(40.0, -95.0, metadata, now_monotonic=100.0)
  assert out is not None
  mm, frame_time = out
  assert frame_time == 1_700_000_000
  assert mm > 5.0


def test_measure_precipitation_returns_zero_for_transparent_tile(monkeypatch):
  size = 256
  transparent_tile = bytes([0, 0, 0, 0]) * (size * size)
  _register_meta_and_tile(monkeypatch, _encode_png_rgba(size, size, transparent_tile))
  metadata = weatherd.MetadataCache()
  out = weatherd._measure_precipitation(40.0, -95.0, metadata, now_monotonic=100.0)
  assert out is not None
  mm, _ = out
  assert mm == 0.0


def test_measure_precipitation_returns_none_when_tile_fetch_fails(monkeypatch):
  stub = RequestsStub()
  stub.register("weather-maps.json", FakeResponse(json_body={
    "host": "h", "radar": {"past": [{"time": 1, "path": "/p"}]},
  }))
  stub.register(".png", Exception("tile fetch failed"))
  monkeypatch.setattr(weatherd, "requests", stub)
  metadata = weatherd.MetadataCache()
  out = weatherd._measure_precipitation(40.0, -95.0, metadata, now_monotonic=100.0)
  assert out is None


def test_haversine_km_identity_zero():
  assert weatherd._haversine_km(40.0, -95.0, 40.0, -95.0) == pytest.approx(0.0)


def test_haversine_km_reference_distance():
  # NYC to LA is approximately 3936 km
  nyc = (40.7128, -74.0060)
  la = (34.0522, -118.2437)
  d = weatherd._haversine_km(nyc[0], nyc[1], la[0], la[1])
  assert 3900.0 < d < 4000.0
