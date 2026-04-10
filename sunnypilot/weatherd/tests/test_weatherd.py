#!/usr/bin/env python3

import pytest

from sunnypilot.weatherd import weatherd


class FakeResponse:
  def __init__(self, json_body=None):
    self._json = json_body

  def json(self):
    return self._json

  def raise_for_status(self):
    pass


class RequestsStub:
  """Minimal stand-in for `requests` that routes URLs to configured responses."""

  def __init__(self):
    self.handlers: dict[str, object] = {}
    self.calls: list[tuple[str, dict]] = []

  def register(self, url_substring: str, response_or_exc):
    self.handlers[url_substring] = response_or_exc

  def get(self, url: str, params=None, *args, **kwargs):
    self.calls.append((url, params or {}))
    for substr, resp in self.handlers.items():
      if substr in url:
        if isinstance(resp, Exception):
          raise resp
        return resp
    raise AssertionError(f"No handler registered for URL: {url}")


def test_severity_from_mm_per_hr_matches_controller_anchors():
  # WeatherController PRECIP anchors: LIGHT=0.5, MODERATE=2.5, HEAVY=7.5 mm/hr.
  assert weatherd.severity_from_mm_per_hr(0.0) == "none"
  assert weatherd.severity_from_mm_per_hr(0.3) == "none"
  assert weatherd.severity_from_mm_per_hr(0.5) == "light"
  assert weatherd.severity_from_mm_per_hr(1.5) == "light"
  assert weatherd.severity_from_mm_per_hr(2.5) == "moderate"
  assert weatherd.severity_from_mm_per_hr(5.0) == "moderate"
  assert weatherd.severity_from_mm_per_hr(7.5) == "heavy"
  assert weatherd.severity_from_mm_per_hr(20.0) == "heavy"


def test_haversine_km_identity_zero():
  assert weatherd._haversine_km(40.0, -95.0, 40.0, -95.0) == pytest.approx(0.0)


def test_haversine_km_reference_distance():
  # NYC to LA is approximately 3936 km
  nyc = (40.7128, -74.0060)
  la = (34.0522, -118.2437)
  d = weatherd._haversine_km(nyc[0], nyc[1], la[0], la[1])
  assert 3900.0 < d < 4000.0


def test_fetch_point_conditions_parses_currently_block(monkeypatch):
  stub = RequestsStub()
  stub.register("api.pirateweather.net/forecast", FakeResponse(json_body={
    "currently": {
      "time": 1_700_000_000,
      "precipIntensity": 3.5,
      "precipType": "rain",
      "precipProbability": 0.9,
    },
  }))
  monkeypatch.setattr(weatherd, "requests", stub)
  out = weatherd.fetch_point_conditions("KEY", 40.0, -95.0)
  assert out == (3.5, 1_700_000_000)

  # Confirm we requested units=si so precipIntensity is mm/hr.
  assert stub.calls[0][1] == {"units": weatherd.PIRATE_WEATHER_UNITS}
  assert "KEY/40.0,-95.0" in stub.calls[0][0]


def test_fetch_point_conditions_returns_none_on_http_failure(monkeypatch):
  stub = RequestsStub()
  stub.register("api.pirateweather.net", Exception("network down"))
  monkeypatch.setattr(weatherd, "requests", stub)
  assert weatherd.fetch_point_conditions("KEY", 40.0, -95.0) is None


def test_fetch_point_conditions_clamps_negative_precip(monkeypatch):
  # Defensive: a nonsensical negative intensity should not become a negative mm/hr.
  stub = RequestsStub()
  stub.register("api.pirateweather.net", FakeResponse(json_body={
    "currently": {"time": 1_700_000_000, "precipIntensity": -0.2},
  }))
  monkeypatch.setattr(weatherd, "requests", stub)
  out = weatherd.fetch_point_conditions("KEY", 40.0, -95.0)
  assert out is not None
  mm, _ = out
  assert mm == 0.0


def test_fetch_point_conditions_treats_null_precip_as_zero(monkeypatch):
  stub = RequestsStub()
  stub.register("api.pirateweather.net", FakeResponse(json_body={
    "currently": {"time": 1_700_000_000, "precipIntensity": None},
  }))
  monkeypatch.setattr(weatherd, "requests", stub)
  out = weatherd.fetch_point_conditions("KEY", 40.0, -95.0)
  assert out is not None
  mm, ts = out
  assert mm == 0.0
  assert ts == 1_700_000_000


def test_fetch_point_conditions_rejects_response_without_time(monkeypatch):
  stub = RequestsStub()
  stub.register("api.pirateweather.net", FakeResponse(json_body={
    "currently": {"precipIntensity": 2.0},
  }))
  monkeypatch.setattr(weatherd, "requests", stub)
  assert weatherd.fetch_point_conditions("KEY", 40.0, -95.0) is None


def test_fetch_point_conditions_rejects_missing_currently_block(monkeypatch):
  stub = RequestsStub()
  stub.register("api.pirateweather.net", FakeResponse(json_body={"minutely": {}}))
  monkeypatch.setattr(weatherd, "requests", stub)
  assert weatherd.fetch_point_conditions("KEY", 40.0, -95.0) is None


def test_fetch_point_conditions_rejects_non_numeric_precip(monkeypatch):
  stub = RequestsStub()
  stub.register("api.pirateweather.net", FakeResponse(json_body={
    "currently": {"time": 1_700_000_000, "precipIntensity": "wet"},
  }))
  monkeypatch.setattr(weatherd, "requests", stub)
  assert weatherd.fetch_point_conditions("KEY", 40.0, -95.0) is None
