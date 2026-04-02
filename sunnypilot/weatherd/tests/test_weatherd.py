#!/usr/bin/env python3

import pytest

from sunnypilot.weatherd.weatherd import _classify_weather


@pytest.mark.parametrize("component_key", ["rain", "showers"])
def test_classify_weather_falls_back_to_component_precipitation_when_total_is_zero(component_key):
  current = {
    "precipitation": 0.0,
    "rain": 0.0,
    "showers": 0.0,
    "snowfall": 0.0,
    "weather_code": 82,
  }
  current[component_key] = 7.5

  result = _classify_weather({
    "current": current,
  })

  assert result["severity"] == "heavy"
  assert result[f"{component_key}_mm"] == pytest.approx(7.5)
  assert result["precipitation_mm"] == pytest.approx(7.5)
