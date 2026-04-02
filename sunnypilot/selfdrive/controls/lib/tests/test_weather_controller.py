#!/usr/bin/env python3

import json

import pytest

from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from sunnypilot.selfdrive.controls.lib import weather_controller as weather_mod


class FakeParams:
  def __init__(self):
    self._bools = {
      "WeatherAwareControlEnabled": True,
    }
    self._vals = {
      "WeatherSpeedReductionLight": 5,
      "WeatherSpeedReductionModerate": 10,
      "WeatherSpeedReductionHeavy": 15,
    }

  def get_bool(self, key):
    return bool(self._bools.get(key, False))

  def get(self, key):
    return self._vals.get(key)

  def put_weather(self, payload):
    self._vals["WeatherCondition"] = json.dumps(payload)


@pytest.mark.parametrize("component_key", ["rain_mm", "showers_mm"])
def test_weather_controller_keeps_slowdown_when_component_precipitation_is_present(monkeypatch, component_key):
  fake_params = FakeParams()
  controller = weather_mod.WeatherController()
  controller.params = fake_params

  fake_now = 1234.0
  payload = {
    "severity": "heavy",
    "precipitation_mm": 0.0,
    "rain_mm": 0.0,
    "showers_mm": 0.0,
    "snowfall_mm": 0.0,
    "timestamp": fake_now,
  }
  payload[component_key] = 7.5
  fake_params.put_weather({
    **payload,
  })

  monotonic_values = iter([100.0, 101.0])
  monkeypatch.setattr(weather_mod.time, "monotonic", lambda: next(monotonic_values))
  monkeypatch.setattr(weather_mod.time, "time", lambda: fake_now)

  v_cruise = 30.0
  controller.update(v_ego=v_cruise, v_cruise=v_cruise)
  controller.update(v_ego=v_cruise, v_cruise=v_cruise)

  assert controller.is_active is True
  assert controller.severity == "heavy"
  assert controller.speed_recommendation != V_CRUISE_UNSET
  assert controller.speed_recommendation < v_cruise
