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


def _make_controller(monkeypatch, payload, monotonic_values=None):
  fake_params = FakeParams()
  controller = weather_mod.WeatherController()
  controller.params = fake_params
  fake_now = 1234.0
  fake_params.put_weather({**payload, "timestamp": fake_now})
  if monotonic_values is None:
    monotonic_values = [100.0, 101.0]
  it = iter(monotonic_values)
  monkeypatch.setattr(weather_mod.time, "monotonic", lambda: next(it))
  monkeypatch.setattr(weather_mod.time, "time", lambda: fake_now)
  return controller


def test_weather_controller_activates_on_radar_precipitation(monkeypatch):
  # New RainViewer-sourced payload: just precipitation_mm + severity + timestamp
  payload = {
    "precipitation_mm": 5.0,  # moderate radar observation
    "severity": "moderate",
  }
  controller = _make_controller(monkeypatch, payload)

  v_cruise = 30.0
  controller.update(v_ego=v_cruise, v_cruise=v_cruise)
  controller.update(v_ego=v_cruise, v_cruise=v_cruise)

  assert controller.is_active is True
  assert controller.severity == "moderate"
  assert controller.speed_recommendation != V_CRUISE_UNSET
  assert controller.speed_recommendation < v_cruise


def test_weather_controller_refuses_to_activate_with_trivial_precipitation(monkeypatch):
  payload = {
    "precipitation_mm": 0.05,  # below MIN_PRECIP_MM_PER_HR
    "severity": "none",
  }
  controller = _make_controller(monkeypatch, payload)
  controller.update(v_ego=30.0, v_cruise=30.0)
  assert controller.is_active is False
  assert controller.speed_recommendation == V_CRUISE_UNSET


def test_weather_controller_activates_with_light_radar_precipitation(monkeypatch):
  payload = {
    "precipitation_mm": 1.0,
    "severity": "light",
  }
  controller = _make_controller(monkeypatch, payload)
  controller.update(v_ego=30.0, v_cruise=30.0)
  controller.update(v_ego=30.0, v_cruise=30.0)
  assert controller.is_active is True
  assert controller.speed_recommendation < 30.0


def test_weather_controller_resets_when_precipitation_drops_to_zero(monkeypatch):
  # Start with active rain
  payload_rain = {"precipitation_mm": 4.0, "severity": "moderate"}
  controller = _make_controller(monkeypatch, payload_rain, monotonic_values=[100.0, 101.0, 102.0])
  controller.update(v_ego=30.0, v_cruise=30.0)
  controller.update(v_ego=30.0, v_cruise=30.0)
  assert controller.is_active is True

  # Then radar clears — precipitation drops to zero
  controller.params.put_weather({"precipitation_mm": 0.0, "severity": "none", "timestamp": 1234.0})
  controller.update(v_ego=30.0, v_cruise=30.0)
  assert controller.is_active is False


def test_weather_controller_ramps_up_faster_than_it_ramps_down(monkeypatch):
  # Sanity-check: WEATHER_ACCEL_RATE is strictly greater than WEATHER_DECEL_RATE.
  assert weather_mod.WEATHER_ACCEL_RATE > weather_mod.WEATHER_DECEL_RATE

  payload = {"precipitation_mm": 3.0, "severity": "moderate"}
  ticks = [100.0 + i for i in range(20)]
  controller = _make_controller(monkeypatch, payload, monotonic_values=ticks)

  v_cruise = 30.0
  controller.update(v_ego=v_cruise, v_cruise=v_cruise)
  controller.update(v_ego=v_cruise, v_cruise=v_cruise)
  first_cap = controller.speed_recommendation
  # Ramp down over several ticks
  for _ in range(5):
    controller.update(v_ego=v_cruise, v_cruise=v_cruise)
  ramped_down = controller.speed_recommendation
  assert ramped_down < first_cap

  # Driver raises cruise — ramp-up should rise at WEATHER_ACCEL_RATE
  before = controller._ramped_speed
  controller.update(v_ego=v_cruise, v_cruise=v_cruise + 20.0)
  after = controller._ramped_speed
  rise = after - before
  # dt=1.0 between ticks, rise should be ~WEATHER_ACCEL_RATE.
  assert rise == pytest.approx(weather_mod.WEATHER_ACCEL_RATE, rel=0.1)


def test_weather_controller_rejects_stale_timestamp(monkeypatch):
  # frame_time is a radar-scan wall-clock time. If it's > MAX_DATA_AGE_S old,
  # the controller refuses to activate.
  fake_params = FakeParams()
  controller = weather_mod.WeatherController()
  controller.params = fake_params
  fake_now = 2000.0
  stale_frame_time = fake_now - weather_mod.MAX_DATA_AGE_S - 10
  fake_params.put_weather({
    "precipitation_mm": 5.0,
    "severity": "moderate",
    "timestamp": stale_frame_time,
  })
  monkeypatch.setattr(weather_mod.time, "monotonic", lambda: 100.0)
  monkeypatch.setattr(weather_mod.time, "time", lambda: fake_now)
  controller.update(v_ego=30.0, v_cruise=30.0)
  assert controller.is_active is False


def test_weather_controller_disabled_by_param(monkeypatch):
  payload = {"precipitation_mm": 5.0, "severity": "moderate"}
  controller = _make_controller(monkeypatch, payload)
  controller.params._bools["WeatherAwareControlEnabled"] = False
  controller.update(v_ego=30.0, v_cruise=30.0)
  assert controller.is_active is False


# ---------------------------------------------------------------------------
# Speed-reduction curve shape: smoothstep between four anchor points
# ---------------------------------------------------------------------------
#
# The curve goes through:
#   (PRECIP_NONE,      0.0)
#   (PRECIP_LIGHT,     red_light)
#   (PRECIP_MODERATE,  red_moderate)
#   (PRECIP_HEAVY,     red_heavy)
# using a cubic smoothstep so slopes match at knots (no hard breakpoints).

RED_LIGHT = 5.0       # mph-equivalent units for readability in tests
RED_MODERATE = 10.0
RED_HEAVY = 15.0


def _reduce(precip_mm: float) -> float:
  return weather_mod._interpolate_reduction(
    precip_mm,
    red_none=0.0,
    red_light=RED_LIGHT,
    red_moderate=RED_MODERATE,
    red_heavy=RED_HEAVY,
  )


def test_interpolate_reduction_hits_each_anchor_exactly():
  assert _reduce(weather_mod.PRECIP_NONE) == 0.0
  assert _reduce(weather_mod.PRECIP_LIGHT) == pytest.approx(RED_LIGHT)
  assert _reduce(weather_mod.PRECIP_MODERATE) == pytest.approx(RED_MODERATE)
  assert _reduce(weather_mod.PRECIP_HEAVY) == pytest.approx(RED_HEAVY)


def test_interpolate_reduction_clamps_above_heavy_anchor():
  assert _reduce(weather_mod.PRECIP_HEAVY + 0.1) == pytest.approx(RED_HEAVY)
  assert _reduce(50.0) == pytest.approx(RED_HEAVY)
  assert _reduce(1000.0) == pytest.approx(RED_HEAVY)


def test_interpolate_reduction_zero_below_none_anchor():
  assert _reduce(0.0) == 0.0
  assert _reduce(-0.5) == 0.0


def test_drizzle_produces_much_smaller_reduction_than_red_light():
  # 0.3 mm/hr is classical drizzle. The new curve should give much less
  # than the full red_light value (contrast with the old linear code,
  # where 0.3 mm/hr → 3.0 mph once the LIGHT anchor was at 0.5 mm/hr).
  drizzle = _reduce(0.3)
  assert drizzle > 0.0
  assert drizzle < RED_LIGHT * 0.3, f"drizzle too aggressive: {drizzle}"


def test_interpolate_reduction_monotonic_across_spectrum():
  samples = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 6.0, 7.5, 10.0, 15.0, 30.0]
  values = [_reduce(p) for p in samples]
  for prev, curr in zip(values, values[1:], strict=False):
    assert curr >= prev - 1e-9, f"non-monotonic: {prev} → {curr}"


def test_smoothstep_c1_continuity_at_anchor_knots():
  # A numerical derivative on both sides of each anchor should match
  # closely — smoothstep has f'(0)=f'(1)=0, so the curve's slope approaches
  # zero from both sides at each knot. We assert the slopes match to within
  # a small tolerance (they're both near zero).
  eps = 1e-4
  for anchor in (weather_mod.PRECIP_LIGHT, weather_mod.PRECIP_MODERATE, weather_mod.PRECIP_HEAVY):
    left = (_reduce(anchor) - _reduce(anchor - eps)) / eps
    right = (_reduce(anchor + eps) - _reduce(anchor)) / eps
    assert abs(left - right) < 0.01, (
      f"slope discontinuity at {anchor}: left={left}, right={right}"
    )


def test_interpolate_reduction_midpoint_equals_anchor_average():
  # Smoothstep is antisymmetric about t=0.5, so the midpoint between two
  # anchors should produce exactly the average of the two Y values.
  mid_0_to_light = (weather_mod.PRECIP_NONE + weather_mod.PRECIP_LIGHT) / 2.0
  assert _reduce(mid_0_to_light) == pytest.approx(RED_LIGHT / 2.0, rel=1e-6)

  mid_light_to_mod = (weather_mod.PRECIP_LIGHT + weather_mod.PRECIP_MODERATE) / 2.0
  assert _reduce(mid_light_to_mod) == pytest.approx((RED_LIGHT + RED_MODERATE) / 2.0, rel=1e-6)

  mid_mod_to_heavy = (weather_mod.PRECIP_MODERATE + weather_mod.PRECIP_HEAVY) / 2.0
  assert _reduce(mid_mod_to_heavy) == pytest.approx((RED_MODERATE + RED_HEAVY) / 2.0, rel=1e-6)
