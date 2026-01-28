import pytest

from openpilot.sunnypilot.livedelay.helpers import get_lat_delay


class FakeParams:
  def __init__(self, lagd_toggle: bool, lagd_value_cache: str | None):
    self._lagd_toggle = lagd_toggle
    self._lagd_value_cache = lagd_value_cache

  def get_bool(self, key: str) -> bool:
    assert key == "LagdToggle"
    return self._lagd_toggle

  def get(self, key: str, return_default: bool = False):
    assert key == "LagdValueCache"
    if self._lagd_value_cache is not None:
      return self._lagd_value_cache
    return "0.2" if return_default else None


def test_get_lat_delay_toggle_on_uses_stock():
  params = FakeParams(True, "0.42")
  assert get_lat_delay(params, 0.31) == pytest.approx(0.31)


def test_get_lat_delay_toggle_off_uses_cached_manual_delay():
  params = FakeParams(False, "0.42")
  assert get_lat_delay(params, 0.31) == pytest.approx(0.42)


def test_get_lat_delay_toggle_off_falls_back_when_cache_invalid():
  params = FakeParams(False, "0.0")
  assert get_lat_delay(params, 0.31) == pytest.approx(0.31)
