import math

from openpilot.sunnypilot.mapd.live_map_data.base_map_data import BaseMapData


class DummyGPS:
  pass


def test_extract_bearing_prefers_bearing_deg():
  gps = DummyGPS()
  gps.bearingDeg = 123.4
  gps.bearing = 90.0
  assert math.isclose(BaseMapData.extract_bearing_deg(gps), 123.4, rel_tol=0.0, abs_tol=1e-6)


def test_extract_bearing_falls_back_to_bearing():
  gps = DummyGPS()
  gps.bearing = 87.5
  assert math.isclose(BaseMapData.extract_bearing_deg(gps), 87.5, rel_tol=0.0, abs_tol=1e-6)


def test_extract_bearing_uses_vned_when_needed():
  gps = DummyGPS()
  gps.vNED = [0.0, 5.0, 0.0]
  # Eastward velocity should map to heading 90 deg.
  assert math.isclose(BaseMapData.extract_bearing_deg(gps), 90.0, rel_tol=0.0, abs_tol=1e-6)


def test_extract_bearing_defaults_to_zero():
  gps = DummyGPS()
  assert BaseMapData.extract_bearing_deg(gps) == 0.0
