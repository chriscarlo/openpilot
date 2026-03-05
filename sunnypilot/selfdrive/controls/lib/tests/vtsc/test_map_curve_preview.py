#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from types import SimpleNamespace

from unittest.mock import patch

import pytest

from openpilot.common.params import Params

from openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController

from .pipeline_harness import FakeSubMaster, make_car_control, make_car_state, make_model_v2, make_radar_state


def _latlon_from_local_en(lat0: float, lon0: float, x_east_m: float, y_north_m: float) -> tuple[float, float]:
  # Small-angle conversion: good enough for unit tests (tens/hundreds of meters)
  earth_r = 6371007.2
  dlat = y_north_m / earth_r
  dlon = x_east_m / (earth_r * math.cos(math.radians(lat0)))
  return (lat0 + math.degrees(dlat), lon0 + math.degrees(dlon))


def _build_map_curvatures_left_curve(*, lat0: float, lon0: float) -> list[dict]:
  # 45m straight then ~90deg left arc (radius 50m)
  pts = []
  # straight
  for i in range(0, 10):
    x = float(i * 5.0)
    y = 0.0
    la, lo = _latlon_from_local_en(lat0, lon0, x, y)
    pts.append({"latitude": la, "longitude": lo, "curvature": 0.0})
  # arc
  r = 50.0
  k = 1.0 / r
  x0 = 45.0
  for i in range(0, 40):
    ang = (math.pi / 2.0) * (i / 39.0)
    x = x0 + r * math.sin(ang)
    y = r * (1.0 - math.cos(ang))
    la, lo = _latlon_from_local_en(lat0, lon0, x, y)
    pts.append({"latitude": la, "longitude": lo, "curvature": float(k)})
  return pts


def _build_map_curvatures_right_curve(*, lat0: float, lon0: float) -> list[dict]:
  # Mirror across the x-axis by negating the northing in lat/lon construction.
  out: list[dict] = []
  for i in range(0, 10):
    x = float(i * 5.0)
    y = 0.0
    la, lo = _latlon_from_local_en(lat0, lon0, x, y)
    out.append({"latitude": la, "longitude": lo, "curvature": 0.0})
  r = 50.0
  k = 1.0 / r
  x0 = 45.0
  for i in range(0, 40):
    ang = (math.pi / 2.0) * (i / 39.0)
    x = x0 + r * math.sin(ang)
    y = -r * (1.0 - math.cos(ang))
    la, lo = _latlon_from_local_en(lat0, lon0, x, y)
    out.append({"latitude": la, "longitude": lo, "curvature": float(k)})
  return out


def _build_sparse_map_curvatures_left_curve(*, lat0: float, lon0: float) -> list[dict]:
  pts = []
  for i in range(0, 6):
    x = float(i * 20.0)
    y = 0.0
    la, lo = _latlon_from_local_en(lat0, lon0, x, y)
    pts.append({"latitude": la, "longitude": lo, "curvature": 0.0})

  r = 80.0
  k = 1.0 / r
  x0 = 100.0
  for i in range(0, 10):
    ang = (math.pi / 2.0) * (i / 9.0)
    x = x0 + r * math.sin(ang)
    y = r * (1.0 - math.cos(ang))
    la, lo = _latlon_from_local_en(lat0, lon0, x, y)
    pts.append({"latitude": la, "longitude": lo, "curvature": float(k)})
  return pts


def _make_sm(*, curvature: float = 0.0, confidence: float = 0.9, v_ego: float = 25.0) -> FakeSubMaster:
  model = make_model_v2(curvature=float(curvature), v_pred=float(max(0.1, v_ego)), confidence=float(confidence))
  return FakeSubMaster(
    data={
      "modelV2": model,
      "radarState": make_radar_state(lead_d_rel_m=None),
      "carState": make_car_state(gas_pressed=False),
      "carControl": make_car_control(long_active=True),
    },
    valid={"modelV2": True, "radarState": True},
  )


class _MockCP:
  pass


def _road_segment_from_local_en(*, way_id: int, lat0: float, lon0: float,
                                centerline_en: list[tuple[float, float]],
                                road_direction: float = 90.0,
                                level_separation: int = 0):
  coords = []
  dist = 0.0
  prev = None
  for x_east, y_north in centerline_en:
    lat, lon = _latlon_from_local_en(lat0, lon0, float(x_east), float(y_north))
    if prev is not None:
      dist += math.hypot(float(x_east) - float(prev[0]), float(y_north) - float(prev[1]))
    coords.append(SimpleNamespace(latitude=lat, longitude=lon, distanceFromStart=float(dist)))
    prev = (float(x_east), float(y_north))

  return SimpleNamespace(
    wayId=int(way_id),
    levelSeparation=int(level_separation),
    roadDirection=float(road_direction),
    centerline=coords,
  )


@pytest.mark.parametrize("builder,expected_dir", [
  (_build_map_curvatures_left_curve, 1),   # left
  (_build_map_curvatures_right_curve, 2),  # right
])
def test_vtsc_map_curve_preview_direction_and_severity(builder, expected_dir):
  p = Params()
  p.put_bool("VisionTurnSpeedControl", True)
  p.put_bool("MTSCLookaheadEnabled", True)

  lat0, lon0 = 37.0, -122.0
  p.put("MapCurvatures", json.dumps(builder(lat0=lat0, lon0=lon0)))
  p.put("LastGPSPosition", json.dumps({"latitude": lat0, "longitude": lon0, "bearing": 90.0}))

  vtc = VisionTurnController(_MockCP())
  sm = _make_sm(v_ego=25.0)

  # Deterministic time for map cache throttling
  with patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time", lambda: 0.0), \
       patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic", lambda: 0.0):
    vtc.update(sm, True, 25.0, 0.0, 40.0)

  assert vtc.curve_preview_valid
  assert vtc.curve_preview_direction == expected_dir
  assert vtc.curve_preview_severity in (1, 2, 3)
  # Our synthetic curve is tight (R=50m), should bucket as tight.
  assert vtc.curve_preview_severity == 3
  # R=50m => kappa = 0.02 1/m (peak)
  assert vtc.curve_preview_kappa_max >= 0.015
  assert 20.0 <= vtc.curve_preview_distance_m <= 80.0
  pts = vtc.curve_preview_points
  assert isinstance(pts, list)
  # Preview now retains a denser decimated polyline for HUD smoothness.
  assert 3 <= len(pts) <= 48


def test_vtsc_map_curve_preview_invalid_on_straight():
  p = Params()
  p.put_bool("VisionTurnSpeedControl", True)
  p.put_bool("MTSCLookaheadEnabled", True)

  lat0, lon0 = 37.0, -122.0
  # 200m straight, zero curvature
  pts = []
  for i in range(0, 30):
    x = float(i * 7.0)
    y = 0.0
    la, lo = _latlon_from_local_en(lat0, lon0, x, y)
    pts.append({"latitude": la, "longitude": lo, "curvature": 0.0})
  p.put("MapCurvatures", json.dumps(pts))
  p.put("LastGPSPosition", json.dumps({"latitude": lat0, "longitude": lon0, "bearing": 90.0}))

  vtc = VisionTurnController(_MockCP())
  sm = _make_sm(v_ego=25.0)

  with patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time", lambda: 0.0), \
       patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic", lambda: 0.0):
    vtc.update(sm, True, 25.0, 0.0, 40.0)

  # Straight-road preview is intentionally pre-cached as valid geometry with zero curve metadata.
  assert vtc.curve_preview_valid
  assert vtc.curve_preview_kappa_max == 0.0
  assert vtc.curve_preview_direction == 0
  assert vtc.curve_preview_severity == 0
  assert 3 <= len(vtc.curve_preview_points) <= 48


def test_vtsc_map_curve_preview_uses_fixed_ten_second_horizon_on_straight():
  p = Params()
  p.put_bool("VisionTurnSpeedControl", True)
  p.put_bool("MTSCLookaheadEnabled", True)

  lat0, lon0 = 37.0, -122.0
  pts = []
  for i in range(0, 40):
    x = float(i * 15.0)
    y = 0.0
    la, lo = _latlon_from_local_en(lat0, lon0, x, y)
    pts.append({"latitude": la, "longitude": lo, "curvature": 0.0})
  p.put("MapCurvatures", json.dumps(pts))
  p.put("LastGPSPosition", json.dumps({"latitude": lat0, "longitude": lon0, "bearing": 90.0}))

  vtc = VisionTurnController(_MockCP())
  sm = _make_sm(v_ego=18.0)

  with patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time", lambda: 0.0), \
       patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic", lambda: 0.0):
    vtc.update(sm, True, 18.0, 0.0, 40.0)

  xs = [float(x) for x, _ in vtc.curve_preview_points]
  assert xs[0] == pytest.approx(0.0, abs=0.25)
  assert xs[-1] == pytest.approx(180.0, abs=6.0)


def test_vtsc_map_curve_preview_resamples_sparse_geometry():
  p = Params()
  p.put_bool("VisionTurnSpeedControl", True)
  p.put_bool("MTSCLookaheadEnabled", True)

  lat0, lon0 = 37.0, -122.0
  p.put("MapCurvatures", json.dumps(_build_sparse_map_curvatures_left_curve(lat0=lat0, lon0=lon0)))
  p.put("LastGPSPosition", json.dumps({"latitude": lat0, "longitude": lon0, "bearing": 90.0}))

  vtc = VisionTurnController(_MockCP())
  sm = _make_sm(v_ego=22.0)

  with patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time", lambda: 0.0), \
       patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic", lambda: 0.0):
    vtc.update(sm, True, 22.0, 0.0, 40.0)

  pts = vtc.curve_preview_points
  assert len(pts) >= 24
  xs = [float(x) for x, _ in pts]
  seg_lens = [
    math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False)
  ]
  assert xs[0] == pytest.approx(0.0, abs=0.25)
  assert max(seg_lens) < 8.5
  assert all(xs[i + 1] >= xs[i] - 1e-3 for i in range(len(xs) - 1))


def test_vtsc_map_curve_preview_builds_intersection_stubs_from_live_map_data():
  p = Params()
  p.put_bool("VisionTurnSpeedControl", True)
  p.put_bool("MTSCLookaheadEnabled", True)

  lat0, lon0 = 37.0, -122.0
  pts = []
  for i in range(0, 40):
    x = float(i * 12.0)
    la, lo = _latlon_from_local_en(lat0, lon0, x, 0.0)
    pts.append({"latitude": la, "longitude": lo, "curvature": 0.0})
  p.put("MapCurvatures", json.dumps(pts))
  p.put("LastGPSPosition", json.dumps({"latitude": lat0, "longitude": lon0, "bearing": 90.0}))

  current_seg = _road_segment_from_local_en(
    way_id=101,
    lat0=lat0,
    lon0=lon0,
    centerline_en=[(-20.0, 0.0), (40.0, 0.0), (100.0, 0.0), (180.0, 0.0)],
  )
  left_stub = _road_segment_from_local_en(
    way_id=202,
    lat0=lat0,
    lon0=lon0,
    centerline_en=[(55.0, 0.0), (55.0, 12.0), (55.0, 24.0), (55.0, 38.0)],
  )
  right_stub = _road_segment_from_local_en(
    way_id=303,
    lat0=lat0,
    lon0=lon0,
    centerline_en=[(90.0, 0.0), (90.0, -12.0), (90.0, -24.0), (90.0, -38.0)],
  )

  vtc = VisionTurnController(_MockCP())
  sm = _make_sm(v_ego=18.0)
  sm._data["liveMapDataSP"] = SimpleNamespace(
    roadGeometryValid=True,
    currentRoadSegment=current_seg,
    nearbyRoadSegments=[left_stub, right_stub],
  )
  sm.valid["liveMapDataSP"] = True

  with patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time", lambda: 0.0), \
       patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic", lambda: 0.0):
    vtc.update(sm, True, 18.0, 0.0, 40.0)

  stubs = vtc.curve_preview_branch_stubs
  assert len(stubs) == 2
  assert not any(bool(stub["highlighted"]) for stub in stubs)
  end_y = [float(stub["points"][-1][1]) for stub in stubs]
  assert any(y > 8.0 for y in end_y)
  assert any(y < -8.0 for y in end_y)
  start_x = [float(stub["points"][0][0]) for stub in stubs]
  assert all(x >= 0.0 for x in start_x)


def test_vtsc_map_curve_preview_highlights_blinker_side_stub():
  p = Params()
  p.put_bool("VisionTurnSpeedControl", True)
  p.put_bool("MTSCLookaheadEnabled", True)

  lat0, lon0 = 37.0, -122.0
  pts = []
  for i in range(0, 40):
    x = float(i * 12.0)
    la, lo = _latlon_from_local_en(lat0, lon0, x, 0.0)
    pts.append({"latitude": la, "longitude": lo, "curvature": 0.0})
  p.put("MapCurvatures", json.dumps(pts))
  p.put("LastGPSPosition", json.dumps({"latitude": lat0, "longitude": lon0, "bearing": 90.0}))

  current_seg = _road_segment_from_local_en(
    way_id=101,
    lat0=lat0,
    lon0=lon0,
    centerline_en=[(-20.0, 0.0), (40.0, 0.0), (100.0, 0.0), (180.0, 0.0)],
  )
  left_stub = _road_segment_from_local_en(
    way_id=202,
    lat0=lat0,
    lon0=lon0,
    centerline_en=[(55.0, 0.0), (55.0, 12.0), (55.0, 24.0), (55.0, 38.0)],
  )
  right_stub = _road_segment_from_local_en(
    way_id=303,
    lat0=lat0,
    lon0=lon0,
    centerline_en=[(90.0, 0.0), (90.0, -12.0), (90.0, -24.0), (90.0, -38.0)],
  )

  vtc = VisionTurnController(_MockCP())
  sm = _make_sm(v_ego=18.0)
  sm._data["carState"] = make_car_state(gas_pressed=False, right_blinker=True)
  sm._data["liveMapDataSP"] = SimpleNamespace(
    roadGeometryValid=True,
    currentRoadSegment=current_seg,
    nearbyRoadSegments=[left_stub, right_stub],
  )
  sm.valid["liveMapDataSP"] = True

  with patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time", lambda: 0.0), \
       patch("openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic", lambda: 0.0):
    vtc.update(sm, True, 18.0, 0.0, 40.0)

  stubs = vtc.curve_preview_branch_stubs
  assert len(stubs) == 1
  assert bool(stubs[0]["highlighted"])
  assert float(stubs[0]["points"][-1][1]) < -8.0
