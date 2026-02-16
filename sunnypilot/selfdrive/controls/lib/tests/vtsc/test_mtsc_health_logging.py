#!/usr/bin/env python3
import json

from .harness import mk_vtsc_with_params


class _ParamStub:
  def __init__(self, values=None, bools=None):
    self._values = values or {}
    self._bools = bools or {}

  def get(self, key: str):
    return self._values.get(key)

  def get_bool(self, key: str) -> bool:
    return bool(self._bools.get(key, False))


def _make_controller(values: dict[str, bytes], bools: dict[str, bool], v_ego: float = 20.0):
  ctrl = mk_vtsc_with_params()
  stub = _ParamStub(values=values, bools=bools)
  ctrl._params = stub
  ctrl._mem_params = stub
  ctrl._v_ego = float(v_ego)
  ctrl._mtsc_health_last_ts = 0.0
  return ctrl


def test_mtsc_health_disabled_status():
  ctrl = _make_controller(
    values={
      "MapCurvatures": b"[]",
      "LastGPSPosition": json.dumps({"latitude": 38.0, "longitude": -121.0, "bearing": 42.0}).encode(),
    },
    bools={"MTSCLookaheadEnabled": False},
  )
  snap = ctrl.snapshot_debug_state()
  assert snap["mtsc_enabled"] is False
  assert snap["mtsc_status"] == "disabled"


def test_mtsc_health_flags_missing_bearing():
  ctrl = _make_controller(
    values={
      "MapCurvatures": json.dumps([{"latitude": 38.0, "longitude": -121.0, "curvature": 0.001}]).encode(),
      "LastGPSPosition": json.dumps({"latitude": 38.0, "longitude": -121.0}).encode(),
    },
    bools={"MTSCLookaheadEnabled": True},
  )
  snap = ctrl.snapshot_debug_state()
  assert snap["mtsc_enabled"] is True
  assert snap["mtsc_status"] == "missing_gps_bearing"
  assert snap["mtsc_gps_has_bearing"] is False


def test_mtsc_health_flags_empty_curvature_while_moving():
  ctrl = _make_controller(
    values={
      "MapCurvatures": b"[]",
      "LastGPSPosition": json.dumps({"latitude": 38.0, "longitude": -121.0, "bearing": 15.0}).encode(),
    },
    bools={"MTSCLookaheadEnabled": True},
    v_ego=26.0,
  )
  snap = ctrl.snapshot_debug_state()
  assert snap["mtsc_status"] == "no_curvature_while_moving"
  assert snap["mtsc_curv_count"] == 0


def test_mtsc_health_ok_when_curvature_and_bearing_present():
  ctrl = _make_controller(
    values={
      "MapCurvatures": json.dumps([
        {"latitude": 38.0, "longitude": -121.0, "curvature": 0.001},
        {"latitude": 38.1, "longitude": -121.1, "curvature": 0.002},
      ]).encode(),
      "LastGPSPosition": json.dumps({"latitude": 38.0, "longitude": -121.0, "bearing": 270.0}).encode(),
    },
    bools={"MTSCLookaheadEnabled": True},
    v_ego=18.0,
  )
  snap = ctrl.snapshot_debug_state()
  assert snap["mtsc_status"] == "ok"
  assert snap["mtsc_curv_count"] == 2
  assert snap["mtsc_gps_has_bearing"] is True
  assert abs(float(snap["mtsc_gps_bearing_deg"]) - 270.0) < 1e-6
