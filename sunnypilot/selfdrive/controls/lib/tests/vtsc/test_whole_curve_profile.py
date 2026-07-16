import json
import math

import pytest

import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vtc_mod
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
  EARTH_R_M,
  MAP_WHOLE_CURVE_ESTIMATOR_VERSION,
  MapWholeCurvePoint,
  VisionTurnController,
  _compute_map_whole_curve_route_fingerprint,
  curvature_to_speed,
)

from .harness import mk_vtsc_with_params


EVENT_ID = "0123456789abcdefabcd-a"


def _profile_payload(*, now_s: float | None = None, count: int = 40,
                     curvature: float = -0.02, generation: int = 7,
                     sigmoid_hash: str | None = None) -> dict:
  now_s = 1_800_000_000.0 if now_s is None else float(now_s)
  sigmoid_hash = vtc_mod._compute_runtime_sigmoid_hash() if sigmoid_hash is None else sigmoid_hash
  step_deg = math.degrees(5.0 / EARTH_R_M)
  points = []
  typed_points = []
  event_start = max(2, count // 4)
  event_end = min(count - 2, event_start + max(3, count // 5))
  for index in range(count):
    event_id = EVENT_ID if event_start <= index <= event_end else ""
    kappa = curvature if event_id else 0.0
    point = {
      "latitude": 37.0 + step_deg * index,
      "longitude": -122.0,
      "distanceMeters": 5.0 * index,
      "curvature": kappa,
      "curvatureCoefficient": 0.65 if event_id else 1.0,
      "baseSafeSpeedMPS": 12.5 if event_id else vtc_mod.MAX_SPEED_DEFAULT,
      "eventID": event_id,
      "confidence": 0.9 if event_id else 1.0,
      "flags": [],
    }
    points.append(point)
    typed_points.append(MapWholeCurvePoint(
      point["latitude"], point["longitude"], point["distanceMeters"],
      point["curvature"], point["curvatureCoefficient"], point["baseSafeSpeedMPS"],
      point["eventID"], point["confidence"], (),
    ))
  return {
    "estimatorVersion": MAP_WHOLE_CURVE_ESTIMATOR_VERSION,
    "generatedAtUnixMillis": now_s * 1000.0,
    "routeFingerprint": _compute_map_whole_curve_route_fingerprint(generation, sigmoid_hash, typed_points),
    "sigmoidHash": sigmoid_hash,
    "generation": generation,
    "points": points,
    "events": [{
      "eventID": EVENT_ID,
      "startIndex": event_start,
      "endIndex": event_end,
      "apexIndex": event_start + (event_end - event_start) // 2,
      "controllingCurvature": curvature,
      "confidence": 0.9,
      "flags": [],
    }],
    "fatalAmbiguity": False,
  }


def _parse(payload: dict, *, now_s: float):
  return VisionTurnController._parse_map_whole_curve_profile(
    json.dumps(payload),
    (37.0, -122.0, 0.0),
    now_s=now_s,
  )


def test_whole_curve_fingerprint_cross_language_vector():
  points = [
    MapWholeCurvePoint(37.0, -122.0, 0.0, 0.0, 1.0, 70.0, ""),
    MapWholeCurvePoint(37.000045, -122.0, 5.0, -0.0123456785, 1.25, 11.25, "0123456789abcdefabcd-a"),
    MapWholeCurvePoint(37.000090, -122.0, 10.0, 0.0123456785, 1.5, 10.5, "0123456789abcdefabcd-b"),
  ]
  assert _compute_map_whole_curve_route_fingerprint(7, "85a608e68945", points) == \
    "a0d4823aa1f16ae4bd70f3d80a1d379f02fd944ac7804ad193b9875ec40710db"


def test_source_defaults_match_persisted_whole_curve_production_tune():
  assert vtc_mod.PHYSICS_A == pytest.approx(-2.548675, abs=1e-9)
  assert vtc_mod.PHYSICS_B == pytest.approx(-1024.629261, abs=1e-9)
  assert vtc_mod.PHYSICS_C == pytest.approx(0.006053, abs=1e-12)
  assert vtc_mod.PHYSICS_D == pytest.approx(4.107103, abs=1e-9)
  assert vtc_mod.PHYSICS_MIN_LAT_ACCEL == pytest.approx(1.5584, abs=1e-9)
  assert vtc_mod.PHYSICS_MAX_LAT_ACCEL == pytest.approx(4.1071, abs=1e-9)
  assert vtc_mod.MAP_PRECURVE_SPEEDS_ESTIMATOR_ALIGNED is False


@pytest.mark.parametrize("gps_payload", [
  {"latitude": math.nan, "longitude": -122.0, "bearing": 0.0},
  {"latitude": math.inf, "longitude": -122.0, "bearing": 0.0},
  {"latitude": 37.0, "longitude": math.nan, "bearing": 0.0},
  {"latitude": 37.0, "longitude": -math.inf, "bearing": 0.0},
  {"latitude": 91.0, "longitude": -122.0, "bearing": 0.0},
  {"latitude": 37.0, "longitude": -181.0, "bearing": 0.0},
  {"latitude": 0.0, "longitude": 0.0, "bearing": 0.0},
  {"latitude": True, "longitude": -122.0, "bearing": 0.0},
  {"longitude": -122.0, "bearing": 0.0},
  [37.0, -122.0],
])
def test_last_gps_pose_rejects_malformed_or_out_of_range_coordinates(gps_payload):
  vtsc = mk_vtsc_with_params(value_overrides={"LastGPSPosition": json.dumps(gps_payload)})

  assert vtsc._get_last_gps_pose() is None


def test_last_gps_pose_preserves_valid_coordinates_and_normalizes_bearing():
  payload = {"latitude": 37.0, "longitude": -122.0, "altitude": 42.0, "bearing": 450.0}
  vtsc = mk_vtsc_with_params(value_overrides={"LastGPSPosition": json.dumps(payload)})

  assert vtsc._get_last_gps_pose() == pytest.approx((37.0, -122.0, 90.0))


def test_whole_curve_profile_preserves_signed_full_resolution_route():
  now_s = 1_800_000_000.0
  profile = _parse(_profile_payload(now_s=now_s, count=200), now_s=now_s)
  assert len(profile.points) == 200
  assert profile.points[50].curvature < 0.0
  assert profile.points[-1].distance_m == pytest.approx(995.0)


def test_unchanged_whole_curve_payload_parses_and_fingerprints_once(monkeypatch):
  now_s = 1_800_000_000.0
  raw = json.dumps(_profile_payload(now_s=now_s, count=260))
  vtsc = mk_vtsc_with_params(value_overrides={"MapWholeCurveProfile": raw})
  parse_calls = []
  fingerprint_calls = []
  original_parse = vtsc._parse_map_whole_curve_profile
  original_fingerprint = vtc_mod._compute_map_whole_curve_route_fingerprint

  def counting_parse(raw_json, gps_pose, *, now_s=None):
    parse_calls.append(raw_json)
    return original_parse(raw_json, gps_pose, now_s=now_s)

  def counting_fingerprint(generation, sigmoid_hash, points):
    fingerprint_calls.append((generation, sigmoid_hash, len(points)))
    return original_fingerprint(generation, sigmoid_hash, points)

  monkeypatch.setattr(vtc_mod.time, "time", lambda: now_s)
  monkeypatch.setattr(vtsc, "_parse_map_whole_curve_profile", counting_parse, raising=True)
  monkeypatch.setattr(vtc_mod, "_compute_map_whole_curve_route_fingerprint", counting_fingerprint)

  first = vtsc._load_map_whole_curve_profile((37.0, -122.0, 0.0))
  repeated = [vtsc._load_map_whole_curve_profile((37.0, -122.0, 0.0)) for _ in range(5)]

  assert first is not None
  assert all(profile is first for profile in repeated)
  assert len(parse_calls) == 1
  assert fingerprint_calls == [(7, vtc_mod._compute_runtime_sigmoid_hash(), 260)]


def test_whole_curve_cache_invalidates_on_raw_or_timestamp_change_and_rechecks_freshness(monkeypatch):
  now_s = 1_800_000_000.0
  payload = _profile_payload(now_s=now_s, count=80)
  overrides = {"MapWholeCurveProfile": json.dumps(payload)}
  vtsc = mk_vtsc_with_params(value_overrides=overrides)
  clock = [now_s]
  parse_calls = []
  original_parse = vtsc._parse_map_whole_curve_profile

  def counting_parse(raw_json, gps_pose, *, now_s=None):
    parse_calls.append(raw_json)
    return original_parse(raw_json, gps_pose, now_s=now_s)

  monkeypatch.setattr(vtc_mod.time, "time", lambda: clock[0])
  monkeypatch.setattr(vtsc, "_parse_map_whole_curve_profile", counting_parse, raising=True)

  first = vtsc._load_map_whole_curve_profile((37.0, -122.0, 0.0))
  assert vtsc._load_map_whole_curve_profile((37.0, -122.0, 0.0)) is first
  assert len(parse_calls) == 1

  clock[0] = now_s + 0.5
  payload["generatedAtUnixMillis"] = clock[0] * 1000.0
  overrides["MapWholeCurveProfile"] = json.dumps(payload)
  refreshed = vtsc._load_map_whole_curve_profile((37.0, -122.0, 0.0))
  assert refreshed is not None and refreshed is not first
  assert len(parse_calls) == 2

  overrides["MapWholeCurveProfile"] = json.dumps(payload, separators=(",", ":"))
  reformatted = vtsc._load_map_whole_curve_profile((37.0, -122.0, 0.0))
  assert reformatted is not None and reformatted is not refreshed
  assert len(parse_calls) == 3

  clock[0] = now_s + 4.0
  assert vtsc._load_map_whole_curve_profile((37.0, -122.0, 0.0)) is None
  assert len(parse_calls) == 3
  assert vtsc._map_whole_curve_reason == "rejected_stale"


@pytest.mark.parametrize("mutation, expected", [
  (lambda payload: payload.update(estimatorVersion="whole-curve-v0"), "version_mismatch"),
  (lambda payload: payload.update(generatedAtUnixMillis=1.0), "stale"),
  (lambda payload: payload.pop("sigmoidHash"), "invalid_sigmoid_hash"),
  (lambda payload: payload.update(fatalAmbiguity=True), "fatal_ambiguity"),
  (lambda payload: payload.update(routeFingerprint="0" * 64), "fingerprint_mismatch"),
  (lambda payload: payload["points"][5].update(curvature=float("nan")), "not_finite"),
  (lambda payload: payload["points"][5].update(curvatureCoefficient=0.0), "curvature_coefficient_range"),
  (lambda payload: payload["points"][5].update(baseSafeSpeedMPS=vtc_mod.MAX_SPEED_DEFAULT + 1.0), "base_safe_speed_range"),
  (lambda payload: payload["points"][5].update(distanceMeters=payload["points"][4]["distanceMeters"]), "nonmonotonic"),
  (lambda payload: payload["points"][5].update(latitude=payload["points"][5]["latitude"] + 0.001), "distance_mismatch"),
])
def test_whole_curve_profile_rejects_invalid_data(mutation, expected):
  now_s = 1_800_000_000.0
  payload = _profile_payload(now_s=now_s)
  mutation(payload)
  with pytest.raises(ValueError, match=expected):
    _parse(payload, now_s=now_s)


def test_rejected_whole_curve_profile_falls_back_to_legacy_map_curvatures(monkeypatch):
  now_s = 1_800_000_000.0
  payload = _profile_payload(now_s=now_s)
  payload["estimatorVersion"] = "whole-curve-v0"
  vtsc = mk_vtsc_with_params(value_overrides={"MapWholeCurveProfile": json.dumps(payload)})
  vtsc._v_ego = 20.0
  vtsc._v_cruise_setpoint = 25.0
  vtsc._map_strategy_mode = "advisory"
  step_deg = math.degrees(10.0 / EARTH_R_M)
  legacy = [(37.0 + step_deg * index, -122.0, 0.015 if 5 <= index <= 10 else 0.0) for index in range(30)]
  monkeypatch.setattr(vtsc, "_get_last_gps_pose", lambda: (37.0, -122.0, 0.0), raising=True)
  monkeypatch.setattr(vtsc, "_load_map_curvatures", lambda: legacy, raising=True)

  vtsc._map_tail_cap()

  assert vtsc._map_profile_source == "legacy_map_curvatures"
  assert vtsc._map_whole_curve_reason == "rejected_version_mismatch"


def test_malformed_gps_fails_open_before_whole_profile_or_legacy_fallback(monkeypatch):
  now_s = 1_800_000_000.0
  vtsc = mk_vtsc_with_params(value_overrides={
    "LastGPSPosition": json.dumps({"latitude": math.nan, "longitude": -122.0, "bearing": 0.0}),
    "MapWholeCurveProfile": json.dumps(_profile_payload(now_s=now_s)),
  })
  whole_calls = []
  legacy_calls = []
  vtsc._map_profile_source = MAP_WHOLE_CURVE_ESTIMATOR_VERSION
  monkeypatch.setattr(vtsc, "_load_map_whole_curve_profile", lambda _pose: whole_calls.append(True), raising=True)
  monkeypatch.setattr(vtsc, "_load_map_curvatures", lambda: legacy_calls.append(True), raising=True)

  assert vtsc._map_tail_cap() == (None, 0.0, 0.0)
  assert vtsc._map_tail_compute_reason == "no_gps"
  assert vtsc._map_profile_source == "none"
  assert whole_calls == []
  assert legacy_calls == []


def test_cached_whole_curve_route_drift_falls_back_to_legacy(monkeypatch):
  now_s = 1_800_000_000.0
  payload = _profile_payload(now_s=now_s, count=80)
  vtsc = mk_vtsc_with_params(value_overrides={"MapWholeCurveProfile": json.dumps(payload)})
  vtsc._v_ego = 20.0
  vtsc._v_cruise_setpoint = 25.0
  vtsc._map_strategy_mode = "advisory"
  pose = [(37.0, -122.0, 0.0)]
  step_deg = math.degrees(10.0 / EARTH_R_M)
  legacy = [(38.0 + step_deg * index, -122.0, 0.015 if 5 <= index <= 10 else 0.0) for index in range(30)]
  monkeypatch.setattr(vtc_mod.time, "time", lambda: now_s)
  monkeypatch.setattr(vtsc, "_get_last_gps_pose", lambda: pose[0], raising=True)
  monkeypatch.setattr(vtsc, "_load_map_curvatures", lambda: legacy, raising=True)

  vtsc._map_tail_cap()
  cached_profile = vtsc._map_whole_curve_cache
  assert cached_profile is not None
  assert vtsc._map_profile_source == MAP_WHOLE_CURVE_ESTIMATOR_VERSION

  pose[0] = (38.0, -122.0, 0.0)
  vtsc._map_tail_cap()

  assert vtsc._map_profile_source == "legacy_map_curvatures"
  assert vtsc._map_whole_curve_reason == "rejected_route_not_near_ego"
  assert vtsc._map_whole_curve_cache is cached_profile


def test_whole_curve_path_uses_runtime_sigmoid_q_without_learned_calibration(monkeypatch):
  monkeypatch.setattr(vtc_mod, "PHYSICS_A", -1.5)
  monkeypatch.setattr(vtc_mod, "PHYSICS_B", -1200.0)
  monkeypatch.setattr(vtc_mod, "PHYSICS_C", 0.006)
  monkeypatch.setattr(vtc_mod, "PHYSICS_D", 4.0)
  monkeypatch.setattr(vtc_mod, "PHYSICS_MIN_LAT_ACCEL", 2.0)
  monkeypatch.setattr(vtc_mod, "PHYSICS_MAX_LAT_ACCEL", 4.0)
  monkeypatch.setattr(vtc_mod, "Q_CURVE_ENABLED", True)
  monkeypatch.setattr(vtc_mod, "Q_CURVE_POINTS", [(1e-4, 1.1), (0.1, 1.1)])
  kappa = 0.02
  vtsc = mk_vtsc_with_params()
  monkeypatch.setattr(vtsc, "_low_speed_calibration_scale", lambda _k: 1.2, raising=True)

  expected = curvature_to_speed(kappa, low_speed_sigmoid_scale=1.0)
  assert vtsc._whole_curve_speed(-kappa) == pytest.approx(expected)
  assert vtsc._whole_curve_speed(-kappa) != pytest.approx(vtsc._curve_speed(kappa))


def test_whole_curve_matching_hash_consumes_route_baked_physics_speed(monkeypatch):
  now_s = 1_800_000_000.0
  payload = _profile_payload(now_s=now_s, count=80)
  vtsc = mk_vtsc_with_params(value_overrides={"MapWholeCurveProfile": json.dumps(payload)})
  vtsc._v_ego = 20.0
  vtsc._v_cruise_setpoint = 30.0
  baked_calls = []
  monkeypatch.setattr(vtc_mod.time, "time", lambda: now_s)
  monkeypatch.setattr(vtsc, "_get_last_gps_pose", lambda: (37.0, -122.0, 0.0), raising=True)
  monkeypatch.setattr(
    vtsc,
    "_baked_vsafe_with_runtime_multipliers",
    lambda speed, kappa: baked_calls.append((speed, kappa)) or float(speed),
    raising=True,
  )
  monkeypatch.setattr(
    vtsc,
    "_whole_curve_speed",
    lambda _kappa: pytest.fail("matching v3 hash should not run the live sigmoid"),
    raising=True,
  )

  vtsc._map_tail_cap()

  assert baked_calls
  assert any(speed == pytest.approx(12.5) for speed, _kappa in baked_calls)
  first_call_count = len(baked_calls)
  vtsc._map_tail_cap()
  assert len(baked_calls) == first_call_count


def test_whole_curve_hash_mismatch_falls_back_to_live_sigmoid(monkeypatch):
  now_s = 1_800_000_000.0
  payload = _profile_payload(now_s=now_s, count=80, sigmoid_hash="000000000000")
  vtsc = mk_vtsc_with_params(value_overrides={"MapWholeCurveProfile": json.dumps(payload)})
  vtsc._v_ego = 20.0
  vtsc._v_cruise_setpoint = 30.0
  live_calls = []
  monkeypatch.setattr(vtc_mod.time, "time", lambda: now_s)
  monkeypatch.setattr(vtsc, "_get_last_gps_pose", lambda: (37.0, -122.0, 0.0), raising=True)
  monkeypatch.setattr(
    vtsc,
    "_whole_curve_speed",
    lambda kappa: live_calls.append(kappa) or 14.0,
    raising=True,
  )
  monkeypatch.setattr(
    vtsc,
    "_baked_vsafe_with_runtime_multipliers",
    lambda _speed, _kappa: pytest.fail("mismatched v3 hash must not consume baked speed"),
    raising=True,
  )

  vtsc._map_tail_cap()

  assert live_calls


def test_map_lookahead_falling_edge_clears_map_cap_on_first_disabled_tick(monkeypatch):
  vtsc = mk_vtsc_with_params()
  vtsc._op_enabled = True
  vtsc._is_enabled = True
  vtsc._gas_pressed = False
  vtsc._v_ego = 25.0
  vtsc._v_cruise_setpoint = 30.0
  vtsc._prev_target_speed = 25.0
  vtsc._map_lookahead_enabled_prev = True
  vtsc._map_tail_active = True
  vtsc._map_tail_candidate = object()
  vtsc._map_tail_last_cap = 12.0
  vtsc._map_holdover_cap = 12.0
  vtsc._map_holdover_candidate = object()
  vtsc._v_turn_hold_until = 1e20
  vtsc._v_turn_hold_min = 12.0
  vtsc._v_turn_release_shape_active = True
  monkeypatch.setattr(vtsc, "_plan_advanced_speed_trajectory", lambda: 30.0, raising=True)
  original_get_bool = vtsc._get_bool_param
  monkeypatch.setattr(
    vtsc,
    "_get_bool_param",
    lambda key, default=False: False if key == "MTSCLookaheadEnabled" else original_get_bool(key, default),
    raising=True,
  )

  vtsc._update_solution(None)

  assert vtsc._map_tail_reason == "toggle_off"
  assert vtsc._map_tail_active is False
  assert vtsc._map_tail_candidate is None
  assert vtsc._map_tail_last_cap is None
  assert vtsc._map_holdover_cap is None
  assert vtsc._map_holdover_candidate is None
  assert vtsc._v_turn_hold_until == 0.0
  assert vtsc._v_turn_hold_min == pytest.approx(vtc_mod.INF_SPEED)
  assert vtsc._v_turn_release_shape_active is False
  assert vtsc.v_turn > 12.0
