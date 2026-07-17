import numpy as np
import time
import math
import json
import os
import hashlib
import re
from dataclasses import dataclass
from enum import IntEnum

from cereal import custom, log
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.numpy_fast import clip
from opendbc.car.common.conversions import Conversions as CV
from openpilot.selfdrive.controls.lib.longitudinal_response_model import CruiseResponseModel
try:
  from opendbc.car.vehicle_model import VehicleModel
except Exception:
  VehicleModel = None
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.modeld.constants import ModelConstants
from .planner_lag_debug import (
  SPAN_MAP_TAIL_CAP,
  SPAN_PREVIEW_BRANCH_STUBS,
  SPAN_PREVIEW_FROM_MAP,
  end_span,
  start_span,
)
from .vision_turn_params import update_vtsc_params
from .vtsc_map_strategy import (
  MAP_STRATEGY_ADVISORY,
  DEFAULT_MAP_STRATEGY,
  DEFAULT_WINDING_BEHAVIOR_PROFILE,
  MAP_STRATEGY_STRATEGIC,
  MapCapCandidate,
  MapStrategyState,
  WindingBehaviorProfile,
  WindingRoadContext,
  classify_winding_road_context,
  compute_map_cap_candidate,
  effective_curve_phase_offset_s,
  evaluate_map_strategy,
  normalize_map_strategy,
  resolve_winding_behavior_profile,
)
try:
  from .vtsc_curve_tuning import Q_CURVE_ENABLED, Q_CURVE_POINTS
except Exception:
  Q_CURVE_ENABLED = False
  Q_CURVE_POINTS = []

VisionTurnControllerState = custom.LongitudinalPlanSP.VisionTurnSpeedControl.VisionTurnSpeedControlState
LaneChangeState = log.LaneChangeState

N_POINTS = int(min(33, len(ModelConstants.T_IDXS)))  # Use available trajectory points

# ===== Freeway Fail-Open Guard Tunables =====
# If path is straight, visibility is long, and confidence is good, ignore occlusion effects.
FREEWAY_CURV_EPS = 1e-5       # effectively straight (1/m)
FREEWAY_MIN_VISIBLE_M = 120.0 # visible horizon long enough (m)
FREEWAY_MIN_CONF = 0.60       # path/model confidence threshold

# ===== Freeway Cap Hold (Planner Response Latency) =====
# At freeway speeds, the VTSC cap can briefly dip for a single model frame as curvature predictions
# fluctuate. The longitudinal planner/MPC typically cannot respond meaningfully to these sub-0.5s
# pulses, which presents as "VTSC is late to slow". Hold material cap reductions briefly so the
# planner sees a stable target and begins braking sooner.
VTURN_HOLD_MIN_V_MPS = 27.0    # only engage hold above ~60 mph
VTURN_HOLD_DELTA_MPS = 1.0    # only hold when cap reduces cruise by ≥ this
VTURN_HOLD_S = 1.2            # tuned from rlogs: typical planner response ≈ 0.9–1.2s
# Under degraded vision we still want a hold, but make it shorter to avoid "sticky" slowdowns after
# a curve ends (especially when the UI state machine keeps predicted curvature slightly non-zero).
VTURN_HOLD_S_OCCLUDED = 0.85
# Shape the final VTSC cap release after a real curve-limited phase so longitudinal MPC does not
# snap from a low curve cap straight back to set speed in one frame.
VTURN_RELEASE_SHAPE_ENTRY_DELTA_MPS = 0.75
VTURN_RELEASE_SHAPE_HANDOFF_MARGIN_MPS = 0.35
VTURN_LOCAL_CURVATURE_CORROBORATION_KAPPA = 0.001
# Minimum predicted lateral acceleration (m/s^2) to treat as real turn evidence for hold gating.
_ENTERING_PRED_LAT_ACC_TH = 1.3

# Global model-horizon phase advance (seconds). Shifts both braking onset and post-apex
# release earlier to compensate planner/actuation latency.
VTSC_TRAJECTORY_PHASE_ADVANCE_S = 1.0

# ===== Feature Flags & Thresholds =====
# Use a large sentinel for "no cap" speed contributions when disabling a channel
INF_SPEED = 1e9

# ===== Steering-curvature fallback =====
# If the model curvature stays near-flat while the steering input indicates a real curve,
# use steering-derived curvature as a last-resort signal to avoid entering a curve at cruise.
STEER_CURVATURE_FALLBACK_MODEL_KAPPA_MAX = 0.003  # 1/m: model says "straight-ish"
STEER_CURVATURE_FALLBACK_MIN_KAPPA = 0.003        # 1/m: car is actually turning
STEER_CURVATURE_FALLBACK_MIN_V_MPS = 13.0         # only consider at ~29 mph+
# During a confirmed lane change, the model horizon can arc across lanes strongly enough to look
# like road curvature. If steering-derived curvature is still below this threshold, suppress the
# model-only curve signal and let map/steering evidence decide whether VTSC should slow.
LANE_CHANGE_CURVATURE_SUPPRESS_STEER_KAPPA_MAX = 0.003
LANE_CHANGE_CURVATURE_SUPPRESS_MIN_V_MPS = 1.0
# When pre-apex unwind starts before the geometric apex, begin with a gentler release slew and
# ramp up as the measured lateral acceleration closes in on the predicted apex load.
PRE_APEX_RELEASE_SLEW_MIN_SCALE = 0.55
# Raw strategic MapCurvatures can occasionally latch onto a ramp/branch before live road geometry
# resolves the mainline. During a blinker/lane-change context, only trust that raw strategic floor
# when the local curve evidence is in the same ballpark.
AMBIGUOUS_RAW_MAP_ANCHOR_MIN_DIST_M = 45.0
AMBIGUOUS_RAW_MAP_ANCHOR_MIN_KAPPA = 0.010
AMBIGUOUS_RAW_MAP_CURVATURE_RATIO = 3.0
# When live mainline geometry is valid and map says "already in the curve" but local vision/steer
# evidence remains much milder for a sustained window, suppress the strategic map floor entirely.
VISIBLE_MAINLINE_RELAX_DWELL_S = 0.45
VISIBLE_MAINLINE_RELAX_CURVE_START_MAX_M = 8.0
VISIBLE_MAINLINE_RELAX_MIN_ANCHOR_KAPPA = 0.003
VISIBLE_MAINLINE_RELAX_CURVATURE_RATIO = 3.0
VISIBLE_MAINLINE_RELAX_MIN_LOCAL_SPEED_DELTA_MPS = 4.5
VISIBLE_MAINLINE_RELAX_MIN_MAP_LATACC_NOW = 2.5
VISIBLE_MAINLINE_RELAX_MAX_LOCAL_LATACC_RATIO = 0.45

# ===== Severe-confidence overshoot conservatism =====
# If lane-line confidence is extremely low, the model often "discovers" tight off-ramp curvature late.
# Make overshoot detection slightly more conservative so VTSC begins slowing earlier in SEVERE/LOST,
# without changing the baseline curvature→speed mapping used in good visibility.
SEVERE_OVERSHOOT_SPEED_SCALE_MIN = 0.90  # multiplicative on safe speeds (lower => more conservative)

# Hidden-turn early deceleration feature flag (disabled fully per request)
HIDDEN_TURN_ENABLED = False

# The FOV/degraded-confidence occlusion state machine has repeatedly produced
# worse on-road behavior than simply trusting the visible path plus map preview.
# Keep the code inert so it cannot throttle acceleration or handoff timing.
VTSC_OCCLUSION_ENABLED = False

# Highway override threshold: start any bypass/relax behavior at 55 mph
HIGHWAY_MIN_MPH = 55.0
HIGHWAY_MIN_MPS = float(HIGHWAY_MIN_MPH * CV.MPH_TO_MS)

# ===== Low-speed occlusion margin relax =====
# At very low speeds on effectively-straight roads, VTSC's occlusion math can trip "negative margin"
# and enforce a decel/hold that feels like a crawl. Allow a small override in this regime.
LOW_SPEED_MARGIN_MAX_V_MPS = 12.5      # taper ends ≈28 mph (dominates town speeds)
LOW_SPEED_MARGIN_CURV_THRESH = 3.5e-4  # below this treat as effectively straight

# Lead-bypass headway floor: avoid inflated headway at crawl speeds behind a lead
OCCL_BYPASS_HEADWAY_V_FLOOR_MPS = 5.0  # ~11 mph
# Lead-bypass low-speed close-lead fallback
OCCL_BYPASS_LOW_SPEED_V_MPS = 7.0      # ~16 mph
OCCL_BYPASS_LEAD_D_REL_MAX_M = 27.0    # ~89 ft

# ===== PSI / Occlusion arbitration tunables (defaults; overridden via Params) =====
# PSI gate to qualify occlusion influence during FOV occlusion
PSI_THRESH_RAD = 0.020     # default gate open threshold (radians)
PSI_HYST_RAD  = 0.005      # hysteresis
# Double-cap guard: if pre-cap target already ≤ occl vmin + eps, don't re-apply occlusion cap
DOUBLE_CAP_EPS_MPS = 0.30
# fov_exit recovery when confidence is near-zero and psi gate is closed
OCCL_CONF_FLOOR    = 0.05
FOV_EXIT_RELAX_S   = 0.60
OCCL_VMIN_NUDGE_MPS = 0.50

# ===== Map lookahead helpers =====
EARTH_R_M = 6371007.2
MAP_WHOLE_CURVE_ESTIMATOR_VERSION = "whole-curve-v3"
MAP_WHOLE_CURVE_PROFILE_MAX_AGE_S = 3.0
MAP_WHOLE_CURVE_PROFILE_FUTURE_TOLERANCE_S = 1.0
MAP_WHOLE_CURVE_PROFILE_MAX_BYTES = 512 * 1024
MAP_WHOLE_CURVE_PROFILE_MAX_POINTS = 512
MAP_WHOLE_CURVE_PROFILE_MAX_EVENTS = 128
# 1.2 km forward coverage plus retained predecessor/rollover context and a
# small endpoint overshoot from strict 5 m resampling.
MAP_WHOLE_CURVE_PROFILE_MAX_DISTANCE_M = 1600.0
MAP_WHOLE_CURVE_PROFILE_MAX_EGO_DISTANCE_M = 75.0
MAP_WHOLE_CURVE_PROFILE_HORIZON_M = 1200.0
_MAP_WHOLE_CURVE_EVENT_ID_RE = re.compile(r"(?:[0-9a-f]{20}-[ab])?")
_MAP_WHOLE_CURVE_FLAG_RE = re.compile(r"[A-Za-z0-9_.:-]{1,96}")


@dataclass(frozen=True)
class MapWholeCurvePoint:
  latitude: float
  longitude: float
  distance_m: float
  curvature: float
  curvature_coefficient: float
  base_safe_speed_mps: float
  event_id: str = ""
  confidence: float | str | None = None
  flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class MapWholeCurveProfile:
  generated_at_unix_ms: float
  route_fingerprint: str
  sigmoid_hash: str
  generation: int
  points: tuple[MapWholeCurvePoint, ...]
  events: tuple[dict, ...]


def _round_half_away_from_zero_scaled(value: float, scale: float) -> int:
  """Cross-language integer quantization used by the whole-curve fingerprint."""
  value_f = float(value)
  if not math.isfinite(value_f):
    raise ValueError("non-finite fingerprint value")
  magnitude = int(math.floor(abs(value_f) * float(scale) + 0.5))
  return -magnitude if value_f < 0.0 else magnitude


def _compute_map_whole_curve_route_fingerprint(generation: int,
                                                sigmoid_hash: str,
                                                points: tuple[MapWholeCurvePoint, ...] | list[MapWholeCurvePoint]) -> str:
  """Hash the exact ordered route/control profile using the Go/Swift v3 contract."""
  digest = hashlib.sha256()
  digest.update(f"MapWholeCurveProfile|{MAP_WHOLE_CURVE_ESTIMATOR_VERSION}|{int(generation)}|{sigmoid_hash}\n".encode("utf-8"))
  for point in points:
    lat_e7 = _round_half_away_from_zero_scaled(point.latitude, 1e7)
    lon_e7 = _round_half_away_from_zero_scaled(point.longitude, 1e7)
    distance_mm = _round_half_away_from_zero_scaled(point.distance_m, 1e3)
    curvature_e9 = _round_half_away_from_zero_scaled(point.curvature, 1e9)
    coefficient_e6 = _round_half_away_from_zero_scaled(point.curvature_coefficient, 1e6)
    base_speed_e6 = _round_half_away_from_zero_scaled(point.base_safe_speed_mps, 1e6)
    digest.update(
      f"{lat_e7},{lon_e7},{distance_mm},{curvature_e9},{coefficient_e6},{base_speed_e6},{point.event_id}\n".encode("utf-8")
    )
  return digest.hexdigest()

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
  c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
  return EARTH_R_M * c

def _xy_from_latlon_m(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
  """Equirectangular approximation in meters in an (east, north) local frame around (lat0, lon0)."""
  dlat = math.radians(lat - lat0)
  dlon = math.radians(lon - lon0)
  x_east = EARTH_R_M * dlon * math.cos(math.radians(lat0))
  y_north = EARTH_R_M * dlat
  return float(x_east), float(y_north)


def _bearing_deg_to_unit_en(bearing_deg: float) -> tuple[float, float] | None:
  if not math.isfinite(bearing_deg):
    return None
  heading_rad = math.radians(bearing_deg)
  fx = math.sin(heading_rad)
  fy = math.cos(heading_rad)
  norm = math.hypot(fx, fy)
  if not (norm > 1e-6 and math.isfinite(norm)):
    return None
  return (float(fx / norm), float(fy / norm))


def _normalize_en_vec(x: float, y: float) -> tuple[float, float] | None:
  norm = math.hypot(float(x), float(y))
  if not (norm > 1e-6 and math.isfinite(norm)):
    return None
  return (float(x / norm), float(y / norm))


def _sm_get_optional(sm, key: str):
  try:
    valid = getattr(sm, 'valid', None)
    if isinstance(valid, dict) and key in valid and not bool(valid.get(key, False)):
      return None
  except Exception:
    pass

  try:
    return sm[key]
  except Exception:
    pass

  data = getattr(sm, '_data', None)
  if isinstance(data, dict):
    return data.get(key, None)
  return getattr(sm, key, None)


def _extract_centerline_coords(seg) -> list[tuple[float, float, float]]:
  centerline = getattr(seg, 'centerline', None)
  if centerline is None:
    return []

  coords: list[tuple[float, float, float]] = []
  needs_synth_s = False
  for coord in centerline:
    try:
      lat = float(getattr(coord, 'latitude'))
      lon = float(getattr(coord, 'longitude'))
    except Exception:
      continue
    if not (math.isfinite(lat) and math.isfinite(lon)):
      continue
    try:
      s = float(getattr(coord, 'distanceFromStart'))
    except Exception:
      s = float('nan')
    if not math.isfinite(s) or (coords and s < coords[-1][2]):
      needs_synth_s = True
    coords.append((lat, lon, s))

  if len(coords) < 2:
    return []

  if needs_synth_s:
    rebuilt = [(coords[0][0], coords[0][1], 0.0)]
    cumulative = 0.0
    for prev, cur in zip(coords, coords[1:], strict=False):
      cumulative += _haversine_m(float(prev[0]), float(prev[1]), float(cur[0]), float(cur[1]))
      rebuilt.append((float(cur[0]), float(cur[1]), float(cumulative)))
    coords = rebuilt

  return coords


def _project_latlon_to_centerline(lat: float, lon: float, coords: list[tuple[float, float, float]]) -> dict | None:
  best: dict | None = None
  for p0, p1 in zip(coords, coords[1:], strict=False):
    ax, ay = _xy_from_latlon_m(float(p0[0]), float(p0[1]), float(lat), float(lon))
    bx, by = _xy_from_latlon_m(float(p1[0]), float(p1[1]), float(lat), float(lon))
    dx = bx - ax
    dy = by - ay
    seg_len2 = dx * dx + dy * dy
    if not (seg_len2 > 1e-6 and math.isfinite(seg_len2)):
      continue

    t = max(0.0, min(1.0, -((ax * dx) + (ay * dy)) / seg_len2))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    d = math.hypot(proj_x, proj_y)
    tangent_en = _normalize_en_vec(dx, dy)
    if tangent_en is None:
      continue

    s = float(p0[2]) + (float(p1[2]) - float(p0[2])) * t
    cand = {
      'd': float(d),
      's': float(s),
      'proj_lat': float(p0[0] + (float(p1[0]) - float(p0[0])) * t),
      'proj_lon': float(p0[1] + (float(p1[1]) - float(p0[1])) * t),
      'tangent_en': tangent_en,
    }
    if best is None or cand['d'] < best['d']:
      best = cand
  return best


def _interp_xy(p0: tuple[float, float], p1: tuple[float, float], t: float) -> tuple[float, float]:
  return (
    float(p0[0] + (p1[0] - p0[0]) * t),
    float(p0[1] + (p1[1] - p0[1]) * t),
  )


def _append_point_if_distinct(out: list[tuple[float, float]], pt: tuple[float, float], eps: float = 1e-3) -> None:
  if not out:
    out.append((float(pt[0]), float(pt[1])))
    return
  last = out[-1]
  if math.hypot(float(pt[0]) - float(last[0]), float(pt[1]) - float(last[1])) > eps:
    out.append((float(pt[0]), float(pt[1])))


def _clip_polyline_x(points: list[tuple[float, float]], x_min: float, x_max: float) -> list[tuple[float, float]]:
  if len(points) < 2 or not math.isfinite(x_min) or not math.isfinite(x_max) or x_max <= x_min:
    return list(points)

  clipped: list[tuple[float, float]] = []
  prev = (float(points[0][0]), float(points[0][1]))
  if x_min <= prev[0] <= x_max:
    clipped.append(prev)

  for curr_raw in points[1:]:
    curr = (float(curr_raw[0]), float(curr_raw[1]))
    x0 = prev[0]
    x1 = curr[0]
    dx = x1 - x0

    if abs(dx) > 1e-6:
      for bound in (x_min, x_max):
        crosses = (x0 < bound <= x1) or (x1 <= bound < x0)
        if crosses:
          t = max(0.0, min(1.0, (bound - x0) / dx))
          _append_point_if_distinct(clipped, _interp_xy(prev, curr, t))

    if x_min <= x1 <= x_max:
      _append_point_if_distinct(clipped, curr)

    if (x0 <= x_max < x1) or (x1 < x_min <= x0):
      break
    prev = curr

  return clipped


def _densify_polyline(points: list[tuple[float, float]], max_step_m: float) -> list[tuple[float, float]]:
  if len(points) < 2 or not (max_step_m > 0.1 and math.isfinite(max_step_m)):
    return list(points)

  dense = [(float(points[0][0]), float(points[0][1]))]
  for p0_raw, p1_raw in zip(points, points[1:], strict=False):
    p0 = (float(p0_raw[0]), float(p0_raw[1]))
    p1 = (float(p1_raw[0]), float(p1_raw[1]))
    seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    steps = max(1, int(math.ceil(seg_len / max_step_m)))
    for step_idx in range(1, steps + 1):
      dense.append(_interp_xy(p0, p1, float(step_idx) / float(steps)))
  return dense


def _chaikin_smooth_polyline(points: list[tuple[float, float]], passes: int = 1) -> list[tuple[float, float]]:
  smoothed = [(float(x), float(y)) for x, y in points]
  for _ in range(max(0, int(passes))):
    if len(smoothed) < 3:
      break
    nxt = [smoothed[0]]
    for p0, p1 in zip(smoothed, smoothed[1:], strict=False):
      q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
      r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
      nxt.append((float(q[0]), float(q[1])))
      nxt.append((float(r[0]), float(r[1])))
    nxt.append(smoothed[-1])
    smoothed = nxt
  return smoothed


def _resample_polyline(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
  if not points or count <= 0:
    return []
  if len(points) == 1 or count == 1:
    return [(float(points[0][0]), float(points[0][1]))]

  cumulative = [0.0]
  for p0, p1 in zip(points, points[1:], strict=False):
    cumulative.append(cumulative[-1] + math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))

  total_len = cumulative[-1]
  if not (total_len > 1e-6 and math.isfinite(total_len)):
    return [(float(points[0][0]), float(points[0][1]))]

  out: list[tuple[float, float]] = []
  seg_idx = 0
  for i in range(count):
    target = total_len * (float(i) / float(max(1, count - 1)))
    while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < target:
      seg_idx += 1
    if seg_idx + 1 >= len(points):
      out.append((float(points[-1][0]), float(points[-1][1])))
      continue
    span = cumulative[seg_idx + 1] - cumulative[seg_idx]
    t = 0.0 if span < 1e-6 else (target - cumulative[seg_idx]) / span
    out.append(_interp_xy(points[seg_idx], points[seg_idx + 1], t))
  return out


def _slice_polyline_by_s(points: list[tuple[float, float]], s_vals: list[float], s_min: float, s_max: float) -> list[tuple[float, float]]:
  if len(points) < 2 or len(points) != len(s_vals):
    return list(points)
  if not (math.isfinite(s_min) and math.isfinite(s_max) and s_max > s_min):
    return list(points)

  clipped: list[tuple[float, float]] = []
  prev = (float(points[0][0]), float(points[0][1]))
  prev_s = float(s_vals[0])
  if s_min <= prev_s <= s_max:
    _append_point_if_distinct(clipped, prev)

  for curr_raw, curr_s_raw in zip(points[1:], s_vals[1:], strict=False):
    curr = (float(curr_raw[0]), float(curr_raw[1]))
    curr_s = float(curr_s_raw)
    ds = curr_s - prev_s
    if not (ds > 1e-6 and math.isfinite(ds)):
      prev = curr
      prev_s = curr_s
      continue

    for bound in (s_min, s_max):
      if prev_s < bound <= curr_s:
        t = max(0.0, min(1.0, (bound - prev_s) / ds))
        _append_point_if_distinct(clipped, _interp_xy(prev, curr, t))

    if s_min <= curr_s <= s_max:
      _append_point_if_distinct(clipped, curr)

    if prev_s <= s_max < curr_s:
      break

    prev = curr
    prev_s = curr_s

  return clipped


def _curve_direction_from_points(points: list[tuple[float, float]]) -> int:
  cross_sum = 0.0
  for p0, p1, p2 in zip(points, points[1:], points[2:], strict=False):
    v1x = float(p1[0]) - float(p0[0])
    v1y = float(p1[1]) - float(p0[1])
    v2x = float(p2[0]) - float(p1[0])
    v2y = float(p2[1]) - float(p1[1])
    cross_sum += (v1x * v2y - v1y * v2x)

  if cross_sum > 1e-3:
    return 1
  if cross_sum < -1e-3:
    return 2
  return 0


def _normalize_polyline_to_entry_frame(points: list[tuple[float, float]], *, densify_step_m: float = 3.0,
                                       smooth_passes: int = 4, resample_count: int = 24) -> list[tuple[float, float]]:
  base: list[tuple[float, float]] = []
  for x_raw, y_raw in points:
    x = float(x_raw)
    y = float(y_raw)
    if math.isfinite(x) and math.isfinite(y):
      base.append((x, y))
  if len(base) < 2:
    return []

  origin = base[0]
  tangent = None
  for cand in base[1:]:
    dx = float(cand[0]) - float(origin[0])
    dy = float(cand[1]) - float(origin[1])
    norm = math.hypot(dx, dy)
    if norm > 0.5 and math.isfinite(norm):
      tangent = (float(dx / norm), float(dy / norm))
      break
  if tangent is None:
    return []

  tx, ty = tangent
  lx, ly = -ty, tx
  local: list[tuple[float, float]] = []
  for pt in base:
    dx = float(pt[0]) - float(origin[0])
    dy = float(pt[1]) - float(origin[1])
    local.append((
      float(dx * tx + dy * ty),
      float(dx * lx + dy * ly),
    ))

  local = _densify_polyline(local, max(0.5, float(densify_step_m)))
  local = _chaikin_smooth_polyline(local, passes=max(0, int(smooth_passes)))
  local = _resample_polyline(local, max(8, int(resample_count)))
  if len(local) < 2:
    return []

  normalized: list[tuple[float, float]] = [(0.0, 0.0)]
  prev_x = 0.0
  for idx, (x_raw, y_raw) in enumerate(local[1:], start=1):
    x = max(prev_x, float(x_raw))
    y = float(y_raw)
    pt = (x, y)
    if math.hypot(pt[0] - normalized[-1][0], pt[1] - normalized[-1][1]) > 1e-3:
      normalized.append(pt)
      prev_x = x
    else:
      prev_x = max(prev_x, x)

  return normalized if len(normalized) >= 2 else []

# ===== ADAPTIVE DECELERATION SYSTEM =====
# Physics-based deceleration management for vision update lag scenarios
# Goal: Target comfort rates, but escalate to minimum decel/jerk needed to reach target speed at curve

# Comfort deceleration limits (m/s²) - primary targets
COMFORT_DECEL_LIMIT = -1.47  # -0.15g - comfortable deceleration
COMFORT_JERK_LIMIT = -2.0    # m/s³ - comfortable jerk

# Safety limits for adaptive escalation (m/s²)
MAX_ADAPTIVE_DECEL = -6.0    # System maximum deceleration
MAX_ADAPTIVE_JERK = -6.0     # System maximum jerk

# Default noise filtering parameters
DEFAULT_FILTER_ALPHA = 0.3      # EMA filter coefficient (0.1-0.9)
DEFAULT_HYSTERESIS_THRESHOLD = 0.15  # Hysteresis threshold (0.1-0.5)
DEFAULT_SAFETY_BIAS = 0.1        # Safety bias factor (0.0-0.5)

# ===== VISION OCCLUSION HANDLING (SIMPLIFIED) =====
# Smoothed confidence with a small hysteresis band; hold last curvature during occlusion
CONF_ALPHA = 0.28       # Faster EMA to track recovery
CONF_GOOD_TH = 0.70     # Threshold to (re)enter good vision
CONF_BAD_TH = 0.65      # Threshold to enter occlusion
# Public hysteresis thresholds (compatibility for tests)
CONFIDENCE_ENTER_PARTIAL = CONF_BAD_TH  # align status with control gate
CONFIDENCE_EXIT_TO_FULL = CONF_GOOD_TH  # align status with control gate
CONFIDENCE_ENTER_SEVERE = 0.45
CONFIDENCE_EXIT_TO_PARTIAL = 0.55

class VisionStatus(IntEnum):
    FULL_VISIBILITY = 0
    PARTIAL_OCCLUSION = 1
    SEVERE_OCCLUSION = 2
    VISION_LOST = 3


@dataclass
class VisionOcclusionState:
    """Smoothed confidence with monotonic-decay occlusion handling.

    Behavior while occluded (vision_good == False):
    - Pre-apex trend (tightening): grow curvature conservatively with distance using gamma_per_m.
    - Post-apex trend (easing): bounded decay toward a fraction of entry curvature to avoid crawl.
    Always maintains monotonic speed by blocking positive acceleration (enforced upstream).
    """
    last_valid_curvature: float = 0.0
    smoothed_confidence: float = 1.0
    vision_good: bool = True
    # Compatibility fields expected by tests
    vision_status: VisionStatus = VisionStatus.FULL_VISIBILITY
    confidence_decay_factor: float = 1.0
    extrapolated_curvature: float = 0.0
    good_vision_frames: int = 0
    occlusion_start_time: float = 0.0
    updated_once: bool = False
    alpha: float = CONF_ALPHA
    good_threshold: float = CONF_GOOD_TH
    bad_threshold: float = CONF_BAD_TH
    # Monotonic occlusion model state
    prev_curvature_good: float = 0.0
    occluded_since_time: float = 0.0
    reacquired_at: float = 0.0
    tail_started: bool = False
    tail_start_time: float = 0.0
    entry_curvature: float = 0.0
    est_curvature: float = 0.0
    trend_sign: int = 0
    last_time: float = 0.0
    distance_since_m: float = 0.0
    mode_monotonic: bool = True
    gamma_per_m: float = 2.5e-4
    lat_jerk_cap: float = 2.0
    vis_horizon_s: float = 1.2
    envelope_horizon_s: float = 1.0
    # Two-stage decay
    decay_tau_fast_s: float = 1.2
    decay_tau_slow_s: float = 2.0
    min_frac_initial: float = 0.6
    min_frac: float = 0.20
    # Trend multipliers (decreasing multiplier set to 1.0 per design)
    increasing_trend_mul: float = 1.0
    decreasing_trend_mul: float = 1.0
    # Dwell timers
    enter_dwell_s: float = 0.20
    exit_dwell_s: float = 0.10
    below_bad_time_s: float = 0.0
    above_good_time_s: float = 0.0
    # Last-known-good (LKG) anchor for occlusion re-anchoring
    _lkg_kappa: float = 0.0
    _lkg_time: float = 0.0
    _lkg_speed: float = 0.0
    _lkg_ramp_s: float = 0.9  # ramp time to allow occluded estimate to exceed LKG

    def update(self, current_curvature: float, vision_confidence: float, v_ego_or_tm, tm=None):
        # Support both 3-arg (tm) and 4-arg (v_ego, tm) signatures
        if tm is None:
            tm = float(v_ego_or_tm)
            v_ego = 0.0
        else:
            v_ego = float(v_ego_or_tm)
        # EMA smoothing of confidence
        self.smoothed_confidence = (1.0 - self.alpha) * self.smoothed_confidence + self.alpha * vision_confidence

        # Track time and distance progression
        dt = 0.0 if self.last_time == 0.0 else max(0.0, tm - self.last_time)
        self.last_time = tm
        # Dwell accumulation based on tri-state thresholds
        if self.smoothed_confidence < self.bad_threshold:
            self.below_bad_time_s += dt
            self.above_good_time_s = 0.0
        elif self.smoothed_confidence > self.good_threshold:
            self.above_good_time_s += dt
            self.below_bad_time_s = 0.0
        else:
            # Borderline band: reset both dwell timers
            self.below_bad_time_s = 0.0
            self.above_good_time_s = 0.0

        # State transitions with dwell
        if self.vision_good:
            # Enter occlusion only after dwell below bad
            if self.below_bad_time_s >= self.enter_dwell_s:
                self.vision_good = False
                self.occluded_since_time = tm
                self.occlusion_start_time = tm
                self.distance_since_m = 0.0
                self.tail_started = False
                self.tail_start_time = 0.0
                base = self.last_valid_curvature if self.updated_once else current_curvature
                self.entry_curvature = max(0.0, float(base))
                self.est_curvature = self.entry_curvature
                dcur = current_curvature - self.prev_curvature_good
                self.trend_sign = 1 if dcur > 0.0 else (-1 if dcur < 0.0 else 0)
                # reset enter dwell
                self.below_bad_time_s = 0.0
            else:
                # Vision is good but not entering occlusion: defer last_valid_curvature updates
                # to the compatibility section (requires 3 strong-good frames)
                self.prev_curvature_good = current_curvature
                self.updated_once = True
        else:
            # Exit occlusion only after dwell above good
            if self.above_good_time_s >= self.exit_dwell_s:
                self.vision_good = True
                self.last_valid_curvature = current_curvature
                # Reacquired: mark timestamp for downstream smoothing aids
                self.reacquired_at = tm
                self.prev_curvature_good = current_curvature
                self.updated_once = True
                self.occluded_since_time = 0.0
                self.distance_since_m = 0.0
                self.tail_started = False
                self.tail_start_time = 0.0
                self.entry_curvature = 0.0
                self.est_curvature = 0.0
                self.trend_sign = 0
                self.above_good_time_s = 0.0
            else:
                # Remain occluded: update distance and estimate curvature for the tail only
                self.distance_since_m += v_ego * dt
                if not self.mode_monotonic:
                    self.est_curvature = self.last_valid_curvature
                else:
                    elapsed = max(0.0, tm - self.occluded_since_time)
                    # Visible horizon in meters
                    s_vis = max(0.0, self.vis_horizon_s * max(0.0, v_ego))
                    s_tail_raw = max(0.0, self.distance_since_m - s_vis)
                    # Tail growth window timing
                    if (s_tail_raw > 1e-3) and (not self.tail_started):
                        self.tail_started = True
                        self.tail_start_time = tm
                    tail_elapsed = (tm - self.tail_start_time) if self.tail_started else 0.0
                    # Curvature- and speed-aware tail allowance
                    k_now_for_window = max(0.0, max(self.entry_curvature, self.est_curvature))
                    # Tail window selection: prioritize safety but ensure sufficient growth at moderate speeds
                    # Baseline: very short for sweepers, long for tight curves
                    if k_now_for_window <= 0.0035:
                        t_allow = 0.10
                    else:
                        t_allow = 1.20
                    # Production alignment: at mountain speeds (≤ ~30 m/s), allow a larger tail window
                    # so curvature growth can reflect an upcoming bend even when entry curvature was near-zero.
                    if v_ego <= 30.0:
                        t_allow = max(t_allow, 0.80)
                    # Clamp to configured envelope horizon
                    t_allow = max(0.10, min(t_allow, self.envelope_horizon_s))
                    s_tail_allow = max(0.0, v_ego) * t_allow
                    s_tail = min(s_tail_raw, s_tail_allow)

                    if self.trend_sign >= 0:
                        # Growth only beyond visible horizon; allow conservative growth with jerk-capped gamma
                        if (s_tail > 0.0) and (tail_elapsed <= self.envelope_horizon_s):
                            # Speed-based cap (piecewise, interpolated) + jerk cap
                            v_cap_speed = 22.0  # soften jerk cap below ~49 mph to preserve mountain decel
                            eff_v = min(max(0.0, v_ego), v_cap_speed)
                            gamma_cap_jerk = getattr(self, 'lat_jerk_cap', 2.0) / max(eff_v**3, 1e-3)
                            # Piecewise speed cap table (m/s -> gamma cap)
                            sp = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0]
                            gp = [8e-4,6e-4,5e-4,4e-4,3e-4,1.8e-4,1.2e-4,1.0e-4]
                            vv = max(0.0, v_ego)
                            if vv <= sp[0]:
                                gamma_cap_speed = gp[0]
                            elif vv >= sp[-1]:
                                gamma_cap_speed = gp[-1]
                            else:
                                for i in range(len(sp)-1):
                                    if sp[i] <= vv <= sp[i+1]:
                                        t = (vv - sp[i]) / max(1e-6, (sp[i+1] - sp[i]))
                                        gamma_cap_speed = gp[i] + (gp[i+1] - gp[i]) * t
                                        break
                            # Allow growth up to configured gamma with both speed and jerk caps
                            gamma_eff = min(self.gamma_per_m, gamma_cap_speed, gamma_cap_jerk)
                            # Optional trend multipliers (default 1.0)
                            try:
                                inc_mul = float(getattr(self, 'increasing_trend_mul', 1.0))
                            except Exception:
                                inc_mul = 1.0
                            self.est_curvature = max(0.0, self.entry_curvature + inc_mul * (gamma_eff * s_tail))
                        else:
                            # Freeze growth beyond tail window
                            self.est_curvature = max(0.0, self.est_curvature)
                    else:
                        # Two-stage decay, with time-to-floor behavior
                        if self.entry_curvature <= 0.0:
                            self.est_curvature = 0.0
                        else:
                            # Floor ramps down after 0.7s occluded
                            floor_frac = self.min_frac if elapsed >= 0.7 else self.min_frac_initial
                            floor_val = floor_frac * self.entry_curvature
                            tau = self.decay_tau_fast_s if elapsed <= 0.8 else self.decay_tau_slow_s
                            decay = math.exp(-elapsed / max(1e-3, tau))
                            try:
                                dec_mul = float(getattr(self, 'decreasing_trend_mul', 1.0))
                            except Exception:
                                dec_mul = 1.0
                            self.est_curvature = max(floor_val, (self.entry_curvature * decay) * dec_mul)


        # ===== Compatibility: expose VisionStatus with hysteresis thresholds =====
        c = float(vision_confidence)
        prev = self.vision_status
        # Immediate escalate to SEVERE on very low confidence
        if c < CONFIDENCE_ENTER_SEVERE:
            self.vision_status = VisionStatus.SEVERE_OCCLUSION
        else:
            if prev == VisionStatus.FULL_VISIBILITY:
                # Leave FULL only below 0.75
                self.vision_status = VisionStatus.PARTIAL_OCCLUSION if c < CONFIDENCE_ENTER_PARTIAL else VisionStatus.FULL_VISIBILITY
            elif prev == VisionStatus.PARTIAL_OCCLUSION:
                # Return to FULL at 0.85; otherwise remain PARTIAL
                if c >= CONFIDENCE_EXIT_TO_FULL:
                    self.vision_status = VisionStatus.FULL_VISIBILITY
                else:
                    self.vision_status = VisionStatus.PARTIAL_OCCLUSION
            elif prev == VisionStatus.SEVERE_OCCLUSION:
                # Recover to PARTIAL at 0.55
                self.vision_status = VisionStatus.PARTIAL_OCCLUSION if c >= CONFIDENCE_EXIT_TO_PARTIAL else VisionStatus.SEVERE_OCCLUSION
            else:
                # VISION_LOST -> treat as severe until recovery
                self.vision_status = VisionStatus.PARTIAL_OCCLUSION if c >= CONFIDENCE_EXIT_TO_PARTIAL else VisionStatus.VISION_LOST
        # Track compat timestamps
        if self.vision_status != VisionStatus.FULL_VISIBILITY and self.occlusion_start_time == 0.0:
            self.occlusion_start_time = tm
        if self.vision_status == VisionStatus.FULL_VISIBILITY:
            self.reacquired_at = tm
# ===== Compatibility: last_valid_curvature update after 3 strong good frames =====
        # Count strong-good frames regardless of current status for compatibility
        if c >= CONFIDENCE_EXIT_TO_FULL:
            self.good_vision_frames = int(self.good_vision_frames) + 1
        else:
            self.good_vision_frames = 0
        if self.good_vision_frames >= 3:
            self.last_valid_curvature = float(current_curvature)
            self.updated_once = True

        # ===== Compatibility: no-decay fix =====
        self.confidence_decay_factor = 1.0
        # Vision-first extrapolated curvature with LKG anchoring during occlusion.
        # - When vision is good: update LKG anchor and use current curvature.
        # - When occluded: blend from LKG toward occluded estimate with a short time ramp.
        now_ts = float(tm)
        if self.vision_good:
            try:
                self._lkg_kappa = float(abs(current_curvature))
                self._lkg_time = now_ts
                # curvature_to_speed defined below; safe to call at runtime
                self._lkg_speed = float(curvature_to_speed(max(1e-8, self._lkg_kappa)))
            except Exception:
                pass
            self.extrapolated_curvature = float(current_curvature)
        else:
            try:
                k_occ = float(max(0.0, self.est_curvature))
                k_lkg = float(getattr(self, '_lkg_kappa', k_occ))
                age_s = max(0.0, now_ts - float(getattr(self, '_lkg_time', 0.0)))
                ramp_s = float(getattr(self, '_lkg_ramp_s', 1.5))
                alpha = min(1.0, age_s / max(0.1, ramp_s))
                k_blend = k_lkg + alpha * max(0.0, k_occ - k_lkg)
                self.extrapolated_curvature = float(k_blend)
            except Exception:
                self.extrapolated_curvature = float(max(0.0, self.est_curvature))

# ===== ORIGINAL PHYSICS-BASED VTSC CONSTANTS =====
_MIN_V = 2.24  # Do not operate under 5mph (was 5.6 m/s = 12.5mph)

_DEBUG = False

# Advanced vision-based functions extracted from chauffeur_vtsc.py

# Constants for advanced curvature-based speed calculation
MAX_SPEED_DEFAULT = 70.0  # m/s, fallback for straight roads (overridden by param)
SPEED_INCREASE_FACTOR = 1.0  # Global multiplier on target speeds (overridden by param)

# Schema-v1 MapPreCurveSpeeds were baked from one raw OSM vertex while the
# parallel MapCurvatures stream is a route-context, arc-weighted average.  Do
# not combine those two different curvature semantics.  Re-enable only after
# mapd publishes an estimator-versioned speed stream derived from the exact
# same curvature samples.
MAP_PRECURVE_SPEEDS_ESTIMATOR_ALIGNED = False

# Physics sigmoid tunables (overridden by params). These source defaults are
# the source-rounded values from the persisted whole-curve production tune.
PHYSICS_A = -3.514849
PHYSICS_B = -4147.818738
PHYSICS_C = 0.005966
PHYSICS_D = 4.538425
PHYSICS_MIN_LAT_ACCEL = 1.0236
PHYSICS_MAX_LAT_ACCEL = 4.5384

# Low-speed bias (applied as +Δ mph under a taper)
LOW_SPEED_BIAS_MPH = 5.0         # +speed boost at tight curves (tapers to 0 by END_MPH)
LOW_SPEED_BIAS_END_MPH = 55.0    # taper covers up to ~55 mph base speed (was 50.0)

# Low-speed envelope calibration. This keeps the learned adjustment bounded to the low-speed
# portion of the physics sigmoid, where steering authority is the real limit, without distorting
# the higher-speed envelope.
LOW_SPEED_CALIB_TARGET_START_MPH = 10.0
LOW_SPEED_CALIB_TARGET_FULL_MIN_MPH = 15.0
LOW_SPEED_CALIB_TARGET_END_MPH = 40.0
LOW_SPEED_CALIB_TARGET_HIGH_END_FADE_MPH = 5.0
LOW_SPEED_CALIB_MIN_CURVATURE = 0.0015
LOW_SPEED_CALIB_MIN_CAP_DELTA_MPS = 0.5
LOW_SPEED_CALIB_MAX_RELAX = 0.04
LOW_SPEED_CALIB_MAX_TIGHTEN = 0.08
LOW_SPEED_CALIB_RELAX_RATE_PER_S = 0.006
LOW_SPEED_CALIB_TIGHTEN_RATE_PER_S = 0.014
LOW_SPEED_CALIB_DECAY_RATE_PER_S = 0.005
LOW_SPEED_CALIB_HEADROOM_TAU_S = 0.75
LOW_SPEED_CALIB_RELAX_OUTPUT_MAX = 0.45
LOW_SPEED_CALIB_TIGHTEN_OUTPUT_MIN = 0.82
LOW_SPEED_CALIB_TRACKING_RELAX_RATIO_MAX = 0.10
LOW_SPEED_CALIB_TRACKING_TIGHTEN_RATIO_MIN = 0.22
LOW_SPEED_CALIB_ENABLE_TIGHTEN_EFFORT = False
LOW_SPEED_CALIB_ENABLE_TRACKING_TRIGGER = False
# Driver gas overrides above the current VTSC request are strong relax evidence. Learn that path
# faster than the passive low-speed headroom path, but only for the local curvature neighborhood.
LOW_SPEED_CALIB_OVERRIDE_MAX_RELAX = 0.12
LOW_SPEED_CALIB_OVERRIDE_RELAX_RATE_PER_S = 0.050
LOW_SPEED_CALIB_OVERRIDE_TIGHTEN_RATE_PER_S = 0.060
LOW_SPEED_CALIB_OVERRIDE_DIVERGENCE_DEADBAND_MPS = 0.05
LOW_SPEED_CALIB_OVERRIDE_DIVERGENCE_FULL_SCALE_MPS = 0.75
LOW_SPEED_CALIB_OVERRIDE_EMA_TAU_S = 0.15
LOW_SPEED_CALIB_OVERRIDE_PROFILE_MIN_CURVATURE = 1.0e-4
LOW_SPEED_CALIB_OVERRIDE_PROFILE_MAX_CURVATURE = 0.08
LOW_SPEED_CALIB_OVERRIDE_PROFILE_BINS = 49
LOW_SPEED_CALIB_OVERRIDE_PROFILE_SIGMA_LOG10 = 0.12
LOW_SPEED_CALIB_OVERRIDE_PROFILE_CUTOFF_SIGMA = 3.0
LOW_SPEED_CALIB_PERSIST_WRITE_S = 5.0
LOW_SPEED_CALIB_PERSIST_DELTA = 0.005
LOW_SPEED_CALIB_PARAM_SYNC_EPS = 1e-4
# When steering headroom (low effort, good tracking) is detected in the speed-tapered
# calibration band, also write a slow per-curvature relax to the override profile at
# 1/10 the gas override rate.  Gated by the speed-band taper so that going slow at a
# gentle freeway curve because of a speed limit doesn't contaminate the profile.
LOW_SPEED_CALIB_STEERING_HEADROOM_PER_CURVATURE_RATE_FACTOR = 0.10
# When a gas override fires and the experienced curvature differs from the map anchor
# curvature by more than this many sigma in log10 space, also update the secondary
# (experienced) bin at a reduced weight.
LOW_SPEED_CALIB_MAP_OVERRIDE_DUAL_BIN_MIN_SIGMA_DISTANCE = 1.0
LOW_SPEED_CALIB_MAP_OVERRIDE_DUAL_BIN_SECONDARY_WEIGHT = 0.5

# ===== Hidden-turn early deceleration trigger (occlusion-only, sub-65 mph) =====
# Allows jerk-limited early braking when a short-horizon physics deficit is provably large
# despite a transiently positive visible-margin condition.
# Align hidden-turn speed gate with highway threshold (~55 mph)
HIDDEN_TURN_V_MAX_MPS = HIGHWAY_MIN_MPS  # ~55 mph; above this we run pure physics
HIDDEN_TURN_T_H_S = 1.8        # short horizon (~40 m at 50 mph)
HIDDEN_TURN_DELTA_V_MPS = 2.0  # ~6 mph speed gap
HIDDEN_TURN_MIN_OCC_S = 0.30   # require persisting occlusion ≥ 600 ms
HIDDEN_TURN_AVAIL_SCALE = 0.50  # slight nudge for earlier activation
HIDDEN_TURN_PHASE_S = 2.0      # only within first ~2 s of occlusion
HIDDEN_TURN_HEADING_WIN_S = 1.2
HIDDEN_TURN_VIS_HEADING_MAX_RAD = math.radians(6.0)  # ~6°, "straight enough"

def _compute_runtime_sigmoid_hash() -> str:
  """Stable 12-hex digest of the current runtime sigmoid tuple. Mirrors
  SigmoidCfg.Hash() in mapd_repo/openpilot-mapd/sigmoid.go exactly so the
  on-device hash can be compared to the tile's baked sigmoidHash. Recomputed
  on every call (cheap; PHYSICS_* may be live-tuned at runtime, in which
  case we want the new hash so tiles fall back to the live path)."""
  canon = "%.6f|%.6f|%.6f|%.6f|%.4f|%.4f|%.2f" % (
    float(PHYSICS_A), float(PHYSICS_B), float(PHYSICS_C), float(PHYSICS_D),
    float(PHYSICS_MIN_LAT_ACCEL), float(PHYSICS_MAX_LAT_ACCEL),
    float(MAX_SPEED_DEFAULT),
  )
  return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def _physics_based_lateral_acceleration(curvature: float, *, low_speed_sigmoid_scale: float = 1.0) -> float:
    """
    Continuous sigmoid-based lateral acceleration function (scipy optimized)
    
    Replaces piecewise function with smooth continuous alternative that provides:
    - Highway zone (≤0.0029): Maintains ~3.12 m/s² performance  
    - Transition zone: Smooth exponential decay
    - Tight curves (>0.0053): 20%+ more aggressive than original (1.8-2.04 vs 1.5-1.7 m/s²)
    
    Mathematical model: Optimized sigmoid with R² = 0.9747
    Benefits: Perfect continuity, no discontinuous jumps, more aggressive low-speed cornering
    """
    curvature = max(1e-8, min(curvature, 1.0))
    # Use globally-tunable sigmoid parameters
    result = PHYSICS_A / (1.0 + math.exp(PHYSICS_B * (curvature - PHYSICS_C))) + PHYSICS_D
    try:
        sigmoid_scale = clip(float(low_speed_sigmoid_scale), 0.75, 1.25)
    except Exception:
        sigmoid_scale = 1.0
    tuned_min = max(0.1, float(PHYSICS_MIN_LAT_ACCEL) * sigmoid_scale)
    tuned_max = max(tuned_min, float(PHYSICS_MAX_LAT_ACCEL) * sigmoid_scale)
    result *= sigmoid_scale
    return max(tuned_min, min(result, tuned_max))

def _q_curve_multiplier(abs_curvature_meters: float) -> float:
    if not Q_CURVE_ENABLED or len(Q_CURVE_POINTS) < 2:
        return 1.0
    if not (abs_curvature_meters > 0.0 and math.isfinite(abs_curvature_meters)):
        return 1.0

    pts = []
    for k, q in Q_CURVE_POINTS:
        try:
            kf = float(k)
            qf = float(q)
        except Exception:
            continue
        if not (kf > 0.0 and math.isfinite(qf)):
            continue
        pts.append((kf, qf))
    if len(pts) < 2:
        return 1.0
    pts.sort(key=lambda kv: kv[0])

    kappa = clip(float(abs_curvature_meters), pts[0][0], pts[-1][0])
    logk = math.log10(max(kappa, 1e-12))

    for i in range(len(pts) - 1):
        k0, q0 = pts[i]
        k1, q1 = pts[i + 1]
        if kappa <= k1:
            log0 = math.log10(max(k0, 1e-12))
            log1 = math.log10(max(k1, 1e-12))
            if log1 <= log0:
                return clip(q0, 0.5, 1.5)
            t = (logk - log0) / (log1 - log0)
            q = q0 + (q1 - q0) * t
            return clip(q, 0.5, 1.5)
    return clip(pts[-1][1], 0.5, 1.5)

def curvature_to_speed(abs_curvature_meters: float, *, low_speed_sigmoid_scale: float = 1.0) -> float:
    """Calculates target speed (m/s) directly from curvature with optional low-speed sigmoid tuning."""
    if abs_curvature_meters < 1e-7:  # Handle straight roads
        return MAX_SPEED_DEFAULT

    # Get safe lateral acceleration using tuned sigmoid
    safe_lat_accel = _physics_based_lateral_acceleration(
      abs_curvature_meters,
      low_speed_sigmoid_scale=low_speed_sigmoid_scale,
    )

    # Calculate speed using physics formula v = sqrt(a / k) with CONSISTENT curvature
    try:
        base_speed_mps = math.sqrt(safe_lat_accel / abs_curvature_meters)
    except (ValueError, ZeroDivisionError):
        base_speed_mps = 0.0

    # Apply simple low-speed bias in mph with taper
    base_speed_mph = base_speed_mps * CV.MS_TO_MPH
    if LOW_SPEED_BIAS_MPH != 0.0 and base_speed_mph < LOW_SPEED_BIAS_END_MPH:
        taper = 1.0 - (base_speed_mph / max(LOW_SPEED_BIAS_END_MPH, 1e-3))
        base_speed_mph = base_speed_mph + LOW_SPEED_BIAS_MPH * max(0.0, min(1.0, taper))
        base_speed_mps = max(0.0, base_speed_mph * CV.MPH_TO_MS)

    # Apply speed increase factor and clip to reasonable maximum
    target_speed_mps = base_speed_mps * SPEED_INCREASE_FACTOR
    target_speed_mps = clip(target_speed_mps, 0.0, MAX_SPEED_DEFAULT)

    # Optional post-scale for tuning the curvature→speed relationship (multiplicative in speed)
    q = _q_curve_multiplier(abs_curvature_meters)
    return clip(target_speed_mps * q, 0.0, MAX_SPEED_DEFAULT)

def find_apexes_enhanced(curvature_array: np.ndarray, threshold: float = 5e-5, min_prominence: float = 1e-4) -> list:
    """
    Find local maxima (apex points) in curvature data with noise filtering.
    
    Args:
        curvature_array: Array of curvature values
        threshold: Minimum curvature to consider as potential apex
        min_prominence: Minimum peak prominence to filter noise
        
    Returns:
        List of indices where apexes are detected
    """
    if len(curvature_array) < 3:
        return []

    # Apply light smoothing (3-point moving average) to reduce noise
    # Use 'same' mode to maintain array size
    smoothed = np.convolve(curvature_array, [0.25, 0.5, 0.25], mode='same')

    apex_indices = []
    for i in range(1, len(smoothed) - 1):
        # Check if this is a local maximum above threshold
        if (smoothed[i] > threshold and
            smoothed[i] >= smoothed[i + 1] and
            smoothed[i] > smoothed[i - 1]):
            # Calculate prominence (peak height above neighbors)
            prominence = smoothed[i] - min(smoothed[i-1], smoothed[i+1])
            if prominence > min_prominence:
                apex_indices.append(i)

    return apex_indices

def calculate_anticipation_time(v_ego_ms: float, target_speed_ms: float, max_pred_lat_acc: float, aggressiveness: float = 1.0) -> float:
    """
    OPTIMIZED anticipation time calculation with research-validated parameters.
    17.6% improvement over original algorithm through Bayesian optimization.
    
    Optimized through synthetic testing across 108 scenarios covering:
    - Speed ranges: 5-85 mph across parking, residential, urban, highway contexts
    - Research-validated comfort deceleration limits (0.295g)
    - Human factors timing expectations from literature
    
    Key improvements:
    - Fixes "eternity at low speed" (parking: -5.5% timing)  
    - Fixes "insufficient buffer at high speed" (highway: +29% timing)
    - Context-aware scaling based on driving environment
    
    Args:
        v_ego_ms: Current vehicle speed (m/s)
        target_speed_ms: Target speed for curve (m/s)  
        max_pred_lat_acc: Maximum predicted lateral acceleration (m/s²)
        aggressiveness: User-configurable multiplier (0.5-2.0, default 1.0)
        
    Returns:
        Optimized anticipation time in seconds
    """

    # Optimized parameters from research study
    reaction_time_base = 1.185        # vs original 1.5s - faster response
    speed_normalization = 15.0        # vs original 20.0 - tuned for city driving
    speed_factor_min = 0.642          # vs original 0.7
    speed_factor_max = 1.975          # vs original 1.5 - wider range
    delta_factor_gain = 0.683         # vs original 0.5 - more sensitive
    delta_factor_max = 1.665          # vs original 1.5
    severity_normalization = 2.424    # vs original 1.5 - less lat acc impact
    timing_min = 0.565               # vs original 1.0 - allows faster reactions
    timing_max = 8.0                 # vs original 3.0 - wider range

    # Context-aware multipliers based on speed
    v_ego_mph = v_ego_ms * CV.MS_TO_MPH
    if v_ego_mph <= 15:
        context_multiplier = 0.973      # Parking: slightly faster
    elif v_ego_mph <= 35:
        context_multiplier = 1.144      # Residential: moderate increase
    elif v_ego_mph <= 55:
        context_multiplier = 1.200      # Urban: efficiency balance
    else:
        context_multiplier = 1.384      # Highway: maximum safety margin

    # Speed factor: Optimized scaling
    speed_factor = clip(v_ego_ms / speed_normalization, speed_factor_min, speed_factor_max)

    # Speed reduction factor: Enhanced sensitivity
    if v_ego_ms > 0.1:
        delta_ratio = (v_ego_ms - target_speed_ms) / v_ego_ms
        delta_factor = clip(1.0 + delta_ratio * delta_factor_gain, 1.0, delta_factor_max)
    else:
        delta_factor = 1.0

    # Curve severity factor: Simplified (optimization showed minimal impact)
    severity_factor = clip(max_pred_lat_acc / severity_normalization, 1.0, 1.0)

    # Calculate optimized timing
    base_timing = reaction_time_base * speed_factor * delta_factor * severity_factor
    timing = base_timing * context_multiplier * aggressiveness

    # Adjust max timing limit based on aggressiveness to allow more pre-emptive slowing
    adjusted_timing_max = timing_max * aggressiveness

    return clip(timing, timing_min, adjusted_timing_max)

def _debug(msg):
  if not _DEBUG:
    return
  print(msg)

def _description_for_state(turn_controller_state):
  if turn_controller_state == VisionTurnControllerState.disabled:
    return 'DISABLED'
  if turn_controller_state == VisionTurnControllerState.entering:
    return 'ENTERING'
  if turn_controller_state == VisionTurnControllerState.turning:
    return 'TURNING'
  if turn_controller_state == VisionTurnControllerState.leaving:
    return 'LEAVING'


class VisionTurnController:
  def __init__(self, CP):
    self._params = Params()
    try:
      self._mem_params = Params("/dev/shm/params")
    except Exception:
      self._mem_params = self._params
    self._CP = CP
    self._vm = None
    try:
      if VehicleModel is not None:
        self._vm = VehicleModel(CP)
    except Exception:
      self._vm = None
    self._steering_angle_deg = 0.0
    self._op_enabled = False
    self._gas_pressed = False
    # Defaults; Params applied by update_vtsc_params(force=True) below
    self._is_enabled = False
    # User-configurable aggressiveness for pre-emptive slowing (0.5-2.0, default 1.0)
    # Higher values = earlier/more conservative slowing before curves
    self._aggressiveness = 1.0

    # Optional fixed lead time override (seconds). 0.0 = disabled
    self._fixed_lead_time_s = 0.0
    # Signed timing offsets (seconds): 0 = default timing, lower = earlier, higher = later.
    # Curve offset affects horizon interpretation; overshoot offset affects braking onset timing;
    # apex exit offset affects when post-apex acceleration logic begins.
    self._curve_phase_offset_s = 0.0
    self._overshoot_phase_offset_s = 0.0
    self._apex_exit_phase_offset_s = 0.0
    
    # ===== ADAPTIVE DECELERATION PARAMETERS =====
    # User-configurable noise filtering parameters
    self._filter_alpha = DEFAULT_FILTER_ALPHA
    self._base_filter_alpha = self._filter_alpha
    
    self._hysteresis_threshold = DEFAULT_HYSTERESIS_THRESHOLD
    
    self._safety_bias = DEFAULT_SAFETY_BIAS
    
    self._last_params_update = 0.
    self._v_cruise_setpoint = 0.
    self._v_ego = 0.
    self._a_ego = 0.
    self._a_target = 0.
    self._v_overshoot = 0.
    self._state = VisionTurnControllerState.disabled

    # ===== ADAPTIVE DECELERATION SYSTEM =====
    self._current_decel = 0.0
    self._filtered_decel_requirement = 0.0
    self._decel_hysteresis_state = False

    # ===== VISION OCCLUSION HANDLING =====
    self._occlusion_state = VisionOcclusionState()
    # Fast reacquisition window management
    self._base_filter_alpha = DEFAULT_FILTER_ALPHA
    self._fast_reacq_until = 0.0
    self._fast_reacq_alpha = 0.85
    self._fast_reacq_window_s = 0.90
    # Visibility barrier params (defaults; Params override in update)
    self._vis_horizon_s = 1.4
    self._vis_margin_m = 10.0
    self._last_vision_confidence = 1.0
    self._gamma_per_meter = 0.00035
    self._lat_jerk_cap = 2.0
    # Sentinel: <= 0 disables cap in tests
    if self._lat_jerk_cap <= 0.0:
      self._lat_jerk_cap = 1e9
    # Anticipation moderation state
    self._prev_smoothed_conf = 1.0
    self._prev_filtered_curvature = 0.0
    self._anticipation_budget_window_start = 0.0
    self._cum_anticipation_reduction = 0.0
    self._last_high_conf_target_speed = 0.0
    self._anticipation_max_reduction_mps = 2.0

    # ===== Freeway cap hold (avoid flicker) =====
    self._v_turn_hold_until = 0.0
    self._v_turn_hold_min = float(INF_SPEED)
    self._v_turn_release_shape_active = False

    # Advanced controller state
    self._current_accel = 0.0
    self._prev_target_speed = 0.0
    # Smoothing bounds (tunable)
    self._max_decel = 3.5  # VisionTurnSpeedControlSmoothingMaxDecel
    self._max_jerk = 6.0   # VisionTurnSpeedControlSmoothingMaxJerk
    self._accel_to_decel_ratio = 1.3  # VisionTurnSpeedControlAccelToDecelRatio
    self._jerk_accel_multiplier = 2.0 # VisionTurnSpeedControlJerkAccelMultiplier
    self._max_accel = self._accel_to_decel_ratio * self._max_decel
    self._max_jerk_accel = self._jerk_accel_multiplier * self._max_jerk

    # EMA filtering for curvature (tunable)
    self._curvature_ema_ratio = 0.3  # VisionTurnSpeedControlCurvatureEMAFactor
    self._filtered_curvature = 0.0

    # Anticipatory deceleration state
    self._is_decelerating_for_curve = False

    # Anticipation/Overshoot planning tunables
    self._planning_decel_limit = 3.5            # VisionTurnSpeedControlPlanningDecelLimit (m/s²)
    self._overshoot_safety_margin = 1.2         # VisionTurnSpeedControlOvershootSafetyMargin (multiplier)
    self._overshoot_min_distance = 10.0         # VisionTurnSpeedControlOvershootMinDistance (m)
    self._anticipation_target_reduction = 0.95  # VisionTurnSpeedControlAnticipationTargetReduction
    # Time-to-brake trigger for planner-facing overshoot cap.
    # <= 0 means we should already be applying overshoot braking.
    self._overshoot_trigger_in_s = float('inf')
    self._overshoot_cap_active = False
    self._near_apex_release_until = 0.0

    # Apex detection and tracking
    self._apex_indices = []  # Indices of detected apexes in trajectory
    self._last_apex_passed_time = 0.0  # For hysteresis
    self._distance_past_apex = 0.0  # Meters past most recent apex
    self._apex_exit_ready = False
    self._prev_apex_exit_ready = False
    self._apex_trigger_idx = 0
    self._curve_sample_idx = 0
    # Detection
    self._apex_threshold = 5e-5          # VisionTurnSpeedControlApexThreshold
    self._apex_prominence = 1e-4         # VisionTurnSpeedControlApexProminence
    self._apex_hysteresis_time = 2.0     # VisionTurnSpeedControlApexHysteresisTime (s)
    self._apex_meters_per_index = 2.0    # VisionTurnSpeedControlApexMetersPerIndex (m)
    self._apex_near_index = 3            # VisionTurnSpeedControlApexNearIndex (indices)
    # Boost
    self._apex_boost_distance = 50.0     # VisionTurnSpeedControlApexBoostDistance (m)
    self._apex_boost_factor = 0.1        # VisionTurnSpeedControlApexBoostFactor (0..0.2)
    self._apex_boost_min_lat_accel = 1.0 # VisionTurnSpeedControlApexBoostMinLatAccel (m/s²)
    self._apex_boost_center = 2.0        # VisionTurnSpeedControlApexBoostCenter (m/s²)
    self._apex_boost_width = 0.5         # VisionTurnSpeedControlApexBoostWidth (m/s²)
    self._boost_safety_curvature_scale = 0.7 # VisionTurnSpeedControlBoostSafetyCurvatureScale

    # Comfort/adaptive deceleration limits (tunable)
    self._comfort_decel_limit = COMFORT_DECEL_LIMIT
    self._comfort_jerk_limit = COMFORT_JERK_LIMIT
    self._max_adaptive_decel = MAX_ADAPTIVE_DECEL
    self._max_adaptive_jerk = MAX_ADAPTIVE_JERK
    self._curvature_trajectory = []  # Store curvature array for apex detection

    self._reset()
    # Force an initial params refresh to ensure runtime knobs reflect latest Params
    try:
      update_vtsc_params(self, force=True)
    except Exception:
      # Safe to continue with defaults if Params not available at startup
      pass

    # Map lookahead cache/state
    self._map_curv_cache_raw = None
    self._map_curv_cache = []
    self._map_curv_last_ts = 0.0
    # Versioned route-level whole-curve profile. This stays separate from the
    # legacy MapCurvatures cache so its ~5 m / ~1.2 km representation is never
    # decimated to the legacy 160-point / 800 m path.
    self._map_whole_curve_cache_raw = None
    self._map_whole_curve_cache: MapWholeCurveProfile | None = None
    self._map_whole_curve_cache_rejection: str | None = None
    self._map_whole_curve_speed_cache_key = None
    self._map_whole_curve_speed_cache: list[float] = []
    self._map_whole_curve_last_ts = 0.0
    self._map_whole_curve_reason = "missing"
    self._map_profile_source = "none"
    self._map_lookahead_enabled_prev = False
    # Sigmoid-baked per-curvature speeds from mapd (schema v1+ tiles).
    # Cached at the same 5 Hz as MapCurvatures; nil/empty list means
    # "fall back to live curvature_to_speed". Hash compared per-tick.
    self._map_pre_curve_speeds_cache_raw = None
    self._map_pre_curve_speeds_cache: list[float] = []
    self._map_pre_curve_speeds_last_ts = 0.0
    self._map_tiles_sigmoid_hash = ""
    self._map_strategy_mode = DEFAULT_MAP_STRATEGY
    self._map_strategy_state = MapStrategyState()
    self._turn_visible_sticky = False
    self._map_tail_candidate = None
    self._map_tail_active = False
    self._map_tail_last_cap = None
    # Holdover: bridge transient map data dropouts (GPS flicker, road geometry
    # invalidation, empty MapCurvatures) so the strategic cap doesn't release
    # for 1-2 frames and then slam back on, causing bucking.
    self._map_holdover_cap = None       # last valid v_cap (m/s)
    self._map_holdover_candidate = None # last valid candidate
    self._map_holdover_ts = 0.0         # monotonic time when holdover was saved
    self._map_tail_advisory_cap = None
    self._map_tail_strategic_cap = None
    self._map_tail_last_start = 0.0
    self._map_tail_last_coverage = 0.0
    self._map_tail_anchor_dist_m = 0.0
    self._map_tail_anchor_k = 0.0
    self._map_tail_anchor_vsafe = 0.0
    self._map_tail_anchor_index = -1
    self._longitudinal_response_model = None
    self._road_geometry_valid = False
    self._lane_change_active = False
    self._single_blinker_active = False
    # Debug-only map lookahead diagnostics (why map cap is inactive this frame)
    self._map_tail_reason = "init"
    self._map_tail_compute_reason = "init"
    self._visible_mainline_counterevidence_since = 0.0

    # Rally co-pilot / HUD curve preview derived from VTSC's map lookahead inputs.
    # The HUD must not compute curves; it only renders these fields.
    self._curve_preview_valid = False
    self._curve_preview_distance_m = 0.0
    self._curve_preview_time_to_s = 0.0
    self._curve_preview_kappa_max = 0.0
    self._curve_preview_direction = 0  # VisionTurnSpeedControl.TurnDirection (unknown=0)
    self._curve_preview_severity = 0   # VisionTurnSpeedControl.CurveSeverity (unknown=0)
    self._curve_preview_points: list[tuple[float, float]] = []
    self._curve_preview_tiles: list[dict] = []
    self._curve_preview_branch_stubs: list[dict] = []
    self._curve_preview_last_ts = 0.0
    self._curve_preview_last_cache_raw = None
    self._curve_preview_last_latlon: tuple[float, float] | None = None
    self._clear_winding_road_context()
    self._clear_mapd_winding_context()
    self._clear_winding_context()
    self._clear_winding_behavior_profile()

    # Lead-aware occlusion bypass
    self._occl_bypass_with_lead = True
    self._occl_bypass_headway_s = 3.0
    self._lead_present = False
    self._lead_headway_s = 99.0
    self._occl_lead_bypass_active = False

    # Low-speed envelope calibration state. Positive values relax the low-speed sigmoid slightly,
    # negative values tighten it. The state evolves slowly from repeated steering headroom evidence.
    self._low_speed_calibration_enabled = bool(getattr(self, "_low_speed_calibration_enabled", True))
    self._low_speed_calibration_override_profile_log10_bins = np.linspace(
      math.log10(float(LOW_SPEED_CALIB_OVERRIDE_PROFILE_MIN_CURVATURE)),
      math.log10(float(LOW_SPEED_CALIB_OVERRIDE_PROFILE_MAX_CURVATURE)),
      int(LOW_SPEED_CALIB_OVERRIDE_PROFILE_BINS),
    )
    self._low_speed_calibration_override_profile = self._empty_low_speed_calibration_override_profile()
    self._low_speed_calibration_override_profile_param_state = self._empty_low_speed_calibration_override_profile()
    self._low_speed_calibration_override_profile_persisted_state = self._empty_low_speed_calibration_override_profile()
    self._low_speed_calibration_state = 0.0
    self._low_speed_calibration_base_state = 0.0
    self._low_speed_calibration_override_state = 0.0
    self._low_speed_calibration_headroom_ema = 0.0
    self._low_speed_calibration_override_ema = 0.0
    self._low_speed_calibration_last_update_s = 0.0
    self._dbg_low_speed_calibration_active = False
    self._dbg_low_speed_calibration_reason = "init"
    self._dbg_low_speed_calibration_headroom = 0.0
    self._dbg_low_speed_calibration_headroom_ema = 0.0
    self._dbg_low_speed_calibration_override_ema = 0.0
    self._dbg_low_speed_calibration_divergence_mps = 0.0
    self._dbg_low_speed_calibration_scale = 1.0
    self._dbg_low_speed_calibration_curve_mph = 0.0
    self._dbg_low_speed_calibration_output = 0.0
    self._dbg_low_speed_calibration_gap = 0.0
    self._dbg_low_speed_calibration_gap_ratio = 0.0
    self._dbg_low_speed_calibration_saturated = False
    self._dbg_low_speed_calibration_map_active = False
    self._dbg_low_speed_calibration_map_anchor_k = 0.0
    self._dbg_low_speed_calibration_dual_bin = False
    self._low_speed_calibration_param_state = 0.0
    self._low_speed_calibration_persisted_state = 0.0
    self._low_speed_calibration_last_persist_s = 0.0
    self._low_speed_calibration_high_end_mph = float(
      getattr(self, "_low_speed_calibration_high_end_mph", LOW_SPEED_CALIB_TARGET_END_MPH)
    )
    try:
      self._sync_low_speed_calibration_param(force=True)
    except Exception:
      pass

    # Telemetry/debug controls
    self._dbg_enabled = False
    self._dbg_emit_interval_s = 0.5  # ~2 Hz
    self._dbg_next_emit_ts = 0.0
    self._dbg_refresh_ts = 0.0
    self._dbg_write_file = False
    # Snapshot fields
    self._dbg_k_model = 0.0
    self._dbg_k_steer = 0.0
    self._dbg_steer_fallback_active = False
    self._dbg_target_raw = 0.0
    self._dbg_target_final = 0.0
    self._dbg_controls_desired_curvature = 0.0
    self._dbg_controls_actual_curvature = 0.0
    self._dbg_occl_positive_margin = False
    self._dbg_early_no_raise = False
    self._dbg_tail_frac = 0.0
    self._dbg_s_tail = 0.0
    self._dbg_jerk_cmd = 0.0
    # FOV gating + units diagnostics
    self._psi_fov_rad = 0.49
    # Unify margin to ~5° across all code paths
    self._psi_margin_rad = 0.087  # ~5 deg
    self._fov_occluded = False
    self._fov_on_cnt = 0
    self._fov_off_cnt = 0
    self._fov_reason = ''
    self._fov_pretrigger_time_s = 1.5
    # Disable onset stickiness/overshoot windows (no-raise window effectively 0)
    self._fov_onset_boost_frames = 0
    self._fov_overshoot_frames = 0
    self._fov_boost_left = 0
    self._fov_overshoot_left = 0
    self._fov_ewma_tau_s = 0.4
    self._fov_kappa_ewma = 0.0
    self._fov_N_on = 2
    self._fov_N_off = 12
    self._dbg_psi_vis = 0.0
    self._dbg_psi_thresh = 0.0
    self._dbg_ttfov_s = 0.0
    self._dbg_units_ok = True
    self._dbg_gamma_eff = 0.0
    # Occlusion arbitration breadcrumbs (defaults)
    self._dbg_psi_est = 0.0
    self._dbg_consider_occl = False
    self._dbg_double_cap_guard = False
    # Onset tracking for occlusion window and early no-raise
    self._occlusion_prev = False
    self._occlusion_onset_timer_s = 0.0
    self._v_cap_active_at_onset_mps = 0.0
    self._onset_no_raise_active = False
    # Cap selection + freeway guard debug fields
    self._dbg_active_cap = ""
    self._dbg_cap_visible_vmin = 0.0
    self._dbg_cap_occl_vmin = 0.0
    self._dbg_cap_map_vmin = 0.0
    self._dbg_vtsc_cmd = 0.0
    self._dbg_strategy_mode = DEFAULT_MAP_STRATEGY
    self._dbg_strategy_state = "idle"
    self._dbg_map_advisory_cap = 0.0
    self._dbg_map_strategic_cap = 0.0
    self._dbg_vision_local_cap = 0.0
    self._dbg_selected_cap = 0.0
    self._dbg_map_floor_active = False
    self._dbg_map_floor_reason = ""
    self._dbg_vision_relax_allowed = False
    self._dbg_vision_relax_reason = ""
    self._dbg_map_anchor_dist_m = 0.0
    self._dbg_map_anchor_k = 0.0
    self._dbg_map_takeover_dwell_s = 0.0
    self._dbg_map_counterevidence_dwell_s = 0.0
    self._dbg_planner_min_accel = 0.0
    self._dbg_planner_response_decel = 0.0
    self._dbg_planner_delay_s = 0.0
    self._dbg_winding_profile_level = 0
    self._dbg_winding_profile_name = DEFAULT_WINDING_BEHAVIOR_PROFILE.name
    self._dbg_winding_profile_source = "none"
    self._dbg_winding_release_up_slew_mps2 = 0.0
    self._dbg_winding_release_limited = False
    self._dbg_winding_release_shape_active = False
    self._dbg_kappa_vis = 0.0
    self._dbg_s_visible_m = 0.0
    self._dbg_path_conf = 0.0
    self._dbg_fail_open = False
    self._freeway_failopen_active = False

    # Vision-floor and dropout discrimination (aggressive bias)
    # 0=off, 1=TTL floor after last-known-good, 2=strict floor always
    self._vision_floor_ttl_s = 3.0
    self._vision_floor_mult = 1.00
    # Treat short model frame loss as transient dropout; suppress occlusion pretrigger briefly
    self._dropout_grace_s = 0.40
    self._dropout_until = 0.0

  # ===== Internal Param Helpers (decode/parse/clip) =====
  def _get_float_param(self, key: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """Read a float param from Params with robust decoding and optional clipping.

    - Accepts bytes or string; falls back to default on any parse error.
    - If bounds provided, applies clip to [lo, hi].
    """
    try:
      raw = self._params.get(key)
      if raw is None:
        val = float(default)
      else:
        s = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else raw
        val = float(s)
    except Exception:
      # Be robust to UnknownKeyName or any decode/parse failure in off-road/harness contexts
      val = float(default)
    if lo is not None and hi is not None:
      return clip(val, lo, hi)
    return val

  def _get_bool_param(self, key: str, default: bool = False) -> bool:
    """Read a boolean param; returns default if underlying access fails."""
    try:
      return bool(self._params.get_bool(key))
    except Exception:
      return bool(default)

  def _get_string_param(self, key: str, default: str = "") -> str:
    """Read a string param; returns default on any decode/access failure."""
    try:
      raw = self._params.get(key)
      if raw is None:
        return str(default)
      return str(raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else raw)
    except Exception:
      return str(default)

  @staticmethod
  def _clip_low_speed_calibration_state(value: float) -> float:
    try:
      state = float(value)
    except Exception:
      state = 0.0
    return float(clip(state, -float(LOW_SPEED_CALIB_MAX_TIGHTEN), float(LOW_SPEED_CALIB_MAX_RELAX)))

  @staticmethod
  def _clip_low_speed_calibration_override_state(value: float) -> float:
    try:
      state = float(value)
    except Exception:
      state = 0.0
    return float(clip(state, 0.0, float(LOW_SPEED_CALIB_OVERRIDE_MAX_RELAX)))

  @staticmethod
  def _empty_low_speed_calibration_override_profile() -> np.ndarray:
    return np.zeros(int(LOW_SPEED_CALIB_OVERRIDE_PROFILE_BINS), dtype=np.float64)

  @staticmethod
  def _clip_low_speed_calibration_override_profile(values) -> np.ndarray:
    try:
      arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
      arr = np.zeros(0, dtype=np.float64)
    if arr.size != int(LOW_SPEED_CALIB_OVERRIDE_PROFILE_BINS):
      return VisionTurnController._empty_low_speed_calibration_override_profile()
    return np.clip(arr, 0.0, float(LOW_SPEED_CALIB_OVERRIDE_MAX_RELAX)).astype(np.float64, copy=False)

  @staticmethod
  def _low_speed_calibration_override_profile_changed(a: np.ndarray, b: np.ndarray, eps: float) -> bool:
    aa = VisionTurnController._clip_low_speed_calibration_override_profile(a)
    bb = VisionTurnController._clip_low_speed_calibration_override_profile(b)
    if aa.shape != bb.shape:
      return True
    try:
      return bool(np.max(np.abs(aa - bb)) > float(eps))
    except Exception:
      return True

  def _parse_low_speed_calibration_override_profile(self, raw, *, fallback: np.ndarray | None = None) -> np.ndarray:
    fallback_profile = self._clip_low_speed_calibration_override_profile(
      self._empty_low_speed_calibration_override_profile() if fallback is None else fallback
    )
    if raw is None:
      return fallback_profile.copy()
    try:
      s = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
      obj = json.loads(s)
    except Exception:
      return fallback_profile.copy()
    values = obj.get("values") if isinstance(obj, dict) else obj
    if not isinstance(values, list):
      return fallback_profile.copy()
    try:
      arr = np.asarray([float(v) for v in values], dtype=np.float64)
    except Exception:
      return fallback_profile.copy()
    clipped = self._clip_low_speed_calibration_override_profile(arr)
    if clipped.size != int(LOW_SPEED_CALIB_OVERRIDE_PROFILE_BINS):
      return fallback_profile.copy()
    return clipped.copy()

  def _serialize_low_speed_calibration_override_profile(self, profile: np.ndarray) -> str:
    clipped = self._clip_low_speed_calibration_override_profile(profile)
    payload = {
      "version": 1,
      "values": [round(float(v), 6) for v in clipped.tolist()],
    }
    return json.dumps(payload, separators=(",", ":"))

  def _low_speed_calibration_override_profile_weights(self, abs_curvature_meters: float) -> np.ndarray:
    try:
      kappa = float(abs_curvature_meters)
    except Exception:
      return self._empty_low_speed_calibration_override_profile()
    if not (kappa > 0.0 and math.isfinite(kappa)):
      return self._empty_low_speed_calibration_override_profile()
    logk = math.log10(clip(
      kappa,
      float(LOW_SPEED_CALIB_OVERRIDE_PROFILE_MIN_CURVATURE),
      float(LOW_SPEED_CALIB_OVERRIDE_PROFILE_MAX_CURVATURE),
    ))
    sigma = max(1e-3, float(LOW_SPEED_CALIB_OVERRIDE_PROFILE_SIGMA_LOG10))
    d = (self._low_speed_calibration_override_profile_log10_bins - logk) / sigma
    weights = np.exp(-0.5 * np.square(d))
    weights[np.abs(d) > float(LOW_SPEED_CALIB_OVERRIDE_PROFILE_CUTOFF_SIGMA)] = 0.0
    denom = float(np.sum(weights))
    if denom <= 1e-12:
      return self._empty_low_speed_calibration_override_profile()
    return (weights / denom).astype(np.float64, copy=False)

  def _low_speed_calibration_override_state_for_curvature(self, abs_curvature_meters: float) -> float:
    profile = self._clip_low_speed_calibration_override_profile(
      getattr(self, "_low_speed_calibration_override_profile", self._empty_low_speed_calibration_override_profile())
    )
    weights = self._low_speed_calibration_override_profile_weights(abs_curvature_meters)
    if profile.shape != weights.shape:
      return 0.0
    try:
      return self._clip_low_speed_calibration_override_state(float(np.dot(profile, weights)))
    except Exception:
      return 0.0

  def _apply_low_speed_calibration_override_local_delta(self, abs_curvature_meters: float, delta_local: float) -> float:
    try:
      delta = float(delta_local)
    except Exception:
      return self._low_speed_calibration_override_state_for_curvature(abs_curvature_meters)
    if abs(delta) <= 1e-9:
      return self._low_speed_calibration_override_state_for_curvature(abs_curvature_meters)
    profile = self._clip_low_speed_calibration_override_profile(
      getattr(self, "_low_speed_calibration_override_profile", self._empty_low_speed_calibration_override_profile())
    )
    weights = self._low_speed_calibration_override_profile_weights(abs_curvature_meters)
    if profile.shape != weights.shape:
      return 0.0
    support = float(np.dot(weights, weights))
    if support <= 1e-9:
      return self._low_speed_calibration_override_state_for_curvature(abs_curvature_meters)
    profile = np.clip(profile + (delta / support) * weights, 0.0, float(LOW_SPEED_CALIB_OVERRIDE_MAX_RELAX)).astype(np.float64, copy=False)
    self._low_speed_calibration_override_profile = profile
    return self._clip_low_speed_calibration_override_state(float(np.dot(profile, weights)))

  def _refresh_low_speed_calibration_state(self) -> None:
    base_state = self._clip_low_speed_calibration_state(getattr(self, "_low_speed_calibration_base_state", 0.0))
    override_state = self._clip_low_speed_calibration_override_state(getattr(self, "_low_speed_calibration_override_state", 0.0))
    self._low_speed_calibration_base_state = float(base_state)
    self._low_speed_calibration_override_state = float(override_state)
    self._low_speed_calibration_state = float(base_state + override_state)

  def _sync_low_speed_calibration_param(self, *, force: bool = False) -> None:
    if not bool(getattr(self, "_low_speed_calibration_enabled", True)):
      self._low_speed_calibration_state = 0.0
      self._low_speed_calibration_base_state = 0.0
      self._low_speed_calibration_override_state = 0.0
      self._low_speed_calibration_override_profile = self._empty_low_speed_calibration_override_profile()
      self._low_speed_calibration_override_profile_param_state = self._empty_low_speed_calibration_override_profile()
      self._low_speed_calibration_override_profile_persisted_state = self._empty_low_speed_calibration_override_profile()
      self._low_speed_calibration_param_state = 0.0
      self._low_speed_calibration_persisted_state = 0.0
      self._low_speed_calibration_headroom_ema = 0.0
      self._low_speed_calibration_override_ema = 0.0
      self._low_speed_calibration_last_persist_s = 0.0
      return

    raw_state = self._get_float_param(
      "VisionTurnSpeedControlLowSpeedLearnedState",
      getattr(self, "_low_speed_calibration_param_state", 0.0),
      -float(LOW_SPEED_CALIB_MAX_TIGHTEN),
      float(LOW_SPEED_CALIB_MAX_RELAX),
    )
    try:
      raw_override_profile = self._params.get("VisionTurnSpeedControlDriverOverrideCurveProfile")
    except Exception:
      raw_override_profile = None
    loaded_override_profile = self._parse_low_speed_calibration_override_profile(
      raw_override_profile,
      fallback=getattr(self, "_low_speed_calibration_override_profile_param_state", self._empty_low_speed_calibration_override_profile()),
    )
    loaded_state = self._clip_low_speed_calibration_state(raw_state)
    tracked_state = self._clip_low_speed_calibration_state(getattr(self, "_low_speed_calibration_param_state", 0.0))
    if (
      force or
      abs(float(loaded_state) - float(tracked_state)) > float(LOW_SPEED_CALIB_PARAM_SYNC_EPS) or
      self._low_speed_calibration_override_profile_changed(
        loaded_override_profile,
        getattr(self, "_low_speed_calibration_override_profile_param_state", self._empty_low_speed_calibration_override_profile()),
        float(LOW_SPEED_CALIB_PARAM_SYNC_EPS),
      )
    ):
      self._low_speed_calibration_base_state = float(loaded_state)
      self._low_speed_calibration_override_profile = loaded_override_profile.copy()
      self._low_speed_calibration_override_state = 0.0
      self._low_speed_calibration_param_state = float(loaded_state)
      self._low_speed_calibration_persisted_state = float(loaded_state)
      self._low_speed_calibration_override_profile_param_state = loaded_override_profile.copy()
      self._low_speed_calibration_override_profile_persisted_state = loaded_override_profile.copy()
      self._low_speed_calibration_headroom_ema = 0.0
      self._low_speed_calibration_override_ema = 0.0
      self._refresh_low_speed_calibration_state()

  def _maybe_persist_low_speed_calibration_state(self, now_s: float) -> None:
    if not bool(getattr(self, "_low_speed_calibration_enabled", True)):
      return
    state = self._clip_low_speed_calibration_state(getattr(self, "_low_speed_calibration_base_state", 0.0))
    persisted = self._clip_low_speed_calibration_state(getattr(self, "_low_speed_calibration_persisted_state", 0.0))
    override_profile = self._clip_low_speed_calibration_override_profile(
      getattr(self, "_low_speed_calibration_override_profile", self._empty_low_speed_calibration_override_profile())
    )
    override_persisted = self._clip_low_speed_calibration_override_profile(
      getattr(self, "_low_speed_calibration_override_profile_persisted_state", self._empty_low_speed_calibration_override_profile())
    )
    last_write_s = float(getattr(self, "_low_speed_calibration_last_persist_s", 0.0) or 0.0)
    if (
      abs(float(state) - float(persisted)) < float(LOW_SPEED_CALIB_PERSIST_DELTA) and
      not self._low_speed_calibration_override_profile_changed(
        override_profile,
        override_persisted,
        float(LOW_SPEED_CALIB_PERSIST_DELTA),
      )
    ):
      return
    if float(now_s) < (last_write_s + float(LOW_SPEED_CALIB_PERSIST_WRITE_S)):
      return
    try:
      if abs(float(state) - float(persisted)) >= float(LOW_SPEED_CALIB_PERSIST_DELTA):
        self._params.put_nonblocking("VisionTurnSpeedControlLowSpeedLearnedState", f"{float(state):.6f}")
      if self._low_speed_calibration_override_profile_changed(
        override_profile,
        override_persisted,
        float(LOW_SPEED_CALIB_PERSIST_DELTA),
      ):
        self._params.put_nonblocking(
          "VisionTurnSpeedControlDriverOverrideCurveProfile",
          self._serialize_low_speed_calibration_override_profile(override_profile),
        )
      self._low_speed_calibration_persisted_state = float(state)
      self._low_speed_calibration_override_profile_persisted_state = override_profile.copy()
      self._low_speed_calibration_last_persist_s = float(now_s)
    except Exception:
      pass

  def set_longitudinal_response_model(self, response_model: CruiseResponseModel | None) -> None:
    self._longitudinal_response_model = response_model

  def _should_emit_debug(self, now_s: float) -> bool:
    try:
      if now_s >= float(getattr(self, '_dbg_refresh_ts', 0.0)):
        self._dbg_enabled = bool(self._get_bool_param('VTSCVerboseDebug', False))
        # refresh file writer toggle alongside verbose debug
        self._dbg_write_file = bool(self._get_bool_param('VTSCWriteSnapshotFile', False))
        self._dbg_refresh_ts = now_s + 2.0
    except Exception:
      self._dbg_enabled = False
    if not self._dbg_enabled:
      return False
    nxt = float(getattr(self, '_dbg_next_emit_ts', 0.0))
    if now_s >= nxt:
      self._dbg_next_emit_ts = now_s + float(getattr(self, '_dbg_emit_interval_s', 0.5))
      return True
    return False

  def _vision_status_str(self) -> str:
    try:
      vs = getattr(self._occlusion_state, 'vision_status', None)
    except Exception:
      vs = None
    if vs == VisionStatus.FULL_VISIBILITY:
      return 'FULL'
    if vs == VisionStatus.PARTIAL_OCCLUSION:
      return 'PARTIAL'
    if vs == VisionStatus.SEVERE_OCCLUSION:
      return 'SEVERE'
    if vs == VisionStatus.VISION_LOST:
      return 'LOST'
    return 'UNKNOWN'

  def _append_snapshot_to_file(self, snap: dict, now_s: float) -> None:
    """Append a compact JSON snapshot line to a small rotating file on device."""
    try:
      base_dir = "/data/media/0/VTSCDebug"
      path = os.path.join(base_dir, "vtsc_snapshots.jsonl")
      os.makedirs(base_dir, exist_ok=True)
      # Attach timestamp to snapshot
      snap_out = dict(snap)
      snap_out['ts'] = float(now_s)
      line = json.dumps(snap_out, separators=(',', ':')) + "\n"
      # Rotate if file grows beyond ~512 KB (simple strategy)
      try:
        if os.path.exists(path) and os.path.getsize(path) > 512 * 1024:
          # Truncate by replacing with empty file; keep a single backup
          try:
            os.replace(path, path + ".1")
          except Exception:
            pass
      except Exception:
        pass
      with open(path, 'a', encoding='utf-8') as f:
        f.write(line)
    except Exception:
      # Never raise from telemetry path
      pass

  def snapshot_debug_state(self) -> dict:
    try:
      v_ego = float(getattr(self, '_v_ego', 0.0))
      v_cruise = float(getattr(self, '_v_cruise_setpoint', 0.0))
      lead = bool(getattr(self, '_lead_present', False))
      hw = float(getattr(self, '_lead_headway_s', 99.0))
      conf = float(getattr(self._occlusion_state, 'smoothed_confidence', 0.0))
      raw_conf = float(getattr(self, '_last_vision_confidence', conf))
      k_model = float(getattr(self, '_dbg_k_model', 0.0))
      k_steer = float(getattr(self, '_dbg_k_steer', 0.0))
      steer_fallback_active = bool(getattr(self, '_dbg_steer_fallback_active', False))
      curve_phase_offset_raw = float(getattr(self, '_curve_phase_offset_s', 0.0))
      curve_phase_offset_effective = float(effective_curve_phase_offset_s(curve_phase_offset_raw))
      k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
      k_vis = float(getattr(self._occlusion_state, 'last_valid_curvature', 0.0))
      is_easing = bool(getattr(self, '_is_easing', False))
      abs_cr = float(getattr(self, '_abs_curvature_rate', 0.0))
      # Speeds
      v_phys_base = float(min(v_cruise, self._curve_speed(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))))
      v_occ = float(self._curve_speed(max(1e-8, k_est)))
      v_vis = float(self._curve_speed(max(1e-8, k_vis)))
      raw = float(getattr(self, '_dbg_target_raw', 0.0))
      final = float(getattr(self, '_dbg_target_final', raw))
      # Occlusion gating
      occl_margin = bool(getattr(self, '_dbg_occl_positive_margin', False))
      bypass = bool(getattr(self, '_occl_lead_bypass_active', False))
      low_speed_margin = bool(getattr(self, '_dbg_low_speed_margin', False))
      fov_occ = bool(getattr(self, '_fov_occluded', False))
      vis_h = float(getattr(self, '_vis_horizon_s', 1.4))
      tail_frac = float(getattr(self, '_dbg_tail_frac', 0.0))
      s_tail = float(getattr(self, '_dbg_s_tail', 0.0))
      enr = bool(getattr(self, '_dbg_early_no_raise', False))
      # Lookahead
      map_active = bool(getattr(self, '_map_tail_active', False))
      map_cap = float(getattr(self, '_map_tail_last_cap', 0.0) or 0.0)
      map_start = float(getattr(self, '_map_tail_last_start', 0.0) or 0.0)
      map_cov = float(getattr(self, '_map_tail_last_coverage', 0.0) or 0.0)
      map_reason = str(getattr(self, '_map_tail_reason', '') or '')
      map_compute_reason = str(getattr(self, '_map_tail_compute_reason', '') or '')
      # Limits and commands
      comfort = float(getattr(self, '_comfort_decel_limit', -1.47))
      max_adapt = float(getattr(self, '_max_adaptive_decel', -6.0))
      decel_cmd = float(getattr(self, '_current_decel', 0.0))
      jerk_cmd = float(getattr(self, '_dbg_jerk_cmd', 0.0))
      a_cmd = float(getattr(self, '_a_target', 0.0))
      # Newly added diagnostics populated in _update_solution
      cap_vis = float(getattr(self, '_dbg_cap_visible_vmin', 0.0))
      cap_occ = float(getattr(self, '_dbg_cap_occl_vmin', 0.0))
      cap_map = float(getattr(self, '_dbg_cap_map_vmin', 0.0))
      active_cap = str(getattr(self, '_dbg_active_cap', '') or '')
      vtsc_cmd = float(getattr(self, '_dbg_vtsc_cmd', 0.0) or 0.0)
      strategy_mode = str(getattr(self, '_dbg_strategy_mode', DEFAULT_MAP_STRATEGY) or DEFAULT_MAP_STRATEGY)
      strategy_state = str(getattr(self, '_dbg_strategy_state', 'idle') or 'idle')
      map_advisory_cap = float(getattr(self, '_dbg_map_advisory_cap', 0.0) or 0.0)
      map_strategic_cap = float(getattr(self, '_dbg_map_strategic_cap', 0.0) or 0.0)
      vision_local_cap = float(getattr(self, '_dbg_vision_local_cap', 0.0) or 0.0)
      selected_cap = float(getattr(self, '_dbg_selected_cap', 0.0) or 0.0)
      map_floor_active = bool(getattr(self, '_dbg_map_floor_active', False))
      map_floor_reason = str(getattr(self, '_dbg_map_floor_reason', '') or '')
      vision_relax_allowed = bool(getattr(self, '_dbg_vision_relax_allowed', False))
      vision_relax_reason = str(getattr(self, '_dbg_vision_relax_reason', '') or '')
      map_anchor_dist = float(getattr(self, '_dbg_map_anchor_dist_m', 0.0) or 0.0)
      map_anchor_k = float(getattr(self, '_dbg_map_anchor_k', 0.0) or 0.0)
      takeover_dwell_s = float(getattr(self, '_dbg_map_takeover_dwell_s', 0.0) or 0.0)
      counterevidence_dwell_s = float(getattr(self, '_dbg_map_counterevidence_dwell_s', 0.0) or 0.0)
      planner_min_accel = float(getattr(self, '_dbg_planner_min_accel', 0.0) or 0.0)
      planner_response_decel = float(getattr(self, '_dbg_planner_response_decel', 0.0) or 0.0)
      planner_delay_s = float(getattr(self, '_dbg_planner_delay_s', 0.0) or 0.0)
      winding_profile_level = int(getattr(self, '_dbg_winding_profile_level', 0) or 0)
      winding_profile_name = str(getattr(self, '_dbg_winding_profile_name', DEFAULT_WINDING_BEHAVIOR_PROFILE.name) or DEFAULT_WINDING_BEHAVIOR_PROFILE.name)
      winding_profile_source = str(getattr(self, '_dbg_winding_profile_source', 'none') or 'none')
      winding_release_up_slew = float(getattr(self, '_dbg_winding_release_up_slew_mps2', 0.0) or 0.0)
      winding_release_limited = bool(getattr(self, '_dbg_winding_release_limited', False))
      winding_release_shape_active = bool(getattr(self, '_dbg_winding_release_shape_active', False))
      near_apex_release_ready = bool(getattr(self, '_dbg_near_apex_release_ready', False))
      apex_release_lat_acc_ratio = float(getattr(self, '_dbg_apex_release_lat_acc_ratio', 0.0) or 0.0)
      s_vis_m = float(getattr(self, '_dbg_s_visible_m', 0.0))
      fail_open = bool(getattr(self, '_dbg_fail_open', False))
      # FOV/units helpers (may be unset on older builds; default sensibly)
      psi_fov = float(getattr(self, '_psi_fov_rad', 0.49))
      psi_margin = float(getattr(self, '_psi_margin_rad', 0.087))
      ttfov = float(getattr(self, '_dbg_ttfov_s', 0.0))
      psi_vis = float(abs(k_vis) * max(0.0, s_vis_m))
      psi_thresh = float(max(0.0, psi_fov - psi_margin))
      occl_reason = str(getattr(self, '_fov_reason', '') or '')
      occl_on = int(getattr(self, '_fov_on_cnt', 0))
      occl_off = int(getattr(self, '_fov_off_cnt', 0))
      gamma_eff = float(getattr(self, '_dbg_gamma_eff', 0.0))
      units_ok = bool(getattr(self, '_dbg_units_ok', True))
      boost_left = int(getattr(self, '_fov_boost_left', 0))
      overshoot_left = int(getattr(self, '_fov_overshoot_left', 0))
      cap_hold_active = bool(getattr(self, '_dbg_vturn_hold_active', False))
      cap_hold_min = float(getattr(self, '_dbg_vturn_hold_min', 0.0) or 0.0)
      low_speed_calibration_active = bool(getattr(self, '_dbg_low_speed_calibration_active', False))
      low_speed_calibration_reason = str(getattr(self, '_dbg_low_speed_calibration_reason', '') or '')
      low_speed_calibration_headroom = float(getattr(self, '_dbg_low_speed_calibration_headroom', 0.0) or 0.0)
      low_speed_calibration_headroom_ema = float(getattr(self, '_dbg_low_speed_calibration_headroom_ema', 0.0) or 0.0)
      low_speed_calibration_override_ema = float(getattr(self, '_dbg_low_speed_calibration_override_ema', 0.0) or 0.0)
      low_speed_calibration_divergence_mps = float(getattr(self, '_dbg_low_speed_calibration_divergence_mps', 0.0) or 0.0)
      low_speed_calibration_scale = float(getattr(self, '_dbg_low_speed_calibration_scale', 1.0) or 1.0)
      low_speed_calibration_curve_mph = float(getattr(self, '_dbg_low_speed_calibration_curve_mph', 0.0) or 0.0)
      low_speed_calibration_output = float(getattr(self, '_dbg_low_speed_calibration_output', 0.0) or 0.0)
      low_speed_calibration_gap = float(getattr(self, '_dbg_low_speed_calibration_gap', 0.0) or 0.0)
      low_speed_calibration_gap_ratio = float(getattr(self, '_dbg_low_speed_calibration_gap_ratio', 0.0) or 0.0)
      low_speed_calibration_saturated = bool(getattr(self, '_dbg_low_speed_calibration_saturated', False))
      low_speed_calibration_map_active = bool(getattr(self, '_dbg_low_speed_calibration_map_active', False))
      low_speed_calibration_map_anchor_k = float(getattr(self, '_dbg_low_speed_calibration_map_anchor_k', 0.0) or 0.0)
      low_speed_calibration_dual_bin = bool(getattr(self, '_dbg_low_speed_calibration_dual_bin', False))
      low_speed_calibration_base_state = float(getattr(self, '_low_speed_calibration_base_state', 0.0) or 0.0)
      low_speed_calibration_override_state = float(getattr(self, '_low_speed_calibration_override_state', 0.0) or 0.0)
      return {
        'v': v_ego, 'cruise': v_cruise, 'lead': lead, 'hw': hw,
        'conf': conf, 'vision_status': self._vision_status_str(),
        'k_model': k_model, 'k_steer': k_steer, 'steer_fallback_active': steer_fallback_active,
        'k_occ': k_est, 'k_vis_last': k_vis,
        'is_easing': is_easing, 'abs_curv_rate': abs_cr,
        'v_base': v_phys_base, 'v_occ': v_occ, 'v_vis': v_vis,
        'raw': raw, 'final': final,
        'occl_positive_margin': occl_margin, 'occl_lead_bypass_active': bypass,
        'low_speed_margin_override': low_speed_margin,
        'fov_occluded': fov_occ,
        'vis_horizon_s': vis_h, 'tail_frac': tail_frac, 's_tail': s_tail, 'early_no_raise': enr,
        'map_tail_active': map_active, 'map_tail_cap': map_cap, 'map_tail_start_m': map_start, 'map_tail_coverage': map_cov,
        'map_tail_reason': map_reason, 'map_tail_compute_reason': map_compute_reason,
        'map_profile_source': str(getattr(self, '_map_profile_source', 'none') or 'none'),
        'map_whole_curve_reason': str(getattr(self, '_map_whole_curve_reason', 'missing') or 'missing'),
        'map_whole_curve_generation': int(getattr(getattr(self, '_map_whole_curve_cache', None), 'generation', 0) or 0),
        'map_whole_curve_fingerprint': str(getattr(getattr(self, '_map_whole_curve_cache', None), 'route_fingerprint', '') or ''),
        'comfort_decel': comfort, 'max_adaptive_decel': max_adapt, 'decel_cmd': decel_cmd, 'jerk_cmd': jerk_cmd, 'a_cmd': a_cmd,
        # New fields for quick triage on road
        'active_cap': active_cap, 'vtsc_cmd': vtsc_cmd,
        'cap_source': str(getattr(self, '_dbg_cap_source', '')),
        'cap_visible_vmin': cap_vis, 'cap_occl_vmin': cap_occ, 'cap_map_vmin': cap_map,
        'strategy_mode': strategy_mode, 'strategy_state': strategy_state,
        'map_advisory_cap': map_advisory_cap, 'map_strategic_cap': map_strategic_cap,
        'vision_local_cap': vision_local_cap, 'selected_cap': selected_cap,
        'map_floor_active': map_floor_active, 'map_floor_reason': map_floor_reason,
        'vision_relax_allowed': vision_relax_allowed, 'vision_relax_reason': vision_relax_reason,
        'map_floor_anchor_dist_m': map_anchor_dist, 'map_floor_anchor_k': map_anchor_k,
        'map_takeover_dwell_s': takeover_dwell_s, 'map_counterevidence_dwell_s': counterevidence_dwell_s,
        'planner_min_accel_mps2': planner_min_accel, 'planner_response_decel_mps2': planner_response_decel,
        'planner_response_delay_s': planner_delay_s,
        'winding_profile_level': winding_profile_level,
        'winding_profile_name': winding_profile_name,
        'winding_profile_source': winding_profile_source,
        'winding_release_up_slew_mps2': winding_release_up_slew,
        'winding_release_limited': winding_release_limited,
        'winding_release_shape_active': winding_release_shape_active,
        'near_apex_release_ready': near_apex_release_ready,
        'apex_release_lat_acc_ratio': apex_release_lat_acc_ratio,
        's_visible_m': s_vis_m, 'kappa_vis': k_vis, 'path_conf': raw_conf, 'raw_path_conf': raw_conf,
        'winding_road_active': bool(getattr(self, '_winding_road_active', False)),
        'winding_road_score': float(getattr(self, '_winding_road_score', 0.0) or 0.0),
        'winding_road_horizon_m': float(getattr(self, '_winding_road_horizon_m', 0.0) or 0.0),
        'winding_reference_vsafe_mps': float(getattr(self, '_winding_reference_vsafe_mps', 0.0) or 0.0),
        'winding_min_anchor_vsafe_mps': float(getattr(self, '_winding_min_anchor_vsafe_mps', 0.0) or 0.0),
        'winding_anchor_count': int(getattr(self, '_winding_anchor_count', 0) or 0),
        'winding_short_gap_count': int(getattr(self, '_winding_short_gap_count', 0) or 0),
        'winding_curve_distance_m': float(getattr(self, '_winding_curve_distance_m', 0.0) or 0.0),
        'mapd_winding_valid': bool(getattr(self, '_mapd_winding_valid', False)),
        'mapd_winding_level': int(getattr(self, '_mapd_winding_level', 0) or 0),
        'mapd_winding_score': int(getattr(self, '_mapd_winding_score', 0) or 0),
        'mapd_winding_confidence': int(getattr(self, '_mapd_winding_confidence', 0) or 0),
        'mapd_winding_current_level': int(getattr(self, '_mapd_winding_current_level', 0) or 0),
        'mapd_winding_current_score': int(getattr(self, '_mapd_winding_current_score', 0) or 0),
        'mapd_winding_current_confidence': int(getattr(self, '_mapd_winding_current_confidence', 0) or 0),
        'mapd_winding_way_count': int(getattr(self, '_mapd_winding_way_count', 0) or 0),
        'winding_context_active': bool(getattr(self, '_winding_context_active', False)),
        'winding_context_score': float(getattr(self, '_winding_context_score', 0.0) or 0.0),
        'winding_context_level': int(getattr(self, '_winding_context_level', 0) or 0),
        'winding_context_confidence': float(getattr(self, '_winding_context_confidence', 0.0) or 0.0),
        'winding_context_source': str(getattr(self, '_winding_context_source', 'none') or 'none'),
        'occluded': bool(not getattr(self._occlusion_state, 'vision_good', True)),
        'fail_open': fail_open,
        'psi_vis': psi_vis, 'psi_thresh': psi_thresh, 'ttfov_s': ttfov, 'psi_fov_rad': psi_fov, 'psi_margin_rad': psi_margin,
        'occlusion_reason': occl_reason, 'occl_on_cnt': occl_on, 'occl_off_cnt': occl_off, 'onset_boost_left': boost_left, 'overshoot_left': overshoot_left,
        'cap_hold_active': cap_hold_active, 'cap_hold_min': cap_hold_min,
        'low_speed_calibration_active': low_speed_calibration_active,
        'low_speed_calibration_reason': low_speed_calibration_reason,
        'low_speed_calibration_headroom': low_speed_calibration_headroom,
        'low_speed_calibration_headroom_ema': low_speed_calibration_headroom_ema,
        'low_speed_calibration_override_ema': low_speed_calibration_override_ema,
        'low_speed_calibration_divergence_mps': low_speed_calibration_divergence_mps,
        'low_speed_calibration_state': float(getattr(self, '_low_speed_calibration_state', 0.0) or 0.0),
        'low_speed_calibration_base_state': low_speed_calibration_base_state,
        'low_speed_calibration_override_state': low_speed_calibration_override_state,
        'low_speed_calibration_scale': low_speed_calibration_scale,
        'low_speed_calibration_curve_mph': low_speed_calibration_curve_mph,
        'low_speed_calibration_output': low_speed_calibration_output,
        'low_speed_calibration_gap': low_speed_calibration_gap,
        'low_speed_calibration_gap_ratio': low_speed_calibration_gap_ratio,
        'low_speed_calibration_saturated': low_speed_calibration_saturated,
        'low_speed_calibration_map_active': low_speed_calibration_map_active,
        'low_speed_calibration_map_anchor_k': low_speed_calibration_map_anchor_k,
        'low_speed_calibration_dual_bin': low_speed_calibration_dual_bin,
        # κ-bias diagnostics
        'onset_bias_active': bool(getattr(self, '_dbg_onset_bias_active', False)),
        'onset_gate_reason': getattr(self, '_dbg_onset_gate_reason', None),
        'gamma_eff': gamma_eff, 'units_ok': units_ok,
        # Occlusion arbitration breadcrumbs
        'psi_gate_est': float(getattr(self, '_dbg_psi_est', 0.0)),
        'psi_gate_thresh': float(getattr(self, '_psi_thresh_rad', PSI_THRESH_RAD)),
        'consider_occl_gate': bool(getattr(self, '_dbg_consider_occl', False)),
        'double_cap_guard': bool(getattr(self, '_dbg_double_cap_guard', False)),
        'pre_cap_target': float(getattr(self, '_pre_cap_target_speed', 0.0)),
        # Phase-offset diagnostics
        'curve_phase_offset_s': curve_phase_offset_raw,
        'curve_phase_offset_effective_s': curve_phase_offset_effective,
        'curve_sample_idx': int(getattr(self, '_curve_sample_idx', 0)),
        'overshoot_trigger_in_s': float(getattr(self, '_overshoot_trigger_in_s', 0.0)),
        'overshoot_cap_active': bool(getattr(self, '_overshoot_cap_active', False)),
        'apex_trigger_idx': int(getattr(self, '_apex_trigger_idx', 0)),
        'apex_exit_ready': bool(getattr(self, '_apex_exit_ready', False)),
        # Duplicated with _dbg_* names for watcher compatibility
        '_dbg_psi_est': float(getattr(self, '_dbg_psi_est', 0.0)),
        '_dbg_psi_thresh': float(getattr(self, '_psi_thresh_rad', PSI_THRESH_RAD)),
        '_dbg_consider_occl': bool(getattr(self, '_dbg_consider_occl', False)),
        '_dbg_double_cap_guard': bool(getattr(self, '_dbg_double_cap_guard', False)),
        '_dbg_pre_cap_target': float(getattr(self, '_pre_cap_target_speed', 0.0)),
      }
    except Exception:
      return {}

  # Pure FOV-based occlusion gating helper
  @staticmethod
  def occlusion_gate(kappa_vis: float, s_visible_m: float, path_conf: float,
                     psi_fov_rad: float, psi_margin_rad: float,
                     k_freeway: float = FREEWAY_CURV_EPS,
                     k_min: float = 2e-4, s_long: float = 120.0,
                     state: dict | None = None) -> tuple[bool, dict, str, dict]:
    state = dict(state or {})
    on_cnt = int(state.get('on_cnt', 0))
    off_cnt = int(state.get('off_cnt', 0))
    occluded = bool(state.get('occluded', False))
    psi_vis = abs(float(kappa_vis)) * max(0.0, float(s_visible_m))
    psi_thresh = max(0.0, float(psi_fov_rad) - float(psi_margin_rad))
    onset = (abs(kappa_vis) >= k_min) and (psi_vis >= psi_thresh)
    clear = (abs(kappa_vis) < k_freeway) or ((s_visible_m >= s_long) and (psi_vis < psi_thresh) and (path_conf >= 0.6))
    N_on, N_off = 5, 10
    reason = 'none'
    if onset:
      on_cnt += 1
      off_cnt = 0
      if on_cnt >= N_on:
        occluded = True
        reason = 'fov_exit'
    elif clear:
      off_cnt += 1
      on_cnt = 0
      if off_cnt >= N_off:
        occluded = False
        reason = 'freeway' if abs(kappa_vis) < k_freeway else 'short_vis'
    else:
      on_cnt = max(0, on_cnt - 1)
      off_cnt = max(0, off_cnt - 1)
    return occluded, {'on_cnt': on_cnt, 'off_cnt': off_cnt, 'occluded': occluded}, reason, {'psi_vis': psi_vis, 'psi_thresh': psi_thresh}

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self, value):
    if value != self._state:
      _debug(f'TVC: TurnVisionController state: {_description_for_state(value)}')
    self._state = value

  @property
  def adaptive_decel_active(self):
    """Adaptive deceleration system status for external monitoring."""
    return abs(self._current_decel) > abs(self._comfort_decel_limit) * 1.1

  @property
  def decel_requirement(self):
    """Current deceleration requirement for external monitoring."""
    return self._filtered_decel_requirement

  @property
  def a_target(self):
    return self._a_target if self.is_active else self._a_ego

  @property
  def v_turn(self):
    # VTSC output for longitudinal planner ingestion.
    #
    # IMPORTANT:
    # - The longitudinal planner consumes `v_turn` as a *speed cap* (min-of-sources).
    # - `v_turn` must therefore represent VTSC's latest computed recommendation, not a
    #   speed trajectory integrated from VTSC's internal accel state (which can create
    #   a feedback loop where the cap sticks near `v_ego` and prevents recovery).
    #
    # `_v_turn_output` is set each update() in `_update_solution()`.
    try:
      v_out = float(getattr(self, '_v_turn_output', 0.0) or 0.0)
    except Exception:
      v_out = 0.0
    if v_out > 0.0:
      return v_out
    # Fallback for very early init / offline contexts.
    return float(getattr(self, '_v_cruise_setpoint', 0.0) or 0.0)

  @property
  def current_lat_acc(self):
    return self._current_lat_acc

  @property
  def max_pred_lat_acc(self):
    return self._max_pred_lat_acc

  @property
  def is_active(self):
    # SIMPLIFIED: Always active when system is enabled - let longitudinal planner's min() decide usage
    return self._op_enabled and self._is_enabled and not self._gas_pressed

  @property
  def is_entering(self):
    return self._state == VisionTurnControllerState.entering

  @property
  def is_turning(self):
    return self._state == VisionTurnControllerState.turning

  @property
  def is_leaving(self):
    return self._state == VisionTurnControllerState.leaving

  @property
  def distance(self):
    """Distance to lateral acceleration overshoot point."""
    return self._v_overshoot_distance if hasattr(self, '_v_overshoot_distance') else 200.0

  # ===== Rally co-pilot / HUD curve preview (map-enriched) =====
  @property
  def curve_preview_valid(self) -> bool:
    return bool(getattr(self, '_curve_preview_valid', False))

  @property
  def curve_preview_distance_m(self) -> float:
    return float(getattr(self, '_curve_preview_distance_m', 0.0) or 0.0)

  @property
  def curve_preview_time_to_s(self) -> float:
    return float(getattr(self, '_curve_preview_time_to_s', 0.0) or 0.0)

  @property
  def curve_preview_kappa_max(self) -> float:
    return float(getattr(self, '_curve_preview_kappa_max', 0.0) or 0.0)

  @property
  def curve_preview_direction(self) -> int:
    # capnp enum value for VisionTurnSpeedControl.TurnDirection
    return int(getattr(self, '_curve_preview_direction', 0) or 0)

  @property
  def curve_preview_severity(self) -> int:
    # capnp enum value for VisionTurnSpeedControl.CurveSeverity
    return int(getattr(self, '_curve_preview_severity', 0) or 0)

  @property
  def curve_preview_points(self) -> list[tuple[float, float]]:
    pts = getattr(self, '_curve_preview_points', None)
    return list(pts) if isinstance(pts, list) else []

  @property
  def curve_preview_tiles(self) -> list[dict]:
    tiles = getattr(self, '_curve_preview_tiles', None)
    if not isinstance(tiles, list):
      return []

    out: list[dict] = []
    for tile in tiles:
      if not isinstance(tile, dict):
        continue
      pts_raw = tile.get('points', [])
      pts: list[tuple[float, float]] = []
      if isinstance(pts_raw, list):
        for pt in pts_raw:
          try:
            x_fwd = float(pt[0])
            y_left = float(pt[1])
          except Exception:
            continue
          if math.isfinite(x_fwd) and math.isfinite(y_left):
            pts.append((x_fwd, y_left))
      if len(pts) < 2:
        continue
      try:
        tile_id = int(tile.get('id', 0) or 0)
        distance_m = float(tile.get('distance_m', 0.0) or 0.0)
        time_to_s = float(tile.get('time_to_s', 0.0) or 0.0)
        direction = int(tile.get('direction', 0) or 0)
        severity = int(tile.get('severity', 0) or 0)
        max_curvature = float(tile.get('max_curvature', 0.0) or 0.0)
        advisory_speed_mps = float(tile.get('advisory_speed_mps', 0.0) or 0.0)
      except Exception:
        continue
      out.append({
        'id': tile_id,
        'distance_m': distance_m,
        'time_to_s': time_to_s,
        'direction': direction,
        'severity': severity,
        'max_curvature': max_curvature,
        'advisory_speed_mps': advisory_speed_mps,
        'points': pts,
      })
    return out

  @property
  def curve_preview_branch_stubs(self) -> list[dict]:
    stubs = getattr(self, '_curve_preview_branch_stubs', None)
    if not isinstance(stubs, list):
      return []

    out: list[dict] = []
    for stub in stubs:
      if not isinstance(stub, dict):
        continue
      pts_raw = stub.get('points', [])
      pts: list[tuple[float, float]] = []
      if isinstance(pts_raw, list):
        for pt in pts_raw:
          try:
            x_fwd = float(pt[0])
            y_left = float(pt[1])
          except Exception:
            continue
          if math.isfinite(x_fwd) and math.isfinite(y_left):
            pts.append((x_fwd, y_left))
      if len(pts) >= 2:
        out.append({
          'highlighted': bool(stub.get('highlighted', False)),
          'points': pts,
        })
    return out

  def getCurrentLateralAccel(self):
    """Return current lateral acceleration for HUD display."""
    return self._current_lat_acc

  def _reset(self):
    self._current_lat_acc = 0.
    self._max_v_for_current_curvature = 0.
    self._max_pred_lat_acc = 0.
    self._v_overshoot_distance = 200.
    self._lat_acc_overshoot_ahead = False
    self._overshoot_trigger_in_s = float('inf')
    self._overshoot_cap_active = False
    self._near_apex_release_until = 0.0

    # Reset adaptive deceleration system
    self._current_decel = 0.0
    self._filtered_decel_requirement = 0.0
    self._decel_hysteresis_state = False

    # Reset vision occlusion state
    self._occlusion_state = VisionOcclusionState()

    # Reset apex tracking
    self._apex_indices = []
    self._distance_past_apex = 0.0
    self._apex_exit_ready = False
    self._prev_apex_exit_ready = False
    self._apex_trigger_idx = 0
    self._curve_sample_idx = 0
    self._curvature_trajectory = []

    # Reset advanced controller state (preserve current_accel to avoid jerk spikes)
    # Do not zero _current_accel here; preserve continuity across state transitions
    self._prev_target_speed = self._v_ego if hasattr(self, '_v_ego') else 0.0
    self._filtered_curvature = 0.0
    # Track cruise setpoint changes (e.g., speed-limit steps)
    self._prev_v_cruise_setpoint = getattr(self, '_prev_v_cruise_setpoint', 0.0)
    self._limit_step_until = 0.0
    self._suppress_raise_due_to_limit = False

    # Reset anticipatory deceleration state
    self._is_decelerating_for_curve = False
    self._v_turn_release_shape_active = False
    self._clear_winding_behavior_profile()

  def _low_speed_calibration_full_max_mph(self) -> float:
    high_end_mph = float(max(
      float(LOW_SPEED_CALIB_TARGET_FULL_MIN_MPH),
      float(getattr(self, "_low_speed_calibration_high_end_mph", LOW_SPEED_CALIB_TARGET_END_MPH)),
    ))
    fade_width_mph = float(max(1.0, float(LOW_SPEED_CALIB_TARGET_HIGH_END_FADE_MPH)))
    return float(max(float(LOW_SPEED_CALIB_TARGET_FULL_MIN_MPH), high_end_mph - fade_width_mph))

  def _low_speed_calibration_taper(self, target_speed_mph: float) -> float:
    speed_mph = float(target_speed_mph)
    if speed_mph <= float(LOW_SPEED_CALIB_TARGET_START_MPH):
      return 0.0
    if speed_mph < float(LOW_SPEED_CALIB_TARGET_FULL_MIN_MPH):
      return float((speed_mph - float(LOW_SPEED_CALIB_TARGET_START_MPH)) /
                   max(1e-3, float(LOW_SPEED_CALIB_TARGET_FULL_MIN_MPH - LOW_SPEED_CALIB_TARGET_START_MPH)))
    high_full_max_mph = float(self._low_speed_calibration_full_max_mph())
    high_end_mph = float(max(
      high_full_max_mph,
      float(getattr(self, "_low_speed_calibration_high_end_mph", LOW_SPEED_CALIB_TARGET_END_MPH)),
    ))
    if speed_mph <= high_full_max_mph:
      return 1.0
    if speed_mph < high_end_mph:
      return float((high_end_mph - speed_mph) /
                   max(1e-3, high_end_mph - high_full_max_mph))
    return 0.0

  def _low_speed_calibration_scale(self, abs_curvature_meters: float) -> float:
    try:
      kappa = float(max(1e-8, abs_curvature_meters))
    except Exception:
      return 1.0
    try:
      base_speed_mph = float(curvature_to_speed(kappa)) * CV.MS_TO_MPH
    except Exception:
      return 1.0
    taper = float(self._low_speed_calibration_taper(base_speed_mph))
    base_state = float(getattr(self, '_low_speed_calibration_base_state', 0.0) or 0.0)
    override_state = float(self._low_speed_calibration_override_state_for_curvature(kappa))
    effective_state = float(taper * base_state + override_state)
    return float(clip(
      1.0 + effective_state,
      1.0 - float(LOW_SPEED_CALIB_MAX_TIGHTEN),
      1.0 + float(LOW_SPEED_CALIB_MAX_RELAX + LOW_SPEED_CALIB_OVERRIDE_MAX_RELAX),
    ))

  def _curve_speed(self, abs_curvature_meters: float) -> float:
    return float(curvature_to_speed(
      abs_curvature_meters,
      low_speed_sigmoid_scale=self._low_speed_calibration_scale(abs_curvature_meters),
    ))

  @staticmethod
  def _whole_curve_speed(abs_curvature_meters: float) -> float:
    """Live fallback for v3 hash mismatch, without learned calibration.

    A matching v3 profile supplies its physics-only speed. Until mapd
    republishes after a live physics edit, this path converts the published
    profile curvature with the current sigmoid/Q data instead.
    """
    return float(curvature_to_speed(abs(float(abs_curvature_meters)), low_speed_sigmoid_scale=1.0))

  def _read_lateral_feedback(self, sm) -> dict | None:
    controls_state = _sm_get_optional(sm, 'controlsState')
    if controls_state is None:
      return None

    try:
      desired_curvature = abs(float(getattr(controls_state, 'desiredCurvature', 0.0) or 0.0))
    except Exception:
      desired_curvature = 0.0
    try:
      actual_curvature = abs(float(getattr(controls_state, 'curvature', 0.0) or 0.0))
    except Exception:
      actual_curvature = 0.0

    state = None
    try:
      lateral_state = getattr(controls_state, 'lateralControlState', None)
      which = lateral_state.which() if lateral_state is not None and hasattr(lateral_state, 'which') else None
      state = getattr(lateral_state, which) if which else None
    except Exception:
      state = None

    try:
      output = abs(float(getattr(state, 'output', 0.0) or 0.0))
    except Exception:
      output = 0.0
    try:
      saturated = bool(getattr(state, 'saturated', False))
    except Exception:
      saturated = False
    try:
      active = bool(getattr(state, 'active', True))
    except Exception:
      active = True

    return {
      'active': bool(active),
      'desired_curvature': float(desired_curvature),
      'actual_curvature': float(actual_curvature),
      'tracking_gap': float(abs(desired_curvature - actual_curvature)),
      'output': float(output),
      'saturated': bool(saturated),
    }

  def _update_low_speed_calibration(self, sm, *, reference_curvature: float) -> None:
    try:
      now_s = float(getattr(time, 'monotonic', time.time)())
    except Exception:
      now_s = time.time()

    last_s = float(getattr(self, '_low_speed_calibration_last_update_s', 0.0) or 0.0)
    dt = 0.05 if last_s <= 0.0 else float(clip(now_s - last_s, 0.01, 0.25))
    self._low_speed_calibration_last_update_s = now_s

    feedback = self._read_lateral_feedback(sm)
    self._dbg_low_speed_calibration_active = False
    self._dbg_low_speed_calibration_reason = "decay_no_feedback"
    self._dbg_low_speed_calibration_headroom = 0.0
    self._dbg_low_speed_calibration_output = 0.0
    self._dbg_low_speed_calibration_gap = 0.0
    self._dbg_low_speed_calibration_gap_ratio = 0.0
    self._dbg_low_speed_calibration_saturated = False
    self._dbg_low_speed_calibration_override_ema = float(getattr(self, '_low_speed_calibration_override_ema', 0.0) or 0.0)
    self._dbg_low_speed_calibration_divergence_mps = 0.0
    self._dbg_low_speed_calibration_map_active = False
    self._dbg_low_speed_calibration_map_anchor_k = 0.0
    self._dbg_low_speed_calibration_dual_bin = False

    if not bool(getattr(self, "_low_speed_calibration_enabled", True)):
      self._low_speed_calibration_state = 0.0
      self._low_speed_calibration_base_state = 0.0
      self._low_speed_calibration_override_state = 0.0
      self._low_speed_calibration_override_profile = self._empty_low_speed_calibration_override_profile()
      self._low_speed_calibration_override_profile_param_state = self._empty_low_speed_calibration_override_profile()
      self._low_speed_calibration_override_profile_persisted_state = self._empty_low_speed_calibration_override_profile()
      self._low_speed_calibration_param_state = 0.0
      self._low_speed_calibration_persisted_state = 0.0
      self._low_speed_calibration_headroom_ema = 0.0
      self._low_speed_calibration_override_ema = 0.0
      self._dbg_low_speed_calibration_reason = "disabled_by_toggle"
      self._dbg_low_speed_calibration_headroom_ema = 0.0
      self._dbg_low_speed_calibration_override_ema = 0.0
      self._dbg_low_speed_calibration_scale = 1.0
      self._refresh_low_speed_calibration_state()
      return

    scale_curvature = float(reference_curvature)
    override_ema = float(getattr(self, '_low_speed_calibration_override_ema', 0.0) or 0.0)
    override_target = 0.0
    driver_override_active = False
    driver_gas_pressed = bool(getattr(self, '_gas_pressed', False))
    # Previous-frame map context (set in _update_solution, read here with 1-frame lag)
    map_tail_active_prev = bool(getattr(self, '_map_tail_active', False))
    map_anchor_k_prev = float(getattr(self, '_map_tail_anchor_k', 0.0) or 0.0)
    map_last_cap_prev = float(getattr(self, '_map_tail_last_cap', 0.0) or 0.0)
    map_context_valid = bool(
      map_tail_active_prev and
      map_anchor_k_prev >= float(LOW_SPEED_CALIB_MIN_CURVATURE) and
      map_last_cap_prev > 0.0
    )
    primary_override_curvature = 0.0
    relevant_common = False
    override_relevant = False
    low_speed_relevant = False
    raw_score = 0.0
    curve_basis_speed_mph = 0.0
    tighten_score = 0.0
    if feedback is None:
      override_alpha = float(dt / max(float(LOW_SPEED_CALIB_OVERRIDE_EMA_TAU_S), dt))
      override_ema = float(max(0.0, override_ema + override_alpha * (0.0 - override_ema)))
    else:
      feedback_curvature = max(
        float(reference_curvature),
        float(feedback['desired_curvature']),
        float(feedback['actual_curvature']),
      )
      scale_curvature = float(feedback_curvature)
      try:
        curve_basis_speed_mph = float(curvature_to_speed(max(1e-8, feedback_curvature))) * CV.MS_TO_MPH
      except Exception:
        curve_basis_speed_mph = 0.0
      taper = float(self._low_speed_calibration_taper(curve_basis_speed_mph))
      baseline_target_cap = float(min(self._v_cruise_setpoint, curvature_to_speed(max(1e-8, feedback_curvature))))
      requested_cap = float(min(self._v_cruise_setpoint, self._curve_speed(max(1e-8, feedback_curvature))))
      vision_curvature_relevant = bool(
        (feedback_curvature >= float(LOW_SPEED_CALIB_MIN_CURVATURE)) and
        ((float(self._v_cruise_setpoint) - baseline_target_cap) >= float(LOW_SPEED_CALIB_MIN_CAP_DELTA_MPS))
      )
      map_curvature_relevant = bool(
        map_context_valid and
        ((float(self._v_cruise_setpoint) - map_last_cap_prev) >= float(LOW_SPEED_CALIB_MIN_CAP_DELTA_MPS))
      )
      relevant_common = bool(
        bool(self._is_enabled) and
        bool(self._op_enabled) and
        bool(feedback['active']) and
        (vision_curvature_relevant or map_curvature_relevant)
      )
      low_speed_relevant = bool(relevant_common and (taper > 0.0) and not driver_gas_pressed)
      override_relevant = bool(relevant_common and not bool(feedback['saturated']))
      driver_override_active = bool(override_relevant and driver_gas_pressed)

      # When map is the binding constraint during a gas override, route the override
      # profile update to the map anchor curvature so the correct bin learns.
      if map_context_valid and driver_gas_pressed:
        primary_override_curvature = float(map_anchor_k_prev)
        self._dbg_low_speed_calibration_map_active = True
        self._dbg_low_speed_calibration_map_anchor_k = float(map_anchor_k_prev)
      else:
        primary_override_curvature = float(scale_curvature)

      gap_ratio = float(feedback['tracking_gap']) / max(float(feedback['desired_curvature']), float(LOW_SPEED_CALIB_MIN_CURVATURE))
      base_relax_score = 0.0

      if low_speed_relevant:
        if bool(feedback['saturated']):
          tighten_score = 1.0
          self._dbg_low_speed_calibration_reason = "tighten_saturated"
        else:
          relax_effort = clip(
            (float(LOW_SPEED_CALIB_RELAX_OUTPUT_MAX) - float(feedback['output'])) /
            max(1e-3, float(LOW_SPEED_CALIB_RELAX_OUTPUT_MAX)),
            0.0, 1.0,
          )
          if bool(LOW_SPEED_CALIB_ENABLE_TRACKING_TRIGGER):
            relax_tracking = clip(
              (float(LOW_SPEED_CALIB_TRACKING_RELAX_RATIO_MAX) - gap_ratio) /
              max(1e-3, float(LOW_SPEED_CALIB_TRACKING_RELAX_RATIO_MAX)),
              0.0, 1.0,
            )
          else:
            relax_tracking = 1.0
          base_relax_score = float(relax_effort * relax_tracking)

          if bool(LOW_SPEED_CALIB_ENABLE_TIGHTEN_EFFORT):
            tighten_effort = clip(
              (float(feedback['output']) - float(LOW_SPEED_CALIB_TIGHTEN_OUTPUT_MIN)) /
              max(1e-3, 1.0 - float(LOW_SPEED_CALIB_TIGHTEN_OUTPUT_MIN)),
              0.0, 1.0,
            )
          else:
            tighten_effort = 0.0
          if bool(LOW_SPEED_CALIB_ENABLE_TRACKING_TRIGGER):
            tighten_tracking = clip(
              (gap_ratio - float(LOW_SPEED_CALIB_TRACKING_TIGHTEN_RATIO_MIN)) /
              max(1e-3, 1.0 - float(LOW_SPEED_CALIB_TRACKING_TIGHTEN_RATIO_MIN)),
              0.0, 1.0,
            )
          else:
            tighten_tracking = 0.0
          tighten_score = float(max(tighten_effort, tighten_tracking))

      if driver_override_active:
        # When map is the binding constraint, measure divergence against the map cap
        # (the constraint the driver is actually overriding), not just the vision cap.
        effective_cap = min(float(requested_cap), float(map_last_cap_prev)) if map_context_valid else float(requested_cap)
        divergence_mps = max(0.0, float(self._v_ego) - float(effective_cap))
        self._dbg_low_speed_calibration_divergence_mps = float(divergence_mps)
        divergence_ratio = clip(
          (float(divergence_mps) - float(LOW_SPEED_CALIB_OVERRIDE_DIVERGENCE_DEADBAND_MPS)) /
          max(1e-3, float(LOW_SPEED_CALIB_OVERRIDE_DIVERGENCE_FULL_SCALE_MPS)),
          0.0, 1.0,
        )
        # Weight larger driver-taken divergences increasingly heavier than mild bumps.
        override_target = float(divergence_ratio * (0.5 + 0.5 * divergence_ratio))

      override_alpha = float(dt / max(float(LOW_SPEED_CALIB_OVERRIDE_EMA_TAU_S), dt))
      override_ema = float(clip(override_ema + override_alpha * (float(override_target) - override_ema), 0.0, 1.0))

      if low_speed_relevant:
        if tighten_score > max(base_relax_score, override_ema, 0.0):
          self._dbg_low_speed_calibration_reason = "tighten_saturated" if bool(feedback['saturated']) else "tighten_effort"
        elif override_ema > max(base_relax_score, 0.0):
          self._dbg_low_speed_calibration_reason = "relax_override"
        elif base_relax_score > 0.0:
          self._dbg_low_speed_calibration_reason = "relax_clean"
        else:
          self._dbg_low_speed_calibration_reason = "decay_ambiguous"
      elif driver_override_active:
        self._dbg_low_speed_calibration_reason = "relax_override" if override_ema > 0.0 else "decay_ambiguous"
      elif relevant_common and driver_gas_pressed:
        self._dbg_low_speed_calibration_reason = "driver_override_passthrough"
      else:
        self._dbg_low_speed_calibration_reason = "decay_not_relevant"

      raw_score = float(clip(base_relax_score - tighten_score, -1.0, 1.0))
      self._dbg_low_speed_calibration_output = float(feedback['output'])
      self._dbg_low_speed_calibration_gap = float(feedback['tracking_gap'])
      self._dbg_low_speed_calibration_gap_ratio = float(gap_ratio)
      self._dbg_low_speed_calibration_saturated = bool(feedback['saturated'])

    self._low_speed_calibration_override_ema = float(override_ema)
    ema_alpha = float(dt / max(float(LOW_SPEED_CALIB_HEADROOM_TAU_S), dt))
    ema_prev = float(getattr(self, '_low_speed_calibration_headroom_ema', 0.0) or 0.0)
    ema = float(clip(ema_prev + ema_alpha * (float(raw_score) - ema_prev), -1.0, 1.0))
    self._low_speed_calibration_headroom_ema = ema

    state = float(getattr(self, '_low_speed_calibration_base_state', 0.0) or 0.0)
    if feedback is None or not relevant_common:
      decay = float(LOW_SPEED_CALIB_DECAY_RATE_PER_S) * dt
      if state > 0.0:
        state = max(0.0, state - decay)
      else:
        state = min(0.0, state + decay)
    elif driver_gas_pressed:
      state = float(state)
    elif not low_speed_relevant:
      decay = float(LOW_SPEED_CALIB_DECAY_RATE_PER_S) * dt
      if state > 0.0:
        state = max(0.0, state - decay)
      else:
        state = min(0.0, state + decay)
    elif ema > 0.0:
      state = min(float(LOW_SPEED_CALIB_MAX_RELAX), state + float(LOW_SPEED_CALIB_RELAX_RATE_PER_S) * dt * ema)
      # Per-curvature micro-relax from steering headroom, gated by taper so that going
      # slow at a gentle freeway curve because of a speed limit doesn't contaminate.
      if (float(scale_curvature) >= float(LOW_SPEED_CALIB_MIN_CURVATURE) and
          float(base_relax_score) > 0.0 and float(taper) > 0.0):
        self._apply_low_speed_calibration_override_local_delta(
          max(1e-8, float(scale_curvature)),
          float(LOW_SPEED_CALIB_OVERRIDE_RELAX_RATE_PER_S) *
          float(LOW_SPEED_CALIB_STEERING_HEADROOM_PER_CURVATURE_RATE_FACTOR) *
          dt * float(base_relax_score) * float(taper),
        )
    elif ema < 0.0:
      state = max(-float(LOW_SPEED_CALIB_MAX_TIGHTEN), state + float(LOW_SPEED_CALIB_TIGHTEN_RATE_PER_S) * dt * ema)

    if driver_override_active and override_ema > 0.0:
      delta = float(LOW_SPEED_CALIB_OVERRIDE_RELAX_RATE_PER_S) * dt * override_ema
      override_state = self._apply_low_speed_calibration_override_local_delta(
        max(1e-8, float(primary_override_curvature)),
        delta,
      )
      # Dual-bin: also update experienced-curvature bin when it differs from map anchor
      dual_bin = False
      if (map_context_valid and
          float(scale_curvature) >= float(LOW_SPEED_CALIB_MIN_CURVATURE) and
          abs(float(primary_override_curvature) - float(scale_curvature)) > 1e-6):
        log10_dist = abs(
          math.log10(max(1e-8, float(primary_override_curvature))) -
          math.log10(max(1e-8, float(scale_curvature)))
        )
        sigma = max(1e-3, float(LOW_SPEED_CALIB_OVERRIDE_PROFILE_SIGMA_LOG10))
        if log10_dist > float(LOW_SPEED_CALIB_MAP_OVERRIDE_DUAL_BIN_MIN_SIGMA_DISTANCE) * sigma:
          self._apply_low_speed_calibration_override_local_delta(
            max(1e-8, float(scale_curvature)),
            delta * float(LOW_SPEED_CALIB_MAP_OVERRIDE_DUAL_BIN_SECONDARY_WEIGHT),
          )
          dual_bin = True
      self._dbg_low_speed_calibration_dual_bin = bool(dual_bin)
    elif relevant_common and not driver_gas_pressed and tighten_score > 0.0:
      # Tighten uses experienced curvature — saturation is about what the car is
      # physically doing, not what the map predicts ahead.
      override_state = self._apply_low_speed_calibration_override_local_delta(
        max(1e-8, float(scale_curvature)),
        -float(LOW_SPEED_CALIB_OVERRIDE_TIGHTEN_RATE_PER_S) * dt * tighten_score,
      )
    else:
      read_curvature = float(primary_override_curvature) if map_context_valid else float(scale_curvature)
      override_state = float(self._low_speed_calibration_override_state_for_curvature(max(1e-8, read_curvature)))

    self._low_speed_calibration_base_state = float(clip(state, -float(LOW_SPEED_CALIB_MAX_TIGHTEN), float(LOW_SPEED_CALIB_MAX_RELAX)))
    self._low_speed_calibration_override_state = float(clip(override_state, 0.0, float(LOW_SPEED_CALIB_OVERRIDE_MAX_RELAX)))
    self._low_speed_calibration_param_state = float(self._low_speed_calibration_base_state)
    self._low_speed_calibration_override_profile_param_state = self._clip_low_speed_calibration_override_profile(
      getattr(self, "_low_speed_calibration_override_profile", self._empty_low_speed_calibration_override_profile())
    )
    self._refresh_low_speed_calibration_state()
    scale_report_curvature = float(primary_override_curvature) if map_context_valid else float(scale_curvature)
    try:
      applied_scale = float(self._low_speed_calibration_scale(max(1e-8, scale_report_curvature)))
    except Exception:
      applied_scale = 1.0
    self._dbg_low_speed_calibration_active = bool(abs(applied_scale - 1.0) > 1e-3)
    self._dbg_low_speed_calibration_headroom = float(raw_score)
    self._dbg_low_speed_calibration_headroom_ema = float(ema)
    self._dbg_low_speed_calibration_override_ema = float(override_ema)
    self._dbg_low_speed_calibration_scale = float(applied_scale)
    self._dbg_low_speed_calibration_curve_mph = float(curve_basis_speed_mph)
    self._maybe_persist_low_speed_calibration_state(now_s)

  def _apply_freeway_v_turn_hold(self, v_cap: float) -> float:
    """Hold material VTSC cap reductions briefly to bridge model flicker.

    Motivation: The model curvature horizon can flicker, producing short (<0.5s) cap dips. The
    longitudinal planner/MPC often cannot react within that window, so braking begins late and the
    driver intervenes. Holding the lowest cap briefly makes the cap persistent enough to be acted
    upon, while keeping release responsive.
    """
    try:
      now = float(time.time())
    except Exception:
      now = 0.0

    # Default: no hold
    self._dbg_vturn_hold_active = False

    try:
      v_ego = float(self._v_ego)
    except Exception:
      v_ego = 0.0

    # Gate hold behavior:
    # - At freeway speeds with good confidence, hold helps when the horizon "pulses" a curve.
    # - Under degraded confidence, hold can still stabilize a brief but meaningful cap without
    #   relying on the deprecated occlusion state machine.
    try:
      raw_conf = float(getattr(self, '_last_vision_confidence', getattr(self._occlusion_state, 'smoothed_confidence', 1.0)))
    except Exception:
      raw_conf = 1.0
    try:
      good_conf = float(getattr(self._occlusion_state, 'good_threshold', 0.70))
    except Exception:
      good_conf = 0.70
    confidence_good = raw_conf >= good_conf
    try:
      failopen = bool(getattr(self, '_freeway_failopen_active', False))
    except Exception:
      failopen = False
    fov_occluded = bool(getattr(self, '_fov_occluded', False))
    try:
      turn_evidence = float(getattr(self, '_max_pred_lat_acc', 0.0)) >= float(_ENTERING_PRED_LAT_ACC_TH)
    except Exception:
      turn_evidence = False
    try:
      curve_evidence = abs(float(getattr(self, '_filtered_curvature', 0.0))) >= 0.004
    except Exception:
      curve_evidence = False

    freeway_speed = v_ego >= float(VTURN_HOLD_MIN_V_MPS)
    local_curve_corroborated = bool(self._has_local_curve_corroboration())
    freeway_clean = freeway_speed and confidence_good and (not fov_occluded)
    freeway_hold_supported = bool(local_curve_corroborated)
    low_conf_trigger_ok = (raw_conf < good_conf) and (not failopen) and (turn_evidence or curve_evidence)
    low_conf_hold_supported = bool(low_conf_trigger_ok and ((not freeway_speed) or local_curve_corroborated))

    try:
      v_cruise = float(self._v_cruise_setpoint)
    except Exception:
      v_cruise = float(v_cap)

    # Update/extend hold window only when VTSC is asking for a meaningful reduction.
    overspeed = (v_ego - float(v_cap)) >= 0.5
    hold_trigger_ok = low_conf_hold_supported or (freeway_clean and freeway_hold_supported)
    if hold_trigger_ok and ((v_cruise - float(v_cap)) >= float(VTURN_HOLD_DELTA_MPS)) and (overspeed or freeway_clean):
      hold_until = float(getattr(self, '_v_turn_hold_until', 0.0) or 0.0)
      hold_min = float(getattr(self, '_v_turn_hold_min', float(v_cap)) or float(v_cap))
      if now >= hold_until:
        hold_min = float(v_cap)
      else:
        hold_min = min(hold_min, float(v_cap))
      self._v_turn_hold_min = hold_min
      hold_s = float(VTURN_HOLD_S_OCCLUDED) if low_conf_hold_supported else float(VTURN_HOLD_S)
      self._v_turn_hold_until = now + hold_s

    # Apply hold if active
    hold_until2 = float(getattr(self, '_v_turn_hold_until', 0.0) or 0.0)
    hold_min2 = float(getattr(self, '_v_turn_hold_min', float(v_cap)) or float(v_cap))
    if now < hold_until2:
      self._dbg_vturn_hold_active = True
      self._dbg_vturn_hold_min = float(hold_min2)
      return float(min(float(v_cap), hold_min2))

    # Hold expired: allow immediate release
    self._v_turn_hold_min = float(INF_SPEED)
    return float(v_cap)

  def _has_local_curve_corroboration(self) -> bool:
    try:
      control_curve_evidence = max(
        abs(float(getattr(self, '_dbg_controls_actual_curvature', 0.0) or 0.0)),
        abs(float(getattr(self, '_dbg_controls_desired_curvature', 0.0) or 0.0)),
      ) >= float(VTURN_LOCAL_CURVATURE_CORROBORATION_KAPPA)
    except Exception:
      control_curve_evidence = False
    try:
      steer_curve_evidence = abs(float(getattr(self, '_dbg_k_steer', 0.0) or 0.0)) >= float(STEER_CURVATURE_FALLBACK_MIN_KAPPA)
    except Exception:
      steer_curve_evidence = False
    try:
      map_evidence = bool(getattr(self, '_map_tail_active', False))
    except Exception:
      map_evidence = False
    # Preview-only overshoot state is derived from the same model horizon that can flicker for a
    # single frame on straight freeway segments. Don't treat it as corroboration when deciding
    # whether to latch a published cap or keep the post-hold release shaping alive.
    return bool(control_curve_evidence or steer_curve_evidence or map_evidence)

  def _has_release_shape_curve_evidence(self) -> bool:
    if bool(self._has_local_curve_corroboration()):
      return True

    try:
      freeway_speed = float(getattr(self, '_v_ego', 0.0) or 0.0) >= float(VTURN_HOLD_MIN_V_MPS)
    except Exception:
      freeway_speed = False
    if freeway_speed:
      return False

    try:
      return bool(
        bool(getattr(self, '_overshoot_cap_active', False)) or
        bool(getattr(self, '_lat_acc_overshoot_ahead', False)) or
        (abs(float(getattr(self, '_filtered_curvature', 0.0) or 0.0)) >= 0.0015) or
        (float(getattr(self, '_max_pred_lat_acc', 0.0) or 0.0) >= float(_ENTERING_PRED_LAT_ACC_TH))
      )
    except Exception:
      return False

  def _apply_winding_v_turn_release_slew(self, v_cap: float, dt: float) -> float:
    self._dbg_winding_release_limited = False
    self._dbg_winding_release_up_slew_mps2 = 0.0
    self._dbg_winding_release_shape_active = bool(getattr(self, '_v_turn_release_shape_active', False))

    try:
      desired_cap = max(0.0, float(v_cap))
    except Exception:
      return float(getattr(self, '_v_cruise_setpoint', 0.0) or 0.0)

    profile = getattr(self, '_winding_behavior_profile', DEFAULT_WINDING_BEHAVIOR_PROFILE)
    if getattr(profile, 'level', None) is None:
      return desired_cap

    try:
      prev_cap = max(0.0, float(getattr(self, '_v_turn_output', desired_cap) or desired_cap))
    except Exception:
      prev_cap = desired_cap

    try:
      v_cruise = max(0.0, float(getattr(self, '_v_cruise_setpoint', desired_cap) or desired_cap))
    except Exception:
      v_cruise = desired_cap
    cap_limited_delta = float(VTURN_RELEASE_SHAPE_ENTRY_DELTA_MPS)
    handoff_margin = float(VTURN_RELEASE_SHAPE_HANDOFF_MARGIN_MPS)
    curve_evidence = bool(self._has_release_shape_curve_evidence())

    if curve_evidence and min(desired_cap, prev_cap) <= v_cruise - cap_limited_delta:
      self._v_turn_release_shape_active = True
    elif bool(getattr(self, '_v_turn_release_shape_active', False)) and (desired_cap <= prev_cap + handoff_margin) and (not curve_evidence):
      self._v_turn_release_shape_active = False

    self._dbg_winding_release_shape_active = bool(getattr(self, '_v_turn_release_shape_active', False))
    if desired_cap <= prev_cap + 1e-6:
      return desired_cap
    if not bool(getattr(self, '_v_turn_release_shape_active', False)):
      return desired_cap

    slew_limit = float(profile.v_turn_release_up_slew_mps2)

    if (not math.isfinite(slew_limit)) or slew_limit <= 0.0:
      return desired_cap

    slew_limit *= float(self._pre_apex_release_slew_scale())
    self._dbg_winding_release_up_slew_mps2 = float(slew_limit)
    limited_cap = min(desired_cap, prev_cap + float(slew_limit) * max(0.0, float(dt)))
    self._dbg_winding_release_limited = bool(limited_cap < desired_cap - 1e-6)
    if (limited_cap >= desired_cap - handoff_margin) and (not curve_evidence):
      self._v_turn_release_shape_active = False
      self._dbg_winding_release_shape_active = False
    return float(limited_cap)

  def _effective_apex_release_lat_acc_ratio(self) -> float:
    profile = getattr(self, '_winding_behavior_profile', DEFAULT_WINDING_BEHAVIOR_PROFILE)
    try:
      ratio = float(getattr(profile, 'apex_release_lat_acc_ratio', 0.92))
    except Exception:
      ratio = 0.92
    try:
      # Keep the early-unwind path under the same user timing contract as the geometric
      # apex-exit release: negative offsets advance release, positive offsets delay it.
      ratio += 0.05 * float(getattr(self, '_apex_exit_phase_offset_s', 0.0) or 0.0)
    except Exception:
      pass
    ratio = min(1.0, max(0.20, ratio))
    self._dbg_apex_release_lat_acc_ratio = float(ratio)
    return float(ratio)

  def _pre_apex_release_slew_scale(self) -> float:
    profile = getattr(self, '_winding_behavior_profile', DEFAULT_WINDING_BEHAVIOR_PROFILE)
    if int(getattr(profile, 'level', 0) or 0) <= 0:
      return 1.0
    if bool(getattr(self, '_apex_exit_ready', False)):
      return 1.0
    try:
      peak_lat_acc = max(
        abs(float(getattr(self, '_current_lat_acc', 0.0) or 0.0)),
        abs(float(getattr(self, '_max_pred_lat_acc', 0.0) or 0.0)),
      )
      current_lat_acc = abs(float(getattr(self, '_current_lat_acc', 0.0) or 0.0))
    except Exception:
      return 1.0
    if peak_lat_acc < float(_ENTERING_PRED_LAT_ACC_TH):
      return 1.0

    ratio = float(self._effective_apex_release_lat_acc_ratio())
    trigger_lat_acc = float(ratio) * float(peak_lat_acc)
    if current_lat_acc < trigger_lat_acc - 1e-6:
      return 1.0

    denom = max(1e-3, float(peak_lat_acc) - float(trigger_lat_acc))
    progress = clip((float(current_lat_acc) - float(trigger_lat_acc)) / denom, 0.0, 1.0)
    return float(PRE_APEX_RELEASE_SLEW_MIN_SCALE + (1.0 - PRE_APEX_RELEASE_SLEW_MIN_SCALE) * float(progress))

  def _is_near_apex_release_ready(self) -> bool:
    ratio = float(self._effective_apex_release_lat_acc_ratio())

    try:
      peak_lat_acc = max(
        abs(float(getattr(self, '_current_lat_acc', 0.0) or 0.0)),
        abs(float(getattr(self, '_max_pred_lat_acc', 0.0) or 0.0)),
      )
      current_lat_acc = abs(float(getattr(self, '_current_lat_acc', 0.0) or 0.0))
      has_curve_evidence = bool(
        bool(getattr(self, '_lat_acc_overshoot_ahead', False)) or
        bool(getattr(self, '_overshoot_cap_active', False)) or
        (abs(float(getattr(self, '_filtered_curvature', 0.0) or 0.0)) >= 0.0015)
      )
    except Exception:
      self._dbg_near_apex_release_ready = False
      return False

    ready = bool(
      has_curve_evidence and
      peak_lat_acc >= float(_ENTERING_PRED_LAT_ACC_TH) and
      current_lat_acc >= (ratio * peak_lat_acc)
    )
    self._dbg_near_apex_release_ready = bool(ready)
    return bool(ready)

  def _update_params(self):
    # Delegate to shared reader to avoid duplicating logic here
    update_vtsc_params(self)

  def _calculate_required_deceleration(self, v_current: float, v_target: float, distance: float) -> float:
    """Calculate minimum deceleration required using physics: a = (v_f² - v_i²) / (2d)"""
    if distance <= 0.1:  # Avoid division by zero
      return self._max_adaptive_decel
    
    # Physics formula: a = (v_target² - v_current²) / (2 × distance)
    required_decel = (v_target * v_target - v_current * v_current) / (2.0 * distance)
    
    # Apply safety bias - slightly more aggressive to ensure we reach target
    safety_biased_decel = required_decel * (1.0 + self._safety_bias)
    
    # Clamp to system limits
    return max(safety_biased_decel, self._max_adaptive_decel)

  def _get_optimal_deceleration(self, raw_decel: float, dt: float) -> float:
    """Get optimal deceleration using adaptive physics-based approach with noise filtering."""
    
    # Step 1: Apply EMA filtering to smooth deceleration requirements
    # Initialize filter to first value if not yet initialized (was 0)
    if self._filtered_decel_requirement == 0.0 and raw_decel != 0.0:
        self._filtered_decel_requirement = raw_decel
    else:
        self._filtered_decel_requirement = ((1.0 - self._filter_alpha) * self._filtered_decel_requirement +
                                           self._filter_alpha * raw_decel)
    
    # Step 2: Determine if we should use comfort or adaptive deceleration
    comfort_sufficient = (abs(self._filtered_decel_requirement) <= abs(self._comfort_decel_limit))
    
    # Step 3: Apply hysteresis to prevent oscillation between comfort/adaptive modes
    if comfort_sufficient and not self._decel_hysteresis_state:
        # Comfort deceleration is sufficient and we're not in adaptive mode
        target_decel = max(self._filtered_decel_requirement, self._comfort_decel_limit)
        target_jerk_limit = abs(self._comfort_jerk_limit)
    elif not comfort_sufficient and not self._decel_hysteresis_state:
        # Need to switch to adaptive mode
        self._decel_hysteresis_state = True
        # Use physics-based calculation with system limits
        target_decel = max(self._filtered_decel_requirement, self._max_adaptive_decel)
        target_jerk_limit = abs(self._max_adaptive_jerk)
    elif self._decel_hysteresis_state:
        # Currently in adaptive mode - check if we can return to comfort with hysteresis
        hysteresis_threshold = abs(self._comfort_decel_limit) * (1.0 - self._hysteresis_threshold)
        if abs(self._filtered_decel_requirement) <= hysteresis_threshold:
            self._decel_hysteresis_state = False
            target_decel = max(self._filtered_decel_requirement, self._comfort_decel_limit)
            target_jerk_limit = abs(self._comfort_jerk_limit)
        else:
            # Stay in adaptive mode
            target_decel = max(self._filtered_decel_requirement, self._max_adaptive_decel)
            target_jerk_limit = abs(self._max_adaptive_jerk)
    else:
        # Default case
        target_decel = max(self._filtered_decel_requirement, self._comfort_decel_limit)
        target_jerk_limit = abs(self._comfort_jerk_limit)
    
    # Step 4: Apply jerk limiting for smooth transitions
    max_decel_change = abs(target_jerk_limit) * dt  # Ensure positive limit
    decel_change = target_decel - self._current_decel
    
    if abs(decel_change) > max_decel_change:
        if decel_change > 0:
            self._current_decel += max_decel_change
        else:
            self._current_decel -= max_decel_change
    else:
        self._current_decel = target_decel
    
    return self._current_decel

  def _update_vision_occlusion(self, model_data, current_time: float):
    """Update vision occlusion state and return curvature to use under monotonic occlusion mode."""
    # Dropout detection: if the model publishes too few orientationRate frames, consider it a transient
    # and open a short grace window during which we avoid pretrigger occlusion.
    try:
      z = getattr(getattr(model_data, 'orientationRate', None), 'z', None)
      n_frames = int(len(z)) if z is not None else 0
      if n_frames and n_frames < 33:
        self._dropout_until = float(time.time()) + float(getattr(self, '_dropout_grace_s', 0.40))
    except Exception:
      pass
    try:
      setattr(self._occlusion_state, 'dropout_active', bool(time.time() < float(getattr(self, '_dropout_until', 0.0))))
    except Exception:
      pass
    # Estimate vision confidence
    if model_data is None:
        vision_confidence = 0.0
    else:
        if hasattr(model_data, 'laneLineProbs') and model_data.laneLineProbs:
            vision_confidence = float(np.mean(model_data.laneLineProbs))
        else:
            vision_confidence = 1.0
    self._last_vision_confidence = vision_confidence

    # Candidate current curvature is the filtered curvature we track
    current_curvature = self._filtered_curvature

    if not VTSC_OCCLUSION_ENABLED:
      occ = self._occlusion_state
      good_conf = float(getattr(occ, 'good_threshold', 0.70))
      occ.smoothed_confidence = max(float(vision_confidence), good_conf)
      occ.prev_smoothed_conf = float(occ.smoothed_confidence)
      occ.vision_good = True
      occ.vision_status = VisionStatus.FULL_VISIBILITY
      occ.last_valid_curvature = current_curvature
      occ.extrapolated_curvature = current_curvature
      occ.est_curvature = current_curvature
      occ.entry_curvature = current_curvature
      occ.distance_since_m = 0.0
      occ.occlusion_start_time = 0.0
      occ.occluded_since_time = 0.0
      occ.reacquired_at = 0.0
      occ.dropout_active = False
      self._fast_reacq_until = 0.0
      self._fov_occluded = False
      self._fov_on_cnt = 0
      self._fov_off_cnt = 0
      self._fov_reason = ''
      self._fov_boost_left = 0
      self._fov_overshoot_left = 0
      self._fov_kappa_ewma = current_curvature
      self._occlusion_prev = False
      self._occlusion_onset_timer_s = 0.0
      self._v_cap_active_at_onset_mps = 0.0
      self._onset_no_raise_active = False
      self._occl_lead_bypass_active = False
      return current_curvature

    # Remember vision_good before update to detect reacquisition
    prev_good = self._occlusion_state.vision_good

    # Update occlusion state with vehicle speed and time
    self._occlusion_state.update(current_curvature, vision_confidence, self._v_ego, current_time)
    # While vision is good, keep last_valid_curvature fresh at controller level (compatibility with acceptance tests)
    if self._occlusion_state.vision_good:
      self._occlusion_state.last_valid_curvature = current_curvature

    # On reacquisition, arm a fast filtering window to improve recovery time
    if (not prev_good) and self._occlusion_state.vision_good:
      # Fast window; effective alpha increased later when applied
      now_t = time.time()
      self._fast_reacq_until = now_t + float(getattr(self, '_fast_reacq_window_s', 0.9))
      # Record precise reacquisition moment for accel floor logic
      try:
        self._occlusion_state.reacquired_at = float(now_t)
      except Exception:
        pass
      # Immediately clear any FOV-gated occlusion artifacts to let fresh vision take over
      try:
        self._fov_occluded = False
        self._fov_on_cnt = 0
        self._fov_off_cnt = 0
        self._fov_boost_left = 0
        self._fov_overshoot_left = 0
        self._onset_no_raise_active = False
      except Exception:
        pass

    # If vision is good, use current curvature; otherwise, use estimated curvature under monotonic model
    return current_curvature if self._occlusion_state.vision_good else self._occlusion_state.extrapolated_curvature

  def _monitor_adaptive_deceleration(self, required_decel: float, remaining_distance: float) -> bool:
    """Monitor adaptive deceleration system performance and detect extreme scenarios."""
    # Check if we're using maximum system deceleration
    is_max_decel = abs(self._current_decel) >= abs(self._max_adaptive_decel) * 0.95
    
    # Check if distance is critically short
    is_critical_distance = remaining_distance < 20.0  # meters
    
    # Log adaptive deceleration activation for debugging
    if self._decel_hysteresis_state:
        _debug(f'VTSC: Adaptive decel active - current: {self._current_decel:.2f}, required: {required_decel:.2f}, distance: {remaining_distance:.1f}m')
    
    # Return true if we're in a challenging scenario (for external monitoring)
    return is_max_decel and is_critical_distance

  def _update_calculations(self, sm):
    """Advanced vision-based curvature calculation using direct model outputs."""
    # Be tolerant of lightweight SM stubs in offline tests
    try:
      model_data = sm['modelV2'] if getattr(sm, 'valid', {}).get('modelV2', False) else None
    except Exception:
      model_data = getattr(sm, 'modelV2', None)
      if model_data is None:
        data = getattr(sm, '_data', None)
        if isinstance(data, dict):
          model_data = data.get('modelV2', None)
    # Lead presence/headway estimation from radarState (if available)
    try:
      try:
        rs = sm['radarState'] if getattr(sm, 'valid', {}).get('radarState', False) else None
      except Exception:
        rs = getattr(sm, 'radarState', None)
        if rs is None:
          data = getattr(sm, '_data', None)
          if isinstance(data, dict):
            rs = data.get('radarState', None)
      lead = getattr(rs, 'leadOne', None) if rs is not None else None
      status = bool(getattr(lead, 'status', False)) if lead is not None else False
      d_rel = float(getattr(lead, 'dRel', 1e9)) if lead is not None else 1e9
      v_ego_safe = max(0.1, float(self._v_ego))
      v_floor = max(float(OCCL_BYPASS_HEADWAY_V_FLOOR_MPS), float(_MIN_V))
      v_for_headway = max(v_ego_safe, v_floor)
      headway_s = float(d_rel) / v_for_headway
      self._lead_present = status
      self._lead_d_rel_m = float(d_rel)
      self._lead_headway_s = headway_s
    except Exception:
      self._lead_present = False
      self._lead_headway_s = 99.0
    current_time = time.time()
    controls_state = _sm_get_optional(sm, 'controlsState')
    try:
      self._dbg_controls_desired_curvature = abs(float(getattr(controls_state, 'desiredCurvature', 0.0) or 0.0))
    except Exception:
      self._dbg_controls_desired_curvature = 0.0
    try:
      self._dbg_controls_actual_curvature = abs(float(getattr(controls_state, 'curvature', 0.0) or 0.0))
    except Exception:
      self._dbg_controls_actual_curvature = 0.0

    # Initialize defaults for edge cases (use the last filtered curvature if present).
    # NOTE: vision occlusion state is updated *after* we update `_filtered_curvature` for this frame.
    # This avoids a 1-frame lag where `last_valid_curvature` can be stuck at ~0 when entering a curve,
    # which in turn breaks FOV-occlusion gating and fail-open logic.
    current_curvature_signed = 0.0
    current_curvature = float(getattr(self, '_filtered_curvature', 0.0))
    max_pred_curvature = current_curvature
    # Recomputed per-frame; used to gate planner-facing overshoot cap timing.
    self._overshoot_trigger_in_s = float('inf')
    self._curve_sample_idx = 0

    # Lead-aware occlusion bypass activation
    try:
      # Lead-bypass is intended to prevent the occlusion subsystem from interfering while following
      # a lead at close headway (e.g., lane-line confidence drops behind a lead should not "stick"
      # VTSC in occlusion/no-raise behavior on otherwise benign segments).
      low_speed_lead_close = bool(
        self._lead_present and (v_ego_safe <= float(OCCL_BYPASS_LOW_SPEED_V_MPS)) and (float(self._lead_d_rel_m) <= float(OCCL_BYPASS_LEAD_D_REL_MAX_M))
      )
      self._occl_lead_bypass_active = bool(
        self._occl_bypass_with_lead and self._lead_present and (
          (self._lead_headway_s <= float(self._occl_bypass_headway_s)) or low_speed_lead_close
        )
      )
    except Exception:
      self._occl_lead_bypass_active = False

    # Use advanced method: direct model data access whenever model is available
    if (model_data is not None and
        hasattr(model_data, 'orientationRate') and hasattr(model_data, 'velocity') and
        model_data.orientationRate.z is not None and model_data.velocity.x is not None):

      orientation_rate_raw = model_data.orientationRate.z
      velocity_pred_raw = model_data.velocity.x

      MIN_POINTS = 3
      if (len(orientation_rate_raw) >= MIN_POINTS and len(velocity_pred_raw) >= MIN_POINTS):
        # Use direct model outputs for curvature calculation
        n_points = int(min(len(orientation_rate_raw), len(velocity_pred_raw), N_POINTS))
        # Ensure n_points is a pure Python int for Cap'n Proto compatibility
        n_points = int(n_points)
        # Advance model-time interpretation to remove systematic VTSC lag.
        # User offset convention: lower values start earlier, higher values start later.
        times_nominal = np.array(ModelConstants.T_IDXS[:n_points], dtype=float)
        base_phase_advance_s = float(VTSC_TRAJECTORY_PHASE_ADVANCE_S)
        curve_phase_offset_s = float(effective_curve_phase_offset_s(float(getattr(self, '_curve_phase_offset_s', 0.0))))
        phase_advance_s = float(clip(base_phase_advance_s - curve_phase_offset_s, 0.0, float(times_nominal[-1])))
        lead_idx = int(np.searchsorted(times_nominal, phase_advance_s, side='left'))
        lead_idx = int(min(max(lead_idx, 0), n_points - 1))
        self._curve_sample_idx = int(lead_idx)
        times_for_planning = np.maximum(0.0, times_nominal - phase_advance_s)
        # FIXED: Preserve sign information - don't use np.abs() here!
        orientation_rate_signed = np.array(list(orientation_rate_raw)[:n_points], dtype=float)
        velocity_pred = np.array(list(velocity_pred_raw)[:n_points], dtype=float)

        # Compute curvature array with SIGNED values.
        # Model orientationRate.z is yaw rate (rad/s). Curvature κ = yaw_rate / speed (1/m).
        # Use predicted velocity to convert; clamp very low speeds to avoid blow-ups.
        v_clip = np.clip(velocity_pred, 0.1, None)
        curvature_array_signed = orientation_rate_signed / v_clip
        # For max calculations, use absolute values
        curvature_array_abs = np.abs(curvature_array_signed)
        max_pred_curvature = float(np.max(curvature_array_abs))
        # expose for debug snapshot
        self._dbg_k_model = max_pred_curvature

        lane_change_active = False
        try:
          lane_change_active = bool(getattr(getattr(model_data, 'meta', None), 'laneChangeState', LaneChangeState.off) != LaneChangeState.off)
        except Exception:
          lane_change_active = False
        self._lane_change_active = bool(lane_change_active)

        # Calculate lateral acceleration using model-predicted curvature
        # This is more accurate than steering angle at highway speeds
        # Use the current model-predicted curvature WITH SIGN preserved
        if len(curvature_array_signed) > 0:
          sample_idx = int(min(max(self._curve_sample_idx, 0), len(curvature_array_signed) - 1))
          current_curvature = float(curvature_array_abs[sample_idx])  # Absolute value for calculations
          current_curvature_signed = float(curvature_array_signed[sample_idx])  # Signed value for lateral accel

        # Steering-curvature fallback when the model says "straight" but steering indicates a
        # real curve. This avoids "fail open" behavior on sharp bends when the model curvature
        # momentarily flattens.
        self._dbg_k_steer = 0.0
        self._dbg_steer_fallback_active = False
        try:
          llp = getattr(model_data, 'laneLineProbs', None)
          vision_confidence = float(np.mean(llp)) if llp else 1.0
        except Exception:
          vision_confidence = 1.0
        kappa_steer = 0.0
        kappa_steer_abs = 0.0
        try:
          if self._vm is not None and float(self._v_ego) >= LANE_CHANGE_CURVATURE_SUPPRESS_MIN_V_MPS:
            sa_rad = math.radians(float(getattr(self, '_steering_angle_deg', 0.0)))
            kappa_steer = float(self._vm.calc_curvature(sa_rad, float(self._v_ego), 0.0))
            kappa_steer_abs = abs(kappa_steer)
            self._dbg_k_steer = float(kappa_steer_abs)
            if (float(max_pred_curvature) <= STEER_CURVATURE_FALLBACK_MODEL_KAPPA_MAX and
                float(self._v_ego) >= STEER_CURVATURE_FALLBACK_MIN_V_MPS and
                kappa_steer_abs >= STEER_CURVATURE_FALLBACK_MIN_KAPPA):
              current_curvature = max(float(current_curvature), float(kappa_steer_abs))
              current_curvature_signed = float(kappa_steer)
              self._dbg_steer_fallback_active = True
        except Exception:
          pass

        if lane_change_active and kappa_steer_abs < LANE_CHANGE_CURVATURE_SUPPRESS_STEER_KAPPA_MAX:
          curvature_array_signed = np.zeros_like(curvature_array_signed)
          curvature_array_abs = np.zeros_like(curvature_array_abs)
          max_pred_curvature = 0.0
          current_curvature = 0.0
          current_curvature_signed = 0.0
          self._dbg_k_model = 0.0
          self._dbg_steer_fallback_active = False

        # Store curvature trajectory and detect apexes (use absolute values for apex detection)
        self._curvature_trajectory = curvature_array_abs.tolist()
        raw_apex_indices = find_apexes_enhanced(curvature_array_abs, self._apex_threshold, self._apex_prominence)
        if lead_idx > 0 and raw_apex_indices:
          self._apex_indices = sorted({max(0, int(i) - lead_idx) for i in raw_apex_indices})
        else:
          self._apex_indices = raw_apex_indices
        _debug(f'TVC: Found {len(self._apex_indices)} apexes at indices: {self._apex_indices}')

        # Update filtered curvature using EMA of the NEAR-TERM curvature, not the horizon max.
        # Using the max across the horizon makes the "visible" path act like an occlusion cap
        # and causes premature, persistent overslow. Filter toward the instantaneous curvature instead.
        self._filtered_curvature = ((1 - self._curvature_ema_ratio) * self._filtered_curvature +
                                   self._curvature_ema_ratio * current_curvature)

        # Calculate lateral accelerations using model predictions (not steering angle)
        self._current_lat_acc = current_curvature_signed * self._v_ego**2
        self._max_pred_lat_acc = self._v_ego**2 * max_pred_curvature

        self._update_low_speed_calibration(sm, reference_curvature=float(current_curvature))

        # Calculate safe speed using the calibrated low-speed envelope.
        self._max_v_for_current_curvature = self._curve_speed(current_curvature) if current_curvature > 0 else V_CRUISE_MAX * CV.KPH_TO_MS

        # Check for overshoot using curvature_to_speed method (use absolute values)
        safe_speeds = np.array([self._curve_speed(curv) for curv in curvature_array_abs])
        # Under very low lane-line confidence, be mildly conservative when deciding whether we need
        # to start slowing for a curve ahead. This helps blind off-ramps where the model curvature
        # estimate can rise sharply only very late in the approach.
        try:
          conf_for_scale = float(vision_confidence)
        except Exception:
          conf_for_scale = 1.0
        try:
          lead_bypass = bool(self._occl_lead_bypass_active)
        except Exception:
          lead_bypass = False
        if (not lead_bypass) and (conf_for_scale < CONFIDENCE_EXIT_TO_PARTIAL):
          try:
            denom = float(max(1e-6, CONFIDENCE_EXIT_TO_PARTIAL - CONFIDENCE_ENTER_SEVERE))
            t = float(clip((conf_for_scale - CONFIDENCE_ENTER_SEVERE) / denom, 0.0, 1.0))
          except Exception:
            t = 0.0
          scale = float(SEVERE_OVERSHOOT_SPEED_SCALE_MIN + (1.0 - SEVERE_OVERSHOOT_SPEED_SCALE_MIN) * t)
          safe_speeds = safe_speeds * scale
        overshoot_mask = safe_speeds < self._v_ego
        self._lat_acc_overshoot_ahead = np.any(overshoot_mask)

        # Update occlusion state using the curvature we've just computed/filtered for this frame.
        # This keeps `last_valid_curvature` and confidence gating aligned with the *current* frame's
        # `_filtered_curvature`, avoiding a 1-frame lag at curve entry.
        _ = self._update_vision_occlusion(model_data, current_time)

        if self._lat_acc_overshoot_ahead:
          # PROPER FIX: Consider ALL points requiring deceleration, not just first or tightest
          # Calculate which points need immediate action based on deceleration requirements
          overshoot_indices = np.where(overshoot_mask)[0]

          # For each point that needs slowing, calculate if we need to start NOW
          max_decel = max(0.1, float(self._planning_decel_limit))  # m/s² planning decel limit
          immediate_requirements = []

          for idx in overshoot_indices:
            # How much distance do we need to slow down to this point's safe speed?
            speed_diff_sq = safe_speeds[idx]**2 - self._v_ego**2
            decel_distance_needed = abs(speed_diff_sq) / max(2e-3, (2 * max_decel))

            # How far away is this point?
            point_distance = times_for_planning[idx] * self._v_ego

            # Do we need to start slowing NOW for this point?
            if point_distance <= decel_distance_needed * max(1.0, float(self._overshoot_safety_margin)):
              immediate_requirements.append((idx, safe_speeds[idx], point_distance))

          if immediate_requirements:
            # Among all points needing immediate action, target the MINIMUM safe speed
            # This ensures we plan for the tightest part of the curve
            min_required_speed = min([speed for _, speed, _ in immediate_requirements])
            # Find the index with that minimum speed
            for idx, speed, dist in immediate_requirements:
              if speed == min_required_speed:
                overshoot_idx = idx
                self._v_overshoot_distance = dist
                break
          else:
            # FIX: No immediate requirements, but still plan for the TIGHTEST point ahead
            # Don't just use the first overshoot - find the point with minimum safe speed
            tightest_idx = overshoot_indices[np.argmin(safe_speeds[overshoot_indices])]
            overshoot_idx = tightest_idx
            self._v_overshoot_distance = times_for_planning[overshoot_idx] * self._v_ego

          self._v_overshoot = min(safe_speeds[overshoot_idx], self._v_cruise_setpoint)
          # Distance already set above based on immediate requirements or tightest point
          # Ensure minimum distance for safety
          self._v_overshoot_distance = max(self._v_overshoot_distance, float(self._overshoot_min_distance))
          # Calculate anticipation time for early deceleration
          anticipation_time = calculate_anticipation_time(
              self._v_ego,
              self._v_overshoot,
              max_pred_curvature * self._v_ego**2,
              self._aggressiveness
          )

          # Optional override: use fixed lead time in seconds if configured (> 0)
          if getattr(self, '_fixed_lead_time_s', 0.0) > 0.0:
            anticipation_time = clip(self._fixed_lead_time_s, 0.1, 10.0)

          # Adjust the overshoot distance to start deceleration earlier
          # This makes us reach target speed BEFORE the apex
          anticipation_distance = anticipation_time * self._v_ego
          raw_trigger_distance = float(self._v_overshoot_distance - anticipation_distance)
          # Signed user timing offset for overshoot-based braking onset.
          # Lower values (negative) begin slowing earlier; higher values delay onset.
          overshoot_phase_offset_s = float(getattr(self, '_overshoot_phase_offset_s', 0.0))
          self._overshoot_trigger_in_s = raw_trigger_distance / max(self._v_ego, 0.1) + overshoot_phase_offset_s
          self._v_overshoot_distance = max(raw_trigger_distance, float(self._overshoot_min_distance))

          _debug(
            f"TVC: Advanced High LatAcc. Dist: {self._v_overshoot_distance:.2f}, "
            f"v: {self._v_overshoot * CV.MS_TO_KPH:.2f}, anticipation: {anticipation_time:.1f}s"
          )

        return  # Successfully processed vision data

    # Vision not good or model not available: use held curvature (adjusted_curvature)
    adjusted_curvature = self._update_vision_occlusion(model_data, current_time)
    current_curvature = max(0.0, float(adjusted_curvature))
    current_curvature_signed = 0.0
    max_pred_curvature = current_curvature

    # Update filtered curvature and dependent quantities
    self._filtered_curvature = ((1 - self._curvature_ema_ratio) * self._filtered_curvature +
                               self._curvature_ema_ratio * max_pred_curvature)
    self._current_lat_acc = current_curvature_signed * self._v_ego**2
    self._max_pred_lat_acc = self._v_ego**2 * max_pred_curvature
    self._update_low_speed_calibration(sm, reference_curvature=float(current_curvature))
    self._max_v_for_current_curvature = self._curve_speed(current_curvature) if current_curvature > 0 else V_CRUISE_MAX * CV.KPH_TO_MS
    self._lat_acc_overshoot_ahead = (self._max_v_for_current_curvature < self._v_ego)
    self._v_overshoot = min(self._max_v_for_current_curvature, self._v_cruise_setpoint)
    # Conservative default distance handling
    if self._lat_acc_overshoot_ahead:
      # Default conservative distance handling when vision not good: ensure a reasonable floor
      default_floor = max(20.0, 2.0 * float(self._overshoot_min_distance))
      self._v_overshoot_distance = max(getattr(self, '_v_overshoot_distance', default_floor), default_floor)
      self._overshoot_trigger_in_s = self._v_overshoot_distance / max(self._v_ego, 0.1)
    else:
      self._v_overshoot_distance = getattr(self, '_v_overshoot_distance', 200.0)
      self._overshoot_trigger_in_s = float('inf')
    
    # Track curvature change rate for anticipation moderation (20 Hz assumed)
    try:
      prev = self._prev_filtered_curvature
    except AttributeError:
      prev = self._filtered_curvature
    rate = (abs(self._filtered_curvature) - abs(prev)) / 0.05
    self._abs_curvature_rate = rate
    self._is_easing = (rate <= 0.0)
    self._prev_filtered_curvature = self._filtered_curvature
  def _state_transition(self):
    """Compatibility shell for legacy VTSC state telemetry.

    The state machine no longer drives VTSC behavior; it is retained only so traces and
    tooling that expect this method/field continue to function.
    """
    # System-level disable conditions still clear hold state.
    if not self._op_enabled or not self._is_enabled or self._gas_pressed:
      self._v_turn_hold_until = 0.0
      self._v_turn_hold_min = float(INF_SPEED)
      self.state = VisionTurnControllerState.disabled
      return
    self.state = VisionTurnControllerState.disabled

  def _update_solution(self, sm=None):
    """SIMPLIFIED: Always run physics calculations - let longitudinal planner decide usage."""
    dt = 0.05  # 20Hz

    # Apply temporary fast reacquisition filter alpha if armed
    now = time.time()
    # Time since occlusion started (for early-phase behaviors)
    try:
      _t0_occ = float(getattr(self._occlusion_state, 'occluded_since_time', 0.0) or 0.0)
    except Exception:
      _t0_occ = 0.0
    occ_age = max(0.0, now - _t0_occ)
    if now < getattr(self, '_fast_reacq_until', 0.0):
      self._filter_alpha = max(self._base_filter_alpha, float(getattr(self, '_fast_reacq_alpha', 0.85)))
    else:
      self._filter_alpha = self._base_filter_alpha
    occl_bypass = bool(getattr(self, '_occl_lead_bypass_active', False))

    # SIMPLIFIED: Always run advanced planning logic - no activation thresholds
    # On straight roads: will return cruise setpoint, longitudinal planner ignores (other sources lower)
    # On curves: will return physics speed, longitudinal planner uses it (lowest source)
    # Calculate target speed using advanced planning
    raw_target = self._plan_advanced_speed_trajectory()
    if raw_target is None:
      raw_target = self._prev_target_speed if hasattr(self, '_prev_target_speed') else self._v_ego
    # Debug: record raw target prior to map caps
    try:
      self._dbg_target_raw = float(raw_target)
    except Exception:
      self._dbg_target_raw = float(self._prev_target_speed if hasattr(self, '_prev_target_speed') else self._v_ego)

    # Keep jerk scaling constant to respect caps
    scale_jerk = 1.0

    # Optional: apply map-based lookahead cap to extend horizon.
    strategy_mode = normalize_map_strategy(getattr(self, '_map_strategy_mode', DEFAULT_MAP_STRATEGY))
    self._dbg_strategy_mode = strategy_mode
    self._dbg_strategy_state = "idle"
    self._dbg_map_advisory_cap = float(getattr(self, '_map_tail_advisory_cap', 0.0) or 0.0)
    self._dbg_map_strategic_cap = float(getattr(self, '_map_tail_strategic_cap', 0.0) or 0.0)
    self._dbg_vision_local_cap = float(raw_target)
    self._dbg_selected_cap = float(raw_target)
    self._dbg_map_floor_active = False
    self._dbg_map_floor_reason = ""
    self._dbg_vision_relax_allowed = False
    self._dbg_vision_relax_reason = ""
    self._dbg_map_anchor_dist_m = 0.0
    self._dbg_map_anchor_k = 0.0
    self._dbg_map_takeover_dwell_s = 0.0
    self._dbg_map_counterevidence_dwell_s = 0.0
    response_model = getattr(self, '_longitudinal_response_model', None)
    self._dbg_planner_min_accel = float(getattr(response_model, 'min_accel_mps2', 0.0) or 0.0)
    self._dbg_planner_response_decel = float(getattr(response_model, 'planning_decel_mps2', 0.0) or 0.0)
    self._dbg_planner_delay_s = float(getattr(response_model, 'actuation_delay_s', 0.0) or 0.0)
    self._clear_winding_behavior_profile()
    self._map_tail_reason = "toggle_off"
    try:
      map_lookahead_enabled = bool(self._get_bool_param('MTSCLookaheadEnabled', False))
      map_lookahead_falling_edge = bool(self._map_lookahead_enabled_prev and not map_lookahead_enabled)
      self._map_lookahead_enabled_prev = map_lookahead_enabled
      if map_lookahead_enabled:
        self._map_tail_reason = "enabled_no_cap"
        map_tail_span = start_span(SPAN_MAP_TAIL_CAP)
        try:
          v_cap, s_start, coverage = self._map_tail_cap(sm)
        finally:
          end_span(map_tail_span)
        candidate = getattr(self, '_map_tail_candidate', None)
        self._dbg_map_advisory_cap = float(getattr(self, '_map_tail_advisory_cap', 0.0) or 0.0)
        self._dbg_map_strategic_cap = float(getattr(self, '_map_tail_strategic_cap', 0.0) or 0.0)
        if candidate is not None:
          self._map_tail_last_cap = float(getattr(candidate, 'cap_mps', 0.0) or 0.0)
          self._map_tail_last_start = float(getattr(candidate, 'start_m', s_start) or 0.0)
          self._map_tail_last_coverage = float(getattr(candidate, 'coverage', coverage) or 0.0)
          self._dbg_map_anchor_dist_m = float(getattr(candidate, 'anchor_dist_m', 0.0) or 0.0)
          self._dbg_map_anchor_k = float(getattr(candidate, 'anchor_curvature', 0.0) or 0.0)
        if v_cap is not None and candidate is not None:
          v_cap_f = float(v_cap)
          # Save holdover for bridging transient map dropouts
          self._map_holdover_cap = v_cap_f
          self._map_holdover_candidate = candidate
          self._map_holdover_ts = now
          raw_target_pre_map = float(raw_target)
          try:
            v_ego_local = float(max(0.0, self._v_ego))
            s_visible = float(max(0.0, getattr(self, '_vis_horizon_s', 1.4)) * v_ego_local)
            vis_margin = float(max(0.0, getattr(self, '_vis_margin_m', 10.0)))
            k_turn_min = float(max(1e-6, getattr(self, '_fov_k_min', 2e-4)))
            k_now = float(abs(getattr(self, '_filtered_curvature', 0.0)))
            # Hysteresis: onset at k_turn_min, clear at 0.6 * k_turn_min
            if self._turn_visible_sticky:
              turn_visible_now = bool(k_now >= k_turn_min * 0.6)
            else:
              turn_visible_now = bool(k_now >= k_turn_min)
            self._turn_visible_sticky = turn_visible_now
            turn_visible_ahead = bool(
              bool(getattr(self, '_lat_acc_overshoot_ahead', False)) and
              (float(getattr(self, '_v_overshoot_distance', 1e9)) <= (s_visible + vis_margin))
            )
            vision_good = bool(getattr(self._occlusion_state, 'vision_good', True))
            vision_status = getattr(self._occlusion_state, 'vision_status', VisionStatus.FULL_VISIBILITY)
            full_visibility = bool(int(vision_status) == int(VisionStatus.FULL_VISIBILITY))
            decision = evaluate_map_strategy(
              mode=strategy_mode,
              state=self._map_strategy_state,
              candidate=candidate,
              raw_target_pre_map=raw_target_pre_map,
              full_visibility=full_visibility,
              vision_good=vision_good,
              turn_visible=bool(turn_visible_now or turn_visible_ahead),
              s_visible_m=s_visible,
              vis_margin_m=vis_margin,
              now_s=now,
              apex_exit_ready=bool(getattr(self, '_apex_exit_ready', False)),
              winding_profile=getattr(self, '_winding_behavior_profile', DEFAULT_WINDING_BEHAVIOR_PROFILE),
            )
          except Exception:
            decision = None

          if decision is not None:
            self._map_tail_active = bool(decision.map_floor_active)
            self._map_tail_reason = str(decision.map_reason or "applied")
            self._dbg_strategy_state = str(decision.strategy_state or "idle")
            self._dbg_map_floor_active = bool(decision.map_floor_active)
            self._dbg_map_floor_reason = str(decision.map_reason or "")
            self._dbg_vision_relax_allowed = bool(decision.vision_relax_allowed)
            self._dbg_vision_relax_reason = str(decision.vision_relax_reason or "")
            self._dbg_map_takeover_dwell_s = float(decision.takeover_dwell_s)
            self._dbg_map_counterevidence_dwell_s = float(decision.counterevidence_dwell_s)
            if bool(decision.apply_map_cap):
              raw_target = min(raw_target, v_cap_f)
          else:
            self._map_tail_active = True
            self._map_tail_reason = "applied"
            raw_target = min(raw_target, v_cap_f)
        else:
          # Holdover: bridge transient map data dropouts (1-2 frames of GPS
          # flicker, road geometry invalidation, or empty MapCurvatures) by
          # keeping the last valid cap active for up to 1 second.
          # Do NOT holdover intentional suppressions (counterevidence, lane
          # change ambiguity) — those should release immediately.
          _MAP_HOLDOVER_TIMEOUT_S = 1.0
          _MAP_HOLDOVER_DATA_DROPOUT_REASONS = frozenset({
            'no_gps', 'road_geometry_invalid', 'no_map_curvatures',
            'insufficient_map_points', 'unknown', 'no_cap',
            'enabled_no_cap', '',
          })
          _compute_reason = str(getattr(self, '_map_tail_compute_reason', '') or '')
          _is_data_dropout = _compute_reason in _MAP_HOLDOVER_DATA_DROPOUT_REASONS
          if (_is_data_dropout and
              self._map_holdover_cap is not None and
              self._map_holdover_candidate is not None and
              (now - self._map_holdover_ts) < _MAP_HOLDOVER_TIMEOUT_S):
            raw_target = min(raw_target, self._map_holdover_cap)
            self._map_tail_active = True
            self._map_tail_last_cap = self._map_holdover_cap
            self._map_tail_reason = "holdover"
            self._dbg_map_floor_active = True
            self._dbg_map_floor_reason = "holdover"
            self._dbg_map_strategic_cap = self._map_holdover_cap
          else:
            self._map_strategy_state.reset()
            self._turn_visible_sticky = False
            self._map_tail_active = False
            self._map_tail_reason = str(_compute_reason or 'no_cap')
            self._map_holdover_cap = None
            self._map_holdover_candidate = None
      else:
        self._clear_map_lookahead_state(clear_latched_output=map_lookahead_falling_edge)
        self._map_tail_reason = "toggle_off"
    except Exception:
      self._map_strategy_state.reset()
      self._turn_visible_sticky = False
      self._map_tail_active = False
      self._map_tail_reason = "exception"
      self._clear_winding_road_context()
    self._dbg_selected_cap = float(raw_target)
    self._dbg_map_floor_reason = str(self._map_tail_reason or "")
    # Debug: record final target after map caps
    try:
      self._dbg_target_final = float(raw_target)
    except Exception:
      self._dbg_target_final = float(self._prev_target_speed if hasattr(self, '_prev_target_speed') else self._v_ego)

    # Track the speed cap VTSC intends to publish to the planner. This starts from the
    # base/vision target and is tightened by occlusion/map/other constraints.
    try:
      v_target_cap = float(raw_target)
    except Exception:
      v_target_cap = float(self._v_cruise_setpoint)
    # Overshoot cap timing gate:
    # - Engage once computed "time-to-start-braking" is reached.
    # - Release this extra cap once apex-exit logic says we're past apex in an easing phase, so the
    #   planner can start accelerating out while still obeying the visible-curve cap.
    prev_overshoot_cap_active = bool(getattr(self, '_overshoot_cap_active', False))
    self._overshoot_cap_active = False
    self._dbg_near_apex_release_ready = False
    try:
      profile = getattr(self, '_winding_behavior_profile', DEFAULT_WINDING_BEHAVIOR_PROFILE)
      self._dbg_apex_release_lat_acc_ratio = float(getattr(profile, 'apex_release_lat_acc_ratio', DEFAULT_WINDING_BEHAVIOR_PROFILE.apex_release_lat_acc_ratio))
    except Exception:
      self._dbg_apex_release_lat_acc_ratio = float(DEFAULT_WINDING_BEHAVIOR_PROFILE.apex_release_lat_acc_ratio)
    try:
      if bool(getattr(self, '_lat_acc_overshoot_ahead', False)):
        now_release_s = float(time.time())
        prev_published_cap = max(0.0, float(getattr(self, '_v_turn_output', 0.0) or 0.0))
        trigger_in_s = float(getattr(self, '_overshoot_trigger_in_s', float('inf')))
        should_start = bool(trigger_in_s <= 0.0)
        try:
          profile = getattr(self, '_winding_behavior_profile', DEFAULT_WINDING_BEHAVIOR_PROFILE)
          near_apex_hold_s = max(0.0, float(getattr(profile, 'apex_release_hold_s', 0.0) or 0.0))
        except Exception:
          near_apex_hold_s = 0.0
        near_apex_release = bool(prev_overshoot_cap_active and self._is_near_apex_release_ready())
        if near_apex_release and near_apex_hold_s > 0.0:
          self._near_apex_release_until = max(
            float(getattr(self, '_near_apex_release_until', 0.0) or 0.0),
            float(now_release_s) + float(near_apex_hold_s),
          )
        latched_near_apex_release = bool(float(getattr(self, '_near_apex_release_until', 0.0) or 0.0) > float(now_release_s))
        apex_release = bool(
          (getattr(self, '_apex_exit_ready', False) and getattr(self, '_is_easing', False)) or
          latched_near_apex_release
        )
        soft_handoff_block = bool(
          bool(getattr(self, '_apex_exit_ready', False)) and
          (not prev_overshoot_cap_active) and
          prev_published_cap > 1.0 and
          prev_published_cap <= (float(getattr(self, '_v_ego', 0.0) or 0.0) + 2.0)
        )
        if should_start and not apex_release and not soft_handoff_block:
          v_target_cap = min(v_target_cap, float(getattr(self, '_v_overshoot', v_target_cap)))
          self._overshoot_cap_active = True
      else:
        self._near_apex_release_until = 0.0
    except Exception:
      self._overshoot_cap_active = False

    # ===== Freeway sanity guard (fail-open) =====
    try:
      kappa_vis = float(abs(getattr(self._occlusion_state, 'last_valid_curvature', 0.0)))
      s_visible_m = float(max(0.0, getattr(self, '_vis_horizon_s', 1.4) * max(0.0, self._v_ego)))
      path_conf = float(getattr(self._occlusion_state, 'smoothed_confidence', 0.0))
    except Exception:
      kappa_vis, s_visible_m, path_conf = 0.0, 0.0, 0.0
    # Param override to force fail-open during on-road triage
    failopen_param = bool(self._get_bool_param('VTSCFailOpen', False))
    fail_open = bool((kappa_vis <= FREEWAY_CURV_EPS) and (s_visible_m >= FREEWAY_MIN_VISIBLE_M) and (path_conf >= FREEWAY_MIN_CONF))
    self._freeway_failopen_active = bool(fail_open or failopen_param)
    # Snapshot fields for triage
    self._dbg_kappa_vis = kappa_vis
    self._dbg_s_visible_m = s_visible_m
    self._dbg_path_conf = path_conf
    self._dbg_fail_open = bool(self._freeway_failopen_active)

    # FOV-based occlusion gate computation with pre-trigger and stickiness
    try:
      psi_fov = float(getattr(self, '_psi_fov_rad', 0.49))
      psi_margin = float(getattr(self, '_psi_margin_rad', 0.087))
    except Exception:
      psi_fov, psi_margin = 0.49, 0.087
    # instantaneous psi (use filtered curvature to anticipate FOV exit)
    try:
      kappa_gate = float(getattr(self, '_filtered_curvature', 0.0))
    except Exception:
      kappa_gate = kappa_vis
    self._dbg_psi_vis = float(abs(kappa_gate) * max(0.0, s_visible_m))
    self._dbg_psi_thresh = float(max(0.0, psi_fov - psi_margin))
    # Pre-trigger by time-to-FOV-exit (TTFOV)
    k_min = float(getattr(self, '_fov_k_min', 2e-4))
    k_free = float(getattr(self, '_fov_k_freeway', FREEWAY_CURV_EPS))
    try:
      delta_psi = max(0.0, self._dbg_psi_thresh - self._dbg_psi_vis)
      delta_s_to_exit = delta_psi / max(abs(kappa_gate), 1e-9)
      ttfov_s = delta_s_to_exit / max(self._v_ego, 0.1)
    except Exception:
      ttfov_s = 999.0
    self._dbg_ttfov_s = float(ttfov_s)
    pretrigger_time = float(getattr(self, '_fov_pretrigger_time_s', 1.2))
    # Only allow pretrigger when confidence is actually degraded; otherwise, favor fresh vision
    try:
      _conf_s = float(getattr(self._occlusion_state, 'smoothed_confidence', 1.0))
      _conf_bad = float(getattr(self._occlusion_state, 'bad_threshold', 0.65))
      _conf_good = float(getattr(self._occlusion_state, 'good_threshold', 0.70))
      _vis_good = bool(getattr(self._occlusion_state, 'vision_good', True)) and (_conf_s >= _conf_good)
      # Track slope to suppress pretrigger while confidence is rising from borderline
      try:
        _conf_prev = float(getattr(self._occlusion_state, 'prev_smoothed_conf', _conf_s))
      except Exception:
        _conf_prev = _conf_s
      setattr(self._occlusion_state, 'prev_smoothed_conf', _conf_s)
      conf_rising = (_conf_s >= _conf_prev + 0.005)  # ~0.5% absolute rise per update
    except Exception:
      _conf_s, _conf_bad, _conf_good, _vis_good, conf_rising = 1.0, 0.65, 0.70, True, False
    pretrigger = (ttfov_s <= pretrigger_time) and (abs(kappa_gate) >= k_min) and (_conf_s < _conf_bad) and (not conf_rising)
    # Suppress pretrigger during short model dropouts to avoid overreacting
    try:
      if bool(getattr(self._occlusion_state, 'dropout_active', False)):
        pretrigger = False
    except Exception:
      pass
    # Hysteretic onset/clear
    onset_geom = (abs(kappa_gate) >= k_min) and (self._dbg_psi_vis >= self._dbg_psi_thresh)
    onset = (onset_geom or pretrigger) and (not _vis_good)
    # Clear when geometry says we're no longer near a field-of-view exit.
    #
    # Rationale:
    # - Real roads often have mediocre lane-line confidence for benign reasons (wear, glare, lead vehicle).
    # - VTSC is an assistant; once the path is safely within FoV again, we should recover like a
    #   reasonable human would, instead of "sticking" in occlusion due to confidence alone.
    clear = (abs(kappa_vis) < k_free) or (self._dbg_psi_vis < self._dbg_psi_thresh) or _vis_good
    # If vision is good, forcibly clear occlusion state and reset counters immediately
    if _vis_good and self._fov_occluded:
      self._fov_occluded = False
      self._fov_on_cnt = 0
      self._fov_off_cnt = 0
      self._fov_boost_left = 0
      self._fov_overshoot_left = 0
      self._onset_no_raise_active = False
    if onset and not self._fov_occluded:
      self._fov_on_cnt = int(self._fov_on_cnt) + 1
      self._fov_off_cnt = 0
      if self._fov_on_cnt >= int(getattr(self, '_fov_N_on', 5)):
        self._fov_occluded = True
        self._fov_reason = 'pretrigger' if pretrigger else 'fov_exit'
        self._fov_boost_left = int(getattr(self, '_fov_onset_boost_frames', 10))
        self._fov_overshoot_left = int(getattr(self, '_fov_overshoot_frames', 10))
        # initialize EWMA curvature at current filtered
        try:
          self._fov_kappa_ewma = float(getattr(self, '_filtered_curvature', 0.0))
        except Exception:
          self._fov_kappa_ewma = 0.0
    elif clear and self._fov_occluded:
      # During onset stickiness window, do not clear on small margin
      if int(getattr(self, '_fov_boost_left', 0)) <= 0:
        self._fov_off_cnt = int(self._fov_off_cnt) + 1
        self._fov_on_cnt = 0
        if self._fov_off_cnt >= int(getattr(self, '_fov_N_off', 10)):
          self._fov_occluded = False
          self._fov_reason = 'freeway' if abs(kappa_vis) < k_free else 'short_vis'
    else:
      self._fov_on_cnt = max(0, int(self._fov_on_cnt) - 1)
      self._fov_off_cnt = max(0, int(self._fov_off_cnt) - 1)
    # decay windows
    if int(getattr(self, '_fov_boost_left', 0)) > 0:
      self._fov_boost_left -= 1
    if int(getattr(self, '_fov_overshoot_left', 0)) > 0:
      self._fov_overshoot_left -= 1

    # Compute acceleration command to drive current speed toward target
    accel_cmd = (raw_target - self._v_ego) / dt
    # ===== Severe confidence occlusion guard (non-FOV) =====
    # If vision confidence is extremely low (SEVERE/LOST), do not allow increasing speed.
    #
    # This is intentionally *not* tied to FOV geometry; it protects against cases where the
    # model has effectively no reliable lane-line confidence (e.g., glare/washed-out markings),
    # but the FOV gate might not trigger (gentle curvature / low psi).
    #
    # Lead bypass MUST override this, since lead presence can reduce lane-line confidence
    # without representing a true visibility occlusion.
    try:
      vs = getattr(self._occlusion_state, 'vision_status', None)
      severe_conf = (vs == VisionStatus.SEVERE_OCCLUSION) or (vs == VisionStatus.VISION_LOST)
    except Exception:
      severe_conf = False
    # Only apply this guard when there is meaningful curvature (i.e., when VTSC is relevant).
    # On straight roads, low lane-line confidence can happen for reasons unrelated to "can't see around a bend"
    # (e.g., worn paint, glare, lead vehicle covering lane lines), and we don't want VTSC to interfere.
    try:
      k_gate_abs = abs(float(kappa_gate))
    except Exception:
      k_gate_abs = 0.0
    try:
      min_operating_v = float(_MIN_V)
    except Exception:
      min_operating_v = 0.0
    low_speed_guard = bool(self._v_ego <= max(0.0, min_operating_v))
    severe_conf_no_raise = bool(
      severe_conf and (not occl_bypass) and (not self._freeway_failopen_active)
      and (k_gate_abs >= float(k_min)) and (not low_speed_guard)
    )
    if severe_conf_no_raise:
      accel_cmd = min(accel_cmd, 0.0)
      try:
        v_target_cap = min(float(v_target_cap), float(self._v_ego))
      except Exception:
        pass
    # Fast reacquisition nudge: only if physics base supports acceleration above current speed
    try:
      now_t = time.time()
      if bool(getattr(self._occlusion_state, 'vision_good', True)) and (now_t <= float(getattr(self, '_fast_reacq_until', 0.0))):
        try:
          base_cap = float(min(self._v_cruise_setpoint, self._curve_speed(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))))
        except Exception:
          base_cap = float(self._v_cruise_setpoint)
        # Require a small margin to ensure this is a raise scenario
        if (base_cap > (self._v_ego + 0.05)) and (float(getattr(self, '_prev_target_speed', self._v_ego)) < 0.98 * base_cap):
          accel_cmd = max(accel_cmd, 0.18)
    except Exception:
      pass
    # If FOV-gated occlusion is active, fold in a conservative occlusion cap immediately
    self._dbg_low_speed_margin = False
    if self._fov_occluded and (not occl_bypass) and (not self._freeway_failopen_active):
      try:
        k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
      except Exception:
        k_est = 0.0
      k_filt = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
      # During early overshoot window, use EWMA to be conservative
      if int(getattr(self, '_fov_overshoot_left', 0)) > 0:
        try:
          tau = float(getattr(self, '_fov_ewma_tau_s', 0.5))
          alpha = 1.0 - math.exp(-dt / max(1e-3, tau))
        except Exception:
          alpha = 0.5
        self._fov_kappa_ewma = (1.0 - alpha) * float(getattr(self, '_fov_kappa_ewma', k_filt)) + alpha * k_filt
        k_cons = max(1e-8, min(self._fov_kappa_ewma, k_est))
      else:
        k_cons = max(1e-8, min(k_filt, k_est))
      # === κ-bias at occluded onset: briefly bias occlusion cap to ensure decel starts ===
      # Use existing onset stickiness window (_fov_boost_left) as a safe timing window.
      try:
        psi_margin_deg = float(getattr(self, '_psi_margin_rad', 0.087)) * 57.2957795
      except Exception:
        psi_margin_deg = 0.0
      try:
        ttfov_s = float(getattr(self, '_dbg_ttfov_s', 999.0))
      except Exception:
        ttfov_s = 999.0
      pretrigger_time = float(getattr(self, '_fov_pretrigger_time_s', 1.5))
      slack_s = 0.10
      v_max_mps = 36.0
      try:
        boost_left = int(getattr(self, '_fov_boost_left', 0))
      except Exception:
        boost_left = 0
      onset_gate = (boost_left > 0) and (psi_margin_deg >= 5.0) and (self._v_ego <= v_max_mps) and (ttfov_s <= (pretrigger_time + slack_s))
      try:
        if onset_gate:
          # Disable curvature inflation at onset; do not alter k_cons
          self._dbg_onset_bias_active = False
          self._dbg_onset_gate_reason = {
            'psi_ok': True,
            'speed_ok': True,
            'ttfov_ok': True,
            'boost_left': int(boost_left),
          }
        else:
          self._dbg_onset_bias_active = False
          self._dbg_onset_gate_reason = {
            'psi_ok': bool(psi_margin_deg >= 5.0),
            'speed_ok': bool(self._v_ego <= v_max_mps),
            'ttfov_ok': bool(ttfov_s <= (pretrigger_time + slack_s)),
            'boost_left': int(boost_left),
          }
      except Exception:
        self._dbg_onset_bias_active = False
        self._dbg_onset_gate_reason = {'error': True}
      v_occ_cap = float(self._curve_speed(k_cons))
      accel_cmd = min(accel_cmd, (v_occ_cap - self._v_ego) / dt)
      # Also constrain the published cap to reflect this occlusion cap.
      try:
        v_target_cap = min(float(v_target_cap), float(v_occ_cap))
      except Exception:
        pass

      # === Consolidated curvature cap and early no-raise during onset window ===
      # Maintain onset timers on rising edge of occlusion
      if self._fov_occluded and not getattr(self, '_occlusion_prev', False):
        self._occlusion_onset_timer_s = 0.0
        # Estimate visible cap at onset for reference (min of cruise and filtered-curvature speed)
        try:
          cap_vis_onset = float(min(self._v_cruise_setpoint, self._curve_speed(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))))
        except Exception:
          cap_vis_onset = float(self._v_cruise_setpoint)
        # Active cap approx at onset
        self._v_cap_active_at_onset_mps = float(min(cap_vis_onset, v_occ_cap))
        # Arm a short vision-floor TTL to avoid immediate depression below LKG
        try:
          self._vision_floor_until = float(time.time()) + float(getattr(self, '_vision_floor_ttl_s', 2.5))
        except Exception:
          self._vision_floor_until = 0.0
      if self._fov_occluded:
        self._occlusion_onset_timer_s += dt
      self._occlusion_prev = bool(self._fov_occluded)

      # Onset window remains visible in debug breadcrumbs; no-raise behavior is disabled.
      try:
        psi_margin_deg = float(getattr(self, '_psi_margin_rad', 0.087)) * 57.2957795
      except Exception:
        psi_margin_deg = 0.0
      v_max_mps = 36.0
      window_s = 0.8
      # Relax TTFOV gating inside onset window to ensure assist engages
      onset_gate = (self._fov_occluded and (psi_margin_deg >= 5.0) and (self._v_ego <= v_max_mps) and (self._occlusion_onset_timer_s <= window_s))
      if onset_gate:
        # Disable onset curvature inflation and decel floors; rely on v_occ_cap from k_cons only
        self._dbg_cap_source = 'occluded_onset_disabled'
      self._onset_no_raise_active = False

    # Occlusion-time accel gating: allow positive accel only with positive margin
    occl_positive_margin = False
    early_no_raise = False  # suppress positive accel in early hidden-turn phase
    occl_effects_active = bool(self._fov_occluded and (not occl_bypass) and (not self._freeway_failopen_active))
    if occl_effects_active:
      v_gate_hi = HIGHWAY_MIN_MPS  # ~55 mph
      if self._v_ego < v_gate_hi:
        # Gradual bias toward pure physics mode between ~50 and 65 mph (no hard bypass)
        # Compute barrier context to determine margin (near vs. far)
        try:
          v_vis = self._curve_speed(max(1e-8, float(self._occlusion_state.last_valid_curvature)))
          # When occluded, do NOT let the filtered/model curvature drive the near cap;
          # rely on last-visible curvature (v_vis) for the near bound.
          v_near = min(v_vis, self._v_cruise_setpoint)
          # Use the more conservative (lower speed) of occlusion est curvature and filtered curvature
          try:
            k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
          except Exception:
            k_est = 0.0
          k_filt = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
          v_occ_raw = min(self._curve_speed(max(1e-8, k_est)), self._curve_speed(k_filt))
          s_vis = max(0.0, float(getattr(self, '_vis_horizon_s', 1.4)) * max(0.0, self._v_ego))
          a_cap = abs(float(self._comfort_decel_limit))
          v_now = max(self._prev_target_speed, self._v_ego)
          # Tail-aware far bound for gating
          try:
            dist_since = float(getattr(self._occlusion_state, 'distance_since_m', 0.0))
          except Exception:
            dist_since = 0.0
          s_tail = max(0.0, dist_since - s_vis)
          try:
            k_now = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
          except Exception:
            k_now = 0.0
          if k_now <= 0.004:
            tail_frac = 0.10
          elif k_now >= 0.008:
            tail_frac = 0.90
          else:
            # Interpolate from 0.10 at 0.004 to 0.90 at 0.008 (match harness diagnostics)
            tail_frac = 0.10 + 0.80 * ((k_now - 0.004) / 0.004)
          # Speed-based gating: ignore tail floor above ~55 mph (pure physics) and taper between 50–55 mph
          v_gate_lo = 22.35  # m/s ~50 mph
          v_gate_hi = 24.5872  # m/s ~55 mph
          v_now_for_gate = max(0.0, self._v_ego)
          blend = (
            0.0 if v_now_for_gate <= v_gate_lo else
            (1.0 if v_now_for_gate >= v_gate_hi else (v_now_for_gate - v_gate_lo) / max(1e-6, (v_gate_hi - v_gate_lo)))
          )
          tail_frac_eff = tail_frac * (1.0 - blend)
          try:
            v_cap_tail_eff = math.sqrt(max(0.0, v_now * v_now - 2.0 * a_cap * (tail_frac_eff * s_tail)))
          except Exception:
            v_cap_tail_eff = v_now
          # Compute two far bounds: one with speed-blended tail (for internal use) and one matching harness (for gating)
          v_far_gate = min(v_occ_raw, v_cap_tail_eff, self._v_cruise_setpoint)
          # Harness-equivalent far bound (no speed-based blend on tail fraction)
          try:
            v_cap_tail_h = math.sqrt(max(0.0, v_now * v_now - 2.0 * a_cap * (tail_frac * s_tail)))
          except Exception:
            v_cap_tail_h = v_now
          v_far_h = min(v_occ_raw, v_cap_tail_h, self._v_cruise_setpoint)
          d_req_h = max(0.0, (v_now * v_now - v_far_h * v_far_h) / max(2e-3, 2.0 * a_cap))
          margin_dist = float(getattr(self, '_vis_margin_m', 10.0))
          # Use harness-equivalent margin for gating decisions with small buffer (≈2 m)
          positive_margin = (d_req_h <= (s_vis - (margin_dist + 2.0)))
          early_phase = int(getattr(self, '_fov_on_cnt', 0)) <= 10
          occl_positive_margin = bool(positive_margin and not early_phase)
          # Snapshot for telemetry
          try:
            self._dbg_s_tail = float(s_tail)
            self._dbg_tail_frac = float(tail_frac)
          except Exception:
            self._dbg_s_tail = 0.0
            self._dbg_tail_frac = 0.0
          # ===== Hidden-turn early deceleration trigger (short-horizon critical deficit) =====
          if HIDDEN_TURN_ENABLED and (self._v_ego <= HIDDEN_TURN_V_MAX_MPS):
            try:
              now = time.time()
            except Exception:
              now = 0.0
            t0 = float(getattr(self._occlusion_state, 'occluded_since_time', 0.0) or 0.0)
            occ_elapsed = max(0.0, now - t0)
            if occ_elapsed >= HIDDEN_TURN_MIN_OCC_S and occ_elapsed <= HIDDEN_TURN_PHASE_S:
              v_req_hidden = float(v_cap_tail_eff)
              deficit = max(0.0, v_now - v_req_hidden)
              try:
                vs = getattr(self._occlusion_state, 'vision_status', None)
              except Exception:
                vs = None
              use_shed = (vs == VisionStatus.SEVERE_OCCLUSION or vs == VisionStatus.VISION_LOST)
              if deficit >= HIDDEN_TURN_DELTA_V_MPS or use_shed:
                d_avail_time = max(0.0, self._v_ego * HIDDEN_TURN_T_H_S)
                d_avail_vis = max(0.0, s_vis - margin_dist)
                d_avail = min(d_avail_time, d_avail_vis)
                if use_shed:
                  v_req_min = max(0.0, v_now - HIDDEN_TURN_DELTA_V_MPS)
                  d_req_hidden = max(0.0, (v_now * v_now - v_req_min * v_req_min) / max(2e-3, 2.0 * a_cap))
                else:
                  d_req_hidden = max(0.0, (v_now * v_now - v_req_hidden * v_req_hidden) / max(2e-3, 2.0 * a_cap))
                # Soft early-phase visible-heading tightening for hidden-turns
                try:
                  entry_kappa = abs(float(getattr(self._occlusion_state, 'entry_curvature', 0.0) or 0.0))
                except Exception:
                  entry_kappa = 0.0
                s_head = max(6.0, self._v_ego * HIDDEN_TURN_HEADING_WIN_S)
                vis_heading_rad = entry_kappa * s_head
                straightness_gain = max(0.0, min(1.0,
                  (HIDDEN_TURN_VIS_HEADING_MAX_RAD - vis_heading_rad) / max(1e-6, HIDDEN_TURN_VIS_HEADING_MAX_RAD)))
                phase_progress = max(0.0, min(1.0, occ_elapsed / max(1e-6, HIDDEN_TURN_PHASE_S)))
                early_tighten = 0.55 * straightness_gain * (1.0 - phase_progress)
                short_h_avail = (HIDDEN_TURN_AVAIL_SCALE * d_avail) * (1.0 - early_tighten)
                if d_req_hidden > short_h_avail:
                  v_occ_raw = min(v_occ_raw, v_req_hidden)
                  positive_margin = False
                  # Suppress positive accel during early hidden-turn phase (~1.5s)
                  early_no_raise = True
          reachable_cap = v_near
        except Exception:
          positive_margin = False
          reachable_cap = self._v_cruise_setpoint
        # Under positive margin allow non-negative acceleration; otherwise decelerate toward barrier target
        if positive_margin:
          # Drive a gentle raise toward reachable_cap using existing jerk limits
          pos_limit = max(0.0, float(getattr(self, '_max_accel', 1.0)))
          desired_target = max(self._v_ego, min(reachable_cap, self._v_ego + pos_limit * 0.05))
          if not early_no_raise:
            accel_cmd = max(accel_cmd, (desired_target - self._v_ego) / 0.05)
          # Never decelerate while margin is positive
          accel_cmd = max(accel_cmd, 0.0)
        else:
          # Produce barrier target using near/far policy and fold into accel (favor decel)
          # Include far-field bound for:
          # - Highway (>36 m/s)
          # - Moderate/mountain speeds when curvature is meaningful (k_now ≥ 0.004)
          # - All sub-30 m/s regimes to ensure timely slowing for hidden/abrupt turns outside FoV
          use_far = (self._v_ego > 36.0) or (self._v_ego <= 36.0 and k_now >= 0.004) or (self._v_ego <= 30.0)
          if use_far:
            barrier_target_speed = min(min(v_near, v_far_gate), v_now)
          else:
            # Very low curvature at low speeds: stick to near bound to avoid crawl
            barrier_target_speed = min(v_near, v_now)
          # Fold barrier target into commanded deceleration (respect jerk limits downstream):
          # Pull toward barrier target; prefer more conservative (more negative) acceleration
          accel_cmd = min(accel_cmd, (barrier_target_speed - self._v_ego) / 0.05)
          # Constrain published cap for planner ingestion.
          try:
            v_target_cap = min(float(v_target_cap), float(barrier_target_speed))
          except Exception:
            pass

        if (not occl_positive_margin) and (self._v_ego <= LOW_SPEED_MARGIN_MAX_V_MPS) and (k_now <= LOW_SPEED_MARGIN_CURV_THRESH):
          occl_positive_margin = True
          self._dbg_low_speed_margin = True
      else:
        # Highway-speed occlusion path: compute near/far context locally.
        # NOTE: Tail-floor terms are intentionally omitted here; above ~55 mph we operate in
        # "pure physics" mode (v_occ_raw + last-visible curvature), and rely on downstream
        # occlusion "no-raise" gating to prevent inappropriate acceleration.
        try:
          v_vis = self._curve_speed(max(1e-8, float(self._occlusion_state.last_valid_curvature)))
        except Exception:
          v_vis = float(self._v_cruise_setpoint)
        v_near = min(float(v_vis), float(self._v_cruise_setpoint))
        try:
          k_now = abs(float(getattr(self._occlusion_state, 'est_curvature', 0.0) or 0.0))
        except Exception:
          k_now = 0.0
        try:
          k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
        except Exception:
          k_est = 0.0
        k_filt = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
        v_occ_raw = min(self._curve_speed(max(1e-8, k_est)), self._curve_speed(k_filt))
        v_now = max(float(getattr(self, '_prev_target_speed', self._v_ego)), float(self._v_ego))
        v_far_gate = min(float(v_occ_raw), float(self._v_cruise_setpoint))

        # Produce barrier target using near/far policy and fold into accel (favor decel)
        # Include far-field bound for:
        # - Highway (>36 m/s)
        # - Moderate/mountain speeds when curvature is meaningful (k_now ≥ 0.004)
        # - All sub-30 m/s regimes to ensure timely slowing for hidden/abrupt turns outside FoV
        use_far = (self._v_ego > 36.0) or (self._v_ego <= 36.0 and k_now >= 0.004) or (self._v_ego <= 30.0)
        if use_far:
          barrier_target_speed = min(min(v_near, v_far_gate), v_now)
        else:
          barrier_target_speed = min(v_near, v_now)
        # Fold barrier target into commanded deceleration (respect jerk limits downstream):
        # Pull toward barrier target; prefer more conservative (more negative) acceleration
        accel_cmd = min(accel_cmd, (barrier_target_speed - self._v_ego) / 0.05)
        # Constrain published cap for planner ingestion.
        try:
          v_target_cap = min(float(v_target_cap), float(barrier_target_speed))
        except Exception:
          pass
    # ===== APPLY ADAPTIVE DECELERATION SYSTEM =====
    # Enforce no positive acceleration while occluded unless positive margin exists.
    # Additionally, suppress positive accel in early hidden-turn phase.
    if occl_effects_active:
      # Suppress raising during recent speed-limit step down while occluded
      try:
        if time.time() < getattr(self, '_limit_step_until', 0.0):
          early_no_raise = True
      except Exception:
        pass
      # If a speed-limit down-step occurred, suppress raising entirely until vision is good again
      if getattr(self, '_suppress_raise_due_to_limit', False):
        early_no_raise = True
      # Straight-road exemption: when curvature is near-zero, there is no hidden turn
      # to protect against.  Holding the cap at v_ego on a straight road after a lead
      # car turns off traps the system at low speed with no hazard justification.
      try:
        _k_for_noraise = float(abs(getattr(self, '_filtered_curvature', 0.0)))
      except Exception:
        _k_for_noraise = 1.0  # fail conservative
      straight_road = (_k_for_noraise < 5e-4)  # ~2000m radius — effectively straight
      if (not occl_positive_margin) or early_no_raise:
        if not straight_road:
          accel_cmd = min(accel_cmd, 0.0)
          # Mirror the "no-raise" behavior in the published speed cap:
          # if we are occluded and disallowing positive accel, we must not publish a cap above v_ego
          # (otherwise the planner/MPC will accelerate).
          try:
            v_target_cap = min(float(v_target_cap), float(self._v_ego))
          except Exception:
            pass
    # record for telemetry
    self._dbg_occl_positive_margin = bool(occl_positive_margin)
    self._dbg_early_no_raise = bool(early_no_raise)
    # Check if deceleration is required
    if accel_cmd < 0:
        # For curve scenarios, use physics-based calculation if needed
        if self._lat_acc_overshoot_ahead:
            remaining_distance = self._v_overshoot_distance
            physics_required_decel = self._calculate_required_deceleration(
                self._v_ego, self._v_overshoot, remaining_distance)
            # Use the more conservative (more negative) of commanded or physics-required decel
            accel_cmd = min(accel_cmd, physics_required_decel)

        # While occluded, avoid over-braking: cap to comfort decel limit
        if occl_effects_active:
            accel_cmd = max(accel_cmd, self._comfort_decel_limit)

        # Apply adaptive deceleration system with noise filtering
        accel_cmd = self._get_optimal_deceleration(accel_cmd, dt)
        # While occluded, ensure decel command does not exceed comfort cap after filtering
        if occl_effects_active:
          try:
            self._current_decel = max(self._current_decel, self._comfort_decel_limit)
          except Exception:
            pass

        # Monitor adaptive deceleration performance
        remaining_distance = self._v_overshoot_distance if self._lat_acc_overshoot_ahead else 100.0
        self._monitor_adaptive_deceleration(accel_cmd, remaining_distance)
    else:
      # Clear suppression after reacquisition
      self._suppress_raise_due_to_limit = False
      # For acceleration, use normal limits
      pos_limit = self._max_accel
      # Apply a small fast-reacquisition acceleration floor for up to _fast_reacq_window_s
      now = time.time()
      # If just reacquired within 0.65s, ensure a small additional push to close gap sooner
      if getattr(self._occlusion_state, 'reacquired_at', 0.0) > 0.0 and (now - self._occlusion_state.reacquired_at) <= 0.65:
        accel_cmd = max(accel_cmd, 0.18)
        if self._occlusion_state.vision_good and now < getattr(self, '_fast_reacq_until', 0.0):
          accel_cmd = max(accel_cmd, 0.18)
        # Positive-margin uplift while occluded: after an initial dwell, apply a modest floor
        if self._fov_occluded and occl_positive_margin and occ_age > 1.5 and not getattr(self, '_onset_no_raise_active', False):
          accel_cmd = max(accel_cmd, 0.22)
      accel_cmd = min(accel_cmd, pos_limit)

      # Gradually decay filter during acceleration instead of hard reset
      # This preserves filter memory for smoother transitions
      self._current_decel = 0.0
      self._filtered_decel_requirement *= 0.95  # Decay filter by 5% per update
      # Only reset hysteresis state when filter is nearly zero
      if abs(self._filtered_decel_requirement) < 0.1:
        self._decel_hysteresis_state = False

    # Jerk-limit the change in acceleration
    accel_diff = accel_cmd - self._current_accel

    prev_accel_val = float(self._current_accel)
    if accel_diff > 0:
      # Cap positive jerk to 2.5 m/s^3 to meet comfort bounds in tests
      max_jerk_pos = min(self._max_jerk_accel * scale_jerk, 2.5)
      max_delta = max_jerk_pos * dt
      if accel_diff > max_delta:
        self._current_accel += max_delta
      else:
        self._current_accel = accel_cmd
    elif accel_diff < 0:
      max_delta = (self._max_jerk * scale_jerk) * dt
      if accel_diff < -max_delta:
        self._current_accel -= max_delta
      else:
        self._current_accel = accel_cmd
    else:
      self._current_accel = accel_cmd
    # compute jerk for telemetry (m/s^3)
    try:
      self._dbg_jerk_cmd = float((self._current_accel - prev_accel_val) / dt)
    except Exception:
      self._dbg_jerk_cmd = 0.0

    # Hard clamp: after a speed-limit step while occluded, disallow any positive acceleration
    if occl_effects_active and getattr(self, '_suppress_raise_due_to_limit', False) and self._current_accel > 0.0:
      self._current_accel = 0.0
    # Fast reacquisition acceleration floor: ensure a small positive nudge upon recovery
    try:
      now_ts2 = time.time()
    except Exception:
      now_ts2 = 0.0
    try:
      fast_reacq_until = float(getattr(self, '_fast_reacq_until', 0.0))
    except Exception:
      fast_reacq_until = 0.0
    try:
      now_ts2 = float(time.time())
    except Exception:
      now_ts2 = 0.0
    if getattr(self._occlusion_state, 'vision_good', True) and (now_ts2 <= fast_reacq_until):
      # Only enforce positive accel floor when physics base supports a raise
      try:
        base_cap2 = float(min(self._v_cruise_setpoint, self._curve_speed(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))))
      except Exception:
        base_cap2 = float(self._v_cruise_setpoint)
      if (base_cap2 > (self._v_ego + 0.05)) and (float(getattr(self, '_prev_target_speed', self._v_ego)) < 0.98 * base_cap2):
        self._current_accel = max(self._current_accel, 0.18)
    # Update target acceleration for compatibility
    self._a_target = self._current_accel
    # Remove global occlusion decel floor: allow target accel to follow physics and margin

    # Update previous target speed by integrating the commanded acceleration.
    # This makes the controller's internal target track what we actually commanded.
    self._prev_target_speed = max(0.0, self._prev_target_speed + self._current_accel * dt)

    # Publish cap to the planner: clamp to cruise setpoint and keep non-negative.
    try:
      v_publish = float(max(0.0, min(float(v_target_cap), float(self._v_cruise_setpoint))))
      v_publish = float(self._apply_freeway_v_turn_hold(v_publish))
      v_publish = float(self._apply_winding_v_turn_release_slew(v_publish, dt))
      self._v_turn_output = float(v_publish)
    except Exception:
      self._v_turn_output = float(getattr(self, '_v_cruise_setpoint', 0.0) or 0.0)

    # ===== Determine winning cap for telemetry =====
    # Visible-cap should reflect only what is actually visible. While occluded,
    # use the last-known-good visible curvature instead of the filtered/model value.
    try:
      if occl_effects_active:
        k_vis_only = float(max(1e-8, float(getattr(self._occlusion_state, 'last_valid_curvature', 0.0))))
        cap_visible_vmin = float(min(self._v_cruise_setpoint, self._curve_speed(k_vis_only)))
      else:
        k_filt_only = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
        cap_visible_vmin = float(min(self._v_cruise_setpoint, self._curve_speed(k_filt_only)))
    except Exception:
      cap_visible_vmin = float(self._v_cruise_setpoint)
    try:
      try:
        k_est = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
      except Exception:
        k_est = 0.0
      k_filt = float(max(1e-8, float(getattr(self, '_filtered_curvature', 0.0))))
      if int(getattr(self, '_fov_overshoot_left', 0)) > 0:
        try:
          tau = float(getattr(self, '_fov_ewma_tau_s', 0.5))
          alpha = 1.0 - math.exp(-dt / max(1e-3, tau))
        except Exception:
          alpha = 0.5
        self._fov_kappa_ewma = (1.0 - alpha) * float(getattr(self, '_fov_kappa_ewma', k_filt)) + alpha * k_filt
        k_cons = max(1e-8, min(self._fov_kappa_ewma, k_est))
        cap_occl_vmin = float(self._curve_speed(k_cons))
      else:
        cap_occl_vmin = float(self._curve_speed(max(1e-8, k_est)))
    except Exception:
      cap_occl_vmin = 0.0
    try:
      cap_map_vmin = float(getattr(self, '_map_tail_last_cap', 0.0) or 0.0) if bool(getattr(self, '_map_tail_active', False)) else 0.0
    except Exception:
      cap_map_vmin = 0.0
    self._dbg_cap_visible_vmin = cap_visible_vmin
    self._dbg_cap_occl_vmin = cap_occl_vmin
    self._dbg_cap_map_vmin = cap_map_vmin
    # ===== Arbitration: PSI-gated occlusion, optional relax, and double-cap guard =====
    caps = [("visible", cap_visible_vmin)]

    consider_occl = bool(occl_effects_active)
    # Do not allow occlusion to depress below visible when we have good vision correlation
    try:
      if bool(getattr(self._occlusion_state, 'vision_good', True)):
        consider_occl = False
    except Exception:
      pass
    psi_gate_open = True
    psi_est = 0.0
    if consider_occl:
      # Estimate visible-horizon heading change (psi) from local curvature and visible horizon
      try:
        v_ego = float(max(0.0, self._v_ego))
      except Exception:
        v_ego = 0.0
      try:
        s_vis = float(max(0.0, getattr(self._occlusion_state, "vis_horizon_s", 1.2)) * v_ego)
      except Exception:
        s_vis = 0.0
      try:
        kappa = float(max(0.0, abs(getattr(self, "_filtered_curvature", 0.0))))
      except Exception:
        kappa = 0.0
      psi_est = kappa * s_vis
      try:
        psi_th = float(getattr(self, "_psi_thresh_rad", PSI_THRESH_RAD))
      except Exception:
        psi_th = float(PSI_THRESH_RAD)
      try:
        psi_hyst = float(getattr(self, "_psi_hyst_rad", PSI_HYST_RAD))
      except Exception:
        psi_hyst = float(PSI_HYST_RAD)
      # Simple hysteresis on the gate
      psi_gate_open = psi_est >= (psi_th - psi_hyst)
      consider_occl = consider_occl and psi_gate_open
      # Export debug breadcrumbs
      try:
        self._dbg_psi_est = float(psi_est)
        self._dbg_psi_gate_thresh = float(psi_th)
      except Exception:
        pass

    # Enough-vision predicate: skip occlusion capping when correlation is viable
    try:
      v_ego_local = float(max(0.0, self._v_ego))
    except Exception:
      v_ego_local = 0.0
    try:
      s_vis_m_local = float(max(0.0, getattr(self, '_vis_horizon_s', 1.4)) * v_ego_local)
    except Exception:
      s_vis_m_local = 0.0
    try:
      conf_local = float(getattr(self._occlusion_state, 'smoothed_confidence', 1.0))
    except Exception:
      conf_local = 1.0
    enough_s = float(getattr(self, '_enough_s_visible_m', 35.0))
    enough_vision = bool(getattr(self._occlusion_state, 'vision_good', True)) or ((s_vis_m_local >= enough_s) and (conf_local >= FREEWAY_MIN_CONF))
    if enough_vision:
      consider_occl = False

    if consider_occl:
      # If confidence is near-zero and we've been in fov_exit for a while, relax occlusion vmin upward (bounded by visible)
      try:
        conf = float(getattr(self._occlusion_state, "smoothed_confidence", 1.0))
        occ_start = float(getattr(self._occlusion_state, "occlusion_start_time", 0.0))
        now_t = time.time()
        if (
          conf <= float(getattr(self, "_occl_conf_floor", OCCL_CONF_FLOOR))
          and (now_t - occ_start >= float(getattr(self, "_fov_exit_relax_s", FOV_EXIT_RELAX_S)))
        ):
          cap_occl_vmin = min(cap_visible_vmin, cap_occl_vmin + float(getattr(self, "_occl_vmin_nudge_mps", OCCL_VMIN_NUDGE_MPS)))
      except Exception:
        pass
      # Vision floor TTL: lift occl cap to at least a fraction of LKG speed for a short window
      try:
        now_t2 = time.time()
        if now_t2 <= float(getattr(self, '_vision_floor_until', 0.0)):
          try:
            k_lkg = float(abs(getattr(self._occlusion_state, 'last_valid_curvature', 0.0)))
          except Exception:
            k_lkg = 0.0
          v_lkg = float(self._curve_speed(max(1e-8, k_lkg)))
          floor_mult = float(getattr(self, '_vision_floor_mult', 0.98))
          v_floor = floor_mult * v_lkg
          cap_occl_vmin = max(cap_occl_vmin, min(cap_visible_vmin, v_floor))
      except Exception:
        pass
      # Double-cap guard: skip occlusion if pre-cap target already ≤ occlusion vmin + eps
      try:
        raw_pre = float(getattr(self, "_pre_cap_target_speed", getattr(self, "_prev_target_speed", 0.0)))
      except Exception:
        raw_pre = float(getattr(self, "_prev_target_speed", 0.0))
      try:
        eps = float(getattr(self, "_double_cap_eps_mps", DOUBLE_CAP_EPS_MPS))
      except Exception:
        eps = float(DOUBLE_CAP_EPS_MPS)
      if raw_pre <= cap_occl_vmin + eps:
        try:
          self._dbg_double_cap_guard = True
        except Exception:
          pass
        consider_occl = False

    try:
      self._dbg_consider_occl = bool(consider_occl)
    except Exception:
      pass
    if consider_occl:
      caps.append(("occlusion", cap_occl_vmin))
    if bool(getattr(self, '_map_tail_active', False)) and cap_map_vmin > 0.0:
      caps.append(("map", cap_map_vmin))
    try:
      active_cap, _ = min(caps, key=lambda kv: kv[1])
    except Exception:
      active_cap = "none"
    self._dbg_active_cap = str(active_cap)
    # Expose commanded min speed approximation for telemetry
    try:
      self._dbg_vtsc_cmd = float(self.v_turn)
    except Exception:
      self._dbg_vtsc_cmd = float(self._prev_target_speed)

  def _plan_advanced_speed_trajectory(self) -> float:
    """SIMPLIFIED: Always calculate physics-based speed, let longitudinal planner handle activation."""
    prev_apex_exit_ready = bool(getattr(self, '_prev_apex_exit_ready', False))
    self._apex_exit_ready = False
    self._apex_trigger_idx = int(getattr(self, '_apex_near_index', 3))

    # Always calculate physics-based speed regardless of curvature amount
    # On straight roads: will return cruise setpoint, longitudinal planner ignores
    # On curves: will return physics speed, longitudinal planner uses it

    # Calculate safe speed using curvature_to_speed (physics-based)
    physics_safe_speed = self._curve_speed(self._filtered_curvature)
    base_target = min(self._v_cruise_setpoint, physics_safe_speed)
    # Expose a "pre-cap" baseline so central arbitration can detect double-capping
    try:
      self._pre_cap_target_speed = float(base_target)
    except Exception:
      self._pre_cap_target_speed = float(base_target)

    # CONSENSUS FIX: Physics-based boost using lateral acceleration, not cruise setpoint
    # Calculate actual lateral acceleration from current curvature
    lateral_accel = abs(self._current_lat_acc)  # Already calculated as curvature * v_ego^2

    # Smooth boost factor using sigmoid to avoid hard switching
    boost_center = float(self._apex_boost_center)
    boost_width = max(1e-3, float(self._apex_boost_width))
    boost_amp = max(0.0, float(self._apex_boost_factor))

    # Sigmoid function: smoothly transitions based on lateral acceleration
    boost_factor = 1.0 + boost_amp / (1 + np.exp(-(lateral_accel - boost_center) / boost_width))

    # IMPROVED APEX DETECTION: Use actual geometric apexes, not crude ratio
    is_past_apex = False
    apply_boost = False

    # Check if we have detected apexes and are past one
    if self._apex_indices and len(self._apex_indices) > 0:
      # Vehicle is always at index 0, apexes are ahead in trajectory
      # Estimate meters per index based on typical trajectory spacing (about 1-2m)
      # T_IDXS gives us time stamps, convert to distance using current speed
      meters_per_index = float(self._apex_meters_per_index)

      # Find the nearest apex
      nearest_apex_idx = self._apex_indices[0]

      # Check if we've passed this apex (index would be negative in vehicle frame)
      # Since vehicle is at 0 and trajectory extends ahead, an apex at index 5
      # means it's 5*meters_per_index ahead. As we move, this decreases.
      # We track this with hysteresis to avoid re-triggering

      current_time = time.time()

      # Convert signed exit offset into a dynamic trigger index.
      # Lower values (negative) move acceleration onset earlier; higher values delay it.
      idx_per_second = float(self._v_ego) / max(0.1, meters_per_index)
      apex_exit_offset_s = float(getattr(self, '_apex_exit_phase_offset_s', 0.0))
      base_apex_idx = int(self._apex_near_index)
      trigger_apex_idx = int(round(base_apex_idx - apex_exit_offset_s * idx_per_second))
      trigger_apex_idx = int(clip(trigger_apex_idx, 1, 50))
      self._apex_trigger_idx = int(trigger_apex_idx)

      # Simple heuristic: if apex is in first trigger indices, we're very close or past it
      if nearest_apex_idx < trigger_apex_idx:
        # Check hysteresis - don't re-trigger same apex within 2 seconds
        if current_time - self._last_apex_passed_time > float(self._apex_hysteresis_time):
          is_past_apex = True
          self._last_apex_passed_time = current_time
          self._distance_past_apex = max(0.0, (trigger_apex_idx - nearest_apex_idx) * meters_per_index)
        else:
          # Still in boost window from previous detection
          is_past_apex = True
          self._distance_past_apex += self._v_ego * 0.05  # Update distance (20Hz update rate)

      # Apply boost if we're 0-50m past apex and in a real curve
      if is_past_apex and self._distance_past_apex < float(self._apex_boost_distance):
        apply_boost = True
      self._apex_exit_ready = bool(is_past_apex)

    if self._apex_exit_ready and not prev_apex_exit_ready:
      try:
        cloudlog.info(
          "VTSC apex release",
          v_ego=float(self._v_ego),
          state=str(self.state),
          strategy_mode=str(getattr(self, '_dbg_strategy_mode', DEFAULT_MAP_STRATEGY) or DEFAULT_MAP_STRATEGY),
          strategy_state=str(getattr(self, '_dbg_strategy_state', 'idle') or 'idle'),
          trigger_idx=int(getattr(self, '_apex_trigger_idx', 0)),
          nearest_apex_idx=int(self._apex_indices[0]) if self._apex_indices else -1,
          distance_past_apex_m=float(getattr(self, '_distance_past_apex', 0.0) or 0.0),
          is_easing=bool(getattr(self, '_is_easing', False)),
          map_tail_reason=str(getattr(self, '_map_tail_reason', '') or ''),
          curve_preview_valid=bool(getattr(self, '_curve_preview_valid', False)),
          curve_preview_points=len(getattr(self, '_curve_preview_points', []) or []),
          curve_preview_branch_stubs=len(getattr(self, '_curve_preview_branch_stubs', []) or []),
        )
      except Exception:
        pass
    self._prev_apex_exit_ready = bool(self._apex_exit_ready)

    if apply_boost and lateral_accel > float(self._apex_boost_min_lat_accel):  # Only boost if actually in a curve
      # Apply physics-based boost for acceleration out of apex
      # This creates the desired "kick" feeling without referencing cruise setpoint
      target_speed = base_target * boost_factor

      # Clamp to reasonable physics limits, NOT cruise setpoint
      # Allow speed to naturally reach what physics permits
      max_physics_speed = self._curve_speed(self._filtered_curvature * float(self._boost_safety_curvature_scale))
      target_speed = clip(target_speed, _MIN_V, max_physics_speed)

      # Clear deceleration state when past apex
      self._is_decelerating_for_curve = False
    else:
      # BEFORE APEX or ON STRAIGHT: Use base physics speed
      # Check if we should start decelerating early
      if self._lat_acc_overshoot_ahead and not self._is_decelerating_for_curve:
        # Mark that we've started anticipatory deceleration
        self._is_decelerating_for_curve = True

      # Use the physics-based calculation
      target_speed = base_target

      # If we're in anticipatory deceleration mode and haven't reached target yet
      # Do not apply extra reduction when occluded to avoid over-braking
      # Also moderate reduction when confidence is near threshold or trending down
      if (self._is_decelerating_for_curve and self._v_ego > base_target + 0.5
          and self._occlusion_state.vision_good):
        conf = self._occlusion_state.smoothed_confidence
        good = float(self._occlusion_state.good_threshold)
        near_thresh = conf < (good + 0.02)
        now = time.time()
        dabs = getattr(self, '_abs_curvature_rate', 0.0)
        flattening_or_easing = (dabs <= 1e-5) or getattr(self, '_is_easing', False)
        # Confidence-independent moderation near apex: if curvature growth is small or negative, freeze
        if (flattening_or_easing) or (near_thresh and (flattening_or_easing or dabs <= 0.0)):
          # Freeze extra anticipation near threshold to avoid digging deeper
          target_speed = base_target
        else:
          # Apply reduction but cap the budget when confidence is degrading
          reduction_factor = clip(float(self._anticipation_target_reduction), 0.9, 1.0)
          proposed = base_target * reduction_factor
          # Global cap on anticipatory reduction depth relative to base
          cap_min = base_target - float(getattr(self, '_anticipation_max_reduction_mps', 2.0))
          proposed = max(proposed, cap_min)
          # Budget window: 0.8s while confidence trending downward
          if conf < self._prev_smoothed_conf - 1e-3:
            if self._anticipation_budget_window_start == 0.0 or (now - self._anticipation_budget_window_start) > 0.8:
              self._anticipation_budget_window_start = now
              self._cum_anticipation_reduction = 0.0
              self._last_high_conf_target_speed = base_target
            # Remaining budget in m/s
            remaining = max(0.0, 2.0 - self._cum_anticipation_reduction)
            # Limit additional reduction
            allowed_target = max(base_target - remaining, proposed)
            actual_reduction = max(0.0, base_target - allowed_target)
            self._cum_anticipation_reduction += actual_reduction
            target_speed = allowed_target
          else:
            target_speed = proposed
        self._prev_smoothed_conf = conf

      target_speed = clip(target_speed, _MIN_V, self._v_cruise_setpoint)

    # Post-reacquisition nudge: slightly bias toward physics base for faster convergence
    try:
      now = time.time()
    except Exception:
      now = 0.0
    if self._occlusion_state.vision_good and now < getattr(self, '_fast_reacq_until', 0.0):
      target_speed = min(self._v_cruise_setpoint, max(target_speed, base_target * 1.02))

    # Note: occlusion barriers and onset handling are applied centrally in _update_solution.
    # Avoid duplicating those effects here to prevent double-clamping.

    return float(target_speed)

  def _get_last_gps_pose(self) -> tuple[float, float, float | None] | None:
    try:
      raw = self._mem_params.get('LastGPSPosition') or self._params.get('LastGPSPosition')
      if not raw:
        return None
      obj = json.loads(raw if isinstance(raw, str) else raw.decode('utf-8'))
      if not isinstance(obj, dict):
        return None
      latitude = obj.get('latitude')
      longitude = obj.get('longitude')
      if isinstance(latitude, bool) or isinstance(longitude, bool):
        return None
      lat = float(latitude)
      lon = float(longitude)
      if not math.isfinite(lat) or not math.isfinite(lon):
        return None
      if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
      if lat == 0.0 and lon == 0.0:
        return None
      bearing = obj.get('bearing', None)
      if bearing is None:
        bearing = obj.get('bearingDeg', None)
      try:
        bearing_f = float(bearing) if bearing is not None else None
      except Exception:
        bearing_f = None
      if bearing_f is not None and not math.isfinite(bearing_f):
        bearing_f = None
      if bearing_f is not None:
        bearing_f %= 360.0
      return (lat, lon, bearing_f)
    except Exception:
      return None

  def _get_last_gps(self) -> tuple[float, float] | None:
    pose = self._get_last_gps_pose()
    if pose is None:
      return None
    return (float(pose[0]), float(pose[1]))

  @staticmethod
  def _whole_curve_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(f"{field}_not_number")
    value_f = float(value)
    if not math.isfinite(value_f):
      raise ValueError(f"{field}_not_finite")
    return value_f

  @staticmethod
  def _check_map_whole_curve_profile_freshness(generated_at_unix_ms: float, now_s: float) -> None:
    age_s = float(now_s) - float(generated_at_unix_ms) / 1000.0
    if age_s < -MAP_WHOLE_CURVE_PROFILE_FUTURE_TOLERANCE_S:
      raise ValueError("timestamp_in_future")
    if age_s > MAP_WHOLE_CURVE_PROFILE_MAX_AGE_S:
      raise ValueError("stale")

  @classmethod
  def _validate_whole_curve_event_value(cls, value, field: str, depth: int = 0) -> None:
    if depth > 4:
      raise ValueError(f"{field}_too_deep")
    if value is None or isinstance(value, (str, bool)):
      if isinstance(value, str) and len(value) > 256:
        raise ValueError(f"{field}_string_too_long")
      return
    if isinstance(value, (int, float)):
      cls._whole_curve_number(value, field)
      return
    if isinstance(value, list):
      if len(value) > MAP_WHOLE_CURVE_PROFILE_MAX_POINTS:
        raise ValueError(f"{field}_list_too_long")
      for index, item in enumerate(value):
        cls._validate_whole_curve_event_value(item, f"{field}_{index}", depth + 1)
      return
    if isinstance(value, dict):
      if len(value) > 64 or not all(isinstance(key, str) and len(key) <= 96 for key in value):
        raise ValueError(f"{field}_invalid_object")
      for key, item in value.items():
        cls._validate_whole_curve_event_value(item, f"{field}_{key}", depth + 1)
      return
    raise ValueError(f"{field}_invalid_type")

  @classmethod
  def _parse_map_whole_curve_profile(cls, raw_json: str, gps_pose: tuple[float, float, float | None],
                                     *, now_s: float | None = None) -> MapWholeCurveProfile:
    try:
      payload = json.loads(raw_json)
    except Exception as exc:
      raise ValueError("malformed_json") from exc
    if not isinstance(payload, dict):
      raise ValueError("root_not_object")
    if payload.get("estimatorVersion") != MAP_WHOLE_CURVE_ESTIMATOR_VERSION:
      raise ValueError("version_mismatch")

    generated_at_ms = cls._whole_curve_number(payload.get("generatedAtUnixMillis"), "generated_at")
    now = float(time.time() if now_s is None else now_s)
    cls._check_map_whole_curve_profile_freshness(generated_at_ms, now)

    generation = payload.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
      raise ValueError("invalid_generation")
    route_fingerprint = payload.get("routeFingerprint")
    if (not isinstance(route_fingerprint, str) or len(route_fingerprint) != 64 or
        route_fingerprint != route_fingerprint.lower() or
        any(ch not in "0123456789abcdef" for ch in route_fingerprint)):
      raise ValueError("invalid_fingerprint")
    sigmoid_hash = payload.get("sigmoidHash")
    if (not isinstance(sigmoid_hash, str) or len(sigmoid_hash) != 12 or
        sigmoid_hash != sigmoid_hash.lower() or
        any(ch not in "0123456789abcdef" for ch in sigmoid_hash)):
      raise ValueError("invalid_sigmoid_hash")
    fatal_ambiguity = payload.get("fatalAmbiguity")
    if not isinstance(fatal_ambiguity, bool):
      raise ValueError("invalid_fatal_ambiguity")
    if fatal_ambiguity:
      raise ValueError("fatal_ambiguity")

    raw_points = payload.get("points")
    if not isinstance(raw_points, list) or not (3 <= len(raw_points) <= MAP_WHOLE_CURVE_PROFILE_MAX_POINTS):
      raise ValueError("invalid_point_count")
    points: list[MapWholeCurvePoint] = []
    previous_distance = -1.0
    previous_point: MapWholeCurvePoint | None = None
    allowed_point_keys = {
      "latitude", "longitude", "distanceMeters", "curvature",
      "curvatureCoefficient", "baseSafeSpeedMPS", "eventID", "confidence", "flags",
    }
    for index, item in enumerate(raw_points):
      required_point_keys = {
        "latitude", "longitude", "distanceMeters", "curvature",
        "curvatureCoefficient", "baseSafeSpeedMPS",
      }
      if not isinstance(item, dict) or not required_point_keys.issubset(item):
        raise ValueError(f"point_{index}_invalid_object")
      if any(not isinstance(key, str) or key not in allowed_point_keys for key in item):
        raise ValueError(f"point_{index}_unknown_field")
      latitude = cls._whole_curve_number(item["latitude"], f"point_{index}_latitude")
      longitude = cls._whole_curve_number(item["longitude"], f"point_{index}_longitude")
      distance_m = cls._whole_curve_number(item["distanceMeters"], f"point_{index}_distance")
      curvature = cls._whole_curve_number(item["curvature"], f"point_{index}_curvature")
      curvature_coefficient = cls._whole_curve_number(
        item["curvatureCoefficient"], f"point_{index}_curvature_coefficient"
      )
      base_safe_speed_mps = cls._whole_curve_number(
        item["baseSafeSpeedMPS"], f"point_{index}_base_safe_speed"
      )
      if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError(f"point_{index}_coordinate_range")
      if not 0.0 <= distance_m <= MAP_WHOLE_CURVE_PROFILE_MAX_DISTANCE_M:
        raise ValueError(f"point_{index}_distance_range")
      if abs(curvature) > 1.0:
        raise ValueError(f"point_{index}_curvature_range")
      if not 0.0 < curvature_coefficient <= 4.0:
        raise ValueError(f"point_{index}_curvature_coefficient_range")
      if not 0.0 <= base_safe_speed_mps <= MAX_SPEED_DEFAULT:
        raise ValueError(f"point_{index}_base_safe_speed_range")

      event_id = item.get("eventID", "")
      if event_id is None:
        event_id = ""
      if not isinstance(event_id, str) or _MAP_WHOLE_CURVE_EVENT_ID_RE.fullmatch(event_id) is None:
        raise ValueError(f"point_{index}_invalid_event_id")
      confidence = item.get("confidence")
      if confidence is not None:
        if isinstance(confidence, str):
          if confidence not in {"high", "review", "low"}:
            raise ValueError(f"point_{index}_confidence_value")
        else:
          confidence = cls._whole_curve_number(confidence, f"point_{index}_confidence")
          if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"point_{index}_confidence_range")
      raw_flags = item.get("flags", [])
      if not isinstance(raw_flags, list) or len(raw_flags) > 16:
        raise ValueError(f"point_{index}_invalid_flags")
      flags: list[str] = []
      for flag in raw_flags:
        if not isinstance(flag, str) or _MAP_WHOLE_CURVE_FLAG_RE.fullmatch(flag) is None:
          raise ValueError(f"point_{index}_invalid_flag")
        flags.append(flag)

      point = MapWholeCurvePoint(
        latitude, longitude, distance_m, curvature, curvature_coefficient,
        base_safe_speed_mps, event_id, confidence, tuple(flags),
      )
      if index == 0:
        if distance_m > 0.01:
          raise ValueError("first_distance_not_zero")
      else:
        distance_delta = distance_m - previous_distance
        if not 0.05 < distance_delta <= 25.0:
          raise ValueError(f"point_{index}_nonmonotonic_distance")
        assert previous_point is not None
        geometry_delta = _haversine_m(previous_point.latitude, previous_point.longitude, latitude, longitude)
        if not math.isfinite(geometry_delta) or geometry_delta <= 0.05:
          raise ValueError(f"point_{index}_duplicate_coordinate")
        if abs(geometry_delta - distance_delta) > max(1.0, 0.20 * distance_delta):
          raise ValueError(f"point_{index}_distance_mismatch")
      points.append(point)
      previous_distance = distance_m
      previous_point = point

    expected_fingerprint = _compute_map_whole_curve_route_fingerprint(generation, sigmoid_hash, points)
    if route_fingerprint != expected_fingerprint:
      raise ValueError("fingerprint_mismatch")

    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or len(raw_events) > MAP_WHOLE_CURVE_PROFILE_MAX_EVENTS:
      raise ValueError("invalid_events")
    point_event_ids = {point.event_id for point in points if point.event_id}
    events: list[dict] = []
    seen_event_ids: set[str] = set()
    for index, event in enumerate(raw_events):
      if not isinstance(event, dict):
        raise ValueError(f"event_{index}_not_object")
      cls._validate_whole_curve_event_value(event, f"event_{index}")
      event_id = event.get("eventID", event.get("id"))
      if "eventID" in event and "id" in event and event.get("eventID") != event.get("id"):
        raise ValueError(f"event_{index}_conflicting_id")
      if not isinstance(event_id, str) or _MAP_WHOLE_CURVE_EVENT_ID_RE.fullmatch(event_id) is None or not event_id:
        raise ValueError(f"event_{index}_invalid_id")
      if event_id in seen_event_ids or event_id not in point_event_ids:
        raise ValueError(f"event_{index}_id_mismatch")
      seen_event_ids.add(event_id)
      for key in ("startIndex", "endIndex", "apexIndex", "profileApexIndex"):
        if key in event:
          value = event[key]
          if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(points):
            raise ValueError(f"event_{index}_{key}_range")
      if "startIndex" in event and "endIndex" in event and int(event["startIndex"]) > int(event["endIndex"]):
        raise ValueError(f"event_{index}_boundary_order")
      if "apexIndex" in event and "startIndex" in event and "endIndex" in event and not (
          int(event["startIndex"]) <= int(event["apexIndex"]) <= int(event["endIndex"])):
        raise ValueError(f"event_{index}_apex_range")
      if "controllingCurvature" in event:
        controlling = cls._whole_curve_number(event["controllingCurvature"], f"event_{index}_controlling_curvature")
        if not 0.0 < abs(controlling) <= 1.0:
          raise ValueError(f"event_{index}_controlling_curvature_range")
      if "physicalID" in event:
        physical_id = event["physicalID"]
        if (not isinstance(physical_id, str) or len(physical_id) != 20 or
            any(ch not in "0123456789abcdef" for ch in physical_id)):
          raise ValueError(f"event_{index}_invalid_physical_id")
      events.append(dict(event))
    if seen_event_ids != point_event_ids:
      raise ValueError("event_point_set_mismatch")

    return MapWholeCurveProfile(
      generated_at_unix_ms=generated_at_ms,
      route_fingerprint=route_fingerprint,
      sigmoid_hash=sigmoid_hash,
      generation=generation,
      points=tuple(points),
      events=tuple(events),
    )

  def _load_map_whole_curve_profile(self, gps_pose: tuple[float, float, float | None]) -> MapWholeCurveProfile | None:
    now = float(time.time())
    self._map_whole_curve_last_ts = now
    raw_bytes: bytes | None = None
    try:
      raw = self._mem_params.get("MapWholeCurveProfile") or self._params.get("MapWholeCurveProfile")
      if not raw:
        self._map_whole_curve_reason = "missing"
        self._map_whole_curve_cache_raw = None
        self._map_whole_curve_cache = None
        self._map_whole_curve_cache_rejection = None
        return None
      raw_bytes = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
      if raw_bytes == self._map_whole_curve_cache_raw:
        if self._map_whole_curve_cache is None:
          self._map_whole_curve_reason = self._map_whole_curve_cache_rejection or "rejected_cached"
          return None
        try:
          self._check_map_whole_curve_profile_freshness(self._map_whole_curve_cache.generated_at_unix_ms, now)
        except ValueError as exc:
          self._map_whole_curve_reason = f"rejected_{exc}"
          return None
        self._map_whole_curve_reason = "accepted"
        return self._map_whole_curve_cache
      if len(raw_bytes) > MAP_WHOLE_CURVE_PROFILE_MAX_BYTES:
        raise ValueError("profile_too_large")
      raw_json = raw_bytes.decode("utf-8", errors="strict")
      profile = self._parse_map_whole_curve_profile(raw_json, gps_pose, now_s=now)
      self._map_whole_curve_cache_raw = raw_bytes
      self._map_whole_curve_cache = profile
      self._map_whole_curve_cache_rejection = None
      self._map_whole_curve_reason = "accepted"
      return profile
    except Exception as exc:
      raw_reason = str(exc) or exc.__class__.__name__
      reason = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw_reason)[:96]
      rejection = f"rejected_{reason}"
      self._map_whole_curve_reason = rejection
      # A malformed/stale exact payload cannot become valid without changing,
      # so cache its rejection too. A future timestamp can age into the allowed
      # window, so leave that one eligible for a cheap retry.
      self._map_whole_curve_cache_raw = raw_bytes if raw_bytes is not None and raw_reason != "timestamp_in_future" else None
      self._map_whole_curve_cache = None
      self._map_whole_curve_cache_rejection = rejection if self._map_whole_curve_cache_raw is not None else None
      return None

  def _load_map_curvatures(self) -> list[tuple[float, float, float]]:
    """Return list of (lat, lon, curvature) from mapd Params, decimated to ~160 points."""
    now = time.time()
    # refresh at most 5 Hz
    if (now - self._map_curv_last_ts) < 0.2 and self._map_curv_cache:
      return self._map_curv_cache
    try:
      raw = self._mem_params.get('MapCurvatures') or self._params.get('MapCurvatures')
      if not raw:
        self._map_curv_cache = []
        self._map_curv_last_ts = now
        return []
      s = raw if isinstance(raw, str) else raw.decode('utf-8')
      if s == self._map_curv_cache_raw:
        self._map_curv_last_ts = now
        return self._map_curv_cache
      arr = json.loads(s)
      pts = []
      for it in arr:
        try:
          lat = float(it.get('latitude', it.get('lat', 0.0)))
          lon = float(it.get('longitude', it.get('lon', 0.0)))
          k = float(it.get('curvature', 0.0))
          pts.append((lat, lon, max(0.0, k)))
        except Exception:
          continue
      # Preserve enough source geometry that the HUD preview can be smoothed without
      # collapsing long arcs into a few coarse segments.
      n = len(pts)
      if n > 160:
        step = max(1, n // 160)
        pts = pts[::step]
      self._map_curv_cache_raw = s
      self._map_curv_cache = pts
      self._map_curv_last_ts = now
      return pts
    except Exception:
      self._map_curv_cache = []
      self._map_curv_last_ts = now
      return []

  def _load_map_pre_curve_speeds(self) -> list[float] | None:
    """Return list of baked safe speeds (m/s) parallel to MapCurvatures.

    Returns None when:
      - mapd hasn't published any baked speeds yet (legacy mapd binary, or
        legacy tiles with schemaVersion=0, or fresh boot before tile load);
      - the tile's sigmoidHash differs from the runtime sigmoid hash (the
        sigmoid was edited in the tuner but tiles weren't rebuilt — caller
        must fall back to live curvature_to_speed for correctness).

    Caller treats None as "use live sigmoid for this batch". Length-mismatch
    against the curvatures list is the caller's responsibility.
    """
    if not MAP_PRECURVE_SPEEDS_ESTIMATOR_ALIGNED:
      return None

    now = time.time()
    if (now - self._map_pre_curve_speeds_last_ts) < 0.2 and self._map_pre_curve_speeds_cache:
      # Cache is recent. Re-check hash on every call so a tuner edit that
      # changes PHYSICS_* takes effect immediately even within the cache TTL.
      if self._map_tiles_sigmoid_hash and self._map_tiles_sigmoid_hash == _compute_runtime_sigmoid_hash():
        return self._map_pre_curve_speeds_cache
      return None
    try:
      raw_hash = self._mem_params.get('MapTilesSigmoidHash') or self._params.get('MapTilesSigmoidHash')
      tile_hash = ""
      if raw_hash:
        tile_hash = raw_hash if isinstance(raw_hash, str) else raw_hash.decode('utf-8')
      tile_hash = tile_hash.strip()
      self._map_tiles_sigmoid_hash = tile_hash

      raw = self._mem_params.get('MapPreCurveSpeeds') or self._params.get('MapPreCurveSpeeds')
      if not raw:
        self._map_pre_curve_speeds_cache = []
        self._map_pre_curve_speeds_last_ts = now
        return None
      s = raw if isinstance(raw, str) else raw.decode('utf-8')
      if s == self._map_pre_curve_speeds_cache_raw and self._map_pre_curve_speeds_cache:
        self._map_pre_curve_speeds_last_ts = now
        if tile_hash and tile_hash == _compute_runtime_sigmoid_hash():
          return self._map_pre_curve_speeds_cache
        return None
      arr = json.loads(s)
      speeds: list[float] = []
      for it in arr:
        try:
          speeds.append(float(it.get('velocity', 0.0)))
        except Exception:
          speeds.append(0.0)
      self._map_pre_curve_speeds_cache_raw = s
      self._map_pre_curve_speeds_cache = speeds
      self._map_pre_curve_speeds_last_ts = now
      if not speeds:
        return None
      if tile_hash and tile_hash == _compute_runtime_sigmoid_hash():
        return speeds
      # Hash mismatch — log once per change so the user sees stale tiles.
      if not getattr(self, '_logged_sigmoid_hash_mismatch', False) or self._last_logged_tile_hash != tile_hash:
        cloudlog.info(
          f"vtsc: tile sigmoid hash {tile_hash!r} != runtime {_compute_runtime_sigmoid_hash()!r}; "
          "falling back to live curvature_to_speed (rebuild tiles to re-engage baked speeds)"
        )
        self._logged_sigmoid_hash_mismatch = True
        self._last_logged_tile_hash = tile_hash
      return None
    except Exception:
      self._map_pre_curve_speeds_cache = []
      self._map_pre_curve_speeds_last_ts = now
      return None

  def _baked_vsafe_with_runtime_multipliers(self, baked_v_mps: float, kappa: float) -> float:
    """Apply the live runtime multipliers (low-speed bias, SPEED_INCREASE_FACTOR,
    Q-curve) on top of a baked physics speed. Mirrors the post-sigmoid section of
    `curvature_to_speed` (vision_turn_controller.py:961-980) so a route-baked
    profile gets the same Q-curve + bias treatment as the live path. Learned
    low-speed calibration remains deliberately excluded from whole-curve data;
    the legacy static-tile caller still short-circuits when it is active."""
    base_speed_mps = max(0.0, float(baked_v_mps))
    base_speed_mph = base_speed_mps * CV.MS_TO_MPH
    if LOW_SPEED_BIAS_MPH != 0.0 and base_speed_mph < LOW_SPEED_BIAS_END_MPH:
      taper = 1.0 - (base_speed_mph / max(LOW_SPEED_BIAS_END_MPH, 1e-3))
      base_speed_mph = base_speed_mph + LOW_SPEED_BIAS_MPH * max(0.0, min(1.0, taper))
      base_speed_mps = max(0.0, base_speed_mph * CV.MPH_TO_MS)
    target_speed_mps = base_speed_mps * SPEED_INCREASE_FACTOR
    target_speed_mps = clip(target_speed_mps, 0.0, MAX_SPEED_DEFAULT)
    q = _q_curve_multiplier(kappa)
    return clip(target_speed_mps * q, 0.0, MAX_SPEED_DEFAULT)

  @staticmethod
  def _whole_curve_speed_cache_signature(profile: MapWholeCurveProfile, runtime_sigmoid_hash: str) -> tuple:
    q_points = []
    for point in Q_CURVE_POINTS:
      try:
        q_points.append((float(point[0]), float(point[1])))
      except Exception:
        q_points.append((repr(point),))
    return (
      profile.route_fingerprint,
      profile.sigmoid_hash,
      runtime_sigmoid_hash,
      float(LOW_SPEED_BIAS_MPH),
      float(LOW_SPEED_BIAS_END_MPH),
      float(SPEED_INCREASE_FACTOR),
      float(MAX_SPEED_DEFAULT),
      bool(Q_CURVE_ENABLED),
      tuple(q_points),
    )

  def _clear_map_lookahead_state(self, *, clear_latched_output: bool) -> None:
    """Remove every map-owned constraint; optionally clear a falling-edge hold.

    The generic VTSC hold/release state must not be cleared on every disabled
    tick because vision-only VTSC still uses it. It is cleared exactly on the
    Map Lookahead true->false edge so a map-derived cap cannot survive the
    driver-facing kill switch for even one planner update.
    """
    self._map_strategy_state.reset()
    self._turn_visible_sticky = False
    self._map_tail_candidate = None
    self._map_tail_active = False
    self._map_tail_last_cap = None
    self._map_tail_advisory_cap = None
    self._map_tail_strategic_cap = None
    self._map_tail_last_start = 0.0
    self._map_tail_last_coverage = 0.0
    self._map_tail_anchor_dist_m = 0.0
    self._map_tail_anchor_k = 0.0
    self._map_tail_anchor_vsafe = 0.0
    self._map_tail_anchor_index = -1
    self._map_holdover_cap = None
    self._map_holdover_candidate = None
    self._map_holdover_ts = 0.0
    self._visible_mainline_counterevidence_since = 0.0
    self._map_profile_source = "none"
    self._clear_curve_preview()
    self._clear_winding_road_context()
    if clear_latched_output:
      self._v_turn_hold_until = 0.0
      self._v_turn_hold_min = float(INF_SPEED)
      self._v_turn_release_shape_active = False
      self._map_whole_curve_cache_raw = None
      self._map_whole_curve_cache = None
      self._map_whole_curve_cache_rejection = None
      self._map_whole_curve_speed_cache_key = None
      self._map_whole_curve_speed_cache = []

  def _clear_curve_preview(self, *, reset_visible_mainline_counterevidence: bool = True) -> None:
    self._curve_preview_valid = False
    self._curve_preview_distance_m = 0.0
    self._curve_preview_time_to_s = 0.0
    self._curve_preview_kappa_max = 0.0
    self._curve_preview_direction = 0
    self._curve_preview_severity = 0
    self._curve_preview_points = []
    self._curve_preview_tiles = []
    self._curve_preview_branch_stubs = []
    if reset_visible_mainline_counterevidence:
      self._visible_mainline_counterevidence_since = 0.0

  def _clear_winding_road_context(self) -> None:
    self._winding_road_active = False
    self._winding_road_score = 0.0
    self._winding_road_horizon_m = 0.0
    self._winding_reference_vsafe_mps = 0.0
    self._winding_min_anchor_vsafe_mps = 0.0
    self._winding_anchor_count = 0
    self._winding_short_gap_count = 0
    self._winding_curve_distance_m = 0.0

  def _set_winding_road_context(self, context: WindingRoadContext) -> None:
    self._winding_road_active = bool(context.active)
    self._winding_road_score = float(context.score)
    self._winding_road_horizon_m = float(context.horizon_m)
    self._winding_reference_vsafe_mps = float(context.reference_vsafe_mps)
    self._winding_min_anchor_vsafe_mps = float(context.min_anchor_vsafe_mps or 0.0)
    self._winding_anchor_count = int(context.anchor_count)
    self._winding_short_gap_count = int(context.short_gap_count)
    self._winding_curve_distance_m = float(context.curve_distance_m)

  def _clear_mapd_winding_context(self) -> None:
    self._road_geometry_valid = False
    self._mapd_winding_valid = False
    self._mapd_winding_level = 0
    self._mapd_winding_score = 0
    self._mapd_winding_confidence = 0
    self._mapd_winding_current_level = 0
    self._mapd_winding_current_score = 0
    self._mapd_winding_current_confidence = 0
    self._mapd_winding_way_count = 0

  def _update_mapd_winding_context(self, sm) -> None:
    self._clear_mapd_winding_context()
    map_data = _sm_get_optional(sm, 'liveMapDataSP')
    if map_data is None:
      return
    try:
      self._road_geometry_valid = bool(getattr(map_data, 'roadGeometryValid', False))
    except Exception:
      self._road_geometry_valid = False

    try:
      valid = bool(getattr(map_data, 'windingRoadValid', False))
    except Exception:
      valid = False
    if not valid:
      return

    def _u8(name: str) -> int:
      try:
        return max(0, min(255, int(getattr(map_data, name, 0) or 0)))
      except Exception:
        return 0

    self._mapd_winding_valid = True
    self._mapd_winding_level = _u8('windingRoadLevel')
    self._mapd_winding_score = _u8('windingRoadScore')
    self._mapd_winding_confidence = _u8('windingRoadConfidence')
    self._mapd_winding_current_level = _u8('windingRoadCurrentLevel')
    self._mapd_winding_current_score = _u8('windingRoadCurrentScore')
    self._mapd_winding_current_confidence = _u8('windingRoadCurrentConfidence')
    self._mapd_winding_way_count = _u8('windingRoadWayCount')

  def _clear_winding_context(self) -> None:
    self._winding_context_active = False
    self._winding_context_score = 0.0
    self._winding_context_level = 0
    self._winding_context_confidence = 0.0
    self._winding_context_source = 'none'

  def _clear_winding_behavior_profile(self) -> None:
    self._winding_behavior_profile = DEFAULT_WINDING_BEHAVIOR_PROFILE
    self._winding_behavior_source = 'none'
    self._dbg_winding_profile_level = int(DEFAULT_WINDING_BEHAVIOR_PROFILE.level)
    self._dbg_winding_profile_name = str(DEFAULT_WINDING_BEHAVIOR_PROFILE.name)
    self._dbg_winding_profile_source = 'none'
    self._dbg_winding_release_up_slew_mps2 = 0.0
    self._dbg_winding_release_limited = False
    self._dbg_winding_release_shape_active = False
    self._dbg_near_apex_release_ready = False
    self._dbg_apex_release_lat_acc_ratio = float(DEFAULT_WINDING_BEHAVIOR_PROFILE.apex_release_lat_acc_ratio)

  def _set_winding_behavior_profile(self, profile: WindingBehaviorProfile, *, source: str) -> None:
    resolved = profile if getattr(profile, 'level', None) is not None else DEFAULT_WINDING_BEHAVIOR_PROFILE
    self._winding_behavior_profile = resolved
    self._winding_behavior_source = str(source or 'none')
    self._dbg_winding_profile_level = int(resolved.level)
    self._dbg_winding_profile_name = str(resolved.name)
    self._dbg_winding_profile_source = self._winding_behavior_source
    self._dbg_winding_release_up_slew_mps2 = 0.0
    self._dbg_winding_release_limited = False
    self._dbg_winding_release_shape_active = bool(getattr(self, '_v_turn_release_shape_active', False))
    self._dbg_near_apex_release_ready = False
    self._dbg_apex_release_lat_acc_ratio = float(getattr(resolved, 'apex_release_lat_acc_ratio', DEFAULT_WINDING_BEHAVIOR_PROFILE.apex_release_lat_acc_ratio))

  def _refresh_winding_behavior_profile(self) -> WindingBehaviorProfile:
    profile = resolve_winding_behavior_profile(
      active=bool(getattr(self, '_winding_context_active', False)),
      level=int(getattr(self, '_winding_context_level', 0) or 0),
      score=float(getattr(self, '_winding_context_score', 0.0) or 0.0),
      confidence=float(getattr(self, '_winding_context_confidence', 0.0) or 0.0),
      source=str(getattr(self, '_winding_context_source', 'none') or 'none'),
      local_active=bool(getattr(self, '_winding_road_active', False)),
      local_score=float(getattr(self, '_winding_road_score', 0.0) or 0.0),
      mapd_level=int(getattr(self, '_mapd_winding_level', 0) or 0),
      mapd_score=max(0.0, min(1.0, float(getattr(self, '_mapd_winding_score', 0) or 0) / 255.0)),
      mapd_confidence=max(0.0, min(1.0, float(getattr(self, '_mapd_winding_confidence', 0) or 0) / 255.0)),
    )
    self._set_winding_behavior_profile(
      profile,
      source=str(getattr(self, '_winding_context_source', 'none') or 'none'),
    )
    return profile

  def _update_winding_context(self) -> None:
    self._clear_winding_context()

    local_active = bool(getattr(self, '_winding_road_active', False))
    local_score = float(getattr(self, '_winding_road_score', 0.0) or 0.0)

    mapd_valid = bool(getattr(self, '_mapd_winding_valid', False))
    mapd_level = int(getattr(self, '_mapd_winding_level', 0) or 0)
    mapd_score = max(0.0, min(1.0, float(getattr(self, '_mapd_winding_score', 0) or 0) / 255.0))
    mapd_conf = max(0.0, min(1.0, float(getattr(self, '_mapd_winding_confidence', 0) or 0) / 255.0))
    mapd_active = bool(mapd_valid and mapd_level >= 3 and mapd_conf >= 0.35 and mapd_score >= 0.45)

    if local_active and mapd_active:
      self._winding_context_active = True
      self._winding_context_score = max(local_score, mapd_score)
      self._winding_context_level = max(3, mapd_level)
      self._winding_context_confidence = max(mapd_conf, min(1.0, 0.5 + 0.5 * local_score))
      self._winding_context_source = 'blended'
      return

    if mapd_active:
      self._winding_context_active = True
      self._winding_context_score = mapd_score
      self._winding_context_level = mapd_level
      self._winding_context_confidence = mapd_conf
      self._winding_context_source = 'mapd'
      return

    if local_active:
      self._winding_context_active = True
      self._winding_context_score = local_score
      self._winding_context_level = 0
      self._winding_context_confidence = min(1.0, 0.5 + 0.5 * local_score)
      self._winding_context_source = 'local'

  def _should_relax_strategic_map_candidate(self, candidate) -> bool:
    if candidate is None or candidate.cap_mps is None:
      return False
    if bool(getattr(self, '_road_geometry_valid', False)):
      return False
    if bool(getattr(self, '_mapd_winding_valid', False)):
      return False
    if not (bool(getattr(self, '_lane_change_active', False)) or bool(getattr(self, '_single_blinker_active', False))):
      return False

    try:
      anchor_dist_m = float(candidate.anchor_dist_m or 0.0)
      anchor_k = abs(float(candidate.anchor_curvature or 0.0))
    except Exception:
      return False
    if anchor_dist_m < float(AMBIGUOUS_RAW_MAP_ANCHOR_MIN_DIST_M):
      return False
    if anchor_k < float(AMBIGUOUS_RAW_MAP_ANCHOR_MIN_KAPPA):
      return False

    local_k = max(
      abs(float(getattr(self, '_dbg_k_model', 0.0) or 0.0)),
      abs(float(getattr(self, '_dbg_k_steer', 0.0) or 0.0)),
      abs(float(getattr(self, '_filtered_curvature', 0.0) or 0.0)),
    )
    return bool(anchor_k >= max(float(AMBIGUOUS_RAW_MAP_ANCHOR_MIN_KAPPA), local_k * float(AMBIGUOUS_RAW_MAP_CURVATURE_RATIO)))

  def _should_suppress_map_candidate_for_visible_mainline_counterevidence(self, candidate) -> bool:
    if candidate is None or candidate.cap_mps is None:
      return False
    if not bool(getattr(self, '_road_geometry_valid', False)):
      return False
    if bool(getattr(self, '_mapd_winding_valid', False)) or bool(getattr(self, '_winding_context_active', False)):
      return False
    if bool(getattr(self, '_lane_change_active', False)) or bool(getattr(self, '_single_blinker_active', False)):
      return False
    if not bool(getattr(self, '_curve_preview_valid', False)):
      return False
    if len(getattr(self, '_curve_preview_branch_stubs', []) or []) > 0:
      return False
    try:
      if float(getattr(self, '_curve_preview_distance_m', 0.0) or 0.0) > float(VISIBLE_MAINLINE_RELAX_CURVE_START_MAX_M):
        return False
    except Exception:
      return False
    try:
      if float(getattr(self, '_v_ego', 0.0) or 0.0) < float(HIGHWAY_MIN_MPS):
        return False
    except Exception:
      return False

    try:
      if int(getattr(self._occlusion_state, 'vision_status', VisionStatus.FULL_VISIBILITY)) != int(VisionStatus.FULL_VISIBILITY):
        return False
    except Exception:
      return False

    try:
      anchor_k = abs(float(candidate.anchor_curvature or 0.0))
      candidate_cap = float(candidate.cap_mps or 0.0)
      v_ego = float(getattr(self, '_v_ego', 0.0) or 0.0)
    except Exception:
      return False
    if anchor_k < float(VISIBLE_MAINLINE_RELAX_MIN_ANCHOR_KAPPA):
      return False

    local_k = max(
      abs(float(getattr(self, '_dbg_k_model', 0.0) or 0.0)),
      abs(float(getattr(self, '_dbg_k_steer', 0.0) or 0.0)),
      abs(float(getattr(self, '_filtered_curvature', 0.0) or 0.0)),
    )
    if anchor_k < max(float(VISIBLE_MAINLINE_RELAX_MIN_ANCHOR_KAPPA), local_k * float(VISIBLE_MAINLINE_RELAX_CURVATURE_RATIO)):
      return False

    try:
      local_curve_speed = float(curvature_to_speed(max(local_k, 1e-8)))
    except Exception:
      local_curve_speed = float(MAX_SPEED_DEFAULT)
    if local_curve_speed < candidate_cap + float(VISIBLE_MAINLINE_RELAX_MIN_LOCAL_SPEED_DELTA_MPS):
      return False

    map_lat_acc_now = anchor_k * v_ego * v_ego
    if map_lat_acc_now < float(VISIBLE_MAINLINE_RELAX_MIN_MAP_LATACC_NOW):
      return False
    local_lat_acc = max(
      abs(float(getattr(self, '_current_lat_acc', 0.0) or 0.0)),
      abs(float(getattr(self, '_max_pred_lat_acc', 0.0) or 0.0)),
    )
    if local_lat_acc > map_lat_acc_now * float(VISIBLE_MAINLINE_RELAX_MAX_LOCAL_LATACC_RATIO):
      return False

    return True

  def _update_curve_preview_from_map(self, *, gps_lat: float, gps_lon: float, pts: list[tuple[float, float, float]], i0: int,
                                     gps_bearing_deg: float | None = None) -> None:
    """Update HUD curve preview from mapd curvature samples.

    This is intentionally display-oriented: the HUD must not run its own curve math.
    """
    now = time.time()
    cache_raw = getattr(self, '_map_curv_cache_raw', None)
    last_raw = getattr(self, '_curve_preview_last_cache_raw', None)
    last_latlon = getattr(self, '_curve_preview_last_latlon', None)

    moved_far = True
    try:
      if last_latlon is not None:
        moved_far = _haversine_m(float(gps_lat), float(gps_lon), float(last_latlon[0]), float(last_latlon[1])) > 5.0
    except Exception:
      moved_far = True

    prev_tiles_by_id: dict[int, dict] = {}
    prev_tiles = getattr(self, '_curve_preview_tiles', None)
    if isinstance(prev_tiles, list):
      for tile in prev_tiles:
        if not isinstance(tile, dict):
          continue
        try:
          tile_id = int(tile.get('id', 0) or 0)
        except Exception:
          continue
        if tile_id != 0:
          prev_tiles_by_id[tile_id] = dict(tile)

    # Recompute at most 5 Hz unless map changed or ego moved materially.
    if (now - float(getattr(self, '_curve_preview_last_ts', 0.0) or 0.0)) < 0.2 and (cache_raw == last_raw) and (not moved_far):
      # Still refresh time-to-curve using latest speed.
      try:
        self._curve_preview_time_to_s = float(self._curve_preview_distance_m) / max(0.1, float(self._v_ego))
        refreshed_tiles: list[dict] = []
        for tile in self.curve_preview_tiles:
          refreshed_tiles.append({
            **tile,
            'time_to_s': float(tile['distance_m']) / max(0.1, float(self._v_ego)),
          })
        self._curve_preview_tiles = refreshed_tiles
      except Exception:
        pass
      return

    # Default to invalid; set valid only when we can build a sane preview.
    self._clear_curve_preview(reset_visible_mainline_counterevidence=False)

    PREVIEW_TIME_HORIZON_S = 10.0
    PREVIEW_MIN_M = 30.0
    PREVIEW_MAX_M = 350.0
    TILE_TIME_HORIZON_S = 20.0
    TILE_MIN_M = 60.0
    TILE_MAX_M = 450.0
    PREVIEW_SOURCE_MARGIN_M = 25.0
    PREVIEW_TRAILING_M = 8.0
    PREVIEW_RESAMPLE_STEP_M = 4.0
    SMOOTHING_PASSES = 2
    MAX_POINTS = 48
    KAPPA_MIN = 1.0e-3  # 1/m, ~1000 m radius (detect gentler curves)
    RUN = 2             # consecutive samples to start/end a curve
    TILE_MAX_COUNT = 6
    TILE_MERGE_GAP_M = 18.0
    TILE_LEAD_IN_MIN_M = 10.0
    TILE_LEAD_IN_MAX_M = 24.0
    TILE_LEAD_OUT_MIN_M = 10.0
    TILE_LEAD_OUT_MAX_M = 20.0
    TILE_SMOOTHING_PASSES = 4
    TILE_RESAMPLE_COUNT = 24

    try:
      i0 = int(max(0, min(len(pts) - 1, i0)))
    except Exception:
      i0 = 0
    base_idx = max(0, i0 - 1)
    preview_s_max_m = max(PREVIEW_MIN_M, min(float(self._v_ego) * PREVIEW_TIME_HORIZON_S, PREVIEW_MAX_M))
    tile_s_max_m = max(TILE_MIN_M, min(float(self._v_ego) * TILE_TIME_HORIZON_S, TILE_MAX_M))
    source_s_max_m = max(preview_s_max_m, tile_s_max_m) + PREVIEW_SOURCE_MARGIN_M

    w_lat: list[float] = [float(pts[base_idx][0])]
    w_lon: list[float] = [float(pts[base_idx][1])]
    w_k: list[float] = [max(0.0, float(pts[base_idx][2]))]
    s_pts: list[float] = [0.0]

    s = 0.0
    # Hard cap on iterations to keep this bounded even if points are dense.
    for j in range(base_idx, min(len(pts) - 1, base_idx + 160)):
      ds = _haversine_m(float(pts[j][0]), float(pts[j][1]), float(pts[j+1][0]), float(pts[j+1][1]))
      if not (ds > 0.05 and math.isfinite(ds)):
        continue
      s += float(ds)
      w_lat.append(float(pts[j+1][0]))
      w_lon.append(float(pts[j+1][1]))
      w_k.append(max(0.0, float(pts[j+1][2])))
      s_pts.append(float(s))
      if s >= source_s_max_m:
        break

    if len(w_lat) < 4:
      return

    # Convert to local EN (east, north) around the live ego pose rather than the
    # nearest matched map sample so the strip map aligns with the road ahead.
    en: list[tuple[float, float]] = []
    for la, lo in zip(w_lat, w_lon, strict=False):
      en.append(_xy_from_latlon_m(float(la), float(lo), float(gps_lat), float(gps_lon)))

    # Prefer live GPS bearing at speed; otherwise fall back to the local map tangent.
    basis = None
    if gps_bearing_deg is not None and float(self._v_ego) >= 2.0:
      basis = _bearing_deg_to_unit_en(float(gps_bearing_deg))
    if basis is None:
      for idx in range(1, len(en)):
        dx = float(en[idx][0] - en[idx - 1][0])
        dy = float(en[idx][1] - en[idx - 1][1])
        norm = math.hypot(dx, dy)
        if norm > 1.0 and math.isfinite(norm):
          basis = (float(dx / norm), float(dy / norm))
          break
    if basis is None:
      return
    fx, fy = basis
    lx, ly = -fy, fx

    # Rotate into ego-local (x forward, y left).
    fwd_left: list[tuple[float, float]] = []
    for (xe, yn) in en:
      x_fwd = float(xe) * fx + float(yn) * fy
      y_left = float(xe) * lx + float(yn) * ly
      fwd_left.append((x_fwd, y_left))

    # Estimate the along-track offset between the nearest map point and the ego pose.
    # This keeps distance-to-curve metadata counting down smoothly when GPS lies between
    # sparse map samples.
    s_zero_m = 0.0
    have_zero = False
    if fwd_left and fwd_left[0][0] >= 0.0:
      have_zero = True
      s_zero_m = float(s_pts[0])
    for idx in range(1, len(fwd_left)):
      x0 = float(fwd_left[idx - 1][0])
      x1 = float(fwd_left[idx][0])
      if x0 <= 0.0 <= x1 and abs(x1 - x0) > 1e-6:
        t = max(0.0, min(1.0, -x0 / (x1 - x0)))
        s_zero_m = float(s_pts[idx - 1] + t * (s_pts[idx] - s_pts[idx - 1]))
        have_zero = True
        break
    if not have_zero and max((pt[0] for pt in fwd_left), default=-1.0) < 0.0:
      return

    pts_out = _clip_polyline_x(fwd_left, -PREVIEW_TRAILING_M, preview_s_max_m)
    if len(pts_out) < 2:
      return
    pts_out = _densify_polyline(pts_out, PREVIEW_RESAMPLE_STEP_M)
    pts_out = _chaikin_smooth_polyline(pts_out, passes=SMOOTHING_PASSES)
    pts_out = _clip_polyline_x(pts_out, 0.0, preview_s_max_m)
    if len(pts_out) < 2:
      return
    target_points = max(18, min(MAX_POINTS, int(math.ceil(preview_s_max_m / PREVIEW_RESAMPLE_STEP_M)) + 1))
    pts_out = _resample_polyline(pts_out, target_points)
    if pts_out:
      pts_out[0] = (0.0, float(pts_out[0][1]))

    curve_regions: list[tuple[int, int, int]] = []
    search_idx = 0
    while search_idx < len(w_k):
      start_idx = None
      consec = 0
      for idx in range(search_idx, len(w_k)):
        if float(s_pts[idx]) - float(s_zero_m) > tile_s_max_m:
          search_idx = len(w_k)
          break
        if float(w_k[idx]) >= KAPPA_MIN:
          consec += 1
        else:
          consec = 0
        if consec >= RUN:
          start_idx = idx - (RUN - 1)
          break
      if start_idx is None:
        break

      end_idx = len(w_k) - 1
      consec_below = 0
      for idx in range(int(start_idx), len(w_k)):
        if float(s_pts[idx]) - float(s_zero_m) > tile_s_max_m:
          end_idx = max(int(start_idx), idx - 1)
          break
        if float(w_k[idx]) < KAPPA_MIN:
          consec_below += 1
        else:
          consec_below = 0
        if consec_below >= RUN:
          end_idx = max(int(start_idx), idx - RUN)
          break

      direction = _curve_direction_from_points(fwd_left[int(start_idx):int(end_idx) + 1])
      if curve_regions:
        prev_start, prev_end, prev_direction = curve_regions[-1]
        gap_m = float(s_pts[int(start_idx)]) - float(s_pts[int(prev_end)])
        if prev_direction != 0 and prev_direction == direction and gap_m <= TILE_MERGE_GAP_M:
          curve_regions[-1] = (prev_start, max(prev_end, int(end_idx)), prev_direction)
        else:
          curve_regions.append((int(start_idx), int(end_idx), direction))
      else:
        curve_regions.append((int(start_idx), int(end_idx), direction))

      search_idx = max(int(end_idx) + 1, int(start_idx) + RUN)

    if not curve_regions:
      # No curve detected — still publish a dense road-ahead polyline so the HUD can
      # render the same road frame continuously as the next bend comes into view.
      try:
        self._curve_preview_valid = True
        self._curve_preview_distance_m = 0.0
        self._curve_preview_time_to_s = 0.0
        self._curve_preview_kappa_max = 0.0
        self._curve_preview_direction = 0
        self._curve_preview_severity = 0
        self._curve_preview_points = [(float(x), float(y)) for (x, y) in pts_out]
        self._curve_preview_tiles = []
        self._curve_preview_last_ts = float(now)
        self._curve_preview_last_cache_raw = cache_raw
        self._curve_preview_last_latlon = (float(gps_lat), float(gps_lon))
      except Exception:
        self._clear_curve_preview()
      return

    preview_tiles: list[dict] = []
    matched_prev_ids: set[int] = set()
    preview_meta = None
    for start_idx, end_idx, direction in curve_regions:
      if len(preview_tiles) >= TILE_MAX_COUNT:
        break

      try:
        kappa_max = 0.0
        vs_min = float('inf')
        for idx in range(int(start_idx), int(end_idx) + 1):
          kappa = float(w_k[idx])
          kappa_max = max(kappa_max, kappa)
          vs_min = min(vs_min, float(self._curve_speed(kappa)))
        if not (kappa_max > 0.0 and math.isfinite(kappa_max)):
          kappa_max = 0.0
      except Exception:
        kappa_max = 0.0
        vs_min = float('inf')

      if not math.isfinite(vs_min):
        severity = 0
        vs_min = 0.0
      elif vs_min < 12.0:
        severity = 3
      elif vs_min < 20.0:
        severity = 2
      else:
        severity = 1

      curve_start_s = float(s_pts[int(start_idx)])
      curve_end_s = float(s_pts[int(end_idx)])
      curve_len_m = max(1.0, curve_end_s - curve_start_s)
      lead_in_m = max(TILE_LEAD_IN_MIN_M, min(0.35 * curve_len_m, TILE_LEAD_IN_MAX_M))
      lead_out_m = max(TILE_LEAD_OUT_MIN_M, min(0.25 * curve_len_m, TILE_LEAD_OUT_MAX_M))
      tile_s_start = max(float(s_zero_m), curve_start_s - lead_in_m)
      tile_s_end = min(float(s_pts[-1]), curve_end_s + lead_out_m)
      tile_src_pts = _slice_polyline_by_s(fwd_left, s_pts, tile_s_start, tile_s_end)
      tile_pts = _normalize_polyline_to_entry_frame(
        tile_src_pts,
        densify_step_m=3.0,
        smooth_passes=TILE_SMOOTHING_PASSES,
        resample_count=TILE_RESAMPLE_COUNT,
      )
      if len(tile_pts) < 2:
        continue

      distance_m = max(0.0, curve_start_s - float(s_zero_m))
      prev_tile = None
      prev_tile_score = float('inf')
      for prev_id, cand in prev_tiles_by_id.items():
        if prev_id in matched_prev_ids:
          continue
        try:
          cand_distance = float(cand.get('distance_m', 0.0) or 0.0)
          cand_direction = int(cand.get('direction', 0) or 0)
          cand_severity = int(cand.get('severity', 0) or 0)
          cand_kappa = float(cand.get('max_curvature', 0.0) or 0.0)
        except Exception:
          continue
        if cand_direction != int(direction):
          continue
        dist_delta = abs(cand_distance - float(distance_m))
        if dist_delta > 55.0:
          continue
        score = dist_delta + 6.0 * abs(cand_severity - int(severity)) + 700.0 * abs(cand_kappa - float(kappa_max))
        if score < prev_tile_score:
          prev_tile = cand
          prev_tile_score = score

      if isinstance(prev_tile, dict):
        try:
          matched_prev_ids.add(int(prev_tile.get('id', 0) or 0))
        except Exception:
          pass
        prev_points = prev_tile.get('points', [])
        if isinstance(prev_points, list) and len(prev_points) >= 2:
          tile_pts = [(float(pt[0]), float(pt[1])) for pt in prev_points]
        tile_id = int(prev_tile.get('id', 0) or 0)
      else:
        abs_start_idx = int(base_idx + int(start_idx))
        abs_end_idx = int(base_idx + int(end_idx))
        tile_id = int((((abs_start_idx + 1) & 0xFFFF) << 16) | ((abs_end_idx + 1) & 0xFFFF))

      tile = {
        'id': tile_id,
        'distance_m': float(distance_m),
        'time_to_s': float(distance_m) / max(0.1, float(self._v_ego)),
        'direction': int(direction),
        'severity': int(severity),
        'max_curvature': float(kappa_max),
        'advisory_speed_mps': float(vs_min),
        'points': [(float(x), float(y)) for (x, y) in tile_pts],
      }
      preview_tiles.append(tile)

      if preview_meta is None:
        preview_meta = {
          'distance_m': float(distance_m),
          'kappa_max': float(kappa_max),
          'direction': int(direction),
          'severity': int(severity),
        }

    # Publish preview fields (used by HUD only).
    try:
      self._curve_preview_valid = True
      self._curve_preview_distance_m = float(preview_meta['distance_m']) if preview_meta is not None else 0.0
      self._curve_preview_time_to_s = float(self._curve_preview_distance_m) / max(0.1, float(self._v_ego))
      self._curve_preview_kappa_max = float(preview_meta['kappa_max']) if preview_meta is not None else 0.0
      self._curve_preview_direction = int(preview_meta['direction']) if preview_meta is not None else 0
      self._curve_preview_severity = int(preview_meta['severity']) if preview_meta is not None else 0
      self._curve_preview_points = [(float(x), float(y)) for (x, y) in pts_out]
      self._curve_preview_tiles = preview_tiles
      self._curve_preview_last_ts = float(now)
      self._curve_preview_last_cache_raw = cache_raw
      self._curve_preview_last_latlon = (float(gps_lat), float(gps_lon))
    except Exception:
      self._clear_curve_preview()

  def _update_curve_preview_branch_stubs(self, sm, *, gps_lat: float, gps_lon: float,
                                         gps_bearing_deg: float | None = None) -> None:
    self._curve_preview_branch_stubs = []
    if not bool(getattr(self, '_curve_preview_valid', False)):
      return

    base_pts = getattr(self, '_curve_preview_points', None)
    if not isinstance(base_pts, list) or len(base_pts) < 2:
      return

    map_data = _sm_get_optional(sm, 'liveMapDataSP')
    if map_data is None or not bool(getattr(map_data, 'roadGeometryValid', False)):
      return

    current_seg = getattr(map_data, 'currentRoadSegment', None)
    current_coords = _extract_centerline_coords(current_seg)
    if len(current_coords) < 2:
      return

    ego_proj = _project_latlon_to_centerline(float(gps_lat), float(gps_lon), current_coords)
    if ego_proj is None:
      return

    basis = None
    if gps_bearing_deg is not None and float(self._v_ego) >= 1.0:
      basis = _bearing_deg_to_unit_en(float(gps_bearing_deg))
    if basis is None:
      basis = ego_proj.get('tangent_en', None)
    if basis is None:
      return
    fx, fy = basis
    lx, ly = -fy, fx

    lookahead_m = max(30.0, min(float(self._v_ego) * 10.0, 350.0))
    conn_max_dist_m = 14.0
    conn_back_margin_m = 4.0
    conn_ahead_margin_m = 24.0
    stub_length_m = 34.0
    min_branch_angle_deg = 28.0
    max_branch_angle_deg = 160.0

    car_state = _sm_get_optional(sm, 'carState')
    left_blinker = bool(getattr(car_state, 'leftBlinker', False))
    right_blinker = bool(getattr(car_state, 'rightBlinker', False))
    desired_side = 0
    if left_blinker != right_blinker:
      desired_side = 1 if left_blinker else -1

    current_way_id = int(getattr(current_seg, 'wayId', 0) or 0)
    current_level = int(getattr(current_seg, 'levelSeparation', 0) or 0)
    candidates: list[dict] = []

    nearby_segments = getattr(map_data, 'nearbyRoadSegments', [])
    for seg in nearby_segments:
      try:
        way_id = int(getattr(seg, 'wayId', 0) or 0)
      except Exception:
        way_id = 0
      if way_id == current_way_id:
        continue

      try:
        level = int(getattr(seg, 'levelSeparation', 0) or 0)
      except Exception:
        level = current_level
      if level != current_level:
        continue

      seg_coords = _extract_centerline_coords(seg)
      if len(seg_coords) < 2:
        continue

      best_endpoint: dict | None = None
      for endpoint_idx in (0, len(seg_coords) - 1):
        end_lat = float(seg_coords[endpoint_idx][0])
        end_lon = float(seg_coords[endpoint_idx][1])
        conn = _project_latlon_to_centerline(end_lat, end_lon, current_coords)
        if conn is None or float(conn['d']) > conn_max_dist_m:
          continue

        s_ahead = float(conn['s']) - float(ego_proj['s'])
        if s_ahead < -conn_back_margin_m or s_ahead > lookahead_m + conn_ahead_margin_m:
          continue

        away_idx = 1 if endpoint_idx == 0 else len(seg_coords) - 2
        end_x, end_y = _xy_from_latlon_m(end_lat, end_lon, float(gps_lat), float(gps_lon))
        away_x, away_y = _xy_from_latlon_m(float(seg_coords[away_idx][0]), float(seg_coords[away_idx][1]),
                                           float(gps_lat), float(gps_lon))
        branch_tangent = _normalize_en_vec(away_x - end_x, away_y - end_y)
        if branch_tangent is None:
          continue

        main_tangent = conn.get('tangent_en', None)
        if main_tangent is None:
          continue
        dot = float(main_tangent[0]) * float(branch_tangent[0]) + float(main_tangent[1]) * float(branch_tangent[1])
        cross = float(main_tangent[0]) * float(branch_tangent[1]) - float(main_tangent[1]) * float(branch_tangent[0])
        angle_deg = abs(math.degrees(math.atan2(cross, dot)))
        if angle_deg < min_branch_angle_deg or angle_deg > max_branch_angle_deg:
          continue

        side = 1 if cross > 0.0 else -1
        endpoint_info = {
          'endpoint_idx': int(endpoint_idx),
          's_ahead': float(s_ahead),
          'angle_deg': float(angle_deg),
          'side': int(side),
          'conn': conn,
        }
        if best_endpoint is None:
          best_endpoint = endpoint_info
          continue

        prev_score = (abs(float(best_endpoint['s_ahead'])), -float(best_endpoint['angle_deg']))
        curr_score = (abs(float(endpoint_info['s_ahead'])), -float(endpoint_info['angle_deg']))
        if curr_score < prev_score:
          best_endpoint = endpoint_info

      if best_endpoint is None:
        continue

      endpoint_idx = int(best_endpoint['endpoint_idx'])
      if endpoint_idx == 0:
        branch_slice = seg_coords
      else:
        branch_slice = list(reversed(seg_coords))

      branch_latlon: list[tuple[float, float]] = [
        (float(best_endpoint['conn']['proj_lat']), float(best_endpoint['conn']['proj_lon'])),
      ]
      branch_accum_m = 0.0
      for idx, pt in enumerate(branch_slice):
        lat = float(pt[0])
        lon = float(pt[1])
        if idx > 0:
          branch_accum_m += _haversine_m(float(branch_slice[idx - 1][0]), float(branch_slice[idx - 1][1]), lat, lon)
        branch_latlon.append((lat, lon))
        if branch_accum_m >= stub_length_m:
          break

      stub_pts: list[tuple[float, float]] = []
      for lat, lon in branch_latlon:
        x_east, y_north = _xy_from_latlon_m(float(lat), float(lon), float(gps_lat), float(gps_lon))
        x_fwd = x_east * fx + y_north * fy
        y_left = x_east * lx + y_north * ly
        stub_pts.append((float(x_fwd), float(y_left)))

      stub_pts = _densify_polyline(stub_pts, 5.0)
      stub_pts = _chaikin_smooth_polyline(stub_pts, passes=1)
      stub_pts = _clip_polyline_x(stub_pts, -2.0, lookahead_m + 8.0)
      if len(stub_pts) < 2:
        continue

      if max((float(pt[0]) for pt in stub_pts), default=-1.0) < 0.0:
        continue

      resample_n = max(4, min(10, len(stub_pts)))
      stub_pts = _resample_polyline(stub_pts, resample_n)

      candidates.append({
        'way_id': int(way_id),
        'side': int(best_endpoint['side']),
        's_ahead': float(best_endpoint['s_ahead']),
        'angle_deg': float(best_endpoint['angle_deg']),
        'highlighted': False,
        'points': [(float(x), float(y)) for x, y in stub_pts],
      })

    if not candidates:
      return

    selected: list[dict] = []
    if desired_side != 0:
      same_side = [cand for cand in candidates if int(cand['side']) == desired_side]
      if same_side:
        same_side.sort(key=lambda cand: (abs(float(cand['s_ahead'])), -float(cand['angle_deg'])))
        chosen = dict(same_side[0])
        chosen['highlighted'] = True
        selected = [chosen]

    if not selected:
      per_side: dict[int, dict] = {}
      for cand in candidates:
        side = int(cand['side'])
        prev = per_side.get(side)
        if prev is None or (abs(float(cand['s_ahead'])), -float(cand['angle_deg'])) < (abs(float(prev['s_ahead'])), -float(prev['angle_deg'])):
          per_side[side] = cand
      selected = [dict(cand) for _side, cand in sorted(per_side.items(), key=lambda item: item[1]['s_ahead'])]

    self._curve_preview_branch_stubs = selected[:2]

  def _map_tail_cap(self, sm=None) -> tuple[float | None, float, float]:
    """
    Compute a comfort-reachable cap on current speed from map curvature tail.

    Returns (v_cap_mps|None, start_distance_m, coverage_frac)
    """
    self._map_tail_candidate = None
    self._map_tail_advisory_cap = None
    self._map_tail_strategic_cap = None
    self._map_tail_anchor_dist_m = 0.0
    self._map_tail_anchor_k = 0.0
    self._map_tail_anchor_vsafe = 0.0
    self._map_tail_anchor_index = -1
    self._map_tail_compute_reason = "unknown"
    self._map_profile_source = "none"
    self._clear_winding_behavior_profile()
    self._clear_winding_road_context()
    gps_pose = self._get_last_gps_pose()
    if gps_pose is None:
      self._map_tail_compute_reason = "no_gps"
      self._clear_curve_preview()
      return (None, 0.0, 0.0)
    map_data = _sm_get_optional(sm, 'liveMapDataSP') if sm is not None else None
    if map_data is not None and hasattr(map_data, 'roadGeometryValid'):
      try:
        if not bool(getattr(map_data, 'roadGeometryValid', False)):
          self._map_tail_compute_reason = "road_geometry_invalid"
          self._map_curv_cache = []
          self._map_curv_cache_raw = None
          self._clear_curve_preview()
          return (None, 0.0, 0.0)
      except Exception:
        pass
    lat0, lon0, bearing_deg = gps_pose
    whole_profile = self._load_map_whole_curve_profile(gps_pose)
    use_whole_profile = whole_profile is not None
    if use_whole_profile:
      assert whole_profile is not None
      profile_points = list(whole_profile.points)
      # Preserve the signed route output for fingerprinting/diagnostics, but
      # speed physics always uses curvature magnitude for either turn direction.
      pts = [(point.latitude, point.longitude, abs(point.curvature)) for point in profile_points]
      self._map_profile_source = MAP_WHOLE_CURVE_ESTIMATOR_VERSION
    else:
      profile_points = []
      pts = self._load_map_curvatures()
      self._map_profile_source = "legacy_map_curvatures" if len(pts) >= 3 else "none"
    if len(pts) < 3:
      if self._map_whole_curve_reason.startswith("rejected_"):
        self._map_tail_compute_reason = f"whole_curve_{self._map_whole_curve_reason}"
      else:
        self._map_tail_compute_reason = "no_map_curvatures"
      self._clear_curve_preview()
      return (None, 0.0, 0.0)

    # Find nearest index to ego.
    try:
      dists = [_haversine_m(lat0, lon0, p[0], p[1]) for p in pts]
      i0 = int(np.argmin(dists))
      nearest_distance = float(dists[i0])
    except Exception:
      i0 = 0
      nearest_distance = float("inf") if use_whole_profile else 0.0

    # GPS proximity is intentionally checked here, not in the cached parser:
    # this is the one route scan already required to choose i0 each tick. A
    # cached profile therefore cannot remain active after ego leaves its route.
    if use_whole_profile and (
        nearest_distance > MAP_WHOLE_CURVE_PROFILE_MAX_EGO_DISTANCE_M or
        i0 > len(pts) - 3):
      if nearest_distance > MAP_WHOLE_CURVE_PROFILE_MAX_EGO_DISTANCE_M:
        self._map_whole_curve_reason = "rejected_route_not_near_ego"
      else:
        self._map_whole_curve_reason = "rejected_insufficient_forward_context"
      use_whole_profile = False
      profile_points = []
      pts = self._load_map_curvatures()
      self._map_profile_source = "legacy_map_curvatures" if len(pts) >= 3 else "none"
      if len(pts) < 3:
        self._map_tail_compute_reason = f"whole_curve_{self._map_whole_curve_reason}"
        self._clear_curve_preview()
        return (None, 0.0, 0.0)
      try:
        dists = [_haversine_m(lat0, lon0, p[0], p[1]) for p in pts]
        i0 = int(np.argmin(dists))
      except Exception:
        i0 = 0

    if use_whole_profile:
      # Consume the estimator's verified cumulative route distances directly;
      # this retains strict ~5 m sampling and predecessor context through a
      # current-way rollover.
      base_distance = float(profile_points[i0].distance_m)
      s_list = [float(point.distance_m - base_distance) for point in profile_points[i0 + 1:]]
      k_list = [abs(float(point.curvature)) for point in profile_points[i0 + 1:]]
    else:
      # Legacy MapCurvatures has no cumulative-distance field.
      s_list = [0.0]
      for i in range(i0, len(pts)-1):
        s_list.append(s_list[-1] + _haversine_m(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]))
      if len(s_list) > 0:
        s_list = s_list[1:]
      k_list = [max(0.0, pts[j][2]) for j in range(i0+1, len(pts))]
    if not s_list or not k_list:
      self._map_tail_compute_reason = "insufficient_map_points"
      self._clear_curve_preview()
      return (None, 0.0, 0.0)

    # Update HUD preview from the same map lookahead inputs VTSC already uses.
    try:
      preview_span = start_span(SPAN_PREVIEW_FROM_MAP)
      try:
        self._update_curve_preview_from_map(
          gps_lat=float(lat0),
          gps_lon=float(lon0),
          gps_bearing_deg=None if bearing_deg is None else float(bearing_deg),
          pts=pts,
          i0=i0,
        )
      finally:
        end_span(preview_span)
    except Exception:
      # Never let preview failures affect longitudinal behavior.
      self._clear_curve_preview()
    else:
      try:
        if sm is not None:
          branch_stub_span = start_span(SPAN_PREVIEW_BRANCH_STUBS)
          try:
            self._update_curve_preview_branch_stubs(
              sm,
              gps_lat=float(lat0),
              gps_lon=float(lon0),
              gps_bearing_deg=None if bearing_deg is None else float(bearing_deg),
            )
          finally:
            end_span(branch_stub_span)
      except Exception:
        self._curve_preview_branch_stubs = []

    # Keep the legacy cap horizon unchanged; the versioned whole-curve profile
    # carries the full production 1.2 km route context.
    S_MAX = MAP_WHOLE_CURVE_PROFILE_HORIZON_M if use_whole_profile else 800.0
    L = len(s_list)
    cut = L
    for idx, s in enumerate(s_list):
      if s >= S_MAX:
        cut = idx+1
        break
    s_list = s_list[:cut]
    k_list = k_list[:cut]
    abs_indices = list(range(i0 + 1, len(pts)))[:cut]

    # Whole-curve v3 publishes an estimator-aligned physics speed. Consume it
    # only when its tune hash matches the live sigmoid, then apply the cheap
    # runtime Q/bias layers. A live tune edit fails safely back to conversion
    # until mapd republishes the route profile with the new hash.
    baked = None if use_whole_profile else self._load_map_pre_curve_speeds()
    if use_whole_profile:
      runtime_sigmoid_hash = _compute_runtime_sigmoid_hash()
      if whole_profile is not None and whole_profile.sigmoid_hash == runtime_sigmoid_hash:
        speed_cache_key = self._whole_curve_speed_cache_signature(whole_profile, runtime_sigmoid_hash)
        if (self._map_whole_curve_speed_cache_key != speed_cache_key or
            len(self._map_whole_curve_speed_cache) != len(whole_profile.points)):
          self._map_whole_curve_speed_cache = [
            self._baked_vsafe_with_runtime_multipliers(point.base_safe_speed_mps, abs(point.curvature))
            for point in whole_profile.points
          ]
          self._map_whole_curve_speed_cache_key = speed_cache_key
        vsafe = self._map_whole_curve_speed_cache[i0 + 1:i0 + 1 + cut]
      else:
        self._map_whole_curve_speed_cache_key = None
        self._map_whole_curve_speed_cache = []
        vsafe = [self._whole_curve_speed(k) for k in k_list]
        profile_hash = "" if whole_profile is None else whole_profile.sigmoid_hash
        mismatch_key = (profile_hash, runtime_sigmoid_hash)
        if getattr(self, '_last_logged_whole_curve_sigmoid_mismatch', None) != mismatch_key:
          cloudlog.info(
            f"vtsc: whole-curve profile sigmoid hash {profile_hash!r} != runtime {runtime_sigmoid_hash!r}; "
            "falling back to live curvature_to_speed until mapd republishes"
          )
          self._last_logged_whole_curve_sigmoid_mismatch = mismatch_key
    elif baked is not None and len(baked) == len(k_list):
      try:
        # Skip baked path when low-speed calibration is non-trivially active —
        # the bake captures scale=1.0 and the calibration only matters at
        # sub-40 mph speeds where the live path is the source of truth.
        max_scale_dev = 0.0
        for k_check in k_list:
          dev = abs(self._low_speed_calibration_scale(k_check) - 1.0)
          if dev > max_scale_dev:
            max_scale_dev = dev
            if dev > 1e-3:
              break
        if max_scale_dev <= 1e-3:
          vsafe = [ self._baked_vsafe_with_runtime_multipliers(b, k) for b, k in zip(baked, k_list) ]
        else:
          vsafe = [ self._curve_speed(k) for k in k_list ]
      except Exception:
        vsafe = [ self._curve_speed(k) for k in k_list ]
    else:
      vsafe = [ self._curve_speed(k) for k in k_list ]
    self._set_winding_road_context(classify_winding_road_context(
      s_list=s_list,
      vsafe_list=vsafe,
      abs_indices=abs_indices,
    ))
    self._update_winding_context()
    winding_profile = self._refresh_winding_behavior_profile()

    try:
      vs = getattr(self._occlusion_state, 'vision_status', VisionStatus.FULL_VISIBILITY)
      severe_vision = bool(int(vs) >= int(VisionStatus.SEVERE_OCCLUSION))
    except Exception:
      vs = VisionStatus.FULL_VISIBILITY
      severe_vision = False
    partial_vision = bool(vs == VisionStatus.PARTIAL_OCCLUSION)
    vis_margin = float(getattr(self, '_vis_margin_m', 10.0))
    vis_horizon_s = float(getattr(self, '_vis_horizon_s', 1.4))
    try:
      conf_now = float(getattr(self, '_last_vision_confidence', getattr(self._occlusion_state, 'smoothed_confidence', 1.0)))
    except Exception:
      conf_now = float(getattr(self._occlusion_state, 'smoothed_confidence', 1.0))

    advisory_candidate = compute_map_cap_candidate(
      mode=MAP_STRATEGY_ADVISORY,
      s_list=s_list,
      k_list=k_list,
      vsafe_list=vsafe,
      abs_indices=abs_indices,
      v_ego=float(self._v_ego),
      v_cruise=float(self._v_cruise_setpoint),
      vis_horizon_s=vis_horizon_s,
      vis_margin_m=vis_margin,
      severe_vision=severe_vision,
      partial_vision=partial_vision,
      vision_confidence=conf_now,
      conf_lo=float(CONFIDENCE_EXIT_TO_PARTIAL),
      conf_hi=float(CONFIDENCE_EXIT_TO_FULL),
      max_decel=float(getattr(self, '_max_decel', 3.5)),
      horizon_limit_m=S_MAX,
    )
    strategic_candidate = compute_map_cap_candidate(
      mode=MAP_STRATEGY_STRATEGIC,
      s_list=s_list,
      k_list=k_list,
      vsafe_list=vsafe,
      abs_indices=abs_indices,
      v_ego=float(self._v_ego),
      v_cruise=float(self._v_cruise_setpoint),
      vis_horizon_s=vis_horizon_s,
      vis_margin_m=vis_margin,
      severe_vision=severe_vision,
      partial_vision=partial_vision,
      vision_confidence=conf_now,
      conf_lo=float(CONFIDENCE_EXIT_TO_PARTIAL),
      conf_hi=float(CONFIDENCE_EXIT_TO_FULL),
      max_decel=float(getattr(self, '_max_decel', 3.5)),
      horizon_limit_m=S_MAX,
      response_model=getattr(self, '_longitudinal_response_model', None),
      fixed_lead_time_s=float(getattr(self, '_fixed_lead_time_s', 0.0)),
      curve_phase_offset_s=float(getattr(self, '_curve_phase_offset_s', 0.0)),
      overshoot_phase_offset_s=float(getattr(self, '_overshoot_phase_offset_s', 0.0)),
      reference_speed_mps=float(getattr(self, '_dbg_target_raw', self._v_ego) or self._v_ego),
      winding_profile=winding_profile,
      precomputed_route_profile=use_whole_profile,
    )

    self._map_tail_advisory_cap = advisory_candidate.cap_mps
    self._map_tail_strategic_cap = strategic_candidate.cap_mps

    strategy_mode = normalize_map_strategy(getattr(self, '_map_strategy_mode', DEFAULT_MAP_STRATEGY))
    candidate = strategic_candidate if strategy_mode == MAP_STRATEGY_STRATEGIC else advisory_candidate
    relaxed_for_ambiguity = False
    suppressed_for_counterevidence = False
    self._visible_mainline_counterevidence_since = 0.0 if strategy_mode != MAP_STRATEGY_STRATEGIC else float(getattr(self, '_visible_mainline_counterevidence_since', 0.0) or 0.0)
    if strategy_mode == MAP_STRATEGY_STRATEGIC and self._should_relax_strategic_map_candidate(strategic_candidate):
      candidate = advisory_candidate
      relaxed_for_ambiguity = True
      self._visible_mainline_counterevidence_since = 0.0
    elif strategy_mode == MAP_STRATEGY_STRATEGIC:
      now_s = float(time.time())
      if self._should_suppress_map_candidate_for_visible_mainline_counterevidence(strategic_candidate):
        if self._visible_mainline_counterevidence_since <= 0.0:
          self._visible_mainline_counterevidence_since = now_s
        if (now_s - self._visible_mainline_counterevidence_since) >= float(VISIBLE_MAINLINE_RELAX_DWELL_S):
          candidate = MapCapCandidate(
            mode=MAP_STRATEGY_STRATEGIC,
            cap_mps=None,
            start_m=float(strategic_candidate.start_m),
            coverage=float(strategic_candidate.coverage),
            reason="visible_mainline_counterevidence",
            anchor_dist_m=strategic_candidate.anchor_dist_m,
            anchor_vsafe_mps=strategic_candidate.anchor_vsafe_mps,
            anchor_curvature=strategic_candidate.anchor_curvature,
            anchor_index=strategic_candidate.anchor_index,
          )
          suppressed_for_counterevidence = True
      else:
        self._visible_mainline_counterevidence_since = 0.0
    self._map_tail_candidate = candidate
    self._map_tail_anchor_dist_m = float(candidate.anchor_dist_m or 0.0)
    self._map_tail_anchor_k = float(candidate.anchor_curvature or 0.0)
    self._map_tail_anchor_vsafe = float(candidate.anchor_vsafe_mps or 0.0)
    self._map_tail_anchor_index = int(candidate.anchor_index) if candidate.anchor_index is not None else -1
    if relaxed_for_ambiguity:
      self._map_tail_compute_reason = "lane_change_map_ambiguity"
    elif suppressed_for_counterevidence:
      self._map_tail_compute_reason = "visible_mainline_counterevidence"
    else:
      self._map_tail_compute_reason = str(candidate.reason or "unknown")
    if candidate.cap_mps is None:
      return (None, float(candidate.start_m), float(candidate.coverage))
    return (float(candidate.cap_mps), float(candidate.start_m), float(candidate.coverage))

  def update(self, sm, enabled, v_ego, a_ego, v_cruise_setpoint, v_cruise_cluster_setpoint=None):
    self._op_enabled = enabled
    # Be defensive about SM shape in offline/testing environments
    try:
      cs = sm['carState']
    except Exception:
      try:
        cs = getattr(sm, 'carState', None)
        if cs is None:
          data = getattr(sm, '_data', None)
          if isinstance(data, dict):
            cs = data.get('carState', None)
      except Exception:
        cs = None
    self._gas_pressed = bool(getattr(cs, 'gasPressed', False))
    try:
      self._steering_angle_deg = float(getattr(cs, 'steeringAngleDeg', 0.0))
    except Exception:
      self._steering_angle_deg = 0.0
    left_blinker = bool(getattr(cs, 'leftBlinker', False))
    right_blinker = bool(getattr(cs, 'rightBlinker', False))
    self._single_blinker_active = bool(left_blinker != right_blinker)
    self._lane_change_active = False
    self._v_ego = v_ego
    self._a_ego = a_ego
    # Use cluster speed as source of truth if available, otherwise fall back to v_cruise
    prev_limit = getattr(self, '_v_cruise_setpoint', 0.0)
    self._v_cruise_setpoint = v_cruise_cluster_setpoint if v_cruise_cluster_setpoint is not None else v_cruise_setpoint
    try:
      if self._v_cruise_setpoint < prev_limit - 0.2:
        self._limit_step_until = time.time() + 1.0
        self._suppress_raise_due_to_limit = True
    except Exception:
      pass

    # Initialize advanced controller state on first run or when speed changes significantly
    if (self._prev_target_speed == 0.0 or
        abs(self._prev_target_speed - v_ego) > 5.0):
      self._prev_target_speed = v_ego
      self._current_accel = a_ego

    self._update_params()
    self._update_mapd_winding_context(sm)
    self._update_calculations(sm)
    self._state_transition()
    self._update_solution(sm)
    self._update_winding_context()
    # Emit compact debug snapshot if enabled and rate allows
    try:
      now_s = float(getattr(time, 'monotonic', time.time)())
    except Exception:
      now_s = time.time()
    if self._should_emit_debug(now_s):
      try:
        snap = self.snapshot_debug_state()
        if snap:
          cloudlog.debug("VTSCDBG %s", json.dumps(snap, separators=(',', ':')))
          if bool(getattr(self, '_dbg_write_file', False)):
            self._append_snapshot_to_file(snap, now_s)
      except Exception:
        pass
