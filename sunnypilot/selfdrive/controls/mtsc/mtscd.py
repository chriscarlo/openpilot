#!/usr/bin/env python3
"""
Map Turn Speed Controller Daemon (mtscd)

Publishes strategic, map-derived turn speed recommendations to mapTurnSpeedControlSP.
Initial skeleton: publishes unavailable=false recommendations with diagnostics only,
gated by MTSCEnabled and onroad status via process manager.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cereal.messaging as messaging
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper, Priority, config_realtime_process
import os
from openpilot.system.statsd import statlog
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.navd.helpers import Coordinate, minimum_distance, coordinate_from_param
try:
  # Reuse physics mapping from VTSC
  from openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed
except Exception:
  curvature_to_speed = None  # type: ignore


@dataclass
class Inputs:
  v_ego: float = 0.0
  onroad: bool = False
  map_valid: bool = False
  gps_fresh: bool = False
  vis_horizon_m: float = 0.0
  matched_way_id: int = 0
  road_class: int = 7  # unclassified default
  level: int = 0
  heading_err_deg: float = 0.0
  d_center_m: float = math.inf


def _normalize_angle_deg(a: float) -> float:
  # normalize to [-180, 180]
  a = (a + 180.0) % 360.0 - 180.0
  return a


def _heading_error_deg(ego_deg: float, road_deg: float) -> float:
  return abs(_normalize_angle_deg(ego_deg - road_deg))


def read_inputs(sm: messaging.SubMaster, gps_service: str) -> Inputs:
  inp = Inputs()
  try:
    cs = sm['carState']
    sds = sm['selfdriveState']
    mapd = sm['liveMapDataSP']
    gps = sm[gps_service]
    inp.v_ego = float(cs.vEgo)
    inp.onroad = bool(sds.enabled)
    inp.map_valid = bool(getattr(mapd, 'roadGeometryValid', False))
    # freshness: consider GPS fresh if we got an update this cycle or the message is very recent
    inp.gps_fresh = bool(sm.updated[gps_service] or (time.monotonic() - sm.logMonoTime[gps_service] / 1e9) < 0.5)
    if getattr(mapd, 'currentRoadSegment', None) is not None:
      seg = mapd.currentRoadSegment
      inp.matched_way_id = int(getattr(seg, 'wayId', 0))
      rc = getattr(seg, 'roadClass', 7)
      inp.road_class = rc if isinstance(rc, int) else 7
      inp.level = int(getattr(seg, 'levelSeparation', 0))
      try:
        ego_bearing = float(getattr(gps, 'bearingDeg', 0.0))
      except Exception:
        ego_bearing = 0.0
      road_dir = float(getattr(seg, 'roadDirection', 0.0))
      inp.heading_err_deg = _heading_error_deg(ego_bearing, road_dir)
      # Distance to centerline if GPS position available
      try:
        lat = float(gps.latitude)
        lon = float(gps.longitude)
        ego = Coordinate(lat, lon)
        pts = getattr(seg, 'centerline', [])
        dmin = math.inf
        for i in range(max(0, len(pts) - 1)):
          a = Coordinate(float(pts[i].latitude), float(pts[i].longitude))
          b = Coordinate(float(pts[i+1].latitude), float(pts[i+1].longitude))
          dmin = min(dmin, minimum_distance(a, b, ego))
        inp.d_center_m = float(dmin)
      except Exception:
        inp.d_center_m = math.inf
  except Exception:
    cloudlog.exception('mtscd: failed to read inputs')
  return inp


def _class_score(road_class: int) -> float:
  # 0: motorway, 1: trunk, 2: primary, 3: secondary, 4: tertiary, 5: residential, 6: service, 7: unclassified
  if road_class in (0, 1):
    return 1.0
  if road_class == 2:
    return 0.7
  if road_class in (3, 4):
    return 0.5
  return 0.3


def _level_score(level_sep: int) -> float:
  # Prefer ground level; degrade slightly for over/under unless later continuity confirms
  if level_sep == 0:
    return 1.0
  return 0.8


def compute_confidence(di: Inputs) -> float:
  # Distance-to-centerline score: <=3 m -> 1.0, >=20 m -> 0.0
  if math.isfinite(di.d_center_m):
    if di.d_center_m <= 3.0:
      s_dist = 1.0
    elif di.d_center_m >= 20.0:
      s_dist = 0.0
    else:
      s_dist = max(0.0, min(1.0, 1.0 - (di.d_center_m - 3.0) / (20.0 - 3.0)))
  else:
    s_dist = 0.0

  # Heading alignment score: <=4° -> 1.0, >=25° -> 0.0
  if di.heading_err_deg <= 4.0:
    s_head = 1.0
  elif di.heading_err_deg >= 25.0:
    s_head = 0.0
  else:
    s_head = max(0.0, min(1.0, 1.0 - (di.heading_err_deg - 4.0) / (25.0 - 4.0)))

  s_class = _class_score(di.road_class)
  s_level = _level_score(di.level)

  # Weighted average
  w_dist, w_head, w_class, w_level = 0.35, 0.35, 0.15, 0.15
  conf = (w_dist * s_dist + w_head * s_head + w_class * s_class + w_level * s_level)
  return float(max(0.0, min(1.0, conf)))


def publish_unavailable(pm: messaging.PubMaster, vis_horizon_m: float, diag: Inputs) -> None:
  msg = messaging.new_message('mapTurnSpeedControlSP')
  msg.valid = True
  out = msg.mapTurnSpeedControlSP
  out.timeStamp = int(time.monotonic() * 1e9)
  out.available = False
  out.confidence = compute_confidence(diag)
  out.targetSpeedMps = 0.0
  out.startDistanceM = 0.0
  out.horizonCoverage = getattr(diag, 'horizon_coverage', 0.0)
  out.minSpeedMps = getattr(diag, 'min_speed_mps', 0.0)
  out.minSpeedAtDistanceM = getattr(diag, 'min_speed_at_m', 0.0)
  out.matchedWayId = int(diag.matched_way_id)
  out.roadClass = int(diag.road_class)
  out.levelSeparation = int(diag.level)
  out.headingErrorDeg = float(diag.heading_err_deg)
  out.distanceToCenterlineM = float(diag.d_center_m if math.isfinite(diag.d_center_m) else 0.0)
  out.visHorizonM = float(max(0.0, vis_horizon_m))
  # Optional debug vectors if present on diag
  # Optional debug vectors if explicitly enabled via param
  try:
    if Params().get_int('MTSCLogDetail') > 0:
      if hasattr(diag, 'distances_m') and isinstance(diag.distances_m, list):
        out.distancesM = [float(x) for x in diag.distances_m[:30]]
      if hasattr(diag, 'kappas_per_m') and isinstance(diag.kappas_per_m, list):
        out.kappasPerM = [float(x) for x in diag.kappas_per_m[:30]]
      if hasattr(diag, 'vsafe_mps') and isinstance(diag.vsafe_mps, list):
        out.vSafeMps = [float(x) for x in diag.vsafe_mps[:30]]
  except Exception:
    pass
  try:
    pm.send('mapTurnSpeedControlSP', msg)
  except Exception:
    # Never crash publisher on transport errors
    pass


def publish_recommendation(pm: messaging.PubMaster,
                           vis_horizon_m: float,
                           diag: Inputs,
                           target_speed_mps: float,
                           start_distance_m: float,
                           confidence: float) -> None:
  msg = messaging.new_message('mapTurnSpeedControlSP')
  msg.valid = True
  out = msg.mapTurnSpeedControlSP
  out.timeStamp = int(time.monotonic() * 1e9)
  out.available = True
  out.confidence = float(max(0.0, min(1.0, confidence)))
  out.targetSpeedMps = float(max(0.0, target_speed_mps))
  out.startDistanceM = float(max(0.0, start_distance_m))
  out.horizonCoverage = getattr(diag, 'horizon_coverage', 0.0)
  out.minSpeedMps = getattr(diag, 'min_speed_mps', 0.0)
  out.minSpeedAtDistanceM = getattr(diag, 'min_speed_at_m', 0.0)
  out.matchedWayId = int(diag.matched_way_id)
  out.roadClass = int(diag.road_class)
  out.levelSeparation = int(diag.level)
  out.headingErrorDeg = float(diag.heading_err_deg)
  out.distanceToCenterlineM = float(diag.d_center_m if math.isfinite(diag.d_center_m) else 0.0)
  out.visHorizonM = float(max(0.0, vis_horizon_m))
  # Optional debug vectors if explicitly enabled via param
  try:
    if Params().get_int('MTSCLogDetail') > 0:
      if hasattr(diag, 'distances_m') and isinstance(diag.distances_m, list):
        out.distancesM = [float(x) for x in diag.distances_m[:30]]
      if hasattr(diag, 'kappas_per_m') and isinstance(diag.kappas_per_m, list):
        out.kappasPerM = [float(x) for x in diag.kappas_per_m[:30]]
      if hasattr(diag, 'vsafe_mps') and isinstance(diag.vsafe_mps, list):
        out.vSafeMps = [float(x) for x in diag.vsafe_mps[:30]]
  except Exception:
    pass
  try:
    pm.send('mapTurnSpeedControlSP', msg)
  except Exception:
    # Never crash publisher on transport errors
    pass


# ===== M3: Horizon & Curvature helpers =====
EARTH_R = 6371007.2


def _xy_from_latlon(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
  # Equirectangular approximation in meters relative to (lat0, lon0)
  dlat = math.radians(lat - lat0)
  dlon = math.radians(lon - lon0)
  x = EARTH_R * dlon * math.cos(math.radians(lat0))
  y = EARTH_R * dlat
  return x, y


def _project_s_on_centerline(centerline: list, lat: float, lon: float) -> tuple[float, float]:
  """Return (s_proj, d_perp) where s is distanceFromStart at projection and d_perp is perp distance (m)."""
  best_d = math.inf
  best_s = 0.0
  n = len(centerline)
  if n == 0:
    return 0.0, math.inf
  # ensure distanceFromStart exists and is monotonic; if missing, synthesize
  s_list = []
  lat_list = []
  lon_list = []
  ok = True
  for i in range(n):
    try:
      s_list.append(float(centerline[i].distanceFromStart))
      lat_list.append(float(centerline[i].latitude))
      lon_list.append(float(centerline[i].longitude))
    except Exception:
      ok = False
      break
  if not ok or any(s_list[i] > s_list[i+1] for i in range(len(s_list)-1)):
    # synthesize cumulative distances
    s_list = [0.0]
    lat_list = [] if lat_list else [float(centerline[0].latitude)]
    lon_list = [] if lon_list else [float(centerline[0].longitude)]
    lat_list = [float(getattr(centerline[0], 'latitude', 0.0))]
    lon_list = [float(getattr(centerline[0], 'longitude', 0.0))]
    for i in range(1, n):
      lat_i = float(getattr(centerline[i], 'latitude', 0.0))
      lon_i = float(getattr(centerline[i], 'longitude', 0.0))
      lat_prev = float(getattr(centerline[i-1], 'latitude', 0.0))
      lon_prev = float(getattr(centerline[i-1], 'longitude', 0.0))
      ds = Coordinate(lat_prev, lon_prev).distance_to(Coordinate(lat_i, lon_i))
      s_list.append(s_list[-1] + ds)
      lat_list.append(lat_i)
      lon_list.append(lon_i)

  for i in range(n - 1):
    latA, lonA = lat_list[i], lon_list[i]
    latB, lonB = lat_list[i+1], lon_list[i+1]
    Ax, Ay = 0.0, 0.0
    Bx, By = _xy_from_latlon(latB, lonB, latA, lonA)
    Px, Py = _xy_from_latlon(lat, lon, latA, lonA)
    ABx, ABy = Bx - Ax, By - Ay
    AB2 = ABx*ABx + ABy*ABy
    if AB2 <= 1e-6:
      continue
    t = max(0.0, min(1.0, (Px*ABx + Py*ABy) / AB2))
    Qx, Qy = Ax + t*ABx, Ay + t*ABy
    d = math.hypot(Px - Qx, Py - Qy)
    if d < best_d:
      best_d = d
      sA, sB = s_list[i], s_list[i+1]
      best_s = sA + t * (sB - sA)
  return best_s, best_d


def _interpolate_latlon_from_s(s_list: list[float], lat_list: list[float], lon_list: list[float], s_target: float) -> tuple[float, float]:
  if s_target <= s_list[0]:
    return lat_list[0], lon_list[0]
  if s_target >= s_list[-1]:
    return lat_list[-1], lon_list[-1]
  # binary search for segment
  lo, hi = 0, len(s_list) - 1
  while lo + 1 < hi:
    mid = (lo + hi) // 2
    if s_list[mid] <= s_target:
      lo = mid
    else:
      hi = mid
  s0, s1 = s_list[lo], s_list[lo+1]
  t = 0.0 if s1 <= s0 else (s_target - s0) / (s1 - s0)
  lat = lat_list[lo] + t * (lat_list[lo+1] - lat_list[lo])
  lon = lon_list[lo] + t * (lon_list[lo+1] - lon_list[lo])
  return lat, lon


def _compute_curvature(xs: list[float], ys: list[float]) -> list[float]:
  """Return |curvature| per middle point for triples (i,i+1,i+2)."""
  k = []
  for i in range(len(xs) - 2):
    x1, y1 = xs[i], ys[i]
    x2, y2 = xs[i+1], ys[i+1]
    x3, y3 = xs[i+2], ys[i+2]
    a = math.hypot(x2 - x1, y2 - y1)
    b = math.hypot(x3 - x2, y3 - y2)
    c = math.hypot(x3 - x1, y3 - y1)
    if a <= 1e-3 or b <= 1e-3 or c <= 1e-3:
      k.append(0.0)
      continue
    # triangle area via shoelace
    A = abs(0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)))
    try:
      kappa = 4.0 * A / (a * b * c)
    except ZeroDivisionError:
      kappa = 0.0
    k.append(float(max(0.0, kappa)))
  return k


def build_horizon_and_diagnostics(seg, lat: float, lon: float, v_ego: float,
                                  resample_m: float = 3.0,
                                  t_min_s: float = 15.0,
                                  t_max_s: float = 30.0,
                                  min_dist_m: float = 80.0,
                                  max_dist_m: float = 450.0) -> dict:
  # Extract centerline
  pts = getattr(seg, 'centerline', [])
  n = len(pts)
  if n < 3:
    return {'coverage': 0.0}
  s_list = []
  lat_list = []
  lon_list = []
  try:
    for i in range(n):
      s_list.append(float(pts[i].distanceFromStart))
      lat_list.append(float(pts[i].latitude))
      lon_list.append(float(pts[i].longitude))
  except Exception:
    # synthesize distances
    s_list = [0.0]
    lat_list = [float(getattr(pts[0], 'latitude', 0.0))]
    lon_list = [float(getattr(pts[0], 'longitude', 0.0))]
    for i in range(1, n):
      lat_i = float(getattr(pts[i], 'latitude', 0.0))
      lon_i = float(getattr(pts[i], 'longitude', 0.0))
      ds = Coordinate(lat_list[-1], lon_list[-1]).distance_to(Coordinate(lat_i, lon_i))
      s_list.append(s_list[-1] + ds)
      lat_list.append(lat_i)
      lon_list.append(lon_i)

  s0, d_perp = _project_s_on_centerline(pts, lat, lon)
  s_end = s_list[-1]

  # Horizon length selection
  # T scales from t_min to t_max by speed up to 30 m/s
  sp = max(0.0, min(30.0, v_ego))
  T = t_min_s + (t_max_s - t_min_s) * (sp / 30.0)
  S = max(min_dist_m, min(max_dist_m, v_ego * T))
  s_goal = min(s0 + S, s_end)
  avail = max(0.0, s_goal - s0)
  coverage = avail / S if S > 0 else 0.0

  if avail < resample_m * 2:
    return {'coverage': coverage}

  # Resample lat/lon
  num = int(avail // resample_m) + 3  # ensure at least 3 points
  s_vals = [s0 + i * resample_m for i in range(num)]
  if s_vals[-1] > s_goal:
    s_vals[-1] = s_goal
  lats = []
  lons = []
  lat_ref, lon_ref = _interpolate_latlon_from_s(s_list, lat_list, lon_list, s_vals[0])
  xs: list[float] = []
  ys: list[float] = []
  for sv in s_vals:
    la, lo = _interpolate_latlon_from_s(s_list, lat_list, lon_list, sv)
    lats.append(la)
    lons.append(lo)
    x, y = _xy_from_latlon(la, lo, lat_ref, lon_ref)
    xs.append(x)
    ys.append(y)

  # Curvature per middle sample
  kappas = _compute_curvature(xs, ys)
  # Map to distances aligned to middle points
  d_mids = [i * resample_m + resample_m for i in range(len(kappas))]
  # Physics speeds if function available
  vsafe = []
  if curvature_to_speed is not None:
    for k in kappas:
      try:
        vsafe.append(float(curvature_to_speed(max(1e-8, float(k)))))
      except Exception:
        vsafe.append(0.0)
  else:
    vsafe = [0.0 for _ in kappas]

  # Min speed stats
  min_speed = 0.0
  min_at = 0.0
  if len(vsafe) > 0:
    idx = int(min(range(len(vsafe)), key=lambda i: vsafe[i]))
    min_speed = float(vsafe[idx])
    min_at = float(d_mids[idx])

  # Decimate vectors to ≤30 elements for message
  def decimate(arr: list[float], maxn: int = 30) -> list[float]:
    if len(arr) <= maxn:
      return arr
    step = len(arr) / maxn
    return [arr[int(i * step)] for i in range(maxn)]

  diag = {
    'coverage': float(coverage),
    'distances_m': [float(x) for x in decimate(d_mids)],
    'kappas_per_m': [float(x) for x in decimate(kappas)],
    'vsafe_mps': [float(x) for x in decimate(vsafe)],
    'min_speed_mps': float(min_speed),
    'min_speed_at_m': float(min_at),
  }
  return diag


def main() -> None:
  params = Params()
  # Optional realtime scheduling, disabled by default to prevent starvation
  use_rt = False
  try:
    use_rt = bool(params.get_bool('MTSCRealtime'))
  except Exception:
    use_rt = False
  if not use_rt:
    try:
      ev = os.getenv('MTSC_REALTIME')
      if ev and ev.lower() not in ('0', 'false', 'no'):
        use_rt = True
    except Exception:
      pass
  if use_rt:
    try:
      config_realtime_process(5, Priority.CTRL_LOW)
    except Exception as e:
      try:
        cloudlog.event('mtscd: realtime config failed', error=str(e))
      except Exception:
        pass

  # Subscribe to both GPS services to avoid misselection and stalls
  gps_qcom = 'gpsLocation'
  gps_ublox = 'gpsLocationExternal'
  sm = messaging.SubMaster(['liveMapDataSP', 'carState', 'selfdriveState', gps_qcom, gps_ublox],
                           ignore_avg_freq=True, poll='selfdriveState')
  pm = messaging.PubMaster(['mapTurnSpeedControlSP'])

  rk = Ratekeeper(10, print_delay_threshold=None)
  cloudlog.info('mtscd: started')

  # M2 continuity state
  stable_way_id: int | None = None
  stable_score: float = 0.0
  stable_since: float = 0.0
  best_way_id: int | None = None
  best_since: float = 0.0

  # M2 tunables (could be Params in later pass)
  MIN_CONF = 0.70
  DROP_CONF = 0.45
  SWITCH_MARGIN = 0.08
  SWITCH_DWELL_S = 0.40

  # profiling controls
  try:
    prof_enabled = bool(params.get_bool('MTSCProfile'))
  except Exception:
    prof_enabled = False
  if not prof_enabled:
    try:
      ev = os.getenv('MTSC_PROFILE')
      if ev and ev.lower() not in ('0', 'false', 'no'):
        prof_enabled = True
    except Exception:
      pass

  while True:
    # Initialize timing markers to avoid unbound vars on early exceptions
    t_loop_start = time.monotonic()
    t_after_update = t_loop_start
    t_after_inputs = t_loop_start
    t_after_select = t_loop_start
    t_after_horizon = t_loop_start
    # Block modestly to cooperate with scheduler and other daemons
    sm.update(100)

    # Visible horizon estimate: use a default until integrated with VTSC
    vis_horizon_s = 1.3
    v_ego = float(sm['carState'].vEgo) if sm['carState'] is not None else 0.0
    vis_horizon_m = max(0.0, v_ego * vis_horizon_s)

    # Choose GPS source dynamically with preference for Qualcomm if alive
    gps_service = gps_qcom if sm.alive.get(gps_qcom, False) else (gps_ublox if sm.alive.get(gps_ublox, False) else gps_qcom)
    t_after_update = time.monotonic()
    inp = read_inputs(sm, gps_service)
    t_after_inputs = time.monotonic()

    # Candidate selection (M2): choose best segment based on distance/heading/class/level, apply continuity/hysteresis
    try:
      mapd = sm['liveMapDataSP']
      gps = sm[gps_service]
      candidates = []
      if getattr(mapd, 'currentRoadSegment', None) is not None:
        candidates.append(mapd.currentRoadSegment)
      try:
        nearby = getattr(mapd, 'nearbyRoadSegments', [])
        # Allow param to cap nearby segments for performance tuning
        try:
          limit = int(params.get('MTSCNearbyLimit') or 10)
        except Exception:
          limit = 10
        # env override
        try:
          ev = os.getenv('MTSC_NEARBY_LIMIT')
          if ev is not None and ev.strip() != '':
            limit = int(ev)
        except Exception:
          pass
        limit = max(0, min(10, limit))
        for seg in nearby[:limit]:
          candidates.append(seg)
      except Exception:
        pass

      def seg_metrics(seg) -> tuple[float, float, int, int, int]:
        try:
          ego_bearing = float(getattr(gps, 'bearingDeg', 0.0))
        except Exception:
          ego_bearing = 0.0
        road_dir = float(getattr(seg, 'roadDirection', 0.0))
        heading_err = _heading_error_deg(ego_bearing, road_dir)
        # Distance to centerline
        dmin = math.inf
        try:
          lat = float(gps.latitude)
          lon = float(gps.longitude)
          ego = Coordinate(lat, lon)
          pts = getattr(seg, 'centerline', [])
          for i in range(max(0, len(pts) - 1)):
            a = Coordinate(float(pts[i].latitude), float(pts[i].longitude))
            b = Coordinate(float(pts[i+1].latitude), float(pts[i+1].longitude))
            dmin = min(dmin, minimum_distance(a, b, ego))
        except Exception:
          dmin = math.inf
        rc = getattr(seg, 'roadClass', 7)
        road_class = rc if isinstance(rc, int) else 7
        level = int(getattr(seg, 'levelSeparation', 0))
        way_id = int(getattr(seg, 'wayId', 0))
        return dmin, heading_err, road_class, level, way_id

      def seg_score(dist_m: float, head_deg: float, rclass: int, level: int) -> float:
        # Reuse confidence components
        di = Inputs(d_center_m=dist_m, heading_err_deg=head_deg, road_class=rclass, level=level)
        return compute_confidence(di)

      best = None
      best_sc = -1.0
      best_metrics = None
      # Evaluate candidates
      for seg in candidates:
        dmin, herr, rclass, lvl, wid = seg_metrics(seg)
        sc = seg_score(dmin, herr, rclass, lvl)
        if sc > best_sc:
          best_sc = sc
          best = seg
          best_metrics = (dmin, herr, rclass, lvl, wid)

      now = time.monotonic()

      # Track who is best over time (for dwell logic)
      if best is not None:
        wid_best = int(getattr(best, 'wayId', 0))
        if best_way_id != wid_best:
          best_way_id = wid_best
          best_since = now

      # Initialize stable on first good candidate
      if stable_way_id is None and best is not None and best_sc >= MIN_CONF:
        stable_way_id = int(getattr(best, 'wayId', 0))
        stable_score = best_sc
        stable_since = now

      # Consider switching if a new best persists and is meaningfully better
      if best is not None and stable_way_id is not None:
        wid_best = int(getattr(best, 'wayId', 0))
        if wid_best != stable_way_id:
          if (best_sc >= (stable_score + SWITCH_MARGIN)) and ((now - best_since) >= SWITCH_DWELL_S):
            stable_way_id = wid_best
            stable_score = best_sc
            stable_since = now
        else:
          # Update stable score with some inertia
          stable_score = 0.8 * stable_score + 0.2 * best_sc
          stable_since = stable_since if stable_since > 0 else now

      # Drop stable if confidence degraded significantly
      if stable_way_id is not None and stable_score < DROP_CONF:
        stable_way_id = None
        stable_score = 0.0
        stable_since = 0.0

      # Expose selected (stable if set, else best) as diagnostics
      active_metrics = best_metrics
      if stable_way_id is not None and best_metrics is not None:
        # if best is not stable, recompute metrics for stable to publish
        if stable_way_id != best_metrics[4]:
          # find stable seg among candidates
          for seg in candidates:
            if int(getattr(seg, 'wayId', 0)) == stable_way_id:
              active_metrics = seg_metrics(seg)
              break
        # use stable score for confidence if stable exists
        active_score = stable_score
      else:
        active_score = best_sc if best_sc >= 0.0 else 0.0

      if active_metrics is not None:
        dmin, herr, rclass, lvl, wid = active_metrics
        inp.matched_way_id = wid
        inp.road_class = rclass
        inp.level = lvl
        inp.heading_err_deg = herr
        inp.d_center_m = dmin
        # override confidence in publish by reflecting active score via compute_confidence(inp)
      t_after_select = time.monotonic()
    except Exception:
      cloudlog.exception('mtscd: candidate selection failed')
      t_after_select = time.monotonic()

    # Default horizon timestamp for profiler; may be updated if we build a horizon
    t_after_horizon = t_after_select

    # Skeleton behavior with M3 diagnostics: publish unavailable unless onroad, gps fresh, and map valid
    if not (inp.onroad and inp.gps_fresh and inp.map_valid):
      publish_unavailable(pm, vis_horizon_m, inp)
    else:
      # Build M3 horizon diagnostics if we have a selected/diagnosed segment
      try:
        gps = sm[gps_service]
        lat = float(getattr(gps, 'latitude', 0.0))
        lon = float(getattr(gps, 'longitude', 0.0))
        # Fallback to mapd's fused GPS if unavailable
        if not (lat or lon):
          raise ValueError('empty gps')
      except Exception:
        lat = lon = 0.0
        try:
          # Prefer memparams from mapd_manager
          mempos = coordinate_from_param('LastGPSPosition', Params('/dev/shm/params'))
          if mempos is None:
            mempos = coordinate_from_param('LastGPSPosition', Params())
          if mempos is not None:
            lat = float(mempos.latitude)
            lon = float(mempos.longitude)
        except Exception:
          pass

      # Find the segment whose wayId matches our current diagnostics (stable or best)
      seg_use = None
      try:
        mapd = sm['liveMapDataSP']
        if getattr(mapd, 'currentRoadSegment', None) is not None and int(getattr(mapd.currentRoadSegment, 'wayId', 0)) == inp.matched_way_id:
          seg_use = mapd.currentRoadSegment
        else:
          for seg in getattr(mapd, 'nearbyRoadSegments', [])[:10]:
            if int(getattr(seg, 'wayId', 0)) == inp.matched_way_id:
              seg_use = seg
              break
      except Exception:
        seg_use = None

      if seg_use is not None:
        diag = build_horizon_and_diagnostics(seg_use, lat, lon, inp.v_ego)
        # Attach diagnostic fields to inp for publishing
        inp.horizon_coverage = float(diag.get('coverage', 0.0))  # type: ignore[attr-defined]
        inp.distances_m = list(diag.get('distances_m', []))      # type: ignore[attr-defined]
        inp.kappas_per_m = list(diag.get('kappas_per_m', []))    # type: ignore[attr-defined]
        inp.vsafe_mps = list(diag.get('vsafe_mps', []))          # type: ignore[attr-defined]
        inp.min_speed_mps = float(diag.get('min_speed_mps', 0.0))# type: ignore[attr-defined]
        inp.min_speed_at_m = float(diag.get('min_speed_at_m', 0.0))# type: ignore[attr-defined]
      # Always set horizon timestamp for profiler
      t_after_horizon = time.monotonic()

      # M4 gating and target computation
      conf = compute_confidence(inp)
      coverage = float(getattr(inp, 'horizon_coverage', 0.0))
      vsafe = list(getattr(inp, 'vsafe_mps', []))
      dgrid = list(getattr(inp, 'distances_m', []))
      speed_gate = 29.06  # ~65 mph
      min_conf = 0.70
      min_cov = 0.60
      margin_m = 10.0
      start_dist = max(0.0, vis_horizon_m + margin_m)

      def reachable_cap(vsafe: list[float], dgrid: list[float], s_start: float, v_now: float, a_comf: float = 1.47) -> float:
        if not vsafe or not dgrid or len(vsafe) != len(dgrid):
          return v_now
        vmax = v_now
        for vi, di in zip(vsafe, dgrid, strict=False):
          if di < s_start:
            continue
          # max current speed to decel comfortably to vi over (di - s_start)
          d = max(0.0, di - s_start)
          try:
            v_allow = math.sqrt(max(0.0, vi*vi + 2.0 * a_comf * d))
          except Exception:
            v_allow = v_now
          vmax = min(vmax, v_allow)
        return vmax

      can_offer = (inp.v_ego <= speed_gate) and (conf >= min_conf) and (coverage >= min_cov) and (len(vsafe) >= 3)
      if can_offer and curvature_to_speed is not None:
        v_cap = reachable_cap(vsafe, dgrid, start_dist, inp.v_ego)
        # Do not suggest acceleration; clamp to current speed
        v_target = min(inp.v_ego, v_cap)
        publish_recommendation(pm, vis_horizon_m, inp, v_target, start_dist, conf)
      else:
        publish_unavailable(pm, vis_horizon_m, inp)

    # Profiling and watchdog
    lagged = rk.keep_time()
    if prof_enabled:
      t_end = time.monotonic()
      dt_update = (t_after_update - t_loop_start) * 1000.0
      dt_inputs = (t_after_inputs - t_after_update) * 1000.0
      dt_select = (t_after_select - t_after_inputs) * 1000.0
      try:
        base_h = t_after_horizon
      except NameError:
        base_h = t_after_select
      dt_horizon = (base_h - t_after_select) * 1000.0
      dt_total = (t_end - t_loop_start) * 1000.0
      # statsd gauges (lightweight)
      try:
        statlog.gauge('mtscd_dt_update_ms', dt_update)
        statlog.gauge('mtscd_dt_inputs_ms', dt_inputs)
        statlog.gauge('mtscd_dt_select_ms', dt_select)
        statlog.gauge('mtscd_dt_horizon_ms', dt_horizon)
        statlog.gauge('mtscd_dt_total_ms', dt_total)
        statlog.gauge('mtscd_lagged', int(bool(lagged)))
      except Exception:
        pass
      # occasional log if over budget (>8ms per tick)
      if dt_total > 8.0:
        try:
          cloudlog.event('mtscd.profile', dt_total_ms=dt_total, dt_update_ms=dt_update,
                         dt_inputs_ms=dt_inputs, dt_select_ms=dt_select, dt_horizon_ms=dt_horizon)
        except Exception:
          pass


if __name__ == '__main__':
  main()
