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
from typing import Optional, Tuple

import cereal.messaging as messaging
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.navd.helpers import Coordinate, minimum_distance


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
      inp.road_class = int(getattr(seg, 'roadClass', 7))
      inp.level = int(getattr(seg, 'levelSeparation', 0))
      try:
        ego_bearing = float(getattr(gps, 'bearingDeg', 0.0))
      except Exception:
        ego_bearing = 0.0
      road_dir = float(getattr(seg, 'roadDirection', 0.0))
      inp.heading_err_deg = _heading_error_deg(ego_bearing, road_dir)
      # Distance to centerline if GPS position available
      try:
        lat = float(getattr(gps, 'latitude'))
        lon = float(getattr(gps, 'longitude'))
        ego = Coordinate(lat, lon)
        pts = getattr(seg, 'centerline', [])
        dmin = math.inf
        for i in range(0, max(0, len(pts) - 1)):
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
  out.horizonCoverage = 0.0
  out.minSpeedMps = 0.0
  out.minSpeedAtDistanceM = 0.0
  out.matchedWayId = int(diag.matched_way_id)
  out.roadClass = int(diag.road_class)
  out.levelSeparation = int(diag.level)
  out.headingErrorDeg = float(diag.heading_err_deg)
  out.distanceToCenterlineM = float(diag.d_center_m if math.isfinite(diag.d_center_m) else 0.0)
  out.visHorizonM = float(max(0.0, vis_horizon_m))
  pm.send('mapTurnSpeedControlSP', msg)


def main() -> None:
  params = Params()
  # real-time priority for low overhead
  config_realtime_process(5)
  gps_service = get_gps_location_service(params)
  sm = messaging.SubMaster(['liveMapDataSP', 'carState', 'selfdriveState', gps_service], ignore_avg_freq=True)
  pm = messaging.PubMaster(['mapTurnSpeedControlSP'])

  rk = Ratekeeper(10, print_delay_threshold=None)
  cloudlog.info('mtscd: started')

  while True:
    sm.update(0)

    # Visible horizon estimate: use a default until integrated with VTSC
    vis_horizon_s = 1.3
    v_ego = float(sm['carState'].vEgo) if sm['carState'] is not None else 0.0
    vis_horizon_m = max(0.0, v_ego * vis_horizon_s)

    inp = read_inputs(sm, gps_service)

    # Skeleton behavior: publish unavailable unless onroad, gps fresh, and map valid
    if not (inp.onroad and inp.gps_fresh and inp.map_valid):
      publish_unavailable(pm, vis_horizon_m, inp)
    else:
      # Placeholder: until M2–M4, do not provide recommendations
      publish_unavailable(pm, vis_horizon_m, inp)

    rk.keep_time()


if __name__ == '__main__':
  main()
