"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import math
import os
import platform
import time

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.mapd.live_map_data.base_map_data import BaseMapData
from openpilot.sunnypilot.navd.helpers import Coordinate
from openpilot.sunnypilot.mapd.road_geometry import (
    OfflineRoadGeometryExtractor, OSMRoadGeometryExtractor, RoadGeometryCache, RoadSegment
)
from openpilot.system.hardware.hw import Paths


# LastGPSPosition is persistent so mapd can bootstrap after a restart, but GPS
# arrives continuously. Bound disk writes while still refreshing the restart
# anchor during a drive and periodically while stationary.
LAST_GPS_PERSIST_MIN_INTERVAL_S = 300.0
LAST_GPS_PERSIST_MAX_INTERVAL_S = 1800.0
LAST_GPS_PERSIST_MIN_DISTANCE_M = 1000.0


class OsmMapData(BaseMapData):
  def __init__(self):
    super().__init__()
    self.params = Params()
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params

    # Initialize road geometry components
    self.road_geometry_extractor: OSMRoadGeometryExtractor | OfflineRoadGeometryExtractor | None = None
    self.road_geometry_cache = RoadGeometryCache()
    self.current_road_segment: RoadSegment | None = None
    self.nearby_road_segments: list[RoadSegment] = []
    self.road_geometry_valid = False
    self.road_geometry_failure_reason: str | None = None
    self.local_map_health_issue: str | None = None
    self._missing_context_cycles = 0
    self._last_persisted_gps_monotonic: float | None = None
    self._last_persisted_gps_position: Coordinate | None = None

    # Initialize road geometry extractor
    self._initialize_road_geometry()

  def _initialize_road_geometry(self):
    """Initialize road geometry extractor from the best available local source."""
    try:
      mapd_root = Paths.mapd_root()
      db_candidates = [
        os.path.join(mapd_root, "osm.db"),
        os.path.join(mapd_root, "db", "osm.db"),
        os.path.join(mapd_root, "data.db"),
      ]
      offline_root = os.path.join(mapd_root, "offline")

      db_path = None
      for candidate in db_candidates:
        if os.path.exists(candidate):
          db_path = candidate
          break

      if db_path:
        self.road_geometry_extractor = OSMRoadGeometryExtractor(db_path)
        cloudlog.info(f"Road geometry extractor initialized with database: {db_path}")
      elif os.path.isdir(offline_root):
        self.road_geometry_extractor = OfflineRoadGeometryExtractor(mapd_root)
        cloudlog.info(f"Road geometry extractor initialized with offline tiles: {offline_root}")
      else:
        self.road_geometry_failure_reason = (
          f"No supported local road geometry source found under {mapd_root}. "
          "Expected an OSM database or extracted offline tiles."
        )
        cloudlog.error(self.road_geometry_failure_reason)

    except Exception as e:
      self.road_geometry_failure_reason = f"Failed to initialize road geometry extractor: {e}"
      cloudlog.error(self.road_geometry_failure_reason)
      self.road_geometry_extractor = None

  @staticmethod
  def _valid_coordinate(latitude, longitude) -> tuple[float, float] | None:
    try:
      if isinstance(latitude, bool) or isinstance(longitude, bool):
        return None
      latitude_f = float(latitude)
      longitude_f = float(longitude)
    except (TypeError, ValueError, OverflowError):
      return None
    if not math.isfinite(latitude_f) or not math.isfinite(longitude_f):
      return None
    if not -90.0 <= latitude_f <= 90.0 or not -180.0 <= longitude_f <= 180.0:
      return None
    if latitude_f == 0.0 and longitude_f == 0.0:
      return None
    return latitude_f, longitude_f

  def _validated_live_gps_payload(self) -> dict | None:
    """Return a real, newly received GPS fix suitable for Params publication."""
    service = self.gps_location_service
    try:
      # Requiring all three SubMaster states prevents startup defaults, stale
      # messages, and invalid cereal messages from becoming a restart anchor.
      if (self.sm.updated[service] is not True or
          self.sm.valid[service] is not True or
          self.sm.alive[service] is not True):
        return None
      gps = self.sm[service]
      if getattr(gps, "hasFix", False) is not True:
        return None
    except Exception:
      return None

    coordinate = self._valid_coordinate(getattr(gps, "latitude", None), getattr(gps, "longitude", None))
    if coordinate is None:
      return None
    latitude, longitude = coordinate

    # openpilot-mapd expects bearing degrees to disambiguate direction on
    # one-way roads. extract_bearing_deg also derives it from velocity when an
    # explicit bearing is unavailable.
    bearing_deg = self.extract_bearing_deg(gps)
    if not math.isfinite(bearing_deg):
      bearing_deg = 0.0
    bearing_deg %= 360.0

    payload = {
      "latitude": latitude,
      "longitude": longitude,
      "bearing": bearing_deg,
    }
    try:
      altitude = getattr(gps, "altitude", None)
      if not isinstance(altitude, bool):
        altitude_f = float(altitude)
        if math.isfinite(altitude_f):
          payload["altitude"] = altitude_f
    except (TypeError, ValueError, OverflowError):
      pass
    return payload

  def _should_persist_gps(self, position: Coordinate, now_monotonic: float) -> bool:
    if self._last_persisted_gps_monotonic is None or self._last_persisted_gps_position is None:
      return True

    elapsed_s = now_monotonic - self._last_persisted_gps_monotonic
    if elapsed_s < LAST_GPS_PERSIST_MIN_INTERVAL_S:
      return False
    if elapsed_s >= LAST_GPS_PERSIST_MAX_INTERVAL_S:
      return True

    try:
      moved_m = self._last_persisted_gps_position.distance_to(position)
    except Exception:
      return False
    return math.isfinite(moved_m) and moved_m >= LAST_GPS_PERSIST_MIN_DISTANCE_M

  def _publish_validated_gps(self, payload: dict) -> None:
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    position = Coordinate(payload["latitude"], payload["longitude"])

    # Shared-memory publication remains live on tici and never touches flash.
    # On Darwin mem_params aliases persistent Params, so the throttled write
    # below is the only publication path.
    if self.mem_params is not self.params:
      self.mem_params.put("LastGPSPosition", serialized)

    now_monotonic = time.monotonic()
    if not self._should_persist_gps(position, now_monotonic):
      return
    try:
      self.params.put_nonblocking("LastGPSPosition", serialized)
    except Exception as e:
      cloudlog.error(f"Failed to persist validated LastGPSPosition: {e}")
      return
    self._last_persisted_gps_monotonic = now_monotonic
    self._last_persisted_gps_position = position

  def update_location(self) -> None:
    payload = self._validated_live_gps_payload()
    if payload is not None:
      self.last_position = Coordinate(payload["latitude"], payload["longitude"])
      self.last_altitude = payload.get("altitude")
      self.last_bearing = payload["bearing"]
      self._publish_validated_gps(payload)

    if (self.last_position is None or self.last_altitude is None or
        self._valid_coordinate(self.last_position.latitude, self.last_position.longitude) is None):
      self.road_geometry_valid = False
      self._update_local_map_health_issue()
      return

    # Update road geometry information, but never let geometry failures break
    # legacy mapd behavior.
    try:
      self._update_road_geometry()
    except Exception as e:
      cloudlog.error(f"Error updating road geometry: {e}")
      self.road_geometry_valid = False
    self._update_local_map_health_issue()

  def _update_road_geometry(self):
    """Update road geometry data for current position."""
    if not self.road_geometry_extractor or not self.last_position:
      self.road_geometry_valid = False
      return

    try:
      # Get nearby road segments
      self.nearby_road_segments = self.road_geometry_cache.get_road_segments_near(
        self.last_position, self.road_geometry_extractor
      )

      # Find current road segment
      self.current_road_segment = self.road_geometry_cache.find_current_road_segment(
        self.last_position
      )

      self.road_geometry_valid = len(self.nearby_road_segments) > 0

    except Exception as e:
      cloudlog.error(f"Error updating road geometry: {e}")
      self.road_geometry_valid = False

  def get_current_speed_limit(self) -> float:
    mem_speed_limit = float(self.mem_params.get("MapSpeedLimit") or 0.0)
    if mem_speed_limit > 0.0:
      return mem_speed_limit

    try:
      current_segment_speed = float(getattr(self.current_road_segment, "max_speed", 0.0) or 0.0)
    except (TypeError, ValueError):
      current_segment_speed = 0.0

    if current_segment_speed > 0.0:
      return current_segment_speed

    return 0.0

  def _read_mem_json(self, key: str) -> dict:
    raw = self.mem_params.get(key)
    if not raw:
      return {}
    if isinstance(raw, dict):
      return raw
    if isinstance(raw, bytes):
      try:
        raw = raw.decode('utf-8')
      except Exception:
        return {}
    if isinstance(raw, str):
      try:
        parsed = json.loads(raw)
      except Exception:
        return {}
      return parsed if isinstance(parsed, dict) else {}
    return {}

  def get_current_road_name(self) -> str:
    try:
      if self.current_road_segment and getattr(self.current_road_segment, 'name', ""):
        return str(self.current_road_segment.name)
    except Exception:
      pass
    # Fallback to legacy shared memory param if available
    return str(self.mem_params.get("RoadName") or "")

  def get_next_speed_limit_and_distance(self) -> tuple[float, float]:
    next_speed_limit_section = self._read_mem_json("NextMapSpeedLimit")
    next_speed_limit = next_speed_limit_section.get('speedlimit', 0.0)
    next_speed_limit_latitude = next_speed_limit_section.get('latitude')
    next_speed_limit_longitude = next_speed_limit_section.get('longitude')
    next_speed_limit_distance = 0.0

    if next_speed_limit_latitude and next_speed_limit_longitude:
      next_speed_limit_coordinates = Coordinate(next_speed_limit_latitude, next_speed_limit_longitude)
      next_speed_limit_distance = (self.last_position or Coordinate(0, 0)).distance_to(next_speed_limit_coordinates)

    return next_speed_limit, next_speed_limit_distance

  def get_winding_road_summary(self) -> dict | None:
    return self._read_mem_json("MapWindingSummary")

  def _update_local_map_health_issue(self) -> None:
    issue = None
    if self.params.get_bool("OsmLocal"):
      if self.road_geometry_extractor is None:
        issue = self.road_geometry_failure_reason or "No supported local road geometry source is available."
        self._missing_context_cycles = 0
      else:
        has_context = self.road_geometry_valid or bool(self.get_current_road_name().strip()) or self.get_current_speed_limit() > 0.0
        if self.last_position and not has_context:
          self._missing_context_cycles += 1
          if self._missing_context_cycles >= 5:
            issue = (
              "Local map data is enabled, but the current GPS position still has no "
              "road name, speed limit, or road geometry context."
            )
        else:
          self._missing_context_cycles = 0
    else:
      self._missing_context_cycles = 0

    if issue and issue != self.local_map_health_issue:
      cloudlog.error(f"Local map data issue: {issue}")
    self.local_map_health_issue = issue

  def get_local_map_health_issue(self) -> str | None:
    return self.local_map_health_issue

  # Road geometry access methods for RTI integration
  def get_road_geometry_valid(self) -> bool:
    """Check if road geometry data is available and valid."""
    return self.road_geometry_valid

  def get_current_road_segment(self) -> RoadSegment | None:
    """Get the road segment vehicle is currently on."""
    return self.current_road_segment

  def get_nearby_road_segments(self) -> list[RoadSegment]:
    """Get all road segments within detection radius."""
    return self.nearby_road_segments
