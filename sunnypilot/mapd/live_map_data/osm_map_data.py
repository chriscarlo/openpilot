"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import math
import os
import platform

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.mapd.live_map_data.base_map_data import BaseMapData
from openpilot.sunnypilot.navd.helpers import Coordinate
from openpilot.sunnypilot.mapd.road_geometry import (
    OfflineRoadGeometryExtractor, OSMRoadGeometryExtractor, RoadGeometryCache, RoadSegment
)
from openpilot.system.hardware.hw import Paths


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

  def update_location(self) -> None:
    if self.last_position is None or self.last_altitude is None:
      return

    # openpilot-mapd expects LastGPSPosition to include `bearing` (degrees) to
    # disambiguate direction on one-way roads. Without it, mapd can fail to match
    # the current way and MTSC/MapCurvatures can stay empty (`[]`).
    try:
      bearing_deg = self.extract_bearing_deg(self.sm[self.gps_location_service])
    except Exception:
      bearing_deg = 0.0

    # Avoid serializing NaN/Inf into params JSON. Go's json parser rejects these.
    if not math.isfinite(bearing_deg):
      bearing_deg = float(getattr(self, 'last_bearing', 0.0) or 0.0)
    if not math.isfinite(bearing_deg):
      bearing_deg = 0.0

    params = {
      "latitude": self.last_position.latitude,
      "longitude": self.last_position.longitude,
      "altitude": self.last_altitude,
      "bearing": bearing_deg,
    }

    self.mem_params.put("LastGPSPosition", json.dumps(params))

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
