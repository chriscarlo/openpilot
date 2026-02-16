"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import os
import platform

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.mapd.live_map_data.base_map_data import BaseMapData
from openpilot.sunnypilot.navd.helpers import Coordinate
from openpilot.sunnypilot.mapd.road_geometry import (
    OSMRoadGeometryExtractor, RoadGeometryCache, RoadSegment
)
from openpilot.system.hardware.hw import Paths


class OsmMapData(BaseMapData):
  def __init__(self):
    super().__init__()
    self.params = Params()
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params

    # Initialize road geometry components
    self.road_geometry_extractor: OSMRoadGeometryExtractor | None = None
    self.road_geometry_cache = RoadGeometryCache()
    self.current_road_segment: RoadSegment | None = None
    self.nearby_road_segments: list[RoadSegment] = []
    self.road_geometry_valid = False

    # Initialize road geometry extractor
    self._initialize_road_geometry()

  def _initialize_road_geometry(self):
    """Initialize road geometry extractor with OSM database."""
    try:
      # Find OSM database file
      mapd_root = Paths.mapd_root()
      db_candidates = [
        os.path.join(mapd_root, "osm.db"),
        os.path.join(mapd_root, "db", "osm.db"),
        os.path.join(mapd_root, "data.db"),
      ]

      db_path = None
      for candidate in db_candidates:
        if os.path.exists(candidate):
          db_path = candidate
          break

      if db_path:
        self.road_geometry_extractor = OSMRoadGeometryExtractor(db_path)
        cloudlog.info(f"Road geometry extractor initialized with database: {db_path}")
      else:
        offline_root = os.path.join(mapd_root, "offline")
        if os.path.isdir(offline_root):
          cloudlog.info("No OSM sqlite DB found; live map speed/curvature remains available via mapd")
        else:
          cloudlog.warning("No OSM database found for road geometry extraction")

    except Exception as e:
      cloudlog.error(f"Failed to initialize road geometry extractor: {e}")
      self.road_geometry_extractor = None

  def update_location(self) -> None:
    if self.last_position is None or self.last_altitude is None:
      return

    params = {
      "latitude": self.last_position.latitude,
      "longitude": self.last_position.longitude,
      "altitude": self.last_altitude,
      "bearing": float(getattr(self, 'last_bearing', 0.0) or 0.0),
    }

    self.mem_params.put("LastGPSPosition", json.dumps(params))

    # Update road geometry information
    self._update_road_geometry()

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
    return float(self.mem_params.get("MapSpeedLimit") or 0.0)

  def get_current_road_name(self) -> str:
    try:
      if self.current_road_segment and getattr(self.current_road_segment, 'name', ""):
        return str(self.current_road_segment.name)
    except Exception:
      pass
    # Fallback to legacy shared memory param if available
    return str(self.mem_params.get("RoadName"))

  def get_next_speed_limit_and_distance(self) -> tuple[float, float]:
    next_speed_limit_section_str = self.mem_params.get("NextMapSpeedLimit")
    next_speed_limit_section = next_speed_limit_section_str if next_speed_limit_section_str else {}
    next_speed_limit = next_speed_limit_section.get('speedlimit', 0.0)
    next_speed_limit_latitude = next_speed_limit_section.get('latitude')
    next_speed_limit_longitude = next_speed_limit_section.get('longitude')
    next_speed_limit_distance = 0.0

    if next_speed_limit_latitude and next_speed_limit_longitude:
      next_speed_limit_coordinates = Coordinate(next_speed_limit_latitude, next_speed_limit_longitude)
      next_speed_limit_distance = (self.last_position or Coordinate(0, 0)).distance_to(next_speed_limit_coordinates)

    return next_speed_limit, next_speed_limit_distance

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
