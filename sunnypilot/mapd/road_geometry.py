#!/usr/bin/env python3
"""
Road Geometry Processing Module

Extracts detailed road geometry, lane information, and barrier data from OSM data
to support intelligent RTI threat detection and road-aware positioning.

Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import json
import math
import sqlite3
import time
import zlib
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

import capnp

from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.navd.helpers import Coordinate, minimum_distance


class RoadClass(Enum):
    """OSM highway classification mapping."""
    MOTORWAY = "motorway"
    TRUNK = "trunk"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    RESIDENTIAL = "residential"
    SERVICE = "service"
    UNCLASSIFIED = "unclassified"


class LaneType(Enum):
    """Lane type classification."""
    DRIVING = "driving"
    BUS = "bus"
    BICYCLE = "bicycle"
    PARKING = "parking"
    SHOULDER = "shoulder"
    MEDIAN = "median"


class BarrierType(Enum):
    """Barrier type classification."""
    MEDIAN = "median"
    GUARDRAIL = "guardrail"
    WALL = "wall"
    FENCE = "fence"
    CURB = "curb"


@dataclass
class RoadCoordinate:
    """Enhanced coordinate with distance tracking."""
    latitude: float
    longitude: float
    distance_from_start: float = 0.0

    def to_coordinate(self) -> Coordinate:
        """Convert to navd Coordinate object."""
        return Coordinate(self.latitude, self.longitude)

    @classmethod
    def from_coordinate(cls, coord: Coordinate, distance: float = 0.0) -> 'RoadCoordinate':
        """Create from navd Coordinate."""
        return cls(coord.latitude, coord.longitude, distance)


@dataclass
class Lane:
    """Lane information."""
    lane_index: int  # 0 = rightmost lane
    width: float  # meters
    lane_type: LaneType
    centerline: list[RoadCoordinate]


@dataclass
class Barrier:
    """Barrier/median information."""
    barrier_type: BarrierType
    coordinates: list[RoadCoordinate]


@dataclass
class RoadSegment:
  """Complete road segment with geometry and lane data."""
  way_id: int
  name: str
  road_class: RoadClass
  centerline: list[RoadCoordinate]
  lanes: list[Lane]
  barriers: list[Barrier]
  level_separation: int  # -1=under, 0=ground, 1=bridge
  max_speed: float  # m/s
  road_direction: float  # bearing in degrees

  def get_length(self) -> float:
    """Calculate total road segment length in meters."""
    if len(self.centerline) < 2:
      return 0.0
    return self.centerline[-1].distance_from_start

  def get_closest_point(self, position: Coordinate) -> tuple[RoadCoordinate, float]:
    """
    Find closest point on road centerline to given position.
    Returns (closest_point, distance_to_road).
    """
    if not self.centerline:
      return RoadCoordinate(0, 0), float('inf')

    min_distance = float('inf')
    closest_point = self.centerline[0]

    # Check distance to each segment
    for i in range(len(self.centerline) - 1):
      p1 = self.centerline[i].to_coordinate()
      p2 = self.centerline[i + 1].to_coordinate()

      distance = minimum_distance(p1, p2, position)
      if distance < min_distance:
        min_distance = distance
        # Calculate projection point
        closest_point = self._project_to_segment(
          position,
          self.centerline[i],
          self.centerline[i + 1],
        )

    return closest_point, min_distance

  def _project_to_segment(
    self,
    point: Coordinate,
    seg_start: RoadCoordinate,
    seg_end: RoadCoordinate,
  ) -> RoadCoordinate:
    """Project point onto road segment and calculate distance along road."""
    p1 = seg_start.to_coordinate()
    p2 = seg_end.to_coordinate()

    # Vector from p1 to p2
    dx = p2.longitude - p1.longitude
    dy = p2.latitude - p1.latitude

    # Vector from p1 to point
    px = point.longitude - p1.longitude
    py = point.latitude - p1.latitude

    # Calculate projection parameter t
    segment_length_sq = dx * dx + dy * dy
    if segment_length_sq == 0:
      # Degenerate segment
      return seg_start

    t = max(0, min(1, (px * dx + py * dy) / segment_length_sq))

    # Calculate projected point
    proj_lat = p1.latitude + t * dy
    proj_lon = p1.longitude + t * dx

    # Calculate distance along road
    segment_distance = seg_end.distance_from_start - seg_start.distance_from_start
    distance_along = seg_start.distance_from_start + t * segment_distance

    return RoadCoordinate(proj_lat, proj_lon, distance_along)


class GeoUtils:
    """Geographic utility functions."""

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points in meters."""
        R = 6371000  # Earth radius in meters
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @staticmethod
    def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2 in degrees."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)

        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))

        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)

        return (bearing_deg + 360) % 360


OFFLINE_TILE_DEGREES = 0.25
OFFLINE_TILE_GROUP_DEGREES = 2.0
OFFLINE_CAPNP_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "mapd_repo" / "openpilot-mapd" / "offline.capnp"
OFFLINE_CAPNP_IMPORT_DIR = Path(__file__).resolve().parent / "capnp"


@lru_cache(maxsize=1)
def _load_offline_capnp_schema():
    return capnp.load(str(OFFLINE_CAPNP_SCHEMA_PATH), imports=[str(OFFLINE_CAPNP_IMPORT_DIR)])


class OSMRoadGeometryExtractor:
    """Extracts road geometry from OSM database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection: sqlite3.Connection | None = None

    def connect(self) -> bool:
        """Connect to OSM database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            return True
        except Exception as e:
            cloudlog.error(f"Failed to connect to OSM database: {e}")
            return False

    def disconnect(self):
        """Disconnect from database."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def extract_road_segments_near_position(self,
                                          latitude: float,
                                          longitude: float,
                                          radius_meters: float = 500) -> list[RoadSegment]:
        """Extract road segments within radius of given position."""
        if not self.connection:
            if not self.connect():
                return []

        try:
            # Calculate approximate lat/lon bounds for radius
            lat_offset = radius_meters / 111000  # ~111km per degree
            lon_offset = radius_meters / (111000 * math.cos(math.radians(latitude)))

            # Query for ways within bounds
            query = """
            SELECT DISTINCT w.way_id, w.tags
            FROM ways w
            JOIN way_nodes wn ON w.way_id = wn.way_id
            JOIN nodes n ON wn.node_id = n.node_id
            WHERE n.latitude BETWEEN ? AND ?
            AND n.longitude BETWEEN ? AND ?
            AND w.tags LIKE '%highway%'
            """

            cursor = self.connection.cursor()
            cursor.execute(query, [
                latitude - lat_offset, latitude + lat_offset,
                longitude - lon_offset, longitude + lon_offset
            ])

            road_segments = []
            for row in cursor.fetchall():
                way_id = row['way_id']
                tags = json.loads(row['tags']) if row['tags'] else {}

                # Extract road segment
                segment = self._extract_road_segment(way_id, tags)
                if segment:
                    road_segments.append(segment)

            return road_segments

        except Exception as e:
            cloudlog.error(f"Error extracting road segments: {e}")
            return []

    def _extract_road_segment(self, way_id: int, tags: dict) -> RoadSegment | None:
        """Extract detailed road segment from OSM way."""
        try:
            # Get road classification
            highway_type = tags.get('highway', 'unclassified')
            road_class = self._classify_highway(highway_type)

            # Extract centerline geometry
            centerline = self._extract_centerline(way_id)
            if not centerline:
                return None

            # Extract lane information
            lanes = self._extract_lanes(tags, centerline)

            # Extract barriers
            barriers = self._extract_barriers(way_id)

            # Get level separation (bridge/tunnel)
            level_separation = self._get_level_separation(tags)

            # Get speed limit
            max_speed = self._extract_speed_limit(tags)

            # Calculate road direction at start
            road_direction = self._calculate_road_direction(centerline)

            # Derive human-readable road name if available
            name = tags.get('name') or tags.get('name:en') or tags.get('ref') or ""

            return RoadSegment(
                way_id=way_id,
                name=name,
                road_class=road_class,
                centerline=centerline,
                lanes=lanes,
                barriers=barriers,
                level_separation=level_separation,
                max_speed=max_speed,
                road_direction=road_direction,
            )

        except Exception as e:
            cloudlog.error(f"Error extracting road segment {way_id}: {e}")
            return None

    def _classify_highway(self, highway_type: str) -> RoadClass:
        """Classify OSM highway type."""
        highway_mapping = {
            'motorway': RoadClass.MOTORWAY,
            'motorway_link': RoadClass.MOTORWAY,
            'trunk': RoadClass.TRUNK,
            'trunk_link': RoadClass.TRUNK,
            'primary': RoadClass.PRIMARY,
            'primary_link': RoadClass.PRIMARY,
            'secondary': RoadClass.SECONDARY,
            'secondary_link': RoadClass.SECONDARY,
            'tertiary': RoadClass.TERTIARY,
            'tertiary_link': RoadClass.TERTIARY,
            'residential': RoadClass.RESIDENTIAL,
            'service': RoadClass.SERVICE,
            'living_street': RoadClass.RESIDENTIAL,
        }
        return highway_mapping.get(highway_type, RoadClass.UNCLASSIFIED)

    def _extract_centerline(self, way_id: int) -> list[RoadCoordinate]:
        """Extract road centerline coordinates."""
        try:
            query = """
            SELECT n.latitude, n.longitude, wn.sequence_id
            FROM way_nodes wn
            JOIN nodes n ON wn.node_id = n.node_id
            WHERE wn.way_id = ?
            ORDER BY wn.sequence_id
            """

            cursor = self.connection.cursor()
            cursor.execute(query, [way_id])

            coordinates = []
            total_distance = 0.0
            prev_coord = None

            for row in cursor.fetchall():
                coord = Coordinate(row['latitude'], row['longitude'])

                # Calculate distance from start
                if prev_coord:
                    total_distance += prev_coord.distance_to(coord)

                road_coord = RoadCoordinate(
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    distance_from_start=total_distance
                )
                coordinates.append(road_coord)
                prev_coord = coord

            return coordinates

        except Exception as e:
            cloudlog.error(f"Error extracting centerline for way {way_id}: {e}")
            return []

    def _extract_lanes(self, tags: dict, centerline: list[RoadCoordinate]) -> list[Lane]:
        """Extract lane information from OSM tags."""
        lanes = []

        # Get lane count
        lanes_count = self._parse_lanes_count(tags)

        # Get lane width (use default if not specified)
        lane_width = self._parse_lane_width(tags)

        # Create lanes (simplified - assumes all driving lanes)
        for i in range(lanes_count):
            lane = Lane(
                lane_index=i,
                width=lane_width,
                lane_type=LaneType.DRIVING,
                centerline=[]  # Lane centerlines would require more complex geometry processing
            )
            lanes.append(lane)

        return lanes

    def _parse_lanes_count(self, tags: dict) -> int:
        """Parse lane count from OSM tags."""
        lanes_str = tags.get('lanes')
        if lanes_str:
            try:
                return int(lanes_str)
            except (ValueError, TypeError):
                pass

        # Default based on road type when no lanes tag or invalid value
        highway_type = tags.get('highway', 'residential')
        if highway_type in ['motorway', 'trunk']:
            return 4
        elif highway_type in ['primary', 'secondary']:
            return 2
        else:
            return 1

    def _parse_lane_width(self, tags: dict) -> float:
        """Parse lane width from OSM tags."""
        width_str = tags.get('width', '')
        if width_str:
            try:
                # Parse width (could be "3.5 m" or just "3.5")
                width_val = float(width_str.split()[0])
                return width_val
            except (ValueError, IndexError):
                pass

        # Default lane widths by road type
        highway_type = tags.get('highway', 'residential')
        if highway_type in ['motorway', 'trunk']:
            return 3.7  # Standard highway lane
        elif highway_type in ['primary', 'secondary']:
            return 3.5  # Arterial road
        else:
            return 3.0  # Local road

    def _extract_barriers(self, way_id: int) -> list[Barrier]:
        """Extract barrier information near road."""
        # For now, return empty list
        # In a full implementation, this would query for nearby barrier ways
        return []

    def _get_level_separation(self, tags: dict) -> int:
        """Determine if road is on bridge/tunnel."""
        if 'bridge' in tags and tags['bridge'] not in ['no', 'false']:
            return 1  # Bridge
        elif 'tunnel' in tags and tags['tunnel'] not in ['no', 'false']:
            return -1  # Tunnel
        else:
            return 0  # Ground level

    def _extract_speed_limit(self, tags: dict) -> float:
        """Extract speed limit from OSM tags."""
        maxspeed = tags.get('maxspeed', '')
        if maxspeed:
            try:
                # Parse speed limit (could be "50 mph", "80 km/h", or just "50")
                speed_str = maxspeed.lower()
                if 'mph' in speed_str:
                    speed_val = float(speed_str.replace('mph', '').strip())
                    return speed_val * 0.44704  # Convert mph to m/s
                elif 'km/h' in speed_str or 'kmh' in speed_str:
                    speed_val = float(speed_str.replace('km/h', '').replace('kmh', '').strip())
                    return speed_val / 3.6  # Convert km/h to m/s
                else:
                    # Assume km/h if no unit specified
                    speed_val = float(speed_str)
                    return speed_val / 3.6
            except (ValueError, AttributeError):
                pass

        # Default speed limits by road type
        highway_type = tags.get('highway', 'residential')
        if highway_type in ['motorway']:
            return 33.3  # 120 km/h = 33.3 m/s
        elif highway_type in ['trunk', 'primary']:
            return 22.2  # 80 km/h = 22.2 m/s
        elif highway_type in ['secondary', 'tertiary']:
            return 13.9  # 50 km/h = 13.9 m/s
        else:
            return 8.3   # 30 km/h = 8.3 m/s

    def _calculate_road_direction(self, centerline: list[RoadCoordinate]) -> float:
        """Calculate road bearing at start of segment."""
        if len(centerline) < 2:
            return 0.0

        start = centerline[0].to_coordinate()
        end = centerline[1].to_coordinate()

        # Calculate bearing from start to next point
        lat1 = math.radians(start.latitude)
        lat2 = math.radians(end.latitude)
        lon_diff = math.radians(end.longitude - start.longitude)

        y = math.sin(lon_diff) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon_diff)

        bearing = math.atan2(y, x)
        bearing_degrees = math.degrees(bearing)

        return (bearing_degrees + 360) % 360


class OfflineRoadGeometryExtractor:
    """Extracts road geometry from mapd Cap'n Proto offline tiles."""

    def __init__(self, mapd_root: str):
        self.offline_root = Path(mapd_root) / "offline"
        self.schema = _load_offline_capnp_schema()
        self._logged_missing_tiles: set[str] = set()

    def extract_road_segments_near_position(self,
                                          latitude: float,
                                          longitude: float,
                                          radius_meters: float = 500) -> list[RoadSegment]:
        """Extract road segments within radius of given position."""
        bounds_path = self._bounds_file_for_position(latitude, longitude)
        if not bounds_path.is_file():
            self._log_missing_tile(bounds_path)
            return []

        try:
            offline = self.schema.Offline.from_bytes_packed(bounds_path.read_bytes())
        except Exception as e:
            cloudlog.error(f"Failed to read offline road geometry tile {bounds_path}: {e}")
            return []

        lat_offset = radius_meters / 111000.0
        cos_lat = max(math.cos(math.radians(latitude)), 0.01)
        lon_offset = radius_meters / (111000.0 * cos_lat)
        min_lat = latitude - lat_offset
        max_lat = latitude + lat_offset
        min_lon = longitude - lon_offset
        max_lon = longitude + lon_offset

        road_segments = []
        for index, way in enumerate(offline.ways):
            if not self._way_intersects_bounds(way, min_lat, max_lat, min_lon, max_lon):
                continue

            segment = self._extract_road_segment(bounds_path, index, way)
            if segment:
                road_segments.append(segment)

        return road_segments

    def _log_missing_tile(self, bounds_path: Path) -> None:
        tile_key = str(bounds_path)
        if tile_key not in self._logged_missing_tiles:
            cloudlog.warning(f"No offline OSM tile found for road geometry extraction: {bounds_path}")
            self._logged_missing_tiles.add(tile_key)

    def _bounds_file_for_position(self, latitude: float, longitude: float) -> Path:
        min_lat = math.floor(latitude / OFFLINE_TILE_DEGREES) * OFFLINE_TILE_DEGREES
        min_lon = math.floor(longitude / OFFLINE_TILE_DEGREES) * OFFLINE_TILE_DEGREES
        max_lat = min_lat + OFFLINE_TILE_DEGREES
        max_lon = min_lon + OFFLINE_TILE_DEGREES

        group_lat_dir = int(math.floor(min_lat / OFFLINE_TILE_GROUP_DEGREES) * OFFLINE_TILE_GROUP_DEGREES)
        group_lon_dir = int(math.floor(min_lon / OFFLINE_TILE_GROUP_DEGREES) * OFFLINE_TILE_GROUP_DEGREES)

        return self.offline_root / str(group_lat_dir) / str(group_lon_dir) / (
            f"{min_lat:.6f}_{min_lon:.6f}_{max_lat:.6f}_{max_lon:.6f}"
        )

    @staticmethod
    def _way_intersects_bounds(way, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> bool:
        return not (
            float(way.maxLat) < min_lat or
            float(way.minLat) > max_lat or
            float(way.maxLon) < min_lon or
            float(way.minLon) > max_lon
        )

    def _extract_road_segment(self, bounds_path: Path, index: int, way) -> RoadSegment | None:
        centerline = self._extract_centerline(way)
        if len(centerline) < 2:
            return None

        road_class = self._infer_road_class(way)
        max_speed = self._extract_speed_limit(way)
        lane_width = 3.7 if road_class in (RoadClass.MOTORWAY, RoadClass.TRUNK) else 3.5 if int(way.lanes or 0) >= 2 else 3.0

        lane_count = max(int(way.lanes or 0), 1)
        lanes = [
            Lane(
                lane_index=i,
                width=lane_width,
                lane_type=LaneType.DRIVING,
                centerline=[],
            )
            for i in range(lane_count)
        ]

        way_key = zlib.crc32(f"{bounds_path}:{index}".encode("utf-8")) & 0xFFFFFFFF

        return RoadSegment(
            way_id=way_key,
            name=str(way.name or way.ref or ""),
            road_class=road_class,
            centerline=centerline,
            lanes=lanes,
            barriers=[],
            level_separation=0,
            max_speed=max_speed,
            road_direction=self._calculate_road_direction(centerline),
        )

    @staticmethod
    def _infer_road_class(way) -> RoadClass:
        ref = str(way.ref or "").upper()
        name = str(way.name or "").upper()
        lanes = int(way.lanes or 0)

        if ref.startswith("I-") or "INTERSTATE" in name:
            return RoadClass.MOTORWAY
        if ref.startswith(("US-", "SR-", "CA-", "STATE ROUTE")):
            return RoadClass.TRUNK if lanes >= 2 else RoadClass.PRIMARY
        if bool(way.oneWay) and lanes >= 3:
            return RoadClass.MOTORWAY
        if lanes >= 4:
            return RoadClass.TRUNK
        if lanes >= 2:
            return RoadClass.PRIMARY
        return RoadClass.RESIDENTIAL

    @staticmethod
    def _extract_speed_limit(way) -> float:
        if bool(way.oneWay):
            for speed in (float(way.maxSpeedForward), float(way.maxSpeedBackward), float(way.maxSpeed)):
                if speed > 0:
                    return speed
            return 0.0

        base_speed = float(way.maxSpeed)
        if base_speed > 0:
            return base_speed

        directional_speeds = [float(speed) for speed in (way.maxSpeedForward, way.maxSpeedBackward) if float(speed) > 0]
        return max(directional_speeds, default=0.0)

    @staticmethod
    def _extract_centerline(way) -> list[RoadCoordinate]:
        coordinates = []
        total_distance = 0.0
        prev_coord = None

        for node in way.nodes:
            coord = Coordinate(float(node.latitude), float(node.longitude))
            if prev_coord is not None:
                total_distance += prev_coord.distance_to(coord)

            coordinates.append(RoadCoordinate(
                latitude=coord.latitude,
                longitude=coord.longitude,
                distance_from_start=total_distance,
            ))
            prev_coord = coord

        return coordinates

    @staticmethod
    def _calculate_road_direction(centerline: list[RoadCoordinate]) -> float:
        if len(centerline) < 2:
            return 0.0

        start = centerline[0].to_coordinate()
        end = centerline[1].to_coordinate()
        return GeoUtils.bearing(start.latitude, start.longitude, end.latitude, end.longitude)


class RoadGeometryCache:
    """Caches road geometry data for fast access."""

    def __init__(self, cache_radius: float = 1000.0):
        self.cache_radius = cache_radius
        self.cached_segments: dict[int, RoadSegment] = {}
        self.cache_center: Coordinate | None = None
        self.cache_timestamp = 0.0
        self.cache_ttl = 30.0  # Cache timeout in seconds

    def get_road_segments_near(self,
                              position: Coordinate,
                              extractor) -> list[RoadSegment]:
        """Get road segments near position, using cache when possible."""
        current_time = time.time()

        # Check if cache is valid
        if (self.cache_center and
            self.cache_center.distance_to(position) < self.cache_radius / 2 and
            current_time - self.cache_timestamp < self.cache_ttl):
            return list(self.cached_segments.values())

        # Refresh cache
        segments = extractor.extract_road_segments_near_position(
            position.latitude, position.longitude, self.cache_radius
        )

        self.cached_segments = {seg.way_id: seg for seg in segments}
        self.cache_center = position
        self.cache_timestamp = current_time

        return segments

    def find_current_road_segment(self, position: Coordinate) -> RoadSegment | None:
        """Find the road segment that vehicle is currently on."""
        best_segment = None
        min_distance = float('inf')

        for segment in self.cached_segments.values():
            _, distance = segment.get_closest_point(position)
            if distance < min_distance:
                min_distance = distance
                best_segment = segment

        # Only return if reasonably close to a road (within 20m)
        if min_distance < 20.0:
            return best_segment

        return None
