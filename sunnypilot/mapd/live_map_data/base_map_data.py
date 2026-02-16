"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time
import math
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

from cereal import messaging
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.navd.helpers import Coordinate, coordinate_from_param

if TYPE_CHECKING:
  pass


class BaseMapData(ABC):
  def __init__(self):
    self.params = Params()

    self.gps_location_service = get_gps_location_service(self.params)
    self.sm = messaging.SubMaster(['livePose', 'carControl'] + [self.gps_location_service])
    self.pm = messaging.PubMaster(['liveMapDataSP'])

    self.last_position = coordinate_from_param("LastGPSPosition", self.params)
    self.last_altitude = None
    self.last_bearing = 0.0

  @abstractmethod
  def update_location(self) -> None:
    pass

  @abstractmethod
  def get_current_speed_limit(self) -> float:
    pass

  @abstractmethod
  def get_next_speed_limit_and_distance(self) -> tuple[float, float]:
    pass

  @abstractmethod
  def get_current_road_name(self) -> str:
    pass

  # Road geometry abstract methods (optional - provide defaults for backward compatibility)
  def get_road_geometry_valid(self) -> bool:
    """Check if road geometry data is available and valid."""
    return False

  def get_current_road_segment(self):
    """Get the road segment vehicle is currently on."""
    return None

  def get_nearby_road_segments(self) -> list:
    """Get all road segments within detection radius."""
    return []

  def get_current_location(self) -> None:
    gps = self.sm[self.gps_location_service]

    # ignore the message if the fix is invalid
    gps_ok = self.sm.updated[self.gps_location_service] or (time.monotonic() - self.sm.logMonoTime[self.gps_location_service] / 1e9) > 2.0
    if not gps_ok and self.sm['livePose'].inputsOK:
      return None

    # livePose has these data, but aren't on cereal
    self.last_position = Coordinate(gps.latitude, gps.longitude)
    self.last_altitude = gps.altitude
    self.last_bearing = self.extract_bearing_deg(gps)

  @staticmethod
  def extract_bearing_deg(gps) -> float:
    # Prefer explicit bearing fields when available.
    for key in ("bearingDeg", "bearing"):
      try:
        val = float(getattr(gps, key))
        if math.isfinite(val):
          return val
      except Exception:
        pass

    # Fallback: derive heading from N/E velocity components when present.
    try:
      v_ned = getattr(gps, "vNED", None)
      if v_ned is not None:
        if hasattr(v_ned, "vN") and hasattr(v_ned, "vE"):
          v_n = float(v_ned.vN)
          v_e = float(v_ned.vE)
        else:
          vals = [float(v) for v in v_ned]
          v_n = vals[0] if len(vals) > 0 else 0.0
          v_e = vals[1] if len(vals) > 1 else 0.0
        if math.hypot(v_n, v_e) > 0.1:
          return (math.degrees(math.atan2(v_e, v_n)) + 360.0) % 360.0
    except Exception:
      pass

    return 0.0

  def publish(self) -> None:
    speed_limit = self.get_current_speed_limit()
    next_speed_limit, next_speed_limit_distance = self.get_next_speed_limit_and_distance()

    mapd_sp_send = messaging.new_message('liveMapDataSP')
    # Follow chubbs wiring: valid only when GPS/livePose checks pass
    mapd_sp_send.valid = self.sm.all_checks(service_list=[self.gps_location_service, 'livePose'])
    live_map_data = mapd_sp_send.liveMapDataSP

    # Existing fields
    live_map_data.speedLimitValid = bool(speed_limit > 0)
    live_map_data.speedLimit = speed_limit
    live_map_data.speedLimitAheadValid = bool(next_speed_limit > 0)
    live_map_data.speedLimitAhead = next_speed_limit
    live_map_data.speedLimitAheadDistance = next_speed_limit_distance
    live_map_data.roadName = self.get_current_road_name()

    # New road geometry fields
    try:
      live_map_data.roadGeometryValid = self.get_road_geometry_valid()

      # Populate current road segment
      current_segment = self.get_current_road_segment()
      if current_segment:
        self._populate_road_segment(live_map_data.currentRoadSegment, current_segment)

      # Populate nearby road segments (limit to avoid message size issues)
      nearby_segments = self.get_nearby_road_segments()[:10]  # Limit to 10 segments
      for segment in nearby_segments:
        segment_msg = live_map_data.nearbyRoadSegments.add()
        self._populate_road_segment(segment_msg, segment)

    except Exception:
      # Gracefully handle road geometry errors - don't break existing functionality
      live_map_data.roadGeometryValid = False

    self.pm.send('liveMapDataSP', mapd_sp_send)

  def _populate_road_segment(self, segment_msg, road_segment):
    """Populate capnp road segment message from RoadSegment object."""
    try:
      # Import road geometry types
      from openpilot.sunnypilot.mapd.road_geometry import RoadClass, LaneType, BarrierType

      # Basic road segment info
      segment_msg.wayId = road_segment.way_id
      segment_msg.maxSpeed = road_segment.max_speed
      segment_msg.roadDirection = road_segment.road_direction
      segment_msg.levelSeparation = road_segment.level_separation

      # Map road class
      road_class_mapping = {
        RoadClass.MOTORWAY: 0,    # motorway
        RoadClass.TRUNK: 1,       # trunk
        RoadClass.PRIMARY: 2,     # primary
        RoadClass.SECONDARY: 3,   # secondary
        RoadClass.TERTIARY: 4,    # tertiary
        RoadClass.RESIDENTIAL: 5, # residential
        RoadClass.SERVICE: 6,     # service
        RoadClass.UNCLASSIFIED: 7 # unclassified
      }
      segment_msg.roadClass = road_class_mapping.get(road_segment.road_class, 7)

      # Populate centerline (limit to avoid message size issues)
      centerline_coords = road_segment.centerline[:50]  # Limit to 50 points
      for coord in centerline_coords:
        coord_msg = segment_msg.centerline.add()
        coord_msg.latitude = coord.latitude
        coord_msg.longitude = coord.longitude
        coord_msg.distanceFromStart = coord.distance_from_start

      # Populate lanes (limit to avoid message size issues)
      for lane in road_segment.lanes[:10]:  # Limit to 10 lanes
        lane_msg = segment_msg.lanes.add()
        lane_msg.laneIndex = lane.lane_index
        lane_msg.width = lane.width

        # Map lane type
        lane_type_mapping = {
          LaneType.DRIVING: 0,   # driving
          LaneType.BUS: 1,       # bus
          LaneType.BICYCLE: 2,   # bicycle
          LaneType.PARKING: 3,   # parking
          LaneType.SHOULDER: 4,  # shoulder
          LaneType.MEDIAN: 5     # median
        }
        lane_msg.type = lane_type_mapping.get(lane.lane_type, 0)

        # Add lane centerline coordinates (limited)
        for coord in lane.centerline[:20]:  # Limit to 20 points per lane
          coord_msg = lane_msg.centerline.add()
          coord_msg.latitude = coord.latitude
          coord_msg.longitude = coord.longitude
          coord_msg.distanceFromStart = coord.distance_from_start

      # Populate barriers (limit to avoid message size issues)
      for barrier in road_segment.barriers[:5]:  # Limit to 5 barriers
        barrier_msg = segment_msg.barriers.add()

        # Map barrier type
        barrier_type_mapping = {
          BarrierType.MEDIAN: 0,     # median
          BarrierType.GUARDRAIL: 1,  # guardrail
          BarrierType.WALL: 2,       # wall
          BarrierType.FENCE: 3,      # fence
          BarrierType.CURB: 4        # curb
        }
        barrier_msg.type = barrier_type_mapping.get(barrier.barrier_type, 0)

        # Add barrier coordinates (limited)
        for coord in barrier.coordinates[:20]:  # Limit to 20 points per barrier
          coord_msg = barrier_msg.coordinates.add()
          coord_msg.latitude = coord.latitude
          coord_msg.longitude = coord.longitude
          coord_msg.distanceFromStart = coord.distance_from_start

    except Exception as e:
      # Gracefully handle errors - don't break the messaging system
      cloudlog.error(f"Error populating road segment message: {e}")
      # Set basic fallback values
      if hasattr(segment_msg, 'wayId'):
        segment_msg.wayId = 0
      if hasattr(segment_msg, 'maxSpeed'):
        segment_msg.maxSpeed = 0.0

  def tick(self) -> None:
    self.sm.update()
    self.get_current_location()
    self.update_location()
    self.publish()
