#!/usr/bin/env python3
"""
Enhanced Road Matcher with Road Geometry Integration

Replaces primitive proximity-based threat detection with actual OSM road geometry
to accurately determine if threats are on the same road and their relative directions.

Solves the core RTI problems:
- Threats across median barriers incorrectly flagged as relevant
- GPS bearing-based direction calculation ignoring road curvature  
- No consideration of level separation (bridges/tunnels)
- Fixed distance thresholds ignoring actual road geometry
"""

import time
from dataclasses import dataclass

from cereal import messaging
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.navd.helpers import Coordinate
from openpilot.sunnypilot.mapd.road_geometry import RoadSegment, RoadCoordinate, BarrierType


@dataclass
class RoadProjection:
    """Result of projecting a GPS position onto a road segment."""
    road_segment: RoadSegment
    projected_point: RoadCoordinate
    distance_to_centerline: float  # meters from GPS position to road centerline
    distance_from_start: float     # meters along road centerline from segment start
    confidence: float              # 0.0-1.0 confidence in projection accuracy


class EnhancedRoadMatcher:
    """
    Road-geometry-aware threat matching using mapd integration.
    
    Provides backward-compatible interface while using actual road data
    instead of primitive proximity thresholds to determine threat relevance.
    """

    def __init__(self):
        """Initialize enhanced road matcher with mapd integration."""
        # Subscribe to mapd road geometry data
        self.mapd_sm = messaging.SubMaster(['liveMapDataSP'], ignore_avg_freq=True)

        # Current road geometry state from mapd
        self.road_geometry_valid = False
        self.current_road_segment: RoadSegment | None = None
        self.nearby_road_segments: list[RoadSegment] = []

        # Fallback thresholds for original proximity logic
        self.proximity_threshold_m = 50.0   # Original 50m threshold
        self.highway_threshold_m = 200.0    # Original 200m highway threshold

        # Road geometry processing parameters
        self.max_distance_to_road_m = 100.0    # Maximum distance to consider for road projection
        self.same_road_threshold_m = 20.0      # Maximum centerline distance for same road
        self.level_separation_tolerance = 0    # No tolerance for level differences

        # Performance tracking
        self.last_update_time = 0.0
        self.geometry_update_interval = 0.1  # Update road geometry every 100ms

    def is_same_road(self, ego_lat: float, ego_lon: float,
                    threat_lat: float, threat_lon: float,
                    ego_speed_ms: float,
                    road_data: dict | None = None) -> bool:
        """
        Enhanced same-road detection using road geometry when available.
        
        Args:
            ego_lat, ego_lon: Ego vehicle GPS position
            threat_lat, threat_lon: Threat GPS position  
            ego_speed_ms: Ego vehicle speed (used for fallback thresholds)
            road_data: Optional road geometry data (for testing)
            
        Returns:
            True if threat is on the same road as ego vehicle
        """
        # Update road geometry from mapd (unless test data provided)
        if road_data is None:
            road_data = self._get_current_road_geometry()

        # Use road geometry if available and valid
        if road_data and road_data.get('road_geometry_valid', False):
            return self._is_same_road_geometry_based(
                Coordinate(ego_lat, ego_lon),
                Coordinate(threat_lat, threat_lon),
                road_data
            )
        else:
            # Fallback to original proximity-based logic
            return self._is_same_road_proximity_based(
                ego_lat, ego_lon, threat_lat, threat_lon, ego_speed_ms
            )

    def get_direction_relative_to_ego(self, ego_lat: float, ego_lon: float,
                                     threat_lat: float, threat_lon: float,
                                     ego_heading: float = 0.0,
                                     road_data: dict | None = None) -> str:
        """
        Enhanced direction calculation using road geometry when available.
        
        Args:
            ego_lat, ego_lon: Ego vehicle GPS position
            threat_lat, threat_lon: Threat GPS position
            ego_heading: Ego vehicle heading (used for fallback)
            road_data: Optional road geometry data (for testing)
            
        Returns:
            Direction string: 'ahead', 'behind', 'left', 'right'
        """
        # Update road geometry from mapd (unless test data provided)
        if road_data is None:
            road_data = self._get_current_road_geometry()

        # Use road geometry if available and valid
        if road_data and road_data.get('road_geometry_valid', False):
            return self._get_direction_road_aware(
                Coordinate(ego_lat, ego_lon),
                Coordinate(threat_lat, threat_lon),
                road_data
            )
        else:
            # Fallback to original GPS bearing-based logic
            return self._get_direction_bearing_based(
                ego_lat, ego_lon, threat_lat, threat_lon, ego_heading
            )

    def _get_current_road_geometry(self) -> dict:
        """Get current road geometry data from mapd."""
        current_time = time.time()

        # Rate limit updates to avoid excessive processing
        if current_time - self.last_update_time < self.geometry_update_interval:
            return self._create_road_data_dict()

        self.last_update_time = current_time

        try:
            # Update from mapd
            if self._update_road_geometry_from_mapd():
                cloudlog.debug("RTI updated road geometry from mapd")

        except Exception as e:
            cloudlog.warning(f"RTI failed to update road geometry: {e}")
            self.road_geometry_valid = False

        return self._create_road_data_dict()

    def _update_road_geometry_from_mapd(self) -> bool:
        """Update road geometry from mapd liveMapDataSP message."""
        self.mapd_sm.update(0)  # Non-blocking update

        if not self.mapd_sm.updated['liveMapDataSP']:
            return False

        try:
            mapd_msg = self.mapd_sm['liveMapDataSP']

            # Update validity flag
            self.road_geometry_valid = mapd_msg.roadGeometryValid

            if not self.road_geometry_valid:
                self.current_road_segment = None
                self.nearby_road_segments = []
                return False

            # Extract current road segment
            if hasattr(mapd_msg, 'currentRoadSegment'):
                self.current_road_segment = self._parse_road_segment_from_msg(
                    mapd_msg.currentRoadSegment
                )

            # Extract nearby road segments
            self.nearby_road_segments = []
            if hasattr(mapd_msg, 'nearbyRoadSegments'):
                for segment_msg in mapd_msg.nearbyRoadSegments:
                    road_segment = self._parse_road_segment_from_msg(segment_msg)
                    if road_segment:
                        self.nearby_road_segments.append(road_segment)

            return True

        except Exception as e:
            cloudlog.error(f"RTI failed to parse mapd road geometry: {e}")
            self.road_geometry_valid = False
            return False

    def _parse_road_segment_from_msg(self, segment_msg) -> RoadSegment | None:
        """Parse RoadSegment from capnp message."""
        try:
            from openpilot.sunnypilot.mapd.road_geometry import (
                RoadClass, LaneType, Lane, Barrier
            )

            # Parse centerline
            centerline = []
            for coord_msg in segment_msg.centerline:
                centerline.append(RoadCoordinate(
                    latitude=coord_msg.latitude,
                    longitude=coord_msg.longitude,
                    distance_from_start=coord_msg.distanceFromStart
                ))

            if not centerline:
                return None

            # Parse lanes
            lanes = []
            for lane_msg in segment_msg.lanes:
                lane_centerline = []
                for coord_msg in lane_msg.centerline:
                    lane_centerline.append(RoadCoordinate(
                        latitude=coord_msg.latitude,
                        longitude=coord_msg.longitude,
                        distance_from_start=coord_msg.distanceFromStart
                    ))

                # Map lane type from message
                lane_type_map = {
                    0: LaneType.DRIVING,
                    1: LaneType.BUS,
                    2: LaneType.BICYCLE,
                    3: LaneType.PARKING,
                    4: LaneType.SHOULDER,
                    5: LaneType.MEDIAN
                }
                lane_type = lane_type_map.get(lane_msg.type, LaneType.DRIVING)

                lanes.append(Lane(
                    lane_index=lane_msg.laneIndex,
                    width=lane_msg.width,
                    lane_type=lane_type,
                    centerline=lane_centerline
                ))

            # Parse barriers
            barriers = []
            for barrier_msg in segment_msg.barriers:
                barrier_coords = []
                for coord_msg in barrier_msg.coordinates:
                    barrier_coords.append(RoadCoordinate(
                        latitude=coord_msg.latitude,
                        longitude=coord_msg.longitude,
                        distance_from_start=coord_msg.distanceFromStart
                    ))

                # Map barrier type from message
                barrier_type_map = {
                    0: BarrierType.MEDIAN,
                    1: BarrierType.GUARDRAIL,
                    2: BarrierType.WALL,
                    3: BarrierType.FENCE,
                    4: BarrierType.CURB
                }
                barrier_type = barrier_type_map.get(barrier_msg.type, BarrierType.MEDIAN)

                barriers.append(Barrier(
                    barrier_type=barrier_type,
                    coordinates=barrier_coords
                ))

            # Map road class from message
            road_class_map = {
                0: RoadClass.MOTORWAY,
                1: RoadClass.TRUNK,
                2: RoadClass.PRIMARY,
                3: RoadClass.SECONDARY,
                4: RoadClass.TERTIARY,
                5: RoadClass.RESIDENTIAL,
                6: RoadClass.SERVICE,
                7: RoadClass.UNCLASSIFIED
            }
            from openpilot.sunnypilot.mapd.road_geometry import RoadClass
            road_class = road_class_map.get(segment_msg.roadClass, RoadClass.UNCLASSIFIED)

            return RoadSegment(
                way_id=segment_msg.wayId,
                road_class=road_class,
                centerline=centerline,
                lanes=lanes,
                barriers=barriers,
                level_separation=segment_msg.levelSeparation,
                max_speed=segment_msg.maxSpeed,
                road_direction=segment_msg.roadDirection
            )

        except Exception as e:
            cloudlog.warning(f"RTI failed to parse road segment: {e}")
            return None

    def _create_road_data_dict(self) -> dict:
        """Create standardized road data dictionary."""
        return {
            'road_geometry_valid': self.road_geometry_valid,
            'current_road_segment': self.current_road_segment,
            'nearby_road_segments': self.nearby_road_segments
        }

    def _is_same_road_geometry_based(self, ego_pos: Coordinate, threat_pos: Coordinate,
                                   road_data: dict) -> bool:
        """
        Determine if threat is on same road using actual road geometry.
        
        Algorithm:
        1. Project both positions onto nearby road segments
        2. Check if both project to the same road segment (way_id)
        3. Consider level separation (bridges/tunnels)
        4. Check for barriers between projected positions
        5. Use road-specific distance thresholds
        """
        try:
            # Get road segments
            current_segment = road_data.get('current_road_segment')
            nearby_segments = road_data.get('nearby_road_segments', [])

            if not current_segment and not nearby_segments:
                return False

            # Build list of segments to check
            segments_to_check = []
            if current_segment:
                segments_to_check.append(current_segment)
            for segment in nearby_segments:
                if not current_segment or segment.way_id != current_segment.way_id:
                    segments_to_check.append(segment)

            # Project ego position onto road segments
            ego_projection = self._find_best_road_projection(ego_pos, segments_to_check)
            if not ego_projection:
                return False

            # Project threat position onto road segments
            threat_projection = self._find_best_road_projection(threat_pos, segments_to_check)
            if not threat_projection:
                return False

            # Check if both project to the same road segment
            if ego_projection.road_segment.way_id != threat_projection.road_segment.way_id:
                return False

            # Check level separation (bridges/tunnels)
            if ego_projection.road_segment.level_separation != threat_projection.road_segment.level_separation:
                return False

            # Check for barriers between positions
            if self._has_barrier_between_positions(ego_projection, threat_projection):
                return False

            # Check distance to centerline (both positions should be reasonably close to road)
            max_distance = self._get_road_width_threshold(ego_projection.road_segment)
            if (ego_projection.distance_to_centerline > max_distance or
                threat_projection.distance_to_centerline > max_distance):
                return False

            return True

        except Exception as e:
            cloudlog.warning(f"RTI geometry-based road matching failed: {e}")
            return False

    def _find_best_road_projection(self, position: Coordinate,
                                 road_segments: list[RoadSegment]) -> RoadProjection | None:
        """Find the best projection of position onto available road segments."""
        # print(f"RTI DEBUG: _find_best_road_projection called with position ({position.latitude:.6f}, {position.longitude:.6f})")
        # print(f"RTI DEBUG: Checking {len(road_segments)} road segments")

        best_projection = None
        min_distance = float('inf')

        # Performance optimization: limit segments and use early exit
        # For large road networks (>10 segments), be very aggressive about limiting search
        max_segments = 2 if len(road_segments) > 10 else min(3, len(road_segments))
        segments_to_check = road_segments[:max_segments]

        for i, segment in enumerate(segments_to_check):
            # print(f"RTI DEBUG: Checking segment {i} (way_id={segment.way_id})")
            try:
                # Quick distance check: approximate distance to first centerline point
                if segment.centerline:
                    first_point = segment.centerline[0].to_coordinate()
                    approx_distance = position.distance_to(first_point)  # Already in meters
                    # print(f"RTI DEBUG: Approx distance to first point: {approx_distance:.2f}m")
                    # Be more aggressive with distance filtering for large road networks
                    distance_threshold = 200.0 if len(road_segments) > 10 else 500.0
                    if approx_distance > distance_threshold:
                        # print(f"RTI DEBUG: Skipping segment - distance {approx_distance:.2f}m > 500m")
                        continue

                # print("RTI DEBUG: Calling _project_position_to_road_segment")
                projection = self._project_position_to_road_segment(position, segment)
                # print(f"RTI DEBUG: Projection result: {projection is not None}")

                if projection:
                    # print(f"RTI DEBUG: Projection distance_to_centerline: {projection.distance_to_centerline:.3f}m")
                    if projection.distance_to_centerline < min_distance:
                        min_distance = projection.distance_to_centerline
                        best_projection = projection
                        # print(f"RTI DEBUG: New best projection with distance {min_distance:.3f}m")

                        # Early exit if very close
                        if min_distance < 1.0:  # Tighter threshold for test precision
                            # print(f"RTI DEBUG: Early exit - distance {min_distance:.3f}m < 1.0m")
                            break

            except Exception as e:
                # print(f"RTI DEBUG: Exception in segment {i}: {e}")
                cloudlog.debug(f"RTI projection failed for segment {segment.way_id}: {e}")
                continue

        # Only return projection if reasonably close to a road
        # print(f"RTI DEBUG: Final best_projection: {best_projection is not None}")
        # if best_projection:
        #     print(f"RTI DEBUG: Final distance: {best_projection.distance_to_centerline:.3f}m, max allowed: {self.max_distance_to_road_m}m")

        if best_projection and best_projection.distance_to_centerline <= self.max_distance_to_road_m:
            # print("RTI DEBUG: Returning best projection")
            return best_projection

        # print("RTI DEBUG: Returning None - no suitable projection found")
        return None

    def _project_position_to_road_segment(self, position: Coordinate,
                                        road_segment: RoadSegment) -> RoadProjection | None:
        """Project GPS position onto road segment centerline."""
        try:
            closest_point, distance_to_centerline = road_segment.get_closest_point(position)

            # Calculate confidence based on distance and road characteristics
            confidence = self._calculate_projection_confidence(
                distance_to_centerline, road_segment
            )

            return RoadProjection(
                road_segment=road_segment,
                projected_point=closest_point,
                distance_to_centerline=distance_to_centerline,
                distance_from_start=closest_point.distance_from_start,
                confidence=confidence
            )

        except Exception as e:
            cloudlog.debug(f"RTI failed to project to road segment {road_segment.way_id}: {e}")
            return None

    def _calculate_projection_confidence(self, distance_to_centerline: float,
                                       road_segment: RoadSegment) -> float:
        """Calculate confidence in road projection based on distance and road type."""
        # Base confidence on distance to centerline
        road_width = self._get_road_width_threshold(road_segment)

        if distance_to_centerline <= road_width / 2:
            # Very close to road - high confidence
            return 1.0 - (distance_to_centerline / (road_width / 2)) * 0.2
        elif distance_to_centerline <= road_width:
            # Within road width - medium confidence
            return 0.8 - ((distance_to_centerline - road_width / 2) / (road_width / 2)) * 0.3
        else:
            # Outside road width - low confidence
            max_distance = min(self.max_distance_to_road_m, road_width * 3)
            return max(0.1, 0.5 - (distance_to_centerline - road_width) / max_distance * 0.4)

    def _get_road_width_threshold(self, road_segment: RoadSegment) -> float:
        """Get appropriate distance threshold based on road characteristics."""
        # Calculate actual road width from lanes
        if road_segment.lanes:
            total_width = sum(lane.width for lane in road_segment.lanes)
            # Add margins for shoulders, medians, etc.
            return total_width + 10.0  # 5m margin on each side

        # Fallback to road class-based estimates
        from openpilot.sunnypilot.mapd.road_geometry import RoadClass
        width_by_class = {
            RoadClass.MOTORWAY: 40.0,    # Wide highways with multiple lanes
            RoadClass.TRUNK: 30.0,       # Major roads
            RoadClass.PRIMARY: 25.0,     # Primary roads
            RoadClass.SECONDARY: 20.0,   # Secondary roads
            RoadClass.TERTIARY: 15.0,    # Tertiary roads
            RoadClass.RESIDENTIAL: 12.0, # Residential streets
            RoadClass.SERVICE: 10.0,     # Service roads
            RoadClass.UNCLASSIFIED: 15.0 # Default
        }

        return width_by_class.get(road_segment.road_class, 15.0)

    def _has_barrier_between_positions(self, ego_projection: RoadProjection,
                                     threat_projection: RoadProjection) -> bool:
        """Check if there's a barrier between ego and threat positions."""
        road_segment = ego_projection.road_segment

        # Check each barrier in the road segment
        for barrier in road_segment.barriers:
            if barrier.barrier_type == BarrierType.MEDIAN:
                # For median barriers, check if positions are on opposite sides
                if self._are_positions_across_median(
                    ego_projection, threat_projection, barrier
                ):
                    return True

        return False

    def _are_positions_across_median(self, ego_projection: RoadProjection,
                                   threat_projection: RoadProjection,
                                   median_barrier: 'Barrier') -> bool:
        """Check if ego and threat are on opposite sides of a median barrier."""
        try:
            # Check if both positions are on the same road segment with median barrier
            if ego_projection.road_segment.way_id != threat_projection.road_segment.way_id:
                # Different road segments - check if they have the same barrier
                ego_has_median = any(b.barrier_type == BarrierType.MEDIAN for b in ego_projection.road_segment.barriers)
                threat_has_median = any(b.barrier_type == BarrierType.MEDIAN for b in threat_projection.road_segment.barriers)

                if ego_has_median and threat_has_median:
                    # Both segments have median barriers - likely opposing directions
                    return True

            # If on same road segment, check if positions are significantly offset from centerline
            MIN_OFFSET_FOR_MEDIAN_CHECK = 1.0  # meters (reduced threshold)

            ego_offset = ego_projection.distance_to_centerline
            threat_offset = threat_projection.distance_to_centerline

            # For median separation, both positions should be reasonably offset from centerline
            if ego_offset > MIN_OFFSET_FOR_MEDIAN_CHECK and threat_offset > MIN_OFFSET_FOR_MEDIAN_CHECK:
                # Both are offset from centerline and there's a median barrier
                if median_barrier and median_barrier.barrier_type == BarrierType.MEDIAN:
                    return True

            return False

        except Exception as e:
            cloudlog.debug(f"RTI median barrier check failed: {e}")
            return False

    def _get_direction_road_aware(self, ego_pos: Coordinate, threat_pos: Coordinate,
                                road_data: dict) -> str:
        """
        Determine threat direction using road centerline geometry.
        
        Algorithm:
        1. Project both positions onto road centerlines
        2. Use distance_from_start along centerline to determine ahead/behind
        3. Use lateral offset from centerline for left/right
        4. Consider road curvature and actual geometry
        """
        try:
            # Get road segments
            current_segment = road_data.get('current_road_segment')
            nearby_segments = road_data.get('nearby_road_segments', [])

            if not current_segment and not nearby_segments:
                return 'ahead'  # Safe default

            segments_to_check = []
            if current_segment:
                segments_to_check.append(current_segment)
            segments_to_check.extend(nearby_segments)

            # Project both positions
            ego_projection = self._find_best_road_projection(ego_pos, segments_to_check)
            threat_projection = self._find_best_road_projection(threat_pos, segments_to_check)

            if not ego_projection or not threat_projection:
                return 'ahead'  # Safe default

            # If on different road segments, determine relative position
            if ego_projection.road_segment.way_id != threat_projection.road_segment.way_id:
                return self._get_direction_between_different_roads(
                    ego_projection, threat_projection
                )

            # Same road segment - use distance along centerline
            ego_distance = ego_projection.distance_from_start
            threat_distance = threat_projection.distance_from_start

            # Determine ahead/behind based on road direction
            if threat_distance > ego_distance + 5.0:  # 5m tolerance
                return 'ahead'
            elif threat_distance < ego_distance - 5.0:  # 5m tolerance
                return 'behind'
            else:
                # Very close along road - check lateral offset for left/right
                return self._get_lateral_direction(ego_projection, threat_projection)

        except Exception as e:
            cloudlog.warning(f"RTI road-aware direction calculation failed: {e}")
            return 'ahead'  # Safe default

    def _get_direction_between_different_roads(self, ego_projection: RoadProjection,
                                             threat_projection: RoadProjection) -> str:
        """Determine direction when ego and threat are on different road segments."""
        # For different roads, use GPS bearing as fallback
        # This could be enhanced with road network topology in the future
        ego_coord = ego_projection.projected_point.to_coordinate()
        threat_coord = threat_projection.projected_point.to_coordinate()

        return self._get_direction_bearing_based(
            ego_coord.latitude, ego_coord.longitude,
            threat_coord.latitude, threat_coord.longitude,
            ego_projection.road_segment.road_direction
        )

    def _get_lateral_direction(self, ego_projection: RoadProjection,
                             threat_projection: RoadProjection) -> str:
        """Determine left/right direction when positions are at similar road distances."""
        # This is a simplified implementation
        # A full implementation would use cross products with road direction vectors

        # For now, use GPS coordinates relative to road direction
        road_direction = ego_projection.road_segment.road_direction

        ego_coord = ego_projection.projected_point.to_coordinate()
        threat_coord = threat_projection.projected_point.to_coordinate()

        return self._get_direction_bearing_based(
            ego_coord.latitude, ego_coord.longitude,
            threat_coord.latitude, threat_coord.longitude,
            road_direction
        )

    def _is_same_road_proximity_based(self, ego_lat: float, ego_lon: float,
                                    threat_lat: float, threat_lon: float,
                                    ego_speed_ms: float) -> bool:
        """Original proximity-based same-road detection (fallback)."""
        from openpilot.sunnypilot.rtid.threat_detector import GeoUtils

        distance = GeoUtils.haversine_distance(ego_lat, ego_lon, threat_lat, threat_lon)

        # Use larger threshold for high-speed roads (likely highways)
        threshold = (self.highway_threshold_m if ego_speed_ms > 25  # >90 km/h
                    else self.proximity_threshold_m)

        return distance <= threshold

    def _get_direction_bearing_based(self, ego_lat: float, ego_lon: float,
                                   threat_lat: float, threat_lon: float,
                                   ego_heading: float) -> str:
        """Original GPS bearing-based direction calculation (fallback)."""
        from openpilot.sunnypilot.rtid.threat_detector import GeoUtils

        threat_bearing = GeoUtils.bearing(ego_lat, ego_lon, threat_lat, threat_lon)

        # Relative bearing (threat bearing - ego heading)
        relative_bearing = (threat_bearing - ego_heading + 360) % 360

        # Classify direction
        if relative_bearing <= 45 or relative_bearing >= 315:
            return 'ahead'
        elif 45 < relative_bearing <= 135:
            return 'right'
        elif 135 < relative_bearing <= 225:
            return 'behind'
        else:
            return 'left'
