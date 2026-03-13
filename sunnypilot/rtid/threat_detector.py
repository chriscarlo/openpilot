#!/usr/bin/env python3
"""
Threat Detector for RTI System

Processes traffic alerts to determine relevance to current route and make
speed recommendations. Handles:
- GPS drift and road matching accuracy
- Threat deduplication (DBSCAN clustering)
- Direction determination (ahead/behind on route)
- Safety validation (speed recommendation constraints)
"""

import math
import time
from dataclasses import dataclass

from openpilot.common.swaglog import cloudlog
from .waze_api_client import WazeAlert


@dataclass
class RTIState:
    """RTI system state for publishing."""
    timestamp: int
    threat_ahead: bool
    threat_distance_m: float
    recommended_speed: float  # m/s
    source: str
    api_status: str
    threats: list['ProcessedThreat']
    active_threat_id: str | None = None  # ID of threat causing speed recommendation


@dataclass
class ProcessedThreat:
    """Processed threat with direction and relevance."""
    id: str
    type: str
    latitude: float
    longitude: float
    distance: float  # meters from ego
    direction: str   # ahead, behind, left, right
    confidence: float
    speed_limit_ms: float  # m/s
    on_same_road: bool
    road_match_confidence: float = 0.0  # Confidence in street name match (0.0-1.0)


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


class ThreatClusterer:
    """Handles threat deduplication using optimized spatial indexing."""

    @staticmethod
    def deduplicate_threats(threats: list[WazeAlert],
                          cluster_radius_m: float = 100) -> list[WazeAlert]:
        """
        Deduplicate threats using optimized greedy clustering to match original behavior.

        Uses spatial hashing for O(n) average case complexity while maintaining
        identical results to the original algorithm for accuracy tests.
        """
        if not threats:
            return []

        if len(threats) == 1:
            return threats

        # Performance optimization: pre-calculate grid coordinates
        grid_size = cluster_radius_m * 1.5
        spatial_grid: dict[tuple[int, int], list[int]] = {}
        threat_grid_coords: dict[int, tuple[int, int]] = {}  # Cache grid coordinates
        threat_lon_scales: dict[int, float] = {}   # Cache longitude scales

        # Step 1: Hash threats into grid cells - O(n) with caching
        for i, threat in enumerate(threats):
            # Use average latitude for longitude scaling to avoid repeated math.cos calls
            lon_scale = 111320 * math.cos(math.radians(threat.latitude))
            threat_lon_scales[i] = lon_scale  # Cache for later use
            lat_scale = 110540

            grid_x = int(threat.longitude * lon_scale / grid_size)
            grid_y = int(threat.latitude * lat_scale / grid_size)
            grid_key = (grid_x, grid_y)

            # Cache coordinates for later use
            threat_grid_coords[i] = grid_key

            if grid_key not in spatial_grid:
                spatial_grid[grid_key] = []
            spatial_grid[grid_key].append(i)

        # Step 2: Replicate original greedy algorithm behavior exactly
        clusters = []
        used = set()

        for i, threat in enumerate(threats):
            if i in used:
                continue

            cluster = [threat]
            used.add(i)

            # Get grid coordinates for this threat
            grid_x, grid_y = threat_grid_coords[i]

            # Check all other threats in nearby grid cells (optimization over O(n²))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell_key = (grid_x + dx, grid_y + dy)
                    if cell_key not in spatial_grid:
                        continue

                    for j in spatial_grid[cell_key]:
                        if j in used or j == i:
                            continue

                        other_threat = threats[j]

                        # Use precise haversine distance (matches original)
                        distance = GeoUtils.haversine_distance(
                            threat.latitude, threat.longitude,
                            other_threat.latitude, other_threat.longitude
                        )

                        if distance <= cluster_radius_m:
                            cluster.append(other_threat)
                            used.add(j)

            clusters.append(cluster)

        # Step 3: Return best threat from each cluster - O(n)
        deduplicated = []
        for cluster in clusters:
            # Select best threat by confidence (matches original algorithm)
            best_threat = max(cluster, key=lambda t: t.confidence)
            deduplicated.append(best_threat)

        return deduplicated


class RoadMatcher:
    """Handles road matching and same-road determination."""

    def __init__(self):
        # Import street name matcher
        from .street_name_matcher import StreetNameMatcher
        self.street_matcher = StreetNameMatcher()

        # Distance thresholds for fallback when street names unavailable
        self.road_proximity_threshold_m = 50  # Assume same road if within 50m
        self.highway_proximity_threshold_m = 200  # Highways are wider
        self.current_segment_match_threshold_m = 35.0
        self.contiguous_segment_match_threshold_m = 20.0
        self.contiguous_endpoint_threshold_m = 40.0
        self.contiguous_direction_threshold_deg = 25.0
        self.contiguous_speed_delta_ms = 8.0
        self.competing_segment_advantage_m = 8.0
        self.route_direction_conflict_threshold_deg = 80.0
        # If names are unavailable, use heading + bearing to reject likely side-street threats.
        self.heading_gate_min_speed_ms = 8.0
        self.side_street_bearing_threshold_deg = 60.0
        self.side_street_min_distance_m = 25.0

    @staticmethod
    def _normalize_180(angle: float) -> float:
        """Normalize angle to [-180, 180] range."""
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def is_same_road(self, ego_lat: float, ego_lon: float,
                    threat_lat: float, threat_lon: float,
                    ego_speed_ms: float,
                    ego_street: str | None = None,
                    threat_street: str | None = None,
                    ego_heading_deg: float | None = None,
                    current_road_segment = None,
                    nearby_road_segments = None) -> tuple[bool, float]:
        """
        Determine if threat is on the same road as ego vehicle with confidence.

        Uses street name matching when available, falls back to distance heuristics.

        Args:
            ego_lat, ego_lon: Ego vehicle GPS coordinates
            threat_lat, threat_lon: Threat GPS coordinates
            ego_speed_ms: Current vehicle speed in m/s
            ego_street: Current street name from map data (optional)
            threat_street: Threat street name from Waze API (optional)
            ego_heading_deg: Current vehicle heading in degrees (optional)

        Returns:
            Tuple of (is_same_road: bool, confidence: float)
            confidence ranges from 0.0 to 1.0
        """
        norm_ego_street = self.street_matcher.normalize_street_name(ego_street)
        norm_threat_street = self.street_matcher.normalize_street_name(threat_street)

        # First try street name matching if both names available
        if norm_ego_street and norm_threat_street:
            # Use strict direction matching for highways or expressways on either name
            strict_direction = (
                self.street_matcher.is_highway_or_expressway(ego_street)
                or self.street_matcher.is_highway_or_expressway(threat_street)
            )
            ego_route = self.street_matcher._extract_route_designator(norm_ego_street)
            threat_route = self.street_matcher._extract_route_designator(norm_threat_street)
            match_result = self.street_matcher.match_street_names(
                ego_street, threat_street, strict_direction
            )

            if (
                strict_direction
                and match_result.is_match
                and ego_route
                and threat_route
                and self.street_matcher._route_designators_equivalent(ego_route, threat_route)
                and self._route_alias_direction_conflicts(
                    ego_route,
                    threat_route,
                    current_road_segment=current_road_segment,
                    ego_heading_deg=ego_heading_deg,
                )
            ):
                cloudlog.debug(
                    "RTI route direction conflict: "
                    + f"ego='{ego_street}', threat='{threat_street}'"
                )
                return False, 0.05

            if match_result.is_match and match_result.confidence >= 0.7:
                # High confidence match
                cloudlog.debug(f"RTI street match: {match_result.reason}")
                return match_result.is_match, match_result.confidence
            elif match_result.is_match and match_result.confidence >= 0.5:
                # Medium confidence - also check distance
                distance = GeoUtils.haversine_distance(ego_lat, ego_lon, threat_lat, threat_lon)
                threshold = (self.highway_proximity_threshold_m if ego_speed_ms > 25
                           else self.road_proximity_threshold_m)

                if match_result.is_match and distance <= threshold * 1.5:
                    cloudlog.debug(f"RTI street+distance match: {match_result.reason}, distance={distance:.0f}m")
                    # Boost confidence slightly when distance also matches
                    return True, min(match_result.confidence + 0.1, 1.0)
                elif match_result.is_match:
                    # Street matches but distance doesn't confirm
                    return match_result.is_match, match_result.confidence * 0.8

            geometry_match, geometry_confidence = self._geometry_override_same_road(
                threat_lat,
                threat_lon,
                current_road_segment,
                nearby_road_segments,
            )
            if geometry_match:
                cloudlog.debug(
                    "RTI geometry override accepted street mismatch: "
                    + f"{match_result.reason}, confidence={geometry_confidence:.2f}"
                )
                return True, geometry_confidence

            # If both street names are present and don't match, do not trust pure proximity fallback.
            cloudlog.debug(f"RTI street mismatch: {match_result.reason}")
            return False, 0.05

        # Fall back to distance-based heuristics
        distance = GeoUtils.haversine_distance(ego_lat, ego_lon, threat_lat, threat_lon)

        # Use larger threshold for high-speed roads (likely highways)
        threshold = (self.highway_proximity_threshold_m if ego_speed_ms > 25  # >90 km/h
                    else self.road_proximity_threshold_m)

        is_same = distance <= threshold

        # Optional heading-gated fallback: when we only have proximity, reject likely side-street threats.
        if (
            is_same and
            ego_heading_deg is not None and
            ego_speed_ms >= self.heading_gate_min_speed_ms and
            distance >= self.side_street_min_distance_m
        ):
            threat_bearing = GeoUtils.bearing(ego_lat, ego_lon, threat_lat, threat_lon)
            relative_bearing = abs(self._normalize_180(threat_bearing - ego_heading_deg))
            if self.side_street_bearing_threshold_deg <= relative_bearing <= (180.0 - self.side_street_bearing_threshold_deg):
                cloudlog.debug(
                    "RTI heading-gated reject: likely side-street threat "
                    + f"(distance={distance:.1f}m, rel_bearing={relative_bearing:.1f}deg)"
                )
                return False, 0.15

        # Calculate confidence based on distance proximity
        if is_same:
            # Confidence decreases with distance
            confidence = max(0.3, 0.5 * (1.0 - distance / threshold))
        else:
            confidence = 0.2  # Low confidence when using distance alone

        if ego_street or threat_street:
            cloudlog.debug(
                f"RTI fallback to distance: ego='{ego_street}', threat='{threat_street}', "
                + f"distance={int(distance)}m, threshold={int(threshold)}m, same={is_same}, confidence={confidence:.2f}"
            )

        return is_same, confidence

    @staticmethod
    def _segment_centerline_distance_m(lat: float, lon: float, segment) -> float:
        """Return minimum distance from a point to any centerline coordinate in a segment."""
        centerline = getattr(segment, 'centerline', None) or []
        min_distance = float('inf')
        for coord in centerline:
            coord_lat = getattr(coord, 'latitude', None)
            coord_lon = getattr(coord, 'longitude', None)
            if coord_lat is None or coord_lon is None:
                continue
            distance = GeoUtils.haversine_distance(lat, lon, coord_lat, coord_lon)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    @staticmethod
    def _segment_endpoints(segment) -> list:
        centerline = getattr(segment, 'centerline', None) or []
        if not centerline:
            return []
        if len(centerline) == 1:
            return [centerline[0]]
        return [centerline[0], centerline[-1]]

    @staticmethod
    def _direction_delta_deg(direction_a: float, direction_b: float) -> float:
        diff = abs(float(direction_a) - float(direction_b)) % 360.0
        return min(diff, 360.0 - diff)

    @staticmethod
    def _direction_token_to_bearing(direction_token: str | None) -> float | None:
        if direction_token is None:
            return None

        mapping = {
            'n': 0.0,
            'nb': 0.0,
            'ne': 45.0,
            'e': 90.0,
            'eb': 90.0,
            'se': 135.0,
            's': 180.0,
            'sb': 180.0,
            'sw': 225.0,
            'w': 270.0,
            'wb': 270.0,
            'nw': 315.0,
        }
        return mapping.get(direction_token)

    def _route_alias_direction_conflicts(self, ego_route, threat_route,
                                         current_road_segment, ego_heading_deg: float | None) -> bool:
        """
        Reject route-alias matches when only one side carries a direction token and the
        current road geometry or heading clearly points the opposite way.
        """
        if ego_route is None or threat_route is None:
            return False

        ego_direction = ego_route[2]
        threat_direction = threat_route[2]
        if ego_direction and threat_direction:
            return False

        expected_direction = self._direction_token_to_bearing(ego_direction or threat_direction)
        if expected_direction is None:
            return False

        observed_direction = None
        if current_road_segment is not None:
            try:
                observed_direction = float(getattr(current_road_segment, 'roadDirection'))
            except Exception:
                observed_direction = None

        if observed_direction is None and ego_heading_deg is not None:
            observed_direction = float(ego_heading_deg)

        if observed_direction is None:
            return False

        return self._direction_delta_deg(observed_direction, expected_direction) > self.route_direction_conflict_threshold_deg

    def _segments_are_contiguous(self, current_segment, candidate_segment) -> bool:
        """Heuristic for same-corridor adjacent ways when names disagree."""
        if current_segment is None or candidate_segment is None:
            return False

        if int(getattr(current_segment, 'wayId', 0) or 0) == int(getattr(candidate_segment, 'wayId', 0) or 0):
            return True

        if getattr(current_segment, 'levelSeparation', 0) != getattr(candidate_segment, 'levelSeparation', 0):
            return False

        if getattr(current_segment, 'roadClass', None) != getattr(candidate_segment, 'roadClass', None):
            return False

        direction_delta = self._direction_delta_deg(
            getattr(current_segment, 'roadDirection', 0.0),
            getattr(candidate_segment, 'roadDirection', 0.0),
        )
        if direction_delta > self.contiguous_direction_threshold_deg:
            return False

        speed_delta = abs(
            float(getattr(current_segment, 'maxSpeed', 0.0) or 0.0) -
            float(getattr(candidate_segment, 'maxSpeed', 0.0) or 0.0)
        )
        if speed_delta > self.contiguous_speed_delta_ms:
            return False

        current_endpoints = self._segment_endpoints(current_segment)
        candidate_endpoints = self._segment_endpoints(candidate_segment)
        if not current_endpoints or not candidate_endpoints:
            return False

        min_endpoint_distance = min(
            GeoUtils.haversine_distance(a.latitude, a.longitude, b.latitude, b.longitude)
            for a in current_endpoints
            for b in candidate_endpoints
        )
        return min_endpoint_distance <= self.contiguous_endpoint_threshold_m

    def _geometry_override_same_road(self, threat_lat: float, threat_lon: float,
                                     current_road_segment, nearby_road_segments) -> tuple[bool, float]:
        """
        Resolve naming mismatches using mapd geometry.

        Only override a street mismatch when the threat lies tightly on the current
        segment or a clearly contiguous same-corridor segment from nearbyRoadSegments.
        """
        if current_road_segment is None:
            return False, 0.0

        current_distance = self._segment_centerline_distance_m(threat_lat, threat_lon, current_road_segment)

        best_segment = None
        best_distance = float('inf')
        seen_way_ids = {int(getattr(current_road_segment, 'wayId', 0) or 0)}

        for segment in nearby_road_segments or []:
            way_id = int(getattr(segment, 'wayId', 0) or 0)
            if way_id in seen_way_ids:
                continue
            seen_way_ids.add(way_id)

            distance = self._segment_centerline_distance_m(threat_lat, threat_lon, segment)
            if distance < best_distance:
                best_distance = distance
                best_segment = segment

        if current_distance <= self.current_segment_match_threshold_m:
            competing_segment_is_better = (
                best_segment is not None
                and best_distance + self.competing_segment_advantage_m < current_distance
                and not self._segments_are_contiguous(current_road_segment, best_segment)
            )
            if not competing_segment_is_better:
                return True, 0.82

        if best_segment is None or best_distance > self.contiguous_segment_match_threshold_m:
            return False, 0.0

        if self._segments_are_contiguous(current_road_segment, best_segment):
            return True, 0.72

        return False, 0.0

    def get_direction_relative_to_ego(self, ego_lat: float, ego_lon: float,
                                     threat_lat: float, threat_lon: float,
                                     ego_heading: float = 0) -> str:
        """
        Determine threat direction relative to ego vehicle.
        For now, uses simple bearing-based logic.
        """
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


class SpeedRecommendationEngine:
    """Generates speed recommendations based on threat analysis."""

    def __init__(self):
        # Load user-configurable thresholds from Params
        from openpilot.common.params import Params
        params = Params()

        # Detection radius used for situational (left/right) threats in recommendations
        # Keep in sync with ThreatDetector default (3 miles = ~4828 m)
        detection_radius = params.get("RTIDetectionRadius")
        if detection_radius:
            try:
                self.detection_radius_m = float(detection_radius)
            except (ValueError, TypeError):
                self.detection_radius_m = 4828  # Default 3 miles
        else:
            self.detection_radius_m = 4828  # Default 3 miles

        # Get forward slowdown range (when to start slowing for threats ahead)
        # Stored in meters in params, default 0.75 miles = 1207 meters
        forward_range = params.get("RTIForwardSlowdownRange")
        if forward_range:
            try:
                self.ahead_distance_threshold_m = float(forward_range)
            except (ValueError, TypeError):
                self.ahead_distance_threshold_m = 1207  # Default 0.75 miles
        else:
            self.ahead_distance_threshold_m = 1207  # Default 0.75 miles

        # Get resume speed distance (when to resume normal speed after passing)
        # Stored in meters in params, default 0.75 miles = 1207 meters
        resume_distance = params.get("RTIResumeSpeedDistance")
        if resume_distance:
            try:
                self.behind_distance_threshold_m = float(resume_distance)
            except (ValueError, TypeError):
                self.behind_distance_threshold_m = 1207  # Default 0.75 miles
        else:
            self.behind_distance_threshold_m = 1207  # Default 0.75 miles

        # Get speed reduction settings
        self.speed_reduction_mode = params.get("RTISpeedReductionMode")
        if self.speed_reduction_mode:
            self.speed_reduction_mode = (
                self.speed_reduction_mode.decode('utf-8')
                if isinstance(self.speed_reduction_mode, bytes)
                else str(self.speed_reduction_mode)
            )
        else:
            self.speed_reduction_mode = "posted"

        # Get custom speed reduction (stored in km/h, default 16 km/h = ~10 mph)
        speed_reduction = params.get("RTISpeedReduction")
        if speed_reduction:
            try:
                speed_reduction_kmh = float(speed_reduction)
            except (ValueError, TypeError):
                speed_reduction_kmh = 16  # Default 10 mph
        else:
            speed_reduction_kmh = 16  # Default 10 mph
        self.speed_reduction_ms = speed_reduction_kmh / 3.6  # Convert km/h to m/s

    def calculate_recommendation(self, threats: list[ProcessedThreat],
                               current_speed_ms: float,
                               current_location: tuple[float, float],
                               v_cruise_ms: float = None) -> tuple[float, bool, str | None]:
        """
        Calculate speed recommendation based on processed threats.

        Args:
            threats: List of processed threats
            current_speed_ms: Current vehicle speed in m/s
            current_location: Current GPS location
            v_cruise_ms: Driver's set cruise speed in m/s (for no-limit scenarios)

        Returns:
            Tuple of (recommended_speed_ms, threat_ahead_bool, active_threat_id)
        """
        relevant_threats = []

        for threat in threats:
            # Only consider threats on same road
            if not threat.on_same_road:
                continue

            # Apply distance thresholds based on direction
            max_distance = (self.ahead_distance_threshold_m if threat.direction == 'ahead'
                          else self.behind_distance_threshold_m if threat.direction == 'behind'
                          else self.detection_radius_m)  # Use detection radius for left/right threats (situational awareness)

            if threat.distance <= max_distance and threat.direction in ['ahead', 'behind', 'left', 'right']:
                relevant_threats.append(threat)

        if not relevant_threats:
            return 0.0, False, None  # No recommendation

        # Find closest ahead threat for speed recommendation
        ahead_threats = [t for t in relevant_threats if t.direction == 'ahead']

        if not ahead_threats:
            return 0.0, False, None

        closest_threat = min(ahead_threats, key=lambda t: t.distance)

        # CRITICAL SAFETY: RTI only operates when cruise control is enabled
        # If cruise is not set, RTI must NOT make any speed recommendations
        # This prevents dangerous accelerations when resuming cruise
        if not v_cruise_ms or v_cruise_ms <= 0:
            # No cruise speed set - RTI is inactive
            return 0.0, False, None

        # Determine target speed based on threat type and current conditions
        if self.speed_reduction_mode == "posted":
            # Use posted speed limit (if available)
            if closest_threat.speed_limit_ms > 0:
                target_speed = closest_threat.speed_limit_ms
            else:
                # No speed limit = no speed recommendation
                # Threat will still appear on HUD for visual awareness
                return 0.0, False, None
        else:
            # Custom mode: reduce by fixed amount from cruise speed
            target_speed = v_cruise_ms - self.speed_reduction_ms
            # But never go below a reasonable minimum (e.g. 10 m/s = 22 mph)
            target_speed = max(target_speed, 10.0)

        # Safety validation: recommended speed must never exceed current speed
        # This ensures we're always recommending deceleration or maintaining speed
        # Never recommend acceleration toward a threat
        target_speed = min(target_speed, current_speed_ms)

        # Ensure non-negative speed
        target_speed = max(0.0, target_speed)

        # Return the speed recommendation and the ID of the threat causing it
        return target_speed, True, closest_threat.id


class ThreatDetector:
    """Main threat detection and processing engine."""

    def __init__(self):
        self.clusterer = ThreatClusterer()

        # Use street-name-based road matcher by default
        # This leverages Waze + mapd street names for same-road determination
        self.road_matcher = RoadMatcher()

        self.speed_engine = SpeedRecommendationEngine()

        # Load user-configurable params
        from openpilot.common.params import Params
        params = Params()

        # Get detection radius (for HUD display of all threats within radius)
        # Stored in meters in params, default 2 miles = 3218 meters
        detection_radius = params.get("RTIDetectionRadius")
        if detection_radius:
            try:
                self.detection_radius_m = float(detection_radius)
            except (ValueError, TypeError):
                self.detection_radius_m = 4828  # Default 3 miles
        else:
            self.detection_radius_m = 4828  # Default 3 miles

        # Get threat filter settings
        # 0 = All, 1 = Police Only, 2 = Speed Cameras Only, 3 = Hazards Only, 4 = Custom
        threat_filter = params.get("RTIThreatFilter")
        if threat_filter:
            try:
                self.threat_filter = int(threat_filter)
            except (ValueError, TypeError):
                self.threat_filter = 0  # Default to all threats
        else:
            self.threat_filter = 0  # Default to all threats

        # Performance tracking
        self.last_process_time = 0

        # Second-pass collapse for duplicate same-road hazards shown on HUD.
        # This is intentionally separate from raw alert clustering so we can use
        # onSameRoad + direction heuristics from processed threats.
        self.duplicate_collapse_radius_m = self._read_optional_float_param(
            params, "RTIDuplicateCollapseRadius", 110.0
        )
        self.police_collapse_radius_m = self._read_optional_float_param(
            params, "RTIPoliceCollapseRadius", 140.0
        )

    def process_threats(self, traffic_data: list[WazeAlert] | None,
                       current_location: tuple[float, float],
                       current_speed: float,
                       timestamp: int,
                       v_cruise: float = None,
                       current_heading_deg: float | None = None,
                       posted_speed_limit: float = 0.0,
                       current_road_name: str | None = None,
                       current_road_segment = None,
                       nearby_road_segments = None) -> RTIState:
        """
        Main threat processing pipeline.

        Args:
            traffic_data: Raw traffic alerts from API
            current_location: (lat, lon) of ego vehicle
            current_speed: Current speed in m/s
            timestamp: Current timestamp in nanoseconds
            v_cruise: Driver's set cruise speed in m/s (optional)
            current_heading_deg: Current vehicle heading in degrees (optional)
            posted_speed_limit: Posted speed limit in m/s from map data (optional)
            current_road_name: Current road name from map data (optional)

        Returns:
            RTIState for publishing
        """
        process_start = time.monotonic()

        try:
            # Initialize empty state
            processed_threats = []
            recommended_speed = 0.0
            threat_ahead = False
            active_threat_id = None

            if traffic_data:
                # Step 1: Apply threat filter
                filtered_threats = self._apply_threat_filter(traffic_data)

                # Step 2: Process each threat with actual speed limit and road name
                for threat in filtered_threats:
                    processed_threat = self._process_single_threat(
                        threat, current_location, current_speed, current_heading_deg,
                        posted_speed_limit, current_road_name,
                        current_road_segment, nearby_road_segments
                    )
                    if processed_threat:
                        processed_threats.append(processed_threat)

                # Step 3: Collapse duplicate same-road hazards for HUD and control logic.
                processed_threats = self._collapse_duplicate_threats(processed_threats)

                # Step 4: Generate speed recommendation
                recommended_speed, threat_ahead, active_threat_id = self.speed_engine.calculate_recommendation(
                    processed_threats, current_speed, current_location, v_cruise
                )

            # Step 5: Safety validation
            if recommended_speed > 0:
                # Ensure recommendation is within safe bounds (never accelerate toward threats)
                if not (0 <= recommended_speed <= current_speed):
                    cloudlog.warning(
                        f"RTI unsafe speed recommendation {recommended_speed:.1f} m/s for current speed {current_speed:.1f} m/s - ignoring"
                    )
                    recommended_speed = 0.0
                    threat_ahead = False
                    active_threat_id = None  # Clear active threat if recommendation is unsafe

            # Sort threats by distance for HUD display
            processed_threats.sort(key=lambda t: t.distance)

            process_time = (time.monotonic() - process_start) * 1000  # Convert to ms
            if process_time > 15:  # Warn if over performance budget
                cloudlog.warning(f"RTI threat processing took {process_time:.1f}ms")

            return RTIState(
                timestamp=timestamp,
                threat_ahead=threat_ahead,
                threat_distance_m=processed_threats[0].distance if processed_threats else 0.0,
                recommended_speed=recommended_speed,
                source='rti',
                api_status='unknown',  # Will be set by caller
                threats=processed_threats[:5],  # Limit to 5 for HUD
                active_threat_id=active_threat_id
            )

        except Exception as e:
            cloudlog.error(f"RTI threat processing failed: {e}")
            return self._create_safe_state(timestamp)

    def _apply_threat_filter(self, threats: list[WazeAlert]) -> list[WazeAlert]:
        """Apply user-configured threat filter."""
        if self.threat_filter == 0:  # All threats
            return threats
        elif self.threat_filter == 1:  # Police only
            return [t for t in threats if t.type in ['police', 'policeHiding']]
        elif self.threat_filter == 2:  # Speed cameras only
            return [t for t in threats if t.type in ['speedTrap', 'speedCamera']]
        elif self.threat_filter == 3:  # Hazards only
            return [t for t in threats if t.type in ['hazard', 'shoulderHazard', 'roadHazard']]
        else:  # Custom (currently same as all)
            return threats

    @staticmethod
    def _threat_family(threat_type: str) -> str:
        """Normalize related threat types into a merge family."""
        if threat_type in ('police', 'policeHiding'):
            return 'police'
        return threat_type

    @staticmethod
    def _read_optional_float_param(params, key: str, default: float) -> float:
        """
        Read a float param safely.

        Unknown keys in Params can raise, so treat missing/invalid as default.
        """
        try:
            raw = params.get(key)
        except Exception:
            return default

        if not raw:
            return default

        try:
            return float(raw)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _direction_bucket(direction: str) -> str:
        """
        Bucket directions for duplicate collapsing.

        Treat ahead/behind as one longitudinal lane bucket so nearby duplicate
        pins from the same cop still collapse when one report lags behind.
        """
        if direction in ('ahead', 'behind'):
            return 'longitudinal'
        return direction

    def _collapse_merge_radius(self, threat: ProcessedThreat) -> float:
        """Return merge radius in meters for a processed threat."""
        if self._threat_family(threat.type) == 'police':
            return self.police_collapse_radius_m
        return self.duplicate_collapse_radius_m

    def _can_collapse_pair(self, a: ProcessedThreat, b: ProcessedThreat) -> bool:
        """Check whether two processed threats should be collapsed into one."""
        if not a.on_same_road or not b.on_same_road:
            return False

        if self._threat_family(a.type) != self._threat_family(b.type):
            return False

        if self._direction_bucket(a.direction) != self._direction_bucket(b.direction):
            return False

        return True

    def _collapse_duplicate_threats(self, threats: list[ProcessedThreat]) -> list[ProcessedThreat]:
        """
        Collapse near-duplicate processed threats (especially police clusters)
        into one representative threat for cleaner HUD output.
        """
        if len(threats) <= 1:
            return threats

        collapsed: list[ProcessedThreat] = []
        used: set[int] = set()

        for i, threat in enumerate(threats):
            if i in used:
                continue

            cluster_indices = {i}
            frontier = [i]
            used.add(i)

            # Transitive merge: A~B and B~C implies A/B/C are one displayed hazard.
            while frontier:
                current_idx = frontier.pop()
                current = threats[current_idx]

                for j, candidate in enumerate(threats):
                    if j in cluster_indices or j in used:
                        continue

                    if not self._can_collapse_pair(current, candidate):
                        continue

                    merge_radius = max(
                        self._collapse_merge_radius(current),
                        self._collapse_merge_radius(candidate),
                    )
                    distance_between = GeoUtils.haversine_distance(
                        current.latitude, current.longitude,
                        candidate.latitude, candidate.longitude,
                    )

                    if distance_between <= merge_radius:
                        cluster_indices.add(j)
                        used.add(j)
                        frontier.append(j)

            if len(cluster_indices) == 1:
                collapsed.append(threat)
                continue

            cluster = [threats[idx] for idx in cluster_indices]

            # Prefer highest confidence; break ties toward the closest threat.
            representative = max(cluster, key=lambda t: (t.confidence, -t.distance))
            collapsed.append(representative)

        return collapsed

    def _process_single_threat(self, threat: WazeAlert,
                             current_location: tuple[float, float],
                             current_speed: float,
                             current_heading_deg: float | None = None,
                             posted_speed_limit: float = 0.0,
                             current_road_name: str | None = None,
                             current_road_segment = None,
                             nearby_road_segments = None) -> ProcessedThreat | None:
        """Process a single threat for relevance and direction.

        Args:
            threat: Raw threat from Waze API
            current_location: (lat, lon) of ego vehicle
            current_speed: Current speed in m/s
            current_heading_deg: Current vehicle heading in degrees (optional)
            posted_speed_limit: Posted speed limit in m/s from map data (optional)
            current_road_name: Current road name from map data (optional)

        Returns:
            ProcessedThreat if relevant, None otherwise
        """
        try:
            ego_lat, ego_lon = current_location

            # Calculate distance
            distance = GeoUtils.haversine_distance(
                ego_lat, ego_lon, threat.latitude, threat.longitude
            )

            # Skip threats that are outside detection radius
            if distance > self.detection_radius_m:
                return None

            # Determine if on same road using street names when available
            on_same_road, road_match_confidence = self.road_matcher.is_same_road(
                ego_lat, ego_lon, threat.latitude, threat.longitude, current_speed,
                ego_street=current_road_name, threat_street=threat.street,
                ego_heading_deg=current_heading_deg,
                current_road_segment=current_road_segment,
                nearby_road_segments=nearby_road_segments,
            )

            # Determine direction relative to ego
            direction = self.road_matcher.get_direction_relative_to_ego(
                ego_lat, ego_lon, threat.latitude, threat.longitude,
                ego_heading=(current_heading_deg if current_heading_deg is not None else 0.0)
            )

            # Determine speed limit for this threat location
            speed_limit_ms = 0.0
            if threat.speed_limit:
                # Handle legacy test data or future integration
                speed_limit_ms = threat.speed_limit / 3.6
            else:
                # Apply same logic to all threat types - use posted speed if available
                if posted_speed_limit > 0:
                    # Use actual posted speed limit from map data
                    speed_limit_ms = posted_speed_limit
                    cloudlog.debug(f"RTI: Using posted speed limit {speed_limit_ms:.1f} m/s for {threat.type}")
                else:
                    # No speed limit available - visual alert only, no slowing
                    speed_limit_ms = 0.0
                    cloudlog.debug(f"RTI: No posted speed limit for {threat.type}, visual alert only")

            return ProcessedThreat(
                id=threat.id,
                type=threat.type,
                latitude=threat.latitude,
                longitude=threat.longitude,
                distance=distance,
                direction=direction,
                confidence=threat.confidence,
                speed_limit_ms=speed_limit_ms,
                on_same_road=on_same_road,
                road_match_confidence=road_match_confidence
            )

        except Exception as e:
            cloudlog.warning(f"RTI failed to process threat {threat.id}: {e}")
            return None

    def _create_safe_state(self, timestamp: int) -> RTIState:
        """Create safe offline state when processing fails."""
        return RTIState(
            timestamp=timestamp,
            threat_ahead=False,
            threat_distance_m=0.0,
            recommended_speed=0.0,
            source='rti',
            api_status='offline',
            threats=[],
            active_threat_id=None
        )
