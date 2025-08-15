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
        spatial_grid = {}
        threat_grid_coords = {}  # Cache grid coordinates
        threat_lon_scales = {}   # Cache longitude scales

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
        # Simple road matching based on distance thresholds
        # In a full implementation, this would integrate with OSM data
        self.road_proximity_threshold_m = 50  # Assume same road if within 50m
        self.highway_proximity_threshold_m = 200  # Highways are wider

    def is_same_road(self, ego_lat: float, ego_lon: float,
                    threat_lat: float, threat_lon: float,
                    ego_speed_ms: float) -> bool:
        """
        Determine if threat is on the same road as ego vehicle.
        Uses distance-based heuristics with speed-aware thresholds.
        """
        distance = GeoUtils.haversine_distance(ego_lat, ego_lon, threat_lat, threat_lon)

        # Use larger threshold for high-speed roads (likely highways)
        threshold = (self.highway_proximity_threshold_m if ego_speed_ms > 25  # >90 km/h
                    else self.road_proximity_threshold_m)

        return distance <= threshold

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

        # Get forward slowdown range (when to start slowing for threats ahead)
        forward_range = params.get("RTIForwardSlowdownRange")
        self.ahead_distance_threshold_m = float(forward_range) if forward_range else 1207  # Default 0.75 miles

        # Get resume speed distance (when to resume normal speed after passing)
        resume_distance = params.get("RTIResumeSpeedDistance")
        self.behind_distance_threshold_m = float(resume_distance) if resume_distance else 805  # Default 0.5 miles

        # Get speed reduction settings
        self.speed_reduction_mode = params.get("RTISpeedReductionMode")
        self.speed_reduction_mode = self.speed_reduction_mode.decode('utf-8') if self.speed_reduction_mode else "posted"

        speed_reduction = params.get("RTISpeedReduction")
        speed_reduction_kmh = float(speed_reduction) if speed_reduction else 16  # Default 10 mph
        self.speed_reduction_ms = speed_reduction_kmh / 3.6  # Convert km/h to m/s

        self.default_speed_limit_ms = 25  # 55 mph default when unknown

    def calculate_recommendation(self, threats: list[ProcessedThreat],
                               current_speed_ms: float,
                               current_location: tuple[float, float],
                               v_cruise_ms: float = None) -> tuple[float, bool]:
        """
        Calculate speed recommendation based on processed threats.
        
        Args:
            threats: List of processed threats
            current_speed_ms: Current vehicle speed in m/s
            current_location: Current GPS location
            v_cruise_ms: Driver's set cruise speed in m/s (for no-limit scenarios)
        
        Returns:
            Tuple of (recommended_speed_ms, threat_ahead_bool)
        """
        relevant_threats = []

        for threat in threats:
            # Only consider threats on same road
            if not threat.on_same_road:
                continue

            # Apply distance thresholds based on direction
            max_distance = (self.ahead_distance_threshold_m if threat.direction == 'ahead'
                          else self.behind_distance_threshold_m if threat.direction == 'behind'
                          else 0)  # Don't consider left/right threats for speed control

            if threat.distance <= max_distance and threat.direction in ['ahead', 'behind']:
                relevant_threats.append(threat)

        if not relevant_threats:
            return 0.0, False  # No recommendation

        # Find closest ahead threat for speed recommendation
        ahead_threats = [t for t in relevant_threats if t.direction == 'ahead']

        if not ahead_threats:
            return 0.0, False

        closest_threat = min(ahead_threats, key=lambda t: t.distance)

        # CRITICAL SAFETY: RTI only operates when cruise control is enabled
        # If cruise is not set, RTI must NOT make any speed recommendations
        # This prevents dangerous accelerations when resuming cruise
        if not v_cruise_ms or v_cruise_ms <= 0:
            # No cruise speed set - RTI is inactive
            return 0.0, False
        
        # Determine target speed based on threat type and current conditions
        if self.speed_reduction_mode == "posted":
            # Use posted speed limit (if available)
            if closest_threat.speed_limit_ms > 0:
                target_speed = closest_threat.speed_limit_ms
            else:
                # No posted speed limit - reduce by 20% of driver's set maximum
                target_speed = v_cruise_ms * 0.8  # 20% reduction from set cruise
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

        return target_speed, True


class ThreatDetector:
    """Main threat detection and processing engine."""

    def __init__(self):
        self.clusterer = ThreatClusterer()

        # Use enhanced road matcher with road geometry integration
        from .enhanced_road_matcher import EnhancedRoadMatcher
        self.road_matcher = EnhancedRoadMatcher()

        self.speed_engine = SpeedRecommendationEngine()

        # Load user-configurable params
        from openpilot.common.params import Params
        params = Params()

        # Get detection radius (for HUD display of all threats within radius)
        detection_radius = params.get("RTIDetectionRadius")
        self.detection_radius_m = float(detection_radius) if detection_radius else 3218  # Default 2 miles

        # Get threat filter settings
        # 0 = All, 1 = Police Only, 2 = Speed Cameras Only, 3 = Hazards Only, 4 = Custom
        threat_filter = params.get("RTIThreatFilter")
        self.threat_filter = int(threat_filter) if threat_filter else 0

        # Performance tracking
        self.last_process_time = 0

    def process_threats(self, traffic_data: list[WazeAlert] | None,
                       current_location: tuple[float, float],
                       current_speed: float,
                       timestamp: int,
                       v_cruise: float = None) -> RTIState:
        """
        Main threat processing pipeline.
        
        Args:
            traffic_data: Raw traffic alerts from API
            current_location: (lat, lon) of ego vehicle
            current_speed: Current speed in m/s
            timestamp: Current timestamp in nanoseconds
            v_cruise: Driver's set cruise speed in m/s (optional)
            
        Returns:
            RTIState for publishing
        """
        process_start = time.time()

        try:
            # Initialize empty state
            processed_threats = []
            recommended_speed = 0.0
            threat_ahead = False

            if traffic_data:
                # Step 1: Apply threat filter
                filtered_threats = self._apply_threat_filter(traffic_data)

                # Step 2: Deduplicate threats
                deduplicated_threats = self.clusterer.deduplicate_threats(filtered_threats)

                # Step 3: Process each threat
                for threat in deduplicated_threats:
                    processed_threat = self._process_single_threat(
                        threat, current_location, current_speed
                    )
                    if processed_threat:
                        processed_threats.append(processed_threat)

                # Step 3: Generate speed recommendation
                recommended_speed, threat_ahead = self.speed_engine.calculate_recommendation(
                    processed_threats, current_speed, current_location, v_cruise
                )

            # Step 4: Safety validation
            if recommended_speed > 0:
                # Ensure recommendation is within safe bounds (never accelerate toward threats)
                if not (0 <= recommended_speed <= current_speed):
                    cloudlog.warning(f"RTI unsafe speed recommendation {recommended_speed:.1f} m/s "
                                   f"for current speed {current_speed:.1f} m/s - ignoring")
                    recommended_speed = 0.0
                    threat_ahead = False

            # Sort threats by distance for HUD display
            processed_threats.sort(key=lambda t: t.distance)

            process_time = (time.time() - process_start) * 1000  # Convert to ms
            if process_time > 15:  # Warn if over performance budget
                cloudlog.warning(f"RTI threat processing took {process_time:.1f}ms")

            return RTIState(
                timestamp=timestamp,
                threat_ahead=threat_ahead,
                threat_distance_m=processed_threats[0].distance if processed_threats else 0.0,
                recommended_speed=recommended_speed,
                source='rti',
                api_status='unknown',  # Will be set by caller
                threats=processed_threats[:5]  # Limit to 5 for HUD
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

    def _process_single_threat(self, threat: WazeAlert,
                             current_location: tuple[float, float],
                             current_speed: float) -> ProcessedThreat | None:
        """Process a single threat for relevance and direction."""
        try:
            ego_lat, ego_lon = current_location

            # Calculate distance
            distance = GeoUtils.haversine_distance(
                ego_lat, ego_lon, threat.latitude, threat.longitude
            )

            # Skip threats that are outside detection radius
            if distance > self.detection_radius_m:
                return None

            # Determine if on same road
            on_same_road = self.road_matcher.is_same_road(
                ego_lat, ego_lon, threat.latitude, threat.longitude, current_speed
            )

            # Determine direction relative to ego
            direction = self.road_matcher.get_direction_relative_to_ego(
                ego_lat, ego_lon, threat.latitude, threat.longitude
            )

            # Note: Real Waze API doesn't provide speed_limit in alerts
            # Future: integrate with SLC (Speed Limit Controller) for actual speed limits
            # For now, use conservative default for police/speed trap locations
            speed_limit_ms = 0.0
            if threat.speed_limit:
                # Handle legacy test data or future integration
                speed_limit_ms = threat.speed_limit / 3.6
            else:
                # Use conservative speed limit estimates based on alert type
                if threat.type in ['police', 'policeHiding', 'speedTrap']:
                    # Conservative estimate for enforcement locations
                    speed_limit_ms = self.speed_engine.default_speed_limit_ms
                else:
                    # For other alerts, don't make speed recommendations
                    speed_limit_ms = 0.0

            return ProcessedThreat(
                id=threat.id,
                type=threat.type,
                latitude=threat.latitude,
                longitude=threat.longitude,
                distance=distance,
                direction=direction,
                confidence=threat.confidence,
                speed_limit_ms=speed_limit_ms,
                on_same_road=on_same_road
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
            threats=[]
        )
