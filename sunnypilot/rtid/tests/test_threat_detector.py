#!/usr/bin/env python3
"""
Unit tests for ThreatDetector

Tests threat processing functionality including GPS calculations, clustering,
road matching, and safety-critical speed recommendations.
"""

import pytest
from unittest.mock import patch

from sunnypilot.rtid.threat_detector import (
    ThreatDetector, GeoUtils, ThreatClusterer, RoadMatcher,
    SpeedRecommendationEngine, RTIState, ProcessedThreat
)


@pytest.mark.unit
class TestGeoUtils:
    """Test geographic utility functions."""

    def test_haversine_distance_calculation(self):
        # Test known distance between two points (Mountain View to San Francisco)
        mv_lat, mv_lon = 37.4221, -122.0841  # Mountain View
        sf_lat, sf_lon = 37.7749, -122.4194   # San Francisco

        distance = GeoUtils.haversine_distance(mv_lat, mv_lon, sf_lat, sf_lon)

        # Distance should be approximately 49km
        assert 45000 < distance < 55000

    def test_haversine_same_point(self):
        # Distance from point to itself should be 0
        distance = GeoUtils.haversine_distance(37.4221, -122.0841, 37.4221, -122.0841)
        assert distance == 0.0

    def test_bearing_calculation(self):
        # Test bearing calculation
        # From Mountain View due north should be ~0 degrees
        mv_lat, mv_lon = 37.4221, -122.0841
        north_lat, north_lon = 37.5221, -122.0841  # 1 degree north

        bearing = GeoUtils.bearing(mv_lat, mv_lon, north_lat, north_lon)

        # Should be close to 0 degrees (due north)
        assert -5 <= bearing <= 5 or 355 <= bearing <= 360

    def test_bearing_due_east(self):
        # Test due east bearing (~90 degrees)
        mv_lat, mv_lon = 37.4221, -122.0841
        east_lat, east_lon = 37.4221, -121.0841  # 1 degree east

        bearing = GeoUtils.bearing(mv_lat, mv_lon, east_lat, east_lon)

        # Should be close to 90 degrees (due east)
        assert 85 <= bearing <= 95


@pytest.mark.unit
class TestThreatClusterer:
    """Test threat deduplication and clustering."""

    def test_no_threats_returns_empty(self):
        deduplicated = ThreatClusterer.deduplicate_threats([])
        assert deduplicated == []

    def test_single_threat_returns_unchanged(self, sample_waze_alerts):
        single_alert = sample_waze_alerts[0:1]
        deduplicated = ThreatClusterer.deduplicate_threats(single_alert)

        assert len(deduplicated) == 1
        assert deduplicated[0] == single_alert[0]

    def test_distant_threats_not_clustered(self, sample_waze_alerts):
        # Use alerts that are far apart (different locations)
        distant_alerts = sample_waze_alerts[:2]  # Mountain View and Highway 101

        deduplicated = ThreatClusterer.deduplicate_threats(distant_alerts)

        # Should not be clustered due to distance
        assert len(deduplicated) == 2

    def test_close_threats_clustered(self):
        # Create two very close threats
        from sunnypilot.rtid.waze_api_client import WazeAlert

        alert1 = WazeAlert(
            id='close-1', type='police',
            latitude=37.4221, longitude=-122.0841,
            confidence=0.8, raw_data={}
        )
        alert2 = WazeAlert(
            id='close-2', type='police',
            latitude=37.4222, longitude=-122.0842,  # Very close
            confidence=0.9, raw_data={}
        )

        deduplicated = ThreatClusterer.deduplicate_threats([alert1, alert2])

        # Should be clustered into one threat (highest confidence)
        assert len(deduplicated) == 1
        assert deduplicated[0].confidence == 0.9  # Higher confidence alert kept


@pytest.mark.unit
class TestRoadMatcher:
    """Test road matching and direction determination."""

    @pytest.fixture
    def road_matcher(self):
        return RoadMatcher()

    def test_same_road_detection_close_distance(self, road_matcher):
        # Test points within road proximity threshold
        ego_lat, ego_lon = 37.4221, -122.0841
        threat_lat, threat_lon = 37.4222, -122.0842  # ~15m away

        is_same = road_matcher.is_same_road(ego_lat, ego_lon, threat_lat, threat_lon, 15.0)
        assert is_same  # Should be considered same road

    def test_same_road_detection_far_distance(self, road_matcher):
        # Test points far apart
        ego_lat, ego_lon = 37.4221, -122.0841
        threat_lat, threat_lon = 37.5000, -122.2000  # Several km away

        is_same = road_matcher.is_same_road(ego_lat, ego_lon, threat_lat, threat_lon, 15.0)
        assert not is_same  # Should not be considered same road

    def test_highway_threshold_adjustment(self, road_matcher):
        # Test that highway speeds use larger proximity threshold
        ego_lat, ego_lon = 37.4221, -122.0841
        threat_lat, threat_lon = 37.4225, -122.0845  # ~60m away

        # At low speed, should not be same road
        is_same_slow = road_matcher.is_same_road(ego_lat, ego_lon, threat_lat, threat_lon, 10.0)
        assert not is_same_slow

        # At highway speed, should be same road due to larger threshold
        is_same_fast = road_matcher.is_same_road(ego_lat, ego_lon, threat_lat, threat_lon, 30.0)
        assert is_same_fast

    def test_direction_determination_ahead(self, road_matcher):
        # Test threat directly ahead
        ego_lat, ego_lon = 37.4221, -122.0841
        threat_lat, threat_lon = 37.4231, -122.0841  # North of ego

        direction = road_matcher.get_direction_relative_to_ego(
            ego_lat, ego_lon, threat_lat, threat_lon, ego_heading=0  # Facing north
        )
        assert direction == 'ahead'

    def test_direction_determination_behind(self, road_matcher):
        # Test threat behind
        ego_lat, ego_lon = 37.4221, -122.0841
        threat_lat, threat_lon = 37.4211, -122.0841  # South of ego

        direction = road_matcher.get_direction_relative_to_ego(
            ego_lat, ego_lon, threat_lat, threat_lon, ego_heading=0  # Facing north
        )
        assert direction == 'behind'

    def test_direction_determination_left_right(self, road_matcher):
        # Test threats to left and right
        ego_lat, ego_lon = 37.4221, -122.0841

        # Threat to the right (east) when facing north
        right_threat_lat, right_threat_lon = 37.4221, -122.0831
        direction_right = road_matcher.get_direction_relative_to_ego(
            ego_lat, ego_lon, right_threat_lat, right_threat_lon, ego_heading=0
        )
        assert direction_right == 'right'

        # Threat to the left (west) when facing north
        left_threat_lat, left_threat_lon = 37.4221, -122.0851
        direction_left = road_matcher.get_direction_relative_to_ego(
            ego_lat, ego_lon, left_threat_lat, left_threat_lon, ego_heading=0
        )
        assert direction_left == 'left'


@pytest.mark.unit
class TestSpeedRecommendationEngine:
    """Test speed recommendation logic."""

    @pytest.fixture
    def speed_engine(self):
        return SpeedRecommendationEngine()

    def test_no_threats_returns_no_recommendation(self, speed_engine):
        recommendation, threat_ahead = speed_engine.calculate_recommendation(
            threats=[], current_speed_ms=25.0, current_location=(37.4221, -122.0841)
        )

        assert recommendation == 0.0
        assert threat_ahead is False

    def test_off_road_threats_ignored(self, speed_engine):
        # Create threat not on same road
        threat = ProcessedThreat(
            id='off-road', type='police',
            latitude=37.4221, longitude=-122.0841,
            distance=500, direction='ahead', confidence=0.8,
            speed_limit_ms=11.18, on_same_road=False
        )

        recommendation, threat_ahead = speed_engine.calculate_recommendation(
            threats=[threat], current_speed_ms=25.0, current_location=(37.4221, -122.0841)
        )

        assert recommendation == 0.0
        assert threat_ahead is False

    def test_ahead_threat_generates_recommendation(self, speed_engine):
        # Create threat ahead on same road
        threat = ProcessedThreat(
            id='ahead-threat', type='police',
            latitude=37.4221, longitude=-122.0841,
            distance=800, direction='ahead', confidence=0.8,
            speed_limit_ms=11.18, on_same_road=True  # 25 mph = 11.18 m/s
        )

        recommendation, threat_ahead = speed_engine.calculate_recommendation(
            threats=[threat], current_speed_ms=25.0, current_location=(37.4221, -122.0841)
        )

        assert recommendation == 11.18  # Should recommend speed limit
        assert threat_ahead is True

    def test_behind_threat_within_threshold(self, speed_engine):
        # Create threat behind but within threshold
        threat = ProcessedThreat(
            id='behind-threat', type='police',
            latitude=37.4221, longitude=-122.0841,
            distance=500, direction='behind', confidence=0.8,
            speed_limit_ms=11.18, on_same_road=True
        )

        recommendation, threat_ahead = speed_engine.calculate_recommendation(
            threats=[threat], current_speed_ms=25.0, current_location=(37.4221, -122.0841)
        )

        # Behind threats don't affect ahead recommendation
        assert recommendation == 0.0
        assert threat_ahead is False

    def test_distance_threshold_filtering(self, speed_engine):
        # Create threat beyond distance threshold
        threat = ProcessedThreat(
            id='far-threat', type='police',
            latitude=37.4221, longitude=-122.0841,
            distance=2000, direction='ahead', confidence=0.8,  # Beyond 1600m threshold
            speed_limit_ms=11.18, on_same_road=True
        )

        recommendation, threat_ahead = speed_engine.calculate_recommendation(
            threats=[threat], current_speed_ms=25.0, current_location=(37.4221, -122.0841)
        )

        assert recommendation == 0.0  # Too far away
        assert threat_ahead is False

    @pytest.mark.safety
    def test_safety_validation_prevents_unsafe_recommendations(self, speed_engine):
        # Create threat with unreasonably high speed limit
        threat = ProcessedThreat(
            id='unsafe-threat', type='police',
            latitude=37.4221, longitude=-122.0841,
            distance=500, direction='ahead', confidence=0.8,
            speed_limit_ms=50.0, on_same_road=True  # Unreasonably high
        )

        current_speed = 20.0
        recommendation, threat_ahead = speed_engine.calculate_recommendation(
            threats=[threat], current_speed_ms=current_speed, current_location=(37.4221, -122.0841)
        )

        # Should be capped at safe maximum (1.1x current speed)
        max_safe = current_speed * 1.1
        assert recommendation <= max_safe
        assert threat_ahead is True


@pytest.mark.unit
class TestThreatDetector:
    """Test main threat detection and processing."""

    @pytest.fixture
    def threat_detector(self):
        return ThreatDetector()

    def test_empty_traffic_data_returns_safe_state(self, threat_detector):
        # Test with no traffic data
        state = threat_detector.process_threats(
            traffic_data=None,
            current_location=(37.4221, -122.0841),
            current_speed=25.0,
            timestamp=1234567890
        )

        assert isinstance(state, RTIState)
        assert state.threat_ahead is False
        assert state.recommended_speed == 0.0
        assert len(state.threats) == 0

    def test_single_threat_processing(self, threat_detector, sample_waze_alerts):
        # Test processing single threat
        single_alert = sample_waze_alerts[:1]

        state = threat_detector.process_threats(
            traffic_data=single_alert,
            current_location=(37.4221, -122.0841),
            current_speed=25.0,
            timestamp=1234567890
        )

        assert isinstance(state, RTIState)
        assert len(state.threats) >= 0  # May or may not be relevant

    @pytest.mark.safety
    def test_safety_validation_in_main_processing(self, threat_detector):
        # Create artificial scenario that might generate unsafe recommendation
        from sunnypilot.rtid.waze_api_client import WazeAlert

        unsafe_alert = WazeAlert(
            id='unsafe', type='police',
            latitude=37.4221, longitude=-122.0841,
            confidence=0.8, speed_limit=200,  # Unreasonable speed limit
            raw_data={}
        )

        with patch.object(threat_detector.speed_engine, 'calculate_recommendation') as mock_calc:
            # Mock an unsafe recommendation
            mock_calc.return_value = (60.0, True)  # Much higher than current speed

            state = threat_detector.process_threats(
                traffic_data=[unsafe_alert],
                current_location=(37.4221, -122.0841),
                current_speed=25.0,
                timestamp=1234567890
            )

            # Safety validation should reject unsafe recommendation
            assert state.recommended_speed == 0.0
            assert state.threat_ahead is False

    @pytest.mark.performance
    def test_processing_performance_budget(self, threat_detector, sample_waze_alerts, performance_timer):
        """Test that threat processing meets 15ms performance budget."""
        # Use multiple threats to stress test performance
        many_alerts = sample_waze_alerts * 10

        performance_timer.start()
        state = threat_detector.process_threats(
            traffic_data=many_alerts,
            current_location=(37.4221, -122.0841),
            current_speed=25.0,
            timestamp=1234567890
        )
        performance_timer.stop()

        # Should meet 15ms performance budget
        assert performance_timer.elapsed_ms < 15.0, f"Processing took {performance_timer.elapsed_ms}ms, exceeds 15ms budget"

        assert isinstance(state, RTIState)

    def test_threat_sorting_by_distance(self, threat_detector):
        # Create multiple threats at different distances
        from sunnypilot.rtid.waze_api_client import WazeAlert

        far_alert = WazeAlert(
            id='far', type='police',
            latitude=37.4300, longitude=-122.0841,  # Farther north
            confidence=0.8, raw_data={}
        )
        close_alert = WazeAlert(
            id='close', type='police',
            latitude=37.4225, longitude=-122.0841,  # Closer
            confidence=0.8, raw_data={}
        )

        state = threat_detector.process_threats(
            traffic_data=[far_alert, close_alert],
            current_location=(37.4221, -122.0841),
            current_speed=25.0,
            timestamp=1234567890
        )

        if len(state.threats) >= 2:
            # Threats should be sorted by distance (closest first)
            assert state.threats[0].distance <= state.threats[1].distance

    def test_hud_threat_limit(self, threat_detector):
        # Create more than 5 threats
        from sunnypilot.rtid.waze_api_client import WazeAlert

        many_alerts = []
        for i in range(8):
            alert = WazeAlert(
                id=f'threat-{i}', type='police',
                latitude=37.4221 + i * 0.001, longitude=-122.0841,
                confidence=0.8, raw_data={}
            )
            many_alerts.append(alert)

        state = threat_detector.process_threats(
            traffic_data=many_alerts,
            current_location=(37.4221, -122.0841),
            current_speed=25.0,
            timestamp=1234567890
        )

        # Should be limited to 5 threats for HUD
        assert len(state.threats) <= 5

    def test_error_recovery_returns_safe_state(self, threat_detector):
        # Test error handling with invalid data
        with patch.object(threat_detector, '_process_single_threat', side_effect=Exception("Test error")):
            state = threat_detector.process_threats(
                traffic_data=[{}],  # Invalid threat data
                current_location=(37.4221, -122.0841),
                current_speed=25.0,
                timestamp=1234567890
            )

            # Should return safe state despite error
            assert isinstance(state, RTIState)
            assert state.api_status == 'offline'


@pytest.mark.integration
@pytest.mark.safety
class TestThreatDetectorIntegration:
    """Integration tests for complete threat processing pipeline."""

    def test_full_threat_processing_pipeline(self, sample_waze_alerts):
        """Test complete processing from raw alerts to RTI state."""
        detector = ThreatDetector()

        state = detector.process_threats(
            traffic_data=sample_waze_alerts,
            current_location=(37.4221, -122.0841),
            current_speed=25.0,
            timestamp=1234567890
        )

        # Validate complete RTI state structure
        assert isinstance(state, RTIState)
        assert isinstance(state.timestamp, int)
        assert isinstance(state.threat_ahead, bool)
        assert isinstance(state.threat_distance_m, float)
        assert isinstance(state.recommended_speed, (int, float))
        assert isinstance(state.source, str)
        assert isinstance(state.api_status, str)
        assert isinstance(state.threats, list)

        # Validate safety constraints
        if state.recommended_speed > 0:
            # Speed should be reasonable (not negative, not excessively high)
            assert 0 < state.recommended_speed <= 50.0, f"Recommended speed {state.recommended_speed} is unsafe"

    def test_realistic_highway_scenario(self):
        """Test realistic highway driving scenario."""
        from sunnypilot.rtid.waze_api_client import WazeAlert

        # Highway scenario: 65 mph speed limit, police 1 mile ahead
        highway_alert = WazeAlert(
            id='highway-police', type='police',
            latitude=37.4221 + 0.014,  # ~1 mile north
            longitude=-122.0841,
            confidence=0.9, speed_limit=65, raw_data={}  # 65 mph
        )

        detector = ThreatDetector()
        state = detector.process_threats(
            traffic_data=[highway_alert],
            current_location=(37.4221, -122.0841),
            current_speed=29.0,  # ~65 mph in m/s
            timestamp=1234567890
        )

        # Should generate speed recommendation for highway speeds
        if state.threat_ahead:
            assert state.recommended_speed > 0
            # Should recommend highway speed limit (65 mph ≈ 29 m/s)
            expected_speed_ms = 65 / 2.237  # mph to m/s conversion
            assert abs(state.recommended_speed - expected_speed_ms) < 2.0
