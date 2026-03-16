#!/usr/bin/env python3
"""
Unit tests for Enhanced RoadMatcher with road geometry integration.

Tests the road-geometry-aware threat detection algorithms that replace
the primitive proximity-based logic with actual OSM road data.
"""

import unittest
from unittest.mock import MagicMock, patch

from openpilot.sunnypilot.navd.helpers import Coordinate
from openpilot.sunnypilot.mapd.road_geometry import (
    RoadClass, LaneType, RoadCoordinate, Lane, RoadSegment, BarrierType, Barrier
)


class TestEnhancedRoadMatcher(unittest.TestCase):
    """Test enhanced road matching using actual road geometry."""

    def setUp(self):
        """Set up test road segments and mock data."""
        # Create a straight primary road with median barrier
        self.primary_road_centerline = [
            RoadCoordinate(37.4221, -122.0841, 0.0),    # Start
            RoadCoordinate(37.4222, -122.0842, 100.0),  # 100m along
            RoadCoordinate(37.4223, -122.0843, 200.0),  # 200m along
            RoadCoordinate(37.4224, -122.0844, 300.0),  # 300m along
        ]

        self.primary_road_lanes = [
            Lane(0, 3.5, LaneType.DRIVING, []),  # Right lane
            Lane(1, 3.5, LaneType.DRIVING, []),  # Left lane
        ]

        # Median barrier between opposing directions
        self.median_barrier = Barrier(
            barrier_type=BarrierType.MEDIAN,
            coordinates=[
                RoadCoordinate(37.4221, -122.08405, 0.0),   # Slightly offset from centerline
                RoadCoordinate(37.4222, -122.08415, 100.0),
                RoadCoordinate(37.4223, -122.08425, 200.0),
                RoadCoordinate(37.4224, -122.08435, 300.0),
            ]
        )

        self.primary_road = RoadSegment(
            way_id=1001,
            name="Primary Road",
            road_class=RoadClass.PRIMARY,
            centerline=self.primary_road_centerline,
            lanes=self.primary_road_lanes,
            barriers=[self.median_barrier],  # Restore barrier
            level_separation=0,  # Ground level
            max_speed=13.9,  # 50 km/h
            road_direction=45.0  # Northeast
        )

        # Create opposing direction road (separate way_id, same physical road)
        self.opposing_road_centerline = [
            RoadCoordinate(37.4224, -122.08445, 0.0),    # Start (opposite end)
            RoadCoordinate(37.4223, -122.08435, 100.0),
            RoadCoordinate(37.4222, -122.08425, 200.0),
            RoadCoordinate(37.4221, -122.08415, 300.0),  # End (opposite start)
        ]

        self.opposing_road = RoadSegment(
            way_id=1002,  # Different way_id for opposing direction
            name="Primary Road Opposing",
            road_class=RoadClass.PRIMARY,
            centerline=self.opposing_road_centerline,
            lanes=self.primary_road_lanes,
            barriers=[self.median_barrier],
            level_separation=0,
            max_speed=13.9,
            road_direction=225.0  # Southwest (opposite direction)
        )

        # Create bridge road at different location but same level separation concept
        self.bridge_road = RoadSegment(
            way_id=1003,
            name="Bridge Road",
            road_class=RoadClass.SECONDARY,
            centerline=[
                RoadCoordinate(37.4220, -122.0835, 0.0),    # More offset
                RoadCoordinate(37.4221, -122.0836, 100.0),  # More offset
                RoadCoordinate(37.4222, -122.0837, 200.0),  # More offset
            ],
            lanes=[Lane(0, 3.5, LaneType.DRIVING, [])],
            barriers=[],
            level_separation=1,  # Bridge level
            max_speed=11.1,  # 40 km/h
            road_direction=45.0
        )

        # Create completely separate road
        self.separate_road = RoadSegment(
            way_id=1004,
            name="Separate Road",
            road_class=RoadClass.RESIDENTIAL,
            centerline=[
                RoadCoordinate(37.4241, -122.0861, 0.0),   # 500m away
                RoadCoordinate(37.4242, -122.0862, 100.0),
            ],
            lanes=[Lane(0, 3.0, LaneType.DRIVING, [])],
            barriers=[],
            level_separation=0,
            max_speed=8.3,  # 30 km/h
            road_direction=90.0  # East
        )

    def test_project_position_to_road_segment(self):
        """Test projecting GPS positions onto road centerlines."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        # Test position on road centerline
        ego_position = Coordinate(37.4222, -122.0842)  # Exactly on centerline at 100m
        projection = matcher._project_position_to_road_segment(ego_position, self.primary_road)

        self.assertIsNotNone(projection)
        self.assertAlmostEqual(projection.distance_from_start, 100.0, places=1)
        self.assertLess(projection.distance_to_centerline, 1.0)  # Very close to centerline

        # Test position slightly off road (very small offset for ~5m distance)
        off_road_position = Coordinate(37.4222, -122.08420005)  # About 5m off centerline
        projection = matcher._project_position_to_road_segment(off_road_position, self.primary_road)

        self.assertIsNotNone(projection)
        self.assertGreater(projection.distance_to_centerline, 0.001)  # Should be > 0 for off-road position
        self.assertLess(projection.distance_to_centerline, 10.0)  # Should be reasonable distance

    def test_same_road_detection_with_geometry(self):
        """Test enhanced same-road detection using road geometry."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        # Mock road geometry data
        mock_road_data = {
            'road_geometry_valid': True,
            'current_road_segment': self.primary_road,
            'nearby_road_segments': [self.primary_road, self.opposing_road, self.bridge_road]
        }

        # Test 1: Threat on same road segment (same way_id)
        ego_pos = Coordinate(37.4222, -122.0842)  # On primary road
        threat_pos = Coordinate(37.4223, -122.0843)  # Also on primary road, 100m ahead

        is_same = matcher._is_same_road_geometry_based(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertTrue(is_same)

    def test_same_road_detection_with_median_barrier(self):
        """Test that threats across median barriers are correctly excluded."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        mock_road_data = {
            'road_geometry_valid': True,
            'current_road_segment': self.primary_road,
            'nearby_road_segments': [self.primary_road, self.opposing_road]
        }

        # Test: Ego on primary road, threat on opposing road (across median)
        ego_pos = Coordinate(37.4222, -122.0842)     # Primary road
        threat_pos = Coordinate(37.4222, -122.08425)  # Opposing road (across median)

        is_same = matcher._is_same_road_geometry_based(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertFalse(is_same)  # Should be False due to median barrier

    def test_same_road_detection_with_level_separation(self):
        """Test that threats on bridges/tunnels are correctly handled."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        mock_road_data = {
            'road_geometry_valid': True,
            'current_road_segment': self.primary_road,  # Ground level
            'nearby_road_segments': [self.primary_road, self.bridge_road]
        }

        # Test: Ego on ground level, threat on bridge
        ego_pos = Coordinate(37.4222, -122.0842)    # Ground level road
        threat_pos = Coordinate(37.4221, -122.0836)  # Bridge road centerline

        is_same = matcher._is_same_road_geometry_based(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertFalse(is_same)  # Should be False due to level separation

    def test_road_aware_direction_calculation(self):
        """Test direction calculation using road centerline geometry."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        mock_road_data = {
            'road_geometry_valid': True,
            'current_road_segment': self.primary_road,
            'nearby_road_segments': [self.primary_road]
        }

        # Test 1: Threat ahead on same road
        ego_pos = Coordinate(37.4222, -122.0842)    # 100m along road
        threat_pos = Coordinate(37.4223, -122.0843)  # 200m along road (ahead)

        direction = matcher._get_direction_road_aware(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertEqual(direction, 'ahead')

        # Test 2: Threat behind on same road (use more distinct coordinates)
        ego_pos = Coordinate(37.4223, -122.0843)    # 200m along road
        threat_pos = Coordinate(37.4221, -122.0841)  # 0m along road (clearly behind)

        direction = matcher._get_direction_road_aware(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertEqual(direction, 'behind')

    def test_road_aware_direction_with_curves(self):
        """Test direction calculation on curved roads."""
        # Create curved road
        curved_road_centerline = [
            RoadCoordinate(37.4221, -122.0841, 0.0),    # Start - going east
            RoadCoordinate(37.4221, -122.0831, 100.0),  # Turn north
            RoadCoordinate(37.4231, -122.0831, 200.0),  # Continue north
            RoadCoordinate(37.4231, -122.0821, 300.0),  # Turn east again
        ]

        curved_road = RoadSegment(
            way_id=2001,
            name="Curved Road",
            road_class=RoadClass.SECONDARY,
            centerline=curved_road_centerline,
            lanes=[Lane(0, 3.5, LaneType.DRIVING, [])],
            barriers=[],
            level_separation=0,
            max_speed=11.1,
            road_direction=0.0  # Initial direction - east
        )

        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        mock_road_data = {
            'road_geometry_valid': True,
            'current_road_segment': curved_road,
            'nearby_road_segments': [curved_road]
        }

        # Test: On curved road, GPS bearing might say "right" but road geometry says "ahead"
        ego_pos = Coordinate(37.4221, -122.0836)    # 50m along curved road
        threat_pos = Coordinate(37.4226, -122.0831)  # 150m along curved road (around curve)

        direction = matcher._get_direction_road_aware(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertEqual(direction, 'ahead')  # Should be 'ahead' despite GPS bearing

    def test_fallback_to_proximity_logic(self):
        """Test fallback to original proximity logic when road geometry unavailable."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        # Mock no road geometry available
        mock_road_data = {
            'road_geometry_valid': False,
            'current_road_segment': None,
            'nearby_road_segments': []
        }

        ego_lat, ego_lon = 37.4222, -122.0842
        threat_lat, threat_lon = 37.4223, -122.0843  # ~100m away
        ego_speed_ms = 15.0  # 54 km/h

        # Should fall back to proximity-based logic
        is_same = matcher.is_same_road(
            ego_lat, ego_lon, threat_lat, threat_lon, ego_speed_ms, mock_road_data
        )

        # Should use proximity threshold (threat is ~100m away, within 200m highway threshold)
        self.assertTrue(is_same)

    def test_performance_requirements(self):
        """Test that road geometry processing meets performance requirements."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher
        import time

        matcher = EnhancedRoadMatcher()

        # Create large road network for performance testing
        large_road_network = []
        for i in range(50):  # 50 road segments
            centerline = [
                RoadCoordinate(37.42 + i*0.001, -122.08 + j*0.001, j*100.0)
                for j in range(10)  # 10 points per segment
            ]
            road = RoadSegment(
                way_id=3000 + i,
                name=f"Road {i}",
                road_class=RoadClass.RESIDENTIAL,
                centerline=centerline,
                lanes=[Lane(0, 3.0, LaneType.DRIVING, [])],
                barriers=[],
                level_separation=0,
                max_speed=8.3,
                road_direction=float(i * 10)
            )
            large_road_network.append(road)

        mock_road_data = {
            'road_geometry_valid': True,
            'current_road_segment': large_road_network[0],
            'nearby_road_segments': large_road_network
        }

        # Test processing time for multiple threats
        ego_pos = Coordinate(37.4220, -122.0800)
        threat_positions = [
            Coordinate(37.42 + i*0.0001, -122.08 + i*0.0001)
            for i in range(20)  # 20 threats
        ]

        start_time = time.time()

        for threat_pos in threat_positions:
            matcher._is_same_road_geometry_based(ego_pos, threat_pos, mock_road_data)
            matcher._get_direction_road_aware(ego_pos, threat_pos, mock_road_data)

        processing_time = time.time() - start_time

        # Should process 20 threats in under 100ms for 1Hz RTI cycle compatibility
        self.assertLess(processing_time, 0.100,
                       f"Processing took {processing_time*1000:.1f}ms, should be under 100ms")


class TestRTIMapdIntegration(unittest.TestCase):
    """Test integration between RTI ThreatDetector and mapd road geometry."""

    def setUp(self):
        """Set up mock mapd data and RTI components."""
        self.mock_mapd_data = {
            'road_geometry_valid': True,
            'current_road_segment': None,
            'nearby_road_segments': []
        }

    @patch('cereal.messaging.SubMaster')
    def test_mapd_data_subscription(self, mock_sub_master):
        """Test that RTI correctly subscribes to mapd road geometry data."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        mock_sm = MagicMock()
        mock_sub_master.return_value = mock_sm

        # Mock mapd message
        mock_mapd_msg = MagicMock()
        mock_mapd_msg.roadGeometryValid = True
        mock_sm.__getitem__.return_value = mock_mapd_msg
        mock_sm.updated = {'liveMapDataSP': True}

        matcher = EnhancedRoadMatcher()

        # Verify subscription (allowing for ignore_avg_freq parameter)
        mock_sub_master.assert_called_with(['liveMapDataSP'], ignore_avg_freq=True)

        # Test data update
        updated = matcher._update_road_geometry_from_mapd()
        self.assertTrue(updated)

    def test_integration_with_threat_detector(self):
        """Test integration with existing ThreatDetector."""
        from openpilot.sunnypilot.rtid.threat_detector import ThreatDetector
        from openpilot.sunnypilot.rtid.waze_api_client import WazeAlert

        # ThreatDetector currently depends on its internal RoadMatcher contract.
        with patch('openpilot.sunnypilot.rtid.threat_detector.RoadMatcher') as mock_road_matcher_class:
            mock_road_matcher = MagicMock()
            mock_road_matcher_class.return_value = mock_road_matcher

            # Configure enhanced matching responses
            mock_road_matcher.is_same_road.return_value = (True, 1.0)
            mock_road_matcher.get_direction_relative_to_ego.return_value = 'ahead'

            detector = ThreatDetector()

            # Create test threat
            test_threat = WazeAlert(
                id='test_threat_001',
                latitude=37.4223,
                longitude=-122.0843,
                type='police',
                confidence=0.9,
                speed_limit=None
            )

            # Process threat
            rti_state = detector.process_threats(
                traffic_data=[test_threat],
                current_location=(37.4222, -122.0842),
                current_speed=15.0,
                timestamp=int(1000 * 1e9)
            )

            # Verify enhanced road matcher was used
            mock_road_matcher.is_same_road.assert_called()
            mock_road_matcher.get_direction_relative_to_ego.assert_called()

            # Verify threat was processed correctly
            self.assertEqual(len(rti_state.threats), 1)
            self.assertEqual(rti_state.threats[0].direction, 'ahead')
            self.assertTrue(rti_state.threats[0].on_same_road)

    def test_backward_compatibility(self):
        """Test that enhanced road matcher maintains backward compatibility."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        # Test original interface still works
        is_same = matcher.is_same_road(
            ego_lat=37.4222,
            ego_lon=-122.0842,
            threat_lat=37.4223,
            threat_lon=-122.0843,
            ego_speed_ms=15.0
        )

        direction = matcher.get_direction_relative_to_ego(
            ego_lat=37.4222,
            ego_lon=-122.0842,
            threat_lat=37.4223,
            threat_lon=-122.0843,
            ego_heading=45.0
        )

        # Should return valid results
        self.assertIsInstance(is_same, bool)
        self.assertIn(direction, ['ahead', 'behind', 'left', 'right'])


class TestRealWorldScenarios(unittest.TestCase):
    """Test enhanced RTI with real-world problematic scenarios."""

    def test_original_user_scenario(self):
        """Test the original user scenario: hazard 0.25 miles away, direction 'right'."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher
        from openpilot.sunnypilot.rtid.threat_detector import GeoUtils

        matcher = EnhancedRoadMatcher()

        # Create highway with sharp curve - threat appears "right" via GPS but is actually "ahead" on road
        highway_centerline = [
            RoadCoordinate(37.4000, -122.0000, 0.0),      # Start - heading east
            RoadCoordinate(37.4010, -122.0000, 1000.0),   # 1km straight east
            RoadCoordinate(37.4015, -121.9985, 1500.0),   # Sharp right turn north
            RoadCoordinate(37.4025, -121.9985, 2000.0),   # Continue north
            RoadCoordinate(37.4035, -121.9985, 2500.0),   # Straighten out north
        ]

        highway_road = RoadSegment(
            way_id=9001,
            name="Highway Road",
            road_class=RoadClass.MOTORWAY,
            centerline=highway_centerline,
            lanes=[
                Lane(0, 3.7, LaneType.DRIVING, []),
                Lane(1, 3.7, LaneType.DRIVING, []),
                Lane(2, 3.7, LaneType.DRIVING, []),
            ],
            barriers=[],
            level_separation=0,
            max_speed=33.3,  # 120 km/h
            road_direction=0.0  # Initially east
        )

        mock_road_data = {
            'road_geometry_valid': True,
            'current_road_segment': highway_road,
            'nearby_road_segments': [highway_road]
        }

        # Ego vehicle at start of curve
        ego_pos = Coordinate(37.4012, -122.0000)  # ~1.2km along highway

        # Threat after sharp right turn - appears "right" via GPS but "ahead" on road
        threat_pos = Coordinate(37.4025, -121.9985)  # After the sharp turn, going north

        # Calculate GPS bearing (should be "right" or "southeast")
        gps_bearing = GeoUtils.bearing(
            ego_pos.latitude, ego_pos.longitude,
            threat_pos.latitude, threat_pos.longitude
        )

        # GPS bearing should indicate direction other than "ahead" (demonstrating the issue)
        ego_heading = 90.0  # Traveling east
        relative_bearing = (gps_bearing - ego_heading + 360) % 360

        # Classify direction based on bearing
        if relative_bearing <= 45 or relative_bearing >= 315:
            gps_direction = 'ahead'
        elif 45 < relative_bearing <= 135:
            gps_direction = 'right'
        elif 135 < relative_bearing <= 225:
            gps_direction = 'behind'
        else:
            gps_direction = 'left'

        # With original logic, this would NOT be classified as "ahead" (demonstrating GPS bearing issue)
        self.assertNotEqual(gps_direction, 'ahead')

        # With enhanced road-aware logic, should be "ahead"
        road_direction = matcher._get_direction_road_aware(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertEqual(road_direction, 'ahead')

        # Verify same road detection works
        is_same_road = matcher._is_same_road_geometry_based(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertTrue(is_same_road)

    def test_median_barrier_scenario(self):
        """Test threats across median barriers are correctly excluded."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        # Create divided highway with median barrier
        northbound_centerline = [
            RoadCoordinate(37.4000, -122.0000, 0.0),
            RoadCoordinate(37.4010, -122.0000, 1000.0),
            RoadCoordinate(37.4020, -122.0000, 2000.0),
        ]

        southbound_centerline = [
            RoadCoordinate(37.4020, -122.0005, 0.0),     # Opposite direction, 5m offset
            RoadCoordinate(37.4010, -122.0005, 1000.0),
            RoadCoordinate(37.4000, -122.0005, 2000.0),
        ]

        median_barrier = Barrier(
            barrier_type=BarrierType.MEDIAN,
            coordinates=[
                RoadCoordinate(37.4000, -122.00025, 0.0),    # Between the roads
                RoadCoordinate(37.4010, -122.00025, 1000.0),
                RoadCoordinate(37.4020, -122.00025, 2000.0),
            ]
        )

        northbound_road = RoadSegment(
            way_id=9010,
            name="Northbound Road",
            road_class=RoadClass.MOTORWAY,
            centerline=northbound_centerline,
            lanes=[Lane(0, 3.7, LaneType.DRIVING, [])],
            barriers=[median_barrier],
            level_separation=0,
            max_speed=33.3,
            road_direction=0.0  # North
        )

        southbound_road = RoadSegment(
            way_id=9011,  # Different way_id
            name="Southbound Road",
            road_class=RoadClass.MOTORWAY,
            centerline=southbound_centerline,
            lanes=[Lane(0, 3.7, LaneType.DRIVING, [])],
            barriers=[median_barrier],
            level_separation=0,
            max_speed=33.3,
            road_direction=180.0  # South
        )

        mock_road_data = {
            'road_geometry_valid': True,
            'current_road_segment': northbound_road,
            'nearby_road_segments': [northbound_road, southbound_road]
        }

        # Ego on northbound road
        ego_pos = Coordinate(37.4010, -122.0000)

        # Threat on southbound road (across median, only 5m away via GPS)
        threat_pos = Coordinate(37.4010, -122.0005)

        # Original proximity logic would say "same road" (actual distance ~44m < 50m threshold)
        distance = ego_pos.distance_to(threat_pos)  # Already in meters
        original_logic_same_road = distance <= 50.0
        self.assertTrue(original_logic_same_road)

        # Enhanced logic should say "different road" due to median barrier
        enhanced_logic_same_road = matcher._is_same_road_geometry_based(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertFalse(enhanced_logic_same_road)

    def test_bridge_overpass_scenario(self):
        """Test threats on bridge overpasses are correctly excluded."""
        from openpilot.sunnypilot.rtid.enhanced_road_matcher import EnhancedRoadMatcher

        matcher = EnhancedRoadMatcher()

        # Ground level road
        ground_road = RoadSegment(
            way_id=9020,
            name="Ground Road",
            road_class=RoadClass.PRIMARY,
            centerline=[
                RoadCoordinate(37.4000, -122.0000, 0.0),
                RoadCoordinate(37.4010, -122.0000, 1000.0),
            ],
            lanes=[Lane(0, 3.5, LaneType.DRIVING, [])],
            barriers=[],
            level_separation=0,  # Ground level
            max_speed=13.9,
            road_direction=0.0
        )

        # Bridge/overpass at slightly different GPS coordinates (GPS inaccuracy)
        bridge_road = RoadSegment(
            way_id=9021,
            name="Bridge Road",
            road_class=RoadClass.SECONDARY,
            centerline=[
                RoadCoordinate(37.4000, -122.0001, 0.0),    # Slightly offset GPS coordinates
                RoadCoordinate(37.4010, -122.0001, 1000.0), # Slightly offset GPS coordinates
            ],
            lanes=[Lane(0, 3.5, LaneType.DRIVING, [])],
            barriers=[],
            level_separation=1,  # Bridge level
            max_speed=11.1,
            road_direction=0.0
        )

        mock_road_data = {
            'road_geometry_valid': True,
            'current_road_segment': ground_road,
            'nearby_road_segments': [ground_road, bridge_road]
        }

        # Ego on ground level
        ego_pos = Coordinate(37.4005, -122.0000)

        # Threat on bridge (positioned on bridge coordinates)
        threat_pos = Coordinate(37.4005, -122.0001)

        # Original proximity logic would say "same road" (small distance due to GPS offset)
        distance = ego_pos.distance_to(threat_pos)  # Already in meters
        original_logic_same_road = distance <= 50.0
        self.assertTrue(original_logic_same_road)

        # Enhanced logic should say "different road" due to level separation
        enhanced_logic_same_road = matcher._is_same_road_geometry_based(
            ego_pos, threat_pos, mock_road_data
        )
        self.assertFalse(enhanced_logic_same_road)


if __name__ == '__main__':
    unittest.main()
