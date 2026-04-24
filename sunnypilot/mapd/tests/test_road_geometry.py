#!/usr/bin/env python3
"""
Unit tests for road geometry processing components.

Tests the road geometry extraction, caching, and matching functionality
to ensure reliability before integration with RTI system.
"""

import unittest
import tempfile
import sqlite3
import json
from unittest.mock import MagicMock

from openpilot.sunnypilot.navd.helpers import Coordinate
from openpilot.sunnypilot.mapd.road_geometry import (
    RoadClass, LaneType, RoadCoordinate, Lane, RoadSegment,
    OSMRoadGeometryExtractor, RoadGeometryCache
)


class TestRoadCoordinate(unittest.TestCase):
    """Test RoadCoordinate data structure."""

    def test_coordinate_conversion(self):
        """Test conversion between RoadCoordinate and Coordinate."""
        road_coord = RoadCoordinate(37.4221, -122.0841, 100.0)
        coord = road_coord.to_coordinate()

        self.assertEqual(coord.latitude, 37.4221)
        self.assertEqual(coord.longitude, -122.0841)

        # Test round-trip conversion
        road_coord2 = RoadCoordinate.from_coordinate(coord, 100.0)
        self.assertEqual(road_coord2.latitude, road_coord.latitude)
        self.assertEqual(road_coord2.longitude, road_coord.longitude)
        self.assertEqual(road_coord2.distance_from_start, 100.0)


class TestRoadSegment(unittest.TestCase):
    """Test RoadSegment functionality."""

    def setUp(self):
        """Set up test road segment."""
        centerline = [
            RoadCoordinate(37.4221, -122.0841, 0.0),
            RoadCoordinate(37.4222, -122.0842, 100.0),
            RoadCoordinate(37.4223, -122.0843, 200.0)
        ]

        lanes = [
            Lane(0, 3.5, LaneType.DRIVING, []),
            Lane(1, 3.5, LaneType.DRIVING, [])
        ]

        self.road_segment = RoadSegment(
            way_id=12345,
            name="Test Road",
            road_class=RoadClass.PRIMARY,
            centerline=centerline,
            lanes=lanes,
            barriers=[],
            level_separation=0,
            max_speed=13.9,  # 50 km/h
            road_direction=45.0
        )

    def test_get_length(self):
        """Test road segment length calculation."""
        length = self.road_segment.get_length()
        self.assertEqual(length, 200.0)

    def test_get_closest_point(self):
        """Test finding closest point on road."""
        # Point very close to start of road
        test_point = Coordinate(37.4221, -122.0841)
        closest_point, distance = self.road_segment.get_closest_point(test_point)

        self.assertLess(distance, 1.0)  # Should be very close
        self.assertAlmostEqual(closest_point.distance_from_start, 0.0, places=1)

        # Point near middle of road
        test_point = Coordinate(37.4222, -122.0842)
        closest_point, distance = self.road_segment.get_closest_point(test_point)

        self.assertLess(distance, 1.0)
        self.assertAlmostEqual(closest_point.distance_from_start, 100.0, places=1)

    def test_project_to_segment(self):
        """Test point projection to road segment."""
        # Test projection to middle of first segment
        point = Coordinate(37.42215, -122.08415)  # Slightly off the line

        projected = self.road_segment._project_to_segment(
            point,
            self.road_segment.centerline[0],
            self.road_segment.centerline[1]
        )

        # Should project somewhere between start and middle
        self.assertGreater(projected.distance_from_start, 0.0)
        self.assertLess(projected.distance_from_start, 100.0)


class TestOSMRoadGeometryExtractor(unittest.TestCase):
    """Test OSM road geometry extraction."""

    def setUp(self):
        """Set up test database."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

        # Create minimal test schema
        self.connection.executescript("""
            CREATE TABLE ways (
                way_id INTEGER PRIMARY KEY,
                tags TEXT
            );
            
            CREATE TABLE way_nodes (
                way_id INTEGER,
                node_id INTEGER,
                sequence_id INTEGER
            );
            
            CREATE TABLE nodes (
                node_id INTEGER PRIMARY KEY,
                latitude REAL,
                longitude REAL
            );
        """)

        # Insert test data
        self._insert_test_data()

        self.extractor = OSMRoadGeometryExtractor(self.db_path)

    def _insert_test_data(self):
        """Insert test road data."""
        # Insert a primary road way
        tags = json.dumps({
            'highway': 'primary',
            'lanes': '2',
            'maxspeed': '50 km/h',
            'name': 'Test Road'
        })

        self.connection.execute(
            "INSERT INTO ways (way_id, tags) VALUES (?, ?)",
            (1001, tags)
        )

        # Insert nodes for the way
        nodes = [
            (1, 37.4221, -122.0841),
            (2, 37.4222, -122.0842),
            (3, 37.4223, -122.0843)
        ]

        for node_id, lat, lon in nodes:
            self.connection.execute(
                "INSERT INTO nodes (node_id, latitude, longitude) VALUES (?, ?, ?)",
                (node_id, lat, lon)
            )

            self.connection.execute(
                "INSERT INTO way_nodes (way_id, node_id, sequence_id) VALUES (?, ?, ?)",
                (1001, node_id, node_id)  # Using node_id as sequence_id for simplicity
            )

        self.connection.commit()

    def tearDown(self):
        """Clean up test database."""
        import os
        self.connection.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_connect(self):
        """Test database connection."""
        self.assertTrue(self.extractor.connect())
        self.assertIsNotNone(self.extractor.connection)
        self.extractor.disconnect()
        self.assertIsNone(self.extractor.connection)

    def test_classify_highway(self):
        """Test highway classification."""
        self.assertEqual(
            self.extractor._classify_highway('primary'),
            RoadClass.PRIMARY
        )
        self.assertEqual(
            self.extractor._classify_highway('motorway'),
            RoadClass.MOTORWAY
        )
        self.assertEqual(
            self.extractor._classify_highway('unknown'),
            RoadClass.UNCLASSIFIED
        )

    def test_extract_centerline(self):
        """Test centerline extraction."""
        self.extractor.connect()
        centerline = self.extractor._extract_centerline(1001)

        self.assertEqual(len(centerline), 3)
        self.assertEqual(centerline[0].distance_from_start, 0.0)
        self.assertGreater(centerline[1].distance_from_start, 0.0)
        self.assertGreater(centerline[2].distance_from_start, centerline[1].distance_from_start)

        self.extractor.disconnect()

    def test_parse_lanes_count(self):
        """Test lane count parsing."""
        tags = {'lanes': '4'}
        self.assertEqual(self.extractor._parse_lanes_count(tags), 4)

        # Test default for motorway
        tags = {'highway': 'motorway'}
        self.assertEqual(self.extractor._parse_lanes_count(tags), 4)

        # Test default for residential
        tags = {'highway': 'residential'}
        self.assertEqual(self.extractor._parse_lanes_count(tags), 1)

    def test_extract_speed_limit(self):
        """Test speed limit extraction."""
        # Test km/h
        tags = {'maxspeed': '50 km/h'}
        speed = self.extractor._extract_speed_limit(tags)
        self.assertAlmostEqual(speed, 50/3.6, places=2)  # 13.89 m/s

        # Test mph
        tags = {'maxspeed': '30 mph'}
        speed = self.extractor._extract_speed_limit(tags)
        self.assertAlmostEqual(speed, 30*0.44704, places=2)  # 13.41 m/s

        # Test default for motorway
        tags = {'highway': 'motorway'}
        speed = self.extractor._extract_speed_limit(tags)
        self.assertAlmostEqual(speed, 33.3, places=1)  # 120 km/h default

    def test_extract_road_segment(self):
        """Test complete road segment extraction."""
        self.extractor.connect()

        tags = {
            'highway': 'primary',
            'lanes': '2',
            'maxspeed': '50 km/h'
        }

        segment = self.extractor._extract_road_segment(1001, tags)

        self.assertIsNotNone(segment)
        self.assertEqual(segment.way_id, 1001)
        self.assertEqual(segment.road_class, RoadClass.PRIMARY)
        self.assertEqual(len(segment.centerline), 3)
        self.assertEqual(len(segment.lanes), 2)
        self.assertAlmostEqual(segment.max_speed, 50/3.6, places=2)

        self.extractor.disconnect()

    def test_extract_road_segments_near_position(self):
        """Test extracting road segments near a position."""
        segments = self.extractor.extract_road_segments_near_position(
            37.4222, -122.0842, 1000  # 1km radius
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].way_id, 1001)


class TestRoadGeometryCache(unittest.TestCase):
    """Test road geometry caching."""

    def setUp(self):
        """Set up test cache."""
        self.cache = RoadGeometryCache(cache_radius=500.0)
        self.mock_extractor = MagicMock(spec=OSMRoadGeometryExtractor)

        # Create mock road segment
        self.mock_segment = RoadSegment(
            way_id=1001,
            name="Test Road",
            road_class=RoadClass.PRIMARY,
            centerline=[
                RoadCoordinate(37.4221, -122.0841, 0.0),
                RoadCoordinate(37.4222, -122.0842, 100.0)
            ],
            lanes=[],
            barriers=[],
            level_separation=0,
            max_speed=13.9,
            road_direction=45.0
        )

        self.mock_extractor.extract_road_segments_near_position.return_value = [self.mock_segment]

    def test_cache_initialization(self):
        """Test cache initialization."""
        self.assertEqual(self.cache.cache_radius, 500.0)
        self.assertEqual(len(self.cache.cached_segments), 0)
        self.assertIsNone(self.cache.cache_center)

    def test_get_road_segments_near(self):
        """Test getting road segments with caching."""
        position = Coordinate(37.4221, -122.0841)

        # First call should hit extractor
        segments = self.cache.get_road_segments_near(position, self.mock_extractor)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].way_id, 1001)
        self.mock_extractor.extract_road_segments_near_position.assert_called_once()

        # Second call with nearby position should use cache
        position2 = Coordinate(37.4221, -122.0842)  # Nearby position
        segments2 = self.cache.get_road_segments_near(position2, self.mock_extractor)

        self.assertEqual(len(segments2), 1)
        # Should still only be called once (cache hit)
        self.assertEqual(self.mock_extractor.extract_road_segments_near_position.call_count, 1)

    def test_find_current_road_segment(self):
        """Test finding current road segment."""
        # First populate cache
        position = Coordinate(37.4221, -122.0841)
        self.cache.get_road_segments_near(position, self.mock_extractor)

        # Test finding segment
        current_segment = self.cache.find_current_road_segment(position)
        self.assertIsNotNone(current_segment)
        self.assertEqual(current_segment.way_id, 1001)

        # Test with position far from any road
        far_position = Coordinate(40.0, -120.0)  # Different location entirely
        current_segment = self.cache.find_current_road_segment(far_position)
        self.assertIsNone(current_segment)  # Should be None due to distance threshold


class TestGeometryMath(unittest.TestCase):
    """Test mathematical functions used in geometry processing."""

    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        from openpilot.sunnypilot.mapd.road_geometry import GeoUtils

        # Test known distance
        lat1, lon1 = 37.4221, -122.0841  # Palo Alto
        lat2, lon2 = 37.4222, -122.0842  # Nearby point

        distance = GeoUtils.haversine_distance(lat1, lon1, lat2, lon2)

        # Should be approximately 10-20 meters (small coordinate change)
        self.assertGreater(distance, 5)
        self.assertLess(distance, 50)

    def test_bearing_calculation(self):
        """Test bearing calculation."""
        from openpilot.sunnypilot.mapd.road_geometry import GeoUtils

        # Test due north bearing
        lat1, lon1 = 37.0, -122.0
        lat2, lon2 = 38.0, -122.0  # Due north

        bearing = GeoUtils.bearing(lat1, lon1, lat2, lon2)

        # Should be close to 0 degrees (due north)
        self.assertLess(abs(bearing), 10)

        # Test due east bearing
        lat1, lon1 = 37.0, -122.0
        lat2, lon2 = 37.0, -121.0  # Due east

        bearing = GeoUtils.bearing(lat1, lon1, lat2, lon2)

        # Should be close to 90 degrees (due east)
        self.assertGreater(bearing, 80)
        self.assertLess(bearing, 100)


if __name__ == '__main__':
    unittest.main()
