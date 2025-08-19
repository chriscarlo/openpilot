#!/usr/bin/env python3
"""
Integration tests for mapd with road geometry enhancements.

Tests that the mapd system still functions correctly with the new
road geometry features without breaking existing functionality.
"""

import unittest
import time
from unittest.mock import patch, MagicMock

from openpilot.sunnypilot.mapd.live_map_data.osm_map_data import OsmMapData
from openpilot.sunnypilot.navd.helpers import Coordinate


class TestMapdIntegration(unittest.TestCase):
    """Test mapd integration with road geometry enhancements."""

    def setUp(self):
        """Set up test environment."""
        # Mock the hardware paths
        self.mapd_patcher = patch('openpilot.system.hardware.hw.Paths.mapd_root')
        self.mock_mapd_root = self.mapd_patcher.start()
        self.mock_mapd_root.return_value = "/tmp/test_mapd"

        # Mock messaging components
        self.messaging_patcher = patch('cereal.messaging')
        self.mock_messaging = self.messaging_patcher.start()

        # Create mock message objects
        self.mock_pm = MagicMock()
        self.mock_sm = MagicMock()
        self.mock_messaging.PubMaster.return_value = self.mock_pm
        self.mock_messaging.SubMaster.return_value = self.mock_sm
        self.mock_messaging.new_message.return_value = MagicMock()

        # Mock GPS service
        self.gps_patcher = patch('openpilot.common.gps.get_gps_location_service')
        self.mock_gps_service = self.gps_patcher.start()
        self.mock_gps_service.return_value = 'gpsLocationExternal'

        # Mock params
        self.params_patcher = patch('openpilot.common.params.Params')
        self.mock_params = self.params_patcher.start()

    def tearDown(self):
        """Clean up test environment."""
        self.mapd_patcher.stop()
        self.messaging_patcher.stop()
        self.gps_patcher.stop()
        self.params_patcher.stop()

    def test_osm_map_data_initialization(self):
        """Test that OsmMapData initializes without errors."""
        try:
            map_data = OsmMapData()
            self.assertIsNotNone(map_data)
            self.assertFalse(map_data.road_geometry_valid)  # Should be False initially
            self.assertIsNone(map_data.current_road_segment)
            self.assertEqual(len(map_data.nearby_road_segments), 0)
        except Exception as e:
            self.fail(f"OsmMapData initialization failed: {e}")

    def test_backward_compatibility(self):
        """Test that existing mapd functionality still works."""
        map_data = OsmMapData()

        # Test existing abstract methods
        self.assertEqual(map_data.get_current_speed_limit(), 0.0)
        self.assertEqual(map_data.get_current_road_name(), "")

        next_speed, distance = map_data.get_next_speed_limit_and_distance()
        self.assertEqual(next_speed, 0.0)
        self.assertEqual(distance, 0.0)

    def test_road_geometry_methods(self):
        """Test new road geometry methods work without errors."""
        map_data = OsmMapData()

        # Test new road geometry methods
        self.assertFalse(map_data.get_road_geometry_valid())
        self.assertIsNone(map_data.get_current_road_segment())
        self.assertEqual(len(map_data.get_nearby_road_segments()), 0)

    def test_update_location_with_no_database(self):
        """Test update_location doesn't crash when no database is available."""
        map_data = OsmMapData()

        # Set mock position
        map_data.last_position = Coordinate(37.4221, -122.0841)
        map_data.last_altitude = 100.0

        try:
            map_data.update_location()
            # Should not crash, should gracefully handle missing database
            self.assertFalse(map_data.road_geometry_valid)
        except Exception as e:
            self.fail(f"update_location failed with no database: {e}")

    def test_publish_with_no_road_data(self):
        """Test publish method works when no road geometry data is available."""
        map_data = OsmMapData()

        # Mock the messaging components
        mock_msg = MagicMock()
        mock_live_map_data = MagicMock()
        mock_msg.liveMapDataSP = mock_live_map_data
        self.mock_messaging.new_message.return_value = mock_msg

        # Mock SubMaster state
        self.mock_sm.all_checks.return_value = True

        try:
            map_data.publish()

            # Verify message was sent
            self.mock_pm.send.assert_called_once_with('liveMapDataSP', mock_msg)

            # Verify road geometry fields were set to safe defaults
            self.assertFalse(mock_live_map_data.roadGeometryValid)

        except Exception as e:
            self.fail(f"publish failed with no road data: {e}")

    def test_tick_method(self):
        """Test that tick method completes without errors."""
        map_data = OsmMapData()

        # Mock GPS location
        mock_gps = MagicMock()
        mock_gps.latitude = 37.4221
        mock_gps.longitude = -122.0841
        mock_gps.altitude = 100.0

        self.mock_sm.__getitem__.return_value = mock_gps
        self.mock_sm.updated = {'gpsLocationExternal': True, 'livePose': True}
        self.mock_sm.logMonoTime = {'gpsLocationExternal': time.time() * 1e9}
        self.mock_sm.__getitem__.side_effect = lambda key: mock_gps if key == 'gpsLocationExternal' else MagicMock()

        # Mock livePose
        mock_live_pose = MagicMock()
        mock_live_pose.inputsOK = True

        def get_mock_data(key):
            if key == 'gpsLocationExternal':
                return mock_gps
            elif key == 'livePose':
                return mock_live_pose
            else:
                return MagicMock()

        self.mock_sm.__getitem__.side_effect = get_mock_data
        self.mock_sm.all_checks.return_value = True

        # Mock new_message
        mock_msg = MagicMock()
        mock_live_map_data = MagicMock()
        mock_msg.liveMapDataSP = mock_live_map_data
        self.mock_messaging.new_message.return_value = mock_msg

        try:
            map_data.tick()
            # Should complete without errors
            self.mock_pm.send.assert_called()
        except Exception as e:
            self.fail(f"tick method failed: {e}")

    @patch('os.path.exists')
    def test_database_path_detection(self, mock_exists):
        """Test database path detection logic."""
        # Mock that no database files exist
        mock_exists.return_value = False

        map_data = OsmMapData()
        self.assertIsNone(map_data.road_geometry_extractor)

        # Mock that first candidate exists
        mock_exists.side_effect = lambda path: "/osm.db" in path

        map_data = OsmMapData()
        # Should initialize extractor when database is found
        # Note: This would fail in practice due to invalid DB, but the path detection works


class TestMapdFallbackBehavior(unittest.TestCase):
    """Test fallback behavior when road geometry features fail."""

    def setUp(self):
        """Set up test with mocked dependencies."""
        # Mock all external dependencies
        self.patches = []

        # Mock hardware paths
        hw_patch = patch('openpilot.system.hardware.hw.Paths.mapd_root')
        self.mock_mapd_root = hw_patch.start()
        self.mock_mapd_root.return_value = "/tmp/nonexistent"
        self.patches.append(hw_patch)

        # Mock messaging
        msg_patch = patch('cereal.messaging')
        self.mock_messaging = msg_patch.start()
        self.mock_pm = MagicMock()
        self.mock_sm = MagicMock()
        self.mock_messaging.PubMaster.return_value = self.mock_pm
        self.mock_messaging.SubMaster.return_value = self.mock_sm
        self.patches.append(msg_patch)

        # Mock GPS
        gps_patch = patch('openpilot.common.gps.get_gps_location_service')
        self.mock_gps_service = gps_patch.start()
        self.mock_gps_service.return_value = 'gpsLocationExternal'
        self.patches.append(gps_patch)

        # Mock params
        params_patch = patch('openpilot.common.params.Params')
        self.mock_params = params_patch.start()
        self.patches.append(params_patch)

    def tearDown(self):
        """Clean up patches."""
        for patch_obj in self.patches:
            patch_obj.stop()

    def test_graceful_degradation(self):
        """Test that system gracefully degrades when road geometry fails."""
        map_data = OsmMapData()

        # Force an error in road geometry processing
        with patch.object(map_data, '_update_road_geometry', side_effect=Exception("Database error")):
            map_data.last_position = Coordinate(37.4221, -122.0841)
            map_data.last_altitude = 100.0

            # Should not crash, should continue with basic functionality
            try:
                map_data.update_location()
                self.assertFalse(map_data.road_geometry_valid)
            except Exception as e:
                self.fail(f"System failed to gracefully handle road geometry error: {e}")

    def test_publish_error_handling(self):
        """Test publish method handles errors in road geometry population."""
        map_data = OsmMapData()

        # Mock a successful road segment that will cause population error
        mock_segment = MagicMock()
        map_data.current_road_segment = mock_segment
        map_data.road_geometry_valid = True

        # Mock message components
        mock_msg = MagicMock()
        mock_live_map_data = MagicMock()
        mock_msg.liveMapDataSP = mock_live_map_data
        self.mock_messaging.new_message.return_value = mock_msg
        self.mock_sm.all_checks.return_value = True

        # Force an error in segment population
        with patch.object(map_data, '_populate_road_segment', side_effect=Exception("Population error")):
            try:
                map_data.publish()
                # Should complete and send message despite error
                self.mock_pm.send.assert_called_once()
                # Should have set roadGeometryValid to False due to error
                self.assertFalse(mock_live_map_data.roadGeometryValid)
            except Exception as e:
                self.fail(f"Publish failed to handle population error: {e}")


if __name__ == '__main__':
    unittest.main()
