#!/usr/bin/env python3
"""
Integration tests for mapd with road geometry enhancements.

Tests that the mapd system still functions correctly with the new
road geometry features without breaking existing functionality.
"""

import unittest
import time
import json
from unittest.mock import patch, MagicMock

from openpilot.sunnypilot.mapd.live_map_data.osm_map_data import OsmMapData
from openpilot.sunnypilot.navd.helpers import Coordinate


def _make_mock_params():
    params = MagicMock()

    def _get(key, *args, **kwargs):
        if key == "LastGPSPosition":
            return "{}"
        if key in ("MapSpeedLimit", "NextMapSpeedLimit", "RoadName", "MapWindingSummary"):
            return None
        return None

    params.get.side_effect = _get
    params.get_bool.return_value = False
    params.put.return_value = None
    params.put_bool.return_value = None
    return params


class TestMapdIntegration(unittest.TestCase):
    """Test mapd integration with road geometry enhancements."""

    def setUp(self):
        """Set up test environment."""
        # Mock the hardware paths
        self.mapd_patcher = patch('openpilot.system.hardware.hw.Paths.mapd_root')
        self.mock_mapd_root = self.mapd_patcher.start()
        self.mock_mapd_root.return_value = "/tmp/test_mapd"

        # Mock messaging components
        self.messaging_patcher = patch('openpilot.sunnypilot.mapd.live_map_data.base_map_data.messaging')
        self.mock_messaging = self.messaging_patcher.start()

        # Create mock message objects
        self.mock_pm = MagicMock()
        self.mock_sm = MagicMock()
        self.mock_messaging.PubMaster.return_value = self.mock_pm
        self.mock_messaging.SubMaster.return_value = self.mock_sm
        self.mock_messaging.new_message.return_value = MagicMock()

        # Mock GPS service
        self.gps_patcher = patch('openpilot.sunnypilot.mapd.live_map_data.base_map_data.get_gps_location_service')
        self.mock_gps_service = self.gps_patcher.start()
        self.mock_gps_service.return_value = 'gpsLocationExternal'

        # Mock params used in both BaseMapData and OsmMapData modules
        self.params_patcher_base = patch('openpilot.sunnypilot.mapd.live_map_data.base_map_data.Params')
        self.mock_params_base = self.params_patcher_base.start()
        self.params_patcher_osm = patch('openpilot.sunnypilot.mapd.live_map_data.osm_map_data.Params')
        self.mock_params_osm = self.params_patcher_osm.start()
        params_instance = _make_mock_params()
        self.mock_params_base.return_value = params_instance
        self.mock_params_osm.return_value = params_instance

    def tearDown(self):
        """Clean up test environment."""
        self.mapd_patcher.stop()
        self.messaging_patcher.stop()
        self.gps_patcher.stop()
        self.params_patcher_base.stop()
        self.params_patcher_osm.stop()

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
        self.assertEqual(map_data.get_winding_road_summary(), {})

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

    def test_update_location_writes_last_gps_with_bearing(self):
        """LastGPSPosition should include bearing for mapd one-way matching."""
        map_data = OsmMapData()
        map_data.mem_params = MagicMock()

        map_data.last_position = Coordinate(38.73152, -120.78821)
        map_data.last_altitude = 1012.5

        # Mock GPS feed with bearing from cereal gpsLocation service
        gps_msg = MagicMock()
        gps_msg.bearingDeg = 274.83
        map_data.sm = MagicMock()
        map_data.sm.__getitem__.return_value = gps_msg

        # Isolate this test from road geometry extractor side effects
        with patch.object(map_data, "_update_road_geometry", return_value=None):
            map_data.update_location()

        # Validate payload written to LastGPSPosition includes bearing
        self.assertTrue(map_data.mem_params.put.called)
        args, _kwargs = map_data.mem_params.put.call_args
        self.assertGreaterEqual(len(args), 2)
        self.assertEqual(args[0], "LastGPSPosition")
        payload = json.loads(args[1])
        self.assertAlmostEqual(payload.get("latitude"), 38.73152, places=5)
        self.assertAlmostEqual(payload.get("longitude"), -120.78821, places=5)
        self.assertAlmostEqual(payload.get("altitude"), 1012.5, places=1)
        self.assertAlmostEqual(payload.get("bearing"), 274.83, places=2)

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
            self.assertFalse(mock_live_map_data.windingRoadValid)

        except Exception as e:
            self.fail(f"publish failed with no road data: {e}")

    def test_json_params_are_parsed_for_speed_limit_and_winding_summary(self):
        map_data = OsmMapData()
        map_data.last_position = Coordinate(38.73152, -120.78821)
        map_data.mem_params = MagicMock()

        next_speed_payload = {
            "speedlimit": 17.5,
            "latitude": 38.73200,
            "longitude": -120.78900,
        }
        winding_payload = {
            "valid": True,
            "level": 4,
            "score": 210,
            "confidence": 190,
            "currentLevel": 2,
            "currentScore": 120,
            "currentConfidence": 180,
            "wayCount": 3,
        }

        def _get(key, *args, **kwargs):
            if key == "NextMapSpeedLimit":
                return json.dumps(next_speed_payload).encode("utf-8")
            if key == "MapWindingSummary":
                return json.dumps(winding_payload).encode("utf-8")
            return None

        map_data.mem_params.get.side_effect = _get

        next_speed, distance = map_data.get_next_speed_limit_and_distance()
        self.assertAlmostEqual(next_speed, 17.5)
        self.assertGreater(distance, 0.0)
        self.assertEqual(map_data.get_winding_road_summary(), winding_payload)

    def test_publish_includes_winding_summary_fields(self):
        map_data = OsmMapData()
        map_data.get_winding_road_summary = MagicMock(return_value={
            "valid": True,
            "level": 4,
            "score": 205,
            "confidence": 200,
            "currentLevel": 2,
            "currentScore": 118,
            "currentConfidence": 176,
            "wayCount": 3,
        })

        mock_msg = MagicMock()
        mock_live_map_data = MagicMock()
        mock_msg.liveMapDataSP = mock_live_map_data
        self.mock_messaging.new_message.return_value = mock_msg
        self.mock_sm.all_checks.return_value = True

        map_data.publish()

        self.assertTrue(mock_live_map_data.windingRoadValid)
        self.assertEqual(mock_live_map_data.windingRoadLevel, 4)
        self.assertEqual(mock_live_map_data.windingRoadScore, 205)
        self.assertEqual(mock_live_map_data.windingRoadConfidence, 200)
        self.assertEqual(mock_live_map_data.windingRoadCurrentLevel, 2)
        self.assertEqual(mock_live_map_data.windingRoadCurrentScore, 118)
        self.assertEqual(mock_live_map_data.windingRoadCurrentConfidence, 176)
        self.assertEqual(mock_live_map_data.windingRoadWayCount, 3)

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
        msg_patch = patch('openpilot.sunnypilot.mapd.live_map_data.base_map_data.messaging')
        self.mock_messaging = msg_patch.start()
        self.mock_pm = MagicMock()
        self.mock_sm = MagicMock()
        self.mock_messaging.PubMaster.return_value = self.mock_pm
        self.mock_messaging.SubMaster.return_value = self.mock_sm
        self.patches.append(msg_patch)

        # Mock GPS
        gps_patch = patch('openpilot.sunnypilot.mapd.live_map_data.base_map_data.get_gps_location_service')
        self.mock_gps_service = gps_patch.start()
        self.mock_gps_service.return_value = 'gpsLocationExternal'
        self.patches.append(gps_patch)

        # Mock params used in both BaseMapData and OsmMapData modules
        params_base_patch = patch('openpilot.sunnypilot.mapd.live_map_data.base_map_data.Params')
        self.mock_params_base = params_base_patch.start()
        self.patches.append(params_base_patch)
        params_osm_patch = patch('openpilot.sunnypilot.mapd.live_map_data.osm_map_data.Params')
        self.mock_params_osm = params_osm_patch.start()
        self.patches.append(params_osm_patch)
        params_instance = _make_mock_params()
        self.mock_params_base.return_value = params_instance
        self.mock_params_osm.return_value = params_instance

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
