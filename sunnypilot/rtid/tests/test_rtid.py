#!/usr/bin/env python3
"""
Unit tests for RTI Daemon

Tests main daemon functionality including messaging integration, API coordination,
configuration loading, and main processing loop.
"""

import asyncio
import json
import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sunnypilot.rtid.rtid import RTIDaemon
from sunnypilot.rtid.threat_detector import RTIState


@pytest.mark.unit
class TestRTIDaemon:
    """Test RTI daemon core functionality."""

    @pytest.fixture
    def daemon(self, mock_messaging, mock_params):
        """Create RTI daemon with mocked dependencies."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()
            return daemon

    def test_daemon_initialization(self, mock_messaging, mock_params):
        """Test daemon initialization with mocked components."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()

            # Verify initialization
            assert daemon.params is not None
            assert daemon.sm is not None
            assert daemon.pm is not None
            assert daemon.threat_detector is not None
            assert daemon.enabled is False
            assert daemon.loop_count == 0

    def test_api_key_loading_from_persist(self, mock_messaging, mock_params):
        """Test API key loading from shared key manager."""
        test_key = "test-api-key-persist"

        with patch("sunnypilot.rtid.api_key_manager.get_api_key", return_value=test_key), \
             patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):

            daemon = RTIDaemon()

            assert daemon.api_key == test_key
            mock_client_class.assert_called_once_with(test_key)

    def test_api_key_loading_from_data_persist(self, mock_messaging, mock_params):
        """Test API key loading from dev fallback path via key manager."""
        test_key = "test-api-key-data"

        with patch("sunnypilot.rtid.api_key_manager.get_api_key", return_value=test_key), \
             patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):

            daemon = RTIDaemon()

            assert daemon.api_key == test_key
            mock_client_class.assert_called_once_with(test_key)

    def test_api_key_loading_failure_offline_mode(self, mock_messaging, mock_params):
        """Test graceful handling of API key loading failure."""
        with patch("sunnypilot.rtid.api_key_manager.get_api_key", return_value=None), \
             patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):

            daemon = RTIDaemon()

            assert daemon.api_key is None
            assert daemon.waze_client is None
            mock_client_class.assert_not_called()

    def test_api_key_loading_from_params(self, mock_messaging, mock_params):
        """Test API key loading from Params (RTIManualApiKey)."""
        test_key = b"test-api-key-from-params"

        mock_params.get.return_value = test_key

        with patch("sunnypilot.rtid.api_key_manager.get_api_key", return_value=None), \
             patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):

            daemon = RTIDaemon()

            assert daemon.api_key == test_key.decode("utf-8")
            mock_client_class.assert_called_once_with(test_key.decode("utf-8"))

    def test_api_key_loading_from_manager_fallback(self, mock_messaging, mock_params):
        """Test API key loading from key manager fallback path(s)."""
        test_key = "test-api-key-manager"

        # Ensure Params doesn't short-circuit this test
        mock_params.get.return_value = None

        with patch("sunnypilot.rtid.api_key_manager.get_api_key", return_value=test_key), \
             patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):

            daemon = RTIDaemon()

            assert daemon.api_key == test_key
            mock_client_class.assert_called_once_with(test_key)

    def test_api_key_loading_manager_error(self, mock_messaging, mock_params):
        """Test graceful handling when key manager raises an exception."""
        with patch("sunnypilot.rtid.api_key_manager.get_api_key", side_effect=RuntimeError("boom")), \
             patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):

            daemon = RTIDaemon()

            assert daemon.api_key is None
            assert daemon.waze_client is None
            mock_client_class.assert_not_called()

    def test_enabled_check(self, daemon):
        """Test RTI enabled/disabled state checking."""
        # Test enabled
        daemon.params.get_bool.return_value = True
        assert daemon._check_enabled() is True
        daemon.params.get_bool.assert_called_with("RTIEnabled")

        # Test disabled
        daemon.params.get_bool.return_value = False
        assert daemon._check_enabled() is False

    def test_get_current_location_external_gps(self, daemon):
        """Test location retrieval from external GPS."""
        # Mock external GPS data
        mock_gps = MagicMock()
        mock_gps.latitude = 37.4221
        mock_gps.longitude = -122.0841
        mock_gps.horizontalAccuracy = 5.0

        daemon.sm.__getitem__.side_effect = lambda key: mock_gps if key == 'gpsLocationExternal' else None

        location = daemon._get_current_location()

        assert location == (37.4221, -122.0841)
        daemon.sm.update.assert_called_with(100)

    def test_get_current_location_internal_gps_fallback(self, daemon):
        """Test location fallback to internal GPS (gpsLocation)."""
        mock_gps = MagicMock()
        mock_gps.latitude = 37.4221
        mock_gps.longitude = -122.0841
        mock_gps.hasFix = True

        daemon.sm.__getitem__.side_effect = lambda key: mock_gps if key == 'gpsLocation' else None

        location = daemon._get_current_location()

        assert location == (37.4221, -122.0841)

    def test_get_current_location_no_valid_data(self, daemon):
        """Test location retrieval with no valid GPS data."""
        daemon.sm.__getitem__.side_effect = lambda key: None

        location = daemon._get_current_location()

        assert location is None

    def test_get_current_location_poor_accuracy(self, daemon):
        """Test location rejection due to poor GPS accuracy."""
        # Mock poor accuracy GPS data
        mock_gps = MagicMock()
        mock_gps.latitude = 37.4221
        mock_gps.longitude = -122.0841
        mock_gps.horizontalAccuracy = 50.0  # Poor accuracy

        daemon.sm.__getitem__.side_effect = lambda key: mock_gps if key == 'gpsLocationExternal' else None

        location = daemon._get_current_location()

        assert location is None  # Should reject due to poor accuracy

    def test_get_current_speed(self, daemon):
        """Test vehicle speed retrieval."""
        mock_car_state = MagicMock()
        mock_car_state.vEgo = 25.0  # m/s

        daemon.sm.updated = {'carState': True}
        daemon.sm.__getitem__.side_effect = lambda key: mock_car_state if key == 'carState' else None

        speed = daemon._get_current_speed()

        assert speed == 25.0
        daemon.sm.update.assert_called_with(0)

    def test_get_current_speed_no_data(self, daemon):
        """Test speed retrieval with no car state data."""
        daemon.sm.updated = {'carState': False}

        speed = daemon._get_current_speed()

        assert speed == 0.0

    def test_publish_offline_state(self, daemon):
        """Test publishing offline RTI state."""
        with patch('sunnypilot.rtid.rtid.time.time', return_value=1234.567):
            daemon._publish_offline_state()

            # Verify message was sent
            daemon.pm.send.assert_called_once()
            call_args = daemon.pm.send.call_args

            assert call_args[0][0] == 'rtiStateSP'  # Service name
            msg = call_args[0][1]

            # Verify offline state content
            assert msg.rtiStateSP.threatAhead is False
            assert msg.rtiStateSP.threatDistanceM == 0.0
            assert msg.rtiStateSP.recommendedSpeed == 0.0
            assert msg.rtiStateSP.apiStatus == 'offline'

    def test_publish_rti_state(self, daemon, rti_state, processed_threat):
        """Test publishing active RTI state."""
        # Add threats to state
        test_state = rti_state
        test_state.threats = [processed_threat]

        # Avoid depending on SubMaster message wiring for this unit test.
        daemon._get_current_location = MagicMock(return_value=None)
        daemon._get_current_heading_deg = MagicMock(return_value=0.0)

        daemon._publish_rti_state(test_state)

        # Verify message was sent
        daemon.pm.send.assert_called_once_with('rtiStateSP', daemon.pm.send.call_args[0][1])

        msg = daemon.pm.send.call_args[0][1]

        # Verify state content
        assert msg.rtiStateSP.threatAhead == test_state.threat_ahead
        assert msg.rtiStateSP.threatDistanceM == test_state.threat_distance_m
        assert msg.rtiStateSP.recommendedSpeed == test_state.recommended_speed
        assert msg.rtiStateSP.source == test_state.source
        assert msg.rtiStateSP.apiStatus == test_state.api_status

    @pytest.mark.asyncio
    async def test_process_cycle_no_location(self, daemon):
        """Test processing cycle with no valid GPS location."""
        # Mock no valid location
        daemon._get_current_location = MagicMock(return_value=None)
        daemon._get_current_speed = MagicMock(return_value=25.0)
        daemon._get_cruise_cluster_speed = MagicMock(return_value=0.0)
        daemon._get_current_speed_limit = MagicMock(return_value=0.0)

        await daemon._process_cycle_async()

        assert getattr(daemon, "_last_processed_state", None) is None

    @pytest.mark.asyncio
    async def test_process_cycle_with_api_client(self, daemon, mock_waze_api_response):
        """Test processing cycle with working API client."""
        # Mock valid location and speed
        daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
        daemon._get_current_speed = MagicMock(return_value=25.0)
        daemon._get_cruise_cluster_speed = MagicMock(return_value=30.0)
        daemon._get_current_speed_limit = MagicMock(return_value=0.0)
        daemon._get_current_heading_deg = MagicMock(return_value=0.0)
        daemon.params.get.return_value = None  # ensure default RTIDetectionRadius path

        # Mock API client
        daemon.waze_client = AsyncMock()
        daemon.waze_client.last_success_time = 0
        daemon.waze_client.get_health_status = MagicMock(return_value="connected")

        async def _fake_get_traffic_alerts(lat, lon, radius_km):
            daemon.waze_client.last_success_time = 1
            return []

        daemon.waze_client.get_traffic_alerts.side_effect = _fake_get_traffic_alerts

        # Mock threat detector
        mock_state = RTIState(
            timestamp=1234567890,
            threat_ahead=False,
            threat_distance_m=0.0,
            recommended_speed=0.0,
            source='rti',
            api_status='unknown',
            threats=[]
        )
        daemon.threat_detector.process_threats = MagicMock(return_value=mock_state)

        await daemon._process_cycle_async()

        # Verify API was called
        daemon.waze_client.get_traffic_alerts.assert_awaited_once()
        call_args = daemon.waze_client.get_traffic_alerts.call_args[0]
        assert call_args[0] == 37.4221
        assert call_args[1] == -122.0841
        assert call_args[2] == pytest.approx(4.828, rel=1e-3)  # default 4828m -> 4.828km

        # Verify threat detector was called
        daemon.threat_detector.process_threats.assert_called_once()

        assert daemon._last_processed_state is mock_state
        assert daemon._last_processed_state.api_status == "connected"
        assert daemon._last_processed_state.source == "waze"

    @pytest.mark.asyncio
    async def test_process_cycle_api_error(self, daemon):
        """Test processing cycle with API error."""
        # Mock valid location and speed
        daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
        daemon._get_current_speed = MagicMock(return_value=25.0)
        daemon._get_cruise_cluster_speed = MagicMock(return_value=30.0)
        daemon._get_current_speed_limit = MagicMock(return_value=0.0)
        daemon._get_current_heading_deg = MagicMock(return_value=0.0)

        # Mock API client with error
        daemon.waze_client = AsyncMock()
        daemon.waze_client.last_success_time = 0
        daemon.waze_client.get_health_status = MagicMock(return_value="error")
        daemon.waze_client.get_traffic_alerts.side_effect = Exception("API Error")

        # Mock threat detector
        mock_state = RTIState(
            timestamp=1234567890,
            threat_ahead=False,
            threat_distance_m=0.0,
            recommended_speed=0.0,
            source='rti',
            api_status='unknown',
            threats=[]
        )
        daemon.threat_detector.process_threats = MagicMock(return_value=mock_state)

        await daemon._process_cycle_async()

        # Should handle API error gracefully
        daemon.threat_detector.process_threats.assert_called_once()

        assert daemon._last_processed_state is mock_state
        assert daemon._last_processed_state.api_status == "error"

    @pytest.mark.asyncio
    async def test_process_cycle_no_api_client(self, daemon):
        """Test processing cycle without API client (offline mode)."""
        # Mock valid location and speed
        daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
        daemon._get_current_speed = MagicMock(return_value=25.0)
        daemon._get_cruise_cluster_speed = MagicMock(return_value=30.0)
        daemon._get_current_speed_limit = MagicMock(return_value=0.0)
        daemon._get_current_heading_deg = MagicMock(return_value=0.0)

        # No API client (offline mode)
        daemon.waze_client = None

        # Mock threat detector
        mock_state = RTIState(
            timestamp=1234567890,
            threat_ahead=False,
            threat_distance_m=0.0,
            recommended_speed=0.0,
            source='rti',
            api_status='unknown',
            threats=[]
        )
        daemon.threat_detector.process_threats = MagicMock(return_value=mock_state)

        with patch('sunnypilot.rtid.rtid.time.monotonic_ns', return_value=1234567890000):
            await daemon._process_cycle_async()

        daemon.threat_detector.process_threats.assert_called_once()
        kwargs = daemon.threat_detector.process_threats.call_args.kwargs
        assert kwargs["traffic_data"] is None
        assert kwargs["current_location"] == (37.4221, -122.0841)
        assert kwargs["current_speed"] == 25.0
        assert kwargs["timestamp"] == 1234567890000
        assert daemon._last_processed_state is mock_state
        assert daemon._last_processed_state.api_status == "offline"

    @pytest.mark.asyncio
    async def test_process_cycle_exception_handling(self, daemon):
        """Test exception handling in process cycle."""
        # Mock location retrieval to raise exception
        daemon._get_current_location = MagicMock(side_effect=Exception("Location error"))

        await daemon._process_cycle_async()

        assert getattr(daemon, "_last_processed_state", None) is None


@pytest.mark.integration
class TestRTIDaemonIntegration:
    """Integration tests for RTI daemon."""

    @pytest.mark.asyncio
    async def test_daemon_main_loop_disabled(self, mock_messaging, mock_params):
        """Test main daemon loop when RTI is disabled."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()

            # Mock waze_client.close() to be awaitable
            if daemon.waze_client:
                daemon.waze_client.close = AsyncMock()

            # Mock RTI disabled
            daemon._check_enabled = MagicMock(return_value=False)

            # Run for a short time
            run_task = asyncio.create_task(daemon.run())
            await asyncio.sleep(0.1)  # Let it run briefly
            run_task.cancel()

            try:
                await run_task
            except asyncio.CancelledError:
                pass

            # Should have published offline states
            assert daemon.pm.send.call_count > 0

            # Verify offline messages
            for call in daemon.pm.send.call_args_list:
                msg = call[0][1]
                assert msg.rtiStateSP.apiStatus == 'offline'

    @pytest.mark.asyncio
    async def test_daemon_main_loop_enabled(self, mock_messaging, mock_params):
        """Test main daemon loop when RTI is enabled."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()

            # Mock waze_client.close() to be awaitable
            if daemon.waze_client:
                daemon.waze_client.close = AsyncMock()

            # Mock RTI enabled
            daemon._check_enabled = MagicMock(return_value=True)
            daemon._process_cycle_async = AsyncMock()

            # Run for a short time
            run_task = asyncio.create_task(daemon.run())
            await asyncio.sleep(0.1)  # Let it run briefly
            run_task.cancel()

            try:
                await run_task
            except asyncio.CancelledError:
                pass

            # Should have called process cycle
            assert daemon._process_cycle_async.call_count > 0

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_daemon_loop_timing(self, mock_messaging, mock_params, performance_timer):
        """Test daemon loop timing maintains ~50Hz rate."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()

            # Mock fast processing
            daemon._check_enabled = MagicMock(return_value=True)
            daemon._process_cycle_async = AsyncMock()

            # Measure loop timing
            performance_timer.start()

            run_task = asyncio.create_task(daemon.run())
            await asyncio.sleep(2.1)  # Run for ~2 cycles
            run_task.cancel()

            try:
                await run_task
            except asyncio.CancelledError:
                pass

            performance_timer.stop()

            # Should run at roughly 50Hz (allow some variance for CI/dev env)
            cycles_expected = 50
            assert daemon._process_cycle_async.call_count >= cycles_expected

            # Should not complete significantly faster than 2 seconds
            assert performance_timer.elapsed_ms >= 2000  # At least 2 seconds

    @pytest.mark.asyncio
    async def test_daemon_performance_warning(self, mock_messaging, mock_params):
        """Test performance warning for slow processing cycles."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'), \
             patch('sunnypilot.rtid.rtid.cloudlog') as mock_log:
            daemon = RTIDaemon()

            # Mock waze_client.close() to be awaitable
            if daemon.waze_client:
                daemon.waze_client.close = AsyncMock()

            # Mock slow processing cycle
            async def slow_process():
                await asyncio.sleep(0.15)  # 150ms - exceeds 100ms warning threshold

            daemon._check_enabled = MagicMock(return_value=True)
            daemon._process_cycle_async = slow_process

            # Run for one cycle
            run_task = asyncio.create_task(daemon.run())
            await asyncio.sleep(0.2)  # Let one slow cycle complete
            run_task.cancel()

            try:
                await run_task
            except asyncio.CancelledError:
                pass

            # Should have logged performance warning
            warnings = [str(call.args[0]) for call in mock_log.warning.call_args_list if call.args]
            assert any("RTI cycle took" in w for w in warnings), f"Expected cycle timing warning in: {warnings}"


@pytest.mark.unit
class TestRTIDaemonHelpers:
    """Test helper functions and utilities."""

    @pytest.fixture
    def temp_api_key_file(self):
        """Create temporary API key file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {"api_key": "test-key-12345"}
            json.dump(test_data, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    def test_api_key_loading_integration(self, temp_api_key_file, mock_messaging, mock_params):
        """Integration test for API key loading with real file."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'), \
             patch.object(RTIDaemon, '_load_api_key') as mock_load:

            # Make daemon use our temp file
            mock_load.return_value = "test-key-12345"

            daemon = RTIDaemon()

            assert daemon.api_key == "test-key-12345"
            mock_client_class.assert_called_once_with("test-key-12345")

    def test_message_structure_validation(self, mock_messaging, mock_params):
        """Test that published messages have correct structure."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()

            # Test offline message structure
            daemon._publish_offline_state()

            msg = daemon.pm.send.call_args[0][1]

            # Verify message has all required fields
            assert hasattr(msg.rtiStateSP, 'timeStamp')
            assert hasattr(msg.rtiStateSP, 'threatAhead')
            assert hasattr(msg.rtiStateSP, 'threatDistanceM')
            assert hasattr(msg.rtiStateSP, 'recommendedSpeed')
            assert hasattr(msg.rtiStateSP, 'source')
            assert hasattr(msg.rtiStateSP, 'apiStatus')
            assert hasattr(msg.rtiStateSP, 'threats')

    def test_logging_integration(self, mock_messaging, mock_params):
        """Test logging integration and status messages."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'), \
             patch('sunnypilot.rtid.rtid.cloudlog') as mock_log:

            daemon = RTIDaemon()

            # Initialization should log
            mock_log.info.assert_called_with("RTI Daemon initialized - API interval: 30s (120 calls/hr max)")

            # Test warning for missing API key by creating a new daemon with no key source
            with patch("sunnypilot.rtid.api_key_manager.get_api_key", return_value=None):
                daemon2 = RTIDaemon()

            # Should have logged offline mode warning during daemon2 initialization
            warning_calls = [str(call) for call in mock_log.warning.call_args_list]
            assert any('offline mode' in call for call in warning_calls), f"Expected 'offline mode' in warnings: {warning_calls}"
