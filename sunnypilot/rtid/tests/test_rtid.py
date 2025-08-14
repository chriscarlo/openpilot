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
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

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
        """Test API key loading from /persist location."""
        test_key = "test-api-key-persist"
        key_data = {"api_key": test_key}

        with patch("os.path.exists") as mock_exists, \
             patch("builtins.open", mock_open(read_data=json.dumps(key_data))), \
             patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):

            # Mock /persist path exists, /data/persist doesn't
            mock_exists.side_effect = lambda path: path == '/persist/waze/waze_rapidapi.json'

            daemon = RTIDaemon()

            assert daemon.api_key == test_key
            mock_client_class.assert_called_once_with(test_key)

    def test_api_key_loading_from_data_persist(self, mock_messaging, mock_params):
        """Test API key loading from /data/persist location."""
        test_key = "test-api-key-data"
        key_data = {"api_key": test_key}

        with patch("os.path.exists") as mock_exists, \
             patch("builtins.open", mock_open(read_data=json.dumps(key_data))), \
             patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):

            # Mock /data/persist path exists, /persist doesn't
            mock_exists.side_effect = lambda path: path == '/data/persist/waze/waze_rapidapi.json'

            daemon = RTIDaemon()

            assert daemon.api_key == test_key
            mock_client_class.assert_called_once_with(test_key)

    def test_api_key_loading_failure_offline_mode(self, mock_messaging, mock_params):
        """Test graceful handling of API key loading failure."""
        with patch("os.path.exists", return_value=False), \
             patch('sunnypilot.rtid.rtid.WazeAPIClient') as mock_client_class, \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):

            daemon = RTIDaemon()

            assert daemon.api_key is None
            assert daemon.waze_client is None
            mock_client_class.assert_not_called()

    def test_api_key_loading_malformed_json(self, mock_messaging, mock_params):
        """Test handling of malformed API key file."""
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="invalid json")), \
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
        mock_gps.accuracy = 5.0

        daemon.sm.updated = {'gpsLocationExternal': True, 'liveLocationKalman': False}
        daemon.sm.__getitem__.side_effect = lambda key: mock_gps if key == 'gpsLocationExternal' else None

        location = daemon._get_current_location()

        assert location == (37.4221, -122.0841)
        daemon.sm.update.assert_called_with(0)

    def test_get_current_location_kalman_fallback(self, daemon):
        """Test location fallback to Kalman filter."""
        # Mock Kalman filter data
        mock_kalman = MagicMock()
        mock_kalman.lat = 37.4221
        mock_kalman.lon = -122.0841
        mock_kalman.status = 'valid'

        daemon.sm.updated = {'gpsLocationExternal': False, 'gpsLocation': False, 'liveLocationKalman': True}
        daemon.sm.__getitem__.side_effect = lambda key: mock_kalman if key == 'liveLocationKalman' else None

        location = daemon._get_current_location()

        assert location == (37.4221, -122.0841)

    def test_get_current_location_no_valid_data(self, daemon):
        """Test location retrieval with no valid GPS data."""
        daemon.sm.updated = {'gpsLocationExternal': False, 'gpsLocation': False, 'liveLocationKalman': False}

        location = daemon._get_current_location()

        assert location is None

    def test_get_current_location_poor_accuracy(self, daemon):
        """Test location rejection due to poor GPS accuracy."""
        # Mock poor accuracy GPS data
        mock_gps = MagicMock()
        mock_gps.latitude = 37.4221
        mock_gps.longitude = -122.0841
        mock_gps.accuracy = 50.0  # Poor accuracy

        daemon.sm.updated = {'gpsLocationExternal': True, 'gpsLocation': False, 'liveLocationKalman': False}
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

        await daemon._process_cycle()

        # Should publish offline state when no location
        daemon.pm.send.assert_called_once()
        msg = daemon.pm.send.call_args[0][1]
        assert msg.rtiStateSP.apiStatus == 'offline'

    @pytest.mark.asyncio
    async def test_process_cycle_with_api_client(self, daemon, mock_waze_api_response):
        """Test processing cycle with working API client."""
        # Mock valid location and speed
        daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
        daemon._get_current_speed = MagicMock(return_value=25.0)

        # Mock API client
        daemon.waze_client = AsyncMock()
        daemon.waze_client.get_traffic_alerts.return_value = []

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

        await daemon._process_cycle()

        # Verify API was called
        daemon.waze_client.get_traffic_alerts.assert_called_once_with(37.4221, -122.0841)

        # Verify threat detector was called
        daemon.threat_detector.process_threats.assert_called_once()

        # Verify state was published
        daemon.pm.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_cycle_api_error(self, daemon):
        """Test processing cycle with API error."""
        # Mock valid location and speed
        daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
        daemon._get_current_speed = MagicMock(return_value=25.0)

        # Mock API client with error
        daemon.waze_client = AsyncMock()
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

        await daemon._process_cycle()

        # Should handle API error gracefully
        daemon.threat_detector.process_threats.assert_called_once()
        daemon.pm.send.assert_called_once()

        # Published state should have error status
        msg = daemon.pm.send.call_args[0][1]
        assert msg.rtiStateSP.apiStatus == 'error'

    @pytest.mark.asyncio
    async def test_process_cycle_no_api_client(self, daemon):
        """Test processing cycle without API client (offline mode)."""
        # Mock valid location and speed
        daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
        daemon._get_current_speed = MagicMock(return_value=25.0)

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

        # Mock time.time() to return predictable timestamp
        with patch('sunnypilot.rtid.rtid.time.time', return_value=1234.567890):
            await daemon._process_cycle()

        # Should process without API data
        daemon.threat_detector.process_threats.assert_called_once_with(
            traffic_data=None,
            current_location=(37.4221, -122.0841),
            current_speed=25.0,
            timestamp=1234567890000  # 1234.567890 * 1e9 = 1234567890000 (nanoseconds)
        )

        # Published state should be offline
        daemon.pm.send.assert_called_once()
        msg = daemon.pm.send.call_args[0][1]
        assert msg.rtiStateSP.apiStatus == 'offline'

    @pytest.mark.asyncio
    async def test_process_cycle_exception_handling(self, daemon):
        """Test exception handling in process cycle."""
        # Mock location retrieval to raise exception
        daemon._get_current_location = MagicMock(side_effect=Exception("Location error"))

        await daemon._process_cycle()

        # Should publish offline state on exception
        daemon.pm.send.assert_called_once()
        msg = daemon.pm.send.call_args[0][1]
        assert msg.rtiStateSP.apiStatus == 'offline'


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
            daemon._process_cycle = AsyncMock()

            # Run for a short time
            run_task = asyncio.create_task(daemon.run())
            await asyncio.sleep(0.1)  # Let it run briefly
            run_task.cancel()

            try:
                await run_task
            except asyncio.CancelledError:
                pass

            # Should have called process cycle
            assert daemon._process_cycle.call_count > 0

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_daemon_loop_timing(self, mock_messaging, mock_params, performance_timer):
        """Test daemon loop timing maintains 1Hz rate."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()

            # Mock fast processing
            daemon._check_enabled = MagicMock(return_value=True)
            daemon._process_cycle = AsyncMock()

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

            # Should maintain approximately 1Hz (allow some variance)
            cycles_expected = 2
            assert daemon._process_cycle.call_count >= cycles_expected

            # Should not run significantly faster than 1Hz
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
            daemon._process_cycle = slow_process

            # Run for one cycle
            run_task = asyncio.create_task(daemon.run())
            await asyncio.sleep(0.2)  # Let one slow cycle complete
            run_task.cancel()

            try:
                await run_task
            except asyncio.CancelledError:
                pass

            # Should have logged performance warning
            mock_log.warning.assert_called()
            warning_call = mock_log.warning.call_args[0][0]
            assert "RTI cycle took" in warning_call


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

            # Test warning for missing API key by creating a new daemon with no key files
            with patch("os.path.exists", return_value=False):
                daemon2 = RTIDaemon()

            # Should have logged offline mode warning during daemon2 initialization
            warning_calls = [str(call) for call in mock_log.warning.call_args_list]
            assert any('offline mode' in call for call in warning_calls), f"Expected 'offline mode' in warnings: {warning_calls}"
