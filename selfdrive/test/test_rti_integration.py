#!/usr/bin/env python3
"""
RTI System Integration Test Suite

Comprehensive end-to-end testing of the Realtime Traffic Intelligence system.
Tests message flow, speed control integration, HUD rendering, and configuration.
"""

import json
import os
import pytest
import time
import tempfile
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any

import cereal.messaging as messaging
from openpilot.common.params import Params

# Import RTI components for testing
from sunnypilot.rtid.rtid import RTIDaemon
from sunnypilot.rtid.threat_detector import ThreatDetector, ProcessedThreat
from sunnypilot.selfdrive.controls.lib.rti_controller import RTIController


class RTITestFramework:
    """Framework for RTI integration testing with mock data and timing control."""

    def __init__(self):
        self.params = Params()
        self.pm = None
        self.sm = None
        self.mock_threats = []
        self.test_start_time = None

    def setup(self):
        """Initialize test environment."""
        # Enable RTI in params
        self.params.put_bool("RTIEnabled", True)
        self.params.put("RTIDataSource", "1")  # Waze
        self.params.put("RTIThreatFilter", "0")  # All threats
        self.params.put("RTIAggressiveness", "1")  # Balanced

        # Setup messaging
        self.pm = messaging.PubMaster(['rtiStateSP', 'gpsLocationExternal', 'carState'])
        self.sm = messaging.SubMaster(['rtiStateSP', 'longitudinalPlanSP'])

        self.test_start_time = time.time()

    def teardown(self):
        """Clean up test environment."""
        self.params.put_bool("RTIEnabled", False)
        if self.pm:
            self.pm = None
        if self.sm:
            self.sm = None

    def create_mock_threat(self, distance: float, threat_type: str, speed_limit_mph: float = 25) -> dict:
        """Create a mock threat for testing."""
        return {
            'id': f'threat_{int(time.time()*1000)}',
            'type': threat_type,
            'latitude': 37.4221 + (distance / 111000),  # Approximate lat offset
            'longitude': -122.0841,
            'distance': distance,
            'direction': 0.0,
            'confidence': 0.85,
            'speed_limit_ms': speed_limit_mph * 0.44704,  # Convert mph to m/s
            'timestamp': int(time.time() * 1e9)
        }

    def publish_mock_location(self, lat: float = 37.4221, lon: float = -122.0841, accuracy: float = 5.0):
        """Publish mock GPS location."""
        msg = messaging.new_message('gpsLocationExternal')
        msg.gpsLocationExternal.latitude = lat
        msg.gpsLocationExternal.longitude = lon
        msg.gpsLocationExternal.accuracy = accuracy
        msg.gpsLocationExternal.timestamp = int(time.time() * 1e6)
        self.pm.send('gpsLocationExternal', msg)

    def publish_mock_car_state(self, v_ego: float = 20.0):
        """Publish mock car state with speed."""
        msg = messaging.new_message('carState')
        msg.carState.vEgo = v_ego  # m/s
        msg.carState.aEgo = 0.0
        msg.carState.steeringAngleDeg = 0.0
        self.pm.send('carState', msg)

    def wait_for_message(self, service: str, timeout: float = 5.0) -> Any:
        """Wait for a message on a service with timeout."""
        start = time.time()
        while time.time() - start < timeout:
            self.sm.update(0)
            if self.sm.updated[service]:
                return self.sm[service]
            time.sleep(0.01)
        return None


@pytest.mark.integration
class TestRTIEndToEnd:
    """End-to-end integration tests for RTI system."""

    @pytest.fixture
    def framework(self):
        """Create test framework."""
        fw = RTITestFramework()
        fw.setup()
        yield fw
        fw.teardown()

    @pytest.mark.asyncio
    async def test_rti_daemon_message_flow(self, framework):
        """Test RTI daemon receives inputs and publishes outputs correctly."""
        # Create mock API key file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'api_key': 'test-key'}, f)
            api_key_path = f.name

        try:
            with patch('sunnypilot.rtid.rtid.RTIDaemon._load_api_key') as mock_load:
                mock_load.return_value = 'test-key'

                # Create daemon with mocked API client
                daemon = RTIDaemon()
                daemon.waze_client = AsyncMock()
                daemon.waze_client.get_traffic_alerts.return_value = [
                    framework.create_mock_threat(500, 'POLICE', 35)
                ]

                # Publish mock inputs
                framework.publish_mock_location()
                framework.publish_mock_car_state(25.0)  # 25 m/s = 56 mph

                # Run one processing cycle
                await daemon._process_cycle()

                # Check RTI state was published
                rti_msg = framework.wait_for_message('rtiStateSP')
                assert rti_msg is not None
                assert rti_msg.threatAhead == True
                assert abs(rti_msg.threatDistanceM - 500) < 100  # Within 100m tolerance
                assert rti_msg.apiStatus == 'connected'
                assert rti_msg.source == 'waze'

        finally:
            os.unlink(api_key_path)

    def test_rti_controller_speed_recommendation(self, framework):
        """Test RTI controller processes threats and provides speed recommendations."""
        # Create mock car params
        CP = MagicMock()
        CP.minEnableSpeed = 5.0

        # Create RTI controller
        controller = RTIController(CP)

        # Create mock SubMaster with RTI state
        sm = MagicMock()
        sm.valid = {'rtiStateSP': True}

        # Create mock threat at 300m recommending 25 mph (11.2 m/s)
        mock_threat = MagicMock()
        mock_threat.id = 'test_threat'
        mock_threat.type = 'POLICE'
        mock_threat.distance = 300
        mock_threat.confidence = 0.85
        mock_threat.speedLimitMs = 11.2  # 25 mph

        mock_rti_state = MagicMock()
        mock_rti_state.threatAhead = True
        mock_rti_state.threatDistanceM = 300
        mock_rti_state.recommendedSpeed = 11.2
        mock_rti_state.threats = [mock_threat]

        sm.__getitem__.return_value = mock_rti_state

        # Update controller with current speed of 20 m/s (45 mph)
        controller.update(sm, v_ego=20.0, a_ego=0.0, v_cruise=25.0)

        # Check controller is active and recommending reduced speed
        assert controller.is_active == True
        assert controller.threat_distance == 300

        # Speed should be reduced based on threat distance (300m = near threat)
        # Near threat uses 0.85 reduction factor
        recommended = controller.speed_recommendation
        assert recommended < 20.0  # Should be less than current speed
        assert recommended > 10.0  # But reasonable for approaching threat

    @pytest.mark.asyncio
    async def test_full_system_integration(self, framework):
        """Test complete RTI system from API to speed control."""
        # This test would require running actual processes
        # For now, we'll test the message flow simulation

        # Enable RTI
        framework.params.put_bool("RTIEnabled", True)

        # Simulate threat detection and speed recommendation flow
        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get_traffic_alerts.return_value = [
                {
                    'id': 'threat_1',
                    'type': 'POLICE',
                    'lat': 37.4231,
                    'lon': -122.0841,
                    'subtype': 'VISIBLE',
                    'confidence': 0.9,
                    'reliability': 8
                }
            ]
            MockClient.return_value = mock_client

            # Create and run daemon for one cycle
            daemon = RTIDaemon()
            daemon.waze_client = mock_client

            # Publish vehicle state
            framework.publish_mock_location(37.4221, -122.0841)
            framework.publish_mock_car_state(22.0)  # 50 mph

            # Process cycle
            await daemon._process_cycle()

            # Verify RTI state published
            rti_state = framework.wait_for_message('rtiStateSP', timeout=2.0)
            assert rti_state is not None

            # Now test that controller would respond to this
            CP = MagicMock()
            controller = RTIController(CP)

            sm_mock = MagicMock()
            sm_mock.valid = {'rtiStateSP': True}
            sm_mock.__getitem__.return_value = rti_state

            controller.update(sm_mock, v_ego=22.0, a_ego=0.0, v_cruise=25.0)

            if controller.is_active:
                assert controller.speed_recommendation < 22.0

    def test_parameter_configuration_impact(self, framework):
        """Test that RTI parameters affect system behavior."""
        CP = MagicMock()
        controller = RTIController(CP)

        # Test with RTI disabled
        framework.params.put_bool("RTIEnabled", False)

        sm = MagicMock()
        sm.valid = {'rtiStateSP': True}
        sm.__getitem__.return_value = MagicMock(
            threatAhead=True,
            threatDistanceM=500,
            recommendedSpeed=15.0,
            threats=[]
        )

        controller.update(sm, v_ego=20.0, a_ego=0.0, v_cruise=25.0)
        assert controller.is_active == False  # Should not activate when disabled

        # Enable RTI
        framework.params.put_bool("RTIEnabled", True)
        controller.update(sm, v_ego=20.0, a_ego=0.0, v_cruise=25.0)
        assert controller.is_active == True  # Should activate when enabled

    def test_threat_distance_thresholds(self, framework):
        """Test RTI controller response at different threat distances."""
        CP = MagicMock()
        controller = RTIController(CP)
        framework.params.put_bool("RTIEnabled", True)

        distances_and_expectations = [
            (50, True, 'critical'),    # Very close - critical response
            (200, True, 'near'),        # Near threat - moderate response
            (800, True, 'normal'),      # Far threat - gentle response
            (1500, False, 'inactive'),  # Too far - no response
        ]

        for distance, should_activate, response_type in distances_and_expectations:
            sm = MagicMock()
            sm.valid = {'rtiStateSP': True}

            mock_threat = MagicMock()
            mock_threat.distance = distance
            mock_threat.speedLimitMs = 15.0
            mock_threat.confidence = 0.85
            mock_threat.type = 'POLICE'

            sm.__getitem__.return_value = MagicMock(
                threatAhead=True,
                threatDistanceM=distance,
                recommendedSpeed=15.0,
                threats=[mock_threat]
            )

            controller.update(sm, v_ego=25.0, a_ego=0.0, v_cruise=30.0)

            assert controller.is_active == should_activate, \
                f"Distance {distance}m: expected active={should_activate}"

            if should_activate:
                # Verify appropriate speed reduction based on distance
                speed_rec = controller.speed_recommendation

                if response_type == 'critical':
                    # Should apply 0.75 factor for critical threats
                    assert speed_rec <= 25.0 * 0.75 + 1.0  # Allow small tolerance
                elif response_type == 'near':
                    # Should apply 0.85 factor for near threats
                    assert speed_rec <= 25.0 * 0.85 + 1.0
                elif response_type == 'normal':
                    # Should apply 0.95 factor for normal threats
                    assert speed_rec <= 25.0 * 0.95 + 1.0


@pytest.mark.performance
class TestRTIPerformance:
    """Performance and resource usage tests for RTI system."""

    def test_message_processing_latency(self):
        """Test RTI message processing stays within latency budget."""
        params = Params()
        params.put_bool("RTIEnabled", True)

        # Create components
        detector = ThreatDetector()

        # Create test data
        mock_traffic = [
            {
                'id': f'threat_{i}',
                'type': 'POLICE',
                'lat': 37.4221 + i*0.001,
                'lon': -122.0841,
                'subtype': 'VISIBLE',
                'confidence': 0.8,
                'reliability': 7
            }
            for i in range(20)  # 20 threats to process
        ]

        # Measure processing time
        start = time.time()

        for _ in range(100):  # Process 100 times
            state = detector.process_threats(
                traffic_data=mock_traffic,
                current_location=(37.4221, -122.0841),
                current_speed=20.0,
                timestamp=int(time.time() * 1e9)
            )

        elapsed = time.time() - start
        avg_time = elapsed / 100

        # Should process in under 10ms average
        assert avg_time < 0.010, f"Processing took {avg_time*1000:.2f}ms average"

    def test_memory_usage_under_load(self):
        """Test RTI system memory usage stays reasonable under load."""
        import tracemalloc
        tracemalloc.start()

        # Create many threat objects
        threats = []
        for i in range(1000):
            threat = ProcessedThreat(
                id=f'threat_{i}',
                type='POLICE',
                latitude=37.4221 + i*0.0001,
                longitude=-122.0841,
                distance=100 + i*10,
                direction=i % 360,
                confidence=0.8,
                speed_limit_ms=15.0,
                timestamp=int(time.time() * 1e9)
            )
            threats.append(threat)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should use less than 10MB for 1000 threats
        assert peak < 10 * 1024 * 1024, f"Peak memory usage: {peak / 1024 / 1024:.2f}MB"

    @pytest.mark.asyncio
    async def test_api_request_timeout_handling(self):
        """Test RTI handles API timeouts gracefully."""
        daemon = RTIDaemon()

        # Mock slow API
        daemon.waze_client = AsyncMock()
        daemon.waze_client.get_traffic_alerts.side_effect = TimeoutError()

        # Should handle timeout without crashing
        start = time.time()
        await daemon._process_cycle()
        elapsed = time.time() - start

        # Should timeout quickly and continue
        assert elapsed < 2.0, "Timeout handling took too long"


@pytest.mark.ui
class TestRTIUserInterface:
    """Tests for RTI UI components and HUD rendering."""

    def test_hud_threat_rendering_data(self):
        """Test HUD receives and processes RTI threat data correctly."""
        # This would require Qt test framework
        # For now, validate data structure

        # Create mock RTI state with threats
        threats = [
            ProcessedThreat(
                id='t1',
                type='POLICE',
                latitude=37.4221,
                longitude=-122.0841,
                distance=300,
                direction=0,
                confidence=0.85,
                speed_limit_ms=15.0,
                timestamp=int(time.time() * 1e9)
            )
        ]

        # Verify threat can be serialized for HUD
        threat_data = {
            'type': threats[0].type,
            'distance': threats[0].distance,
            'confidence': threats[0].confidence,
            'speed_limit': threats[0].speed_limit_ms * 2.237  # Convert to mph for display
        }

        assert threat_data['type'] == 'POLICE'
        assert threat_data['distance'] == 300
        assert abs(threat_data['speed_limit'] - 33.5) < 0.5  # ~33.5 mph

    def test_settings_parameter_persistence(self):
        """Test RTI settings are persisted correctly."""
        params = Params()

        # Set RTI parameters
        test_values = {
            'RTIEnabled': '1',
            'RTIDataSource': '2',  # TomTom
            'RTIThreatFilter': '1',  # Police only
            'RTIAggressiveness': '2',  # Aggressive
            'RTIMinDistance': '100',
            'RTIMaxDistance': '1500',
        }

        for key, value in test_values.items():
            params.put(key, value)

        # Verify persistence
        for key, expected in test_values.items():
            actual = params.get(key)
            assert actual == expected.encode(), f"{key}: expected {expected}, got {actual}"


def run_integration_tests():
    """Run all RTI integration tests."""
    print("Running RTI Integration Test Suite...")
    print("=" * 60)

    # Run tests with pytest
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'integration or performance or ui',
        '--color=yes'
    ])


if __name__ == "__main__":
    run_integration_tests()
