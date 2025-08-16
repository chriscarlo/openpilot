#!/usr/bin/env python3
"""
Integration tests for RTI System

Tests complete end-to-end flow from API data ingestion through threat processing
to speed recommendations, ensuring safety-critical behavior across components.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sunnypilot.rtid.rtid import RTIDaemon
from sunnypilot.rtid.waze_api_client import WazeAPIClient, WazeAlert
from sunnypilot.rtid.threat_detector import ThreatDetector


@pytest.mark.integration
class TestRTISystemIntegration:
    """Test complete RTI system integration."""

    @pytest.mark.asyncio
    async def test_end_to_end_threat_processing_flow(self, mock_messaging, mock_params, sample_waze_alerts):
        """Test complete flow from API data to speed recommendation."""
        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as MockClient, \
             patch('sunnypilot.rtid.rtid.ThreatDetector') as MockDetector:

            # Create real instances to test actual integration
            real_client = WazeAPIClient("test-key")
            real_detector = ThreatDetector()

            # Mock the classes to return our real instances
            MockClient.return_value = real_client
            MockDetector.return_value = real_detector

            daemon = RTIDaemon()

            # Mock valid vehicle state
            daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
            daemon._get_current_speed = MagicMock(return_value=25.0)
            daemon._check_enabled = MagicMock(return_value=True)

            # Mock API client to return test data
            with patch.object(real_client, 'get_traffic_alerts', new_callable=AsyncMock) as mock_api:
                mock_api.return_value = sample_waze_alerts

                # Execute processing cycle
                await daemon._process_cycle()

                # Verify API was called with correct location
                mock_api.assert_called_once_with(37.4221, -122.0841)

                # Verify message was published
                daemon.pm.send.assert_called_once()

                # Verify message structure
                published_msg = daemon.pm.send.call_args[0][1]
                assert hasattr(published_msg.rtiStateSP, 'threatAhead')
                assert hasattr(published_msg.rtiStateSP, 'recommendedSpeed')
                assert hasattr(published_msg.rtiStateSP, 'threats')

    @pytest.mark.asyncio
    async def test_realistic_highway_police_scenario(self, mock_messaging, mock_params):
        """Test realistic scenario: police ahead on highway."""

        # Create realistic highway police alert
        highway_police = WazeAlert(
            id='highway-police-001',
            type='police',
            latitude=37.4221 + 0.005,  # ~500m north
            longitude=-122.0841,
            confidence=0.9,
            speed_limit=65,  # 65 mph highway
            street='Highway 101',
            country='US',
            raw_data={}
        )

        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as MockClient, \
             patch('sunnypilot.rtid.rtid.ThreatDetector') as MockDetector:

            real_client = WazeAPIClient("test-key")
            real_detector = ThreatDetector()
            MockClient.return_value = real_client
            MockDetector.return_value = real_detector

            daemon = RTIDaemon()

            # Highway driving scenario
            daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
            daemon._get_current_speed = MagicMock(return_value=29.0)  # ~65 mph
            daemon._check_enabled = MagicMock(return_value=True)

            with patch.object(real_client, 'get_traffic_alerts', new_callable=AsyncMock) as mock_api:
                mock_api.return_value = [highway_police]

                await daemon._process_cycle()

                # Verify processing completed
                daemon.pm.send.assert_called_once()
                published_msg = daemon.pm.send.call_args[0][1]

                # Should have processed the threat (may or may not affect speed depending on distance/road matching)
                assert published_msg.rtiStateSP.apiStatus in ['connected', 'waze']
                assert len(published_msg.rtiStateSP.threats) <= 5  # HUD limit

    @pytest.mark.asyncio
    async def test_city_street_speed_trap_scenario(self, mock_messaging, mock_params):
        """Test city street scenario with speed trap."""

        # Create city street speed trap
        city_trap = WazeAlert(
            id='city-trap-001',
            type='speedTrap',
            latitude=37.4221 + 0.002,  # ~200m north
            longitude=-122.0841,
            confidence=0.85,
            speed_limit=25,  # 25 mph city street
            street='Main Street',
            country='US',
            raw_data={}
        )

        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as MockClient, \
             patch('sunnypilot.rtid.rtid.ThreatDetector') as MockDetector:

            real_client = WazeAPIClient("test-key")
            real_detector = ThreatDetector()
            MockClient.return_value = real_client
            MockDetector.return_value = real_detector

            daemon = RTIDaemon()

            # City driving scenario
            daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
            daemon._get_current_speed = MagicMock(return_value=15.0)  # ~33 mph
            daemon._check_enabled = MagicMock(return_value=True)

            with patch.object(real_client, 'get_traffic_alerts', new_callable=AsyncMock) as mock_api:
                mock_api.return_value = [city_trap]

                await daemon._process_cycle()

                daemon.pm.send.assert_called_once()
                published_msg = daemon.pm.send.call_args[0][1]

                # Verify safety constraints on any recommendation
                if published_msg.rtiStateSP.recommendedSpeed > 0:
                    from conftest import assert_speed_recommendation_safe
                    assert_speed_recommendation_safe(published_msg.rtiStateSP.recommendedSpeed, 15.0)

    @pytest.mark.asyncio
    async def test_multiple_threats_prioritization(self, mock_messaging, mock_params):
        """Test prioritization with multiple threats at different distances."""

        # Create multiple threats at various distances
        close_threat = WazeAlert(
            id='close-police', type='police',
            latitude=37.4221 + 0.001, longitude=-122.0841,  # ~100m
            confidence=0.8, speed_limit=35, raw_data={}
        )

        medium_threat = WazeAlert(
            id='medium-trap', type='speedTrap',
            latitude=37.4221 + 0.005, longitude=-122.0841,  # ~500m
            confidence=0.9, speed_limit=45, raw_data={}
        )

        far_threat = WazeAlert(
            id='far-accident', type='accident',
            latitude=37.4221 + 0.010, longitude=-122.0841,  # ~1000m
            confidence=0.7, speed_limit=None, raw_data={}
        )

        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as MockClient, \
             patch('sunnypilot.rtid.rtid.ThreatDetector') as MockDetector:

            real_client = WazeAPIClient("test-key")
            real_detector = ThreatDetector()
            MockClient.return_value = real_client
            MockDetector.return_value = real_detector

            daemon = RTIDaemon()

            daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
            daemon._get_current_speed = MagicMock(return_value=20.0)
            daemon._check_enabled = MagicMock(return_value=True)

            with patch.object(real_client, 'get_traffic_alerts', new_callable=AsyncMock) as mock_api:
                mock_api.return_value = [far_threat, close_threat, medium_threat]  # Unsorted

                await daemon._process_cycle()

                daemon.pm.send.assert_called_once()
                published_msg = daemon.pm.send.call_args[0][1]

                # Threats should be sorted by distance (closest first)
                if len(published_msg.rtiStateSP.threats) >= 2:
                    threat1_dist = published_msg.rtiStateSP.threats[0].distance
                    threat2_dist = published_msg.rtiStateSP.threats[1].distance
                    assert threat1_dist <= threat2_dist

    @pytest.mark.asyncio
    async def test_api_failure_graceful_degradation(self, mock_messaging, mock_params):
        """Test graceful degradation when API fails."""

        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as MockClient, \
             patch('sunnypilot.rtid.rtid.ThreatDetector') as MockDetector:

            real_client = WazeAPIClient("test-key")
            real_detector = ThreatDetector()
            MockClient.return_value = real_client
            MockDetector.return_value = real_detector

            daemon = RTIDaemon()

            daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
            daemon._get_current_speed = MagicMock(return_value=25.0)
            daemon._check_enabled = MagicMock(return_value=True)

            # Mock API failure
            with patch.object(real_client, 'get_traffic_alerts', new_callable=AsyncMock) as mock_api:
                mock_api.side_effect = Exception("Network timeout")

                await daemon._process_cycle()

                # Should still publish state with error status
                daemon.pm.send.assert_called_once()
                published_msg = daemon.pm.send.call_args[0][1]

                assert published_msg.rtiStateSP.apiStatus == 'error'
                assert published_msg.rtiStateSP.threatAhead is False
                assert published_msg.rtiStateSP.recommendedSpeed == 0.0

    @pytest.mark.asyncio
    async def test_disabled_rti_publishes_offline_state(self, mock_messaging, mock_params):
        """Test that disabled RTI still publishes offline state."""

        mock_waze_client = AsyncMock()
        mock_waze_client.close = AsyncMock()
        with patch('sunnypilot.rtid.rtid.WazeAPIClient', return_value=mock_waze_client), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()

            # RTI disabled
            daemon._check_enabled = MagicMock(return_value=False)

            # Run one cycle of main loop
            run_task = asyncio.create_task(daemon.run())
            await asyncio.sleep(0.05)  # Let one cycle run
            run_task.cancel()

            try:
                await run_task
            except asyncio.CancelledError:
                pass

            # Should have published offline state
            assert daemon.pm.send.call_count > 0
            last_msg = daemon.pm.send.call_args[0][1]
            assert last_msg.rtiStateSP.apiStatus == 'offline'

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_system_performance_under_load(self, mock_messaging, mock_params, performance_timer):
        """Test system performance with high threat count."""

        # Create many threats to stress test
        many_threats = []
        for i in range(50):  # 50 threats
            threat = WazeAlert(
                id=f'threat-{i}',
                type='police' if i % 2 == 0 else 'speedTrap',
                latitude=37.4221 + (i * 0.001),  # Spread out
                longitude=-122.0841,
                confidence=0.7 + (i % 3) * 0.1,
                speed_limit=25 + (i % 5) * 10,
                raw_data={}
            )
            many_threats.append(threat)

        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as MockClient, \
             patch('sunnypilot.rtid.rtid.ThreatDetector') as MockDetector:

            real_client = WazeAPIClient("test-key")
            real_detector = ThreatDetector()
            MockClient.return_value = real_client
            MockDetector.return_value = real_detector

            daemon = RTIDaemon()

            daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
            daemon._get_current_speed = MagicMock(return_value=25.0)
            daemon._check_enabled = MagicMock(return_value=True)

            with patch.object(real_client, 'get_traffic_alerts', new_callable=AsyncMock) as mock_api:
                mock_api.return_value = many_threats

                performance_timer.start()
                await daemon._process_cycle()
                performance_timer.stop()

                # Should complete within reasonable time even with many threats
                assert performance_timer.elapsed_ms <= 100.0, \
                    f"Performance budget exceeded: {performance_timer.elapsed_ms:.1f}ms > 100.0ms"

                daemon.pm.send.assert_called_once()


@pytest.mark.integration
@pytest.mark.safety
class TestRTISystemSafetyIntegration:
    """Test safety-critical aspects of RTI integration."""

    @pytest.mark.asyncio
    async def test_unsafe_speed_recommendation_rejected(self, mock_messaging, mock_params):
        """Test that unsafe speed recommendations are rejected at system level."""

        # Create threat that could generate unsafe recommendation
        unsafe_threat = WazeAlert(
            id='unsafe-speed-trap',
            type='speedTrap',
            latitude=37.4221 + 0.003,  # ~300m ahead
            longitude=-122.0841,
            confidence=0.9,
            speed_limit=120,  # Unreasonable 120 mph
            raw_data={}
        )

        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as MockClient, \
             patch('sunnypilot.rtid.rtid.ThreatDetector') as MockDetector:

            real_client = WazeAPIClient("test-key")
            real_detector = ThreatDetector()
            MockClient.return_value = real_client
            MockDetector.return_value = real_detector

            daemon = RTIDaemon()

            # Slow current speed makes high recommendation unsafe
            daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
            daemon._get_current_speed = MagicMock(return_value=15.0)  # Slow speed
            daemon._check_enabled = MagicMock(return_value=True)

            with patch.object(real_client, 'get_traffic_alerts', new_callable=AsyncMock) as mock_api:
                mock_api.return_value = [unsafe_threat]

                await daemon._process_cycle()

                daemon.pm.send.assert_called_once()
                published_msg = daemon.pm.send.call_args[0][1]

                # System should reject unsafe recommendation
                if published_msg.rtiStateSP.recommendedSpeed > 0:
                    # Any recommendation should be safe
                    from conftest import assert_speed_recommendation_safe
                    assert_speed_recommendation_safe(published_msg.rtiStateSP.recommendedSpeed, 15.0)

    @pytest.mark.asyncio
    async def test_gps_accuracy_safety_check(self, mock_messaging, mock_params):
        """Test that poor GPS accuracy prevents RTI operation."""

        with patch('sunnypilot.rtid.rtid.WazeAPIClient'), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()

            # Mock poor GPS accuracy
            def poor_gps_location():
                return None  # Simulates rejection due to poor accuracy

            daemon._get_current_location = MagicMock(side_effect=poor_gps_location)
            daemon._get_current_speed = MagicMock(return_value=25.0)
            daemon._check_enabled = MagicMock(return_value=True)

            await daemon._process_cycle()

            # Should publish offline state due to no valid location
            daemon.pm.send.assert_called_once()
            published_msg = daemon.pm.send.call_args[0][1]
            assert published_msg.rtiStateSP.apiStatus == 'offline'

    @pytest.mark.asyncio
    async def test_zero_speed_edge_case(self, mock_messaging, mock_params):
        """Test system behavior when vehicle speed is zero."""

        test_threat = WazeAlert(
            id='stationary-threat', type='police',
            latitude=37.4221 + 0.001, longitude=-122.0841,
            confidence=0.8, speed_limit=25, raw_data={}
        )

        with patch('sunnypilot.rtid.rtid.WazeAPIClient') as MockClient, \
             patch('sunnypilot.rtid.rtid.ThreatDetector') as MockDetector:

            real_client = WazeAPIClient("test-key")
            real_detector = ThreatDetector()
            MockClient.return_value = real_client
            MockDetector.return_value = real_detector

            daemon = RTIDaemon()

            # Vehicle is stationary
            daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
            daemon._get_current_speed = MagicMock(return_value=0.0)  # Stationary
            daemon._check_enabled = MagicMock(return_value=True)

            with patch.object(real_client, 'get_traffic_alerts', new_callable=AsyncMock) as mock_api:
                mock_api.return_value = [test_threat]

                await daemon._process_cycle()

                daemon.pm.send.assert_called_once()
                published_msg = daemon.pm.send.call_args[0][1]

                # System should handle zero speed gracefully
                assert published_msg.rtiStateSP.recommendedSpeed >= 0.0

    @pytest.mark.asyncio
    async def test_system_recovery_from_exceptions(self, mock_messaging, mock_params):
        """Test that system recovers gracefully from various exception scenarios."""

        mock_waze_client = AsyncMock()
        mock_waze_client.close = AsyncMock()
        with patch('sunnypilot.rtid.rtid.WazeAPIClient', return_value=mock_waze_client), \
             patch('sunnypilot.rtid.rtid.ThreatDetector'):
            daemon = RTIDaemon()
            daemon._check_enabled = MagicMock(return_value=True)

            # Test various exception scenarios
            exception_scenarios = [
                ("Location error", lambda: Exception("GPS failure")),
                ("Speed error", lambda: Exception("CAN bus error")),
                ("Messaging error", lambda: Exception("Message send failed"))
            ]

            for scenario_name, exception_factory in exception_scenarios:
                # Reset mocks
                daemon.pm.send.reset_mock()

                if "Location" in scenario_name:
                    daemon._get_current_location = MagicMock(side_effect=exception_factory())
                    daemon._get_current_speed = MagicMock(return_value=25.0)
                elif "Speed" in scenario_name:
                    daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
                    daemon._get_current_speed = MagicMock(side_effect=exception_factory())
                else:
                    daemon._get_current_location = MagicMock(return_value=(37.4221, -122.0841))
                    daemon._get_current_speed = MagicMock(return_value=25.0)
                    daemon.pm.send.side_effect = exception_factory()

                # Should not crash - should handle exception gracefully
                try:
                    await daemon._process_cycle()
                except Exception:
                    pytest.fail(f"System failed to handle {scenario_name} gracefully")

                # Should have attempted to publish state (even if send failed)
                if "Messaging" not in scenario_name:
                    assert daemon.pm.send.called
