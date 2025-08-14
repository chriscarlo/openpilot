#!/usr/bin/env python3
"""
Unit tests for WazeAPIClient

Tests API client functionality including rate limiting, caching, error handling,
and data parsing following safety-critical patterns.
"""

import pytest
from unittest.mock import AsyncMock, patch

from sunnypilot.rtid.waze_api_client import WazeAPIClient, WazeAlert, RateLimiter, LRUCache


@pytest.mark.unit
class TestRateLimiter:
    """Test rate limiting functionality."""

    def test_rate_limiter_allows_requests_under_limit(self):
        limiter = RateLimiter(max_requests=5, time_window=60)

        # Should allow requests under limit
        for _ in range(5):
            assert limiter.can_request()
            limiter.record_request()

        # Should deny request at limit
        assert not limiter.can_request()

    def test_rate_limiter_resets_after_time_window(self):
        limiter = RateLimiter(max_requests=2, time_window=1)

        # Use up limit
        limiter.record_request()
        limiter.record_request()
        assert not limiter.can_request()

        # Mock time passage
        with patch('time.time', return_value=limiter.requests[0] + 2):
            assert limiter.can_request()


@pytest.mark.unit
class TestLRUCache:
    """Test LRU cache functionality."""

    def test_cache_stores_and_retrieves_values(self):
        cache = LRUCache(max_size=3)

        cache.put('key1', 'value1')
        cache.put('key2', 'value2')

        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == 'value2'
        assert cache.get('nonexistent') is None

    def test_cache_evicts_lru_when_full(self):
        cache = LRUCache(max_size=2)

        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')  # Should evict key1

        assert cache.get('key1') is None
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'

    def test_cache_updates_access_order(self):
        cache = LRUCache(max_size=2)

        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.get('key1')  # Access key1, making key2 LRU
        cache.put('key3', 'value3')  # Should evict key2

        assert cache.get('key1') == 'value1'
        assert cache.get('key2') is None
        assert cache.get('key3') == 'value3'


@pytest.mark.unit
@pytest.mark.api
class TestWazeAPIClient:
    """Test Waze API client functionality."""

    @pytest.fixture
    def api_client(self, mock_api_key):
        return WazeAPIClient(mock_api_key)

    def test_api_client_initialization(self, mock_api_key):
        client = WazeAPIClient(mock_api_key)

        assert client.api_key == mock_api_key
        assert client.rate_limiter is not None
        assert client.cache is not None
        assert client.consecutive_failures == 0

    def test_cache_key_generation(self, api_client):
        # Test cache key generation with location rounding
        key1 = api_client._get_cache_key(37.4221, -122.0841)
        key2 = api_client._get_cache_key(37.4222, -122.0842)  # Slight difference

        # Should be same key due to rounding
        assert key1 == key2

        key3 = api_client._get_cache_key(37.5000, -122.0000)  # Different tile
        assert key1 != key3

    def test_log_scrubbing(self, api_client):
        # Test API key scrubbing from logs
        test_string = f"API call failed with key {api_client.api_key}"
        scrubbed = api_client._scrub_logs(test_string)

        assert api_client.api_key not in scrubbed
        assert '[API_KEY_REDACTED]' in scrubbed

    @pytest.mark.asyncio
    async def test_successful_api_request(self, api_client, mock_waze_api_response):
        # Mock successful HTTP response
        with patch.object(api_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_waze_api_response

            alerts = await api_client.get_traffic_alerts(37.4221, -122.0841)

            assert len(alerts) > 0
            assert all(isinstance(alert, WazeAlert) for alert in alerts)
            mock_request.assert_called()

    @pytest.mark.asyncio
    async def test_api_rate_limiting(self, api_client):
        # Fill up rate limiter
        for _ in range(90):
            api_client.rate_limiter.record_request()

        # Verify rate limiter is blocking new requests
        assert not api_client.rate_limiter.can_request()

        # Test that get_traffic_alerts respects rate limiting
        alerts = await api_client.get_traffic_alerts(37.4221, -122.0841)

        # Should return empty list due to rate limiting
        assert alerts == []

    @pytest.mark.asyncio
    async def test_api_caching(self, api_client, mock_waze_api_response):
        # Test cache behavior by making two requests and ensuring the second uses cache
        fixed_time = 1640000000

        # Mock the HTTP session to control responses
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_waze_api_response)
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('time.time', return_value=fixed_time):
            # Set up the session
            api_client.session = mock_session

            # First request - should hit the API
            alerts1 = await api_client.get_traffic_alerts(37.4221, -122.0841)
            assert len(alerts1) > 0
            assert mock_session.get.call_count == 1

            # Second request with same parameters - should use cache
            alerts2 = await api_client.get_traffic_alerts(37.4221, -122.0841)
            assert len(alerts2) > 0
            # Should still be only 1 call (cached)
            assert mock_session.get.call_count == 1

            # Results should be the same
            assert alerts1 == alerts2

    @pytest.mark.asyncio
    async def test_api_error_handling(self, api_client):
        with patch.object(api_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Network error")

            alerts = await api_client.get_traffic_alerts(37.4221, -122.0841)

            assert alerts == []
            assert api_client.consecutive_failures == 0  # Internal error tracking

    @pytest.mark.asyncio
    async def test_consecutive_failure_backoff(self, api_client):
        # Simulate repeated failures
        api_client.consecutive_failures = 6
        api_client.last_success_time = 0  # Long time ago

        alerts = await api_client.get_traffic_alerts(37.4221, -122.0841)

        # Should skip API call due to consecutive failures
        assert alerts == []

    def test_alert_type_mapping(self, api_client):
        # Test various Waze alert type mappings
        assert api_client._map_alert_type('POLICE') == 'police'
        assert api_client._map_alert_type('SPEED_TRAP') == 'speedTrap'
        assert api_client._map_alert_type('ACCIDENT') == 'accident'
        assert api_client._map_alert_type('UNKNOWN_TYPE') == 'hazard'  # Default

    def test_parse_alerts_with_valid_data(self, api_client, sample_waze_alerts):
        # Test parsing of well-formed alert data
        mock_response = {'alerts': [alert.raw_data for alert in sample_waze_alerts]}

        parsed_alerts = api_client._parse_alerts(mock_response)

        assert len(parsed_alerts) >= 1
        alert = parsed_alerts[0]
        assert isinstance(alert, WazeAlert)
        assert alert.id is not None
        assert alert.latitude != 0
        assert alert.longitude != 0

    def test_parse_alerts_with_malformed_data(self, api_client):
        # Test graceful handling of malformed data with real API structure
        mock_response = {
            'alerts': [
                {'invalid': 'data'},  # Missing required fields
                {'latitude': 'invalid', 'longitude': -122.0841},  # Invalid coordinates
                {'alert_id': 'test-001', 'type': 'POLICE', 'latitude': 37.4221, 'longitude': -122.0841, 'alert_confidence': 0.8}  # Valid
            ]
        }

        parsed_alerts = api_client._parse_alerts(mock_response)

        # Should only parse the valid alert
        assert len(parsed_alerts) == 1
        assert parsed_alerts[0].type == 'police'

    def test_health_status_reporting(self, api_client):
        # Test various health status conditions
        assert api_client.get_health_status() == 'disconnected'

        api_client.last_success_time = api_client.last_success_time = 1  # Recent success
        with patch('time.time', return_value=50):
            assert api_client.get_health_status() == 'connected'

        api_client.consecutive_failures = 6
        assert api_client.get_health_status() == 'degraded'

        api_client.consecutive_failures = 11
        assert api_client.get_health_status() == 'error'

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_performance(self, api_client, performance_timer, mock_waze_api_response):
        """Test API response time performance."""
        with patch.object(api_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_waze_api_response

            performance_timer.start()
            await api_client.get_traffic_alerts(37.4221, -122.0841)
            performance_timer.stop()

            # API client should respond quickly
            assert performance_timer.elapsed_ms < 100  # 100ms budget for API client logic


@pytest.mark.integration
@pytest.mark.api
class TestWazeAPIIntegration:
    """Integration tests for Waze API client."""

    @pytest.mark.asyncio
    async def test_full_api_workflow(self, mock_api_key, mock_waze_api_response):
        """Test complete API workflow with mocked responses."""
        async with WazeAPIClient(mock_api_key) as client:
            with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_waze_api_response

                alerts = await client.get_traffic_alerts(37.4221, -122.0841)

                assert len(alerts) > 0
                for alert in alerts:
                    # Validate alert structure
                    assert alert.id
                    assert alert.type in ['police', 'policeHiding', 'speedTrap', 'speedCamera', 'accident', 'hazard', 'roadHazard', 'shoulderHazard', 'jam']
                    assert -90 <= alert.latitude <= 90
                    assert -180 <= alert.longitude <= 180
                    assert 0 <= alert.confidence <= 1

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, mock_api_key):
        """Test proper cleanup of HTTP session."""
        client = WazeAPIClient(mock_api_key)

        async with client:
            assert client.session is not None
            session = client.session

        # Session should be closed after context exit
        assert session.closed
