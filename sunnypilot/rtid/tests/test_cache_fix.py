#!/usr/bin/env python3
"""
Test for cache system fix

Specifically tests that the cache hit/miss rate is working correctly
after fixing the parameter mismatch issue.
"""

import pytest
from unittest.mock import AsyncMock, patch
import time

from sunnypilot.rtid.waze_api_client import WazeAPIClient


@pytest.mark.unit
class TestCacheFix:
    """Test that the cache system is working correctly after the fix."""

    @pytest.fixture
    def api_client(self, mock_api_key):
        return WazeAPIClient(mock_api_key)

    @pytest.mark.asyncio
    async def test_cache_hit_rate_with_same_location(self, api_client, mock_waze_api_response):
        """Test that repeated requests to same location hit cache."""

        # Mock the HTTP request to return test data
        with patch.object(api_client, 'session') as mock_session:
            # Configure mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_waze_api_response)
            mock_session.get.return_value.__aenter__.return_value = mock_response

            # Make first request - should hit API
            alerts1 = await api_client.get_traffic_alerts(37.4221, -122.0841)

            # Make second request to same location - should hit cache
            alerts2 = await api_client.get_traffic_alerts(37.4221, -122.0841)

            # Third request with tiny difference - should also hit cache due to rounding
            alerts3 = await api_client.get_traffic_alerts(37.4222, -122.0842)

            # Should have made only ONE HTTP request (first one)
            assert mock_session.get.call_count == 1, f"Expected 1 API call, got {mock_session.get.call_count}"

            # All requests should return same data
            assert len(alerts1) > 0
            assert len(alerts2) == len(alerts1)
            assert len(alerts3) == len(alerts1)

    @pytest.mark.asyncio
    async def test_cache_miss_with_different_locations(self, api_client, mock_waze_api_response):
        """Test that requests to different locations miss cache appropriately."""

        # Mock the HTTP request
        with patch.object(api_client, 'session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_waze_api_response)
            mock_session.get.return_value.__aenter__.return_value = mock_response

            # Make requests to different locations
            await api_client.get_traffic_alerts(37.4221, -122.0841)  # Bay Area
            await api_client.get_traffic_alerts(40.7128, -74.0060)   # NYC
            await api_client.get_traffic_alerts(34.0522, -118.2437)  # LA

            # Should have made three API calls (no cache hits)
            assert mock_session.get.call_count == 3, f"Expected 3 API calls, got {mock_session.get.call_count}"

    @pytest.mark.asyncio
    async def test_cache_key_generation_consistency(self, api_client):
        """Test that cache key generation is consistent and uses center coordinates."""

        # Test that nearby coordinates generate the same cache key
        key1 = api_client._get_cache_key(37.4221, -122.0841)
        key2 = api_client._get_cache_key(37.4222, -122.0842)  # Slight difference

        # Should be same due to rounding
        assert key1 == key2, "Nearby coordinates should generate same cache key"

        # Test that distant coordinates generate different keys
        key3 = api_client._get_cache_key(40.7128, -74.0060)  # NYC
        assert key1 != key3, "Distant coordinates should generate different cache keys"

    @pytest.mark.asyncio
    async def test_cache_expiration(self, api_client, mock_waze_api_response):
        """Test that cache entries expire after TTL."""

        # Set cache TTL to 1 second for testing
        api_client.CACHE_TTL_SECONDS = 1

        with patch.object(api_client, 'session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_waze_api_response)
            mock_session.get.return_value.__aenter__.return_value = mock_response

            # Make first request
            await api_client.get_traffic_alerts(37.4221, -122.0841)
            assert mock_session.get.call_count == 1

            # Wait for cache to expire
            time.sleep(1.1)

            # Make second request - should hit API again due to expiration
            await api_client.get_traffic_alerts(37.4221, -122.0841)
            assert mock_session.get.call_count == 2, "Cache should have expired, causing second API call"

    def test_cache_statistics_tracking(self, api_client):
        """Test that we can track cache hit/miss statistics."""

        # Add some dummy data to cache
        cache_key = api_client._get_cache_key(37.4221, -122.0841)
        api_client.cache.put(cache_key, {"test": "data"})

        # Test cache hit
        hit_data = api_client.cache.get(cache_key)
        assert hit_data is not None, "Should be cache hit"

        # Test cache miss
        miss_key = api_client._get_cache_key(40.7128, -74.0060)
        miss_data = api_client.cache.get(miss_key)
        assert miss_data is None, "Should be cache miss"
