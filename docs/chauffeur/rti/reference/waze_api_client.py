#!/usr/bin/env python3
"""
Waze API Client for RTI System

Handles communication with Waze traffic data via RapidAPI, including:
- Rate limiting (respecting ~100 req/min quota)
- Response caching (LRU cache keyed by location tile + minute)
- Error handling and retry logic
- Security (API key scrubbing from logs)
"""

import copy
import math
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp
from openpilot.common.swaglog import cloudlog

try:
    from .api_key_manager import get_api_key
except ImportError:
    # Fallback for standalone usage
    def get_api_key():
        import os
        return os.getenv('RAPIDAPI_KEY')



@dataclass
class WazeAlert:
    """Structured Waze alert data."""
    id: str
    type: str
    latitude: float
    longitude: float
    confidence: float
    speed_limit: float | None = None
    street: str | None = None
    country: str | None = None
    raw_data: dict | None = None


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, max_requests: int = 90, time_window: int = 60):
        """Initialize with max requests per time window (90/min for safety margin)."""
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def can_request(self) -> bool:
        """Check if a request can be made without exceeding rate limit."""
        now = time.time()
        # Remove requests older than time window
        self.requests = [req_time for req_time in self.requests
                        if now - req_time < self.time_window]

        return len(self.requests) < self.max_requests

    def record_request(self):
        """Record that a request was made."""
        self.requests.append(time.time())


class LRUCache:
    """Memory-bounded LRU cache with size tracking to prevent unbounded growth."""

    def __init__(self, max_size: int = 100, max_memory_mb: float = 10.0):
        """
        Initialize cache with both entry count and memory limits.
        
        Args:
            max_size: Maximum number of entries (default 100)
            max_memory_mb: Maximum memory usage in MB (default 10MB)
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.current_memory_bytes = 0

        # OrderedDict provides O(1) move_to_end and popitem operations
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.size_map = {}  # Track size of each cached item

        # Statistics for monitoring
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of an object in bytes."""
        import sys

        if obj is None:
            return 0

        # For basic types, use sys.getsizeof
        if isinstance(obj, (str, int, float, bool, bytes)):
            return sys.getsizeof(obj)

        # For collections, recursively estimate
        if isinstance(obj, dict):
            size = sys.getsizeof(obj)
            for k, v in obj.items():
                size += self._estimate_size(k) + self._estimate_size(v)
            return size

        if isinstance(obj, (list, tuple)):
            size = sys.getsizeof(obj)
            for item in obj:
                size += self._estimate_size(item)
            return size

        # For other objects, use a conservative estimate
        try:
            # Try to get actual size
            return sys.getsizeof(obj)
        except:
            # Fallback: assume 1KB for unknown objects
            return 1024

    def get(self, key: str) -> Any | None:
        """Get cached value, updating access order in O(1)."""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used) - O(1) operation
            self.cache.move_to_end(key)
            # Return a deep copy to prevent mutation of cached data
            return copy.deepcopy(self.cache[key])

        self.misses += 1
        return None

    def put(self, key: str, value: Any):
        """Store value in cache, evicting items if necessary to stay within limits."""
        # Estimate size of new value
        new_size = self._estimate_size(value)

        # If single item exceeds memory limit, don't cache it
        if new_size > self.max_memory_bytes:
            cloudlog.warning(f"RTI cache item too large ({new_size/1024:.1f}KB), skipping")
            return

        # If key exists, update it
        if key in self.cache:
            old_size = self.size_map.get(key, 0)
            self.current_memory_bytes -= old_size
            self.cache[key] = value
            self.cache.move_to_end(key)
            self.size_map[key] = new_size
            self.current_memory_bytes += new_size
        else:
            # Evict items until we have space (both count and memory)
            while (len(self.cache) >= self.max_size or
                   self.current_memory_bytes + new_size > self.max_memory_bytes):

                if not self.cache:
                    break

                # Evict LRU (first item) - O(1) operation
                evicted_key, _ = self.cache.popitem(last=False)
                evicted_size = self.size_map.pop(evicted_key, 0)
                self.current_memory_bytes -= evicted_size
                self.evictions += 1

            # Add new item (becomes most recent) - O(1) operation
            self.cache[key] = value
            self.size_map[key] = new_size
            self.current_memory_bytes += new_size

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.size_map.clear()
        self.current_memory_bytes = 0

    def get_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

        return {
            'entries': len(self.cache),
            'memory_mb': self.current_memory_bytes / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }


class WazeAPIClient:
    """Waze API client with rate limiting, caching, and error handling."""

    # Real Waze API endpoints via RapidAPI
    BASE_URL = "https://waze.p.rapidapi.com"
    ALERTS_ENDPOINT = "/alerts-and-jams"  # Try alternative endpoint if this fails

    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
        self.rate_limiter = RateLimiter()
        self.cache = LRUCache(max_size=50, max_memory_mb=5.0)  # Conservative limits for mobile
        self.session: aiohttp.ClientSession | None = None

        # Connection health tracking
        self.consecutive_failures = 0
        self.last_success_time = 0

        # Cache TTL in seconds (default 30 seconds for production)
        self.CACHE_TTL_SECONDS = 30

    @classmethod
    def from_persistent_key(cls) -> 'WazeAPIClient':
        """
        Create client using API key from persistent storage or environment.
        
        Raises:
            ValueError: If no API key is found
            
        Returns:
            WazeAPIClient instance
        """
        api_key = get_api_key()
        if not api_key:
            raise ValueError(
                "No API key found. Please set RAPIDAPI_KEY environment variable "
                "or save key to /data/persist/rapidapi_key"
            )
        return cls(api_key)

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={
                'X-RapidAPI-Key': self.api_key,
                'X-RapidAPI-Host': 'waze.p.rapidapi.com',
                'User-Agent': 'openpilot-rti/1.0'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None

    async def close(self):
        """Explicitly close HTTP session if it exists."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def _get_cache_key(self, lat: float, lon: float) -> str:
        """Generate cache key for location and configurable time window."""
        # Round to ~1km tiles for caching efficiency
        tile_lat = round(lat, 2)
        tile_lon = round(lon, 2)
        # Configurable cache windows based on TTL setting
        cache_window = int(time.time() // self.CACHE_TTL_SECONDS)

        return f"{tile_lat}:{tile_lon}:{cache_window}"

    def _scrub_logs(self, data: str) -> str:
        """Remove API key from log data for security."""
        if self.api_key and self.api_key in data:
            return data.replace(self.api_key, '[API_KEY_REDACTED]')
        return data

    async def _make_request(self, endpoint: str, params: dict[str, Any], center_lat: float, center_lon: float) -> dict | None:
        """Make HTTP request with error handling and rate limiting."""
        # Check rate limit
        if not self.rate_limiter.can_request():
            cloudlog.warning("RTI API rate limited - skipping request")
            return None

        # Check cache first
        cache_key = self._get_cache_key(center_lat, center_lon)
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data

        if not self.session:
            await self.__aenter__()

        try:
            url = f"{self.BASE_URL}{endpoint}"
            query_string = urlencode(params)

            cloudlog.debug(f"RTI API request: {endpoint} with params: {self._scrub_logs(str(params))}")

            async with self.session.get(f"{url}?{query_string}") as response:
                self.rate_limiter.record_request()

                if response.status == 200:
                    data = await response.json()

                    # Cache successful response
                    self.cache.put(cache_key, data)

                    # Reset failure tracking
                    self.consecutive_failures = 0
                    self.last_success_time = time.time()

                    return data

                elif response.status == 429:
                    # Rate limited by API
                    cloudlog.warning("RTI API returned 429 - rate limited by server")
                    return None

                else:
                    error_text = await response.text()
                    cloudlog.error(f"RTI API error {response.status}: {self._scrub_logs(error_text)}")
                    self.consecutive_failures += 1
                    return None

        except TimeoutError:
            cloudlog.error("RTI API request timed out")
            self.consecutive_failures += 1
            return None

        except Exception as e:
            cloudlog.error(f"RTI API request failed: {e}")
            self.consecutive_failures += 1
            return None

    async def get_traffic_alerts(self, latitude: float, longitude: float,
                               radius_km: float = 5.0) -> list[WazeAlert]:
        """
        Fetch traffic alerts from Waze API.
        
        Args:
            latitude: Current GPS latitude
            longitude: Current GPS longitude
            radius_km: Search radius in kilometers
            
        Returns:
            List of WazeAlert objects
        """
        # Skip requests if we've had too many consecutive failures
        if self.consecutive_failures > 5:
            time_since_success = time.time() - self.last_success_time
            if time_since_success < 300:  # Wait 5 minutes after repeated failures
                return []

        # Calculate bounding box from center point and radius
        # Approximate: 1 degree ≈ 111 km
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * abs(math.cos(math.radians(latitude))))

        bottom_left = f"{latitude - lat_offset},{longitude - lon_offset}"
        top_right = f"{latitude + lat_offset},{longitude + lon_offset}"

        params = {
            'bottom_left': bottom_left,
            'top_right': top_right,
            'max_alerts': 20,
            'max_jams': 20,
        }

        try:
            # Fetch alerts and jams from single endpoint
            response_data = await self._make_request(self.ALERTS_ENDPOINT, params, latitude, longitude)

            alerts = []
            if response_data and response_data.get('status') == 'OK':
                data = response_data.get('data', {})

                # Parse alerts
                if 'alerts' in data:
                    alerts.extend(self._parse_alerts(data))

                # Parse jams
                if 'jams' in data:
                    alerts.extend(self._parse_jams(data))

            cloudlog.debug(f"RTI fetched {len(alerts)} alerts near {latitude:.4f},{longitude:.4f}")
            return alerts

        except Exception as e:
            cloudlog.error(f"RTI traffic alert fetch failed: {e}")
            return []

    def _parse_alerts(self, data: dict) -> list[WazeAlert]:
        """Parse alerts from Waze API response using real API structure."""
        alerts = []

        raw_alerts = data.get('alerts', [])

        for alert in raw_alerts:
            try:
                # Real API uses alert_id, not uuid or id
                alert_id = str(alert.get('alert_id', ''))
                if not alert_id:
                    continue

                # Map Waze alert types (includes subtype)
                alert_type = self._map_alert_type(alert.get('type', ''), alert.get('subtype', ''))

                # Real API has direct latitude/longitude fields
                latitude = float(alert.get('latitude', 0))
                longitude = float(alert.get('longitude', 0))

                # Real API uses alert_confidence, not just confidence
                confidence = float(alert.get('alert_confidence', 0.5))

                waze_alert = WazeAlert(
                    id=alert_id,
                    type=alert_type,
                    latitude=latitude,
                    longitude=longitude,
                    confidence=confidence,
                    speed_limit=None,  # Real API doesn't provide speed_limit in alerts
                    street=alert.get('street'),
                    country=alert.get('country'),
                    raw_data=alert
                )

                # Validate coordinates are reasonable
                if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                    alerts.append(waze_alert)

            except (KeyError, ValueError, TypeError) as e:
                cloudlog.warning(f"RTI failed to parse alert: {e}")
                continue

        return alerts

    def _parse_jams(self, data: dict) -> list[WazeAlert]:
        """Parse traffic jams from Waze API response."""
        alerts = []

        raw_jams = data.get('jams', [])

        for jam in raw_jams:
            try:
                # Convert jam to alert format
                waze_alert = WazeAlert(
                    id=str(jam.get('uuid', jam.get('id', ''))),
                    type='jam',
                    latitude=float(jam.get('line', [{}])[0].get('y', 0)),
                    longitude=float(jam.get('line', [{}])[0].get('x', 0)),
                    confidence=float(jam.get('level', 1) / 5.0),  # Normalize to 0-1
                    speed_limit=jam.get('speedKMH'),
                    street=jam.get('street'),
                    raw_data=jam
                )

                if waze_alert.latitude != 0 and waze_alert.longitude != 0:
                    alerts.append(waze_alert)

            except (KeyError, ValueError, TypeError) as e:
                cloudlog.warning(f"RTI failed to parse jam: {e}")
                continue

        return alerts

    def _map_alert_type(self, waze_type: str, subtype: str = '') -> str:
        """Map Waze alert types and subtypes to RTI standard types."""
        waze_type = waze_type.upper() if waze_type else ''
        subtype = subtype.upper() if subtype else ''

        # Handle specific subtypes first
        if waze_type == 'POLICE':
            if subtype == 'POLICE_HIDING':
                return 'policeHiding'
            else:
                return 'police'
        elif waze_type == 'HAZARD':
            if subtype == 'HAZARD_ON_ROAD':
                return 'roadHazard'
            elif subtype == 'HAZARD_ON_SHOULDER':
                return 'shoulderHazard'
            else:
                return 'hazard'
        elif waze_type == 'ROAD_CLOSED':
            return 'roadClosed'

        # Standard type mapping for other cases
        type_mapping = {
            'ACCIDENT': 'accident',
            'CONSTRUCTION': 'construction',
            'SPEED_TRAP': 'speedTrap',
            'SPEED_CAMERA': 'speedCamera',
        }

        return type_mapping.get(waze_type, 'hazard')

    def get_health_status(self) -> str:
        """Get current API connection health status."""
        if self.consecutive_failures > 10:
            return 'error'
        elif self.consecutive_failures > 5:
            return 'degraded'
        elif not self.rate_limiter.can_request():
            return 'rateLimited'
        elif time.time() - self.last_success_time < 60:
            return 'connected'
        else:
            return 'disconnected'
