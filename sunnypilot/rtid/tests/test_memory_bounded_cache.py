#!/usr/bin/env python3
"""
Test memory-bounded LRU cache implementation.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock cloudlog before importing
sys.modules['openpilot.common.swaglog'] = MagicMock()

# Now import after mocking
from sunnypilot.rtid.waze_api_client import LRUCache


class TestMemoryBoundedCache:
    """Test suite for memory-bounded LRU cache."""

    def test_basic_operations(self):
        """Test basic get/put operations."""
        cache = LRUCache(max_size=3, max_memory_mb=1.0)
        
        # Test put and get
        cache.put("key1", {"data": "value1"})
        assert cache.get("key1") == {"data": "value1"}
        
        # Test miss
        assert cache.get("nonexistent") is None
        
        # Test statistics
        stats = cache.get_stats()
        assert stats['entries'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1

    def test_max_entries_eviction(self):
        """Test eviction based on entry count limit."""
        cache = LRUCache(max_size=3, max_memory_mb=100.0)  # Large memory limit
        
        # Add items up to limit
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        
        # Add one more - should evict key1 (LRU)
        cache.put("key4", "value4")
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None
        
        stats = cache.get_stats()
        assert stats['entries'] == 3
        assert stats['evictions'] == 1

    def test_memory_limit_eviction(self):
        """Test eviction based on memory limit."""
        cache = LRUCache(max_size=100, max_memory_mb=0.001)  # 1KB limit
        
        # Create a large object (>1KB)
        large_data = "x" * 1000  # ~1KB string
        
        cache.put("key1", "small")
        cache.put("key2", large_data)
        
        # key1 should be evicted due to memory pressure
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        
        stats = cache.get_stats()
        assert stats['evictions'] >= 1
        assert stats['memory_mb'] <= 0.001

    def test_oversized_item_rejection(self):
        """Test that items larger than memory limit are rejected."""
        cache = LRUCache(max_size=10, max_memory_mb=0.001)  # 1KB limit
        
        # Try to cache something much larger than limit
        huge_data = "x" * 10000  # ~10KB
        
        with patch('sunnypilot.rtid.waze_api_client.cloudlog.warning') as mock_warning:
            cache.put("huge", huge_data)
            mock_warning.assert_called_once()
        
        # Item should not be cached
        assert cache.get("huge") is None
        assert cache.get_stats()['entries'] == 0

    def test_lru_ordering(self):
        """Test that LRU ordering is maintained correctly."""
        cache = LRUCache(max_size=3, max_memory_mb=10.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it most recently used
        _ = cache.get("key1")
        
        # Add key4 - should evict key2 (now the LRU)
        cache.put("key4", "value4")
        
        assert cache.get("key1") is not None  # Still there (was accessed)
        assert cache.get("key2") is None      # Evicted (LRU)
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_memory_estimation(self):
        """Test memory size estimation for different object types."""
        cache = LRUCache()
        
        # Test basic types
        assert cache._estimate_size(None) == 0
        assert cache._estimate_size("test") > 0
        assert cache._estimate_size(123) > 0
        assert cache._estimate_size(True) > 0
        
        # Test collections
        dict_size = cache._estimate_size({"key": "value"})
        assert dict_size > cache._estimate_size("key") + cache._estimate_size("value")
        
        list_size = cache._estimate_size([1, 2, 3])
        assert list_size > cache._estimate_size(1) * 3

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LRUCache(max_size=5, max_memory_mb=10.0)
        
        # Add some items
        for i in range(5):
            cache.put(f"key{i}", f"value{i}")
        
        assert cache.get_stats()['entries'] == 5
        assert cache.current_memory_bytes > 0
        
        # Clear cache
        cache.clear()
        
        assert cache.get_stats()['entries'] == 0
        assert cache.current_memory_bytes == 0
        assert cache.get("key0") is None

    def test_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        cache = LRUCache()
        
        cache.put("key1", "value1")
        
        # Generate some hits and misses
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key3")  # Miss
        cache.get("key1")  # Hit
        
        stats = cache.get_stats()
        assert stats['hits'] == 3
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 0.6  # 3/5

    def test_complex_data_structures(self):
        """Test caching of complex nested data structures."""
        cache = LRUCache(max_size=10, max_memory_mb=1.0)
        
        # Complex Waze-like response
        complex_data = {
            'alerts': [
                {
                    'id': 'alert_1',
                    'type': 'POLICE',
                    'location': {'lat': 37.4221, 'lon': -122.0841},
                    'confidence': 0.85,
                    'metadata': {
                        'reporter': 'user123',
                        'timestamp': 1234567890,
                        'votes': [1, 1, 0, 1]
                    }
                }
            ],
            'jams': [
                {'id': 'jam_1', 'severity': 3, 'length': 500}
            ]
        }
        
        cache.put("complex", complex_data)
        retrieved = cache.get("complex")
        
        # Verify deep copy (mutation doesn't affect cache)
        retrieved['alerts'][0]['type'] = 'MODIFIED'
        original = cache.get("complex")
        assert original['alerts'][0]['type'] == 'POLICE'

    def test_memory_pressure_scenario(self):
        """Test realistic memory pressure scenario."""
        cache = LRUCache(max_size=50, max_memory_mb=5.0)
        
        # Simulate caching Waze responses of varying sizes
        small_response = {'alerts': []}
        medium_response = {'alerts': [{'id': f'a{i}'} for i in range(10)]}
        large_response = {'alerts': [{'id': f'a{i}', 'data': 'x' * 100} for i in range(100)]}
        
        # Fill cache with mixed sizes
        for i in range(20):
            if i % 3 == 0:
                cache.put(f"large_{i}", large_response)
            elif i % 3 == 1:
                cache.put(f"medium_{i}", medium_response)
            else:
                cache.put(f"small_{i}", small_response)
        
        stats = cache.get_stats()
        
        # Verify memory stays within limit
        assert stats['memory_mb'] <= 5.0
        assert stats['entries'] <= 50
        
        # Should have evicted some items
        assert stats['evictions'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])