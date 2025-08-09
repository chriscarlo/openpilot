#!/usr/bin/env python3
"""
Performance benchmark tests for threat clustering optimization.

Validates that the spatial grid optimization meets the 15ms performance budget
and scales linearly with threat count.
"""

import time
import random

import pytest

from sunnypilot.rtid.waze_api_client import WazeAlert
from sunnypilot.rtid.threat_detector import ThreatClusterer


def create_test_threats(count: int, lat_center: float = 37.4221,
                       lon_center: float = -122.0841,
                       spread_km: float = 5.0) -> list[WazeAlert]:
    """Create test threats spread over a geographic area."""
    threats = []

    for i in range(count):
        # Spread threats within specified radius
        lat_offset = random.uniform(-spread_km/111, spread_km/111)  # ~111km per degree
        lon_offset = random.uniform(-spread_km/111, spread_km/111)

        threat = WazeAlert(
            id=f'test-threat-{i}',
            type=random.choice(['police', 'speedTrap', 'accident', 'hazard']),
            latitude=lat_center + lat_offset,
            longitude=lon_center + lon_offset,
            confidence=random.uniform(0.5, 1.0),
            speed_limit=random.choice([25, 35, 45, 55, 65]),
            street=f'Test Street {i}',
            country='US',
            raw_data={}
        )
        threats.append(threat)

    return threats


@pytest.mark.performance
class TestThreatClusteringPerformance:
    """Performance tests for threat clustering algorithm."""

    def test_clustering_performance_scales_linearly(self):
        """Test that clustering performance scales linearly with threat count."""
        threat_counts = [10, 25, 50, 100, 200]
        times = []

        for count in threat_counts:
            threats = create_test_threats(count)

            start_time = time.perf_counter()
            ThreatClusterer.deduplicate_threats(threats)
            end_time = time.perf_counter()

            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)

            print(f"Clustering {count} threats: {elapsed_ms:.3f}ms")

        # Verify linear scaling (should not grow quadratically)
        # Performance should be roughly proportional to threat count
        for i in range(1, len(times)):
            ratio = times[i] / times[0]  # Time ratio
            count_ratio = threat_counts[i] / threat_counts[0]  # Count ratio

            # Time ratio should be no more than 1.5x the count ratio (allowing some overhead)
            assert ratio <= count_ratio * 1.5, f"Performance not linear: {count_ratio}x threats took {ratio}x time"

    def test_clustering_meets_performance_budget(self):
        """Test that clustering meets the 15ms performance budget."""
        # Test with realistic worst-case scenario
        high_density_threats = create_test_threats(100, spread_km=1.0)  # Dense urban scenario

        start_time = time.perf_counter()
        result = ThreatClusterer.deduplicate_threats(high_density_threats)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        print(f"Clustering 100 dense threats: {elapsed_ms:.3f}ms")
        print(f"Result count: {len(result)} clusters")

        # With BFS optimization, should complete within 3ms for 100 threats
        assert elapsed_ms < 3.0, f"Clustering took {elapsed_ms:.3f}ms, exceeds 3ms performance target"

        # Ensure we're well within the 15ms safety budget
        assert elapsed_ms < 15.0, f"Clustering took {elapsed_ms:.3f}ms, exceeds 15ms safety budget"

    def test_clustering_performance_with_varying_density(self):
        """Test performance with different threat density scenarios."""
        scenarios = [
            ("sparse", 50, 10.0),   # 50 threats over 10km
            ("moderate", 50, 5.0),  # 50 threats over 5km
            ("dense", 50, 1.0),     # 50 threats over 1km
            ("very_dense", 50, 0.5) # 50 threats over 0.5km
        ]

        for scenario_name, count, spread_km in scenarios:
            threats = create_test_threats(count, spread_km=spread_km)

            start_time = time.perf_counter()
            result = ThreatClusterer.deduplicate_threats(threats)
            end_time = time.perf_counter()

            elapsed_ms = (end_time - start_time) * 1000

            print(f"{scenario_name}: {elapsed_ms:.3f}ms, {len(result)} clusters from {count} threats")

            # All scenarios should complete quickly
            assert elapsed_ms < 10.0, f"{scenario_name} scenario took {elapsed_ms:.3f}ms"

    def test_clustering_memory_efficiency(self):
        """Test that clustering doesn't create excessive memory overhead."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Baseline memory
        baseline_mb = process.memory_info().rss / 1024 / 1024

        # Process large number of threats
        large_threat_set = create_test_threats(500)
        result = ThreatClusterer.deduplicate_threats(large_threat_set)

        # Check memory usage
        peak_mb = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_mb - baseline_mb

        print(f"Memory usage: baseline={baseline_mb:.1f}MB, peak={peak_mb:.1f}MB, increase={memory_increase:.1f}MB")
        print(f"Processed {len(large_threat_set)} threats -> {len(result)} clusters")

        # Should not use excessive memory (rough check)
        assert memory_increase < 50, f"Memory increase of {memory_increase:.1f}MB seems excessive"

    def test_clustering_with_edge_cases(self):
        """Test performance with edge case inputs."""
        # Empty list
        start = time.perf_counter()
        result = ThreatClusterer.deduplicate_threats([])
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 1.0 and result == []

        # Single threat
        single_threat = create_test_threats(1)
        start = time.perf_counter()
        result = ThreatClusterer.deduplicate_threats(single_threat)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 1.0 and len(result) == 1

        # All threats at same location (should cluster to 1)
        same_location = []
        for i in range(20):
            threat = WazeAlert(
                id=f'same-loc-{i}',
                type='police',
                latitude=37.4221,  # Exact same location
                longitude=-122.0841,
                confidence=0.8,
                speed_limit=35,
                raw_data={}
            )
            same_location.append(threat)

        start = time.perf_counter()
        result = ThreatClusterer.deduplicate_threats(same_location)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Same location clustering: {elapsed:.3f}ms, {len(result)} from {len(same_location)}")
        assert elapsed < 5.0 and len(result) == 1


if __name__ == '__main__':
    # Run performance benchmarks directly
    test = TestThreatClusteringPerformance()

    print("=== Threat Clustering Performance Benchmark ===")
    test.test_clustering_performance_scales_linearly()
    print()
    test.test_clustering_meets_performance_budget()
    print()
    test.test_clustering_performance_with_varying_density()
    print()
    test.test_clustering_with_edge_cases()
    print("\n✅ All performance tests passed!")
