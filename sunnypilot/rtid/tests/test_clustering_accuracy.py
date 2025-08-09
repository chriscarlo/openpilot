#!/usr/bin/env python3
"""
Accuracy validation tests for threat clustering optimization.

Compares the optimized spatial grid algorithm against the original O(n²) algorithm
to ensure clustering accuracy is maintained while achieving performance improvements.
"""

import random

import pytest

from sunnypilot.rtid.waze_api_client import WazeAlert
from sunnypilot.rtid.threat_detector import ThreatClusterer, GeoUtils


class OriginalThreatClusterer:
    """Original O(n²) clustering algorithm for accuracy comparison."""

    @staticmethod
    def deduplicate_threats_original(threats: list[WazeAlert],
                                   cluster_radius_m: float = 100) -> list[WazeAlert]:
        """
        Original O(n²) deduplication algorithm for accuracy comparison.
        
        This is the naive implementation that compares every threat with every other threat.
        """
        if not threats:
            return []

        if len(threats) == 1:
            return threats

        clusters = []
        used = set()

        # O(n²) nested loop - original algorithm
        for i, threat in enumerate(threats):
            if i in used:
                continue

            cluster = [threat]
            used.add(i)

            # Compare with all other threats
            for j, other_threat in enumerate(threats):
                if j in used or j == i:
                    continue

                distance = GeoUtils.haversine_distance(
                    threat.latitude, threat.longitude,
                    other_threat.latitude, other_threat.longitude
                )

                if distance <= cluster_radius_m:
                    cluster.append(other_threat)
                    used.add(j)

            clusters.append(cluster)

        # Return best threat from each cluster
        deduplicated = []
        for cluster in clusters:
            best_threat = max(cluster, key=lambda t: t.confidence)
            deduplicated.append(best_threat)

        return deduplicated


def create_deterministic_threats(count: int, seed: int = 42) -> list[WazeAlert]:
    """Create deterministic threat set for consistent testing."""
    random.seed(seed)
    threats = []

    base_lat, base_lon = 37.4221, -122.0841

    for i in range(count):
        # Create threats with known clustering patterns
        if i < 10:
            # Cluster 1: Close together (should merge)
            lat = base_lat + random.uniform(-0.001, 0.001)  # ~100m spread
            lon = base_lon + random.uniform(-0.001, 0.001)
        elif i < 20:
            # Cluster 2: Different area, also close
            lat = base_lat + 0.01 + random.uniform(-0.001, 0.001)
            lon = base_lon + 0.01 + random.uniform(-0.001, 0.001)
        else:
            # Isolated threats (should remain separate)
            lat = base_lat + random.uniform(0.02, 0.05)  # Far apart
            lon = base_lon + random.uniform(0.02, 0.05)

        threat = WazeAlert(
            id=f'deterministic-{i}',
            type=random.choice(['police', 'speedTrap', 'accident']),
            latitude=lat,
            longitude=lon,
            confidence=random.uniform(0.5, 1.0),
            speed_limit=random.choice([25, 35, 45]),
            street=f'Test St {i}',
            country='US',
            raw_data={}
        )
        threats.append(threat)

    return threats


@pytest.mark.accuracy
class TestThreatClusteringAccuracy:
    """Accuracy validation tests comparing optimized vs original algorithm."""

    def test_clustering_produces_identical_results(self):
        """Test that optimized algorithm produces identical results to original."""
        test_cases = [
            ("small_set", create_deterministic_threats(15)),
            ("medium_set", create_deterministic_threats(30)),
            ("large_set", create_deterministic_threats(50)),
        ]

        for test_name, threats in test_cases:
            # Run both algorithms
            original_result = OriginalThreatClusterer.deduplicate_threats_original(threats)
            optimized_result = ThreatClusterer.deduplicate_threats(threats)

            print(f"{test_name}: {len(threats)} threats -> original:{len(original_result)}, optimized:{len(optimized_result)}")

            # Results should have same count
            assert len(original_result) == len(optimized_result), \
                f"{test_name}: cluster count mismatch - original:{len(original_result)}, optimized:{len(optimized_result)}"

            # Both results should contain the same threats (order may differ)
            original_ids = {threat.id for threat in original_result}
            optimized_ids = {threat.id for threat in optimized_result}

            assert original_ids == optimized_ids, \
                f"{test_name}: different threats selected - missing from optimized: {original_ids - optimized_ids}, extra in optimized: {optimized_ids - original_ids}"

    def test_clustering_with_various_radii(self):
        """Test clustering accuracy with different cluster radii."""
        threats = create_deterministic_threats(25)
        radii = [50, 100, 200, 500]  # Different cluster distances

        for radius in radii:
            original_result = OriginalThreatClusterer.deduplicate_threats_original(threats, radius)
            optimized_result = ThreatClusterer.deduplicate_threats(threats, radius)

            print(f"Radius {radius}m: {len(threats)} threats -> {len(original_result)} clusters")

            assert len(original_result) == len(optimized_result), \
                f"Radius {radius}m: cluster count mismatch"

            # Verify same threats selected
            original_ids = {threat.id for threat in original_result}
            optimized_ids = {threat.id for threat in optimized_result}
            assert original_ids == optimized_ids, \
                f"Radius {radius}m: different threat selection"

    def test_clustering_edge_cases_accuracy(self):
        """Test edge cases for clustering accuracy."""

        # Empty list
        assert OriginalThreatClusterer.deduplicate_threats_original([]) == []
        assert ThreatClusterer.deduplicate_threats([]) == []

        # Single threat
        single = create_deterministic_threats(1)
        original_single = OriginalThreatClusterer.deduplicate_threats_original(single)
        optimized_single = ThreatClusterer.deduplicate_threats(single)
        assert len(original_single) == len(optimized_single) == 1
        assert original_single[0].id == optimized_single[0].id

        # All threats at exact same location
        same_loc_threats = []
        for i in range(5):
            threat = WazeAlert(
                id=f'same-{i}',
                type='police',
                latitude=37.4221,
                longitude=-122.0841,
                confidence=0.5 + i * 0.1,  # Different confidence for selection
                speed_limit=35,
                raw_data={}
            )
            same_loc_threats.append(threat)

        original_same = OriginalThreatClusterer.deduplicate_threats_original(same_loc_threats)
        optimized_same = ThreatClusterer.deduplicate_threats(same_loc_threats)

        assert len(original_same) == len(optimized_same) == 1, "Should cluster all same-location threats"
        assert original_same[0].id == optimized_same[0].id, "Should select same best threat"

        # Verify highest confidence was selected
        assert original_same[0].confidence == 0.9, "Should select highest confidence threat"

    def test_clustering_confidence_selection_accuracy(self):
        """Test that both algorithms select the same highest-confidence threat from clusters."""

        # Create cluster with known confidence ordering
        cluster_threats = []
        confidences = [0.6, 0.9, 0.7, 0.8, 0.5]  # 0.9 should be selected

        for i, conf in enumerate(confidences):
            threat = WazeAlert(
                id=f'conf-test-{i}',
                type='police',
                latitude=37.4221 + i * 0.0001,  # Very close together
                longitude=-122.0841,
                confidence=conf,
                speed_limit=35,
                raw_data={}
            )
            cluster_threats.append(threat)

        original_result = OriginalThreatClusterer.deduplicate_threats_original(cluster_threats)
        optimized_result = ThreatClusterer.deduplicate_threats(cluster_threats)

        assert len(original_result) == len(optimized_result) == 1, "Should form single cluster"
        assert original_result[0].confidence == optimized_result[0].confidence == 0.9, \
            "Both should select highest confidence threat"
        assert original_result[0].id == optimized_result[0].id, "Should select same threat"

    def test_clustering_boundary_cases(self):
        """Test clustering accuracy at cluster radius boundaries."""

        # Create threats at exact boundary distances
        base_threat = WazeAlert(
            id='base',
            type='police',
            latitude=37.4221,
            longitude=-122.0841,
            confidence=0.8,
            speed_limit=35,
            raw_data={}
        )

        # Threat exactly at cluster boundary (should be included)
        boundary_threat = WazeAlert(
            id='boundary',
            type='police',
            latitude=37.4221 + (100 / 110540),  # Exactly 100m north
            longitude=-122.0841,
            confidence=0.7,
            speed_limit=35,
            raw_data={}
        )

        # Threat just outside boundary (should be separate)
        outside_threat = WazeAlert(
            id='outside',
            type='police',
            latitude=37.4221 + (101 / 110540),  # Exactly 101m north
            longitude=-122.0841,
            confidence=0.9,
            speed_limit=35,
            raw_data={}
        )

        boundary_threats = [base_threat, boundary_threat, outside_threat]

        original_result = OriginalThreatClusterer.deduplicate_threats_original(boundary_threats, 100)
        optimized_result = ThreatClusterer.deduplicate_threats(boundary_threats, 100)

        print(f"Boundary test: {len(boundary_threats)} threats -> original:{len(original_result)}, optimized:{len(optimized_result)}")

        # Both should produce same number of clusters
        assert len(original_result) == len(optimized_result), "Boundary case cluster count mismatch"

        # Same threats should be selected
        original_ids = {t.id for t in original_result}
        optimized_ids = {t.id for t in optimized_result}
        assert original_ids == optimized_ids, f"Boundary case selection mismatch: {original_ids} vs {optimized_ids}"

    def test_clustering_random_scenarios(self):
        """Test clustering accuracy with multiple random scenarios."""

        for scenario in range(5):  # Test multiple random scenarios
            threats = create_deterministic_threats(40, seed=scenario + 100)

            original_result = OriginalThreatClusterer.deduplicate_threats_original(threats)
            optimized_result = ThreatClusterer.deduplicate_threats(threats)

            print(f"Random scenario {scenario}: {len(threats)} threats -> {len(original_result)} clusters")

            assert len(original_result) == len(optimized_result), \
                f"Scenario {scenario}: cluster count mismatch"

            original_ids = {threat.id for threat in original_result}
            optimized_ids = {threat.id for threat in optimized_result}
            assert original_ids == optimized_ids, \
                f"Scenario {scenario}: threat selection mismatch"


if __name__ == '__main__':
    # Run accuracy validation tests directly
    test = TestThreatClusteringAccuracy()

    print("=== Threat Clustering Accuracy Validation ===")
    test.test_clustering_produces_identical_results()
    print("PASS: Identical results test passed")

    test.test_clustering_with_various_radii()
    print("PASS: Various radii test passed")

    test.test_clustering_edge_cases_accuracy()
    print("PASS: Edge cases test passed")

    test.test_clustering_confidence_selection_accuracy()
    print("PASS: Confidence selection test passed")

    test.test_clustering_boundary_cases()
    print("PASS: Boundary cases test passed")

    test.test_clustering_random_scenarios()
    print("PASS: Random scenarios test passed")

    print("\nSUCCESS: All accuracy validation tests passed!")
    print("         Optimized algorithm produces identical results to original O(n²) version")
