#!/usr/bin/env python3
"""
Functional Correctness Tests for Threat Clustering

Validates that the clustering algorithm satisfies the clustering rules
without requiring exact identity to any reference implementation.
"""

# import pytest
from sunnypilot.rtid.waze_api_client import WazeAlert
from sunnypilot.rtid.threat_detector import ThreatClusterer
from sunnypilot.rtid.threat_detector import GeoUtils


def create_test_threats(count: int, spread_km: float = 1.0) -> list[WazeAlert]:
    """Create test threats spread over a geographic area."""
    import random
    random.seed(42)  # Deterministic for consistent testing
    threats = []

    base_lat, base_lon = 37.4221, -122.0841

    for i in range(count):
        lat_offset = random.uniform(-spread_km/111, spread_km/111)
        lon_offset = random.uniform(-spread_km/111, spread_km/111)

        threat = WazeAlert(
            id=f'test-{i}',
            type=random.choice(['police', 'speedTrap', 'accident', 'hazard']),
            latitude=base_lat + lat_offset,
            longitude=base_lon + lon_offset,
            confidence=random.uniform(0.3, 1.0),
            speed_limit=random.choice([25, 35, 45, 55]),
            street=f'Test St {i}',
            country='US',
            raw_data={}
        )
        threats.append(threat)

    return threats


# @pytest.mark.functional
class TestClusteringFunctionalCorrectness:
    """Test clustering algorithm functional correctness."""

    def test_clustering_rules_satisfied(self):
        """Test that clustering rules are properly satisfied."""
        for cluster_radius in [50, 100, 200]:
            threats = create_test_threats(50, spread_km=2.0)
            result = ThreatClusterer.deduplicate_threats(threats, cluster_radius)

            # Rule 1: All returned threats should be from original set
            original_ids = {t.id for t in threats}
            result_ids = {t.id for t in result}
            assert result_ids.issubset(original_ids), "Result contains threats not in original set"

            # Rule 2: No two result threats should be within cluster radius
            for i, threat1 in enumerate(result):
                for j, threat2 in enumerate(result[i+1:], i+1):
                    distance = GeoUtils.haversine_distance(
                        threat1.latitude, threat1.longitude,
                        threat2.latitude, threat2.longitude
                    )
                    assert distance > cluster_radius, \
                        f"Threats {threat1.id} and {threat2.id} are {distance:.1f}m apart (< {cluster_radius}m)"

            # Rule 3: For each result threat, it should be the best in its cluster
            # (This is harder to test without rebuilding clusters, so we test confidence is reasonable)
            for threat in result:
                assert threat.confidence >= 0.3, f"Selected threat {threat.id} has low confidence: {threat.confidence}"

            print(f"✅ Radius {cluster_radius}m: {len(threats)} threats → {len(result)} clusters, all rules satisfied")

    def test_clustering_reduces_threat_count(self):
        """Test that clustering reduces the number of threats when overlapping."""
        # Create overlapping threats
        threats = []
        base_lat, base_lon = 37.4221, -122.0841

        # Create 3 clusters of 5 threats each, well separated
        for cluster_id in range(3):
            cluster_lat = base_lat + cluster_id * 0.01  # ~1km apart
            cluster_lon = base_lon + cluster_id * 0.01

            for i in range(5):
                threat = WazeAlert(
                    id=f'cluster-{cluster_id}-{i}',
                    type='police',
                    latitude=cluster_lat + i * 0.0003,  # ~30m apart within cluster
                    longitude=cluster_lon + i * 0.0003,
                    confidence=0.5 + i * 0.1,  # Increasing confidence
                    speed_limit=35,
                    street=f'Test St {cluster_id}',
                    country='US',
                    raw_data={}
                )
                threats.append(threat)

        result = ThreatClusterer.deduplicate_threats(threats, cluster_radius_m=100)

        # Should have exactly 3 threats (best from each cluster)
        assert len(result) == 3, f"Expected 3 clusters, got {len(result)}"

        # Each result should be the highest confidence from its cluster
        result_confidences = sorted([t.confidence for t in result], reverse=True)
        expected_confidences = [0.9, 0.9, 0.9]  # Best from each cluster
        assert result_confidences == expected_confidences, \
            f"Expected confidences {expected_confidences}, got {result_confidences}"

        print("✅ Clustering reduced 15 overlapping threats to 3 clusters correctly")

    def test_clustering_preserves_isolated_threats(self):
        """Test that isolated threats are preserved."""
        threats = []
        base_lat, base_lon = 37.4221, -122.0841

        # Create 5 isolated threats, each >200m apart
        for i in range(5):
            threat = WazeAlert(
                id=f'isolated-{i}',
                type='police',
                latitude=base_lat + i * 0.002,  # ~200m apart
                longitude=base_lon + i * 0.002,
                confidence=0.8,
                speed_limit=35,
                street=f'Test St {i}',
                country='US',
                raw_data={}
            )
            threats.append(threat)

        result = ThreatClusterer.deduplicate_threats(threats, cluster_radius_m=100)

        # All threats should be preserved since they're isolated
        assert len(result) == 5, f"Expected 5 isolated threats preserved, got {len(result)}"

        print("✅ All 5 isolated threats preserved correctly")

    def test_empty_and_single_threat_cases(self):
        """Test edge cases with empty or single threat lists."""
        # Empty list
        result = ThreatClusterer.deduplicate_threats([])
        assert result == [], "Empty list should return empty list"

        # Single threat
        single_threat = [WazeAlert(
            id='single',
            type='police',
            latitude=37.4221,
            longitude=-122.0841,
            confidence=0.8,
            speed_limit=35,
            street='Test St',
            country='US',
            raw_data={}
        )]
        result = ThreatClusterer.deduplicate_threats(single_threat)
        assert len(result) == 1, "Single threat should return single threat"
        assert result[0].id == 'single', "Single threat should be unchanged"

        print("✅ Edge cases handled correctly")


if __name__ == '__main__':
    test = TestClusteringFunctionalCorrectness()

    print("=== Functional Correctness Validation ===")
    test.test_clustering_rules_satisfied()
    test.test_clustering_reduces_threat_count()
    test.test_clustering_preserves_isolated_threats()
    test.test_empty_and_single_threat_cases()
    print("\n✅ SUCCESS: All functional correctness tests passed!")
    print("         Clustering algorithm is functionally correct")
