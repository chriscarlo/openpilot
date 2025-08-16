#!/usr/bin/env python3
"""
RTI Direction Calculator Test Script

Test the mathematical logic for calculating directional arrows from ego vehicle
to threat locations before implementing in the actual HUD widget.

This validates:
1. Bearing calculation between two GPS coordinates
2. Relative direction calculation accounting for ego heading
3. Arrow angle conversion for display
4. Edge cases and accuracy verification
"""

import math
import unittest
from dataclasses import dataclass


@dataclass
class Position:
    """GPS coordinate position"""
    lat: float  # degrees
    lon: float  # degrees


@dataclass
class EgoState:
    """Ego vehicle state"""
    position: Position
    heading: float  # degrees, 0=North, 90=East, 180=South, 270=West


@dataclass
class Threat:
    """Threat with location"""
    position: Position
    threat_type: str


class RTIDirectionCalculator:
    """Calculate directional arrows for RTI threats"""

    @staticmethod
    def calculate_bearing(pos1: Position, pos2: Position) -> float:
        """
        Calculate bearing from pos1 to pos2 in degrees
        Returns: 0-360 degrees (0=North, 90=East, 180=South, 270=West)
        """
        lat1, lon1 = math.radians(pos1.lat), math.radians(pos1.lon)
        lat2, lon2 = math.radians(pos2.lat), math.radians(pos2.lon)

        dlon = lon2 - lon1

        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360  # Normalize to 0-360

        return bearing

    @staticmethod
    def calculate_relative_direction(threat_bearing: float, ego_heading: float) -> float:
        """
        Calculate relative direction of threat from ego perspective
        Args:
            threat_bearing: Absolute bearing to threat (0-360°)
            ego_heading: Ego vehicle heading (0-360°)
        Returns: Relative angle for arrow display (0°=forward, 90°=right, etc.)
        """
        relative = threat_bearing - ego_heading
        relative = (relative + 360) % 360  # Normalize to 0-360
        return relative

    @staticmethod
    def calculate_distance(pos1: Position, pos2: Position) -> float:
        """Calculate distance between two positions in meters using Haversine formula"""
        R = 6371000  # Earth radius in meters

        lat1, lon1 = math.radians(pos1.lat), math.radians(pos1.lon)
        lat2, lon2 = math.radians(pos2.lat), math.radians(pos2.lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    @classmethod
    def calculate_threat_direction(cls, ego: EgoState, threat: Threat) -> tuple[float, float]:
        """
        Calculate arrow direction for threat display
        Returns: (arrow_angle, distance_meters)
        """
        bearing = cls.calculate_bearing(ego.position, threat.position)
        relative_direction = cls.calculate_relative_direction(bearing, ego.heading)
        distance = cls.calculate_distance(ego.position, threat.position)

        return relative_direction, distance


class TestRTIDirectionCalculator(unittest.TestCase):
    """Test cases for direction calculation"""

    def setUp(self):
        self.calc = RTIDirectionCalculator()

        # Test location: San Francisco area
        self.ego_base = Position(37.7749, -122.4194)  # SF downtown

    def test_bearing_calculation_cardinals(self):
        """Test bearing calculation for cardinal directions"""
        ego = Position(37.0, -122.0)

        # North
        north = Position(38.0, -122.0)
        self.assertAlmostEqual(self.calc.calculate_bearing(ego, north), 0.0, delta=1)

        # East
        east = Position(37.0, -121.0)
        self.assertAlmostEqual(self.calc.calculate_bearing(ego, east), 90.0, delta=1)

        # South
        south = Position(36.0, -122.0)
        self.assertAlmostEqual(self.calc.calculate_bearing(ego, south), 180.0, delta=1)

        # West
        west = Position(37.0, -123.0)
        self.assertAlmostEqual(self.calc.calculate_bearing(ego, west), 270.0, delta=1)

    def test_relative_direction_calculation(self):
        """Test relative direction calculation"""
        # Ego heading north (0°), threat to the east (90°)
        relative = self.calc.calculate_relative_direction(90.0, 0.0)
        self.assertEqual(relative, 90.0)  # Should point right

        # Ego heading east (90°), threat to the north (0°)
        relative = self.calc.calculate_relative_direction(0.0, 90.0)
        self.assertEqual(relative, 270.0)  # Should point left

        # Ego heading north (0°), threat behind (180°)
        relative = self.calc.calculate_relative_direction(180.0, 0.0)
        self.assertEqual(relative, 180.0)  # Should point back

    def test_distance_calculation(self):
        """Test distance calculation accuracy"""
        # Known distance: SF to Oakland (~13km)
        sf = Position(37.7749, -122.4194)
        oakland = Position(37.8044, -122.2711)

        distance = self.calc.calculate_distance(sf, oakland)
        self.assertTrue(12000 < distance < 14000)  # Approximately 13km

    def test_real_world_scenario(self):
        """Test complete calculation with realistic scenario"""
        # Ego vehicle on Highway 101 heading north
        ego = EgoState(
            position=Position(37.7749, -122.4194),  # SF
            heading=0.0  # North
        )

        # Police threat ahead and to the right
        threat = Threat(
            position=Position(37.7849, -122.4094),  # Slightly north and east
            threat_type="police"
        )

        arrow_angle, distance = self.calc.calculate_threat_direction(ego, threat)

        # Should point roughly northeast (between 0° and 90°)
        self.assertTrue(0 < arrow_angle < 90)
        self.assertTrue(distance > 0)

        print(f"Real-world test: Arrow angle={arrow_angle:.1f}°, Distance={distance:.0f}m")

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        ego = EgoState(Position(37.0, -122.0), 0.0)

        # Same location
        same_threat = Threat(Position(37.0, -122.0), "police")
        angle, distance = self.calc.calculate_threat_direction(ego, same_threat)
        self.assertAlmostEqual(distance, 0.0, delta=1)

        # Crossing 0°/360° boundary
        ego.heading = 350.0  # Nearly north
        north_threat = Threat(Position(37.1, -122.0), "camera")
        angle, distance = self.calc.calculate_threat_direction(ego, north_threat)
        # Should be small positive angle (threat slightly to the right of forward)
        self.assertTrue(0 <= angle <= 30)


def run_interactive_tests():
    """Interactive test scenarios for validation"""
    calc = RTIDirectionCalculator()

    print("=== RTI Direction Calculator Test ===\n")

    scenarios = [
        {
            "name": "Threat directly ahead",
            "ego": EgoState(Position(37.7749, -122.4194), 0.0),  # North
            "threat": Threat(Position(37.7849, -122.4194), "police"),  # Due north
            "expected_angle": "~0° (forward)"
        },
        {
            "name": "Threat to the right",
            "ego": EgoState(Position(37.7749, -122.4194), 0.0),  # North
            "threat": Threat(Position(37.7749, -122.4094), "camera"),  # Due east
            "expected_angle": "~90° (right)"
        },
        {
            "name": "Threat behind",
            "ego": EgoState(Position(37.7749, -122.4194), 0.0),  # North
            "threat": Threat(Position(37.7649, -122.4194), "accident"),  # Due south
            "expected_angle": "~180° (behind)"
        },
        {
            "name": "Threat to the left",
            "ego": EgoState(Position(37.7749, -122.4194), 0.0),  # North
            "threat": Threat(Position(37.7749, -122.4294), "construction"),  # Due west
            "expected_angle": "~270° (left)"
        },
        {
            "name": "Ego heading east, threat north",
            "ego": EgoState(Position(37.7749, -122.4194), 90.0),  # East
            "threat": Threat(Position(37.7849, -122.4194), "hazard"),  # Due north
            "expected_angle": "~270° (left relative to ego)"
        }
    ]

    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"  Ego: {scenario['ego'].position.lat:.4f}, {scenario['ego'].position.lon:.4f}, heading {scenario['ego'].heading}°")
        print(f"  Threat: {scenario['threat'].position.lat:.4f}, {scenario['threat'].position.lon:.4f}")

        angle, distance = calc.calculate_threat_direction(scenario['ego'], scenario['threat'])

        print(f"  Result: Arrow angle = {angle:.1f}°, Distance = {distance:.0f}m")
        print(f"  Expected: {scenario['expected_angle']}")
        print(f"  Status: {'✓' if abs(angle - float(scenario['expected_angle'].split('~')[1].split('°')[0])) < 10 else '⚠'}")
        print()


def generate_arrow_directions():
    """Generate arrow direction mappings for UI implementation"""
    print("=== Arrow Direction Mappings for UI ===\n")

    directions = [
        (0, "↑", "Forward"),
        (45, "↗", "Forward-Right"),
        (90, "→", "Right"),
        (135, "↘", "Back-Right"),
        (180, "↓", "Behind"),
        (225, "↙", "Back-Left"),
        (270, "←", "Left"),
        (315, "↖", "Forward-Left")
    ]

    print("Angle | Arrow | Description")
    print("------|-------|------------")
    for angle, arrow, desc in directions:
        print(f"{angle:3d}°  |   {arrow}   | {desc}")

    print("\nFor smooth rotation, use calculated angle directly in QPainter::rotate()")


if __name__ == "__main__":
    print("Running RTI Direction Calculator Tests...\n")

    # Run unit tests
    unittest.main(argv=[''], verbosity=2, exit=False)

    print("\n" + "="*50)

    # Run interactive tests
    run_interactive_tests()

    # Show arrow mappings
    generate_arrow_directions()
