#!/usr/bin/env python3
"""
Test script to validate RTI speed limit bug fix.

Tests that RTI correctly uses actual posted speed limits from map data
instead of hardcoded default values.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project to path
sys.path.insert(0, '/projects/chauffeur/data/openpilot')

from sunnypilot.rtid.threat_detector import ThreatDetector, ProcessedThreat
from sunnypilot.rtid.waze_api_client import WazeAlert


class TestRTISpeedLimitFix(unittest.TestCase):
    """Test cases for RTI speed limit bug fix."""

    def setUp(self):
        """Set up test environment."""
        # Mock params
        with patch('openpilot.common.params.Params') as mock_params:
            mock_params_instance = MagicMock()
            mock_params_instance.get.side_effect = self._mock_params_get
            mock_params.return_value = mock_params_instance
            
            self.detector = ThreatDetector()

    def _mock_params_get(self, key):
        """Mock params.get() responses."""
        params = {
            'RTIDetectionRadius': b'4828',  # 3 miles
            'RTIForwardSlowdownRange': b'1609',  # 1 mile
            'RTIResumeSpeedDistance': b'1609',  # 1 mile
            'RTISpeedReductionMode': b'posted',  # Use posted speed limit
            'RTISpeedReduction': b'16',  # 10 mph
            'RTIThreatFilter': b'0',  # All threats
        }
        return params.get(key)

    def test_uses_actual_posted_speed_limit(self):
        """Test that RTI uses actual posted speed limit from map data."""
        # Create a police threat
        threat = WazeAlert(
            id='test_police_1',
            type='police',
            latitude=37.7749,
            longitude=-122.4194,
            confidence=0.9,
            speed_limit=None  # No speed limit in Waze data
        )
        
        # Test location and speed
        current_location = (37.7750, -122.4195)  # Very close to threat
        current_speed = 29.0  # 65 mph
        v_cruise = 29.0  # Cruise set to 65 mph
        
        # Test with actual posted speed limit from map data
        posted_speed_limit = 29.0  # 65 mph speed limit from map
        
        # Process threats
        rti_state = self.detector.process_threats(
            traffic_data=[threat],
            current_location=current_location,
            current_speed=current_speed,
            timestamp=1234567890,
            v_cruise=v_cruise,
            posted_speed_limit=posted_speed_limit
        )
        
        # Verify that recommended speed is based on actual speed limit
        # Should be 29.0 m/s (posted limit) or slightly less, not 25 m/s (hardcoded)
        self.assertGreater(rti_state.recommended_speed, 25.0,
                          "Should use actual speed limit, not hardcoded 56 mph")
        
        # Verify threat has correct speed limit
        if rti_state.threats:
            threat_speed = rti_state.threats[0].speed_limit_ms
            self.assertEqual(threat_speed, 29.0,
                           "Threat should have actual posted speed limit")

    def test_falls_back_when_no_speed_limit(self):
        """Test fallback behavior when no posted speed limit available."""
        # Create a police threat
        threat = WazeAlert(
            id='test_police_2',
            type='police',
            latitude=37.7749,
            longitude=-122.4194,
            confidence=0.9,
            speed_limit=None
        )
        
        # Test location and speed
        current_location = (37.7750, -122.4195)
        current_speed = 29.0  # 65 mph
        v_cruise = 29.0
        
        # No posted speed limit available
        posted_speed_limit = 0.0
        
        # Process threats
        rti_state = self.detector.process_threats(
            traffic_data=[threat],
            current_location=current_location,
            current_speed=current_speed,
            timestamp=1234567890,
            v_cruise=v_cruise,
            posted_speed_limit=posted_speed_limit
        )
        
        # Should fall back to default (25 m/s) when no speed limit available
        if rti_state.threats:
            threat_speed = rti_state.threats[0].speed_limit_ms
            self.assertEqual(threat_speed, 25.0,
                           "Should use default when no posted speed limit")

    def test_speed_limit_conversion(self):
        """Test correct unit conversion for different speed limits."""
        test_cases = [
            (55 * 0.44704, "55 mph highway"),  # 24.59 m/s
            (65 * 0.44704, "65 mph freeway"),  # 29.06 m/s
            (75 * 0.44704, "75 mph interstate"),  # 33.53 m/s
            (35 * 0.44704, "35 mph city street"),  # 15.65 m/s
        ]
        
        for speed_limit_ms, description in test_cases:
            with self.subTest(description=description):
                threat = WazeAlert(
                    id=f'test_{description}',
                    type='speedTrap',
                    latitude=37.7749,
                    longitude=-122.4194,
                    confidence=0.9,
                    speed_limit=None
                )
                
                current_location = (37.7750, -122.4195)
                current_speed = speed_limit_ms + 2  # Slightly over limit
                v_cruise = speed_limit_ms + 2
                
                rti_state = self.detector.process_threats(
                    traffic_data=[threat],
                    current_location=current_location,
                    current_speed=current_speed,
                    timestamp=1234567890,
                    v_cruise=v_cruise,
                    posted_speed_limit=speed_limit_ms
                )
                
                if rti_state.threats:
                    threat_speed = rti_state.threats[0].speed_limit_ms
                    self.assertAlmostEqual(threat_speed, speed_limit_ms, places=1,
                                         msg=f"Speed limit conversion failed for {description}")

    def test_no_dangerous_slowdown(self):
        """Test that RTI doesn't recommend dangerously slow speeds."""
        # Create threat on 65 mph freeway
        threat = WazeAlert(
            id='freeway_police',
            type='police',
            latitude=37.7749,
            longitude=-122.4194,
            confidence=0.9,
            speed_limit=None
        )
        
        current_location = (37.7750, -122.4195)
        current_speed = 29.0  # 65 mph
        v_cruise = 29.0
        posted_speed_limit = 29.0  # 65 mph freeway
        
        rti_state = self.detector.process_threats(
            traffic_data=[threat],
            current_location=current_location,
            current_speed=current_speed,
            timestamp=1234567890,
            v_cruise=v_cruise,
            posted_speed_limit=posted_speed_limit
        )
        
        # Should never recommend below 40 mph (17.9 m/s) on a 65 mph freeway
        if rti_state.recommended_speed > 0:
            self.assertGreater(rti_state.recommended_speed, 17.9,
                             "Should not recommend dangerously slow speeds on freeway")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)