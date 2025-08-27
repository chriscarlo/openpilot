#!/usr/bin/env python3
"""
Unit tests for street name matching utilities.

Tests cover normalization, matching logic, confidence scoring,
and highway/direction handling.
"""

import unittest
from sunnypilot.rtid.street_name_matcher import StreetNameMatcher, StreetMatchResult


class TestStreetNameNormalization(unittest.TestCase):
    """Test street name normalization."""
    
    def test_basic_normalization(self):
        """Test basic normalization of street names."""
        # Case and whitespace
        self.assertEqual(StreetNameMatcher.normalize_street_name("Main Street"), "main st")
        self.assertEqual(StreetNameMatcher.normalize_street_name("  MAIN   STREET  "), "main st")
        
        # Common abbreviations
        self.assertEqual(StreetNameMatcher.normalize_street_name("Lawrence Avenue"), "lawrence ave")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Elm Boulevard"), "elm blvd")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Oak Drive"), "oak dr")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Pine Road"), "pine rd")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Maple Lane"), "maple ln")
        
    def test_directional_normalization(self):
        """Test normalization of directional indicators."""
        self.assertEqual(StreetNameMatcher.normalize_street_name("North Main Street"), "n main st")
        self.assertEqual(StreetNameMatcher.normalize_street_name("South Broadway"), "s broadway")
        self.assertEqual(StreetNameMatcher.normalize_street_name("East 1st Street"), "e 1st st")
        self.assertEqual(StreetNameMatcher.normalize_street_name("West Ave"), "w ave")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Northeast Parkway"), "ne pkwy")
        
    def test_highway_normalization(self):
        """Test normalization of highway names."""
        # Interstate
        self.assertEqual(StreetNameMatcher.normalize_street_name("Interstate 5"), "i-5")
        self.assertEqual(StreetNameMatcher.normalize_street_name("I-5"), "i-5")
        self.assertEqual(StreetNameMatcher.normalize_street_name("I 5"), "i-5")
        
        # US Highway
        self.assertEqual(StreetNameMatcher.normalize_street_name("US Highway 101"), "us-101")
        self.assertEqual(StreetNameMatcher.normalize_street_name("US Route 101"), "us-101")
        self.assertEqual(StreetNameMatcher.normalize_street_name("US-101"), "us-101")
        self.assertEqual(StreetNameMatcher.normalize_street_name("US 101"), "us-101")
        
        # State Routes
        self.assertEqual(StreetNameMatcher.normalize_street_name("State Route 17"), "sr-17")
        self.assertEqual(StreetNameMatcher.normalize_street_name("State Highway 87"), "sr-87")
        self.assertEqual(StreetNameMatcher.normalize_street_name("CA-87"), "ca-87")
        self.assertEqual(StreetNameMatcher.normalize_street_name("SR 17"), "sr-17")
        
    def test_number_word_normalization(self):
        """Test normalization of number words."""
        self.assertEqual(StreetNameMatcher.normalize_street_name("First Street"), "1st st")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Second Avenue"), "2nd ave")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Third Boulevard"), "3rd blvd")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Fourth Road"), "4th rd")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Fifth Lane"), "5th ln")
        
    def test_special_characters(self):
        """Test handling of special characters."""
        self.assertEqual(StreetNameMatcher.normalize_street_name("St. Mary's Road"), "st marys rd")
        self.assertEqual(StreetNameMatcher.normalize_street_name("Martin Luther King Jr. Blvd"), "martin luther king jr blvd")
        self.assertEqual(StreetNameMatcher.normalize_street_name("O'Farrell Street"), "ofarrell st")
        
    def test_empty_and_none(self):
        """Test handling of empty and None inputs."""
        self.assertEqual(StreetNameMatcher.normalize_street_name(None), "")
        self.assertEqual(StreetNameMatcher.normalize_street_name(""), "")
        self.assertEqual(StreetNameMatcher.normalize_street_name("   "), "")


class TestCoreStreetNameExtraction(unittest.TestCase):
    """Test extraction of core street names."""
    
    def test_remove_street_types(self):
        """Test removal of street type suffixes."""
        self.assertEqual(StreetNameMatcher.extract_core_street_name("main st"), "main")
        self.assertEqual(StreetNameMatcher.extract_core_street_name("elm ave"), "elm")
        self.assertEqual(StreetNameMatcher.extract_core_street_name("oak blvd"), "oak")
        
    def test_remove_directions(self):
        """Test removal of directional prefixes."""
        self.assertEqual(StreetNameMatcher.extract_core_street_name("n main st"), "main")
        self.assertEqual(StreetNameMatcher.extract_core_street_name("s broadway ave"), "broadway")
        self.assertEqual(StreetNameMatcher.extract_core_street_name("e 1st st"), "1st")
        
    def test_preserve_core_name(self):
        """Test preservation of core street name."""
        self.assertEqual(StreetNameMatcher.extract_core_street_name("martin luther king jr blvd"), 
                        "martin luther king jr")
        self.assertEqual(StreetNameMatcher.extract_core_street_name("i-5"), "i-5")
        self.assertEqual(StreetNameMatcher.extract_core_street_name("us-101"), "us-101")


class TestStreetNameMatching(unittest.TestCase):
    """Test street name matching logic."""
    
    def test_exact_matches(self):
        """Test exact street name matches."""
        result = StreetNameMatcher.match_street_names("Main Street", "Main St")
        self.assertTrue(result.is_match)
        self.assertEqual(result.confidence, 1.0)
        
        result = StreetNameMatcher.match_street_names("Lawrence Expressway", "Lawrence Expwy")
        self.assertTrue(result.is_match)
        self.assertEqual(result.confidence, 1.0)
        
    def test_core_name_matches(self):
        """Test matches based on core street names."""
        # For non-highway streets, different directions shouldn't matter
        result = StreetNameMatcher.match_street_names("North Main Street", "South Main Avenue", strict_direction=False)
        self.assertTrue(result.is_match)
        self.assertGreaterEqual(result.confidence, 0.8)
        
        result = StreetNameMatcher.match_street_names("Elm Street", "Elm Avenue")
        self.assertTrue(result.is_match)
        self.assertGreaterEqual(result.confidence, 0.8)
        
    def test_no_matches(self):
        """Test non-matching street names."""
        result = StreetNameMatcher.match_street_names("Main Street", "Oak Avenue")
        self.assertFalse(result.is_match)
        self.assertEqual(result.confidence, 0.0)
        
        result = StreetNameMatcher.match_street_names("First Street", "Second Street")
        self.assertFalse(result.is_match)
        self.assertEqual(result.confidence, 0.0)
        
    def test_partial_matches(self):
        """Test partial/substring matches."""
        result = StreetNameMatcher.match_street_names("Main", "Main Street")
        self.assertTrue(result.is_match)
        self.assertGreaterEqual(result.confidence, 0.6)
        
        result = StreetNameMatcher.match_street_names("Lawrence", "Lawrence Expressway")
        self.assertTrue(result.is_match)
        self.assertGreaterEqual(result.confidence, 0.6)
        
    def test_fuzzy_matches(self):
        """Test fuzzy matching for minor variations."""
        # This tests the internal _fuzzy_match function indirectly
        result = StreetNameMatcher.match_street_names("Lawrance Expressway", "Lawrence Expressway")  # Typo
        self.assertTrue(result.is_match)
        self.assertGreaterEqual(result.confidence, 0.7)
        
    def test_missing_data(self):
        """Test handling of missing street names."""
        result = StreetNameMatcher.match_street_names(None, "Main Street")
        self.assertFalse(result.is_match)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("Missing", result.reason)
        
        result = StreetNameMatcher.match_street_names("Main Street", None)
        self.assertFalse(result.is_match)
        self.assertEqual(result.confidence, 0.0)
        
        result = StreetNameMatcher.match_street_names(None, None)
        self.assertFalse(result.is_match)
        self.assertEqual(result.confidence, 0.0)


class TestHighwayDirectionalMatching(unittest.TestCase):
    """Test highway and directional matching logic."""
    
    def test_highway_detection(self):
        """Test detection of highways and expressways."""
        self.assertTrue(StreetNameMatcher.is_highway_or_expressway("I-5"))
        self.assertTrue(StreetNameMatcher.is_highway_or_expressway("US-101"))
        self.assertTrue(StreetNameMatcher.is_highway_or_expressway("CA-87"))
        self.assertTrue(StreetNameMatcher.is_highway_or_expressway("State Route 17"))
        self.assertTrue(StreetNameMatcher.is_highway_or_expressway("Lawrence Expressway"))
        self.assertTrue(StreetNameMatcher.is_highway_or_expressway("Central Freeway"))
        
        self.assertFalse(StreetNameMatcher.is_highway_or_expressway("Main Street"))
        self.assertFalse(StreetNameMatcher.is_highway_or_expressway("Elm Avenue"))
        self.assertFalse(StreetNameMatcher.is_highway_or_expressway("Oak Boulevard"))
        
    def test_strict_directional_matching(self):
        """Test strict directional matching for highways."""
        # With strict direction enabled (default for highways)
        result = StreetNameMatcher.match_street_names("US-101 N", "US-101 S", strict_direction=True)
        self.assertFalse(result.is_match)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("Different directions", result.reason)
        
        result = StreetNameMatcher.match_street_names("I-5 Northbound", "I-5 Southbound", strict_direction=True)
        self.assertFalse(result.is_match)
        
        # Same direction should match
        result = StreetNameMatcher.match_street_names("US-101 N", "US-101 North", strict_direction=True)
        self.assertTrue(result.is_match)
        self.assertEqual(result.confidence, 1.0)
        
    def test_non_strict_directional_matching(self):
        """Test non-strict directional matching."""
        # With strict direction disabled
        result = StreetNameMatcher.match_street_names("US-101 N", "US-101 S", strict_direction=False)
        self.assertTrue(result.is_match)
        self.assertGreaterEqual(result.confidence, 0.8)
        
        result = StreetNameMatcher.match_street_names("Main Street North", "Main Street South", strict_direction=False)
        self.assertTrue(result.is_match)
        self.assertGreaterEqual(result.confidence, 0.8)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world street name scenarios from Waze data."""
    
    def test_waze_street_names(self):
        """Test matching with actual Waze street names."""
        # From the captured Waze data
        test_cases = [
            ("Lawrence Expressway", "Lawrence Expwy", True, 1.0),
            ("US-101 S", "US Highway 101 South", True, 1.0),
            ("US-101 S", "US-101 N", False, 0.0),  # Different directions
            ("I-280", "Interstate 280", True, 1.0),
            ("El Camino Real", "El Camino", True, 0.6),  # Partial match
            ("Stevens Creek Boulevard", "Stevens Creek Blvd", True, 1.0),
            ("De Anza Blvd", "De Anza Boulevard", True, 1.0),
        ]
        
        for ego_street, threat_street, expected_match, min_confidence in test_cases:
            result = StreetNameMatcher.match_street_names(ego_street, threat_street)
            self.assertEqual(result.is_match, expected_match,
                           f"Failed for {ego_street} vs {threat_street}")
            if expected_match:
                self.assertGreaterEqual(result.confidence, min_confidence,
                                      f"Low confidence for {ego_street} vs {threat_street}")
    
    def test_edge_cases(self):
        """Test edge cases and special scenarios."""
        # Very long street names
        long_name = "Martin Luther King Junior Memorial Boulevard"
        result = StreetNameMatcher.match_street_names(long_name, "MLK Jr Blvd")
        # This might not match perfectly due to abbreviation differences
        
        # Numbers in street names
        result = StreetNameMatcher.match_street_names("42nd Street", "42nd St")
        self.assertTrue(result.is_match)
        self.assertEqual(result.confidence, 1.0)
        
        # Mixed case and punctuation
        result = StreetNameMatcher.match_street_names("St. Mary's Road", "Saint Marys Rd")
        self.assertTrue(result.is_match)
        
        # International variations (if supported)
        result = StreetNameMatcher.match_street_names("Route 66", "RT-66")
        self.assertTrue(result.is_match)


if __name__ == '__main__':
    unittest.main()