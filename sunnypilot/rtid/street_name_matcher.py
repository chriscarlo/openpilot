#!/usr/bin/env python3
"""
Street name matching utilities for RTI threat detection.

Provides intelligent street name comparison that handles variations,
abbreviations, and common formatting differences.
"""

import re
from dataclasses import dataclass


@dataclass
class StreetMatchResult:
    """Result of street name matching with confidence."""
    is_match: bool
    confidence: float  # 0.0 to 1.0
    reason: str


class StreetNameMatcher:
    """Intelligent street name matching with normalization."""

    # Common street type abbreviations
    STREET_TYPE_MAPPING = {
        'street': 'st',
        'saint': 'st',  # Handle Saint -> St abbreviation
        'st.': 'st',
        'str': 'st',
        'avenue': 'ave',
        'av': 'ave',
        'av.': 'ave',
        'boulevard': 'blvd',
        'blv': 'blvd',
        'road': 'rd',
        'rd.': 'rd',
        'drive': 'dr',
        'dr.': 'dr',
        'lane': 'ln',
        'ln.': 'ln',
        'court': 'ct',
        'ct.': 'ct',
        'circle': 'cir',
        'cir.': 'cir',
        'place': 'pl',
        'pl.': 'pl',
        'highway': 'hwy',
        'hwy.': 'hwy',
        'freeway': 'fwy',
        'fwy.': 'fwy',
        'expressway': 'expwy',
        'expy': 'expwy',
        'exp': 'expwy',
        'parkway': 'pkwy',
        'pkwy.': 'pkwy',
        'turnpike': 'tpke',
        'tpke.': 'tpke',
        'trail': 'trl',
        'trl.': 'trl',
        'way': 'way',
        'wy': 'way',
        'terrace': 'ter',
        'ter.': 'ter',
        'plaza': 'plz',
        'plz.': 'plz',
        'square': 'sq',
        'sq.': 'sq',
        'alley': 'aly',
        'aly.': 'aly',
    }

    # Directional abbreviations
    DIRECTION_MAPPING = {
        'north': 'n',
        'south': 's',
        'east': 'e',
        'west': 'w',
        'northeast': 'ne',
        'northwest': 'nw',
        'southeast': 'se',
        'southwest': 'sw',
        'northbound': 'nb',
        'southbound': 'sb',
        'eastbound': 'eb',
        'westbound': 'wb',
    }

    # Number word mappings
    NUMBER_WORDS = {
        'first': '1st',
        'second': '2nd',
        'third': '3rd',
        'fourth': '4th',
        'fifth': '5th',
        'sixth': '6th',
        'seventh': '7th',
        'eighth': '8th',
        'ninth': '9th',
        'tenth': '10th',
    }

    # Highway name patterns
    HIGHWAY_PATTERNS = [
        (r'\binterstate[\s-]*(\d+)\b', r'i-\1'),
        (r'\bi[\s-]*(\d+)\b', r'i-\1'),
        (r'\bus[\s-]*highway[\s-]*(\d+)\b', r'us-\1'),
        (r'\bus[\s-]*route[\s-]*(\d+)\b', r'us-\1'),
        (r'\bus[\s-]*(\d+)\b', r'us-\1'),
        (r'\bstate[\s-]*route[\s-]*(\d+)\b', r'sr-\1'),
        (r'\bstate[\s-]*highway[\s-]*(\d+)\b', r'sr-\1'),
        (r'\bsr[\s-]*(\d+)\b', r'sr-\1'),
        (r'\bca[\s-]*(\d+)\b', r'ca-\1'),  # California specific
        (r'\broute[\s-]*(\d+)\b', r'rt-\1'),
        (r'\brt[\s-]*(\d+)\b', r'rt-\1'),
    ]

    @staticmethod
    def normalize_street_name(street_name: str | None) -> str:
        """
        Normalize a street name for comparison.

        Args:
            street_name: Raw street name from API or map data

        Returns:
            Normalized street name in lowercase with standardized abbreviations
        """
        if not street_name:
            return ""

        # Convert to lowercase and strip whitespace
        normalized = street_name.lower().strip()

        # Remove punctuation except hyphens
        normalized = re.sub(r'[^\w\s\-]', '', normalized)

        # Normalize multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Apply highway patterns first (to preserve structure)
        for pattern, replacement in StreetNameMatcher.HIGHWAY_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # Split into words for processing
        words = normalized.split()
        processed_words = []

        for word in words:
            # Check for number words
            if word in StreetNameMatcher.NUMBER_WORDS:
                processed_words.append(StreetNameMatcher.NUMBER_WORDS[word])
            # Check for street type abbreviations
            elif word in StreetNameMatcher.STREET_TYPE_MAPPING:
                processed_words.append(StreetNameMatcher.STREET_TYPE_MAPPING[word])
            # Check for directional abbreviations
            elif word in StreetNameMatcher.DIRECTION_MAPPING:
                processed_words.append(StreetNameMatcher.DIRECTION_MAPPING[word])
            else:
                processed_words.append(word)

        return ' '.join(processed_words)

    @staticmethod
    def extract_core_street_name(normalized_name: str) -> str:
        """
        Extract core street name without directionals or type suffixes.

        Args:
            normalized_name: Already normalized street name

        Returns:
            Core street name component
        """
        if not normalized_name:
            return ""

        # Remove common suffixes
        street_types = set(StreetNameMatcher.STREET_TYPE_MAPPING.values())
        directions = set(StreetNameMatcher.DIRECTION_MAPPING.values())

        words = normalized_name.split()
        core_words = []

        for word in words:
            # Skip if it's a pure street type or direction
            if word not in street_types and word not in directions:
                core_words.append(word)

        return ' '.join(core_words)

    @staticmethod
    def match_street_names(ego_street: str | None,
                           threat_street: str | None,
                           strict_direction: bool = True) -> StreetMatchResult:
        """
        Compare two street names intelligently.

        Args:
            ego_street: Current street name from map data
            threat_street: Threat street name from Waze API
            strict_direction: If True, treats different directions as different roads
                              (e.g., US-101 N vs US-101 S are different)

        Returns:
            StreetMatchResult with match status and confidence
        """
        # Handle missing street names
        if not ego_street or not threat_street:
            return StreetMatchResult(
                is_match=False,
                confidence=0.0,
                reason="Missing street name data",
            )

        # Normalize both street names
        norm_ego = StreetNameMatcher.normalize_street_name(ego_street)
        norm_threat = StreetNameMatcher.normalize_street_name(threat_street)

        # Exact match after normalization
        if norm_ego == norm_threat:
            return StreetMatchResult(
                is_match=True,
                confidence=1.0,
                reason=f"Exact match: '{ego_street}' == '{threat_street}'",
            )

        # Check highway directional variants
        if strict_direction:
            # Different directions = different roads for safety
            directions = ['n', 's', 'e', 'w', 'nb', 'sb', 'eb', 'wb', 'ne', 'nw', 'se', 'sw']
            ego_words = norm_ego.split()
            threat_words = norm_threat.split()

            ego_direction = None
            threat_direction = None

            for d in directions:
                if d in ego_words:
                    ego_direction = d
                    break

            for d in directions:
                if d in threat_words:
                    threat_direction = d
                    break

            if ego_direction and threat_direction and ego_direction != threat_direction:
                # Both have directions and they're different
                return StreetMatchResult(
                    is_match=False,
                    confidence=0.0,
                    reason=f"Different directions: '{ego_street}' != '{threat_street}'",
                )

        # Extract core street names (without type/direction)
        core_ego = StreetNameMatcher.extract_core_street_name(norm_ego)
        core_threat = StreetNameMatcher.extract_core_street_name(norm_threat)

        # Core name match (high confidence)
        if core_ego and core_threat and core_ego == core_threat:
            return StreetMatchResult(
                is_match=True,
                confidence=0.85,
                reason=f"Core name match: '{ego_street}' ~ '{threat_street}'",
            )

        # Fuzzy matching for minor variations
        if StreetNameMatcher._fuzzy_match(core_ego, core_threat):
            return StreetMatchResult(
                is_match=True,
                confidence=0.7,
                reason=f"Fuzzy match: '{ego_street}' ~ '{threat_street}'",
            )

        # Check if one name contains the other
        if core_ego in core_threat or core_threat in core_ego:
            # Substring match (medium confidence)
            return StreetMatchResult(
                is_match=True,
                confidence=0.6,
                reason=f"Partial match: '{ego_street}' ~ '{threat_street}'",
            )

        # No match
        return StreetMatchResult(
            is_match=False,
            confidence=0.0,
            reason=f"No match: '{ego_street}' != '{threat_street}'",
        )

    @staticmethod
    def _fuzzy_match(str1: str, str2: str, threshold: float = 0.85) -> bool:
        """
        Simple fuzzy string matching using Levenshtein distance ratio.

        Args:
            str1: First string
            str2: Second string
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            True if strings are similar enough
        """
        if not str1 or not str2:
            return False

        # Quick exact match check
        if str1 == str2:
            return True

        # Calculate simple similarity ratio
        # This is a simplified version - could use proper Levenshtein
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))

        if max_len == 0:
            return True

        # Too different in length
        if len_diff / max_len > 0.3:
            return False

        # Count matching characters in order
        matches = 0
        j = 0
        for c in str1:
            if j < len(str2) and c == str2[j]:
                matches += 1
                j += 1
            elif j + 1 < len(str2) and c == str2[j + 1]:
                matches += 1
                j += 2  # Allow one insertion/deletion

        similarity = matches / max_len
        return similarity >= threshold

    @staticmethod
    def is_highway_or_expressway(street_name: str | None) -> bool:
        """
        Check if a street name represents a highway or expressway.

        Args:
            street_name: Street name to check

        Returns:
            True if it's a highway/expressway
        """
        if not street_name:
            return False

        # Use normalized version to check patterns
        normalized = StreetNameMatcher.normalize_street_name(street_name)

        # Check for common highway indicators
        highway_indicators = [
            r'\bi-\d+',        # Interstate
            r'\bus-\d+',       # US Highway
            r'\bsr-\d+',       # State Route (normalized)
            r'\bca-\d+',       # California State Route
            r'\bstate route',   # State Route (original)
            r'\bhwy\b',        # Highway
            r'\bfwy\b',        # Freeway
            r'\bexpwy\b',      # Expressway
            r'\bpkwy\b',       # Parkway
            r'\btpke\b',       # Turnpike
        ]

        for pattern in highway_indicators:
            if re.search(pattern, normalized):
                return True

        # Also check original lowercase for patterns not caught by normalization
        original_lower = street_name.lower()
        if any(word in original_lower for word in ['highway', 'freeway', 'expressway', 'parkway', 'turnpike']):
            return True

        return False
