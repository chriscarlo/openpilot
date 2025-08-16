#!/usr/bin/env python3
"""
Pytest configuration and fixtures for RTI test suite.

Provides mock data, fixtures, and testing utilities for comprehensive
RTI system testing following openpilot patterns.
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from sunnypilot.rtid.waze_api_client import WazeAlert
from sunnypilot.rtid.threat_detector import RTIState, ProcessedThreat


# Test data for consistent testing
MOCK_GPS_LOCATIONS = {
    'mountain_view': (37.4221, -122.0841),  # Mountain View, CA
    'highway_101': (37.4419, -122.1430),   # Highway 101 near Google
    'city_street': (37.4045, -122.0785),   # City street in Mountain View
    'freeway': (37.4500, -122.1800),       # Freeway location
}

MOCK_WAZE_ALERTS = [
    {
        'alert_id': 'police-alert-001',
        'type': 'POLICE',
        'subtype': 'POLICE_HIDING',
        'latitude': 37.4221,
        'longitude': -122.0841,
        'alert_confidence': 0.8,
        'alert_reliability': 5,
        'street': 'Main Street',
        'country': 'US'
    },
    {
        'alert_id': 'speed-trap-002',
        'type': 'POLICE',
        'subtype': '',
        'latitude': 37.4419,
        'longitude': -122.1430,
        'alert_confidence': 0.9,
        'alert_reliability': 7,
        'street': 'Highway 101',
        'country': 'US'
    },
    {
        'alert_id': 'accident-003',
        'type': 'HAZARD',
        'subtype': 'HAZARD_ON_ROAD',
        'latitude': 37.4045,
        'longitude': -122.0785,
        'alert_confidence': 0.7,
        'alert_reliability': 4,
        'street': 'City Boulevard',
        'country': 'US'
    }
]

MOCK_WAZE_JAMS = [
    {
        'uuid': 'jam-001',
        'line': [{'x': -122.1800, 'y': 37.4500}],
        'level': 3,  # Traffic level 1-5
        'speedKMH': 40,
        'street': 'Interstate 280'
    }
]


@pytest.fixture
def mock_waze_api_response():
    """Mock Waze API response with real API structure."""
    return {
        'status': 'OK',
        'data': {
            'alerts': MOCK_WAZE_ALERTS,
            'jams': MOCK_WAZE_JAMS
        }
    }


@pytest.fixture
def sample_waze_alerts():
    """List of WazeAlert objects for testing."""
    alerts = []
    for alert_data in MOCK_WAZE_ALERTS:
        alert = WazeAlert(
            id=alert_data['alert_id'],
            type=alert_data['type'].lower(),
            latitude=alert_data['latitude'],
            longitude=alert_data['longitude'],
            confidence=alert_data['alert_confidence'],
            speed_limit=None,  # Real API doesn't provide speed_limit
            street=alert_data.get('street'),
            country=alert_data.get('country'),
            raw_data=alert_data
        )
        alerts.append(alert)
    return alerts


@pytest.fixture
def mock_gps_location():
    """Mock GPS location (Mountain View, CA)."""
    return MOCK_GPS_LOCATIONS['mountain_view']


@pytest.fixture
def mock_car_state():
    """Mock car state for testing."""
    return {
        'vEgo': 25.0,  # 25 m/s (~55 mph)
        'aEgo': 0.0,
        'steerAngle': 0.0,
        'leftBlinker': False,
        'rightBlinker': False
    }


@pytest.fixture
def mock_messaging():
    """Mock cereal messaging for testing."""
    with patch('sunnypilot.rtid.rtid.messaging') as mock_msg:
        # Mock SubMaster
        mock_sm = MagicMock()
        mock_sm.update = MagicMock()
        mock_sm.updated = {'gpsLocationExternal': True, 'carState': True}
        mock_sm.__getitem__ = MagicMock()
        mock_msg.SubMaster.return_value = mock_sm

        # Mock PubMaster
        mock_pm = MagicMock()
        mock_pm.send = MagicMock()
        mock_msg.PubMaster.return_value = mock_pm

        # Mock new_message
        mock_msg.new_message = MagicMock()

        yield mock_msg


@pytest.fixture
def mock_params():
    """Mock openpilot Params for testing."""
    with patch('sunnypilot.rtid.rtid.Params') as mock_params_class:
        mock_params_instance = MagicMock()
        mock_params_instance.get_bool.return_value = True  # RTI enabled by default
        mock_params_class.return_value = mock_params_instance
        yield mock_params_instance


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def mock_api_key_file(tmp_path, mock_api_key):
    """Create mock API key file for testing."""
    key_file = tmp_path / "waze_api_key.json"
    key_data = {"api_key": mock_api_key}
    key_file.write_text(json.dumps(key_data))
    return str(key_file)


@pytest.fixture
async def mock_http_session():
    """Mock aiohttp session for API testing."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"alerts": MOCK_WAZE_ALERTS})
    mock_response.text = AsyncMock(return_value="OK")

    mock_session.get.return_value.__aenter__.return_value = mock_response
    return mock_session


@pytest.fixture
def processed_threat():
    """Sample ProcessedThreat for testing."""
    return ProcessedThreat(
        id='test-threat-001',
        type='police',
        latitude=37.4221,
        longitude=-122.0841,
        distance=500.0,  # 500m ahead
        direction='ahead',
        confidence=0.8,
        speed_limit_ms=11.18,  # 25 mph in m/s
        on_same_road=True
    )


@pytest.fixture
def rti_state():
    """Sample RTI state for testing."""
    return RTIState(
        timestamp=1234567890,
        threat_ahead=True,
        threat_distance_m=500.0,
        recommended_speed=11.18,  # 25 mph in m/s
        source='waze',
        api_status='connected',
        threats=[]
    )


@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return None

    return Timer()


# Pytest markers for test organization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test (budget: 15ms)"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API integration test"
    )
    config.addinivalue_line(
        "markers", "safety: mark test as safety validation test"
    )


# Utility functions for testing
def assert_speed_recommendation_safe(recommended_speed: float, current_speed: float):
    """Assert that speed recommendation follows safety constraints."""
    assert 0.0 <= recommended_speed <= current_speed * 1.2, \
        f"Unsafe speed recommendation: {recommended_speed:.1f} m/s for current {current_speed:.1f} m/s"


def assert_performance_budget(elapsed_ms: float, budget_ms: float = 15.0):
    """Assert that operation completed within performance budget."""
    assert elapsed_ms <= budget_ms, \
        f"Performance budget exceeded: {elapsed_ms:.1f}ms > {budget_ms:.1f}ms"


def create_mock_waze_alert(alert_type: str = 'police',
                          distance_from_ego: float = 500.0,
                          confidence: float = 0.8) -> WazeAlert:
    """Create a mock WazeAlert for testing."""
    # Calculate coordinates roughly distance_from_ego meters north of Mountain View
    lat_offset = distance_from_ego / 111000  # Rough meters to degrees conversion

    return WazeAlert(
        id=f'test-{alert_type}-001',
        type=alert_type,
        latitude=MOCK_GPS_LOCATIONS['mountain_view'][0] + lat_offset,
        longitude=MOCK_GPS_LOCATIONS['mountain_view'][1],
        confidence=confidence,
        speed_limit=25,  # mph
        street='Test Street',
        country='US',
        raw_data={}
    )
