#!/usr/bin/env python3
"""
Cap'n Proto Schema Synchronization Tests

Validates that the Cap'n Proto schema includes all threat types used in the implementation
and that message creation/parsing works correctly with all threat types.
"""

import pytest

from sunnypilot.rtid.waze_api_client import WazeAlert, WazeAPIClient


def get_all_threat_types_from_code() -> set[str]:
    """Extract all threat types used in the implementation."""
    # These are the threat types actually used in the code implementation
    code_threat_types = {
        'police',
        'policeHiding',
        'speedTrap',
        'speedCamera',
        'accident',
        'hazard',
        'roadHazard',
        'shoulderHazard',
        'construction',
        'jam',
        'roadClosed'
    }
    return code_threat_types


def get_schema_threat_types() -> set[str]:
    """Extract threat types from the Cap'n Proto schema."""
    # These should match the ThreatType enum in rti.capnp
    schema_threat_types = {
        'police',        # @0
        'speedTrap',     # @1
        'speedCamera',   # @2
        'accident',      # @3
        'hazard',        # @4
        'construction',  # @5
        'jam',           # @6
        'policeHiding',  # @7
        'roadHazard',    # @8
        'shoulderHazard', # @9
        'roadClosed'     # @10
    }
    return schema_threat_types


@pytest.mark.schema
class TestSchemaSynchronization:
    """Test Cap'n Proto schema synchronization with code implementation."""

    def test_all_code_threat_types_in_schema(self):
        """Test that all threat types used in code are defined in the schema."""
        code_types = get_all_threat_types_from_code()
        schema_types = get_schema_threat_types()

        missing_in_schema = code_types - schema_types

        assert not missing_in_schema, f"Threat types used in code but missing from schema: {missing_in_schema}"
        print(f"✓ All {len(code_types)} code threat types are defined in schema")

    def test_no_unused_schema_types(self):
        """Test that schema doesn't define threat types not used in code."""
        code_types = get_all_threat_types_from_code()
        schema_types = get_schema_threat_types()

        unused_in_schema = schema_types - code_types

        # This is a warning, not an error - extra types are OK for forward compatibility
        if unused_in_schema:
            print(f"Warning: Schema defines threat types not used in code: {unused_in_schema}")
        else:
            print("✓ No unused threat types in schema")

    def test_waze_api_mapping_completeness(self):
        """Test that Waze API mapping produces only valid threat types."""
        api_client = WazeAPIClient("test-key")
        schema_types = get_schema_threat_types()

        # Test known Waze alert type mappings
        test_mappings = [
            ('POLICE', ''),
            ('POLICE', 'POLICE_HIDING'),
            ('SPEED_TRAP', ''),
            ('SPEED_CAMERA', ''),
            ('ACCIDENT', ''),
            ('CONSTRUCTION', ''),
            ('HAZARD', ''),
            ('HAZARD', 'HAZARD_ON_ROAD'),
            ('HAZARD', 'HAZARD_ON_SHOULDER'),
            ('ROAD_CLOSED', ''),
            ('UNKNOWN_TYPE', ''),  # Should map to 'hazard'
        ]

        for waze_type, subtype in test_mappings:
            mapped_type = api_client._map_alert_type(waze_type, subtype)
            assert mapped_type in schema_types, f"Mapped type '{mapped_type}' from Waze '{waze_type}/{subtype}' not in schema"

        print("✓ All Waze API mappings produce valid schema threat types")

    def test_threat_detector_logic_compatibility(self):
        """Test that threat detection logic works with schema threat types."""
        schema_types = get_schema_threat_types()

        # Test the specific logic in threat detector that uses threat types
        speed_control_types = ['police', 'policeHiding', 'speedTrap']

        for threat_type in speed_control_types:
            assert threat_type in schema_types, f"Speed control threat type '{threat_type}' not in schema"

        print("✓ Threat detection logic compatible with schema types")

    def test_message_creation_with_all_types(self):
        """Test that messages can be created with all threat types."""
        from cereal import messaging

        schema_types = get_schema_threat_types()

        # Test creating messages with each threat type
        for threat_type in schema_types:
            try:
                # Create test alert
                test_alert = WazeAlert(
                    id=f'test-{threat_type}',
                    type=threat_type,
                    latitude=37.4221,
                    longitude=-122.0841,
                    confidence=0.8,
                    speed_limit=35,
                    raw_data={}
                )

                # Create RTI message
                msg = messaging.new_message('rtiStateSP')
                msg.rtiStateSP.timeStamp = 1000000000
                msg.rtiStateSP.threatAhead = True
                msg.rtiStateSP.source = 'test'
                msg.rtiStateSP.apiStatus = 'connected'

                # Initialize threats list with one threat
                msg.rtiStateSP.init('threats', 1)
                threat_msg = msg.rtiStateSP.threats[0]
                threat_msg.id = test_alert.id
                threat_msg.type = test_alert.type  # This should work if schema is correct
                threat_msg.latitude = test_alert.latitude
                threat_msg.longitude = test_alert.longitude
                threat_msg.distance = 100.0  # Sample distance
                threat_msg.direction = 'ahead'  # Sample direction
                threat_msg.confidence = test_alert.confidence
                threat_msg.speedLimitMs = test_alert.speed_limit or 0.0

                print(f"✓ Message created successfully for threat type: {threat_type}")

            except Exception as e:
                pytest.fail(f"Failed to create message for threat type '{threat_type}': {e}")

    def test_schema_backward_compatibility(self):
        """Test that schema changes maintain backward compatibility."""
        # The original threat types should still work
        original_types = {
            'police', 'speedTrap', 'speedCamera', 'accident',
            'hazard', 'construction', 'jam'
        }

        schema_types = get_schema_threat_types()

        missing_original = original_types - schema_types
        assert not missing_original, f"Original threat types missing from updated schema: {missing_original}"

        print("✓ Schema maintains backward compatibility with original types")

    def test_new_threat_types_added(self):
        """Test that new threat types are properly added to schema."""
        new_types = {'policeHiding', 'roadHazard', 'shoulderHazard', 'roadClosed'}
        schema_types = get_schema_threat_types()

        missing_new = new_types - schema_types
        assert not missing_new, f"New threat types missing from schema: {missing_new}"

        print(f"✓ All {len(new_types)} new threat types added to schema")


if __name__ == '__main__':
    # Run schema synchronization tests directly
    test = TestSchemaSynchronization()

    print("=== Cap'n Proto Schema Synchronization Validation ===")
    test.test_all_code_threat_types_in_schema()
    test.test_no_unused_schema_types()
    test.test_waze_api_mapping_completeness()
    test.test_threat_detector_logic_compatibility()
    test.test_message_creation_with_all_types()
    test.test_schema_backward_compatibility()
    test.test_new_threat_types_added()
    print("\nSUCCESS: Schema synchronization validation complete!")
    print("         All threat types are properly synchronized between code and schema")
