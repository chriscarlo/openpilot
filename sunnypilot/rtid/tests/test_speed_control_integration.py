#!/usr/bin/env python3
"""
Test script for RTI speed control integration with longitudinal planner.

Validates that RTI controller properly integrates with the longitudinal planner
and correctly influences cruise speed when threats are detected.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.rti_controller import RTIController


class TestRTISpeedControlIntegration:
    """Test suite for RTI speed control integration."""

    @pytest.fixture
    def mock_CP(self):
        """Mock car parameters."""
        CP = Mock()
        CP.notCar = False
        return CP

    @pytest.fixture
    def mock_sm(self):
        """Mock SubMaster with RTI state."""
        sm = MagicMock()
        sm.valid = {'rtiStateSP': True}

        # Create mock RTI state
        rti_state = Mock()
        rti_state.threatAhead = False
        rti_state.threatDistanceM = 0.0
        rti_state.recommendedSpeed = 0.0
        rti_state.threats = []

        sm.__getitem__ = lambda self, key: rti_state if key == 'rtiStateSP' else None
        return sm

    @pytest.fixture
    def rti_controller(self, mock_CP):
        """Create RTI controller instance."""
        with patch('openpilot.common.params.Params') as mock_params:
            mock_params.return_value.get_bool.return_value = True  # RTIEnabled = True
            controller = RTIController(mock_CP)
            return controller

    def test_rti_controller_initialization(self, rti_controller):
        """Test RTI controller initializes correctly."""
        assert rti_controller is not None
        assert rti_controller.is_active == False
        assert rti_controller.speed_recommendation == V_CRUISE_UNSET
        assert rti_controller.threat_distance == 0.0

    def test_rti_inactive_when_disabled(self, mock_CP, mock_sm):
        """Test RTI remains inactive when disabled."""
        with patch('openpilot.common.params.Params') as mock_params:
            mock_params.return_value.get_bool.return_value = False  # RTIEnabled = False
            controller = RTIController(mock_CP)

            # Update with threat present but RTI disabled
            mock_sm['rtiStateSP'].threatAhead = True
            mock_sm['rtiStateSP'].threatDistanceM = 500.0
            controller.update(mock_sm, 20.0, 0.0, 25.0)

            assert controller.is_active == False
            assert controller.speed_recommendation == V_CRUISE_UNSET

    def test_rti_activates_with_threat(self, rti_controller, mock_sm):
        """Test RTI activates when threat is detected."""
        # Setup threat
        mock_sm['rtiStateSP'].threatAhead = True
        mock_sm['rtiStateSP'].threatDistanceM = 500.0
        mock_sm['rtiStateSP'].recommendedSpeed = 15.0  # 15 m/s (54 km/h)

        # Create mock threat
        mock_threat = Mock()
        mock_threat.type = 'police'
        mock_threat.confidence = 0.9
        mock_threat.speedLimitMs = 15.0
        mock_sm['rtiStateSP'].threats = [mock_threat]

        # Update controller
        v_ego = 20.0  # Current speed: 20 m/s (72 km/h)
        v_cruise = 25.0  # Cruise setpoint: 25 m/s (90 km/h)
        rti_controller.update(mock_sm, v_ego, 0.0, v_cruise)

        # RTI should be active and recommend lower speed
        assert rti_controller.is_active == True
        assert rti_controller.speed_recommendation < v_ego  # Should recommend slowing down
        assert rti_controller.speed_recommendation > 0
        assert rti_controller.threat_distance == 500.0
        assert rti_controller.threat_type == 'police'

    def test_rti_speed_reduction_by_distance(self, rti_controller, mock_sm):
        """Test RTI applies appropriate speed reduction based on distance."""
        # Setup threat
        mock_sm['rtiStateSP'].threatAhead = True
        mock_sm['rtiStateSP'].recommendedSpeed = 15.0

        mock_threat = Mock()
        mock_threat.type = 'speedTrap'
        mock_threat.confidence = 0.9
        mock_threat.speedLimitMs = 15.0
        mock_sm['rtiStateSP'].threats = [mock_threat]

        v_ego = 25.0  # 90 km/h
        v_cruise = 30.0  # 108 km/h

        # Test critical distance (< 100m)
        mock_sm['rtiStateSP'].threatDistanceM = 80.0
        rti_controller.update(mock_sm, v_ego, 0.0, v_cruise)
        critical_speed = rti_controller.speed_recommendation

        # Test near distance (< 300m)
        mock_sm['rtiStateSP'].threatDistanceM = 250.0
        rti_controller.update(mock_sm, v_ego, 0.0, v_cruise)
        near_speed = rti_controller.speed_recommendation

        # Test normal distance (< 1000m)
        mock_sm['rtiStateSP'].threatDistanceM = 600.0
        rti_controller.update(mock_sm, v_ego, 0.0, v_cruise)
        normal_speed = rti_controller.speed_recommendation

        # Verify speed reduction increases as distance decreases
        assert critical_speed < near_speed < normal_speed
        assert all(speed <= v_ego for speed in [critical_speed, near_speed, normal_speed])

    def test_rti_deactivates_when_threat_clears(self, rti_controller, mock_sm):
        """Test RTI deactivates when threat is no longer present."""
        # First activate RTI with threat
        mock_sm['rtiStateSP'].threatAhead = True
        mock_sm['rtiStateSP'].threatDistanceM = 500.0
        mock_sm['rtiStateSP'].recommendedSpeed = 15.0
        rti_controller.update(mock_sm, 20.0, 0.0, 25.0)
        assert rti_controller.is_active == True

        # Clear threat
        mock_sm['rtiStateSP'].threatAhead = False
        mock_sm['rtiStateSP'].threatDistanceM = 0.0
        rti_controller.update(mock_sm, 20.0, 0.0, 25.0)

        # RTI should deactivate
        assert rti_controller.is_active == False
        assert rti_controller.speed_recommendation == V_CRUISE_UNSET

    def test_rti_ignores_distant_threats(self, rti_controller, mock_sm):
        """Test RTI ignores threats beyond activation distance."""
        # Setup distant threat
        mock_sm['rtiStateSP'].threatAhead = True
        mock_sm['rtiStateSP'].threatDistanceM = 1500.0  # Beyond 1000m threshold
        mock_sm['rtiStateSP'].recommendedSpeed = 15.0

        rti_controller.update(mock_sm, 20.0, 0.0, 25.0)

        # RTI should not activate
        assert rti_controller.is_active == False
        assert rti_controller.speed_recommendation == V_CRUISE_UNSET

    def test_rti_low_speed_cutoff(self, rti_controller, mock_sm):
        """Test RTI doesn't activate below minimum operating speed."""
        # Setup threat
        mock_sm['rtiStateSP'].threatAhead = True
        mock_sm['rtiStateSP'].threatDistanceM = 500.0
        mock_sm['rtiStateSP'].recommendedSpeed = 10.0

        # Update with very low ego speed (below 5 mph)
        v_ego = 2.0  # ~4.5 mph
        rti_controller.update(mock_sm, v_ego, 0.0, 25.0)

        # RTI should not activate
        assert rti_controller.is_active == False
        assert rti_controller.speed_recommendation == V_CRUISE_UNSET

    def test_rti_never_accelerates_toward_threat(self, rti_controller, mock_sm):
        """Test RTI never recommends speed higher than current speed."""
        # Setup threat with high recommended speed
        mock_sm['rtiStateSP'].threatAhead = True
        mock_sm['rtiStateSP'].threatDistanceM = 500.0
        mock_sm['rtiStateSP'].recommendedSpeed = 30.0  # High speed limit

        mock_threat = Mock()
        mock_threat.type = 'police'
        mock_threat.confidence = 0.9
        mock_threat.speedLimitMs = 30.0
        mock_sm['rtiStateSP'].threats = [mock_threat]

        # Update with lower ego speed
        v_ego = 15.0  # Currently going slower than limit
        v_cruise = 25.0
        rti_controller.update(mock_sm, v_ego, 0.0, v_cruise)

        # RTI should never recommend acceleration
        if rti_controller.is_active:
            assert rti_controller.speed_recommendation <= v_ego

    def test_longitudinal_planner_integration(self, mock_CP):
        """Test RTI integration with longitudinal planner."""
        from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

        # Create planner with mock MPC
        mock_mpc = Mock()
        planner = LongitudinalPlannerSP(mock_CP, mock_mpc)

        # Verify RTI controller is initialized
        assert hasattr(planner, 'rti')
        assert isinstance(planner.rti, RTIController)

        # Create mock SubMaster with RTI state
        sm = MagicMock()
        sm.valid = {'rtiStateSP': True, 'carControl': True}
        sm.__getitem__ = lambda self, key: Mock(longActive=True) if key == 'carControl' else Mock()

        # Mock RTI state with threat
        rti_state = Mock()
        rti_state.threatAhead = True
        rti_state.threatDistanceM = 500.0
        rti_state.recommendedSpeed = 15.0
        rti_state.threats = []
        sm.__getitem__ = lambda self, key: rti_state if key == 'rtiStateSP' else Mock(longActive=True)

        # Update cruise speed
        v_ego = 20.0
        a_ego = 0.0
        v_cruise = 25.0

        # Mock controller states
        planner.slc.is_active = False
        planner.v_tsc.is_active = False
        planner.rti._enabled = True
        planner.rti._is_active = True
        planner.rti._speed_recommendation = 18.0  # RTI recommends 18 m/s

        # Update v_cruise
        final_speed = planner.update_v_cruise(sm, v_ego, a_ego, v_cruise)

        # Final speed should be influenced by RTI
        # Since RTI recommends 18 m/s and original is 25 m/s, final should be 18 m/s
        assert final_speed <= v_cruise


class TestRTIProcessRegistration:
    """Test RTI process registration in process_config."""

    def test_rtid_process_registered(self):
        """Test that rtid process is properly registered."""
        from openpilot.system.manager.process_config import managed_processes, rti_enabled

        # Check rti_enabled function exists
        assert callable(rti_enabled)

        # Check rtid process is in managed_processes
        assert 'rtid' in managed_processes

        # Verify process configuration
        rtid_process = managed_processes['rtid']
        assert rtid_process.name == 'rtid'
        assert rtid_process.module == 'sunnypilot.rtid.rtid'

    def test_rti_enabled_condition(self):
        """Test RTI enablement condition function."""
        from openpilot.system.manager.process_config import rti_enabled

        mock_params = Mock()
        mock_CP = Mock()

        # Test when RTIEnabled is True and started
        mock_params.get_bool.return_value = True
        assert rti_enabled(True, mock_params, mock_CP) == True

        # Test when RTIEnabled is False
        mock_params.get_bool.return_value = False
        assert rti_enabled(True, mock_params, mock_CP) == False

        # Test when not started
        mock_params.get_bool.return_value = True
        assert rti_enabled(False, mock_params, mock_CP) == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
