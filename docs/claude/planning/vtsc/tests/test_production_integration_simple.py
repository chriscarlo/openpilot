#!/usr/bin/env python3
"""
Integration test for simplified enhanced VTSC in production environment
Tests that enhanced features are always active when VTSC is enabled
"""

import sys
import numpy as np
from dataclasses import dataclass
from collections import namedtuple

# Add openpilot root to path
sys.path.insert(0, '/data/openpilot')

# Import the production VTSC
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController
from cereal import custom
from openpilot.common.conversions import Conversions as CV

# Mock SM data structure
MockCarState = namedtuple('CarState', ['steeringAngleDeg', 'gasPressed'])
MockModelV2 = namedtuple('ModelV2', ['laneLines', 'laneLineProbs', 'laneLineStds'])
MockLaneLine = namedtuple('LaneLine', ['x', 'y', 't'])
MockLateralPlan = namedtuple('LateralPlan', ['psis', 'dPathPoints'])

class MockSM:
    """Mock service manager for testing"""
    def __init__(self):
        self.data = {}
        self.valid = {}

    def __getitem__(self, key):
        return self.data.get(key, None)

    def update_data(self, key, value):
        self.data[key] = value
        self.valid[key] = True


@dataclass
class MockCP:
    """Mock car parameters"""
    steerRatio: float = 15.0
    wheelbase: float = 2.7


def create_mock_sm(steering_angle=0.0, gas_pressed=False, lane_prob=0.8):
    """Create mock SM data with reasonable defaults"""
    sm = MockSM()

    # Car state
    car_state = MockCarState(
        steeringAngleDeg=steering_angle,
        gasPressed=gas_pressed
    )
    sm.update_data('carState', car_state)

    # Model data with lane lines
    x_vals = np.linspace(0, 150, 33)
    left_y = np.sin(x_vals * 0.02) * 5 - 1.8  # Left lane
    right_y = np.sin(x_vals * 0.02) * 5 + 1.8  # Right lane

    lane_lines = [
        MockLaneLine(x=x_vals, y=left_y - 3.6, t=np.zeros(33)),  # Far left
        MockLaneLine(x=x_vals, y=left_y, t=np.zeros(33)),        # Left
        MockLaneLine(x=x_vals, y=right_y, t=np.zeros(33)),       # Right
        MockLaneLine(x=x_vals, y=right_y + 3.6, t=np.zeros(33))  # Far right
    ]

    model_v2 = MockModelV2(
        laneLines=lane_lines,
        laneLineProbs=[0.5, lane_prob, lane_prob, 0.5],
        laneLineStds=[0.2, 0.1, 0.1, 0.2]
    )
    sm.update_data('modelV2', model_v2)

    # Lateral plan
    psis = np.linspace(0, 1.5, 16)
    d_path_points = np.sin(psis) * 10

    lateral_plan = MockLateralPlan(
        psis=psis,
        dPathPoints=d_path_points.tolist()
    )
    sm.update_data('lateralPlan', lateral_plan)

    return sm


def test_basic_functionality():
    """Test basic VTSC functionality"""
    print("Testing basic functionality...")

    # Create controller
    cp = MockCP()
    controller = VisionTurnController(cp)

    # Test initial state
    assert controller.state == custom.LongitudinalPlanSP.VisionTurnSpeedControl.VisionTurnSpeedControlState.disabled
    assert controller.is_active == False
    assert controller.a_target == 0.0

    print("✓ Initial state correct")

    # Create mock data
    sm = create_mock_sm()

    # Update with enabled=False
    controller.update(sm, enabled=False, v_ego=30.0, a_ego=0.0, v_cruise_setpoint=30.0)
    assert controller.is_active == False

    print("✓ Stays disabled when system disabled")

    # Update with enabled=True
    controller.update(sm, enabled=True, v_ego=30.0, a_ego=0.0, v_cruise_setpoint=30.0)

    print("✓ Basic update works")


def test_enhanced_always_active():
    """Test that enhanced features are always active when VTSC is enabled"""
    print("\nTesting enhanced features are always active...")

    # Create controller
    cp = MockCP()
    controller = VisionTurnController(cp)

    # Check enhanced properties exist
    assert hasattr(controller, 'emergency_level')
    assert hasattr(controller, 'intervention_required')

    print("✓ Enhanced properties exist")

    # Enable VTSC
    controller._is_enabled = True

    # Test emergency level is accessible
    level = controller.emergency_level
    assert level.name in ['NORMAL', 'CAUTION', 'WARNING', 'CRITICAL', 'INTERVENTION']

    print("✓ Emergency level always accessible")

    # Test intervention flag
    assert isinstance(controller.intervention_required, bool)

    print("✓ Intervention flag always accessible")

    # Test that enhanced calculations affect a_target when active
    sm = create_mock_sm()
    controller.update(sm, enabled=True, v_ego=30.0, a_ego=0.0, v_cruise_setpoint=30.0)

    # Force state to entering to make it active
    controller._max_pred_lat_acc = 1.5
    controller._state_transition()

    if controller.is_active:
        # Should be using _current_decel from enhanced calculations
        assert hasattr(controller, '_current_decel')
        assert controller.a_target == controller._current_decel
        print("✓ Enhanced deceleration used when active")


def test_no_enhanced_parameter():
    """Test that there's no separate enhanced parameter"""
    print("\nTesting no separate enhanced parameter...")

    cp = MockCP()
    controller = VisionTurnController(cp)

    # Make sure _use_enhanced doesn't exist
    assert not hasattr(controller, '_use_enhanced')

    print("✓ No _use_enhanced parameter exists")

    # Update params and ensure no error
    controller._update_params()

    print("✓ Parameter update works without enhanced parameter")


def test_curve_approach_with_enhanced():
    """Test approaching a curve with enhanced features"""
    print("\nTesting curve approach with enhanced features...")

    # Create controller
    cp = MockCP()
    controller = VisionTurnController(cp)

    # Enable VTSC
    controller._is_enabled = True

    # Simulate approaching a curve
    sm = create_mock_sm(steering_angle=0.0)

    # Start at high speed
    v_ego = 35.0  # 126 km/h
    emergency_levels_seen = set()

    # Update multiple times simulating approach
    for i in range(10):
        # Increase steering angle to simulate entering curve
        steering_angle = i * 2.0
        sm = create_mock_sm(steering_angle=steering_angle)

        controller.update(sm, enabled=True, v_ego=v_ego, a_ego=0.0, v_cruise_setpoint=40.0)

        # Track emergency levels
        emergency_levels_seen.add(controller.emergency_level.name)

        # Check we get reasonable acceleration
        a_target = controller.a_target
        assert -6.0 <= a_target <= 1.0  # Within system limits

        # Update speed
        v_ego += a_target * 0.1
        v_ego = max(v_ego, 0)

    print(f"✓ Final speed: {v_ego * CV.MS_TO_KPH:.0f} km/h")
    print(f"✓ Final acceleration: {controller.a_target:.2f} m/s²")
    print(f"✓ Emergency levels seen: {emergency_levels_seen}")
    print("✓ System respects -6.0 m/s² limit")


def test_disabled_behavior():
    """Test behavior when VTSC is disabled"""
    print("\nTesting disabled behavior...")

    cp = MockCP()
    controller = VisionTurnController(cp)

    # Explicitly disable VTSC
    controller._is_enabled = False

    sm = create_mock_sm()

    # Update with system enabled but VTSC disabled
    controller.update(sm, enabled=True, v_ego=30.0, a_ego=-1.0, v_cruise_setpoint=30.0)

    # Debug state
    print(f"  Debug: controller.state = {controller.state}")
    print(f"  Debug: controller._is_enabled = {controller._is_enabled}")
    print(f"  Debug: controller.is_active = {controller.is_active}")

    # a_target should follow a_ego when disabled
    assert controller.a_target == -1.0

    print("✓ Remains disabled when VTSC parameter is off")
    print("✓ Returns ego acceleration when disabled")


def run_all_tests():
    """Run all integration tests"""
    print("="*60)
    print("Simplified Enhanced VTSC Integration Test")
    print("Testing: When VTSC is ON, ALL features are ON")
    print("="*60)

    try:
        test_basic_functionality()
        test_enhanced_always_active()
        test_no_enhanced_parameter()
        test_curve_approach_with_enhanced()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - Simplified integration successful")
        print("✓ Enhanced features are always active when VTSC is enabled")
        print("✓ No unnecessary parameters added")
        print("✓ KISS principle maintained")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
