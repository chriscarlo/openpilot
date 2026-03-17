"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from parameterized import parameterized

from cereal import car, log

from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, LaneChangeDirection, LaneChangeState, TURN_DESIRE_SPEED_MAX


class TestTurnDesires:

  def setup_method(self):
    self.dh = DesireHelper()

  def _car_state(self, speed_mph, *, left=False, right=False, steering_pressed=False, steering_torque=0.0):
    cs = car.CarState.new_message()
    cs.vEgo = speed_mph * CV.MPH_TO_MS
    cs.leftBlinker = left
    cs.rightBlinker = right
    cs.steeringPressed = steering_pressed
    cs.steeringTorque = steering_torque
    cs.leftBlindspot = False
    cs.rightBlindspot = False
    cs.brakePressed = False
    return cs

  @parameterized.expand([
    ("left", True, False, log.Desire.turnLeft),
    ("right", False, True, log.Desire.turnRight),
  ])
  def test_turn_desire_arms_below_lane_change_speed(self, _name, left, right, expected_desire):
    self.dh.update(self._car_state(15.0, left=left, right=right), lateral_active=True, lane_change_prob=0.5)

    assert self.dh.desire == expected_desire
    assert self.dh.turn_desire_active
    assert self.dh.lane_change_state == LaneChangeState.off
    assert self.dh.lane_change_direction == LaneChangeDirection.none

  @parameterized.expand([
    ("left", True, False, log.Desire.turnLeft),
    ("right", False, True, log.Desire.turnRight),
  ])
  def test_first_blinker_cycle_enters_pre_lane_change_before_direction_sets(self, _name, left, right, expected_desire):
    self.dh.update(self._car_state(25.0, left=left, right=right), lateral_active=True, lane_change_prob=0.5)

    assert self.dh.desire == expected_desire
    assert self.dh.turn_desire_active
    assert self.dh.lane_change_state == LaneChangeState.preLaneChange
    assert self.dh.lane_change_direction == LaneChangeDirection.none

  @parameterized.expand([
    ("left", True, False, 1.0, LaneChangeDirection.left, log.Desire.turnLeft),
    ("right", False, True, -1.0, LaneChangeDirection.right, log.Desire.turnRight),
  ])
  def test_turn_desire_override_persists_when_lane_change_starts(self, _name, left, right, steering_torque, expected_direction, expected_desire):
    self.dh.update(self._car_state(25.0, left=left, right=right), lateral_active=True, lane_change_prob=0.5)
    self.dh.update(
      self._car_state(25.0, left=left, right=right, steering_pressed=True, steering_torque=steering_torque),
      lateral_active=True,
      lane_change_prob=0.5,
    )

    assert self.dh.desire == expected_desire
    assert self.dh.turn_desire_active
    assert self.dh.lane_change_state == LaneChangeState.laneChangeStarting
    assert self.dh.lane_change_direction == expected_direction

  @parameterized.expand([
    ("just_below_threshold", TURN_DESIRE_SPEED_MAX * CV.MS_TO_MPH - 0.01, True, False, True, log.Desire.turnLeft),
    ("at_threshold", TURN_DESIRE_SPEED_MAX * CV.MS_TO_MPH, True, False, False, log.Desire.none),
    ("hazards", 15.0, True, True, False, log.Desire.none),
    ("lat_inactive", 15.0, True, False, False, log.Desire.none),
  ])
  def test_turn_desire_guard_conditions(self, case_name, speed_mph, left, right, should_be_active, expected_desire):
    lateral_active = case_name != "lat_inactive"
    self.dh.update(self._car_state(speed_mph, left=left, right=right), lateral_active=lateral_active, lane_change_prob=0.5)

    assert self.dh.turn_desire_active is should_be_active
    assert self.dh.desire == expected_desire
