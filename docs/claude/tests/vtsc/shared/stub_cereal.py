"""Stub cereal module for VTSC testing"""

class MockVisionTurnSpeedControlState:
    disabled = 0
    enabled = 1
    entering = 2
    turning = 3
    leaving = 4
    active = 5
    activeButUnavailable = 6

class MockCustom:
    class LongitudinalPlanSP:
        class VisionTurnSpeedControl:
            VisionTurnSpeedControlState = MockVisionTurnSpeedControlState

custom = MockCustom()