from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature, get_turn_desire_max_lateral_accel_no_roll, \
                                                           TURN_DESIRE_MAX_LATERAL_ACCEL_NO_ROLL, \
                                                           TURN_DESIRE_FULL_AUTHORITY_MAX_LATERAL_ACCEL_NO_ROLL


def test_turn_desire_lateral_accel_taper():
  assert get_turn_desire_max_lateral_accel_no_roll(5.0 * CV.MPH_TO_MS) == TURN_DESIRE_FULL_AUTHORITY_MAX_LATERAL_ACCEL_NO_ROLL
  assert get_turn_desire_max_lateral_accel_no_roll(25.0 * CV.MPH_TO_MS) == TURN_DESIRE_FULL_AUTHORITY_MAX_LATERAL_ACCEL_NO_ROLL
  assert get_turn_desire_max_lateral_accel_no_roll(35.0 * CV.MPH_TO_MS) == TURN_DESIRE_MAX_LATERAL_ACCEL_NO_ROLL

  tapered_limit = get_turn_desire_max_lateral_accel_no_roll(30.0 * CV.MPH_TO_MS)
  assert TURN_DESIRE_MAX_LATERAL_ACCEL_NO_ROLL < tapered_limit < TURN_DESIRE_FULL_AUTHORITY_MAX_LATERAL_ACCEL_NO_ROLL


def test_turn_desire_relaxes_curvature_clip_at_20_mph():
  v_ego = 20.0 * CV.MPH_TO_MS
  requested_curvature = 0.09

  base_curvature, base_limited = clip_curvature(v_ego, requested_curvature, requested_curvature, 0.0)
  turn_curvature, turn_limited = clip_curvature(v_ego, requested_curvature, requested_curvature, 0.0,
                                                max_lateral_accel_no_roll=get_turn_desire_max_lateral_accel_no_roll(v_ego))

  assert base_limited
  assert turn_limited
  assert turn_curvature > base_curvature


def test_turn_desire_relaxes_curvature_clip_at_30_mph():
  v_ego = 30.0 * CV.MPH_TO_MS
  requested_curvature = 0.03

  base_curvature, base_limited = clip_curvature(v_ego, requested_curvature, requested_curvature, 0.0)
  turn_curvature, turn_limited = clip_curvature(v_ego, requested_curvature, requested_curvature, 0.0,
                                                max_lateral_accel_no_roll=get_turn_desire_max_lateral_accel_no_roll(v_ego))

  assert base_limited
  assert turn_limited
  assert turn_curvature > base_curvature
