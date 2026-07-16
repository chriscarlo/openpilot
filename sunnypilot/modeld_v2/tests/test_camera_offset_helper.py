import numpy as np
import pytest
from types import SimpleNamespace

from openpilot.selfdrive.modeld.camera_offset_helper import CameraOffsetHelper
from openpilot.selfdrive.modeld.lane_line_meta import estimate_lane_center


INTRINSICS = np.array([
  [2648.0, 0.0, 1928.0 / 2.0],
  [0.0, 2648.0, 1208.0 / 2.0],
  [0.0, 0.0, 1.0],
], dtype=np.float32)
HEIGHT = 1.22


@pytest.fixture
def helper():
  return CameraOffsetHelper(20.0)


class TestSmoothing:
  def test_smoothing_converges(self, helper):
    """Offset should exponentially converge toward target."""
    helper.set_offset(0.1)
    transform = np.eye(3, dtype=np.float32)

    for _ in range(100):
      helper.update(transform.copy(), transform.copy(), INTRINSICS, INTRINSICS, HEIGHT)

    assert abs(helper.actual_camera_offset - 0.1) < 1e-5

  def test_smoothing_step(self, helper):
    """Single step should apply SMOOTH_ALPHA of the target."""
    helper.set_offset(1.0)
    transform = np.eye(3, dtype=np.float32)

    helper.update(transform.copy(), transform.copy(), INTRINSICS, INTRINSICS, HEIGHT)

    expected = helper.SMOOTH_ALPHA * helper.MAX_TOTAL_OFFSET
    assert abs(helper.actual_camera_offset - expected) < 1e-6

  def test_smoothing_two_steps(self, helper):
    """After two steps, the filter should follow the standard EMA recurrence."""
    helper.set_offset(1.0)
    transform = np.eye(3, dtype=np.float32)

    helper.update(transform.copy(), transform.copy(), INTRINSICS, INTRINSICS, HEIGHT)
    helper.update(transform.copy(), transform.copy(), INTRINSICS, INTRINSICS, HEIGHT)

    alpha = helper.SMOOTH_ALPHA
    target = helper.MAX_TOTAL_OFFSET
    expected = (1.0 - alpha) * (alpha * target) + alpha * target
    assert abs(helper.actual_camera_offset - expected) < 1e-6


class TestShearMatrix:
  def test_zero_offset_no_change(self, helper):
    """Zero offset should return transforms unchanged."""
    transform = np.eye(3, dtype=np.float32)
    main, extra = helper.update(transform.copy(), transform.copy(), INTRINSICS, INTRINSICS, HEIGHT)

    np.testing.assert_array_equal(main, transform)
    np.testing.assert_array_equal(extra, transform)

  def test_nonzero_offset_modifies_transform(self, helper):
    """Non-zero offset should modify the transforms."""
    helper.set_offset(0.1)
    # Force actual offset to target immediately
    helper.actual_camera_offset = 0.1

    transform = np.eye(3, dtype=np.float32)
    main, extra = helper.update(transform.copy(), transform.copy(), INTRINSICS, INTRINSICS, HEIGHT)

    # The transform should be modified
    assert not np.array_equal(main, transform)
    assert not np.array_equal(extra, transform)

  def test_shear_matrix_values(self):
    """Verify shear matrix has correct horizontal shear structure (row 0)."""
    offset = 0.1

    result = CameraOffsetHelper.apply_camera_offset(np.eye(3, dtype=np.float32), INTRINSICS, HEIGHT, offset)

    # Horizontal shear: offset lives in row 0, col 1
    expected_shear = offset / HEIGHT
    assert abs(result[0, 1] - expected_shear) < 1e-6
    # Translation with negative sign to keep principal point centered
    expected_translation = -offset / HEIGHT * INTRINSICS[1, 2]
    assert abs(result[0, 2] - expected_translation) < 1e-4
    # Row 1 must be untouched (no vertical shear)
    assert abs(result[1, 0]) < 1e-6
    assert abs(result[1, 2]) < 1e-6

  def test_apply_to_identity(self):
    """Applying shear to identity should produce the shear matrix itself."""
    offset = 0.2

    result = CameraOffsetHelper.apply_camera_offset(np.eye(3, dtype=np.float32), INTRINSICS, HEIGHT, offset)

    # Diagonal is identity
    assert abs(result[0, 0] - 1.0) < 1e-6
    assert abs(result[1, 1] - 1.0) < 1e-6
    assert abs(result[2, 2] - 1.0) < 1e-6
    # Horizontal shear in row 0
    assert abs(result[0, 1] - offset / HEIGHT) < 1e-6
    assert abs(result[0, 2] - (-offset / HEIGHT * INTRINSICS[1, 2])) < 1e-4
    # Row 1 untouched
    assert abs(result[1, 0]) < 1e-6
    assert abs(result[1, 2]) < 1e-6

  def test_negative_offset_reverses_shear(self):
    """Negative offset should produce opposite shear direction."""
    pos = CameraOffsetHelper.apply_camera_offset(np.eye(3, dtype=np.float32), INTRINSICS, HEIGHT, 0.1)
    neg = CameraOffsetHelper.apply_camera_offset(np.eye(3, dtype=np.float32), INTRINSICS, HEIGHT, -0.1)

    # Shear elements should be equal magnitude, opposite sign
    assert abs(pos[0, 1] + neg[0, 1]) < 1e-6
    assert abs(pos[0, 2] + neg[0, 2]) < 1e-4


def _make_lane_lines(center_y: float, lane_width: float, xs: np.ndarray):
  half_width = lane_width / 2.0
  left = np.full_like(xs, center_y - half_width, dtype=np.float32)
  right = np.full_like(xs, center_y + half_width, dtype=np.float32)
  return [
    SimpleNamespace(x=xs, y=left - lane_width, z=np.zeros_like(xs)),
    SimpleNamespace(x=xs, y=left, z=np.zeros_like(xs)),
    SimpleNamespace(x=xs, y=right, z=np.zeros_like(xs)),
    SimpleNamespace(x=xs, y=right + lane_width, z=np.zeros_like(xs)),
  ]


class TestLaneCenterEstimate:
  def test_estimate_lane_center_uses_both_markers(self):
    xs = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float32)
    lane_lines = _make_lane_lines(center_y=0.18, lane_width=3.65, xs=xs)

    estimate = estimate_lane_center(lane_lines, [0.1, 0.95, 0.93, 0.1])

    assert estimate.center_valid
    assert estimate.center_prob > 0.7
    assert estimate.center_y == pytest.approx(0.18, abs=1e-3)
    assert estimate.lane_width == pytest.approx(3.65, abs=1e-3)

  def test_estimate_lane_center_rejects_implausible_width(self):
    xs = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float32)
    lane_lines = _make_lane_lines(center_y=0.0, lane_width=5.6, xs=xs)

    estimate = estimate_lane_center(lane_lines, [0.1, 0.95, 0.93, 0.1])

    assert not estimate.center_valid
    assert estimate.center_prob == pytest.approx(0.0)


class TestAutoTune:
  def test_auto_tune_ignores_curved_segments(self, helper):
    hold_curve = helper.AUTO_TUNE_SLIGHT_CURVATURE + helper.AUTO_TUNE_CURVATURE_HYSTERESIS + 1e-4
    for _ in range(200):
      helper.observe(
        center_y=0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=hold_curve,
      )

    assert helper.auto_camera_offset == pytest.approx(0.0)

  def test_auto_tune_curve_hold_preserves_learned_offset(self, helper):
    hold_curve = helper.AUTO_TUNE_SLIGHT_CURVATURE + helper.AUTO_TUNE_CURVATURE_HYSTERESIS + 1e-4

    for _ in range(helper.AUTO_TUNE_UPDATE_FRAMES):
      helper.observe(
        center_y=0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=0.0,
      )

    learned_offset = helper.auto_camera_offset
    assert learned_offset < 0.0

    helper.observe(
      center_y=0.18,
      center_prob=0.95,
      lane_width=3.65,
      center_valid=True,
      v_ego=33.5,
      lat_active=True,
      blinkers_active=False,
      lane_change_active=False,
      desired_curvature=hold_curve,
    )

    for _ in range(200):
      helper.observe(
        center_y=0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=helper.AUTO_TUNE_SLIGHT_CURVATURE,
      )

    assert helper.auto_camera_offset == pytest.approx(learned_offset)

  def test_auto_tune_requires_fresh_straight_after_curve_hold(self, helper):
    hold_curve = helper.AUTO_TUNE_SLIGHT_CURVATURE + helper.AUTO_TUNE_CURVATURE_HYSTERESIS + 1e-4

    for _ in range(helper.AUTO_TUNE_UPDATE_FRAMES - 1):
      helper.observe(
        center_y=0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=0.0,
      )

    helper.observe(
      center_y=0.18,
      center_prob=0.95,
      lane_width=3.65,
      center_valid=True,
      v_ego=33.5,
      lat_active=True,
      blinkers_active=False,
      lane_change_active=False,
      desired_curvature=hold_curve,
    )

    helper.observe(
      center_y=0.18,
      center_prob=0.95,
      lane_width=3.65,
      center_valid=True,
      v_ego=33.5,
      lat_active=True,
      blinkers_active=False,
      lane_change_active=False,
      desired_curvature=0.0,
    )

    assert helper.auto_camera_offset == pytest.approx(0.0)

    for _ in range(helper.AUTO_TUNE_UPDATE_FRAMES - 1):
      helper.observe(
        center_y=0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=0.0,
      )

    assert helper.auto_camera_offset < 0.0

  def test_auto_tune_post_curve_uses_fresh_center_sign(self, helper):
    hold_curve = helper.AUTO_TUNE_SLIGHT_CURVATURE + helper.AUTO_TUNE_CURVATURE_HYSTERESIS + 1e-4

    for _ in range(helper.AUTO_TUNE_UPDATE_FRAMES):
      helper.observe(
        center_y=0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=0.0,
      )

    learned_before_curve = helper.auto_camera_offset
    assert learned_before_curve < 0.0

    for _ in range(200):
      helper.observe(
        center_y=-0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=hold_curve,
      )

    for _ in range(helper.AUTO_TUNE_UPDATE_FRAMES):
      helper.observe(
        center_y=-0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=0.0,
      )

    assert helper.auto_camera_offset > learned_before_curve

  @pytest.mark.parametrize(("blinkers_active", "lane_change_active"), [(True, False), (False, True)])
  def test_auto_tune_maneuver_requires_fresh_post_maneuver_window(self, helper, blinkers_active, lane_change_active):
    for _ in range(helper.AUTO_TUNE_UPDATE_FRAMES):
      helper.observe(
        center_y=0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=0.0,
      )

    learned_before_lane_change = helper.auto_camera_offset
    assert learned_before_lane_change < 0.0

    for _ in range(helper.AUTO_TUNE_UPDATE_FRAMES * 2):
      helper.observe(
        center_y=-0.8,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=blinkers_active,
        lane_change_active=lane_change_active,
        desired_curvature=0.0,
      )

    assert helper.auto_camera_offset == pytest.approx(learned_before_lane_change)

    for _ in range(helper.AUTO_TUNE_UPDATE_FRAMES - 1):
      helper.observe(
        center_y=-0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=0.0,
      )

    assert helper.auto_camera_offset == pytest.approx(learned_before_lane_change)

    helper.observe(
      center_y=-0.18,
      center_prob=0.95,
      lane_width=3.65,
      center_valid=True,
      v_ego=33.5,
      lat_active=True,
      blinkers_active=False,
      lane_change_active=False,
      desired_curvature=0.0,
    )

    assert helper.auto_camera_offset > learned_before_lane_change

  def test_curve_hold_hysteresis_survives_lane_change(self, helper):
    hold_enter = helper.AUTO_TUNE_SLIGHT_CURVATURE + helper.AUTO_TUNE_CURVATURE_HYSTERESIS + 1e-4
    hold_band = helper.AUTO_TUNE_SLIGHT_CURVATURE

    helper.observe(
      center_y=0.18,
      center_prob=0.95,
      lane_width=3.65,
      center_valid=True,
      v_ego=33.5,
      lat_active=True,
      blinkers_active=False,
      lane_change_active=False,
      desired_curvature=hold_enter,
    )
    helper.observe(
      center_y=0.18,
      center_prob=0.95,
      lane_width=3.65,
      center_valid=True,
      v_ego=33.5,
      lat_active=True,
      blinkers_active=False,
      lane_change_active=True,
      desired_curvature=hold_band,
    )

    for _ in range(helper.AUTO_TUNE_UPDATE_FRAMES):
      helper.observe(
        center_y=0.18,
        center_prob=0.95,
        lane_width=3.65,
        center_valid=True,
        v_ego=33.5,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=hold_band,
      )

    assert helper.auto_camera_offset == pytest.approx(0.0)

  def test_auto_tune_straight_highway_settles_without_oscillation(self, helper):
    xs = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float32)
    transform = np.eye(3, dtype=np.float32)
    lane_width = 3.65
    static_bias = 0.18

    measured_centers = []
    target_offsets = []

    for _ in range(45 * 20):
      measured_center = static_bias + helper.actual_camera_offset
      lane_lines = _make_lane_lines(measured_center, lane_width, xs)
      estimate = estimate_lane_center(lane_lines, [0.1, 0.97, 0.96, 0.1])
      measured_centers.append(estimate.center_y)
      target_offsets.append(helper.target_camera_offset)

      helper.observe(
        center_y=estimate.center_y,
        center_prob=estimate.center_prob,
        lane_width=estimate.lane_width,
        center_valid=estimate.center_valid,
        v_ego=75.0 * 0.44704,
        lat_active=True,
        blinkers_active=False,
        lane_change_active=False,
        desired_curvature=0.0,
      )
      helper.update(transform.copy(), transform.copy(), INTRINSICS, INTRINSICS, HEIGHT)

    post_settle = measured_centers[20 * 20:]
    assert abs(target_offsets[8 * 20 - 1]) < 0.04
    assert abs(post_settle[-1]) < 0.035

    sign_flips = 0
    prev_sign = None
    for value in post_settle:
      if abs(value) < 0.01:
        continue
      sign = 1 if value > 0 else -1
      if prev_sign is not None and sign != prev_sign:
        sign_flips += 1
      prev_sign = sign

    assert sign_flips <= 1
    assert helper.auto_camera_offset < 0.0
    assert target_offsets[-1] < 0.0
