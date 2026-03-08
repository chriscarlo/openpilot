import numpy as np
import pytest
from unittest.mock import MagicMock

from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper


@pytest.fixture
def helper():
  return CameraOffsetHelper()


@pytest.fixture
def mock_sm():
  sm = MagicMock()
  sm.__getitem__ = MagicMock(side_effect=lambda key: {
    'deviceState': MagicMock(deviceType='tici'),
    'roadCameraState': MagicMock(sensor='ar0231'),
    'liveCalibration': MagicMock(height=[1.22]),
  }[key])
  return sm


class TestSmoothing:
  def test_smoothing_converges(self, helper, mock_sm):
    """Offset should exponentially converge toward target."""
    helper.set_offset(0.1)
    transform = np.eye(3, dtype=np.float32)

    for _ in range(100):
      helper.update(transform.copy(), transform.copy(), mock_sm, False)

    assert abs(helper.actual_camera_offset - 0.1) < 1e-6

  def test_smoothing_step(self, helper, mock_sm):
    """Single step should apply 0.1 of the target (SMOOTH_ALPHA)."""
    helper.set_offset(1.0)
    transform = np.eye(3, dtype=np.float32)

    helper.update(transform.copy(), transform.copy(), mock_sm, False)

    assert abs(helper.actual_camera_offset - 0.1) < 1e-6

  def test_smoothing_two_steps(self, helper, mock_sm):
    """After two steps: 0.9*0.1 + 0.1*1.0 = 0.19."""
    helper.set_offset(1.0)
    transform = np.eye(3, dtype=np.float32)

    helper.update(transform.copy(), transform.copy(), mock_sm, False)
    helper.update(transform.copy(), transform.copy(), mock_sm, False)

    assert abs(helper.actual_camera_offset - 0.19) < 1e-6


class TestShearMatrix:
  def test_zero_offset_no_change(self, helper, mock_sm):
    """Zero offset should return transforms unchanged."""
    transform = np.eye(3, dtype=np.float32)
    main, extra = helper.update(transform.copy(), transform.copy(), mock_sm, False)

    np.testing.assert_array_equal(main, transform)
    np.testing.assert_array_equal(extra, transform)

  def test_nonzero_offset_modifies_transform(self, helper, mock_sm):
    """Non-zero offset should modify the transforms."""
    helper.set_offset(0.1)
    # Force actual offset to target immediately
    helper.actual_camera_offset = 0.1

    transform = np.eye(3, dtype=np.float32)
    main, extra = helper.update(transform.copy(), transform.copy(), mock_sm, False)

    # The transform should be modified
    assert not np.array_equal(main, transform)
    assert not np.array_equal(extra, transform)

  def test_shear_matrix_values(self):
    """Verify shear matrix has expected structure."""
    from openpilot.common.transformations.camera import DEVICE_CAMERAS
    dc = DEVICE_CAMERAS[('tici', 'ar0231')]
    intrinsics = dc.fcam.intrinsics
    cy = intrinsics[1, 2]
    height = 1.22
    offset = 0.1

    result = CameraOffsetHelper.apply_camera_offset(np.eye(3, dtype=np.float32), intrinsics, height, offset)

    # Check the shear element
    expected_shear = offset / height
    assert abs(result[1, 0] - expected_shear) < 1e-6
    # Check the translation element
    expected_translation = cy * offset / height
    assert abs(result[1, 2] - expected_translation) < 1e-4

  def test_apply_to_identity(self):
    """Applying shear to identity should produce the shear matrix itself."""
    from openpilot.common.transformations.camera import DEVICE_CAMERAS
    dc = DEVICE_CAMERAS[('tici', 'ar0231')]
    intrinsics = dc.fcam.intrinsics
    height = 1.22
    offset = 0.2

    result = CameraOffsetHelper.apply_camera_offset(np.eye(3, dtype=np.float32), intrinsics, height, offset)

    cy = intrinsics[1, 2]
    assert abs(result[0, 0] - 1.0) < 1e-6
    assert abs(result[1, 1] - 1.0) < 1e-6
    assert abs(result[2, 2] - 1.0) < 1e-6
    assert abs(result[1, 0] - offset / height) < 1e-6
    assert abs(result[1, 2] - cy * offset / height) < 1e-4
