import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.modeld.camera_offset_helper import (
  CameraOffsetHelper,
  camera_offset_auto_enabled,
  should_persist_auto_offset,
)


class TestCameraOffsetHelper:
  def test_target_offset_uses_manual_and_loaded_auto_offset(self):
    helper = CameraOffsetHelper()
    helper.set_offset(0.04)
    helper.load_auto_tune_offset(0.05)

    assert helper.get_auto_tune_offset() == pytest.approx(0.05)
    assert helper.target_camera_offset == pytest.approx(0.09)

  def test_disable_enable_preserves_loaded_auto_offset(self):
    helper = CameraOffsetHelper()
    helper.set_offset(0.02)
    helper.load_auto_tune_offset(0.07)

    helper.set_auto_enabled(False)
    assert helper.get_auto_tune_offset() == pytest.approx(0.07)
    assert helper.target_camera_offset == pytest.approx(0.02)

    helper.set_auto_enabled(True)
    assert helper.target_camera_offset == pytest.approx(0.09)

  def test_reset_auto_tune_clears_learned_offset(self):
    helper = CameraOffsetHelper()
    helper.load_auto_tune_offset(0.06)

    helper.reset_auto_tune()

    assert helper.get_auto_tune_offset() == pytest.approx(0.0)
    assert helper.target_camera_offset == pytest.approx(0.0)

  def test_loaded_offset_is_clipped(self):
    helper = CameraOffsetHelper()
    helper.load_auto_tune_offset(1.0)

    assert helper.get_auto_tune_offset() == pytest.approx(helper.AUTO_TUNE_MAX_OFFSET)


def test_should_persist_first_crossing_away_from_zero():
  assert should_persist_auto_offset(0.003, 0.0, 1.0, 0.0)


def test_should_not_persist_subthreshold_delta():
  assert not should_persist_auto_offset(0.001, 0.0, 10.0, 0.0)


def test_should_wait_for_debounce_window_after_initial_save():
  assert not should_persist_auto_offset(0.010, 0.005, 4.9, 0.0)
  assert should_persist_auto_offset(0.010, 0.005, 5.0, 0.0)


def test_camera_offset_auto_enabled_uses_param_default_when_unset():
  params = Params()
  params.remove("CameraOffsetAuto")

  assert camera_offset_auto_enabled(params)


def test_camera_offset_auto_enabled_allows_explicit_disable():
  params = Params()
  params.put_bool("CameraOffsetAuto", False)

  assert not camera_offset_auto_enabled(params)
