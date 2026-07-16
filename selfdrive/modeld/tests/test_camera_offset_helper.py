import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.modeld.camera_offset_helper import (
  CAMERA_OFFSET_AUTO_LEARNED_PARAM,
  CAMERA_OFFSET_AUTO_RESET_ACK_PARAM,
  CAMERA_OFFSET_AUTO_RESET_REQUEST_PARAM,
  CAMERA_OFFSET_AUTO_VERSION,
  CAMERA_OFFSET_AUTO_VERSION_PARAM,
  CameraOffsetHelper,
  camera_offset_auto_enabled,
  should_persist_auto_offset,
)


class MemoryParams:
  DEFAULTS = {
    CAMERA_OFFSET_AUTO_LEARNED_PARAM: 0.0,
    CAMERA_OFFSET_AUTO_VERSION_PARAM: 0,
  }

  def __init__(self, values=None, apply_nonblocking=True):
    self.values = dict(values or {})
    self.apply_nonblocking = apply_nonblocking
    self.queued = []

  def get(self, key, return_default=False):
    if key in self.values:
      return self.values[key]
    return self.DEFAULTS.get(key) if return_default else None

  def put_nonblocking(self, key, value):
    self.queued.append((key, value))
    if self.apply_nonblocking:
      self.values[key] = value


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

  def test_load_persisted_auto_tune_keeps_current_state(self):
    params = MemoryParams({
      CAMERA_OFFSET_AUTO_VERSION_PARAM: CAMERA_OFFSET_AUTO_VERSION,
      CAMERA_OFFSET_AUTO_LEARNED_PARAM: 0.06,
    })
    helper = CameraOffsetHelper()

    assert helper.load_persisted_auto_tune(params) == pytest.approx(0.06)

  def test_load_persisted_auto_tune_resets_previous_algorithm_version(self):
    params = MemoryParams({
      CAMERA_OFFSET_AUTO_VERSION_PARAM: CAMERA_OFFSET_AUTO_VERSION - 1,
      CAMERA_OFFSET_AUTO_LEARNED_PARAM: -0.07,
    })
    helper = CameraOffsetHelper()

    assert helper.load_persisted_auto_tune(params) == pytest.approx(0.0)
    assert params.get(CAMERA_OFFSET_AUTO_LEARNED_PARAM) == pytest.approx(0.0)
    assert params.get(CAMERA_OFFSET_AUTO_VERSION_PARAM) == CAMERA_OFFSET_AUTO_VERSION

  def test_load_persisted_auto_tune_consumes_pending_reset(self):
    params = MemoryParams({
      CAMERA_OFFSET_AUTO_VERSION_PARAM: CAMERA_OFFSET_AUTO_VERSION,
      CAMERA_OFFSET_AUTO_LEARNED_PARAM: 0.06,
      CAMERA_OFFSET_AUTO_RESET_REQUEST_PARAM: "reset-1",
    })
    helper = CameraOffsetHelper()

    assert helper.load_persisted_auto_tune(params) == pytest.approx(0.0)
    assert params.get(CAMERA_OFFSET_AUTO_LEARNED_PARAM) == pytest.approx(0.0)
    assert params.get(CAMERA_OFFSET_AUTO_RESET_ACK_PARAM) == "reset-1"

  def test_load_persisted_auto_tune_ignores_acknowledged_reset(self):
    params = MemoryParams({
      CAMERA_OFFSET_AUTO_VERSION_PARAM: CAMERA_OFFSET_AUTO_VERSION,
      CAMERA_OFFSET_AUTO_LEARNED_PARAM: 0.06,
      CAMERA_OFFSET_AUTO_RESET_REQUEST_PARAM: "reset-1",
      CAMERA_OFFSET_AUTO_RESET_ACK_PARAM: "reset-1",
    })
    helper = CameraOffsetHelper()

    assert helper.load_persisted_auto_tune(params) == pytest.approx(0.06)

  def test_consume_auto_tune_reset_queues_zero_before_ack_and_is_idempotent(self):
    params = MemoryParams({
      CAMERA_OFFSET_AUTO_RESET_REQUEST_PARAM: "reset-2",
      CAMERA_OFFSET_AUTO_RESET_ACK_PARAM: "reset-1",
    }, apply_nonblocking=False)
    helper = CameraOffsetHelper()
    helper.load_auto_tune_offset(0.06)

    assert helper.consume_auto_tune_reset(params)
    assert helper.get_auto_tune_offset() == pytest.approx(0.0)
    assert params.queued == [
      (CAMERA_OFFSET_AUTO_LEARNED_PARAM, 0.0),
      (CAMERA_OFFSET_AUTO_VERSION_PARAM, CAMERA_OFFSET_AUTO_VERSION),
      (CAMERA_OFFSET_AUTO_RESET_ACK_PARAM, "reset-2"),
    ]
    assert not helper.consume_auto_tune_reset(params)
    assert len(params.queued) == 3


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
