import math

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter


AUTO_TUNE_PERSIST_MIN_DELTA = 0.002
AUTO_TUNE_PERSIST_INTERVAL_S = 5.0
CAMERA_OFFSET_AUTO_PARAM = "CameraOffsetAuto"
CAMERA_OFFSET_AUTO_LEARNED_PARAM = "CameraOffsetAutoLearned"
CAMERA_OFFSET_AUTO_RESET_ACK_PARAM = "CameraOffsetAutoResetAck"
CAMERA_OFFSET_AUTO_RESET_REQUEST_PARAM = "CameraOffsetAutoResetRequest"
CAMERA_OFFSET_AUTO_VERSION_PARAM = "CameraOffsetAutoVersion"
CAMERA_OFFSET_AUTO_VERSION = 1


def camera_offset_auto_enabled(params) -> bool:
  return bool(params.get(CAMERA_OFFSET_AUTO_PARAM, return_default=True))


def should_persist_auto_offset(current_offset: float, last_saved_offset: float,
                               now_monotonic: float, last_save_monotonic: float,
                               min_delta: float = AUTO_TUNE_PERSIST_MIN_DELTA,
                               min_interval: float = AUTO_TUNE_PERSIST_INTERVAL_S) -> bool:
  if not all(math.isfinite(v) for v in (current_offset, last_saved_offset, now_monotonic, last_save_monotonic)):
    return False

  delta = abs(current_offset - last_saved_offset)
  if delta < min_delta:
    return False

  if abs(last_saved_offset) < min_delta and abs(current_offset) >= min_delta:
    return True

  return (now_monotonic - last_save_monotonic) >= min_interval


class CameraOffsetHelper:
  SMOOTH_ALPHA = 0.1
  MAX_TOTAL_OFFSET = 0.35

  AUTO_TUNE_UPDATE_FRAMES = 25
  AUTO_TUNE_INVALID_RESET_FRAMES = 60
  AUTO_TUNE_FILTER_RC = 3.0
  AUTO_TUNE_MIN_SPEED = 20.0
  AUTO_TUNE_SLIGHT_CURVATURE = 5e-4
  AUTO_TUNE_CURVATURE_HYSTERESIS = 1e-4
  AUTO_TUNE_MIN_PROB = 0.6
  AUTO_TUNE_MIN_LANE_WIDTH = 2.8
  AUTO_TUNE_MAX_LANE_WIDTH = 4.8
  AUTO_TUNE_DEADBAND = 0.016
  AUTO_TUNE_FULL_ERROR = 0.14
  AUTO_TUNE_STEP_GAIN = 0.2
  AUTO_TUNE_MAX_STEP = 0.0065
  AUTO_TUNE_MAX_OFFSET = 0.18

  def __init__(self, model_freq: float = 20.0):
    self.camera_offset = 0.0
    self.auto_enabled = True
    self.auto_camera_offset = 0.0
    self.actual_camera_offset = 0.0
    self._valid_frames = 0
    self._invalid_frames = 0
    self._curve_hold_active = False
    self._last_reset_request: str | None = None
    self._center_filter = FirstOrderFilter(0.0, self.AUTO_TUNE_FILTER_RC, 1.0 / model_freq, initialized=False)

  def set_offset(self, offset: float):
    self.camera_offset = float(offset)

  def set_auto_enabled(self, enabled: bool):
    enabled = bool(enabled)
    if self.auto_enabled and not enabled:
      self.reset_runtime_state()
    self.auto_enabled = enabled

  def load_auto_tune_offset(self, offset: float) -> None:
    self.auto_camera_offset = float(np.clip(offset, -self.AUTO_TUNE_MAX_OFFSET, self.AUTO_TUNE_MAX_OFFSET))

  def get_auto_tune_offset(self) -> float:
    return float(self.auto_camera_offset)

  def _reset_observation_window(self) -> None:
    self._valid_frames = 0
    self._invalid_frames = 0
    self._center_filter.initialized = False

  def reset_runtime_state(self) -> None:
    self._reset_observation_window()
    self._curve_hold_active = False

  def reset_auto_tune(self) -> None:
    self.auto_camera_offset = 0.0
    self.reset_runtime_state()

  def _reset_persisted_auto_tune(self, params, reset_request: str | None = None) -> None:
    self.reset_auto_tune()
    # Keep these writes on modeld's Params queue and acknowledge last. FIFO
    # ordering ensures an older queued learned-offset write cannot win the reset.
    params.put_nonblocking(CAMERA_OFFSET_AUTO_LEARNED_PARAM, 0.0)
    params.put_nonblocking(CAMERA_OFFSET_AUTO_VERSION_PARAM, CAMERA_OFFSET_AUTO_VERSION)
    if reset_request is not None:
      self._last_reset_request = reset_request
      params.put_nonblocking(CAMERA_OFFSET_AUTO_RESET_ACK_PARAM, reset_request)

  def load_persisted_auto_tune(self, params) -> float:
    stored_version = params.get(CAMERA_OFFSET_AUTO_VERSION_PARAM, return_default=True)
    reset_request = params.get(CAMERA_OFFSET_AUTO_RESET_REQUEST_PARAM)
    reset_ack = params.get(CAMERA_OFFSET_AUTO_RESET_ACK_PARAM)
    reset_pending = reset_request is not None and reset_request not in (reset_ack, self._last_reset_request)
    if stored_version != CAMERA_OFFSET_AUTO_VERSION or reset_pending:
      self._reset_persisted_auto_tune(params, reset_request)
    else:
      self.load_auto_tune_offset(params.get(CAMERA_OFFSET_AUTO_LEARNED_PARAM, return_default=True))
    return self.get_auto_tune_offset()

  def consume_auto_tune_reset(self, params) -> bool:
    reset_request = params.get(CAMERA_OFFSET_AUTO_RESET_REQUEST_PARAM)
    if reset_request is None or reset_request in (params.get(CAMERA_OFFSET_AUTO_RESET_ACK_PARAM), self._last_reset_request):
      return False
    self._reset_persisted_auto_tune(params, reset_request)
    return True

  @property
  def target_camera_offset(self) -> float:
    auto_offset = self.auto_camera_offset if self.auto_enabled else 0.0
    return float(np.clip(self.camera_offset + auto_offset, -self.MAX_TOTAL_OFFSET, self.MAX_TOTAL_OFFSET))

  def _update_curve_hold_state(self, desired_curvature: float) -> bool:
    if not math.isfinite(desired_curvature):
      self._curve_hold_active = False
      return False

    abs_curvature = abs(desired_curvature)
    hold_enter_curvature = self.AUTO_TUNE_SLIGHT_CURVATURE + self.AUTO_TUNE_CURVATURE_HYSTERESIS
    hold_exit_curvature = max(0.0, self.AUTO_TUNE_SLIGHT_CURVATURE - self.AUTO_TUNE_CURVATURE_HYSTERESIS)

    if self._curve_hold_active:
      self._curve_hold_active = abs_curvature >= hold_exit_curvature
    else:
      self._curve_hold_active = abs_curvature >= hold_enter_curvature

    return self._curve_hold_active

  def observe(self, center_y: float, center_prob: float, lane_width: float,
              center_valid: bool, v_ego: float, lat_active: bool,
              blinkers_active: bool, lane_change_active: bool,
              desired_curvature: float) -> None:
    if not self.auto_enabled:
      return

    curve_hold_active = self._update_curve_hold_state(desired_curvature)
    if curve_hold_active or not lat_active or blinkers_active or lane_change_active:
      # A curve, maneuver, or disengagement changes the observed lane frame.
      # Hold the learned offset and discard center history so the first update
      # after the interruption is based only on fresh straight-road samples.
      self._reset_observation_window()
      return

    valid = (
      center_valid and
      v_ego >= self.AUTO_TUNE_MIN_SPEED and
      math.isfinite(desired_curvature) and
      math.isfinite(center_y) and
      math.isfinite(center_prob) and
      math.isfinite(lane_width) and
      center_prob >= self.AUTO_TUNE_MIN_PROB and
      self.AUTO_TUNE_MIN_LANE_WIDTH <= lane_width <= self.AUTO_TUNE_MAX_LANE_WIDTH
    )
    if not valid:
      self._valid_frames = 0
      self._invalid_frames += 1
      if self._invalid_frames >= self.AUTO_TUNE_INVALID_RESET_FRAMES:
        self._center_filter.initialized = False
      return

    self._valid_frames += 1
    self._invalid_frames = 0
    filtered_center = float(self._center_filter.update(center_y))
    if self._valid_frames % self.AUTO_TUNE_UPDATE_FRAMES != 0:
      return

    abs_error = abs(filtered_center)
    if abs_error <= self.AUTO_TUNE_DEADBAND:
      return

    span = max(self.AUTO_TUNE_FULL_ERROR - self.AUTO_TUNE_DEADBAND, 1e-6)
    normalized = float(np.clip((abs_error - self.AUTO_TUNE_DEADBAND) / span, 0.0, 1.0))
    softness = normalized * normalized * (3.0 - 2.0 * normalized)
    step = min(self.AUTO_TUNE_MAX_STEP, self.AUTO_TUNE_STEP_GAIN * abs_error * softness)
    if step <= 0.0:
      return

    # Positive center_y means the lane center still appears to the right of the vehicle,
    # so apply the opposite camera offset to drive the observed center back toward zero.
    self.auto_camera_offset = float(np.clip(
      self.auto_camera_offset - math.copysign(step, filtered_center),
      -self.AUTO_TUNE_MAX_OFFSET, self.AUTO_TUNE_MAX_OFFSET,
    ))

  def update(self, transform_main: np.ndarray, transform_extra: np.ndarray,
             intrinsics_main: np.ndarray, intrinsics_extra: np.ndarray, height: float) -> tuple[np.ndarray, np.ndarray]:
    target = self.target_camera_offset
    self.actual_camera_offset = (1.0 - self.SMOOTH_ALPHA) * self.actual_camera_offset + self.SMOOTH_ALPHA * target

    if abs(self.actual_camera_offset) < 1e-6:
      return transform_main, transform_extra

    transform_main = self.apply_camera_offset(transform_main, intrinsics_main, height, self.actual_camera_offset)
    transform_extra = self.apply_camera_offset(transform_extra, intrinsics_extra, height, self.actual_camera_offset)
    return transform_main, transform_extra

  @staticmethod
  def apply_camera_offset(transform: np.ndarray, intrinsics: np.ndarray,
                          height: float, offset: float) -> np.ndarray:
    cy = intrinsics[1, 2]
    shear = np.array([
      [1.0, offset / height, -offset / height * cy],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return shear @ transform
