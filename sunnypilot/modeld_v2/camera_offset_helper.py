import numpy as np
from openpilot.common.transformations.camera import DEVICE_CAMERAS


class CameraOffsetHelper:
  SMOOTH_ALPHA = 0.1  # exponential smoothing factor (0.9 old + 0.1 new)

  def __init__(self):
    self.camera_offset = 0.0
    self.actual_camera_offset = 0.0

  def set_offset(self, offset: float):
    self.camera_offset = offset

  def update(self, transform_main: np.ndarray, transform_extra: np.ndarray,
             sm, main_wide_camera: bool) -> tuple[np.ndarray, np.ndarray]:
    """Apply smoothed camera offset shear to both transforms."""
    self.actual_camera_offset = (1.0 - self.SMOOTH_ALPHA) * self.actual_camera_offset + self.SMOOTH_ALPHA * self.camera_offset

    if self.actual_camera_offset == 0.0:
      return transform_main, transform_extra

    dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
    height = sm['liveCalibration'].height[0] if sm['liveCalibration'].height else 1.22

    transform_main = self.apply_camera_offset(
      transform_main, dc.fcam.intrinsics if not main_wide_camera else dc.ecam.intrinsics,
      height, self.actual_camera_offset)
    transform_extra = self.apply_camera_offset(
      transform_extra, dc.ecam.intrinsics, height, self.actual_camera_offset)

    return transform_main, transform_extra

  @staticmethod
  def apply_camera_offset(transform: np.ndarray, intrinsics: np.ndarray,
                          height: float, offset: float) -> np.ndarray:
    """Apply a horizontal shear to the warp matrix to shift the camera perspective."""
    # Shear matrix: shifts the image horizontally proportional to offset/height,
    # compensating for the camera intrinsics' principal point (cy).
    cy = intrinsics[1, 2]
    shear = np.array([
      [1.0, offset / height, -offset / height * cy],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return shear @ transform
