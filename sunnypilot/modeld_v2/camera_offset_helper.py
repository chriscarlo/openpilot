import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.modeld.camera_offset_helper import CameraOffsetHelper as SharedCameraOffsetHelper


class CameraOffsetHelper(SharedCameraOffsetHelper):
  def update(self, transform_main: np.ndarray, transform_extra: np.ndarray,
             sm, main_wide_camera: bool) -> tuple[np.ndarray, np.ndarray]:
    dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
    height = sm['liveCalibration'].height[0] if sm['liveCalibration'].height else 1.22
    intrinsics_main = dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics
    intrinsics_extra = dc.ecam.intrinsics
    return super().update(transform_main, transform_extra, intrinsics_main, intrinsics_extra, height)
