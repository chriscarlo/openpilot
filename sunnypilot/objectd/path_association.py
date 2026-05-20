from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from openpilot.sunnypilot.objectd.types import Detection

DEFAULT_CAMERA_HEIGHT_M = 1.22
DEFAULT_PATH_WIDTH_M = 1.8
DEFAULT_ACTIVE_CONFIDENCE_THRESHOLD = 0.35
MIN_LOOKAHEAD_M = 3.0
MAX_LOOKAHEAD_M = 40.0


@dataclass(frozen=True)
class ProjectedPath:
  center_px: np.ndarray
  half_width_px: np.ndarray
  forward_m: np.ndarray


class HazardTracker:
  def __init__(self, activate_frames: int = 2, clear_frames: int = 2):
    self.activate_frames = activate_frames
    self.clear_frames = clear_frames
    self.pending_count = 0
    self.missed_count = 0
    self.active_detection: Detection | None = None

  def update(self, detection: Detection | None) -> Detection | None:
    if detection is None:
      self.pending_count = 0
      if self.active_detection is not None:
        self.missed_count += 1
        if self.missed_count >= self.clear_frames:
          self.active_detection = None
      return self.active_detection

    self.missed_count = 0
    self.pending_count += 1
    if self.active_detection is None:
      if self.pending_count >= self.activate_frames:
        self.active_detection = detection
    else:
      self.active_detection = detection
    return self.active_detection


def build_road_calibration_transform(device_type: str, sensor: str, rpy_calib: Iterable[float]) -> np.ndarray:
  from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
  from openpilot.common.transformations.orientation import rot_from_euler

  device_from_calib = rot_from_euler(np.asarray(tuple(rpy_calib), dtype=np.float32))
  intrinsics = DEVICE_CAMERAS[(device_type, sensor)].fcam.intrinsics
  return intrinsics @ view_frame_from_device_frame @ device_from_calib


def project_model_path(model_position, calib_transform: np.ndarray, *, camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
                       path_width_m: float = DEFAULT_PATH_WIDTH_M,
                       min_lookahead_m: float = MIN_LOOKAHEAD_M,
                       max_lookahead_m: float = MAX_LOOKAHEAD_M) -> ProjectedPath:
  points = np.column_stack([
    np.asarray(model_position.x, dtype=np.float32),
    np.asarray(model_position.y, dtype=np.float32),
    np.asarray(model_position.z, dtype=np.float32),
  ])

  if points.size == 0:
    return ProjectedPath(np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32))

  valid = (
    np.isfinite(points).all(axis=1) &
    (points[:, 0] >= min_lookahead_m) &
    (points[:, 0] <= max_lookahead_m)
  )
  points = points[valid]
  if points.shape[0] == 0:
    return ProjectedPath(np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32))

  half_width_m = path_width_m / 2.0
  offsets = np.array([
    [0.0, 0.0, camera_height_m],
    [0.0, -half_width_m, camera_height_m],
    [0.0, half_width_m, camera_height_m],
  ], dtype=np.float32)
  points_3d = (points[None, :, :] + offsets[:, None, :]).reshape(3 * len(points), 3)
  proj = calib_transform @ points_3d.T
  proj = proj.reshape(3, 3, len(points))

  center = proj[:, 0, :]
  left = proj[:, 1, :]
  right = proj[:, 2, :]

  valid_proj = (
    (np.abs(center[2]) >= 1e-6) &
    (np.abs(left[2]) >= 1e-6) &
    (np.abs(right[2]) >= 1e-6) &
    np.isfinite(center).all(axis=0) &
    np.isfinite(left).all(axis=0) &
    np.isfinite(right).all(axis=0)
  )
  valid_proj &= (center[2] > 0.0) & (left[2] > 0.0) & (right[2] > 0.0)
  if not np.any(valid_proj):
    return ProjectedPath(np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32))

  center_screen = (center[:2, valid_proj] / center[2, valid_proj][None, :]).T.astype(np.float32)
  left_screen = (left[:2, valid_proj] / left[2, valid_proj][None, :]).T.astype(np.float32)
  right_screen = (right[:2, valid_proj] / right[2, valid_proj][None, :]).T.astype(np.float32)
  half_width_px = (0.5 * np.linalg.norm(left_screen - right_screen, axis=1)).astype(np.float32)
  forward_m = points[valid_proj, 0].astype(np.float32)

  valid_width = np.isfinite(half_width_px) & (half_width_px > 0.0)
  return ProjectedPath(center_screen[valid_width], half_width_px[valid_width], forward_m[valid_width])


def associate_detections(detections: list[Detection], projected_path: ProjectedPath, *,
                         active_confidence_threshold: float = DEFAULT_ACTIVE_CONFIDENCE_THRESHOLD) -> tuple[list[Detection], Detection | None]:
  if projected_path.center_px.shape[0] == 0:
    return detections, None

  associated: list[Detection] = []
  active_detection: Detection | None = None
  nearest_distance = float("inf")
  center = projected_path.center_px
  widths = projected_path.half_width_px
  forward = projected_path.forward_m

  for detection in detections:
    footpoint_x = 0.5 * (detection.x_min + detection.x_max)
    footpoint_y = detection.y_max
    footpoint = np.array([footpoint_x, footpoint_y], dtype=np.float32)
    deltas = center - footpoint[None, :]
    distances_px = np.linalg.norm(deltas, axis=1)
    best_idx = int(np.argmin(distances_px))
    on_path = bool(distances_px[best_idx] <= widths[best_idx])
    distance_m = float(forward[best_idx]) if on_path else float("inf")
    updated = detection.with_path_state(
      on_path=on_path,
      distance_m=distance_m,
      footpoint_x=float(footpoint_x),
      footpoint_y=float(footpoint_y),
    )
    associated.append(updated)
    if on_path and detection.confidence >= active_confidence_threshold and distance_m < nearest_distance:
      nearest_distance = distance_m
      active_detection = updated

  return associated, active_detection
