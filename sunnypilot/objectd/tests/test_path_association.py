from __future__ import annotations

import math

import numpy as np

from openpilot.sunnypilot.objectd.path_association import (
  DEFAULT_PATH_WIDTH_M,
  HazardTracker,
  ProjectedPath,
  associate_detections,
)
from openpilot.sunnypilot.objectd.types import Detection


def test_associate_detection_marks_centerline_box_on_path():
  projected = ProjectedPath(
    center_px=np.array([[100.0, 420.0], [100.0, 320.0], [100.0, 220.0]], dtype=np.float32),
    half_width_px=np.array([24.0, 20.0, 16.0], dtype=np.float32),
    forward_m=np.array([5.0, 15.0, 25.0], dtype=np.float32),
  )
  detections = [Detection("person", 0.9, 90.0, 280.0, 110.0, 320.0)]

  associated, active = associate_detections(detections, projected)

  assert len(associated) == 1
  assert associated[0].on_path is True
  assert math.isclose(associated[0].distance_m, 15.0)
  assert active is not None
  assert active.class_name == "person"


def test_associate_detection_rejects_box_outside_corridor():
  projected = ProjectedPath(
    center_px=np.array([[100.0, 420.0], [100.0, 320.0], [100.0, 220.0]], dtype=np.float32),
    half_width_px=np.array([18.0, 18.0, 18.0], dtype=np.float32),
    forward_m=np.array([5.0, 15.0, 25.0], dtype=np.float32),
  )
  detections = [Detection("person", 0.9, 150.0, 280.0, 180.0, 320.0)]

  associated, active = associate_detections(detections, projected)

  assert associated[0].on_path is False
  assert math.isinf(associated[0].distance_m)
  assert active is None


def test_association_distance_tracks_nearest_path_point():
  projected = ProjectedPath(
    center_px=np.array([[100.0, 420.0], [100.0, 320.0], [100.0, 220.0]], dtype=np.float32),
    half_width_px=np.array([24.0, 20.0, 16.0], dtype=np.float32),
    forward_m=np.array([5.0, 15.0, 25.0], dtype=np.float32),
  )
  near_detection = Detection("person", 0.9, 92.0, 380.0, 108.0, 420.0)
  far_detection = Detection("person", 0.9, 92.0, 180.0, 108.0, 220.0)

  associated, active = associate_detections([near_detection, far_detection], projected)

  assert associated[0].distance_m == 5.0
  assert associated[1].distance_m == 25.0
  assert active is not None
  assert active.distance_m == 5.0


def test_default_path_width_tracks_vehicle_corridor_not_full_lane():
  assert DEFAULT_PATH_WIDTH_M == 1.8


def test_low_confidence_on_path_detection_does_not_activate_hazard():
  projected = ProjectedPath(
    center_px=np.array([[100.0, 320.0]], dtype=np.float32),
    half_width_px=np.array([20.0], dtype=np.float32),
    forward_m=np.array([15.0], dtype=np.float32),
  )
  detections = [Detection("bench", 0.28, 90.0, 280.0, 110.0, 320.0)]

  associated, active = associate_detections(detections, projected)

  assert associated[0].on_path is True
  assert associated[0].distance_m == 15.0
  assert active is None


def test_lane_edge_detection_inside_old_width_but_outside_vehicle_corridor_is_rejected():
  projected = ProjectedPath(
    center_px=np.array([[100.0, 320.0]], dtype=np.float32),
    half_width_px=np.array([18.0], dtype=np.float32),
    forward_m=np.array([15.0], dtype=np.float32),
  )
  detections = [Detection("chair", 0.9, 114.0, 280.0, 126.0, 320.0)]

  associated, active = associate_detections(detections, projected)

  assert associated[0].on_path is False
  assert active is None


def test_hazard_tracker_requires_two_frames_and_clears_after_two_misses():
  tracker = HazardTracker(activate_frames=2, clear_frames=2)
  detection = Detection("person", 0.9, 90.0, 280.0, 110.0, 320.0, on_path=True, distance_m=12.0, footpoint_x=100.0, footpoint_y=320.0)

  assert tracker.update(detection) is None
  active = tracker.update(detection)
  assert active is not None
  assert active.distance_m == 12.0

  assert tracker.update(None) is not None
  assert tracker.update(None) is None
