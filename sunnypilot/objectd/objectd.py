#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass, field

from cereal import messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
from setproctitle import setproctitle

from openpilot.common.params import Params
from openpilot.common.realtime import Priority, Ratekeeper, config_realtime_process
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.objectd.backend import BackendError, NullDetectorBackend, build_detector_backend
from openpilot.sunnypilot.objectd.config import ObjectdRuntimeConfig
from openpilot.sunnypilot.objectd.path_association import (
  DEFAULT_CAMERA_HEIGHT_M,
  HazardTracker,
  associate_detections,
  build_road_calibration_transform,
  project_model_path,
)
from openpilot.sunnypilot.selfdrive.controls.lib.object_hazard_controller import (
  compute_hazard_speed_recommendation,
  should_stop_for_hazard,
)

PROCESS_NAME = "sunnypilot.objectd.objectd"
ENABLE_PARAM = "ObjectHazardEnabled"


@dataclass
class HazardSnapshot:
  enabled: bool = True
  model_ready: bool = False
  backend: str = "null"
  active: bool = False
  hazard_on_path: bool = False
  stop_required: bool = False
  recommended_speed: float = 0.0
  hazard_distance_m: float = 0.0
  hazard_confidence: float = 0.0
  hazard_class: str = ""
  image_x: float = 0.0
  image_y: float = 0.0
  source_frame_id: int = 0
  detections: list = field(default_factory=list)


def publish_object_hazard_state(pm: messaging.PubMaster, snapshot: HazardSnapshot) -> None:
  msg = messaging.new_message("objectHazardStateSP", valid=True)
  state = msg.objectHazardStateSP
  state.timeStamp = time.monotonic_ns()
  state.enabled = snapshot.enabled
  state.active = snapshot.active
  state.modelReady = snapshot.model_ready
  state.backend = snapshot.backend
  state.hazardOnPath = snapshot.hazard_on_path
  state.stopRequired = snapshot.stop_required
  state.recommendedSpeed = float(snapshot.recommended_speed)
  state.hazardDistanceM = float(snapshot.hazard_distance_m)
  state.hazardConfidence = float(snapshot.hazard_confidence)
  state.hazardClass = snapshot.hazard_class
  state.imageX = float(snapshot.image_x)
  state.imageY = float(snapshot.image_y)
  state.sourceFrameId = int(snapshot.source_frame_id)
  if snapshot.detections:
    detections = state.init("detections", len(snapshot.detections))
    for idx, detection in enumerate(snapshot.detections):
      detections[idx].className = detection.class_name
      detections[idx].confidence = float(detection.confidence)
      detections[idx].xMin = float(detection.x_min)
      detections[idx].yMin = float(detection.y_min)
      detections[idx].xMax = float(detection.x_max)
      detections[idx].yMax = float(detection.y_max)
      detections[idx].onPath = bool(detection.on_path)
      detections[idx].distanceM = float(detection.distance_m if detection.on_path else 0.0)
  pm.send("objectHazardStateSP", msg)


def try_connect_road_camera() -> VisionIpcClient | None:
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  return vipc_client if vipc_client.connect(False) else None


def recv_latest_buffer(vipc_client: VisionIpcClient):
  buf = vipc_client.recv(100)
  if buf is None:
    return None

  while True:
    newer = vipc_client.recv(0)
    if newer is None:
      return buf
    buf = newer


def main() -> None:
  setproctitle(PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  config_realtime_process(4, Priority.CTRL_LOW)
  runtime_config = ObjectdRuntimeConfig.from_env()
  runtime_config.apply_environment_defaults()

  tracker = HazardTracker()
  params = Params()
  backend = NullDetectorBackend("disabled")
  pm = messaging.PubMaster(["objectHazardStateSP"])
  sm = messaging.SubMaster(["carState", "deviceState", "liveCalibration", "modelV2", "roadCameraState"])
  rk = Ratekeeper(max(1, int(round(runtime_config.detector_hz))), print_delay_threshold=None)
  vipc_client = None
  last_backend_status = ""
  last_backend_error = ""

  while True:
    sm.update(0)
    feature_enabled = params.get_bool(ENABLE_PARAM)
    device_started = bool(sm.seen["deviceState"] and sm["deviceState"].started)
    warmup_allowed = (not device_started) or runtime_config.allow_onroad_warmup
    if feature_enabled and isinstance(backend, NullDetectorBackend) and backend.reason == "disabled":
      try:
        backend = build_detector_backend()
      except BackendError as err:
        cloudlog.error("objectd backend unavailable: %s", err)
        backend = NullDetectorBackend(str(err))

    if feature_enabled and warmup_allowed:
      backend.start_warmup()
    if not device_started:
      vipc_client = None
    if feature_enabled and device_started and backend.ready and vipc_client is None:
      try:
        vipc_client = try_connect_road_camera()
      except Exception as err:
        cloudlog.exception("objectd road camera connect failed: %s", err)
        vipc_client = None

    backend_status = getattr(backend, "status_name", backend.backend_name)
    if (feature_enabled and device_started and not runtime_config.allow_onroad_warmup and
        not backend.ready and not getattr(backend, "warming", False) and
        not isinstance(backend, NullDetectorBackend)):
      backend_status = f"{backend_status}:waiting_offroad_warmup"
    if backend_status != last_backend_status:
      cloudlog.info("objectd backend status: %s", backend_status)
      last_backend_status = backend_status
    backend_error = getattr(backend, "last_error", "")
    if backend_error and backend_error != last_backend_error:
      cloudlog.error("objectd backend error: %s", backend_error)
      last_backend_error = backend_error

    snapshot = HazardSnapshot(
      enabled=feature_enabled,
      model_ready=bool(feature_enabled and backend.ready),
      backend=backend_status,
    )
    if (feature_enabled and backend.ready and vipc_client is not None and
        sm.valid.get("modelV2", False) and sm.alive.get("modelV2", False) and
        sm.valid.get("liveCalibration", False) and sm.alive.get("liveCalibration", False)):
      try:
        buf = recv_latest_buffer(vipc_client)
      except Exception as err:
        cloudlog.exception("objectd road camera recv failed: %s", err)
        buf = None

      if buf is not None and sm.seen["roadCameraState"] and sm.seen["deviceState"]:
        try:
          v_ego = float(sm["carState"].vEgo) if sm.valid.get("carState", False) else 0.0
          camera_height = float(sm["liveCalibration"].height[0]) if sm["liveCalibration"].height else DEFAULT_CAMERA_HEIGHT_M
          transform = build_road_calibration_transform(
            str(sm["deviceState"].deviceType),
            str(sm["roadCameraState"].sensor),
            sm["liveCalibration"].rpyCalib,
          )
          projected_path = project_model_path(sm["modelV2"].position, transform, camera_height_m=camera_height)
          detections, candidate = associate_detections(backend.infer(buf), projected_path)
          active_detection = tracker.update(candidate)

          if active_detection is not None:
            recommended_speed = compute_hazard_speed_recommendation(active_detection.distance_m, v_ego, None)
            snapshot.active = True
            snapshot.hazard_on_path = True
            snapshot.hazard_distance_m = float(active_detection.distance_m)
            snapshot.hazard_confidence = float(active_detection.confidence)
            snapshot.hazard_class = active_detection.class_name
            snapshot.image_x = float(active_detection.footpoint_x)
            snapshot.image_y = float(active_detection.footpoint_y)
            snapshot.source_frame_id = int(vipc_client.frame_id)
            snapshot.recommended_speed = float(recommended_speed)
            snapshot.stop_required = should_stop_for_hazard(active_detection.distance_m, recommended_speed)

          snapshot.detections = detections[:runtime_config.debug_detection_limit]
        except BackendError as err:
          cloudlog.error("objectd backend failed closed: %s", err)
          backend.mark_failed(str(err))
          snapshot.model_ready = False
          snapshot.backend = getattr(backend, "status_name", backend.backend_name)
          tracker.update(None)
        except Exception as err:
          cloudlog.exception("objectd inference/path association failed: %s", err)
          tracker.update(None)
      else:
        tracker.update(None)
    else:
      tracker.update(None)

    publish_object_hazard_state(pm, snapshot)
    rk.keep_time()


if __name__ == "__main__":
  main()
