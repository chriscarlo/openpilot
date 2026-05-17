from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import cereal.messaging as messaging
import pytest

from openpilot.common.params import Params
from openpilot.sunnypilot.objectd.backend import BackendError, SnpeYoloDetector
from openpilot.sunnypilot.objectd.prepare_yolo11n_assets import build_metadata
from openpilot.system.manager.process_config import managed_processes, object_hazard_enabled


def test_objectd_process_registered():
  assert "objectd" in managed_processes
  process = managed_processes["objectd"]
  assert process.name == "objectd"
  assert process.module == "sunnypilot.objectd.objectd"


def test_object_hazard_enabled_requires_onroad_param_and_real_car():
  params = Mock()
  cp = SimpleNamespace(notCar=False)

  params.get_bool.return_value = True
  assert object_hazard_enabled(True, params, cp) is True

  params.get_bool.return_value = False
  assert object_hazard_enabled(True, params, cp) is False

  params.get_bool.return_value = True
  assert object_hazard_enabled(False, params, cp) is False
  assert object_hazard_enabled(True, params, SimpleNamespace(notCar=True)) is False


def test_object_hazard_param_defaults_enabled():
  assert Params().get("ObjectHazardEnabled", return_default=True) is True


def test_yolo11n_asset_metadata_matches_backend_contract():
  metadata = build_metadata("0" * 64)

  assert metadata["input_name"] == "image"
  assert metadata["input_width"] == 640
  assert metadata["input_height"] == 640
  assert metadata["input_layout"] == "NCHW"
  assert metadata["prediction_count"] == 8400
  assert metadata["attributes"] == 84
  assert metadata["prediction_layout"] == "attributes_first"
  assert metadata["has_objectness"] is False
  assert metadata["hazard_labels"] == ["bicycle", "cow", "dog", "horse", "person", "sheep"]


def test_snpe_backend_rejects_qnn_assets_before_loading_model():
  SnpeYoloDetector._validate_export_runtime({})

  with pytest.raises(BackendError, match="QNN_DLC"):
    SnpeYoloDetector._validate_export_runtime({"export_runtime": "QNN_DLC"})


def test_object_hazard_messages_expose_new_schema_fields():
  hazard_msg = messaging.new_message("objectHazardStateSP")
  hazard_msg.objectHazardStateSP.backend = "snpe_gpu"
  hazard_msg.objectHazardStateSP.hazardClass = "person"
  detections = hazard_msg.objectHazardStateSP.init("detections", 1)
  detections[0].className = "person"

  plan_msg = messaging.new_message("longitudinalPlanSP")
  plan_msg.longitudinalPlanSP.objectHazardControl.hazardClass = "person"
  plan_msg.longitudinalPlanSP.objectHazardControl.stopRequired = True

  assert str(hazard_msg.objectHazardStateSP.backend) == "snpe_gpu"
  assert str(hazard_msg.objectHazardStateSP.detections[0].className) == "person"
  assert str(plan_msg.longitudinalPlanSP.objectHazardControl.hazardClass) == "person"
  assert bool(plan_msg.longitudinalPlanSP.objectHazardControl.stopRequired) is True
