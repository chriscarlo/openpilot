from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import cereal.messaging as messaging
import numpy as np
import pytest

from openpilot.common.params import Params
from openpilot.sunnypilot.objectd.backend import (
  BackendError,
  OrtCpuYoloDetector,
  OrtQnnYoloDetector,
  QnnNetRunYoloDetector,
  SnpeYoloDetector,
)
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

  assert metadata["source_repo"] == "qualcomm/YOLOv11-Detection"
  assert metadata["source_checkpoint"] == "YOLO11-N / yolo11n.pt"
  assert metadata["export_runtime"] == "QNN_DLC"
  assert metadata["input_name"] == "image"
  assert metadata["input_width"] == 640
  assert metadata["input_height"] == 640
  assert metadata["input_layout"] == "NHWC"
  assert metadata["output_name"] == "detector_output"
  assert metadata["prediction_count"] == 8400
  assert metadata["attributes"] == 84
  assert metadata["prediction_layout"] == "attributes_first"
  assert metadata["has_objectness"] is False
  assert metadata["hazard_labels"] == ["bicycle", "cow", "dog", "horse", "person", "sheep"]


def test_yolov8n_asset_metadata_preserves_current_decoder_contract():
  metadata = build_metadata("0" * 64, model_preset="yolov8n")

  assert metadata["source_repo"] == "qualcomm/YOLOv8-Detection"
  assert metadata["source_checkpoint"] == "YOLOv8-N"
  assert metadata["export_runtime"] == "QNN_DLC"
  assert metadata["input_width"] == 640
  assert metadata["input_height"] == 640
  assert metadata["output_name"] == "detector_output"
  assert metadata["prediction_count"] == 8400
  assert metadata["attributes"] == 84
  assert metadata["prediction_layout"] == "attributes_first"
  assert metadata["hazard_labels"] == ["bicycle", "cow", "dog", "horse", "person", "sheep"]


def test_asset_metadata_allows_explicit_model_overrides():
  metadata = build_metadata(
    "0" * 64,
    model_preset="yolov8n",
    source_repo="local/custom-detector",
    source_checkpoint="custom-mobile-hazard-v1",
    input_width=320,
    input_height=320,
    output_name="boxes",
    prediction_count=2100,
    attributes=85,
    prediction_layout="predictions_first",
    has_objectness=True,
    labels=["person", "bicycle", "car"],
    hazard_labels=["person", "bicycle"],
  )

  assert metadata["source_repo"] == "local/custom-detector"
  assert metadata["source_checkpoint"] == "custom-mobile-hazard-v1"
  assert metadata["input_width"] == 320
  assert metadata["input_height"] == 320
  assert metadata["output_name"] == "boxes"
  assert metadata["prediction_count"] == 2100
  assert metadata["attributes"] == 85
  assert metadata["prediction_layout"] == "predictions_first"
  assert metadata["has_objectness"] is True
  assert metadata["labels"] == ["person", "bicycle", "car"]
  assert metadata["hazard_labels"] == ["person", "bicycle"]


def test_asset_metadata_allows_quantized_qnn_io():
  metadata = build_metadata(
    "0" * 64,
    model_preset="yolov8n",
    export_runtime="QNN_DLC",
    input_dtype="uint8",
    input_scale=1.0 / 255.0,
    input_zero_point=0,
    output_dtype="uint8",
    output_scale=2.5,
    output_zero_point=1,
  )

  assert metadata["input_dtype"] == "uint8"
  assert metadata["input_scale"] == 1.0 / 255.0
  assert metadata["input_zero_point"] == 0
  assert metadata["output_dtype"] == "uint8"
  assert metadata["output_scale"] == 2.5
  assert metadata["output_zero_point"] == 1


def test_yolo11n_asset_metadata_allows_snpe_assets():
  metadata = build_metadata("0" * 64, export_runtime="SNPE_DLC")

  assert metadata["export_runtime"] == "SNPE_DLC"
  assert metadata["input_layout"] == "NCHW"


def test_yolo11n_asset_metadata_allows_precompiled_qnn_onnx_assets():
  metadata = build_metadata("0" * 64, export_runtime="PRECOMPILED_QNN_ONNX")

  assert metadata["export_runtime"] == "PRECOMPILED_QNN_ONNX"
  assert metadata["input_layout"] == "NHWC"


def test_yolov8n_asset_metadata_allows_plain_onnx_measurement_assets():
  metadata = build_metadata("0" * 64, model_preset="yolov8n", export_runtime="ONNX")

  assert metadata["export_runtime"] == "ONNX"
  assert metadata["input_layout"] == "NCHW"


def test_snpe_backend_rejects_qnn_assets_before_loading_model():
  SnpeYoloDetector._validate_export_runtime({})

  with pytest.raises(BackendError, match="QNN_DLC"):
    SnpeYoloDetector._validate_export_runtime({"export_runtime": "QNN_DLC"})
  with pytest.raises(BackendError, match="ONNX"):
    SnpeYoloDetector._validate_export_runtime({"export_runtime": "ONNX"})


def test_snpe_backend_rejects_qairt_2_dlc_before_loading_model(tmp_path):
  model_path = tmp_path / "model.dlc"
  dlc_metadata = {
    "dlcGenerationInfo": [
      {"converterCommand": {"converterVersion": "2.42.0.251225135753_193295"}},
    ],
  }
  with zipfile.ZipFile(model_path, "w", compression=zipfile.ZIP_STORED) as dlc:
    dlc.writestr("dlc.metadata2.1.0", json.dumps(dlc_metadata))

  with pytest.raises(BackendError, match="SNPE runtime is 1.61"):
    SnpeYoloDetector._validate_dlc_compatibility(model_path)


def test_snpe_backend_rejects_qairt_2_legacy_metadata_schema(tmp_path):
  model_path = tmp_path / "model.dlc"
  dlc_metadata = {
    "dlc-generation-info": [
      {"converter-command": {"converter-version": "2.33.0.250327124043_117917"}},
    ],
  }
  with zipfile.ZipFile(model_path, "w", compression=zipfile.ZIP_STORED) as dlc:
    dlc.writestr("dlc.metadata2.0.1", json.dumps(dlc_metadata))

  with pytest.raises(BackendError, match="2.33.0"):
    SnpeYoloDetector._validate_dlc_compatibility(model_path)


def test_snpe_backend_rejects_public_qairt_2_22_dlc(tmp_path):
  model_path = tmp_path / "model.dlc"
  dlc_metadata = "\n".join([
    "converter-command=snpe-onnx-to-dlc --input_network yolov11_det.onnx",
    "converter-version=2.22.6.240515184619_92920",
    "model-version=yolo11n_object_hazard_qairt222",
  ])
  with zipfile.ZipFile(model_path, "w", compression=zipfile.ZIP_STORED) as dlc:
    dlc.writestr("dlc.metadata", dlc_metadata)

  with pytest.raises(BackendError, match="model format 4.x"):
    SnpeYoloDetector._validate_dlc_compatibility(model_path)


def test_snpe_backend_allows_legacy_key_value_metadata(tmp_path):
  model_path = tmp_path / "model.dlc"
  dlc_metadata = "\n".join([
    "converter-command=snpe-onnx-to-dlc --input_network yolov11_det.onnx",
    "converter-version=1.61.0.3358",
    "model-version=yolo11n_object_hazard_snpe161",
  ])
  with zipfile.ZipFile(model_path, "w", compression=zipfile.ZIP_STORED) as dlc:
    dlc.writestr("dlc.metadata", dlc_metadata)

  SnpeYoloDetector._validate_dlc_compatibility(model_path)


def test_snpe_backend_allows_legacy_converter_dlc(tmp_path):
  model_path = tmp_path / "model.dlc"
  dlc_metadata = {
    "dlcGenerationInfo": [
      {"converterCommand": {"converterVersion": "1.61.0.3358"}},
    ],
  }
  with zipfile.ZipFile(model_path, "w", compression=zipfile.ZIP_STORED) as dlc:
    dlc.writestr("dlc.metadata", json.dumps(dlc_metadata))

  SnpeYoloDetector._validate_dlc_compatibility(model_path)


def test_qnn_backend_only_accepts_qnn_dlc_assets():
  QnnNetRunYoloDetector._validate_export_runtime({"export_runtime": "QNN_DLC"})

  with pytest.raises(BackendError, match="SNPE_DLC"):
    QnnNetRunYoloDetector._validate_export_runtime({"export_runtime": "SNPE_DLC"})


def test_qnn_backend_quantizes_uint8_input(tmp_path):
  detector = QnnNetRunYoloDetector.__new__(QnnNetRunYoloDetector)
  detector.input = np.array([0.0, 0.5, 1.0], dtype=np.float32)
  detector.input_dtype = "uint8"
  detector.input_scale = 1.0 / 255.0
  detector.input_zero_point = 0
  input_path = tmp_path / "image.raw"

  detector._write_qnn_input(input_path)

  assert np.fromfile(input_path, dtype=np.uint8).tolist() == [0, 128, 255]


def test_qnn_backend_dequantizes_uint8_output(tmp_path):
  detector = QnnNetRunYoloDetector.__new__(QnnNetRunYoloDetector)
  detector.expected_output_size = 3
  detector.output_dtype = "uint8"
  detector.output_scale = 2.5
  detector.output_zero_point = 1
  output_dir = tmp_path / "output"
  output_dir.mkdir()
  np.array([1, 2, 3], dtype=np.uint8).tofile(output_dir / "detector_output.raw")

  output = detector._read_qnn_output(output_dir)

  assert output.dtype == np.float32
  assert output.tolist() == [0.0, 2.5, 5.0]


def test_qnn_backend_uses_native_files_for_quantized_io(tmp_path):
  detector = QnnNetRunYoloDetector.__new__(QnnNetRunYoloDetector)
  detector.qnn_net_run = "qnn-net-run"
  detector.qnn_backend = "libQnnHtp.so"
  detector.qnn_model_dlc_lib = "libQnnModelDlc.so"
  detector.model_path = tmp_path / "model.dlc"
  detector.input_dtype = "uint8"
  detector.output_dtype = "uint8"

  cmd = detector._build_qnn_command(tmp_path / "input_list.txt", tmp_path / "output")

  assert "--use_native_input_files" in cmd
  assert "--use_native_output_files" in cmd


def test_qnn_backend_omits_native_files_for_float_io(tmp_path):
  detector = QnnNetRunYoloDetector.__new__(QnnNetRunYoloDetector)
  detector.qnn_net_run = "qnn-net-run"
  detector.qnn_backend = "libQnnHtp.so"
  detector.qnn_model_dlc_lib = "libQnnModelDlc.so"
  detector.model_path = tmp_path / "model.dlc"
  detector.input_dtype = "float32"
  detector.output_dtype = "float32"

  cmd = detector._build_qnn_command(tmp_path / "input_list.txt", tmp_path / "output")

  assert "--use_native_input_files" not in cmd
  assert "--use_native_output_files" not in cmd


def test_ort_qnn_backend_only_accepts_precompiled_qnn_onnx_assets():
  OrtQnnYoloDetector._validate_export_runtime({"export_runtime": "PRECOMPILED_QNN_ONNX"})

  with pytest.raises(BackendError, match="QNN_DLC"):
    OrtQnnYoloDetector._validate_export_runtime({"export_runtime": "QNN_DLC"})


def test_onnx_cpu_backend_is_explicit_measurement_only(monkeypatch):
  OrtCpuYoloDetector._validate_export_runtime({"export_runtime": "ONNX"})

  with pytest.raises(BackendError, match="QNN_DLC"):
    OrtCpuYoloDetector._validate_export_runtime({"export_runtime": "QNN_DLC"})

  monkeypatch.delenv("OBJECTD_ALLOW_CPU_INFERENCE", raising=False)
  assert OrtCpuYoloDetector._allow_cpu_inference() is False
  monkeypatch.setenv("OBJECTD_ALLOW_CPU_INFERENCE", "1")
  assert OrtCpuYoloDetector._allow_cpu_inference() is True


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
