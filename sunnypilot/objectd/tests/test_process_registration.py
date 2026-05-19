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
  COCO_80_LABELS,
  DEFAULT_HAZARD_LABELS,
  OrtCpuYoloDetector,
  OrtQnnYoloDetector,
  QnnNetRunYoloDetector,
  SnpeYoloDetector,
  TinygradOnnxYoloDetector,
  YoloDetectorBase,
)
import openpilot.sunnypilot.objectd.backend as objectd_backend
from openpilot.sunnypilot.objectd.config import (
  DEFAULT_ALLOW_ONROAD_WARMUP,
  DEFAULT_BACKEND,
  DEFAULT_DETECTOR_HZ,
  DEFAULT_TINYGRAD_DEVICE,
  DEFAULT_TINYGRAD_WARMUP_RUNS,
  ObjectdRuntimeConfig,
  PRIMARY_MODEL_DIR,
)
from openpilot.sunnypilot.objectd.prepare_yolo11n_assets import build_metadata
from openpilot.sunnypilot.objectd.scheduler import ObjectdInferenceScheduler
from openpilot.system.manager.process_config import managed_processes, object_hazard_enabled

EXPECTED_DEFAULT_HAZARD_LABELS = sorted(DEFAULT_HAZARD_LABELS)


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
  assert object_hazard_enabled(True, params, cp) is True

  params.get_bool.return_value = True
  assert object_hazard_enabled(False, params, cp) is True
  assert object_hazard_enabled(True, params, SimpleNamespace(notCar=True)) is False
  params.get_bool.assert_not_called()


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
  assert metadata["hazard_labels"] == EXPECTED_DEFAULT_HAZARD_LABELS


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
  assert metadata["hazard_labels"] == EXPECTED_DEFAULT_HAZARD_LABELS


def test_yolov8n_tinygrad_160_metadata_matches_managed_runtime():
  metadata = build_metadata("0" * 64, model_preset="yolov8n_tinygrad_160", export_runtime="ONNX")

  assert metadata["source_checkpoint"] == "YOLOv8-N / yolov8n.pt"
  assert metadata["export_runtime"] == "ONNX"
  assert metadata["input_name"] == "images"
  assert metadata["input_width"] == 160
  assert metadata["input_height"] == 160
  assert metadata["input_layout"] == "NCHW"
  assert metadata["output_name"] == "output0"
  assert metadata["prediction_count"] == 525
  assert metadata["attributes"] == 84
  assert metadata["hazard_labels"] == EXPECTED_DEFAULT_HAZARD_LABELS


def test_yolo11n_tinygrad_160_metadata_matches_managed_runtime():
  metadata = build_metadata("0" * 64, model_preset="yolo11n_tinygrad_160")

  assert metadata["source_repo"] == "ultralytics/yolo11"
  assert metadata["source_checkpoint"] == "YOLO11-N / yolo11n.pt"
  assert metadata["export_runtime"] == "ONNX"
  assert metadata["input_name"] == "images"
  assert metadata["input_width"] == 160
  assert metadata["input_height"] == 160
  assert metadata["input_layout"] == "NCHW"
  assert metadata["output_name"] == "output0"
  assert metadata["prediction_count"] == 525
  assert metadata["attributes"] == 84
  assert metadata["prediction_layout"] == "attributes_first"
  assert metadata["has_objectness"] is False
  assert metadata["hazard_labels"] == EXPECTED_DEFAULT_HAZARD_LABELS


def test_yolo11n_tinygrad_fallback_metadata_presets_are_available():
  metadata_192 = build_metadata("0" * 64, model_preset="yolo11n_tinygrad_192")
  metadata_224 = build_metadata("0" * 64, model_preset="yolo11n_tinygrad_224")

  assert metadata_192["input_width"] == 192
  assert metadata_192["input_height"] == 192
  assert metadata_192["prediction_count"] == 756
  assert metadata_224["input_width"] == 224
  assert metadata_224["input_height"] == 224
  assert metadata_224["prediction_count"] == 1029
  assert metadata_192["export_runtime"] == metadata_224["export_runtime"] == "ONNX"


def test_yolov8n_tinygrad_256_metadata_remains_available_for_override():
  metadata = build_metadata("0" * 64, model_preset="yolov8n_tinygrad_256", export_runtime="ONNX")

  assert metadata["input_width"] == 256
  assert metadata["input_height"] == 256
  assert metadata["output_name"] == "output0"
  assert metadata["prediction_count"] == 1344
  assert metadata["attributes"] == 84
  assert metadata["hazard_labels"] == EXPECTED_DEFAULT_HAZARD_LABELS


def test_default_hazard_labels_include_rigid_road_obstacles_but_not_soft_debris():
  for label in ("person", "dog", "car", "motorcycle", "fire hydrant", "chair", "refrigerator"):
    assert label in DEFAULT_HAZARD_LABELS
  for label in ("banana", "apple", "fork", "book", "toothbrush"):
    assert label not in DEFAULT_HAZARD_LABELS


def test_decoder_filters_default_hazard_labels():
  detector = YoloDetectorBase.__new__(YoloDetectorBase)
  detector.attributes = 84
  detector.prediction_count = 2
  detector.prediction_layout = "attributes_first"
  detector.has_objectness = False
  detector.confidence_threshold = 0.25
  detector.iou_threshold = 0.45
  detector.input_width = 10
  detector.input_height = 10
  detector.labels = COCO_80_LABELS
  detector.hazard_labels = set(DEFAULT_HAZARD_LABELS)
  detector._crop_offset_x = 0
  detector._crop_offset_y = 0
  detector.output = np.zeros((detector.attributes, detector.prediction_count), dtype=np.float32)
  detector.output[:4, 0] = [5.0, 5.0, 2.0, 2.0]
  detector.output[:4, 1] = [7.0, 7.0, 2.0, 2.0]
  detector.output[4 + COCO_80_LABELS.index("banana"), 0] = 0.95
  detector.output[4 + COCO_80_LABELS.index("car"), 1] = 0.90

  detections = detector._decode_predictions(10, 10)

  assert [detection.class_name for detection in detections] == ["car"]


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


def test_tinygrad_onnx_backend_only_accepts_plain_onnx_assets():
  TinygradOnnxYoloDetector._validate_export_runtime({"export_runtime": "ONNX"})

  with pytest.raises(BackendError, match="QNN_DLC"):
    TinygradOnnxYoloDetector._validate_export_runtime({"export_runtime": "QNN_DLC"})


def test_tinygrad_onnx_device_selection_is_accelerator_first(monkeypatch):
  monkeypatch.setenv("OBJECTD_TINYGRAD_DEVICE", "QCOM")
  monkeypatch.delenv("DEV", raising=False)
  assert TinygradOnnxYoloDetector._configure_tinygrad_device() == "QCOM"
  assert objectd_backend.os.environ["DEV"] == "QCOM"


def test_tinygrad_onnx_rejects_cpu_without_explicit_measurement_opt_in(monkeypatch):
  monkeypatch.delenv("OBJECTD_TINYGRAD_DEVICE", raising=False)
  monkeypatch.delenv("OBJECTD_ALLOW_CPU_INFERENCE", raising=False)
  monkeypatch.delenv("DEV", raising=False)
  monkeypatch.setattr(objectd_backend.Path, "exists", lambda self: False)

  with pytest.raises(BackendError, match="requires an accelerator"):
    TinygradOnnxYoloDetector._configure_tinygrad_device()


def test_tinygrad_worker_affinity_parser_accepts_lists_and_ranges():
  assert TinygradOnnxYoloDetector._parse_cpu_affinity("") == set()
  assert TinygradOnnxYoloDetector._parse_cpu_affinity("0,2-3,5") == {0, 2, 3, 5}


def test_managed_objectd_runtime_defaults_point_at_tinygrad_onnx(monkeypatch):
  for key in (
    "OBJECTD_BACKEND",
    "OBJECTD_MODEL_DIR",
    "OBJECTD_MODEL_PATH",
    "OBJECTD_MODEL_METADATA",
    "OBJECTD_TINYGRAD_DEVICE",
    "OBJECTD_DETECTOR_HZ",
    "OBJECTD_TINYGRAD_WARMUP_RUNS",
    "OBJECTD_ROI_MODE",
    "OMP_NUM_THREADS",
  ):
    monkeypatch.delenv(key, raising=False)

  config = ObjectdRuntimeConfig.from_env()
  config.apply_environment_defaults()

  assert config.backend == DEFAULT_BACKEND == "tinygrad_onnx"
  assert config.model_dir == PRIMARY_MODEL_DIR
  assert config.model_dir.name == "yolo11n_tinygrad_160_install"
  assert config.tinygrad_device == DEFAULT_TINYGRAD_DEVICE == "QCOM"
  assert config.detector_hz == DEFAULT_DETECTOR_HZ == 2.0
  assert config.tinygrad_warmup_runs == DEFAULT_TINYGRAD_WARMUP_RUNS == 3
  assert config.allow_onroad_warmup == DEFAULT_ALLOW_ONROAD_WARMUP is True
  assert objectd_backend.os.environ["OBJECTD_BACKEND"] == "tinygrad_onnx"
  assert objectd_backend.os.environ["OBJECTD_MODEL_PATH"].endswith("model.onnx")
  assert objectd_backend.os.environ["OMP_NUM_THREADS"] == "1"
  assert objectd_backend.os.environ["OBJECTD_ROI_MODE"] == "road_wide"


def test_objectd_inference_scheduler_skips_one_tick_after_overrun():
  scheduler = ObjectdInferenceScheduler(detector_hz=2.0, infer_budget_ms=100.0, phase_sec=0.0)
  scheduler.next_run_time = 10.0

  assert scheduler.should_run(9.9) is False
  assert scheduler.should_run(10.0) is True
  scheduler.record_runtime(0.150)
  assert scheduler.should_run(10.5) is False
  assert scheduler.should_run(11.0) is True


def test_road_wide_roi_crops_top_of_nv12_frame(monkeypatch):
  if objectd_backend.cv2 is None:
    pytest.skip("opencv unavailable")
  monkeypatch.setenv("OBJECTD_ROI_MODE", "road_wide")
  monkeypatch.setenv("OBJECTD_ROI_TOP_FRACTION", "0.25")
  detector = YoloDetectorBase.__new__(YoloDetectorBase)
  detector.roi_mode = "road_wide"
  detector.roi_top_fraction = 0.25
  y_plane = np.full((8, 8), 64, dtype=np.uint8)
  uv_plane = np.full((4, 8), 128, dtype=np.uint8)
  buf = SimpleNamespace(data=np.vstack((y_plane, uv_plane)).tobytes(), stride=8, width=8, height=8)

  rgb = detector._visionbuf_to_rgb(buf)

  assert rgb.shape == (6, 8, 3)
  assert detector._crop_offset_x == 0
  assert detector._crop_offset_y == 2


def test_tinygrad_warmup_starts_in_background_and_marks_ready():
  detector = TinygradOnnxYoloDetector.__new__(TinygradOnnxYoloDetector)
  detector.ready = False
  detector.warming = False
  detector.last_error = ""
  detector.backend_name = "tinygrad_onnx:qcom"
  detector.warmup_stage = ""
  detector._warmup_runs = 1
  detector._warmup_lock = objectd_backend.threading.Lock()
  detector._worker_thread = None
  detector._request_queue = objectd_backend.queue.Queue(maxsize=1)
  detector._build_worker_runner = lambda: (object, object())
  detector._run_warmup_blocking = lambda _tensor, _jit: None

  detector.start_warmup()
  detector._worker_thread.join(timeout=1.0)

  assert detector.ready is True
  assert detector.warming is False
  assert detector.status_name == "tinygrad_onnx:qcom"


def test_tinygrad_warmup_failure_stays_fail_closed():
  detector = TinygradOnnxYoloDetector.__new__(TinygradOnnxYoloDetector)
  detector.ready = False
  detector.warming = False
  detector.last_error = ""
  detector.backend_name = "tinygrad_onnx:qcom"
  detector.warmup_stage = ""
  detector._warmup_runs = 1
  detector._warmup_lock = objectd_backend.threading.Lock()
  detector._worker_thread = None
  detector._request_queue = objectd_backend.queue.Queue(maxsize=1)
  detector._build_worker_runner = lambda: (object, object())

  def _raise_backend_error(_tensor, _jit):
    raise BackendError("qcom compile failed")

  detector._run_warmup_blocking = _raise_backend_error

  detector.start_warmup()
  detector._worker_thread.join(timeout=1.0)

  assert detector.ready is False
  assert detector.warming is False
  assert detector.last_error == "qcom compile failed"
  assert detector.status_name == "tinygrad_onnx:qcom:error"


def test_auto_onnx_assets_use_tinygrad_backend(monkeypatch):
  detector = object()
  monkeypatch.setenv("OBJECTD_BACKEND", "auto")
  monkeypatch.setattr(objectd_backend, "_read_default_export_runtime", lambda: "ONNX")
  monkeypatch.setattr(objectd_backend, "TinygradOnnxYoloDetector", lambda: detector)

  assert objectd_backend.build_detector_backend() is detector


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
