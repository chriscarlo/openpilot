from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

import numpy as np

from openpilot.sunnypilot.objectd.config import (
  DEFAULT_BACKEND,
  DEFAULT_TINYGRAD_WARMUP_RUNS,
  PRIMARY_MODEL_DIR,
)
from openpilot.sunnypilot.objectd.types import Detection

try:
  import cv2
except ModuleNotFoundError:  # pragma: no cover - platform-specific dependency
  cv2 = None

try:
  from openpilot.sunnypilot.modeld.models.commonmodel_pyx import CLContext
  from openpilot.sunnypilot.modeld.runners.runmodel_pyx import Runtime
  from openpilot.sunnypilot.modeld.runners.snpemodel_pyx import SNPEModel
except ModuleNotFoundError:  # pragma: no cover - absent in dev env until built
  CLContext = None
  Runtime = None
  SNPEModel = None

DEFAULT_MODEL_DIR = PRIMARY_MODEL_DIR
DEFAULT_HAZARD_LABELS = {
  "person",
  "bicycle", "car", "motorcycle", "bus", "train", "truck",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "backpack", "suitcase", "skis", "snowboard", "skateboard",
  "chair", "couch", "potted plant", "bed", "dining table", "toilet",
  "tv", "laptop", "microwave", "oven", "sink", "refrigerator",
}
COCO_80_LABELS = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush",
]


class BackendError(RuntimeError):
  pass


class NullDetectorBackend:
  def __init__(self, reason: str = "disabled"):
    self.reason = reason
    self.backend_name = f"null:{reason}"
    self.ready = False
    self.warming = False
    self.last_error = reason

  def infer(self, _buf) -> list[Detection]:
    return []

  @property
  def status_name(self) -> str:
    return self.backend_name

  def start_warmup(self) -> None:
    pass

  def mark_failed(self, reason: str) -> None:
    self.last_error = reason


class YoloDetectorBase:
  def _init_detector_config(self, default_model_name: str = "model.dlc") -> None:
    if cv2 is None:
      raise BackendError("opencv-python-headless is required for objectd preprocessing")
    try:
      cv2.setNumThreads(max(1, int(os.getenv("OBJECTD_CV2_THREADS", "1"))))
    except Exception:
      pass

    self.model_dir = Path(os.getenv("OBJECTD_MODEL_DIR", DEFAULT_MODEL_DIR))
    self.model_path = Path(os.getenv("OBJECTD_MODEL_PATH", self.model_dir / default_model_name))
    self.metadata_path = Path(os.getenv("OBJECTD_MODEL_METADATA", self.model_dir / "metadata.json"))
    metadata = self._load_metadata(self.metadata_path)
    self._verify_sha256(self.model_path, metadata.get("model_sha256"))

    self.export_runtime = self._normalize_export_runtime(metadata)
    self.input_name = str(metadata["input_name"])
    self.input_width = int(metadata["input_width"])
    self.input_height = int(metadata["input_height"])
    self.input_layout = str(metadata.get("input_layout", "NCHW")).upper()
    self.input_dtype = str(metadata.get("input_dtype", "float32")).lower()
    self.input_scale = float(metadata.get("input_scale", 1.0))
    self.input_zero_point = int(metadata.get("input_zero_point", 0))
    self.prediction_count = int(metadata["prediction_count"])
    self.attributes = int(metadata["attributes"])
    self.prediction_layout = str(metadata.get("prediction_layout", "attributes_first"))
    self.output_name = str(metadata.get("output_name", "detector_output"))
    self.output_dtype = str(metadata.get("output_dtype", "float32")).lower()
    self.output_scale = float(metadata.get("output_scale", 1.0))
    self.output_zero_point = int(metadata.get("output_zero_point", 0))
    self.has_objectness = bool(metadata.get("has_objectness", False))
    self.confidence_threshold = float(metadata.get("confidence_threshold", 0.25))
    self.iou_threshold = float(metadata.get("iou_threshold", 0.45))
    self.labels = list(metadata.get("labels", COCO_80_LABELS))
    self.hazard_labels = set(metadata.get("hazard_labels", sorted(DEFAULT_HAZARD_LABELS)))
    self.expected_output_size = self.prediction_count * self.attributes
    self.roi_mode = os.getenv("OBJECTD_ROI_MODE", "road_wide").strip().lower()
    self.roi_top_fraction = max(0.0, min(0.45, self._env_float("OBJECTD_ROI_TOP_FRACTION", 0.15)))
    self._crop_offset_x = 0
    self._crop_offset_y = 0

    input_channels = int(metadata.get("input_channels", 3))
    input_size = self.input_width * self.input_height * input_channels

    self.output = np.zeros(self.expected_output_size, dtype=np.float32)
    self.input = np.zeros(input_size, dtype=np.float32)
    self.ready = False
    self.warming = False
    self.last_error = ""
    self.warmup_stage = ""

  def infer(self, buf) -> list[Detection]:
    rgb = self._prepare_input(buf)
    self._execute_model()
    return self._decode_predictions(rgb.shape[1], rgb.shape[0])

  @property
  def status_name(self) -> str:
    if self.last_error:
      return f"{self.backend_name}:error"
    if self.warming:
      stage = f":{self.warmup_stage}" if self.warmup_stage else ""
      return f"{self.backend_name}:warming{stage}"
    return self.backend_name

  def start_warmup(self) -> None:
    pass

  def mark_failed(self, reason: str) -> None:
    self.ready = False
    self.warming = False
    self.last_error = reason

  def _execute_model(self) -> None:
    raise NotImplementedError

  @staticmethod
  def _env_float(name: str, default: float) -> float:
    try:
      return float(os.getenv(name, str(default)))
    except ValueError:
      return default

  def _prepare_input(self, buf) -> np.ndarray:
    rgb = self._visionbuf_to_rgb(buf)
    resized = cv2.resize(rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    if self.input_layout == "NCHW":
      self.input[:] = normalized.transpose(2, 0, 1).reshape(-1)
    elif self.input_layout == "NHWC":
      self.input[:] = normalized.reshape(-1)
    else:
      raise BackendError(f"unsupported input layout '{self.input_layout}'")
    return rgb

  @staticmethod
  def _normalize_export_runtime(metadata: dict) -> str:
    return str(metadata.get("export_runtime", "SNPE_DLC")).upper()

  @staticmethod
  def _load_metadata(path: Path) -> dict:
    if not path.is_file():
      raise BackendError(f"objectd metadata not found at {path}")
    with path.open() as f:
      metadata = json.load(f)
    required_keys = {"input_name", "input_width", "input_height", "prediction_count", "attributes"}
    missing = required_keys - metadata.keys()
    if missing:
      raise BackendError(f"objectd metadata missing keys: {sorted(missing)}")
    return metadata

  @staticmethod
  def _verify_sha256(path: Path, expected: str | None) -> None:
    if not path.is_file():
      raise BackendError(f"objectd model not found at {path}")
    if not expected:
      return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected.lower():
      raise BackendError(f"objectd model checksum mismatch for {path}")

  @staticmethod
  def _round_even(value: int) -> int:
    return value - (value % 2)

  def _visionbuf_to_rgb(self, buf) -> np.ndarray:
    frame = np.frombuffer(buf.data, dtype=np.uint8).reshape((-1, buf.stride))
    x0 = 0
    y0 = 0
    x1 = buf.width
    y1 = buf.height
    if self.roi_mode == "road_wide":
      y0 = self._round_even(int(buf.height * self.roi_top_fraction))
    elif self.roi_mode != "full":
      raise BackendError(f"unsupported objectd ROI mode '{self.roi_mode}'")

    y0 = max(0, min(y0, buf.height - 2))
    y1 = max(y0 + 2, self._round_even(y1))
    y_plane = frame[:buf.height, :buf.width]
    uv_plane = frame[buf.height:buf.height + buf.height // 2, :buf.width]
    y_crop = y_plane[y0:y1, x0:x1]
    uv_crop = uv_plane[y0 // 2:y1 // 2, x0:x1]
    yuv = np.vstack((y_crop, uv_crop))
    self._crop_offset_x = x0
    self._crop_offset_y = y0
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)

  def _decode_predictions(self, source_width: int, source_height: int) -> list[Detection]:
    predictions = self.output.reshape(self.attributes, self.prediction_count)
    if self.prediction_layout == "attributes_first":
      predictions = predictions.T
    elif self.prediction_layout == "predictions_first":
      predictions = predictions.reshape(self.prediction_count, self.attributes)
    else:
      raise BackendError(f"unsupported prediction layout '{self.prediction_layout}'")

    boxes = predictions[:, :4]
    if self.has_objectness:
      objectness = predictions[:, 4]
      class_scores = predictions[:, 5:]
      class_indices = np.argmax(class_scores, axis=1)
      confidences = objectness * class_scores[np.arange(len(predictions)), class_indices]
    else:
      class_scores = predictions[:, 4:]
      class_indices = np.argmax(class_scores, axis=1)
      confidences = class_scores[np.arange(len(predictions)), class_indices]

    keep = confidences >= self.confidence_threshold
    if not np.any(keep):
      return []

    boxes = boxes[keep]
    class_indices = class_indices[keep]
    confidences = confidences[keep]
    scale_x = source_width / float(self.input_width)
    scale_y = source_height / float(self.input_height)
    labels = [self.labels[idx] if idx < len(self.labels) else f"class_{idx}" for idx in class_indices]

    filtered: list[tuple[np.ndarray, float, str]] = []
    for box, confidence, label in zip(boxes, confidences, labels, strict=True):
      if label not in self.hazard_labels:
        continue
      cx, cy, width, height = box.tolist()
      x_min = max(0.0, (cx - width / 2.0) * scale_x)
      y_min = max(0.0, (cy - height / 2.0) * scale_y)
      x_max = min(source_width - 1.0, (cx + width / 2.0) * scale_x)
      y_max = min(source_height - 1.0, (cy + height / 2.0) * scale_y)
      x_min += self._crop_offset_x
      x_max += self._crop_offset_x
      y_min += self._crop_offset_y
      y_max += self._crop_offset_y
      filtered.append((np.array([x_min, y_min, x_max, y_max], dtype=np.float32), float(confidence), label))

    if not filtered:
      return []

    detections: list[Detection] = []
    for label in sorted({item[2] for item in filtered}):
      label_items = [(box, confidence) for box, confidence, item_label in filtered if item_label == label]
      kept = self._nms(label_items)
      detections.extend(Detection(label, confidence, *box.tolist()) for box, confidence in kept)
    return detections

  def _nms(self, items: list[tuple[np.ndarray, float]]) -> list[tuple[np.ndarray, float]]:
    if not items:
      return []
    order = sorted(items, key=lambda item: item[1], reverse=True)
    kept: list[tuple[np.ndarray, float]] = []
    while order:
      best_box, best_score = order.pop(0)
      kept.append((best_box, best_score))
      order = [
        (box, score) for box, score in order
        if self._bbox_iou(best_box, box) < self.iou_threshold
      ]
    return kept

  @staticmethod
  def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_min = max(float(box_a[0]), float(box_b[0]))
    y_min = max(float(box_a[1]), float(box_b[1]))
    x_max = min(float(box_a[2]), float(box_b[2]))
    y_max = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x_max - x_min)
    inter_h = max(0.0, y_max - y_min)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
      return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = max(area_a + area_b - intersection, 1e-6)
    return intersection / union


class SnpeYoloDetector(YoloDetectorBase):
  def __init__(self, runtime_name: str):
    if SNPEModel is None or Runtime is None or CLContext is None:
      raise BackendError("SNPE model runner extensions are not available")

    self.runtime_name = runtime_name.lower()
    if self.runtime_name not in {"gpu", "dsp"}:
      raise BackendError(f"unsupported objectd runtime '{runtime_name}'")

    self._init_detector_config()
    self._validate_export_runtime({"export_runtime": self.export_runtime})
    self._validate_dlc_compatibility(self.model_path)
    self.context = CLContext()
    runtime = Runtime.GPU if self.runtime_name == "gpu" else Runtime.DSP
    self.model = SNPEModel(str(self.model_path), self.output, runtime, False, self.context)
    self.model.addInput(self.input_name, self.input)
    self.backend_name = f"snpe_{self.runtime_name}"
    self.ready = True

  def _execute_model(self) -> None:
    self.model.execute()

  @staticmethod
  def _validate_export_runtime(metadata: dict) -> None:
    export_runtime = str(metadata.get("export_runtime", "SNPE_DLC")).upper()
    if export_runtime != "SNPE_DLC":
      raise BackendError(
        f"objectd model export_runtime '{export_runtime}' is not supported by the SNPE backend"
      )

  @staticmethod
  def _validate_dlc_compatibility(model_path: Path) -> None:
    try:
      with zipfile.ZipFile(model_path) as dlc:
        metadata_name = next((name for name in dlc.namelist() if name.startswith("dlc.metadata")), None)
        if metadata_name is None:
          return
        dlc_metadata = SnpeYoloDetector._parse_dlc_metadata(dlc.read(metadata_name))
    except (OSError, zipfile.BadZipFile, ValueError) as err:
      raise BackendError(f"objectd SNPE DLC metadata is unreadable: {err}") from err

    SnpeYoloDetector._reject_unsupported_converter(dlc_metadata)
    for generation in dlc_metadata.get("dlcGenerationInfo", []):
      command = generation.get("converterCommand", {})
      SnpeYoloDetector._reject_unsupported_converter(command)
    for generation in dlc_metadata.get("dlc-generation-info", []):
      command = generation.get("converter-command", {})
      SnpeYoloDetector._reject_unsupported_converter(command)

  @staticmethod
  def _parse_dlc_metadata(raw_metadata: bytes) -> dict:
    text = raw_metadata.decode("utf-8")
    try:
      return json.loads(text)
    except json.JSONDecodeError:
      metadata: dict[str, str] = {}
      for line in text.splitlines():
        if "=" not in line:
          continue
        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()
      if not metadata:
        raise ValueError("metadata is neither JSON nor key=value text")
      return metadata

  @staticmethod
  def _reject_unsupported_converter(command: dict) -> None:
    converter_version = str(command.get("converterVersion", command.get("converter-version", "")))
    major_version = converter_version.split(".", 1)[0]
    if major_version.isdigit() and int(major_version) >= 2:
      raise BackendError(
        "objectd SNPE DLC was produced by QAIRT/SNPE converter "
        f"{converter_version}, but the bundled tici SNPE runtime is 1.61.x and loops on model format 4.x"
      )


class QnnNetRunYoloDetector(YoloDetectorBase):
  def __init__(self):
    self._init_detector_config()
    self._validate_export_runtime({"export_runtime": self.export_runtime})
    self.qnn_net_run = self._resolve_executable(os.getenv("OBJECTD_QNN_NET_RUN", "qnn-net-run"))
    self.qnn_backend = os.getenv("OBJECTD_QNN_BACKEND", "libQnnHtp.so")
    self.qnn_model_dlc_lib = os.getenv("OBJECTD_QNN_MODEL_DLC_LIB", "libQnnModelDlc.so")
    self.timeout = float(os.getenv("OBJECTD_QNN_TIMEOUT", "5.0"))
    self._workdir = tempfile.TemporaryDirectory(prefix="objectd-qnn-")
    self.backend_name = "qnn_net_run"
    self.ready = True

  def _execute_model(self) -> None:
    workdir = Path(self._workdir.name)
    input_path = workdir / "image.raw"
    input_list_path = workdir / "input_list.txt"
    output_dir = workdir / "output"
    if output_dir.exists():
      shutil.rmtree(output_dir)
    output_dir.mkdir()

    self._write_qnn_input(input_path)
    input_list_path.write_text(str(input_path) + "\n", encoding="utf-8")

    cmd = self._build_qnn_command(input_list_path, output_dir)
    try:
      completed = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                                 timeout=self.timeout, check=False)
    except subprocess.TimeoutExpired as err:
      raise BackendError(f"qnn-net-run timed out after {self.timeout:.1f}s") from err

    if completed.returncode != 0:
      stderr = completed.stderr.strip() or completed.stdout.strip()
      raise BackendError(f"qnn-net-run failed with code {completed.returncode}: {stderr[-500:]}")

    self.output[:] = self._read_qnn_output(output_dir)

  def _build_qnn_command(self, input_list_path: Path, output_dir: Path) -> list[str]:
    cmd = [
      self.qnn_net_run,
      "--backend", self.qnn_backend,
      "--model", self.qnn_model_dlc_lib,
      "--dlc_path", str(self.model_path),
      "--input_list", str(input_list_path),
      "--output_dir", str(output_dir),
    ]
    if self._numpy_dtype(self.input_dtype, "input") != np.dtype(np.float32):
      cmd.append("--use_native_input_files")
    if self._numpy_dtype(self.output_dtype, "output") != np.dtype(np.float32):
      cmd.append("--use_native_output_files")
    log_level = os.getenv("OBJECTD_QNN_LOG_LEVEL")
    if log_level:
      cmd += ["--log_level", log_level]
    return cmd

  def _write_qnn_input(self, input_path: Path) -> None:
    if self.input_dtype in {"float", "float32"}:
      self.input.astype(np.float32, copy=False).tofile(input_path)
      return
    if self.input_dtype == "uint8":
      if self.input_scale <= 0.0:
        raise BackendError("qnn uint8 input_scale must be positive")
      quantized = np.clip(
        np.rint(self.input.astype(np.float64) / self.input_scale + self.input_zero_point),
        0,
        np.iinfo(np.uint8).max,
      ).astype(np.uint8)
      quantized.tofile(input_path)
      return
    raise BackendError(f"unsupported qnn input dtype '{self.input_dtype}'")

  def _read_qnn_output(self, output_dir: Path) -> np.ndarray:
    output_dtype = self._numpy_dtype(self.output_dtype, "output")
    expected_bytes = self.expected_output_size * output_dtype.itemsize
    raw_outputs = sorted(output_dir.rglob("*.raw"))
    for raw_output in raw_outputs:
      if raw_output.stat().st_size == expected_bytes:
        raw = np.fromfile(raw_output, dtype=output_dtype, count=self.expected_output_size)
        if output_dtype == np.dtype(np.float32):
          return raw.astype(np.float32, copy=False)
        return (raw.astype(np.float32) - self.output_zero_point) * self.output_scale
    sizes = {str(path): path.stat().st_size for path in raw_outputs}
    raise BackendError(f"qnn-net-run did not produce expected {expected_bytes}-byte output; saw {sizes}")

  @staticmethod
  def _numpy_dtype(dtype_name: str, field_name: str) -> np.dtype:
    normalized = dtype_name.lower()
    if normalized in {"float", "float32"}:
      return np.dtype(np.float32)
    if normalized == "uint8":
      return np.dtype(np.uint8)
    raise BackendError(f"unsupported qnn {field_name} dtype '{dtype_name}'")

  @staticmethod
  def _validate_export_runtime(metadata: dict) -> None:
    export_runtime = str(metadata.get("export_runtime", "")).upper()
    if export_runtime != "QNN_DLC":
      raise BackendError(
        f"objectd model export_runtime '{export_runtime or 'UNKNOWN'}' is not supported by the qnn_net_run backend"
      )

  @staticmethod
  def _resolve_executable(name: str) -> str:
    path = Path(name)
    if path.is_file():
      return str(path)
    resolved = shutil.which(name)
    if resolved is not None:
      return resolved
    raise BackendError(f"qnn-net-run executable not found at '{name}'")


class OrtQnnYoloDetector(YoloDetectorBase):
  _EP_REGISTERED = False

  def __init__(self):
    self._add_qnn_python_path()
    self._init_detector_config("model.onnx")
    self._validate_export_runtime({"export_runtime": self.export_runtime})
    try:
      import onnxruntime as ort
      import onnxruntime_qnn as qnn_ep
    except ModuleNotFoundError as err:
      raise BackendError(
        "onnxruntime-qnn dependencies are not available; install them under "
        f"{self._default_qnn_python_path()}"
      ) from err

    self.ort = ort
    self.qnn_ep = qnn_ep
    self._set_qnn_adsp_library_path(Path(qnn_ep.get_qnn_htp_path()).parent)
    self._register_qnn_ep()
    selected_devices = [device for device in ort.get_ep_devices() if device.ep_name == "QNNExecutionProvider"]
    if not selected_devices:
      raise BackendError("QNNExecutionProvider registered but no QNN EP device was discovered")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {"backend_path": qnn_ep.get_qnn_htp_path()}
    options.add_provider_for_devices(selected_devices, provider_options)
    self.session = ort.InferenceSession(str(self.model_path), sess_options=options)
    self.backend_name = "ort_qnn"
    self.ready = True

  def _execute_model(self) -> None:
    if self.input_layout == "NHWC":
      model_input = self.input.reshape(1, self.input_height, self.input_width, -1)
    elif self.input_layout == "NCHW":
      model_input = self.input.reshape(1, -1, self.input_height, self.input_width)
    else:
      raise BackendError(f"unsupported input layout '{self.input_layout}'")
    result = self.session.run([self.output_name], {self.input_name: model_input})[0]
    flat_output = np.asarray(result, dtype=np.float32).reshape(-1)
    if flat_output.size != self.expected_output_size:
      raise BackendError(
        f"ort_qnn output size mismatch: expected {self.expected_output_size}, got {flat_output.size}"
      )
    self.output[:] = flat_output

  @classmethod
  def _register_qnn_ep(cls) -> None:
    if cls._EP_REGISTERED:
      return
    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep
    try:
      ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    except Exception as err:
      if "already registered" not in str(err).lower():
        raise
    cls._EP_REGISTERED = True

  @staticmethod
  def _validate_export_runtime(metadata: dict) -> None:
    export_runtime = str(metadata.get("export_runtime", "")).upper()
    if export_runtime != "PRECOMPILED_QNN_ONNX":
      raise BackendError(
        f"objectd model export_runtime '{export_runtime or 'UNKNOWN'}' is not supported by the ort_qnn backend"
      )

  @classmethod
  def _add_qnn_python_path(cls) -> None:
    qnn_python_path = Path(os.getenv("OBJECTD_QNN_PYTHONPATH", cls._default_qnn_python_path()))
    if qnn_python_path.is_dir():
      sys.path.insert(0, str(qnn_python_path))

  @staticmethod
  def _default_qnn_python_path() -> Path:
    return DEFAULT_MODEL_DIR.parents[0] / "python"

  @staticmethod
  def _set_qnn_adsp_library_path(qnn_lib_dir: Path) -> None:
    existing = [
      path for path in os.getenv("ADSP_LIBRARY_PATH", "").split(":")
      if path and Path(path) != qnn_lib_dir
    ]
    qnn_paths = [str(qnn_lib_dir), "/usr/lib/rfsa/adsp", "/dsp"]
    os.environ["ADSP_LIBRARY_PATH"] = ":".join(qnn_paths + existing)


class TinygradOnnxYoloDetector(YoloDetectorBase):
  def __init__(self):
    self._init_detector_config("model.onnx")
    self._validate_export_runtime({"export_runtime": self.export_runtime})
    self._add_tinygrad_python_path()
    self.tinygrad_device = self._configure_tinygrad_device()
    self.backend_name = f"tinygrad_onnx:{self.tinygrad_device.lower()}"
    self._warmup_runs = int(os.getenv("OBJECTD_TINYGRAD_WARMUP_RUNS", str(DEFAULT_TINYGRAD_WARMUP_RUNS)))
    self._infer_timeout = float(os.getenv("OBJECTD_TINYGRAD_INFER_TIMEOUT", "5.0"))
    self._warmup_lock = threading.Lock()
    self._worker_thread: threading.Thread | None = None
    self._request_queue: queue.Queue = queue.Queue(maxsize=1)
    self.ready = False

  def _execute_model(self) -> None:
    if not self.ready:
      raise BackendError("tinygrad_onnx inference requested before warmup completed")
    response_queue: queue.Queue = queue.Queue(maxsize=1)
    try:
      self._request_queue.put((self.input.copy(), response_queue), timeout=0.1)
      flat_output, error = response_queue.get(timeout=self._infer_timeout)
    except queue.Full as err:
      raise BackendError("tinygrad_onnx worker is busy") from err
    except queue.Empty as err:
      raise BackendError(f"tinygrad_onnx inference timed out after {self._infer_timeout:.1f}s") from err
    if error:
      raise BackendError(error)
    if flat_output.size != self.expected_output_size:
      raise BackendError(
        f"tinygrad_onnx output size mismatch: expected {self.expected_output_size}, got {flat_output.size}"
    )
    self.output[:] = flat_output

  def _execute_model_on_worker(self, Tensor, jit_run, model_input_flat: np.ndarray) -> np.ndarray:
    if self.input_layout == "NHWC":
      model_input = model_input_flat.reshape(1, self.input_height, self.input_width, -1)
    elif self.input_layout == "NCHW":
      model_input = model_input_flat.reshape(1, -1, self.input_height, self.input_width)
    else:
      raise BackendError(f"unsupported input layout '{self.input_layout}'")
    tensor_input = Tensor(model_input, device=self.tinygrad_device).realize()
    result = jit_run(tensor_input).realize()
    flat_output = np.asarray(result.numpy(), dtype=np.float32).reshape(-1)
    return flat_output

  def start_warmup(self) -> None:
    if self.ready or self.last_error:
      return
    with self._warmup_lock:
      if self.ready or self.warming or self._worker_thread is not None:
        return
      self.warming = True
      self.warmup_stage = "queued"
      self._worker_thread = threading.Thread(
        target=self._run_worker,
        name="objectd-tinygrad-worker",
        daemon=True,
      )
      self._worker_thread.start()

  def _run_worker(self) -> None:
    self._configure_worker_scheduling()
    try:
      Tensor, jit_run = self._build_worker_runner()
      self._run_warmup_blocking(Tensor, jit_run)
    except Exception as err:
      self.mark_failed(str(err))
      return
    self.ready = True
    self.warming = False
    self.warmup_stage = ""
    while True:
      model_input_flat, response_queue = self._request_queue.get()
      try:
        response_queue.put((self._execute_model_on_worker(Tensor, jit_run, model_input_flat), ""))
      except Exception as err:
        response_queue.put((np.array([], dtype=np.float32), str(err)))

  @staticmethod
  def _configure_worker_scheduling() -> None:
    if sys.platform != "linux":
      return
    try:
      if hasattr(os, "SCHED_IDLE"):
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
      elif hasattr(os, "SCHED_BATCH"):
        os.sched_setscheduler(0, os.SCHED_BATCH, os.sched_param(0))
    except OSError:
      pass
    try:
      affinity = TinygradOnnxYoloDetector._parse_cpu_affinity(os.getenv("OBJECTD_CPU_AFFINITY", ""))
      if affinity:
        os.sched_setaffinity(0, affinity)
    except (AttributeError, OSError, ValueError):
      pass

  @staticmethod
  def _parse_cpu_affinity(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
      part = part.strip()
      if not part:
        continue
      if "-" in part:
        start, end = part.split("-", 1)
        cpus.update(range(int(start), int(end) + 1))
      else:
        cpus.add(int(part))
    return cpus

  def _build_worker_runner(self):
    try:
      from tinygrad import Tensor, TinyJit
      from tinygrad.nn.onnx import OnnxRunner
    except ModuleNotFoundError as err:
      raise BackendError(
        "tinygrad dependencies are not available; set OBJECTD_TINYGRAD_PYTHONPATH"
      ) from err

    runner = OnnxRunner(self.model_path)
    if self.input_name not in runner.graph_inputs:
      raise BackendError(f"tinygrad_onnx input '{self.input_name}' not found in model")
    if self.output_name not in runner.graph_outputs:
      raise BackendError(f"tinygrad_onnx output '{self.output_name}' not found in model")

    def run(model_input):
      return runner({self.input_name: model_input})[self.output_name].contiguous()

    return Tensor, TinyJit(run)

  def _run_warmup_blocking(self, Tensor, jit_run) -> None:
    original_input = self.input.copy()
    try:
      self.input.fill(0.0)
      for run_idx in range(self._warmup_runs):
        self.warmup_stage = f"run{run_idx + 1}of{self._warmup_runs}"
        self._execute_model_on_worker(Tensor, jit_run, self.input)
    finally:
      self.input[:] = original_input

  @staticmethod
  def _validate_export_runtime(metadata: dict) -> None:
    export_runtime = str(metadata.get("export_runtime", "")).upper()
    if export_runtime != "ONNX":
      raise BackendError(
        f"objectd model export_runtime '{export_runtime or 'UNKNOWN'}' is not supported by the tinygrad_onnx backend"
      )

  @staticmethod
  def _configure_tinygrad_device() -> str | None:
    configured_device = os.getenv("OBJECTD_TINYGRAD_DEVICE")
    if configured_device:
      os.environ.setdefault("DEV", configured_device)
      return configured_device
    if Path("/dev/kgsl-3d0").exists():
      os.environ.setdefault("DEV", "QCOM")
      return "QCOM"
    if OrtCpuYoloDetector._allow_cpu_inference():
      os.environ.setdefault("DEV", "CPU")
      return "CPU"
    raise BackendError(
      "tinygrad_onnx requires an accelerator device; set OBJECTD_TINYGRAD_DEVICE or "
      "OBJECTD_ALLOW_CPU_INFERENCE=1 for explicit CPU measurement"
    )

  @staticmethod
  def _add_tinygrad_python_path() -> None:
    python_path = Path(os.getenv("OBJECTD_TINYGRAD_PYTHONPATH", DEFAULT_MODEL_DIR.parents[2] / "tinygrad_repo"))
    if python_path.is_dir():
      sys.path.insert(0, str(python_path))


class OrtCpuYoloDetector(YoloDetectorBase):
  def __init__(self):
    self._init_detector_config("model.onnx")
    self._validate_export_runtime({"export_runtime": self.export_runtime})
    if not self._allow_cpu_inference():
      raise BackendError(
        "onnx_cpu is measurement-only; set OBJECTD_ALLOW_CPU_INFERENCE=1 to run it explicitly"
      )
    self._add_onnx_python_path()
    try:
      import onnxruntime as ort
    except ModuleNotFoundError as err:
      raise BackendError(
        "onnxruntime dependencies are not available; install them or set OBJECTD_ONNX_PYTHONPATH"
      ) from err

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(
      str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    self.backend_name = "onnx_cpu:measurement"
    self.ready = True

  def _execute_model(self) -> None:
    if self.input_layout == "NHWC":
      model_input = self.input.reshape(1, self.input_height, self.input_width, -1)
    elif self.input_layout == "NCHW":
      model_input = self.input.reshape(1, -1, self.input_height, self.input_width)
    else:
      raise BackendError(f"unsupported input layout '{self.input_layout}'")
    result = self.session.run([self.output_name], {self.input_name: model_input})[0]
    flat_output = np.asarray(result, dtype=np.float32).reshape(-1)
    if flat_output.size != self.expected_output_size:
      raise BackendError(
        f"onnx_cpu output size mismatch: expected {self.expected_output_size}, got {flat_output.size}"
      )
    self.output[:] = flat_output

  @staticmethod
  def _validate_export_runtime(metadata: dict) -> None:
    export_runtime = str(metadata.get("export_runtime", "")).upper()
    if export_runtime != "ONNX":
      raise BackendError(
        f"objectd model export_runtime '{export_runtime or 'UNKNOWN'}' is not supported by the onnx_cpu backend"
      )

  @staticmethod
  def _allow_cpu_inference() -> bool:
    return os.getenv("OBJECTD_ALLOW_CPU_INFERENCE", "").strip().lower() in {"1", "true", "yes"}

  @staticmethod
  def _add_onnx_python_path() -> None:
    python_path = Path(os.getenv("OBJECTD_ONNX_PYTHONPATH", DEFAULT_MODEL_DIR.parents[0] / "python"))
    if python_path.is_dir():
      sys.path.insert(0, str(python_path))


def _read_default_export_runtime() -> str:
  model_dir = Path(os.getenv("OBJECTD_MODEL_DIR", DEFAULT_MODEL_DIR))
  metadata_path = Path(os.getenv("OBJECTD_MODEL_METADATA", model_dir / "metadata.json"))
  metadata = YoloDetectorBase._load_metadata(metadata_path)
  return YoloDetectorBase._normalize_export_runtime(metadata)


def build_detector_backend():
  backend_name = os.getenv("OBJECTD_BACKEND", DEFAULT_BACKEND).lower()
  if backend_name == "auto":
    export_runtime = _read_default_export_runtime()
    if export_runtime == "PRECOMPILED_QNN_ONNX":
      return OrtQnnYoloDetector()
    if export_runtime == "ONNX":
      return TinygradOnnxYoloDetector()
    if export_runtime == "QNN_DLC":
      return QnnNetRunYoloDetector()
    return SnpeYoloDetector("gpu")
  if backend_name == "null":
    return NullDetectorBackend("forced")
  if backend_name == "snpe_gpu":
    return SnpeYoloDetector("gpu")
  if backend_name == "snpe_dsp":
    return SnpeYoloDetector("dsp")
  if backend_name in {"ort_qnn", "qnn_ort"}:
    return OrtQnnYoloDetector()
  if backend_name in {"onnx_cpu", "ort_cpu"}:
    return OrtCpuYoloDetector()
  if backend_name in {"tinygrad", "tinygrad_onnx"}:
    return TinygradOnnxYoloDetector()
  if backend_name in {"qnn", "qnn_net_run"}:
    return OrtQnnYoloDetector() if backend_name == "qnn" else QnnNetRunYoloDetector()
  raise BackendError(f"unsupported OBJECTD_BACKEND '{backend_name}'")
