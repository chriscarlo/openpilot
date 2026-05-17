from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

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

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / ".cache" / "objectd" / "yolo11n"
DEFAULT_HAZARD_LABELS = {
  "person", "bicycle", "dog", "horse", "sheep", "cow",
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

  def infer(self, _buf) -> list[Detection]:
    return []


class YoloDetectorBase:
  def _init_detector_config(self, default_model_name: str = "model.dlc") -> None:
    if cv2 is None:
      raise BackendError("opencv-python-headless is required for objectd preprocessing")

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
    self.prediction_count = int(metadata["prediction_count"])
    self.attributes = int(metadata["attributes"])
    self.prediction_layout = str(metadata.get("prediction_layout", "attributes_first"))
    self.output_name = str(metadata.get("output_name", "detector_output"))
    self.has_objectness = bool(metadata.get("has_objectness", False))
    self.confidence_threshold = float(metadata.get("confidence_threshold", 0.25))
    self.iou_threshold = float(metadata.get("iou_threshold", 0.45))
    self.labels = list(metadata.get("labels", COCO_80_LABELS))
    self.hazard_labels = set(metadata.get("hazard_labels", sorted(DEFAULT_HAZARD_LABELS)))
    self.expected_output_size = self.prediction_count * self.attributes

    input_channels = int(metadata.get("input_channels", 3))
    input_size = self.input_width * self.input_height * input_channels

    self.output = np.zeros(self.expected_output_size, dtype=np.float32)
    self.input = np.zeros(input_size, dtype=np.float32)

  def infer(self, buf) -> list[Detection]:
    rgb = self._prepare_input(buf)
    self._execute_model()
    return self._decode_predictions(rgb.shape[1], rgb.shape[0])

  def _execute_model(self) -> None:
    raise NotImplementedError

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
  def _visionbuf_to_rgb(buf) -> np.ndarray:
    frame = np.frombuffer(buf.data, dtype=np.uint8).reshape((-1, buf.stride))
    yuv = frame[:buf.height + buf.height // 2, :buf.width]
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
    if export_runtime.startswith("QNN") or export_runtime == "PRECOMPILED_QNN_ONNX":
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

    self.input.astype(np.float32, copy=False).tofile(input_path)
    input_list_path.write_text(str(input_path) + "\n", encoding="utf-8")

    cmd = [
      self.qnn_net_run,
      "--backend", self.qnn_backend,
      "--model", self.qnn_model_dlc_lib,
      "--dlc_path", str(self.model_path),
      "--input_list", str(input_list_path),
      "--output_dir", str(output_dir),
    ]
    try:
      completed = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                                 timeout=self.timeout, check=False)
    except subprocess.TimeoutExpired as err:
      raise BackendError(f"qnn-net-run timed out after {self.timeout:.1f}s") from err

    if completed.returncode != 0:
      stderr = completed.stderr.strip() or completed.stdout.strip()
      raise BackendError(f"qnn-net-run failed with code {completed.returncode}: {stderr[-500:]}")

    self.output[:] = self._read_qnn_output(output_dir)

  def _read_qnn_output(self, output_dir: Path) -> np.ndarray:
    expected_bytes = self.expected_output_size * np.dtype(np.float32).itemsize
    raw_outputs = sorted(output_dir.rglob("*.raw"))
    for raw_output in raw_outputs:
      if raw_output.stat().st_size == expected_bytes:
        return np.fromfile(raw_output, dtype=np.float32, count=self.expected_output_size)
    sizes = {str(path): path.stat().st_size for path in raw_outputs}
    raise BackendError(f"qnn-net-run did not produce expected {expected_bytes}-byte output; saw {sizes}")

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


def _read_default_export_runtime() -> str:
  model_dir = Path(os.getenv("OBJECTD_MODEL_DIR", DEFAULT_MODEL_DIR))
  metadata_path = Path(os.getenv("OBJECTD_MODEL_METADATA", model_dir / "metadata.json"))
  metadata = YoloDetectorBase._load_metadata(metadata_path)
  return YoloDetectorBase._normalize_export_runtime(metadata)


def build_detector_backend():
  backend_name = os.getenv("OBJECTD_BACKEND", "auto").lower()
  if backend_name == "auto":
    export_runtime = _read_default_export_runtime()
    if export_runtime == "PRECOMPILED_QNN_ONNX":
      return OrtQnnYoloDetector()
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
  if backend_name in {"qnn", "qnn_net_run"}:
    return OrtQnnYoloDetector() if backend_name == "qnn" else QnnNetRunYoloDetector()
  raise BackendError(f"unsupported OBJECTD_BACKEND '{backend_name}'")
