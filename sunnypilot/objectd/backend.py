from __future__ import annotations

import hashlib
import json
import os
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


class SnpeYoloDetector:
  def __init__(self, runtime_name: str):
    if SNPEModel is None or Runtime is None or CLContext is None:
      raise BackendError("SNPE model runner extensions are not available")
    if cv2 is None:
      raise BackendError("opencv-python-headless is required for objectd preprocessing")

    self.runtime_name = runtime_name.lower()
    if self.runtime_name not in {"gpu", "dsp"}:
      raise BackendError(f"unsupported objectd runtime '{runtime_name}'")

    self.model_dir = Path(os.getenv("OBJECTD_MODEL_DIR", DEFAULT_MODEL_DIR))
    self.model_path = Path(os.getenv("OBJECTD_MODEL_PATH", self.model_dir / "model.dlc"))
    self.metadata_path = Path(os.getenv("OBJECTD_MODEL_METADATA", self.model_dir / "metadata.json"))
    metadata = self._load_metadata(self.metadata_path)
    self._verify_sha256(self.model_path, metadata.get("model_sha256"))

    self.input_name = str(metadata["input_name"])
    self.input_width = int(metadata["input_width"])
    self.input_height = int(metadata["input_height"])
    self.input_layout = str(metadata.get("input_layout", "NCHW")).upper()
    self.prediction_count = int(metadata["prediction_count"])
    self.attributes = int(metadata["attributes"])
    self.prediction_layout = str(metadata.get("prediction_layout", "attributes_first"))
    self.has_objectness = bool(metadata.get("has_objectness", False))
    self.confidence_threshold = float(metadata.get("confidence_threshold", 0.25))
    self.iou_threshold = float(metadata.get("iou_threshold", 0.45))
    self.labels = list(metadata.get("labels", COCO_80_LABELS))
    self.hazard_labels = set(metadata.get("hazard_labels", sorted(DEFAULT_HAZARD_LABELS)))
    # Keep detector assets isolated from the shared SNPE runner assumptions by requiring a
    # flattened output tensor ([1, N] or [N]) whose length matches prediction_count * attributes.
    expected_output_size = self.prediction_count * self.attributes

    input_channels = int(metadata.get("input_channels", 3))
    input_size = self.input_width * self.input_height * input_channels

    self.output = np.zeros(expected_output_size, dtype=np.float32)
    self.input = np.zeros(input_size, dtype=np.float32)
    self.context = CLContext()
    runtime = Runtime.GPU if self.runtime_name == "gpu" else Runtime.DSP
    self.model = SNPEModel(str(self.model_path), self.output, runtime, False, self.context)
    self.model.addInput(self.input_name, self.input)
    self.backend_name = f"snpe_{self.runtime_name}"
    self.ready = True

  def infer(self, buf) -> list[Detection]:
    rgb = self._visionbuf_to_rgb(buf)
    resized = cv2.resize(rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    if self.input_layout == "NCHW":
      self.input[:] = normalized.transpose(2, 0, 1).reshape(-1)
    elif self.input_layout == "NHWC":
      self.input[:] = normalized.reshape(-1)
    else:
      raise BackendError(f"unsupported input layout '{self.input_layout}'")
    self.model.execute()
    return self._decode_predictions(rgb.shape[1], rgb.shape[0])

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

  @staticmethod
  def _visionbuf_to_rgb(buf) -> np.ndarray:
    frame = np.frombuffer(buf.data, dtype=np.uint8).reshape((-1, buf.stride))
    yuv = frame[:buf.height + buf.height // 2, :buf.width]
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)

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


def build_detector_backend():
  backend_name = os.getenv("OBJECTD_BACKEND", "snpe_gpu").lower()
  if backend_name == "null":
    return NullDetectorBackend("forced")
  if backend_name == "snpe_gpu":
    return SnpeYoloDetector("gpu")
  if backend_name == "snpe_dsp":
    return SnpeYoloDetector("dsp")
  raise BackendError(f"unsupported OBJECTD_BACKEND '{backend_name}'")
