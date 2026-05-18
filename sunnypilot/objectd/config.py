from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_MODEL_DIR = REPO_ROOT / ".cache" / "objectd" / "yolov8n_tinygrad_256_install"
PRIMARY_MODEL_PATH = PRIMARY_MODEL_DIR / "model.onnx"
PRIMARY_METADATA_PATH = PRIMARY_MODEL_DIR / "metadata.json"

DEFAULT_BACKEND = "tinygrad_onnx"
DEFAULT_TINYGRAD_DEVICE = "QCOM"
DEFAULT_DETECTOR_HZ = 2.0
DEFAULT_TINYGRAD_WARMUP_RUNS = 3
DEFAULT_DEBUG_DETECTION_LIMIT = 8


def _env_float(name: str, default: float) -> float:
  try:
    return float(os.getenv(name, str(default)))
  except ValueError:
    return default


def _env_int(name: str, default: int) -> int:
  try:
    return int(os.getenv(name, str(default)))
  except ValueError:
    return default


@dataclass(frozen=True)
class ObjectdRuntimeConfig:
  backend: str = DEFAULT_BACKEND
  model_dir: Path = PRIMARY_MODEL_DIR
  model_path: Path = PRIMARY_MODEL_PATH
  metadata_path: Path = PRIMARY_METADATA_PATH
  tinygrad_device: str = DEFAULT_TINYGRAD_DEVICE
  detector_hz: float = DEFAULT_DETECTOR_HZ
  tinygrad_warmup_runs: int = DEFAULT_TINYGRAD_WARMUP_RUNS
  debug_detection_limit: int = DEFAULT_DEBUG_DETECTION_LIMIT

  @classmethod
  def from_env(cls) -> "ObjectdRuntimeConfig":
    model_dir = Path(os.getenv("OBJECTD_MODEL_DIR", PRIMARY_MODEL_DIR))
    return cls(
      backend=os.getenv("OBJECTD_BACKEND", DEFAULT_BACKEND),
      model_dir=model_dir,
      model_path=Path(os.getenv("OBJECTD_MODEL_PATH", model_dir / "model.onnx")),
      metadata_path=Path(os.getenv("OBJECTD_MODEL_METADATA", model_dir / "metadata.json")),
      tinygrad_device=os.getenv("OBJECTD_TINYGRAD_DEVICE", DEFAULT_TINYGRAD_DEVICE),
      detector_hz=max(1.0, min(2.0, _env_float("OBJECTD_DETECTOR_HZ", DEFAULT_DETECTOR_HZ))),
      tinygrad_warmup_runs=max(0, _env_int("OBJECTD_TINYGRAD_WARMUP_RUNS", DEFAULT_TINYGRAD_WARMUP_RUNS)),
      debug_detection_limit=max(0, _env_int("OBJECTD_DEBUG_DETECTION_LIMIT", DEFAULT_DEBUG_DETECTION_LIMIT)),
    )

  def apply_environment_defaults(self) -> None:
    os.environ.setdefault("OBJECTD_BACKEND", self.backend)
    os.environ.setdefault("OBJECTD_MODEL_DIR", str(self.model_dir))
    os.environ.setdefault("OBJECTD_MODEL_PATH", str(self.model_path))
    os.environ.setdefault("OBJECTD_MODEL_METADATA", str(self.metadata_path))
    os.environ.setdefault("OBJECTD_TINYGRAD_DEVICE", self.tinygrad_device)
    os.environ.setdefault("OBJECTD_DETECTOR_HZ", f"{self.detector_hz:.1f}")
    os.environ.setdefault("OBJECTD_TINYGRAD_WARMUP_RUNS", str(self.tinygrad_warmup_runs))
    os.environ.setdefault("OBJECTD_DEBUG_DETECTION_LIMIT", str(self.debug_detection_limit))
