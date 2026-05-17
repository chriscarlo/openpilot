#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from openpilot.sunnypilot.objectd.backend import COCO_80_LABELS, DEFAULT_HAZARD_LABELS, DEFAULT_MODEL_DIR

HF_REPO_ID = "qualcomm/YOLOv11-Detection"
SUPPORTED_EXPORT_RUNTIMES = ("QNN_DLC", "SNPE_DLC")
SUPPORTED_INPUT_LAYOUTS = ("NCHW", "NHWC")


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def build_metadata(model_sha256: str, export_runtime: str = "QNN_DLC", input_layout: str | None = None) -> dict:
  export_runtime = export_runtime.upper()
  if export_runtime not in SUPPORTED_EXPORT_RUNTIMES:
    raise ValueError(f"unsupported export_runtime '{export_runtime}'")
  if input_layout is None:
    input_layout = "NHWC" if export_runtime == "QNN_DLC" else "NCHW"
  input_layout = input_layout.upper()
  if input_layout not in SUPPORTED_INPUT_LAYOUTS:
    raise ValueError(f"unsupported input_layout '{input_layout}'")

  return {
    "source_repo": HF_REPO_ID,
    "source_checkpoint": "YOLO11-N / yolo11n.pt",
    "export_runtime": export_runtime,
    "export_notes": "Export with include_postprocessing=False and split_output=False so output is [1,84,8400].",
    "model_sha256": model_sha256,
    "input_name": "image",
    "input_width": 640,
    "input_height": 640,
    "input_channels": 3,
    "input_layout": input_layout,
    "prediction_count": 8400,
    "attributes": 84,
    "prediction_layout": "attributes_first",
    "has_objectness": False,
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "labels": COCO_80_LABELS,
    "hazard_labels": sorted(DEFAULT_HAZARD_LABELS),
  }


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Install objectd YOLO11-N DLC assets into the local object-hazard cache."
  )
  parser.add_argument("--model-dlc", type=Path, required=True,
                      help="Path to a YOLO11-N DLC file.")
  parser.add_argument("--export-runtime", choices=SUPPORTED_EXPORT_RUNTIMES, default="QNN_DLC",
                      help="Runtime that produced this DLC. Default: QNN_DLC.")
  parser.add_argument("--input-layout", choices=SUPPORTED_INPUT_LAYOUTS,
                      help="Input tensor layout expected by the DLC. Default: NHWC for QNN_DLC, NCHW for SNPE_DLC.")
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR,
                      help=f"Destination asset directory. Default: {DEFAULT_MODEL_DIR}")
  args = parser.parse_args()

  if not args.model_dlc.is_file():
    raise FileNotFoundError(args.model_dlc)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  model_dst = args.output_dir / "model.dlc"
  metadata_dst = args.output_dir / "metadata.json"
  shutil.copy2(args.model_dlc, model_dst)

  metadata = build_metadata(sha256_file(model_dst), args.export_runtime, args.input_layout)
  metadata_dst.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

  print(f"Installed {model_dst}")
  print(f"Wrote {metadata_dst}")
  print(f"Recorded export_runtime={metadata['export_runtime']} input_layout={metadata['input_layout']}")
  print(f"Note: {HF_REPO_ID} does not distribute pre-exported DLC assets due to licensing.")
  print("Export a compatible DLC, then run this helper with --model-dlc and matching metadata.")


if __name__ == "__main__":
  main()
