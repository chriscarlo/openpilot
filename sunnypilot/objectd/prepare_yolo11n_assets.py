#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from openpilot.sunnypilot.objectd.backend import COCO_80_LABELS, DEFAULT_HAZARD_LABELS, DEFAULT_MODEL_DIR

HF_REPO_ID = "qualcomm/YOLOv11-Detection"
SUPPORTED_EXPORT_RUNTIMES = ("QNN_DLC", "SNPE_DLC", "PRECOMPILED_QNN_ONNX")
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
    input_layout = "NCHW" if export_runtime == "SNPE_DLC" else "NHWC"
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
    "output_name": "detector_output",
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
    description="Install objectd YOLO11-N accelerator assets into the local object-hazard cache."
  )
  model_arg = parser.add_mutually_exclusive_group(required=True)
  model_arg.add_argument("--model-dlc", type=Path,
                         help="Path to a YOLO11-N DLC file.")
  model_arg.add_argument("--model-onnx", type=Path,
                         help="Path to a YOLO11-N precompiled QNN ONNX file.")
  parser.add_argument("--export-runtime", choices=SUPPORTED_EXPORT_RUNTIMES,
                      help="Runtime that produced this asset. Default: QNN_DLC for --model-dlc, PRECOMPILED_QNN_ONNX for --model-onnx.")
  parser.add_argument("--input-layout", choices=SUPPORTED_INPUT_LAYOUTS,
                      help="Input tensor layout expected by the model. Default: NHWC for QNN assets, NCHW for SNPE_DLC.")
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR,
                      help=f"Destination asset directory. Default: {DEFAULT_MODEL_DIR}")
  args = parser.parse_args()

  model_src = args.model_dlc if args.model_dlc is not None else args.model_onnx
  if not model_src.is_file():
    raise FileNotFoundError(model_src)
  export_runtime = args.export_runtime
  if export_runtime is None:
    export_runtime = "PRECOMPILED_QNN_ONNX" if args.model_onnx is not None else "QNN_DLC"

  args.output_dir.mkdir(parents=True, exist_ok=True)
  model_dst = args.output_dir / ("model.onnx" if args.model_onnx is not None else "model.dlc")
  metadata_dst = args.output_dir / "metadata.json"
  shutil.copy2(model_src, model_dst)
  if args.model_onnx is not None:
    for context_bin in model_src.parent.glob("*.bin"):
      shutil.copy2(context_bin, args.output_dir / context_bin.name)

  metadata = build_metadata(sha256_file(model_dst), export_runtime, args.input_layout)
  metadata_dst.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

  print(f"Installed {model_dst}")
  print(f"Wrote {metadata_dst}")
  print(f"Recorded export_runtime={metadata['export_runtime']} input_layout={metadata['input_layout']}")
  print(f"Note: {HF_REPO_ID} does not distribute pre-exported DLC assets due to licensing.")
  print("Export a compatible asset, then run this helper with --model-dlc or --model-onnx and matching metadata.")


if __name__ == "__main__":
  main()
