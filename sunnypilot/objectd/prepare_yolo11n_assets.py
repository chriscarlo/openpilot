#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from openpilot.sunnypilot.objectd.backend import COCO_80_LABELS, DEFAULT_HAZARD_LABELS, DEFAULT_MODEL_DIR

def tinygrad_onnx_preset(
  model_family: str,
  source_repo: str,
  source_checkpoint: str,
  input_size: int,
  prediction_count: int,
) -> dict:
  return {
    "source_repo": source_repo,
    "source_checkpoint": source_checkpoint,
    "export_notes": (
      f"Export with Ultralytics ONNX opset 12, imgsz={input_size}, "
      f"simplify=False, and raw output [1,84,{prediction_count}]."
    ),
    "input_name": "images",
    "input_width": input_size,
    "input_height": input_size,
    "input_channels": 3,
    "output_name": "output0",
    "prediction_count": prediction_count,
    "attributes": 84,
    "prediction_layout": "attributes_first",
    "has_objectness": False,
    "labels": COCO_80_LABELS,
    "hazard_labels": sorted(DEFAULT_HAZARD_LABELS),
    "decoder_family": "yolo_anchor_free",
    "model_family": model_family,
  }


MODEL_PRESETS = {
  "yolo11n": {
    "source_repo": "qualcomm/YOLOv11-Detection",
    "source_checkpoint": "YOLO11-N / yolo11n.pt",
    "export_notes": "Export with include_postprocessing=False and split_output=False so output is [1,84,8400].",
    "input_name": "image",
    "input_width": 640,
    "input_height": 640,
    "input_channels": 3,
    "output_name": "detector_output",
    "prediction_count": 8400,
    "attributes": 84,
    "prediction_layout": "attributes_first",
    "has_objectness": False,
    "labels": COCO_80_LABELS,
    "hazard_labels": sorted(DEFAULT_HAZARD_LABELS),
    "decoder_family": "yolo_anchor_free",
  },
  "yolov8n": {
    "source_repo": "qualcomm/YOLOv8-Detection",
    "source_checkpoint": "YOLOv8-N",
    "export_notes": "Expected YOLOv8-N no-postprocessing detector output [1,84,8400]; verify names/shapes on the exported asset before install.",
    "input_name": "image",
    "input_width": 640,
    "input_height": 640,
    "input_channels": 3,
    "output_name": "detector_output",
    "prediction_count": 8400,
    "attributes": 84,
    "prediction_layout": "attributes_first",
    "has_objectness": False,
    "labels": COCO_80_LABELS,
    "hazard_labels": sorted(DEFAULT_HAZARD_LABELS),
    "decoder_family": "yolo_anchor_free",
  },
  "yolo11n_tinygrad_160": tinygrad_onnx_preset(
    "yolo11n", "ultralytics/yolo11", "YOLO11-N / yolo11n.pt", 160, 525
  ),
  "yolo11n_tinygrad_192": tinygrad_onnx_preset(
    "yolo11n", "ultralytics/yolo11", "YOLO11-N / yolo11n.pt", 192, 756
  ),
  "yolo11n_tinygrad_224": tinygrad_onnx_preset(
    "yolo11n", "ultralytics/yolo11", "YOLO11-N / yolo11n.pt", 224, 1029
  ),
  "yolo11n_tinygrad_256": tinygrad_onnx_preset(
    "yolo11n", "ultralytics/yolo11", "YOLO11-N / yolo11n.pt", 256, 1344
  ),
  "yolov8n_tinygrad_160": tinygrad_onnx_preset(
    "yolov8n", "ultralytics/yolov8", "YOLOv8-N / yolov8n.pt", 160, 525
  ),
  "yolov8n_tinygrad_256": tinygrad_onnx_preset(
    "yolov8n", "ultralytics/yolov8", "YOLOv8-N / yolov8n.pt", 256, 1344
  ),
}
SUPPORTED_EXPORT_RUNTIMES = ("QNN_DLC", "SNPE_DLC", "PRECOMPILED_QNN_ONNX", "ONNX")
SUPPORTED_INPUT_LAYOUTS = ("NCHW", "NHWC")
SUPPORTED_PREDICTION_LAYOUTS = ("attributes_first", "predictions_first")


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def parse_input_size(input_size: str) -> tuple[int, int]:
  try:
    width, height = input_size.lower().split("x", 1)
    return int(width), int(height)
  except ValueError as err:
    raise ValueError(f"input size must be WIDTHxHEIGHT, got '{input_size}'") from err


def load_labels(path: Path) -> list[str]:
  with path.open(encoding="utf-8") as f:
    labels = json.load(f)
  if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
    raise ValueError(f"labels file must contain a JSON string array: {path}")
  return labels


def build_metadata(
  model_sha256: str,
  export_runtime: str | None = None,
  input_layout: str | None = None,
  model_preset: str = "yolo11n",
  **overrides,
) -> dict:
  if export_runtime is None:
    export_runtime = "ONNX" if "_tinygrad_" in model_preset else "QNN_DLC"
  export_runtime = export_runtime.upper()
  if export_runtime not in SUPPORTED_EXPORT_RUNTIMES:
    raise ValueError(f"unsupported export_runtime '{export_runtime}'")
  if input_layout is None:
    input_layout = "NCHW" if export_runtime in {"SNPE_DLC", "ONNX"} else "NHWC"
  input_layout = input_layout.upper()
  if input_layout not in SUPPORTED_INPUT_LAYOUTS:
    raise ValueError(f"unsupported input_layout '{input_layout}'")

  if model_preset not in MODEL_PRESETS:
    raise ValueError(f"unsupported model_preset '{model_preset}'")

  metadata = dict(MODEL_PRESETS[model_preset])
  metadata.update({
    "export_runtime": export_runtime,
    "model_sha256": model_sha256,
    "input_layout": input_layout,
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
  })
  metadata.update({key: value for key, value in overrides.items() if value is not None})

  prediction_layout = str(metadata["prediction_layout"])
  if prediction_layout not in SUPPORTED_PREDICTION_LAYOUTS:
    raise ValueError(f"unsupported prediction_layout '{prediction_layout}'")
  if int(metadata["attributes"]) < 5:
    raise ValueError("attributes must include at least 4 box values and 1 class/objectness value")
  if int(metadata["prediction_count"]) <= 0:
    raise ValueError("prediction_count must be positive")
  return metadata


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Install objectd accelerator assets plus explicit detector metadata into the local object-hazard cache."
  )
  model_arg = parser.add_mutually_exclusive_group(required=True)
  model_arg.add_argument("--model-dlc", type=Path,
                         help="Path to a detector DLC file.")
  model_arg.add_argument("--model-onnx", type=Path,
                         help="Path to a detector precompiled QNN ONNX file.")
  parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="yolo11n",
                      help="Known metadata preset to start from. Default: yolo11n.")
  parser.add_argument("--export-runtime", choices=SUPPORTED_EXPORT_RUNTIMES,
                      help="Runtime that produced this asset. Default: QNN_DLC for --model-dlc, PRECOMPILED_QNN_ONNX for --model-onnx.")
  parser.add_argument("--input-layout", choices=SUPPORTED_INPUT_LAYOUTS,
                      help="Input tensor layout expected by the model. Default: NHWC for QNN assets, NCHW for SNPE_DLC.")
  parser.add_argument("--input-dtype",
                      help="Input tensor dtype, e.g. float32 or uint8.")
  parser.add_argument("--input-scale", type=float,
                      help="Input quantization scale for integer QNN assets.")
  parser.add_argument("--input-zero-point", type=int,
                      help="Input quantization zero point for integer QNN assets.")
  parser.add_argument("--input-size",
                      help="Input tensor size as WIDTHxHEIGHT, overriding the selected preset.")
  parser.add_argument("--input-name",
                      help="Input tensor name, overriding the selected preset.")
  parser.add_argument("--output-name",
                      help="Output tensor name, overriding the selected preset.")
  parser.add_argument("--output-dtype",
                      help="Output tensor dtype, e.g. float32 or uint8.")
  parser.add_argument("--output-scale", type=float,
                      help="Output quantization scale for integer QNN assets.")
  parser.add_argument("--output-zero-point", type=int,
                      help="Output quantization zero point for integer QNN assets.")
  parser.add_argument("--prediction-count", type=int,
                      help="Number of decoded predictions in the output tensor.")
  parser.add_argument("--attributes", type=int,
                      help="Number of attributes per prediction.")
  parser.add_argument("--prediction-layout", choices=SUPPORTED_PREDICTION_LAYOUTS,
                      help="Output tensor memory layout.")
  parser.add_argument("--has-objectness", action="store_true",
                      help="Set when the output layout is [box4, objectness, class scores].")
  parser.add_argument("--source-repo",
                      help="Source model repository or origin, overriding the selected preset.")
  parser.add_argument("--source-checkpoint",
                      help="Source model checkpoint/name, overriding the selected preset.")
  parser.add_argument("--export-notes",
                      help="Short provenance note for how this asset was exported.")
  parser.add_argument("--labels-file", type=Path,
                      help="JSON string array of model labels. Default: COCO 80 labels from the preset.")
  parser.add_argument("--hazard-label", action="append", dest="hazard_labels",
                      help="Hazard label to keep. May be repeated; defaults to the preset hazard labels.")
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR,
                      help=f"Destination asset directory. Default: {DEFAULT_MODEL_DIR}")
  args = parser.parse_args()

  model_src = args.model_dlc if args.model_dlc is not None else args.model_onnx
  if not model_src.is_file():
    raise FileNotFoundError(model_src)
  export_runtime = args.export_runtime
  if export_runtime is None:
    if args.model_onnx is not None and "_tinygrad_" in args.model_preset:
      export_runtime = "ONNX"
    else:
      export_runtime = "PRECOMPILED_QNN_ONNX" if args.model_onnx is not None else "QNN_DLC"

  input_width = input_height = None
  if args.input_size is not None:
    input_width, input_height = parse_input_size(args.input_size)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  model_dst = args.output_dir / ("model.onnx" if args.model_onnx is not None else "model.dlc")
  metadata_dst = args.output_dir / "metadata.json"
  shutil.copy2(model_src, model_dst)
  if args.model_onnx is not None:
    for sidecar in sorted(model_src.parent.glob("*.bin")) + sorted(model_src.parent.glob("*.data")):
      shutil.copy2(sidecar, args.output_dir / sidecar.name)

  metadata = build_metadata(
    sha256_file(model_dst),
    export_runtime,
    args.input_layout,
    model_preset=args.model_preset,
    input_name=args.input_name,
    input_dtype=args.input_dtype,
    input_scale=args.input_scale,
    input_zero_point=args.input_zero_point,
    input_width=input_width,
    input_height=input_height,
    output_name=args.output_name,
    output_dtype=args.output_dtype,
    output_scale=args.output_scale,
    output_zero_point=args.output_zero_point,
    prediction_count=args.prediction_count,
    attributes=args.attributes,
    prediction_layout=args.prediction_layout,
    has_objectness=True if args.has_objectness else None,
    source_repo=args.source_repo,
    source_checkpoint=args.source_checkpoint,
    export_notes=args.export_notes,
    labels=load_labels(args.labels_file) if args.labels_file is not None else None,
    hazard_labels=sorted(args.hazard_labels) if args.hazard_labels is not None else None,
  )
  metadata_dst.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

  print(f"Installed {model_dst}")
  print(f"Wrote {metadata_dst}")
  print(
    "Recorded "
    f"preset={args.model_preset} export_runtime={metadata['export_runtime']} "
    f"input_layout={metadata['input_layout']} output={metadata['output_name']}"
  )
  if metadata["source_repo"] in {"qualcomm/YOLOv11-Detection", "qualcomm/YOLOv8-Detection"}:
    print(f"Note: {metadata['source_repo']} does not distribute pre-exported DLC assets due to licensing.")
  print("Export a compatible asset, then run this helper with --model-dlc or --model-onnx and matching metadata.")


if __name__ == "__main__":
  main()
