# Object Hazard Model / Backend Options - 2026-05-17

## Scope

Chauffeur object hazard detection should cover vulnerable road users and non-car obstacles. Cars and trucks stay out of object-hazard semantics because openpilot already has a normal vehicle lead/planner path.

This note ranks model/backend options after the YOLO11 QNN-DLC-on-SNPE path was proven incompatible with the current tici SNPE 1.61 runtime. Do not use this as permission to remove the SNPE/QNN guards in `sunnypilot/objectd/backend.py`.

## Current Best Path

1. Keep the existing backend fail-closed behavior.
2. Use `qualcomm/YOLOv8-Detection` / YOLOv8-N as the next general COCO detector candidate.
3. Export YOLOv8-N with no postprocessing and verify the produced tensor names/shapes before installing the asset:
   - expected input: `image`, 640x640
   - expected output: `detector_output`, `[1,84,8400]`
   - expected labels: COCO 80
   - object-hazard labels: `person`, `bicycle`, `dog`, `horse`, `sheep`, `cow`
4. Prefer a true SNPE 1.61-compatible DLC if an official/licensed SNPE 1.x converter can be obtained.
5. If the exported asset is QNN-era only, treat it as blocked until a real tici QNN runtime/backend path exists.

`sunnypilot/objectd/prepare_yolo11n_assets.py` now has a `--model-preset yolov8n` path and explicit metadata overrides so the installed asset records model provenance instead of pretending every detector is YOLO11.

## Implementation / Test Result

YOLOv8-N was exported through Qualcomm AI Hub as a plain ONNX asset with `--no-include-postprocessing`. The export hit the same Qualcomm model-card metadata bug seen on YOLO11: the compiled model output is `detector_output`, but `qai_hub_models` expects semantic output `boxes` while merging metadata. A local runtime monkeypatch skipped only `merge_output_metadata`; the exported model metadata itself correctly records:

- input `image`: `[1, 3, 640, 640]`, `float32`
- output `detector_output`: `[1, 84, 8400]`, `float32`
- tools: ONNX `1.19.1`, ONNX Runtime `1.24.1.dev20260212`

Cache-only local assets:

- raw AI Hub export: `.cache/objectd/yolov8n_onnx_patched/yolov8_det-onnx-float/`
- installed objectd test asset: `.cache/objectd/yolov8n_onnx_install/`

Code-side result:

- `ONNX` is now an explicit asset runtime in the metadata helper.
- `onnx_cpu` / `ort_cpu` is available only when `OBJECTD_ALLOW_CPU_INFERENCE=1` is set. This is for measurement and model-contract testing, not production fallback.
- The helper now copies ONNX external-data sidecars such as `yolov8_det.data`.

Local smoke tests:

- Zero-input ONNX Runtime CPU execution initialized and produced the expected `705600` float outputs in about `262 ms` on the Windows laptop CPU.
- On `https://ultralytics.com/images/bus.jpg`, the existing hazard-label filter returned four `person` detections and excluded vehicle classes from object-hazard output.
- Focused object-hazard tests passed: `34 passed`.

This proves YOLOv8-N is semantically compatible with the current decoder and hazard-label contract. It does not prove a production tici accelerator path; the remaining blocker is still SNPE 1.61 asset compatibility or a real QNN/TFLite runtime path.

## Candidate Ranking

| Rank | Candidate | Why it is interesting | Main blocker |
|---|---|---|---|
| 1 | `qualcomm/YOLOv8-Detection` | Best drop-in general detector candidate. Qualcomm-maintained, Android/real-time tagged, COCO-style classes, YOLOv8-N is small enough for low-cadence auxiliary use. Qualcomm reports YOLOv8-N at 640x640, 3.18M parameters, 12.2 MB float, and QNN_DLC SA8295P timings around 7.872 ms float / 4.564 ms w8a16 / 2.215 ms w8a8. | Qualcomm does not distribute pre-exported assets for this model due to licensing, so we still need a compatible export. AI Hub QNN_DLC is not SNPE-loadable on this tici image. |
| 2 | `qualcomm/Person-Foot-Detection` | Best narrow human-hazard prototype. It has downloadable ONNX/QNN_DLC/TFLite assets, 640x480 input, 2 output classes, 2.53M parameters, and reported QNN_DLC SA8295P timings around 7.970 ms float / 5.175 ms w8a16 / 2.205 ms w8a8. | It only covers person/foot. It does not cover bicycles or animals and will need decoder/metadata work before it can feed the current YOLO decoder path. Its QNN_DLC assets are QAIRT 2.45 era, so still not SNPE 1.61 assets. |
| 3 | YOLOv5n or older SSD/MobileNet-style detectors | Best compatibility bet if a legacy SNPE 1.x converter becomes available. Older graphs may avoid some modern YOLO11/YOLOv8 export friction, and a 320px YOLOv5n export is documented in Qualcomm SNPE examples. | Does not solve the missing converter/runtime problem by itself. Ultralytics licensing must be reviewed before product use. |
| 4 | `qualcomm/RF-DETR` or other small transformer detectors | COCO-style detector and Qualcomm-packaged assets exist. | Too heavy for the auxiliary tici path. Qualcomm reports RF-DETR-base at 29M parameters / 116 MB, and QNN_DLC SA8295P around 108.245 ms with up to 871 MB peak memory. Keep it as a research reference, not the next tici run. |
| 5 | Segmentation or free-space anomaly models | Could eventually detect non-class-specific road obstacles. | It does not fit the current `objectHazardStateSP` boxes/classes/scores contract without a new semantic layer, and runtime cost/risk is not yet justified. |

## Backend Ranking

| Rank | Backend path | Status |
|---|---|---|
| 1 | Existing SNPE backend with a SNPE 1.61-compatible DLC | Still the cleanest production path because `objectd` already has `snpe_gpu` / `snpe_dsp` wiring and guardrails. Requires a real SNPE 1.x compatible converter or asset. |
| 2 | Real QNN runtime/backend on tici | Required for AI Hub `QNN_DLC` and precompiled QNN-style assets. Current probes did not find a usable native QNN stack on the tici, so this is runtime bring-up work before model work. |
| 3 | TFLite accelerator path | Qualcomm model cards often report good TFLite/NPU numbers, including YOLOv8 and Person-Foot. A read-only tici inventory found `libQnnTFLiteDelegate.so` only inside cache-copied QAIRT directories, not as a native system runtime. Treat this as experimental runtime bring-up, not a ready backend. |
| 4 | CPU ONNX/TFLite | Measurement-only fallback. Do not present it as a fix unless it is explicitly gated, low cadence, and monitored with the live harness. |

Read-only tici inventory on 2026-05-17:

- `/data/openpilot/third_party/snpe/.../libSNPE.so` is still the only repo-bundled production accelerator runtime found.
- QAIRT/QNN/TFLite delegate files exist under cache-only experiment paths such as `.cache/qairt222_runtime/`, `.cache/objectd/qairt_2.42_ubuntu/`, and `.cache/objectd/python/onnxruntime_qnn/`.
- The tici working tree was at `08207ea` with the known unrelated dirty `live_waze_police_capture.json`.

## Tici Rules For The Next Experiment

- Do not manually edit tracked files on the tici.
- Commit/push local repo changes, then pull on `/data/openpilot`.
- Cache-only model assets may be copied under `.cache/objectd/...`.
- Do not retry QAIRT 2.x DLCs through `SNPEModel`; the guard exists to prevent a known loop.
- After any backend initializes, verify `objectHazardStateSP`, `longitudinalPlanSP.objectHazardControl`, and `longitudinalPlan.shouldStop` with the live monitor before calling it done.

## Sources Checked

- `qualcomm/YOLOv8-Detection` Hugging Face model card and API metadata, checked 2026-05-17: https://huggingface.co/qualcomm/YOLOv8-Detection
- `qualcomm/Person-Foot-Detection` Hugging Face model card and API metadata, checked 2026-05-17: https://huggingface.co/qualcomm/Person-Foot-Detection
- `qualcomm/RF-DETR` Hugging Face model card and API metadata, checked 2026-05-17: https://huggingface.co/qualcomm/RF-DETR
- Qualcomm AI Hub FAQ on AI Engine Direct formats, checked 2026-05-17: https://workbench.aihub.qualcomm.com/docs/hub/faq.html
- Qualcomm SNPE YOLOv5 quick-start PDF, checked 2026-05-17: https://docs.qualcomm.com/bundle/publicresource/KBA-240222225148_REV_1_Quick_Start_Demo_of_SNPE_Yolov5_in_6490.pdf
