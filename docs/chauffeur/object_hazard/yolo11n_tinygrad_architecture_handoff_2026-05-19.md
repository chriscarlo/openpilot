# Object Hazard YOLO11n Tinygrad Architecture Handoff - 2026-05-19

## Purpose

This handoff is for a new agent to create an experimental branch from the current branch and architect the next object-hazard model path: moving the current tinygrad ONNX detector from YOLOv8n to YOLO11n.

The immediate goal is not to retry SNPE/QNN_DLC. The goal is to design and, if viable, implement a reproducible tinygrad ONNX YOLO11n path that can be tested at small input sizes without disturbing the core openpilot stack.

## Required First Reads

Before changing code or judging live behavior, read:

- `.codex/skills/object-hazard-live-monitor/SKILL.md`
- `AGENTS.md`
- `docs/chauffeur/object_hazard/tici_qnn_snpe_handoff_2026-05-17.md`
- this file

Keep the actual pipeline in view:

- `sunnypilot/objectd/`
- `sunnypilot/selfdrive/controls/lib/object_hazard_controller.py`
- `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
- `selfdrive/controls/lib/longitudinal_planner.py`
- `selfdrive/controls/plannerd.py`
- `system/manager/process_config.py`
- `cereal/custom.capnp`
- `cereal/services.py`

## Create An Experimental Branch

Do not build the YOLO11n experiment directly on `chauffeur-exp01`. Create a new experimental branch from the current local branch so it can be pushed and loaded cleanly onto the tici without perturbing `chauffeur-exp01`:

```powershell
cd C:\Users\crimoldi\Documents\codex\chauffeur
git status --short
git branch --show-current
git switch -c codex/object-hazard-yolo11n
```

If `codex/object-hazard-yolo11n` already exists, use a unique `codex/` branch name and say so.

Before pushing, verify that the branch point is the current `chauffeur-exp01` head unless the user explicitly asks to start elsewhere:

```powershell
git merge-base --is-ancestor chauffeur-exp01 HEAD
git log --oneline --decorate -5
```

After validation, push the experimental branch by name. The tici should then pull that branch directly for testing:

```powershell
git push -u origin codex/object-hazard-yolo11n
```

## Current Repo State

- Primary branch: `chauffeur-exp01`
- Experimental branch: `codex/object-hazard-yolo11n`
- Latest relevant pushed commits at update time:
  - `eb630f7a4 Add YOLO11n tinygrad object hazard presets`
  - `46298a611 Optimize YOLO11n object hazard runtime`
- Current managed objectd default:
  - backend: `tinygrad_onnx`
  - tinygrad device: `QCOM`
  - model dir: `.cache/objectd/yolo11n_tinygrad_160_install`
  - cadence: `DEFAULT_DETECTOR_HZ = 2.0`
  - warmup runs: `DEFAULT_TINYGRAD_WARMUP_RUNS = 3`
  - onroad warmup allowed: `DEFAULT_ALLOW_ONROAD_WARMUP = True`
  - inference budget: `DEFAULT_INFER_BUDGET_MS = 250.0`
  - schedule phase: `DEFAULT_SCHEDULE_PHASE_SEC = 0.17`
  - ROI mode: `DEFAULT_ROI_MODE = road_wide`
  - ROI top crop: `DEFAULT_ROI_TOP_FRACTION = 0.15`
- The current branch has been validated locally but still needs tici asset install and live monitoring.

Unrelated dirty files existed in the original checkout at handoff time:

```text
sunnypilot/selfdrive/controls/lib/tests/vtsc/pipeline_harness.py
system/sentry.py
.codex/config.toml
docs/chauffeur/object_hazard/object_hazard_next_agent_prompt_2026-05-17.md
```

Do not revert unrelated dirty files. If they are still present when you branch, leave them unstaged unless they are directly required for the YOLO11n experiment.

## Current Working Detector Path

The only currently viable accelerator-backed path is tinygrad ONNX on QCOM:

- `TinygradOnnxYoloDetector` in `sunnypilot/objectd/backend.py`
- default config in `sunnypilot/objectd/config.py`
- deadline/phase scheduling in `sunnypilot/objectd/scheduler.py`
- asset metadata in `sunnypilot/objectd/prepare_yolo11n_assets.py`

The detector loop is latest-frame only. It does not run inference over a backlog of camera frames:

- `recv_latest_buffer()` takes one blocking VisionIPC receive, then drains any immediately available newer buffers with `recv(0)`.
- Only the newest retained buffer goes into `backend.infer(buf)`.
- The tinygrad worker queue is `maxsize=1`, so it does not build an inference batch.
- Warmup is the only intentional burst: default 3 zero-input model runs once per objectd process start.
- Runtime defaults cap OpenCV and common numeric helper thread fanout at one thread.
- `road_wide` ROI mode crops the top of the NV12 frame before RGB conversion/resizing while preserving original-frame detection coordinates.
- The scheduler skips one detector tick after over-budget inference instead of allowing repeated detector-cadence pressure pulses.
- `OBJECTD_CPU_AFFINITY` is available as an opt-in live-monitor-driven control; no hard pin is enabled by default.

Known YOLOv8n tinygrad ONNX asset sizes:

```text
160: output0 [1,84,525]
224: output0 [1,84,1029]
256: output0 [1,84,1344]
320: output0 [1,84,2100]
```

The current default is 160 because the onroad 256 path caused visible timing pressure in multiple services. The reduction is material:

- 160x160 input: 25,600 pixels, 525 candidate predictions
- 256x256 input: 65,536 pixels, 1,344 candidate predictions
- 160 is about 39% of the 256 input pixel count, with about 39% of the candidate decode count

This is a performance mitigation, not a semantic win. It may miss smaller/farther hazards.

## Live Tici Evidence So Far

Earlier tici testing reached a healthy end-to-end state with YOLOv8n 256 tinygrad ONNX:

- `objectd` manager-owned
- backend `tinygrad_onnx:qcom`
- `modelReady=true`
- `ObjectHazardEnabled=True`
- `objectHazardStateSP` publishing at 2 Hz
- HUD `OBJ` green
- planner propagation wired through `longitudinalPlanSP.objectHazardControl` and main `longitudinalPlan.shouldStop`

But onroad monitoring showed unacceptable timing pressure:

- user observed choppy video and steering-control lag/oscillation-like behavior when object hazard was enabled
- HUD monitor showed several services flickering yellow about every 2 seconds
- 60 s live capture with 256 asset showed objectd around 15.5% CPU, 268 MB RSS, 13 threads
- device CPU average around 64.6%, max sample 98%
- GPU average around 12%
- thermal green
- objectd worker was `SCHED_IDLE`, but helper threads were normal scheduler class and could land on busy cores
- no obvious free core existed; pinning risks colliding with more important services

The current 160 default is an attempt to reduce each 2 Hz burst without reducing cadence further.

## Do Not Retry These Paths

Do not retry the failed SNPE/QNN_DLC paths unless the user explicitly changes direction and you have a new runtime/converter fact that changes the analysis.

Known blocked paths:

- Qualcomm AI Hub `QNN_DLC` artifacts are not SNPE-loadable on this tici.
- Public QAIRT 2.x DLCs report model format 4.x and loop/fail on the bundled tici SNPE 1.61 runtime.
- `SNPEModel` with those assets can hang or loop.
- QAIRT 2.x QNN runtime probes did not produce a usable GPU/DSP/CPU QNN path on this tici.
- Do not remove the SNPE/QNN guards just to "try it anyway."

Keep CPU inference measurement-only:

- Do not silently fall back to CPU.
- `OBJECTD_ALLOW_CPU_INFERENCE=1` is for explicit measurement only.

## Why YOLO11n Is Worth Architecting

Current search/reading suggests YOLO11n is the best next small-model experiment if it can stay on the tinygrad ONNX path:

- YOLO11n is smaller than YOLOv8n by Ultralytics headline numbers:
  - YOLO11n: about 2.6M params / 6.5B FLOPs
  - YOLOv8n: about 3.2M params / 8.7B FLOPs
- Ultralytics reports better COCO mAP for YOLO11n than YOLOv8n in its model tables.
- It should export through the same Ultralytics ONNX flow.
- It may reduce the per-tick burst without changing the runtime architecture.

Compatibility caveat:

- Quantized INT8 YOLOs on Hugging Face are mostly runtime-specific, such as OpenVINO IR, TFLite, or vendor-targeted packages.
- Ultralytics ONNX export supports small `imgsz` exports, but ONNX INT8 is not the straightforward drop-in path here.
- Treat INT8 as a separate runtime-architecture problem, not a simple replacement for `model.onnx`.

Potential references to verify before committing a model choice:

- Ultralytics YOLO11 docs/model table
- Ultralytics YOLOv8 docs/model table
- Ultralytics export docs for ONNX arguments and supported quantization formats
- Hugging Face model cards for any candidate quantized YOLO11n/YOLOv8n artifacts

## Architecture Task For The New Agent

Architect a move from `yolov8n_tinygrad_*` to `yolo11n_tinygrad_*` without changing the fail-closed runtime contract.

Recommended structure:

1. Create cache-only YOLO11n ONNX assets at 160 and one fallback resolution.

Suggested candidate resolutions:

```text
160x160: first performance candidate
192x192: likely compromise candidate if 160 loses too much detail
224x224: fallback if 160/192 are semantically too weak
```

Do not commit model binaries. Store assets under `.cache/objectd/`, for example:

```text
.cache/objectd/yolo11n_tinygrad_160_install/model.onnx
.cache/objectd/yolo11n_tinygrad_160_install/metadata.json
.cache/objectd/yolo11n_tinygrad_192_install/model.onnx
.cache/objectd/yolo11n_tinygrad_192_install/metadata.json
```

2. Export YOLO11n in the same style as the working YOLOv8n tinygrad assets.

Expected shape is likely still YOLO anchor-free attributes-first:

```text
output0 [1,84,N]
input images [1,3,H,W]
```

Do not assume `N`. Inspect the exported ONNX or metadata after export. For stride-8/16/32 YOLO heads, square input candidate counts are expected to be:

```text
160: 20*20 + 10*10 + 5*5 = 525
192: 24*24 + 12*12 + 6*6 = 756
224: 28*28 + 14*14 + 7*7 = 1029
256: 32*32 + 16*16 + 8*8 = 1344
```

3. Add explicit metadata presets.

Likely files:

- `sunnypilot/objectd/prepare_yolo11n_assets.py`
- `sunnypilot/objectd/tests/test_process_registration.py`

Add presets such as:

```text
yolo11n_tinygrad_160
yolo11n_tinygrad_192
```

Metadata should use:

```text
source_repo: ultralytics/yolo11
source_checkpoint: YOLO11-N / yolo11n.pt
export_runtime: ONNX
input_layout: NCHW
input_name: images
output_name: output0
attributes: 84
prediction_layout: attributes_first
has_objectness: false
labels: COCO_80_LABELS
hazard_labels: see Product/Semantic Constraints below
decoder_family: yolo_anchor_free
```

4. Make the managed default switch explicit and reversible.

Likely file:

- `sunnypilot/objectd/config.py`

Do not hard-code a half-working experiment without tests. Options:

- switch default to `.cache/objectd/yolo11n_tinygrad_160_install` only after local metadata/shape tests pass
- or add a short design doc first and leave default unchanged until tici assets are copied and probed

Keep `OBJECTD_MODEL_DIR`, `OBJECTD_MODEL_PATH`, and `OBJECTD_MODEL_METADATA` overrides working.

5. Consider refactoring preset duplication only if it keeps things clearer.

The current preset table already has duplicated YOLOv8n tinygrad entries. A tiny helper/factory may make YOLO11n 160/192/224 less error-prone, but avoid a broad cleanup.

6. Validate tinygrad graph compatibility before live claims.

Local Windows cannot prove QCOM runtime behavior, but it can prove metadata, shape, and tests. If tinygrad and ONNX dependencies are present locally, try a tiny ONNX shape-load check. Otherwise document the missing dependency and leave QCOM validation for the tici.

On tici, validation must be cache-only assets plus manager-owned objectd. Do not manually edit tracked files on tici.

## Suggested Local Verification

Run at minimum:

```powershell
python -m compileall -q sunnypilot\objectd sunnypilot\selfdrive\controls\lib\object_hazard_controller.py
python -m pytest -n 0 --basetemp .cache/pytest_tmp -o cache_dir=.cache/pytest_cache `
  sunnypilot/objectd/tests/test_process_registration.py `
  sunnypilot/objectd/tests/test_path_association.py `
  sunnypilot/selfdrive/controls/lib/tests/test_object_hazard_controller.py `
  sunnypilot/selfdrive/controls/lib/tests/test_object_hazard_pipeline.py
```

Known latest result before this handoff, after the 160 YOLOv8n default:

```text
48 passed
```

If local pytest fails due to missing native/generated Windows modules, report the exact missing import separately from code-health findings.

## Suggested Tici Validation Later

Only after code is committed/pushed and the tici is reachable:

1. Pull the experimental branch on tici, not `chauffeur-exp01`.
2. Copy cache-only YOLO11n ONNX assets to `/data/openpilot/.cache/objectd/...`.
3. Reboot or restart through the normal manager flow, not by ad hoc relaunch of individual tracked services.
4. Use the live monitor:

```bash
python3 .codex/skills/object-hazard-live-monitor/scripts/object_hazard_live_monitor.py \
  --ssh-profile commaHome \
  --duration 45 \
  --save-json .cache/object-hazard-monitor-yolo11n-tinygrad.json
```

Evidence required before calling it viable:

- `objectHazardStateSP` exists, alive, valid, and at expected frequency
- backend reports `tinygrad_onnx:qcom`
- `modelReady=true` only after warmup succeeds
- no repeated backend errors
- `objectd` CPU/RSS/thread count stays acceptable
- HUD monitor no longer produces broad yellow pulses at detector cadence
- planner propagation reaches `longitudinalPlanSP.objectHazardControl`
- main `longitudinalPlan.shouldStop` reflects stop requests only when appropriate

## Product/Semantic Constraints

The object-hazard feature is for general non-car hazards, especially vulnerable road users and animals. Existing openpilot lead/model/planner paths already handle normal car following.

The product direction changed after this handoff was first written: keep classification available for future evade/do-not-evade decisions, but filter detections to classes that could plausibly harm a human/animal or damage the vehicle.

Current default hazard labels:

```text
person
bicycle
car
motorcycle
bus
train
truck
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
traffic light
fire hydrant
stop sign
parking meter
bench
backpack
suitcase
skis
snowboard
skateboard
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
microwave
oven
sink
refrigerator
```

Soft/low-consequence classes such as food, utensils, books, toothbrushes, and similar debris remain filtered out by default.

## Definition Of Done For The Architecture Pass

A good first pass on the experimental branch should produce one of:

- a small, tested code change adding YOLO11n tinygrad ONNX presets and a managed default choice, plus a clear note that tici asset/live validation remains
- or a short design doc explaining why YOLO11n ONNX is not compatible with tinygrad/QCOM and what the next smallest viable model path is

Do not call the feature done until live tici monitoring shows resource stability and planner propagation.
