# Object Hazard Tici Handoff - 2026-05-17

## Purpose

This handoff is for the next session that will implement the real fix for the experimental `objectd` object-hazard feature on the tici.

Current state: the object-hazard pipeline code is present, default-enabled, and guarded, but the available Qualcomm AI Hub `QNN_DLC` asset does not run on the current tici SNPE backend. The next real fix is either a tici-compatible SNPE DLC asset or a real QNN backend/runtime path.

## Repo / Branch State

- Laptop repo: `C:\Users\crimoldi\Documents\codex\chauffeur`
- Branch: `chauffeur-exp01`
- Pushed commits from this work:
  - `a39481015` - `Prepare object hazard assets for device bringup`
  - `e0a486215` - `Guard objectd against QNN DLC assets`
  - `a601c772c` - `Add QNN object hazard backend path`
  - `4ab0b2871` - `Run object hazard with ONNX Runtime QNN`
  - `b2bf0daf0` - `Point ORT QNN at bundled HTP libraries`
  - `35ec38561` - `Guard SNPE backend against QAIRT 2 DLCs`
  - `28978843d` - `Reject legacy QAIRT 2 DLC metadata`
- Tici repo: `/data/openpilot`
- Tici head after pull: `2897884`
- Tici had one pre-existing dirty file after pull: `live_waze_police_capture.json`
- Important workflow rule from user: commit and push from laptop, then pull to tici. Do not leave tracked manual edits on the tici.

## How To SSH To The Tici

The tici is on Wi-Fi at `192.168.1.172`.

Use PuTTY `plink` with the PPK key:

```powershell
& 'C:\Program Files\PuTTY\plink.exe' -batch -ssh `
  -i 'C:\Users\chrisr0122\.ssh\puttygen051426_ppk.ppk' `
  comma@192.168.1.172 'cd /data/openpilot && git status --short'
```

Use `pscp` for cache-only asset copies:

```powershell
& 'C:\Program Files\PuTTY\pscp.exe' -batch `
  -i 'C:\Users\chrisr0122\.ssh\puttygen051426_ppk.ppk' `
  'C:\path\to\asset.dlc' `
  comma@192.168.1.172:/data/openpilot/.cache/objectd/yolo11n/
```

Notes:
- The Windows process was running as `chrisr0122`.
- Direct access to `C:\Users\crimoldi\.ssh` failed with Windows access denied.
- User thought the newest PPK might be under `crimoldi\.ssh`; the usable key found in this session was `C:\Users\chrisr0122\.ssh\puttygen051426_ppk.ppk`.
- OpenSSH keys did not work for `comma@192.168.1.172`; `plink` with the PPK did.

## Tici Python Environment

Use the launch-equivalent environment before Python imports:

```sh
cd /data/openpilot
source launch_env.sh
export PYTHONPATH=$PWD
export VIRTUAL_ENV=/usr/local/venv
export PATH=$VIRTUAL_ENV/bin:$PATH
```

Plain `/usr/bin/python3` failed to import repo dependencies such as `numpy`; `/usr/local/venv/bin/python` worked.

One-line `plink` pattern:

```powershell
& 'C:\Program Files\PuTTY\plink.exe' -batch -ssh `
  -i 'C:\Users\chrisr0122\.ssh\puttygen051426_ppk.ppk' `
  comma@192.168.1.172 'cd /data/openpilot && source launch_env.sh && export PYTHONPATH=$PWD && export VIRTUAL_ENV=/usr/local/venv && export PATH=$VIRTUAL_ENV/bin:$PATH && python -c "import numpy; print(numpy.__version__)"'
```

## Current Feature State

- `ObjectHazardEnabled` default is now `"1"` in `common/params_keys.h`.
- Runtime param on the tici was set to true:

```sh
python -c "from openpilot.common.params import Params; p=Params(); p.put_bool('ObjectHazardEnabled', True); print(p.get_bool('ObjectHazardEnabled'))"
```

- Offroad toggle exists in the longitudinal settings panel as `Object Hazard Detection`.
- Hazard classes are intentionally limited to vulnerable road users and animals:
  - `person`
  - `bicycle`
  - `dog`
  - `horse`
  - `sheep`
  - `cow`
- Do not add cars/trucks to object-hazard semantics; normal vehicle handling belongs to existing openpilot lead/model/planner paths.

## Asset State

Local cache:

```text
.cache/objectd/yolo11n/model.dlc
.cache/objectd/yolo11n/metadata.json
.cache/objectd/source/yolo11n.pt
```

Tici cache:

```text
/data/openpilot/.cache/objectd/yolo11n/model.dlc
/data/openpilot/.cache/objectd/yolo11n/metadata.json
```

Current repaired asset SHA256:

```text
4a6b700675c9184df189dd8dc41c7bac5ee0b7f9905ed56659a19b004739f62c
```

Verify on tici:

```sh
sha256sum .cache/objectd/yolo11n/model.dlc
python -c "import json; print(json.load(open('.cache/objectd/yolo11n/metadata.json'))['model_sha256'])"
```

## What Happened With Qualcomm / Hugging Face

- Hugging Face repo: `qualcomm/YOLOv11-Detection`
- The HF repo does not host pre-exported DLC assets due to licensing; it only exposes docs/model-card-level info.
- Qualcomm AI Hub account/API is configured locally. Do not paste the token into chat.
  - QAI client config: `C:\Users\chrisr0122\.qai_hub\client.ini`
  - Token backup files were saved under `.cache/qai_hub/`
- Working AI Hub export produced a `QNN_DLC` for `SA8295P ADP` using YOLO11-N / `yolo11n.pt`.
- AI Hub model id printed by the exporter: `mmxlg4eym`.
- The direct export crashed during Qualcomm metadata writing because the no-postprocessing output was named `detector_output`, not the stock `boxes` output. A monkeypatch skipping `merge_output_metadata` allowed the DLC download.
- The first downloaded DLC had invalid ZIP CRC metadata. Because entries were stored uncompressed, it was repacked locally into a valid ZIP/DLC container.

The repack fixed archive readability, but not runtime compatibility.

## Current Blocking Failure

The tici `SNPEModel` runner can read the repaired archive, then repeatedly logs:

```text
SNPE model format version detected: 4.1.0
error_code=312; error_message=Undefined error
```

Both `OBJECTD_BACKEND=snpe_gpu` and `OBJECTD_BACKEND=snpe_dsp` timed out after 20 seconds with the same SNPE format loop.

Follow-up converter/runtime checks:

- The tici bundled SNPE runtime is `1.61.0.3358`.
- Official public QAIRT `v2.42.0`, `v2.33.0`, and `v2.22.6.240515` converters all produced DLCs that loop on the bundled SNPE runtime.
- QAIRT `v2.42.0` / `v2.33.0` DLCs report SNPE model format `4.1.0`; QAIRT `v2.22.6.240515` reports model format `4.0.0`.
- QAIRT `v2.22.6.240515` target `snpe-platform-validator --runtime gpu` passes the simple GPU validator on the tici, but `snpe-net-run --use_gpu` on the YOLO DLC fails with `QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED` and model validation error `No backend could validate Op=/model/0/conv/Conv Type=Conv2d`.
- The existing `snpemodel_pyx.so` cannot be rebound to QAIRT 2.x `libSNPE.so`; import fails with `undefined symbol: _ZNK3zdl8DlSystem21UserBufferEncodingTfN14getElementSizeEv`.
- QAIRT 2.x QNN GPU platform validation crashes, and QNN DSP validation finds libraries but fails loading the DSP calculator stub with `undefined symbol: remote_session_control`.

The code now rejects `export_runtime: QNN_DLC` metadata before constructing `SNPEModel`, so `objectd` fails safe instead of hanging:

```text
BackendError: objectd model export_runtime 'QNN_DLC' is not supported by the SNPE backend
```

Verify guard on tici:

```sh
cd /data/openpilot
source launch_env.sh
export PYTHONPATH=$PWD
export VIRTUAL_ENV=/usr/local/venv
export PATH=$VIRTUAL_ENV/bin:$PATH
timeout 20s python -c "from openpilot.sunnypilot.objectd.backend import build_detector_backend; build_detector_backend()" 2>&1 | head -20
```

## QNN Runtime Check

No QNN runtime or tools were found on the tici:

```sh
find /usr /data/openpilot /data -maxdepth 6 \
  \( -name "libQnn*.so" -o -name "qnn-net-run" -o -name "QNN*" -o -name "qnn*" \) \
  2>/dev/null | head -100
```

That command returned nothing in this session.

## Commands Already Verified

Local:

```powershell
python -m compileall -q sunnypilot\objectd sunnypilot\selfdrive\controls\lib\object_hazard_controller.py
python -m pytest -n 0 --basetemp .cache/pytest_tmp -o cache_dir=.cache/pytest_cache `
  sunnypilot/objectd/tests/test_process_registration.py `
  sunnypilot/objectd/tests/test_path_association.py `
  sunnypilot/selfdrive/controls/lib/tests/test_object_hazard_controller.py `
  sunnypilot/selfdrive/controls/lib/tests/test_object_hazard_pipeline.py
```

Latest local result after guard commit:

```text
21 passed
```

Tici:

```sh
cd /data/openpilot
source launch_env.sh
export PYTHONPATH=$PWD
export VIRTUAL_ENV=/usr/local/venv
export PATH=$VIRTUAL_ENV/bin:$PATH
python -m compileall -q sunnypilot/objectd sunnypilot/selfdrive/controls/lib/object_hazard_controller.py
```

Tici compileall passed.

## Next Real Fix Options

Preferred path A: produce a SNPE-compatible DLC.

- The current default viable backend is still SNPE: `sunnypilot/objectd/backend.py` uses `SNPEModel` for `snpe_gpu` / `snpe_dsp`, with experimental QNN paths guarded by asset/runtime metadata.
- The repo includes SNPE runtime libraries under `third_party/snpe`, but no converter tools were found.
- Need a DLC whose model format is supported by the bundled tici SNPE runtime.
- Public QAIRT 2.x converters are not sufficient; use an official SNPE 1.61-era converter package if available through Qualcomm Software Center / QPM or another licensed Qualcomm channel.
- If a SNPE-compatible DLC is produced, update metadata so `export_runtime` is `SNPE_DLC` or omit the field. Do not use `prepare_yolo11n_assets.py` unchanged, because it currently writes `export_runtime: QNN_DLC` and the backend will reject it.
- Add `--export-runtime` or similar to `prepare_yolo11n_assets.py` if using it for SNPE assets.

Path B: implement a real QNN backend.

- Current QNN DLC asset is probably appropriate for QNN, not SNPE.
- The tici currently appears to lack a QNN runtime stack that can initialize for this workload; QAIRT 2.x cache-only runtime probes did not produce a working GPU/DSP path.
- A QNN path means adding/installing runtime support and writing a backend that does not use `SNPEModel`.
- Do not silently add CPU inference fallback; if a CPU/ONNX/tinygrad fallback is explored, gate it explicitly and measure load.

Avoid:

- Do not remove the QNN guard just to let `SNPEModel` try; it loops and can leave stuck processes.
- Do not manually edit tracked files on the tici. Commit/push/pull.
- Do not add cars/trucks to object-hazard labels.

## Useful Files

- `sunnypilot/objectd/backend.py`
- `sunnypilot/objectd/objectd.py`
- `sunnypilot/objectd/prepare_yolo11n_assets.py`
- `sunnypilot/objectd/path_association.py`
- `sunnypilot/objectd/tests/test_process_registration.py`
- `sunnypilot/selfdrive/controls/lib/object_hazard_controller.py`
- `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
- `selfdrive/controls/lib/longitudinal_planner.py`
- `system/manager/process_config.py`
- `cereal/custom.capnp`
- `cereal/services.py`
- `.codex/skills/object-hazard-live-monitor/SKILL.md`

## Minimal Tici Pull / Verify Loop

```powershell
& 'C:\Program Files\PuTTY\plink.exe' -batch -ssh `
  -i 'C:\Users\chrisr0122\.ssh\puttygen051426_ppk.ppk' `
  comma@192.168.1.172 'cd /data/openpilot && git pull --ff-only origin chauffeur-exp01 && git rev-parse --short HEAD && git status --short'
```

```powershell
& 'C:\Program Files\PuTTY\plink.exe' -batch -ssh `
  -i 'C:\Users\chrisr0122\.ssh\puttygen051426_ppk.ppk' `
  comma@192.168.1.172 'cd /data/openpilot && source launch_env.sh && export PYTHONPATH=$PWD && export VIRTUAL_ENV=/usr/local/venv && export PATH=$VIRTUAL_ENV/bin:$PATH && python -m compileall -q sunnypilot/objectd sunnypilot/selfdrive/controls/lib/object_hazard_controller.py'
```
