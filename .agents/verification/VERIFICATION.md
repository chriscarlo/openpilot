## chauffeur-dev4 verification log

### Environment
- Host: `/home/chris/repos/chauffeur-dev3-port-only-chubbs-split`
- Branch: `chauffeur-dev4`

### Python sanity checks
```bash
python3 -m py_compile \
  sunnypilot/models/runners/helpers.py \
  sunnypilot/models/runners/model_runner.py \
  sunnypilot/modeld_v2/modeld.py \
  sunnypilot/modeld_v2/fill_model_msg.py \
  sunnypilot/modeld_v2/parse_model_outputs_split.py \
  selfdrive/modeld/modeld.py \
  selfdrive/modeld/parse_model_outputs.py \
  selfdrive/modeld/dmonitoringmodeld.py
```

### Build (targeted, avoids embedded toolchain)
The full `scons` default build requires `arm-none-eabi-gcc` (panda firmware toolchain) which is not present in this host environment.

This targeted build validates the Cython modules and native libs that the changed code depends on:
```bash
. .venv-dev4/bin/activate
scons -j"$(nproc)" -u \
  common/params_pyx.so \
  common/transformations/transformations.so \
  msgq_repo/msgq/ipc_pyx.so \
  msgq_repo/msgq/visionipc/visionipc_pyx.so \
  selfdrive/modeld/models/commonmodel_pyx.so \
  sunnypilot/modeld_v2/models/commonmodel_pyx.so
```

### VTSC regression (2026-02-11)
```bash
. .venv-dev4/bin/activate

# Fast targeted suite for VTSC changes (hold logic + state reset behavior)
python -m pytest -q sunnypilot/selfdrive/controls/lib/tests/vtsc

# Sanity compile checks for touched scripts/tests
python -m py_compile \
  sunnypilot/selfdrive/controls/lib/vision_turn_controller.py \
  sunnypilot/selfdrive/controls/lib/tests/vtsc/test_regression_rca_events.py \
  tools/vtsc/vtsc_intervention_recorder.py \
  tools/vtsc/extract_vtsc_rca_fixture.py
```

### VTSC steering fallback + recorder summary (2026-02-14)
Baseline quick check:
```bash
cd /home/chris/repos/chauffeur-dev4
/home/chris/.venv-openpilot/bin/pytest -m 'not slow'
```
Result: failed (missing deps: parameterized, hypothesis, jinja2, casadi, aiohttp, etc.).

Targeted checks:
```bash
cd /home/chris/repos/chauffeur-dev4
/home/chris/.venv-openpilot/bin/pytest sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py -k "steering_fallback"
/home/chris/.venv-openpilot/bin/python -m py_compile tools/vtsc/vtsc_intervention_recorder.py
```

### VTSC intervention recorder capture hardening (2026-02-16)
```bash
# Local syntax + targeted unit test for new cruise-source fallback helper
python3 -m py_compile tools/vtsc/vtsc_intervention_recorder.py
.venv/bin/pytest -q tools/vtsc/tests/test_vtsc_intervention_recorder.py

# Remote evidence on tici (commaCar) around user-reported 12:53/12:55 local window
# - Confirmed params/time:
#   MTSC lookahead toggle mtime at ~20:55:40 UTC
# - Compared old vs patched replay gates on segments 61-66:
#   OLD gas_rises=15 triggers=0
#   PATCHED gas_rises=15 triggers=1
# - Verified script runtime status:
#   python3 -m tools.vtsc.vtsc_intervention_recorder running under manager
# - Verified event capture directory state at check time:
#   /data/media/0/VTSCTuner/events had 0 event bundles
```

### MTSC health logging + VTSC handoff replay check (2026-02-16)
```bash
# Syntax checks
python3 -m py_compile \
  sunnypilot/selfdrive/controls/lib/vision_turn_controller.py \
  sunnypilot/selfdrive/controls/lib/tests/vtsc/test_mtsc_health_logging.py

# New targeted tests
.venv/bin/pytest -q sunnypilot/selfdrive/controls/lib/tests/vtsc/test_mtsc_health_logging.py

# Existing targeted VTSC regression sanity (representative)
.venv/bin/pytest -q sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py -k freeway_cap_hold_prevents_single_frame_flicker

# Mapd-bearing fixes regression checks
.venv/bin/pytest -q sunnypilot/mapd/tests/test_base_map_data_bearing.py
.venv/bin/pytest -q sunnypilot/mapd/tests/test_integration.py -k road_geometry_methods

# Recent route replay (segments copied from commaCar)
PYTHONPATH=. .venv/bin/python docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py \
  .cache/commaCar/00000027--c644173ddb/rlog_58.zst --out .cache/commaCar/00000027--c644173ddb/replay_c33_58.jsonl --cruise 33.5
PYTHONPATH=. .venv/bin/python docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py \
  .cache/commaCar/00000027--c644173ddb/rlog_59.zst --out .cache/commaCar/00000027--c644173ddb/replay_c33_59.jsonl --cruise 33.5
PYTHONPATH=. .venv/bin/python docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py \
  .cache/commaCar/00000027--c644173ddb/rlog_60.zst --out .cache/commaCar/00000027--c644173ddb/replay_c33_60.jsonl --cruise 33.5
PYTHONPATH=. .venv/bin/python docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py \
  .cache/commaCar/00000027--c644173ddb/rlog_61.zst --out .cache/commaCar/00000027--c644173ddb/replay_c33_61.jsonl --cruise 33.5
PYTHONPATH=. .venv/bin/python docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py \
  .cache/commaCar/00000027--c644173ddb/rlog_62.zst --out .cache/commaCar/00000027--c644173ddb/replay_c33_62.jsonl --cruise 33.5
```

Observed results:
- New MTSC health test file: `4 passed`.
- VTSC representative regression (`freeway_cap_hold_prevents_single_frame_flicker`): `passed`.
- Mapd bearing/unit tests: `4 passed`.
- Mapd integration road geometry method check: `passed`.
- Replay outputs generated successfully (about 1200 snapshots per segment).

Replay handoff findings (fixed-cruise replay set):
- `replay_c33_58`: jumps >2 m/s = 0, short near-cruise pulses = 0
- `replay_c33_59`: jumps >2 m/s = 0, short near-cruise pulses = 0
- `replay_c33_60`: jumps >2 m/s = 6, short pulses = 2
- `replay_c33_61`: jumps >2 m/s = 8, short pulses = 3
- `replay_c33_62`: jumps >2 m/s = 16, short pulses = 6

Representative pulse windows were short (~0.05s–0.50s), matching the reported “brief lurch” shape (temporary near-cruise windows that re-tighten).

Additional note:
- A wider scenario subset run surfaced an existing red test in this workspace:
  `test_mountain_cap_hold_prevents_single_frame_flicker_under_occlusion` (not introduced by MTSC health-log patch path).
