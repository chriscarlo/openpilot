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
