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

### VTSC map-lookahead reason diagnostics (2026-02-25)
Targeted checks:
```bash
cd /home/chris/repos/chauffeur-dev4
/home/chris/.venv-openpilot/bin/pytest -q sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py -k "map_lookahead or offramp_short_tight_curve_map_cap_applies_when_vision_lost"
/home/chris/.venv-openpilot/bin/pytest -q sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py -k "map_lookahead or map_lookahead_reason or offramp_short_tight_curve_map_cap_applies_when_vision_lost"
/home/chris/.venv-openpilot/bin/python -m py_compile \
  sunnypilot/selfdrive/controls/lib/vision_turn_controller.py \
  sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py
```
Results: passed (`4/4`, then `6/6`, plus `py_compile` success).

Broader VTSC suite gate:
```bash
cd /home/chris/repos/chauffeur-dev4
/home/chris/.venv-openpilot/bin/pytest -q sunnypilot/selfdrive/controls/lib/tests/vtsc
```
Result: failed with 3 existing regressions in this workspace:
- `test_mountain_cap_hold_prevents_single_frame_flicker_under_occlusion`
- `test_throttle_prob_gate_can_prevent_accel_after_vtsc_release`
- `test_fov_occlusion_clears_on_straight_even_with_mediocre_confidence`

Offline log analysis artifacts generated:
```bash
/home/chris/.cache/commaCar/vtsc_pull_latest_last40m/analysis/all_override_rises_last40m.jsonl
/home/chris/.cache/commaCar/vtsc_pull_latest_last40m/analysis/vtsc_override_summary_fallback_last40m.json
/home/chris/.cache/commaCar/vtsc_pull_latest_last40m/analysis/vtsc_turn_source_arbitration_last40m.json
/home/chris/.cache/commaCar/vtsc_pull_latest_last40m/analysis/vtscdbg_map_activity_last40m.json
/home/chris/.cache/commaCar/vtsc_pull_latest_last40m/analysis/liveMapDataSP_health_last40m.json
```

### VTSC intervention recorder vEgo/vCruise fix (2026-02-26)
Targeted sanity check:
```bash
python3 -m py_compile tools/vtsc/vtsc_intervention_recorder.py
```
Result: success.

### Mapd: Publish Bearing In LastGPSPosition For openpilot-mapd (2026-02-26)
Targeted sanity checks:
```bash
cd /home/chris/repos/chauffeur-dev4
/home/chris/.venv-openpilot/bin/python -m py_compile sunnypilot/mapd/live_map_data/osm_map_data.py
```
Result: success.

Targeted tests:
```bash
cd /home/chris/repos/chauffeur-dev4
/home/chris/.venv-openpilot/bin/pytest -q sunnypilot/mapd/tests/test_integration.py -k "update_location_with_no_database or tick_method"
/home/chris/.venv-openpilot/bin/pytest -q sunnypilot/mapd/tests
```
Result: failing with multiple pre-existing issues in this workspace (not triaged/fixed in this change).

### Commit Gate Sweep: VTSC + Mapd (2026-02-26)
Commands run:
```bash
cd /home/chris/repos/chauffeur-dev4
./.venv-dev4/bin/python -m pytest -q \
  sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py::test_mountain_cap_hold_prevents_single_frame_flicker_under_occlusion \
  sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py::test_fov_occlusion_clears_on_straight_even_with_mediocre_confidence

./.venv-dev4/bin/python -m pytest -q sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py

./.venv-dev4/bin/python -m pytest -q \
  sunnypilot/mapd/tests/test_integration.py::TestMapdIntegration::test_update_location_writes_last_gps_with_bearing

./.venv-dev4/bin/python -m pytest -q sunnypilot/mapd/tests/test_integration.py

./.venv-dev4/bin/python -m ruff check --select E,F \
  sunnypilot/mapd/live_map_data/osm_map_data.py \
  sunnypilot/mapd/tests/test_integration.py \
  sunnypilot/selfdrive/controls/lib/vision_turn_controller.py \
  sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py \
  tools/vtsc/vtsc_intervention_recorder.py
```

Results:
- VTSC targeted regressions: **passed** (`2 passed`)
- VTSC scenario suite: **passed** (`27 passed`)
- Mapd targeted bearing test: **passed** (`1 passed`)
- Mapd integration suite: **passed** (`10 passed`)
- Targeted lint sanity (E/F on touched files): **passed** (`All checks passed`)

Notes:
- Repo-wide lint script (`scripts/lint/lint.sh`) reports broad pre-existing baseline issues outside this change set (docs/hooks/shebang/codespell/mypy across unrelated files).
