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

### Paramsd roll-confidence A/B root-cause proof (2026-02-27)

Quick checks:
```bash
cd /home/chris/repos/chauffeur-dev4
python3 -m py_compile selfdrive/locationd/paramsd.py selfdrive/selfdrived/selfdrived.py

cd /home/chris/repos/chauffeur-dev4-ab
python3 -m py_compile selfdrive/locationd/paramsd.py selfdrive/selfdrived/selfdrived.py
```
Result: passed.

Targeted deterministic A/B assertion harness:
```bash
/home/chris/.venvs/chffrverify/bin/python - <<'PY'
import json
import os
import subprocess
import textwrap

PYTHON = '/home/chris/.venvs/chffrverify/bin/python'
BASE = '/home/chris/repos/chauffeur-dev4-ab'      # db285c8c8 (pre-fix)
PATCHED = '/home/chris/repos/chauffeur-dev4'      # 9529c237e (with fix)

child_code = textwrap.dedent('''
import json
import numpy as np
from types import SimpleNamespace
from openpilot.selfdrive.locationd.paramsd import VehicleParamsLearner, States, ROLL_STD_MAX, LOW_ACTIVE_SPEED

def build_dummy(*, active: bool, speed: float, roll_std: float, roll_valid_seed: bool = True, steer_ratio: float = 1.0, stiffness: float = 1.0, angle_offset_deg: float = 0.0, road_roll_rad: float = 0.0, yaw_rate: float = 0.0):
  x = np.zeros((9, 1), dtype=float)
  x[States.STEER_RATIO] = steer_ratio
  x[States.STIFFNESS] = stiffness
  x[States.ANGLE_OFFSET] = np.radians(angle_offset_deg)
  x[States.ANGLE_OFFSET_FAST] = 0.0
  x[States.ROAD_ROLL] = road_roll_rad
  x[States.YAW_RATE] = yaw_rate

  P = np.eye(9, dtype=float) * 1e-6
  P[States.ROAD_ROLL, States.ROAD_ROLL] = roll_std ** 2

  return SimpleNamespace(
    kf=SimpleNamespace(x=x, P=P),
    angle_offset=angle_offset_deg,
    avg_angle_offset=angle_offset_deg,
    roll=road_roll_rad,
    active=active,
    observed_speed=speed,
    observed_yaw_rate=0.0,
    avg_offset_valid=True,
    total_offset_valid=True,
    roll_valid=roll_valid_seed,
    min_sr=0.5,
    max_sr=2.0,
    reset=lambda _t: None,
  )

def run_case(**kwargs):
  dummy = build_dummy(**kwargs)
  msg = VehicleParamsLearner.get_msg(dummy, valid=True, debug=False)
  lp = msg.liveParameters
  return {
    'lp_valid': bool(lp.valid),
    'msg_valid': bool(msg.valid),
    'sensor_valid': bool(lp.sensorValid),
    'steer_ratio_valid': bool(lp.steerRatioValid),
  }

rs = {
  'meta': {
    'roll_std_max': float(ROLL_STD_MAX),
    'low_active_speed': float(LOW_ACTIVE_SPEED),
  },
  'cases': {
    'low_speed_high_roll_std': run_case(active=True, speed=5.0, roll_std=ROLL_STD_MAX * 1.30),
    'high_speed_high_roll_std': run_case(active=True, speed=20.0, roll_std=ROLL_STD_MAX * 1.30),
    'high_speed_good_roll_std': run_case(active=True, speed=20.0, roll_std=ROLL_STD_MAX * 0.25),
    'low_speed_bad_steer_ratio': run_case(active=True, speed=5.0, roll_std=ROLL_STD_MAX * 0.25, steer_ratio=2.5),
  },
}
print(json.dumps(rs))
''')

def run(repo):
  env = os.environ.copy()
  env['PYTHONPATH'] = repo
  out = subprocess.check_output([PYTHON, '-c', child_code], env=env, text=True)
  return json.loads(out)

base = run(BASE)
patched = run(PATCHED)

# Root-cause proof: pre-fix invalid at low speed + high roll_std; patched valid
assert base['cases']['low_speed_high_roll_std']['lp_valid'] is False
assert patched['cases']['low_speed_high_roll_std']['lp_valid'] is True

# Non-regression: high-speed roll gating unchanged
assert base['cases']['high_speed_high_roll_std']['lp_valid'] is False
assert patched['cases']['high_speed_high_roll_std']['lp_valid'] is False
assert base['cases']['high_speed_good_roll_std']['lp_valid'] is True
assert patched['cases']['high_speed_good_roll_std']['lp_valid'] is True

# Non-regression: unrelated validity checks remain enforced
assert base['cases']['low_speed_bad_steer_ratio']['lp_valid'] is False
assert patched['cases']['low_speed_bad_steer_ratio']['lp_valid'] is False

# Message-level valid pass-through unchanged
for label in base['cases']:
  assert base['cases'][label]['msg_valid'] is True
  assert patched['cases'][label]['msg_valid'] is True

print('AB_PARAMS_ROOTCAUSE_CHECK: PASS')
print(json.dumps({'baseline': base, 'patched': patched}, indent=2))
PY
```
Result: passed (`AB_PARAMS_ROOTCAUSE_CHECK: PASS`).

Build/module gate (required runtime modules in both trees):
```bash
# baseline worktree
cd /home/chris/repos/chauffeur-dev4-ab
PATH=/home/chris/.venvs/chffrverify/bin:$PATH /home/chris/.venvs/chffrverify/bin/scons -j"$(nproc)" -u common/params_pyx.so msgq_repo/msgq/ipc_pyx.so
PATH=/home/chris/.venvs/chffrverify/bin:$PATH /home/chris/.venvs/chffrverify/bin/scons -j"$(nproc)" -u common/transformations/transformations.so
PATH=/home/chris/.venvs/chffrverify/bin:$PATH /home/chris/.venvs/chffrverify/bin/scons -j"$(nproc)" -u rednose/helpers/ekf_sym_pyx.so
PATH=/home/chris/.venvs/chffrverify/bin:$PATH /home/chris/.venvs/chffrverify/bin/scons -j"$(nproc)" -u selfdrive/pandad/pandad_api_impl.so

# patched tree
cd /home/chris/repos/chauffeur-dev4
PATH=/home/chris/.venvs/chffrverify/bin:$PATH /home/chris/.venvs/chffrverify/bin/scons -j"$(nproc)" -u rednose/helpers/ekf_sym_pyx.so
PATH=/home/chris/.venvs/chffrverify/bin:$PATH /home/chris/.venvs/chffrverify/bin/scons -j"$(nproc)" -u selfdrive/pandad/pandad_api_impl.so
```
Result: passed.

Known skips:
- Device integration drive/replay verification for this specific event path.
- Reason: device currently unreachable from office network.
- Follow-up: run the same branch on device and confirm no `paramsdTemporaryError` during parking-lot turn-in/out while retaining high-speed roll safeguards.

#### Re-validation on updated dev4 tip (2026-02-27)

After fast-forwarding `chauffeur-dev4` to `67df2ecd88a020855fe7065560723849f551e001` (docs-hygiene/AGENTS update), re-ran the same deterministic A/B assertion harness against:
- baseline worktree: `/home/chris/repos/chauffeur-dev4-ab` (`db285c8c8`)
- patched tree: `/home/chris/repos/chauffeur-dev4` (`67df2ecd8`)

Result: passed (`AB_PARAMS_ROOTCAUSE_CHECK_DEV4_HEAD: PASS`).
