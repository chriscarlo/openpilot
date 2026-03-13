---
name: model-manager-updater
description: >
  Update the Sunnypilot model manager to support a new upstream model list
  version, new model types, or selector version bumps. Use when the user
  reports "only the default model shows", asks to bump the model selector
  version, switch the model list URL, or sync model-manager changes from
  upstream sunnypilot. Triggers include "model manager", "model list",
  "selector version", "driving_models_v", "MODEL_URL",
  "ModelManagerSP", "offPolicy", "model type enum".
---

# Model Manager Updater

## Overview

The Sunnypilot model manager lets users select community driving models
from a remote JSON model list. Updating it typically means bumping
version constants, switching the model list URL, and handling any new
model types or schema changes introduced by the new list.

Every update touches a **fixed set of files** that must stay in sync.
Missing any one of them causes a different silent or runtime failure.

## File Inventory (ordered by dependency)

| File | What to check |
|------|---------------|
| `cereal/custom.capnp` | `ModelManagerSP.Model.Type` enum — must include every `type` string present in the new JSON model list |
| `sunnypilot/models/helpers.py` | `CURRENT_SELECTOR_VERSION`, `REQUIRED_MIN_SELECTOR_VERSION`, `is_bundle_version_compatible()` |
| `sunnypilot/models/fetcher.py` | `MODEL_URL`, `ModelParser._parse_model()`, `ModelParser.parse_models()` |
| `sunnypilot/models/manager.py` | Download loop, param read/write, index checks |
| `selfdrive/ui/sunnypilot/qt/offroad/settings/models_panel.cc` | C++ `switch(model.getType())` — must have a case for every capnp enum value; progress bar + frame per type; if the selector is browseable onroad, it also needs a cache fallback because `models_manager` is offroad-only |
| `selfdrive/ui/sunnypilot/qt/offroad/settings/models_panel.h` | Declares `QProgressBar*` and `QFrame*` members for each model type |
| `sunnypilot/models/runners/tinygrad/model_types.py` | Parser mapping for new runtime model types; `offPolicy` needs its own parser entry |
| `sunnypilot/models/runners/tinygrad/tinygrad_runner.py` | `TinygradSplitRunner` must instantiate and merge every artifact required by the bundle, including `offPolicy` |
| `sunnypilot/modeld_v2/parse_model_outputs_split.py` | `planplus` may arrive without `plan`; parse it independently for three-artifact bundles |
| `sunnypilot/models/tinygrad_ref.py` | If `tinygrad_repo` is vendored instead of a submodule/git checkout, read the pinned runtime ref from tracked metadata first |
| `tinygrad_repo/.vendored_ref` | When vendoring tinygrad, record the exact upstream commit here so compatibility checks can still prove the runtime matches the model list |

## Step-by-Step Workflow

### 1. Identify what changed upstream

```bash
# Fetch upstream sunnypilot
git fetch sp master --depth=1

# Compare key files
git diff HEAD..sp/master -- sunnypilot/models/helpers.py sunnypilot/models/fetcher.py sunnypilot/models/manager.py cereal/custom.capnp
```

Or if bumping manually, inspect the new model list JSON:

```bash
curl -s '<NEW_MODEL_URL>' | python3 -c '
import sys, json
d = json.load(sys.stdin)
types = set()
for b in d.get("bundles", []):
    for m in b.get("models", []):
        types.add(m.get("type"))
print("Model types in JSON:", sorted(types))
print("Bundle count:", len(d.get("bundles", [])))
versions = [b.get("minimum_selector_version") for b in d.get("bundles", [])]
print("min_selector_version range:", min(versions), "-", max(versions))
'
```

For tinygrad runtime issues, also check the runtime pin itself:

```bash
# Upstream sunnypilot stores tinygrad_repo as a gitlink, not a tree
git rev-parse sp/master:tinygrad_repo

# Remote compiled model list pin
python3 - <<'PY'
import requests
from sunnypilot.models.fetcher import ModelFetcher
print(requests.get(ModelFetcher.MODEL_URL, timeout=10).json()["tinygrad_ref"])
PY
```

If this branch vendors `tinygrad_repo`, compare your local tree against a real
checkout/archive of that tinygrad commit. Do **not** rely on
`git diff sp/master:tinygrad_repo ...`; the superproject only stores a gitlink.

### 2. Update version constants (`helpers.py`)

```python
CURRENT_SELECTOR_VERSION = <new_version>
REQUIRED_MIN_SELECTOR_VERSION = <new_min>
```

The compatibility window is: `REQUIRED_MIN <= bundle.minimumSelectorVersion <= CURRENT`.

### 3. Update model list URL (`fetcher.py`)

```python
MODEL_URL = "https://raw.githubusercontent.com/sunnypilot/sunnypilot-docs/refs/heads/gh-pages/docs/driving_models_v<N>.json"
```

### 4. Add any new model types to the capnp enum

Check the JSON `type` values against `cereal/custom.capnp` → `ModelManagerSP.Model.Type`.
**New types must be appended** (capnp enums are positional — never reorder or insert).

### 5. Add UI support for new model types (`models_panel.cc` + `.h`)

For each new enum value, follow the existing pattern — declare `QProgressBar*` and
`QFrame*` in the header, create them in the constructor, add visibility reset in
`handleBundleDownloadProgress()`, and add a `case` in the `switch(model.getType())`.

If the selector can be opened onroad, do not rely on `modelManagerSP` alone.
`system/manager/process_config.py` keeps `models_manager` on `only_offroad`, so
`models_panel.cc` must fall back to cached params (`ModelManager_ModelsCache`
and `ModelManager_ActiveBundle`) or the UI will appear empty while driving.

### 6. Audit runtime support for the new bundle composition

Do not stop after capnp + parser + UI support. If a bundle contains
`offPolicy`, also diff/check:

- `sunnypilot/models/runners/tinygrad/model_types.py`
- `sunnypilot/models/runners/tinygrad/tinygrad_runner.py`
- `sunnypilot/modeld_v2/parse_model_outputs_split.py`

### 7. Build and deploy

```bash
# Local C++ check (fast — just the one object file)
scons -j$(nproc) selfdrive/ui/sunnypilot/qt/offroad/settings/models_panel.o

# On-device
git push && ssh commaCar "cd /data/openpilot && git pull && sudo reboot"
```

### 8. Verify on device

```bash
ssh commaCar "cd /data/openpilot && source /usr/local/venv/bin/activate && python3 -c '
from sunnypilot.models.fetcher import ModelFetcher
from openpilot.common.params import Params
f = ModelFetcher(Params())
bundles = f.get_available_bundles()
print(f\"Bundles: {len(bundles)}\")
for b in bundles[:5]:
    print(f\"  {b.displayName} msv={b.minimumSelectorVersion}\")
'"
```

## Pitfalls, Gotchas, and Foot Guns

### Silent total failure: capnp enum miss
If the JSON model list introduces a type string not in the capnp enum,
`_parse_model()` throws `AttributeError` from pycapnp. If `parse_models()`
uses a bare list comprehension, the entire model list returns empty with
no user-visible error. Always wrap per-bundle parsing in try/except.

### C++ `-Werror` build failure on unhandled enum
Adding a capnp enum value generates a new C++ enumerant. Any
`switch(model.getType())` without a case for it fails scons with `-Werror`.
Catch this locally with `scons ... models_panel.o` before deploying.

### `index == 0` truthiness bug
`if (x := params.get("ModelManager_DownloadIndex")):` is falsy when the
model index is `0`. Always use `is not None`.

### Params type mismatches
`ModelManager_LastSyncTime` is stored as int→str. `ModelManager_ModelsCache`
is stored as JSON string. Upstream syncs sometimes reintroduce raw dict
writes — verify the cache round-trips.

### capnp enum ordering
Capnp enums are positional (`@N` ordinals). Never reorder or insert.
Always append with the next `@N`.

### Full rebuild after capnp changes
Any `cereal/custom.capnp` change invalidates the scons cache for most C++
targets. Expect 15-20 minute full rebuild on-device.

### Runtime half-support on `offPolicy` bundles
Adding `offPolicy` to capnp + the offroad UI is **not** enough. v15 bundles
such as `OMV4` are three-artifact runtime bundles: `policy` can carry
`planplus`, `vision` can carry `pose`/`meta`/`hidden_state`, and `offPolicy`
can carry `plan`, `lane_lines`, `road_edges`, `lead`, and `lead_prob`.
If `TinygradSplitRunner` only runs `vision` + `policy`, onroad can stay
unhealthy with calibration stuck at 0%.

### Standalone `planplus` parsing
For three-artifact bundles, `planplus` may live in the `policy` artifact
without `plan` in the same output dict. `sunnypilot/modeld_v2/parse_model_outputs_split.py`
must parse `planplus` independently, not only inside `if 'plan' in outs:`.

### `tinygrad_ref` mismatch can hard-crash `modeld_tinygrad`
The v15 model list publishes a top-level `tinygrad_ref`. If your branch's
`tinygrad_repo` content does not match that ref, newer compiled tinygrad
pickles can crash during `pickle.load(...)` before onroad publishes any
`modelV2`/`cameraOdometry` data. On affected branches this shows up as:

- `modeld_tinygrad` never stays running
- calibration stuck at `0%`
- `MDL` red with downstream `LOC`/`PLN`/`PRM` red or yellow
- swaglog traceback from `tinygrad/device.py` with a buffer `size mismatch`

Check both:

- remote JSON `tinygrad_ref`
- your branch's actual `tinygrad_repo` state

Do not assume `tinygrad_repo` is a real git checkout; some branches vendor it
as a plain tree snapshot, which makes `get_tinygrad_ref()` ineffective.
When that happens, add a tracked `tinygrad_repo/.vendored_ref` file and make
`sunnypilot/models/tinygrad_ref.py` read it before looking for `.git`.

After any tinygrad runtime sync, smoke-test the actual compiled pickles before
declaring victory. On device, run `pickle.load(...)` on the active
`driving_vision_*_tinygrad.pkl`, `driving_policy_*_tinygrad.pkl`, and
`driving_off_policy_*_tinygrad.pkl` under the target runtime. If that still
crashes in `tinygrad/device.py`, calibration will remain stuck at `0%` because
`modeld_tinygrad` never reaches steady state.

### `CapturedJit` input metadata rename can crash startup after a tinygrad sync
After syncing to newer tinygrad runtimes, compiled model pickles may still load
cleanly while `modeld_tinygrad` crashes during runner initialization. Newer
tinygrad exposes `model_run.captured.expected_input_info`; older code in
`sunnypilot/models/runners/tinygrad/tinygrad_runner.py` may still read the
legacy `expected_st_vars_dtype_device` field. When that happens, startup fails
with:

- `AttributeError: 'CapturedJit' object has no attribute 'expected_st_vars_dtype_device'`
- `modeld_tinygrad` exits with code `1`
- calibration may briefly move if you manually launch `modeld`, then stop again

Keep a compatibility helper/fallback in `tinygrad_runner.py` for both field
names when vendoring newer tinygrad runtimes.

### Gitlink vs vendored tree diff trap
Upstream sunnypilot tracks `tinygrad_repo` as a gitlink/submodule, while some
branches vendor it as a plain tree. `git diff sp/master:tinygrad_repo` is not a
real tree diff and can fail or mislead. To audit local drift correctly:

- resolve the upstream tinygrad commit from `sp/master:tinygrad_repo` or remote
  JSON `tinygrad_ref`
- checkout or archive that commit from the actual tinygrad repo
- diff that real tree against local `tinygrad_repo/`

### Onroad selector can look empty even when the model list is healthy
`models_manager` is registered as `PythonProcess("models_manager", ..., only_offroad)`.
If `models_panel.cc` is changed to allow browsing onroad, there will be no live
`modelManagerSP` publisher while `deviceState.started == True`.

Symptoms:

- the selector shows only the default `phoenix` entry
- the current-model label falls back to `phoenix`
- `ModelFetcher(Params()).get_available_bundles()` still returns the full list
- `ModelManager_ModelsCache` and `ModelManager_ActiveBundle` params are populated

The first broken contract is UI data sourcing, not fetch/parsing. Fix by either:

- re-gating the selector to offroad, or
- making `models_panel.cc` read cached params when the live message is absent

Fast proof on device:

- `deviceState.started == True`
- `managerState.processes["models_manager"].running == False`
- `managerState.processes["models_manager"].shouldBeRunning == False`

### Upstream sync overwrites local fixes
Upstream sunnypilot syncs may completely replace `helpers.py`, `fetcher.py`,
`manager.py`, and the tinygrad split runtime files. Re-verify defensive
patterns (try/except in parser, `is not None` checks, JSON encode/decode in
cache, `offPolicy` runner support, standalone `planplus` parsing,
`tinygrad_ref` compatibility, and vendored tinygrad metadata) after every
sync.

### JSON keys vs capnp field names
Remote JSON uses `snake_case` (`minimum_selector_version`). Capnp uses
`camelCase` (`minimumSelectorVersion`). `_parse_bundle()` reads snake_case.
`is_bundle_version_compatible()` reads camelCase from `.to_dict()`.

## Skill Self-Improvement (mandatory on every invocation)

After completing any model-manager update task, **before finishing**,
review what happened during this session and update both copies of this
skill with any new information learned. This is not optional, and it does
**not** require a user prompt.

Treat this skill as a kaizen loop: every real invocation should leave it
more accurate, more actionable, or more compact than it was before. The
skill should continuously improve as a matter of course while being used,
not only when the user explicitly asks for documentation maintenance.

### Continuous Improvement Policy

- Always compare new evidence from the current session against existing
  guidance before finishing.
- If the new evidence proves an existing bullet incomplete, stale, or wrong,
  **correct or replace** it immediately instead of adding a contradictory note.
- Prefer editing or deleting obsolete guidance over endlessly appending new
  warnings.
- If a new lesson affects nearby workflow steps, file inventory, or reference
  history, update those adjacent sections in the same pass.
- If a failure passes through multiple layers, record the **first broken
  contract** and the most reliable validation command, not just the final
  downstream symptom.
- Keep both skill copies in sync automatically; do not wait for the user to
  ask for the second copy to be updated.

### What to update

- **New gotchas/foot guns** — any failure mode you hit that isn't already
  documented above. Add it to the "Pitfalls" section.
- **New files touched** — if the update required changing a file not in
  the File Inventory table, add it.
- **Version history** — append the new version bump to
  `references/version_history.md` with commit hash, version numbers,
  URL, and a short note about what was new/different.
- **New model types** — update the Model Type Evolution table in
  `references/version_history.md`.
- **Workflow changes** — if a step in the workflow proved wrong, incomplete,
  or needed reordering, fix it.
- **Corrections** — if any existing guidance was wrong or misleading,
  correct it rather than adding a contradictory note.

### Where to update (both locations, always)

1. **Claude Code**: `~/.claude/skills/model-manager-updater/SKILL.md`
   and `~/.claude/skills/model-manager-updater/references/version_history.md`
2. **Codex CLI**: `.codex/skills/model-manager-updater/SKILL.md`
   and `.codex/skills/model-manager-updater/references/version_history.md`

Keep both copies substantively in sync. The only expected differences are
path references (Codex uses `.codex/skills/...` paths, Claude Code uses
`~/.claude/skills/...` paths).

### Quality bar

- Only add things that caused real failures or wasted real time.
- Prefer concrete guidance (file path, command, config key) over vague warnings.
- Prefer correction and replacement over accumulation.
- Remove bullets that become obsolete (e.g., a bug was fixed in code/config).
- Keep the skill compact. If it grows too large, factor detail into
  `references/` files and link to them rather than letting stale bulk pile up.

## References

- Version history: `.codex/skills/model-manager-updater/references/version_history.md`
- Upstream remote: `sp` → `https://github.com/sunnypilot/sunnypilot.git`
- Model list URL pattern: `https://raw.githubusercontent.com/sunnypilot/sunnypilot-docs/refs/heads/gh-pages/docs/driving_models_v<N>.json`
