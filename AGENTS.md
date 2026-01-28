# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `selfdrive/` (controls, modeld, UI helpers) and `system/` (on-device services: `loggerd`, `camerad`, `proclogd`).
- Messaging schemas in `cereal/`; hardware and safety in `panda/`.
- Vendored submodules live under `*_repo/` (e.g., `opendbc_repo/`, `tinygrad_repo/`).
- Fork features/UI: `sunnypilot/`. Chauffeur docs and examples: `docs/chauffeur/`.
- Tests sit alongside modules and at repo root as `test_*.py`.

## Build, Test, and Development Commands
- Environment check: confirm whether you're on the TICI production environment or the WSL Ubuntu dev environment; some scripts, paths, and hardware access differ.
- Environment/persistence report: see `/persist/PERSISTENCE_AND_ENVIRONMENT_REPORT_*.md` (latest). When work depends on storage durability, mount options, partition sizes, or hardware limits, consult this report first and regenerate if stale.
- Create env: `python -m venv .venv && source .venv/bin/activate`.
- Install deps (dev+tests): `pip install -e ".[testing,dev]"`.
- Build native targets: `scons -j$(nproc)` (add `--stock-ui` to build stock UI).
- Fast tests: `pytest -m 'not slow'`; full suite: `pytest`.
- Lint & types: `scripts/lint/lint.sh` (runs `ruff`, `mypy`, `codespell`, etc.).
- Chauffeur examples: `pytest docs/chauffeur -q` (target a file to iterate faster).

- Mapd source inspection (no vendoring):
  - To clone upstream `openpilot-mapd` sources into an ignored cache dir for ad-hoc review, run:
    - `bash scripts/dev/fetch_mapd_source.sh` (defaults to `.cache/openpilot-mapd`)
    - Pin a ref: `bash scripts/dev/fetch_mapd_source.sh -r <tag|branch|commit>`
  - Do not commit these sources; they are for inspection only. Runtime still uses the installed binary at `third_party/mapd_pfeiferj/mapd`.

## Coding Style & Naming Conventions
- Python: 2-space indent, type hints encouraged; files use `snake_case.py`.
- C/C++: Clang/Clang++ (C++17); follow `.clang-tidy`; warnings are errors in SCons.
- Keep functions small, documented; remove dead code and unused params.

## Offroad UI Style Standards
- All offroad settings menus must follow the Chauffeur Offroad Settings UI Brand Style Guide.
- Source of truth: `docs/chauffeur/ui/bsg/offroad/offroad_settings_bsg.md`.
- Panels should mirror the RTI Settings submenu visual language (titles, section cards, toggles, range controls), including:
  - Title 50px/600 centered; section headers 42px/500; control labels 36px.
  - Section cards use `#292929` background, 20px radius, 25px padding; screen margins 50/20/50/20.
  - Toggles use `ToggleSP` 150×80 right-aligned; ± controls are 100×100 circles; Reset 150×80.
  - Descriptions use 32–34px body text, `#999999`.
  - Group related rows into logical sections (e.g., Visibility & Lookahead, Developer Options).

## Testing Guidelines
- Framework: `pytest` with parallelization (`-n auto` if configured).
- Mark long tests with `@pytest.mark.slow`; device-only with `@pytest.mark.tici`.
- Name tests `test_*.py`; place next to code or under a module `tests/` dir.

## Commit & Pull Request Guidelines
- Commits: imperative mood with scoped prefix, e.g., `selfdrive: fix MPC latency`.
- Branch policy: work only on `chauffeur-dev3` in this workspace. PRs target `chauffeur-dev3`.
  - Note: `chubbs-merge` is not an active development branch right now; use it only for debugging comparisons or targeted ports when explicitly requested.
  - Note: `chauffeur-dev2` is deprecated.
- PRs: include rationale, verification steps (routes/logs for car changes), linked issues, and tests. Use templates in `.github/pull_request_template.md`.

## Agent Planning (update_plan tool)

- When the plan tool is available (session launched with `-c include_plan_tool=true`), always maintain a concise task plan using the `update_plan` tool.
- Create the plan at task start, then update it after each meaningful change (file edits, commands, test runs).
- Keep 3–6 steps total. Use statuses: `pending`, `in_progress`, `completed`.
- Exactly one step may be `in_progress` at a time.
- Prefer small, verifiable steps; revise instead of appending long tails.
- If the tool is not enabled, ask to enable it: “Launch Codex with `-c include_plan_tool=true` (e.g., `codexh`/`codexl`) so I can publish plan updates.”

Example intent (do not paste literally; invoke the tool):
- explanation: short reason for changes when the plan structure shifts
- plan:
  - { step: "Scan repo and confirm env", status: completed }
  - { step: "Pin scope and create patch", status: in_progress }
  - { step: "Run fast tests", status: pending }
  - { step: "Summarize changes + next steps", status: pending }

## UI Controls — Sunnypilot Builds
- When building under `SUNNYPILOT`, always use the SP widget variants or alias to them:
  - `LabelControlSP`, `ListWidgetSP`, `ToggleControlSP`, `ParamControlSP`, `LayoutWidgetSP`.
  - You may `#define` base names to SP names inside `#ifdef SUNNYPILOT` blocks for shared code.
- Using base controls in SP builds can leave layouts uninitialized and crash the UI on widget creation.

## Device Editing Policy
- Do not modify files directly on the device. Use the device only for logs, tracing, and context.
- Make code/documentation changes in this repo and push; pull from the device to avoid divergence.

## Chauffeur Porting Rules
- Source branch: `chubbs-ssh-only`. Only port requested changes.
- No submodules: repository is flattened. Vendor code into local `*_repo/...` paths.
- Do not modify `.gitmodules` or add symlinks. Fix imports/includes to local paths.
- Example mapping: `opendbc/` → `opendbc_repo/opendbc/` (import path `opendbc` remains).

## Security & Configuration Tips
- Do not commit private keys, large binaries, or personal drive logs. Use Git LFS when needed.
- Changes to controls/safety require clear justification and tests.

## Logs: Rlogs/Qlogs vs Swaglogs (on-device)

- Where (segments): `/data/media/0/realdata/<dongle>--<route>--<seg>/`
  - Rlog: `rlog.zst` (or `rlog.bz2`)
  - Qlog: `qlog.zst` (or `qlog.bz2`)
  - Video: `fcamera.hevc`, `ecamera.hevc`, optional `qcamera.ts`

- Where (swaglogs): `/data/log/swaglog.*` (rotating newline‑delimited JSON files)

- Parse rlogs/qlogs with `tools/lib/logreader.py` (handles .zst/.bz2):
  - Example (extract VTSCDBG from rlogs):
    `python - <<'PY'
from openpilot.tools.lib.logreader import LogReader
import glob, json
for p in sorted(glob.glob('/data/media/0/realdata/*--*--*/rlog.zst'))[-20:]:
  for m in LogReader(p):
    if m.which()=='logMessage':
      s = m.logMessage              # outer swaglog JSON string embedded in the capnp event
      try:
        outer = json.loads(s)
      except Exception:
        continue
      msg = outer.get('msg','')
      if msg.startswith('VTSCDBG '):
        d = json.loads(msg.split('VTSCDBG ',1)[1])
        print(p, d.get('vision_status'), d.get('v_occ'), d.get('v_vis'))
        break
PY`

- Parse qlogs similarly (force qlog mode):
  - `from openpilot.tools.lib.logreader import LogReader, ReadMode`
  - `for m in LogReader('/data/media/0/realdata/<...>/qlog.zst', default_mode=ReadMode.QLOG): ...`

- Parse swaglogs (newline‑delimited JSON, no capnp):
  - Example:
    `python - <<'PY'
import json, glob
for path in sorted(glob.glob('/data/log/swaglog.*'))[-5:]:
  with open(path,'r',encoding='utf-8',errors='ignore') as f:
    for line in f:
      try:
        rec = json.loads(line)
      except Exception:
        continue
      if isinstance(rec.get('msg'), str) and rec['msg'].startswith('VTSCDBG '):
        d = json.loads(rec['msg'].split('VTSCDBG ',1)[1])
        print(path, d.get('vision_status'), d.get('v_occ'), d.get('v_vis'))
        break
PY`

- Quick grep on compressed rlogs (coarse):
  - `zstd -dc /data/media/0/realdata/<...>/rlog.zst | strings | rg -n "VTSCDBG|Route 50"`

- Tips:
  - Use `ls -lt /data/media/0/realdata | head` to find most recent routes; segment numbers grow over time.
  - For time windows, filter by file mtime or by `created` field inside swaglog JSON lines.
  - Prefer `LogReader` for correctness and speed over raw `zstd | strings` when extracting fields.

## VTSC Monitoring Quick-Start (On-Device)
- Ensure VTSC debug toggles are ON in Offroad → Cruise → VTSC → Settings:
  - `Verbose VTSC Debug Logging` (`VTSCVerboseDebug`)
  - `Write Onroad VTSC Snapshots (JSONL)` (`VTSCWriteSnapshotFile`)
- Start watcher: `nohup python3 tools/vtsc/vtsc_watch.py > .cache/vtsc_watch.out 2>&1 & echo $! > .cache/vtsc_watch.pid`
- Tail output during drive: `tail -f .cache/vtsc_watch.out`
- Snapshots file: `/data/media/0/VTSCDebug/vtsc_snapshots.jsonl` (used by the analyzer)
- Post-drive: `python docs/chauffeur/vtsc/analysis/analyze_snapshots.py /data/media/0/VTSCDebug/vtsc_snapshots.jsonl --dump-tsv OUT.tsv`
- Organize results under `docs/chauffeur/vtsc/debug/debug_YYYY-MM-DD/` as per `docs/chauffeur/vtsc/AGENTS.md`.

## Agent Macros: VTSC Live Monitoring

- Trigger: `/monitor vtsc` or phrase “monitor the vtsc”.
- Agent will:
  - Ensure Params toggles are ON: `VTSCVerboseDebug=1`, `VTSCWriteSnapshotFile=1`.
  - Start watcher: `nohup python3 tools/vtsc/vtsc_watch.py > .cache/vtsc_watch.out 2>&1 & echo $! > .cache/vtsc_watch.pid`.
  - Confirm snapshot file exists: `/data/media/0/VTSCDebug/vtsc_snapshots.jsonl`.
  - Tail output live to the user and surface flags only when requested (`rg -n "flags=(?!-)" .cache/vtsc_watch.out`).
  - If manager/UI are down, relaunch `python3 system/manager/manager.py` and verify `controlsd`, `plannerd`, and UI daemons.
  - After the drive, offer analyzer: `python docs/chauffeur/vtsc/analysis/analyze_snapshots.py <snapshots.jsonl> --dump-tsv OUT.tsv`.
- Stop: `/monitor vtsc stop` → kill PID from `.cache/vtsc_watch.pid`.

Watcher flags explained:
- `freeway_failopen_missed`: Straight, long visibility, good confidence but occlusion cap active.
- `double_occl_cap_suspect`: Raw target ≈ occlusion cap while occlusion cap is active.
- `pretrigger_with_high_conf`: Pretrigger reason while confidence ≥ 0.70.
- `psi_below_thresh`: Occlusion active but `psi_vis < psi_thresh`.

## Agent Macros: VTSC Full-Trace Replay

- Trigger: `/trace vtsc <rlog_path> [--max-frames N] [--cruise MPS] [--disable-failopen]`
- Agent will:
  - Validate `<rlog_path>` exists and is readable.
- Run: `python docs/chauffeur/vtsc/fullTrace/full_trace_replay.py <rlog_path> --out .cache/vtsc_full_trace.jsonl [--max-frames N] [--cruise MPS] [--disable-failopen]`.
  - On success, print the output path, frame count (`wc -l`), and a small head/tail sample.
  - Optionally copy to docs (when requested): `docs/chauffeur/vtsc/testing/examples/<derived_name>.jsonl` for sharing/prompts.
  - If needed, re-run with adjusted flags for iteration.

Examples:
- `/trace vtsc docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_00000085--f247b281ca--67.zst --max-frames 200`
- `/trace vtsc /data/media/0/realdata/<dongle>--<route>--<seg>/rlog.zst --disable-failopen`
