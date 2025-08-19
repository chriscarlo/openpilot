# Repository Guidelines

## Project Structure & Module Organization
- `selfdrive/`: Core processes (controls, modeld, UI helpers), SCons build entry.
- `system/`: On-device services (`loggerd`, `camerad`, `proclogd`, etc.).
- `cereal/`: Messaging schemas and generated code.
- `panda/`: Hardware interfaces and safety code.
- `opendbc_repo/`: Vehicle DBCs and car data; `tools/`: dev tools, replay, UI tests.
- `sunnypilot/`: Fork-specific features/UI integration.
- `docs/chauffeur/`: Chauffeur fork docs, notes, and test scripts.
- Tests live alongside modules and at repo root as `test_*.py`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -e ".[testing,dev]"`
- Build native targets: `scons -j$(nproc)` (add `--stock-ui` to build stock UI)
- Fast tests: `pytest -m 'not slow'`
- Full tests: `pytest` (markers available: `slow`, `tici`, `accuracy`, `schema`)
- Lint & type-check: `scripts/lint/lint.sh` (runs `ruff`, `mypy`, `codespell`, etc.)
 - Chauffeur examples: `pytest docs/chauffeur -q` (target a file for faster runs).

## Coding Style & Naming Conventions
- Python: 2-space indent (`.editorconfig`), type hints encouraged; keep modules and files `snake_case.py`.
- C/C++: Clang/Clang++ toolchain, C++17; follow `.clang-tidy`; warnings are errors in SCons.
- Tests: files named `test_*.py`; fixtures near tests; prefer small, deterministic cases.
- Keep functions small and documented; avoid dead code and unused params.

## Testing Guidelines
- Framework: `pytest` with parallelization (`-n auto`). Ensure local runs pass with repo defaults.
- Mark long-running tests with `@pytest.mark.slow` and gate device-only tests with `@pytest.mark.tici`.
- Add tests with behavior changes; place them next to the code or in the module’s `tests/` dir when present.

## Commit & Pull Request Guidelines
- Commits: imperative mood and scoped prefix, e.g., `selfdrive: fix MPC latency`.
- Branch policy: work only on `chauffeur-dev2` in this workspace. Upstream sync from `chubbs-ssh-only` is maintainer-controlled.
- PRs target `chauffeur-dev2`. Include description, rationale, verification steps (routes/logs for car changes), linked issues, and tests.
- For car ports/bugfixes, use the templates in `.github/pull_request_template.md`.

## Chauffeur Porting Rules
- Source branch: `chubbs-ssh-only` (maintainer syncs from upstream sunnypilot). Only port requested changes.
- No submodules in `chauffeur-dev2`: it is flattened. Do not add submodules or references to them.
- When porting, copy actual files from upstream submodules into their local, vendored paths here (e.g., `*_repo/...`), and fix imports/includes accordingly.
- Never modify `.gitmodules`; avoid symlink changes that reintroduce external refs. Build and tests must pass after the port.

Path mapping examples (upstream → local vendored):
- `opendbc/` → `opendbc_repo/opendbc/` (import path `opendbc` remains valid via local tree)
- `tinygrad/` → `tinygrad_repo/tinygrad/`
- `rednose/` → `rednose_repo/rednose/`
- `msgq/` → `msgq_repo/msgq/`
- `teleoprtc/` → `teleoprtc_repo/teleoprtc/`
- If an include/import references a submodule root, repoint it to the corresponding local vendored path above.

## Security & Configuration Tips
- Do not commit private keys, large binaries, or personal drive logs. Use Git LFS where appropriate.
- Follow safety rules; changes to controls/safety require clear justification and tests.
- Respect shared workspace: do not remove or modify files under `.claude/` or `docs/chauffeur/claude/`.
