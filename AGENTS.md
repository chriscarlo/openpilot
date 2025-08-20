# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `selfdrive/` (controls, modeld, UI helpers) and `system/` (on-device services: `loggerd`, `camerad`, `proclogd`).
- Messaging schemas in `cereal/`; hardware and safety in `panda/`.
- Vendored submodules live under `*_repo/` (e.g., `opendbc_repo/`, `tinygrad_repo/`).
- Fork features/UI: `sunnypilot/`. Chauffeur docs and examples: `docs/chauffeur/`.
- Tests sit alongside modules and at repo root as `test_*.py`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`.
- Install deps (dev+tests): `pip install -e ".[testing,dev]"`.
- Build native targets: `scons -j$(nproc)` (add `--stock-ui` to build stock UI).
- Fast tests: `pytest -m 'not slow'`; full suite: `pytest`.
- Lint & types: `scripts/lint/lint.sh` (runs `ruff`, `mypy`, `codespell`, etc.).
- Chauffeur examples: `pytest docs/chauffeur -q` (target a file to iterate faster).

## Coding Style & Naming Conventions
- Python: 2-space indent, type hints encouraged; files use `snake_case.py`.
- C/C++: Clang/Clang++ (C++17); follow `.clang-tidy`; warnings are errors in SCons.
- Keep functions small, documented; remove dead code and unused params.

## Testing Guidelines
- Framework: `pytest` with parallelization (`-n auto` if configured).
- Mark long tests with `@pytest.mark.slow`; device-only with `@pytest.mark.tici`.
- Name tests `test_*.py`; place next to code or under a module `tests/` dir.

## Commit & Pull Request Guidelines
- Commits: imperative mood with scoped prefix, e.g., `selfdrive: fix MPC latency`.
- Branch policy: work only on `chauffeur-dev2` in this workspace. PRs target `chauffeur-dev2`.
- PRs: include rationale, verification steps (routes/logs for car changes), linked issues, and tests. Use templates in `.github/pull_request_template.md`.

## Chauffeur Porting Rules
- Source branch: `chubbs-ssh-only`. Only port requested changes.
- No submodules: repository is flattened. Vendor code into local `*_repo/...` paths.
- Do not modify `.gitmodules` or add symlinks. Fix imports/includes to local paths.
- Example mapping: `opendbc/` → `opendbc_repo/opendbc/` (import path `opendbc` remains).

## Security & Configuration Tips
- Do not commit private keys, large binaries, or personal drive logs. Use Git LFS when needed.
- Changes to controls/safety require clear justification and tests.
- Do not modify `.claude/` or `docs/chauffeur/claude/`.

