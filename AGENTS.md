# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `selfdrive/` (controls, modeld, UI helpers) and `system/` (on-device services: `loggerd`, `camerad`, `proclogd`).
- Messaging schemas in `cereal/`; hardware and safety in `panda/`.
- Vendored submodules live under `*_repo/` (e.g., `opendbc_repo/`, `tinygrad_repo/`).
- Fork features/UI: `sunnypilot/`. Chauffeur docs and examples: `docs/chauffeur/`.
- Tests sit alongside modules and at repo root as `test_*.py`.

## Build, Test, and Development Commands
- Environment check: confirm whether you're on the TICI production environment or the WSL Ubuntu dev environment; some scripts, paths, and hardware access differ.
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
- Branch policy: work only on `chubbs-merge` in this workspace. PRs target `chubbs-merge`.
  - Note: `chauffeur-dev2` is deprecated. We will still reference it to port custom features into `chubbs-merge` and for debugging comparisons when needed.
- PRs: include rationale, verification steps (routes/logs for car changes), linked issues, and tests. Use templates in `.github/pull_request_template.md`.

## Chauffeur Porting Rules
- Source branch: `chubbs-ssh-only`. Only port requested changes.
- No submodules: repository is flattened. Vendor code into local `*_repo/...` paths.
- Do not modify `.gitmodules` or add symlinks. Fix imports/includes to local paths.
- Example mapping: `opendbc/` → `opendbc_repo/opendbc/` (import path `opendbc` remains).

## Security & Configuration Tips
- Do not commit private keys, large binaries, or personal drive logs. Use Git LFS when needed.
- Changes to controls/safety require clear justification and tests.
