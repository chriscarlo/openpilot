# Agent Instructions (Canonical)

Avoid vague AI-isms such as "clean" or "concrete" when describing work; describe the specific property instead.

## Instruction maintenance
Use `$context-file-librarian` to audit and maintain instruction entrypoints and prevent drift/duplication (this is instruction-file work only; never application code).

Examples:
- "Use $context-file-librarian to update AGENTS.md and remove stale/unverifiable bullets."
- "Run $context-file-librarian and enforce the CLAUDE.md symlink to AGENTS.md."
- "Audit instruction drift and reduce AGENTS.md bloat with $context-file-librarian."

Trigger phrases:
- "update AGENTS.md"
- "audit instruction drift"
- "reduce AGENTS.md bloat"
- "these repo instructions are stale/wrong"
- "enforce CLAUDE.md symlink"
- "add a new landmine/gotcha to agent instructions"

## Non-obvious requirements (must follow)
- `AGENTS.md` files are the source of truth; any `CLAUDE.md` in this repo should be a symlink to the nearest-scope `AGENTS.md`.
- Avoid branch/history-changing git commands (`checkout`, `switch`, `merge`, `rebase`) unless the user explicitly requests them.
- For Sunnypilot Qt UI (`sunnypilot/` and `selfdrive/ui/sunnypilot/`), prefer `*SP` widgets from `selfdrive/ui/sunnypilot/qt/widgets/controls.h`.
- Offroad settings UI must follow `docs/chauffeur/ui/bsg/offroad/offroad_settings_bsg.md`.
- Keep raw artifacts untracked under `.cache/` (`.gitignore` includes it); never commit secrets or personal drive logs.

## Landmines / gotchas (things that fail silently)
- `pytest` runs with strict/parallel defaults (see `pyproject.toml`): `-Werror`, `--strict-markers`, and `-n auto`.
- Repo-root `pytest` intentionally ignores some subtrees via `pyproject.toml` `addopts --ignore=...` (e.g. `tinygrad_repo/`); confirm your changes are actually exercised by the tests you ran.
- If you add a new pytest marker, also add it to `pyproject.toml` `[tool.pytest.ini_options].markers` (strict markers are enforced).
- Claude Code installs a hook that denies branch-changing git commands (`.claude/hooks/block-branch-changes.py`); use `git show origin/<branch>:path` / `git diff <current>..origin/<branch>` when you only need to inspect.

## Verification / definition of done
- Run the smallest relevant check for your change:
  - Python: `pytest <touched_dir_or_test_file>`
  - Quick suite (skip slow): `pytest -m 'not slow'`
  - Lint/types: `scripts/lint/lint.sh`
  - C/C++/Qt/SCons: `scons -j$(nproc)`
- Confirm `git diff` only contains intended source/docs (no accidental `*.o` or `moc_*.cc` churn).

## Updating this file (drift policy)
- Add a bullet only after a real agent/user failure that was not prevented by existing tooling/tests.
- Prefer concrete, locally verifiable guidance (exact file path, command, or config key).
- Remove bullets once the underlying footgun is fixed in code/config or becomes obvious from repo defaults.

## Device access
- SSH profiles (in `~/.ssh/config`; which one works depends on current network/SSID):
  - `commaHome` — home Wi-Fi (192.168.1.172)
  - `commaCar` — car hotspot (192.168.0.229)
  - `commaAdb` — USB via adb port-forward (127.0.0.1:2222, key `~/.ssh/id_comma_device`)
- All use user `comma`. `sudo` requires no password (NOPASSWD).
- Repo on device: `/data/openpilot` (tracks `chauffeur-dev4`).
- Tici only: the device build venv is `/usr/local/venv`; use `source /usr/local/venv/bin/activate` and `/usr/local/venv/bin/scons` rather than assuming `scons` is on `PATH`.
- Deploy workflow: `git push` from dev machine, then `ssh <profile> "cd /data/openpilot && git pull && sudo reboot"`.

## Needs human confirmation (temporary; keep very short)
- Any workflow expectations that live outside this repo (device deployment steps, protected branch name(s), etc.).
