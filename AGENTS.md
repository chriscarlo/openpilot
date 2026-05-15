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

## Needs human confirmation (temporary; keep very short)
- Any workflow expectations that live outside this repo (device deployment steps, protected branch name(s), etc.).

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update tasks/lessons.md with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes -- don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -- then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. Plan First: Write plan to tasks/todo.md with checkable items
2. Verify Plan: Check in before starting implementation
3. Track Progress: Mark items complete as you go
4. Explain Changes: High-level summary at each step
5. Document Results: Add review section to tasks/todo.md
6. Capture Lessons: Update tasks/lessons.md after corrections

### Updating this file (drift policy)
- Add a bullet only after a real agent/user failure that was not prevented by existing tooling/tests.
- Prefer concrete, locally verifiable guidance (exact file path, command, or config key).
- Remove bullets once the underlying footgun is fixed in code/config or becomes obvious from repo defaults.

## Core Principles

- Simplicity First: Make every change as simple as possible. Impact minimal code.
- No Laziness: Find root causes. No temporary fixes. Senior developer standards.
- Minimal Impact: Only touch what's necessary. No side effects with new bugs.

## Device access
- SSH profiles (in `~/.ssh/config`; which one works depends on current network/SSID):
  - `commaHome` — home Wi-Fi (192.168.1.172)
  - `commaCar` — car hotspot (192.168.0.229)
  - `commaAdb` — USB via adb port-forward (127.0.0.1:2222, key `~/.ssh/id_comma_device`)
- All use user `comma`. `sudo` requires no password (NOPASSWD).
- Repo on device: `/data/openpilot` (tracks `chauffeur-dev4`).
- Tici only: the device build venv is `/usr/local/venv`; use `source /usr/local/venv/bin/activate` and `/usr/local/venv/bin/scons` rather than assuming `scons` is on `PATH`.
- Deploy workflow: `git push` from dev machine, then `ssh <profile> "cd /data/openpilot && git pull && sudo reboot"`.

## Verification / definition of done
- Run the smallest relevant check for your change:
  - Python: `pytest <touched_dir_or_test_file>`
  - Quick suite (skip slow): `pytest -m 'not slow'`
  - Lint/types: `scripts/lint/lint.sh`
  - C/C++/Qt/SCons: `scons -j$(nproc)`
- Confirm `git diff` only contains intended source/docs (no accidental `*.o` or `moc_*.cc` churn).