#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/codex/debug.sh [TARGET]
#
# Launches Codex with a standardized debug intake prompt.
# Example: scripts/codex/debug.sh selfdrive/controls/controlsd.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

TARGET="${1-}"

PROMPT="/debug ${TARGET}

You are the Codex agent inside an active dev session. When the user triggers /debug, always run the same structured flow:

1) Confirm environment quickly (TICI vs WSL dev). If unknown, ask succinctly.
2) Build a short plan (3-6 steps) and mark steps as you go using the plan tool.
3) If TARGET is provided and exists, read it first in <=250 line chunks. If not provided, ask for scope.
4) Run fast checks relevant to scope (ruff/mypy/pytest -m 'not slow' or targeted tests), but only where appropriate and minimal.
5) Propose the smallest reproducible fix; prefer surgical patches. Avoid unrelated changes.
6) Offer to run tests/build if it validates the specific change.

Conventions:
- Use ripgrep (rg) for search, chunk reads <=250 lines.
- Prefer focused actions; avoid verbose explanations unless asked.
- Keep messages concise; summarize progress before tool calls.
- If ambiguity blocks progress, ask 1-2 pointed questions.

If TARGET looks like a file path, treat it as the primary artifact to inspect. Otherwise, interpret it as a search key.
"

exec codex -C "${ROOT_DIR}" "${PROMPT}"

