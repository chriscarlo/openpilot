# Codex Slash Commands (v0.30.0)

This document lists the built‑in slash commands available in the Codex TUI for codex-cli 0.30.0 and explains how to use them.

How to invoke:
- Type `/` in the composer to open the command palette.
- Type to filter; Up/Down to select.
- Tab autocompletes the highlighted command into the composer (e.g., `/model `).
- Enter executes the selected built‑in command. Esc closes the popup.
- User prompts: Any Markdown files under `$CODEX_HOME/prompts/*.md` (usually `~/.codex/prompts`) appear in this list by name; selecting one sends its contents as your message.

Built‑in commands (kebab‑case):
- `/model` — choose what model and reasoning effort to use (opens a popup of presets).
- `/approvals` — choose what Codex can do without approval (approval policy + sandbox).
- `/new` — start a new chat during a conversation.
- `/init` — create an AGENTS.md file with instructions for Codex (sends a starter prompt).
- `/compact` — summarize the conversation to free context.
- `/diff` — show git diff (including untracked files).
- `/mention` — insert `@` to mention a file (opens file search when you type after `@`).
- `/status` — show current session configuration and token usage.
- `/mcp` — list configured MCP tools.
- `/logout` — log out of Codex and exit.
- `/quit` — exit Codex.
- `/test-approval` — test approval request (debug builds only; not present in release builds).

Notes and tips:
- Arguments: Built‑in slash commands in 0.30.0 do not parse inline arguments. For example, typing `/model gpt-5-high` will still open the model picker; the `gpt-5-high` text is not parsed.
- Fast model switching:
  - Use CLI on start: `codex -m gpt-5 -c 'model_reasoning_effort="high"'` or a profile like `codex -p gpt5h`.
  - In‑session: type `/model`, then Enter to open the picker; arrow‑keys to select a preset, Enter to confirm.
- Custom prompts:
  - Create `~/.codex/prompts/<name>.md` to add items to the `/` palette. Selecting a custom prompt sends its file contents as your message (it does not execute an action like a built‑in command).
- Status: `/status` displays the active model, reasoning effort, approval mode, sandbox, and current token usage.

Where custom prompts are loaded from:
- Codex scans `$CODEX_HOME/prompts` for `*.md` files (on this setup, `~/.codex` typically points to `/data/.codex`). Non‑Markdown files and subdirectories are ignored.

Limitations (0.30.0):
- No `/model <preset>` inline selection; only the interactive picker.
- Slash commands cannot change CLI‑only options (e.g., `--sandbox` in non‑TUI contexts). Use CLI flags or profiles when launching.

See also:
- Flags and CLI options: `docs/codex/flags.md`
