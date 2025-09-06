# Codex CLI Flags (v0.30.0)

This document lists the Codex CLI flags and their function for codex-cli 0.30.0 as installed in this environment. It includes hidden but supported flags present in this build.

Version: codex-cli 0.30.0  
Binary: `$(which codex)`

See also: `docs/codex/slash_commands.md` for in‑session slash commands.

## Global Invocation

Usage:

```
codex [OPTIONS] [PROMPT]
codex [OPTIONS] [PROMPT] <COMMAND>
```

- `PROMPT` (positional): Optional initial user prompt. If omitted, an interactive TUI starts.

## Global Flags

- `-c, --config <key=value>`
  - Override values from `~/.codex/config.toml`. Dotted paths supported (e.g., `-c model="o3"`, `-c 'shell_environment_policy.inherit=all'`). Values parse as JSON first; falls back to raw string.

- `-i, --image <FILE>...`
  - Attach one or more images to the initial prompt.

- `-m, --model <MODEL>`
  - Select model (e.g., `gpt-5`).

- `--oss`
  - Convenience flag: `-c model_provider=oss` and verifies local Ollama.

- `-p, --profile <CONFIG_PROFILE>`
  - Select a config profile defined in `~/.codex/config.toml` under `[profiles.*]`.

- `-s, --sandbox <SANDBOX_MODE>`
  - Sandbox policy for shell commands. Values: `read-only`, `workspace-write`, `danger-full-access`.

- `-a, --ask-for-approval <APPROVAL_POLICY>`
  - When to require user approval for commands. Values: `untrusted`, `on-failure`, `on-request`, `never`.

- `--full-auto`
  - Convenience alias for `-a on-failure` and `--sandbox workspace-write`.

- `--dangerously-bypass-approvals-and-sandbox` (alias: `--yolo`)
  - Disable sandboxing and approvals entirely. For externally sandboxed environments only.

- `-C, --cd <DIR>`
  - Run with the specified working directory as project root.

- `--search`
  - Enable web search tool in the session (off by default).

- `-h, --help`
  - Show help.

- `-V, --version`
  - Show version.

### Hidden (but available) flags in 0.30.0

- `--resume`
  - Open an interactive picker to resume a prior session recorded on disk. Mutually exclusive with `--continue`. Hidden from `--help` in 0.30.0 but supported.

- `--continue`
  - Continue the most recent session (equivalent to selecting the newest item in the resume picker). Mutually exclusive with `--resume`. Hidden from `--help` in 0.30.0 but supported.

Notes:
- `--resume`/`--continue` first appeared in `rust-v0.29.1-alpha.3`, were visible through `alpha.7`, and became hidden in `0.30.0`.

## Commands

- `exec` (alias: `e`) — Non-interactive run
  - Usage: `codex exec [OPTIONS] [PROMPT]`
  - Flags (in addition to Global):
    - `--skip-git-repo-check` — Allow running outside a Git repo.
    - `--color <always|never|auto>` — Control color output (default `auto`).
    - `--json` — Emit JSONL events to stdout.
    - `--output-last-message <FILE>` — Write the last agent message to a file.

- `apply` (alias: `a`) — Apply latest diff as `git apply`
  - Usage: `codex apply [OPTIONS] <TASK_ID>`
  - Args:
    - `<TASK_ID>` — The task whose latest diff to apply.
  - Flags: inherits `-c/--config` and `-h/--help`.

- `login` — Manage login
  - Usage: `codex login [OPTIONS] [COMMAND]`
  - Subcommands:
    - `status` — Show login status.
  - Flags: `--api-key <API_KEY>`, `-c/--config`, `-h/--help`.

- `logout` — Remove stored credentials
  - Flags: `-c/--config`, `-h/--help`.

- `mcp` — Experimental: run as an MCP server
  - Flags: `-c/--config`, `-h/--help`.

- `proto` — Run the Protocol stream via stdin/stdout
  - Flags: `-c/--config`, `-h/--help`.

- `completion` — Generate shell completions
  - Usage: `codex completion [OPTIONS] [SHELL]`
  - Args: `[SHELL]` — `bash`, `elvish`, `fish`, `powershell`, `zsh` (default `bash`).
  - Flags: `-c/--config`, `-h/--help`.

- `debug` — Internal debugging
  - Usage: `codex debug [OPTIONS] <COMMAND>`
  - Subcommands: `seatbelt` (macOS), `landlock` (Linux), `help`.
  - Flags: `-c/--config`, `-h/--help`.

## Examples

- Start interactive with model and sandbox tuned:
  - `codex -m gpt-5 --full-auto`

- Non-interactive JSON stream from prompt file, outside git:
  - `codex exec --json --skip-git-repo-check - < prompt.md`

- Resume/continue (hidden in help but supported):
  - `codex --resume`
  - `codex --continue`

## See Also

- Config file: `~/.codex/config.toml` (profiles, defaults, sandbox, approvals)
- Session history: `~/.codex/history.jsonl`
