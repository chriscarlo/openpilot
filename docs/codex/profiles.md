# Codex Profiles (config.toml)

This page documents everything that can be configured inside a profile block in `~/.codex/config.toml` for codex-cli 0.30.0, plus nearby options frequently used with profiles. Profiles let you switch model and reasoning presets quickly via `-p/--profile`.

## Where profiles live

- File: `~/.codex/config.toml` (aka `$CODEX_HOME/config.toml`)
- Section: `[profiles.<name>]`
- Select at launch: `codex -p <name>`

Example:

```
[profiles.gpt5h]
model = "gpt-5"
model_provider = "openai"
model_reasoning_effort = "high"        # minimal|low|medium|high
model_reasoning_summary = "detailed"   # auto|concise|detailed|none

[profiles.gpt5m]
model = "gpt-5"
model_provider = "openai"
model_reasoning_effort = "medium"
model_reasoning_summary = "detailed"
```

## Profile fields (authoritative)

The `ConfigProfile` schema accepts the fields below. All are optional; unspecified values inherit from top‑level defaults.

- `model` (string)
  - Model slug (e.g., `gpt-5`, `o3`); must match a supported provider slug.

- `model_provider` (string)
  - Key under `model_providers` to select (e.g., `openai`, `oss`).

- `approval_policy` (enum)
  - Controls when Codex asks before executing commands.
  - Values: `untrusted` (aka UnlessTrusted), `on-failure`, `on-request` (default), `never`.

- `model_reasoning_effort` (enum)
  - Reasoning effort for Responses API.
  - Values: `minimal`, `low`, `medium` (default), `high`.

- `model_reasoning_summary` (enum)
  - Whether to return a reasoning summary and at what detail.
  - Values: `auto` (default), `concise`, `detailed`, `none` (disable).

- `model_verbosity` (enum)
  - GPT‑5 `text.verbosity` control.
  - Values: `low`, `medium` (default), `high`.

- `chatgpt_base_url` (string)
  - Alternate base URL for ChatGPT (UI) backend requests. Default: `https://chatgpt.com/backend-api/`.

- `experimental_instructions_file` (path)
  - Replace built‑in base instructions with a file’s contents. Relative paths resolve against the working directory.

Notes:
- Profiles do not set sandbox policy directly. Use top‑level `sandbox_mode` or CLI.
- Profiles do not define MCP servers. Use top‑level `mcp_servers`.

## Related top‑level keys (outside profiles)

These are common alongside profiles and can be overridden at launch with `-c key=value` when needed.

- `sandbox_mode` (enum): `read-only` | `workspace-write` | `danger-full-access`.
- `approval_policy` (enum): global default when not set in profile.
- `shell_environment_policy` (table): controls environment inheritance for shell/local_shell tools.
  - `inherit`: `all` (default) | `core` | `none`
  - `ignore_default_excludes`: bool
  - `exclude`: [regex strings]
  - `set`: { NAME = "VALUE", ... }
  - `include_only`: [regex strings]
  - `experimental_use_profile`: bool
- `tools` (table): feature toggles
  - `web_search`: bool — expose `web_search` tool to the model
  - `view_image`: bool — enable `view_image` tool to attach local images
- `history` (table)
  - `persistence`: `save-all` (default) | `none`
  - `max_bytes`: int (reserved)
- `file_opener`: `vscode` | `vscode-insiders` | `windsurf` | `cursor` | `none`
- `projects` (table): per‑path trust config, e.g. `projects = { "/data/openpilot" = { trust_level = "trusted" } }`
- `mcp_servers` (table of servers): `{ command = "bash", args = ["-lc", "server"], env = { KEY = "VAL" } }`
- `model_providers` (table): extend/override providers (base URLs, auth, etc.).
- `hide_agent_reasoning`: bool — hide reasoning bubble in TUI.
- `show_raw_agent_reasoning`: bool — show raw reasoning content.
- `experimental_instructions_file`: path — top‑level variant (see profile field).
- `experimental_resume`: path — internal rollout replay file (undocumented; for dev use).
- `notify`: [program and args] — external notifier for user messages.
- `tui` (table): reserved for future TUI options.

CLI override‑only (not in config.toml schema):
- `include_plan_tool` (bool)
- `include_apply_patch_tool` (bool)
- `include_view_image_tool` (bool)
- `tools_web_search_request` (bool)

You can pass these via `codex -c key=true` at launch; they are not read from config.toml.

## Hidden and special CLI flags

Available in 0.30.0 but hidden from `--help`:
- `--resume` — open resume picker (interactive list of recent sessions)
- `--continue` — continue the most recent session

Aliases:
- `--dangerously-bypass-approvals-and-sandbox` has alias `--yolo`.

## Quick recipes

- Switch profiles:
  - `codex -p gpt5h`

- Force a different approval policy for one run:
  - `codex -p gpt5h -a on-request`

- Enable web search for a session (without editing config):
  - `codex -c tools.web_search=true`

- Use a different ChatGPT base during testing:
  - `codex -p gpt5h -c chatgpt_base_url="https://chatgpt.com/backend-api/"`

