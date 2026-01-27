# Subagents and Multi-Agent Workflows (Codex CLI)

This repo supports **two complementary ways** to parallelize work in Codex:

1) **Official Codex CLI spawned agents** (collaboration / multi-agent threads)
   - Best for: read-only repo discovery, doc synthesis, risk reviews, background research.
   - Caveat: spawned agents typically share the same working tree. Avoid parallel edits.

2) **Worktree subagents (homegrown MCP runner)**
   - Best for: parallel tasks that may run commands or edit code without stepping on each other.
   - Provides: isolation via per-task git worktrees + patch capture (`patch`, `hasDeletions`).

The repo-scoped dispatcher skill teaches the master agent how to choose:
- `.codex/skills/subagent-dispatcher/SKILL.md`

## When to use which

Use **official spawned agents** when:
- You want 2-4 "scouts" to find callsites, trace data flow, or summarize subsystems.
- You want multiple independent reads of specs/PRs/issues and a consolidated brief.
- You want a "second set of eyes" review (safety, UX, config/migration risk).

Use **worktree subagents** when:
- Tasks may run commands, modify files, or otherwise risk workspace collisions.
- You want to split work across file trees (e.g., `selfdrive/...` vs `system/...` vs `tools/...`).
- You want captured diffs/patches for review before merging.

## Worktree subagents: configuration

The worktree subagents runner is an MCP server that exposes tools:
- `subagents-start`
- `subagents-poll`
- `subagents-cancel`
- `subagents-cleanup`

### Safety model (worktrees + bubblewrap)

The runner is designed for "yolo-ish" execution without having workers stomp on
each other:
- Each task gets an isolated Codex `HOME` (separate sessions/logs).
- Each task gets an isolated git worktree (separate working trees).
- Tasks run inside a `bubblewrap` mount namespace with a minimal filesystem view:
  - task worktree (rw)
  - a read-only bind of the original repo root (for absolute-path and git common-dir resolution)
  - task HOME (rw)
  - minimal OS/runtime paths (ro)
  - shared network namespace (web search works)

### Environment helpers

Each worktree subagent gets a few convenience env vars:
- `CODEX_SUBAGENT_REPO_ROOT`: the worker's writable worktree path (also the initial CWD).
- `CODEX_SUBAGENT_WORKDIR`: same as `CODEX_SUBAGENT_REPO_ROOT`.
- `CODEX_SUBAGENT_HOST_REPO_ROOT`: the master checkout path (mounted read-only).
- `CODEX_SUBAGENT_USED_WORKTREE`: `1` when a worktree was created; otherwise `0`.

Recommended setup (global, shared across repos): add this to `~/.codex/config.toml`:

```toml
[mcp_servers.subagents]
command = "node"
args = ["/home/<you>/.codex/tools/subagents-mcp-server.mjs"]
working_directory = "/projects"
```

Notes:
- After editing `~/.codex/config.toml`, restart Codex CLI so it reloads MCP tools.
- Use an absolute path in `args` (do not rely on `~` expansion).
- When calling `subagents-start`, always pass `workspace` explicitly (the repo root) so it works regardless of the MCP server working directory.

### Shared resource policy (keep it generic)

The worktree-subagents MCP server script is intentionally a **shared resource**
across projects (BudgetCal, Chauffeur, and any future repo). Treat the code at:

- `$HOME/.codex/tools/subagents-mcp-server.mjs` (example: `/home/chris/.codex/tools/subagents-mcp-server.mjs`)

as the canonical implementation.

Caveats:
- Do not add repo-specific behavior (paths, env vars, domain assumptions) to the
  shared runner just to improve one project. That tends to break other repos.
- Prefer customizing **how you call** subagents instead:
  - worker prompts (role + constraints + output contract)
  - `workspace` per task/repo
  - per-task `profile` / `model` / `reasoningEffort`
  - per-task `sandboxMode` / `enableSearch`

If the runner truly needs an improvement, implement it in the shared script in a
repo-agnostic way, then update docs as needed.

### Per-task controls

Each worker task can optionally set:
- `profile`: selects a profile from `~/.codex/config.toml` (recommended)
- `model`: explicit model override
- `reasoningEffort`: `low` | `medium` | `high` | `xhigh`
- `sandboxMode`: `read-only` | `workspace-write` | `danger-full-access`
- `enableSearch`: enable/disable web search

If omitted, workers inherit your root `model` and `model_reasoning_effort` defaults from `~/.codex/config.toml`.

## Official spawned agents: enablement

Official spawned-agent support is typically gated behind an experimental feature.
You can check current feature flags with:

```sh
codex features list
```

Look for flags related to collaboration / multi-agent (often `collaboration_modes`).
Enable the feature per your Codex CLI version's instructions.

## Skill usage

In a Codex prompt, you can explicitly request the dispatcher:
- `$subagent-dispatcher`

Or you can just describe the need ("run 3 parallel repo scouts and summarize"), and the master agent should choose a backend and delegate appropriately.
