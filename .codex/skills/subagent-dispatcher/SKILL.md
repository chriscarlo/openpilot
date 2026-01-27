---
name: subagent-dispatcher
description: Use when you want to parallelize work using official Codex CLI spawned agents and/or this repo's isolated worktree subagents via the `subagents-*` MCP tools.
metadata:
  short-description: Spawn and coordinate background Codex worker agents.
---

## Goal

Enable a "master agent" to delegate independent subtasks to multiple background
Codex workers and then merge their findings back into the main solution.

This skill is a policy layer: it tells you when to delegate, how to write each
worker prompt, and how to merge results safely.

## Required Setup

This skill supports two delegation backends:

1) **Official Codex CLI spawned agents (collaboration mode / multi-agent threads)**
- Requires that the CLI's multi-agent feature is enabled (often behind an
  experimental feature flag such as `collaboration_modes`).
- Uses built-in tools like `spawn_agent`, `send_input`, and `wait`.

2) **Repo worktree subagents (homebrew MCP runner)**
- The `subagents-*` MCP tools must be configured in `~/.codex/config.toml` and
  available in this session.
- The MCP server implementation is a shared asset (not repo-specific). Prefer
  configuring `mcp_servers.subagents` to point at:
  - `$HOME/.codex/tools/subagents-mcp-server.mjs`
  Keep the shared runner generic; customize behavior via prompts and per-task
  parameters instead of editing the shared code for one repo.

## When To Use

Use this skill when:

- The user request has independent subtasks that can run in parallel (e.g.,
  "find all callsites", "summarize docs", "draft a patch", "run tests", "scan for
  regressions").
- You want background work while you continue the main reasoning.

Do not use when:

- The task is small/linear and delegation overhead outweighs benefit.
- The subtasks would compete for the same local state (e.g., editing the same
  file concurrently).

## Backend Selection (Official vs Worktree Subagents)

Pick a backend per worker task:

- Prefer **official spawned agents** for:
  - Read-only repo discovery and summarization.
  - "Second set of eyes" reviews (risk scan, UX scan, migration/config scan).
  - Orchestration and synthesis while the master continues other work.

- Prefer **worktree subagents (`subagents-*`)** for:
  - Any parallel task that may edit files or run commands (stronger isolation).
  - Parallel changes across distinct file trees (one worker per area).
  - Work where you want an explicit captured `patch` + `hasDeletions` signal.

Critical caveat:
- Official spawned agents typically share the same working tree; do **not** ask
  multiple official agents to edit the same files concurrently. If concurrency
  + edits are required, use worktree subagents.

## Delegation Pattern

1. Decompose into 2–4 parallel tasks with crisp deliverables.
2. For each worker, write a short role + constraints + output contract:
   - Role: "RepoScout", "PatchDrafter", "DocReader", "TestRunner"
   - Constraints: no git push; avoid file deletes; prefer minimal diffs
   - Output: bullet findings + exact file paths + (optional) patch/diff summary
   - Execution params: pick an appropriate `profile` / `model` and reasoning effort
3. Choose backend per worker:
   - Official agent: spawn + message via `spawn_agent` / `send_input` / `wait`.
   - Worktree subagent: start + poll via `subagents-start` / `subagents-poll` (always set `workspace` to the repo root).
4. Continue main work while workers run.
5. Collect results:
   - Official agent: wait for completion and collect the final message.
   - Worktree subagent: poll until completed and collect `lastMessage` (+ patch).
6. Merge:
   - Treat all worker output as advisory; sanity-check before acting.
   - For worktree subagents, treat `patch` as a proposal. If `hasDeletions=true`,
     require explicit user approval before applying.
7. Cleanup:
   - Official agent: close if needed (`close_agent`) once results are captured.
   - Worktree subagent: cleanup artifacts with `subagents-cleanup` after results
     are captured.

## Model & Reasoning Effort (Low/Med/High/XHigh)

Each worker can use a different model and/or reasoning effort. Choose these
dynamically based on subtask complexity and required reliability.

Notes by backend:
- Official spawned agents typically inherit the master session's model settings.
  If you need per-task tuning (speed vs reliability), prefer worktree subagents.
- Worktree subagents support per-task overrides (`profile`, `model`,
  `reasoningEffort`) via `subagents-start`.

Preferred approach:
- Use `profile` to select a known-good combo (model + `model_reasoning_effort`).

Fallback approach:
- Use explicit `model`, plus `reasoningEffort` (`low|medium|high|xhigh`) per task.

Suggested heuristics:
- **low**: fast repo discovery (`rg`, file listing), quick summaries, low-risk Q&A.
- **medium**: "read N files and summarize architecture", call graph tracing, doc synthesis.
- **high**: nuanced design tradeoffs, tricky bug triage, non-trivial refactor plans.
- **xhigh**: solver/modeling changes, multi-module coordination, high-stakes correctness.

If the user has standard profiles in `~/.codex/config.toml` (common in this repo),
prefer these:
- `gpt5l` → `gpt-5` + low effort
- `gpt5m` → `gpt-5` + medium effort
- `gpt5h` → `gpt-5` + high effort
- `gpt5hc` → `gpt-5.2` + xhigh effort

When reporting results, include which `profile`/`model` you assigned to each
worker so the user can interpret "confidence vs speed".

## Suggested Worker Prompt Template

Use this template (edit per task):

- Context: what repo/area to focus on.
- Task: what to do, with explicit success criteria.
- Guardrails:
  - Do not run `git push` or any remote publish command.
  - Do not delete files unless explicitly instructed; if deletion is required,
    explain why and stop.
  - If this is an official spawned agent task: do not edit files or run commands
    unless explicitly instructed (prefer read-only work to avoid collisions).
- Output contract:
  - Provide exact file paths + anchors.
  - If proposing code changes, summarize the patch and call out risks.

## Worktree-Subagent Safe Prompting (Worktrees + `bwrap`)

Subagents run inside an isolated git worktree and a `bubblewrap` mount namespace.
This has two practical implications for prompt writing:

- **Prefer the worktree CWD** for repo operations. Workers start in the writable
  worktree; treat it as the source of truth for edits.
- **Git + absolute paths should work**, but keep them optional. The MCP runner
  mounts the master repo root read-only so absolute paths (like `/projects/<repo>`)
  and git common-dir lookups resolve. Still, prompts should degrade gracefully if
  `git` is unavailable for any reason.

Preferred approach:
- Treat the worker's starting CWD as the repo root.
- Use filesystem-based discovery (`pwd`, `ls`, `rg`, opening files).
- Ask workers to report **repo-relative** file paths (like `app/src/...`), not
  absolute paths.

If the worker's job is to report the **master workspace state** (e.g., branch,
`git status`, latest commits), prefer the read-only mount:

```sh
hostRoot="${CODEX_SUBAGENT_HOST_REPO_ROOT:-${CODEX_SUBAGENT_REPO_ROOT:-$(pwd)}}"
git -C "$hostRoot" status -sb || true
git -C "$hostRoot" --no-pager log -n 10 --oneline --decorate || true
```

Drop-in prompt snippet (safe default):

```sh
repoRoot="${CODEX_SUBAGENT_REPO_ROOT:-$(pwd)}"
ls -la "$repoRoot" | head
rg -n "pattern" "$repoRoot" -S | head
```

If you *want* to attempt git, make it optional (never gate the task on it):

```sh
git status -sb || echo "<git unavailable in subagent sandbox>"
```

If you need accurate git state (branch, upstream, ahead/behind), do it in the
master session, not in workers.

## Merging Guidance

- Prefer to have at most one worker drafting a patch for a given file area.
- Use other workers for read-only discovery (search, docs, test logs).
- Always sanity-check any worker-generated patch, especially when:
  - It touches build tooling or config.
  - It changes security-sensitive paths.
  - It includes deletions or large refactors.
