# Chauffeur Agent Instructions (Claude)

Read `AGENTS.md` in this repo root and follow it as the source of truth for this project — non-obvious requirements, landmines, verification steps, all of it. It was written for Codex, so treat its instructions as directions to follow, not text to imitate literally: where it names a Codex-specific mechanism, do whatever Claude Code needs to do to achieve the same result. Do not edit `AGENTS.md` to "translate" it; adapt on the fly instead.

`.claude/skills/` is a symlink to `.codex/skills/`, so the same skill files back both agents — no duplicate skills to maintain. The same adapt-on-the-fly rule applies inside them.

Examples of adapting Codex-isms to Claude Code:
- AGENTS.md tells you to read a skill's `agents/` subfolder before delegating a subtask -> use the Agent tool (or a matching custom subagent under `.claude/agents/`) to do that step, using the same acceptance criteria.
- AGENTS.md or a skill references a Codex-only CLI flag or tool name -> use whichever Claude Code tool (Read/Edit/Bash/etc.) produces the same outcome.
- A skill's `scripts/` helper assumes Codex's working-directory conventions -> run it the same way but through Bash, adjusting invocation as needed for this environment.
