---
name: context-file-librarian
description: >
  Audits and maintains repo instruction context files: keeps AGENTS.md minimal, verifiable, and
  front-loaded, and enforces that CLAUDE.md and .claude/CLAUDE.md (if present) are symlinks to the
  canonical AGENTS.md. Triggers on phrases like "update AGENTS.md", "audit instruction drift",
  "reduce AGENTS.md bloat", "these repo instructions are stale/wrong", "enforce CLAUDE.md symlink",
  and "add a new landmine/gotcha to agent instructions". Do NOT trigger for feature work, bug
  fixes, tests, or refactors unless the user explicitly asks to maintain instruction files.
---

# Context File Librarian

## Purpose

Keep instruction entrypoints accurate, minimal, and non-duplicative:

- `AGENTS.md` is the *only* canonical truth.
- `CLAUDE.md` and `.claude/CLAUDE.md` (if they exist) must be symlinks to `AGENTS.md`.
- Instruction files should be short, front-loaded, and only contain **high-impact, non-obvious,
  verifiable** requirements and landmines.

This skill is for **instruction maintenance**, not application development.

## Positive triggers (use this skill)

Use `$context-file-librarian` when the user says (or strongly implies) any of:

- “update AGENTS.md”
- “audit instruction drift”
- “reduce AGENTS.md bloat”
- “these repo instructions are stale/wrong”
- “enforce CLAUDE.md symlink”
- “add a new landmine/gotcha to agent instructions”

## Negative triggers (do not use this skill)

Do **not** switch into this skill for:

- “implement feature”
- “fix bug”
- “write tests”
- “refactor code”

…unless the user explicitly asks to maintain instruction/context files as part of the request.

## Scope & safety (hard requirements)

### Files you may edit

Only edit:

- `AGENTS.md`
- `AGENTS.override.md` (only if present)
- `CLAUDE.md` (only if present; enforce symlink to `AGENTS.md`)
- `.claude/CLAUDE.md` (only if present; enforce symlink to `AGENTS.md`)
- `.claude/rules/*.md` (only if present; remove duplication / convert to minimal pointers)

### Files you must not edit

- Never change application code or behavior.
- If other instruction/memory files exist (e.g. `.claude/*.md` beyond the entrypoints), you may
  **report** them, but do not edit them under this skill unless the user explicitly expands scope.

### Symlink policy

- If `CLAUDE.md` and/or `.claude/CLAUDE.md` do not exist: **do not create them**; report “none present”.
- If they exist and are not symlinks to `AGENTS.md`: replace with symlinks.
- If symlink replacement is unsafe due to platform constraints, report exactly why and what would be
  needed to safely enforce it.

## Workflow

### Step 1 — Discovery

1) Locate:
   - canonical `AGENTS.md` (repo root)
   - `AGENTS.override.md` (if present)
   - `CLAUDE.md` / `.claude/CLAUDE.md` (if present)
   - `.claude/rules/*.md` (if present)
2) Confirm whether the Claude entrypoints are symlinks and where they point.

### Step 2 — Generate an audit report (scripted)

Run:

```bash
python3 .agents/skills/context-file-librarian/scripts/audit_context_files.py
```

Use the report to drive deterministic, reviewable edits.

### Step 3 — Audit checks (must cover all)

For the canonical `AGENTS.md` (and `AGENTS.override.md` if present), check:

1) **Redundancy / bloat**
   - Remove overviews, architecture essays, directory listings, dependency encyclopedias.
2) **Unverifiable / stale claims**
   - Commands/paths/policies must be evidenced in-repo or moved to “Needs human confirmation”.
3) **Anchoring risk**
   - Requirements that could mislead if wrong: verify or quarantine.
4) **Placement risk**
   - Move the highest-impact constraints to the top; keep sections short and front-loaded.
5) **Canonicalization drift**
   - Any `CLAUDE.md` / `.claude/CLAUDE.md` that is not a symlink to `AGENTS.md` must be replaced.

### Step 4 — Deterministic edit strategy (reviewable)

For each finding, choose one action:

- **DELETE** (remove redundant or low-signal content)
- **REWRITE SHORTER** (keep requirement, compress phrasing, keep it locally verifiable)
- **MOVE TO “Needs human confirmation”** (when correctness cannot be proven from this repo)

Guardrails:

- Preserve the drift policy: add bullets only after real failures; remove when fixed/obvious.
- Avoid “policy sprawl”: prefer concrete commands, file paths, and proven landmines.

### Step 5 — Maintenance loop (3-pass)

Perform edits in three passes:

1) **Generate:** propose minimal correctness-first `AGENTS.md`.
2) **Reflect:** self-critique against redundancy, unverifiable claims, anchoring hazards; cross-check repo evidence.
3) **Curate:** apply final edits; ensure short, structured, front-loaded.

### Step 6 — Enforce canonicalization (after editing AGENTS.md)

1) Make `AGENTS.md` canonical and final.
2) Enforce symlinks (only if the entrypoints exist):
   - `CLAUDE.md` -> `AGENTS.md`
   - `.claude/CLAUDE.md` -> `../AGENTS.md`
3) If `.claude/rules/*.md` exist, prune duplication and convert to minimal pointers to `AGENTS.md`.

### Step 7 — Finish with a clean review surface

Required outputs for the user:

- A regenerated audit report (after changes)
- Clean `git diff` (only intended instruction/symlink changes)
- A concise changelog of what changed and why

