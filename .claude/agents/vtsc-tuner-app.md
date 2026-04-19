---
name: vtsc-tuner-app
description: Edit, debug, troubleshoot, iterate, and polish the standalone Rust/egui VTSC sigmoid tuner at `tools/vtsc_tuner/`. Use when changing the hero-plot interactions, the plain-English knobs, EQ bands, apply/commit/push/pull-to-tici pipeline, undo/redo, theming/scale, or when debugging WSLg-specific window crashes and detached-launch issues. Triggers include "vtsc tuner app", "vtsc sigmoid tuner", "knob widget", "apply chain", "Revert to baseline", "Pull on tici", and anything under `tools/vtsc_tuner/`.
---

Use the canonical repo-local instructions at `.codex/skills/vtsc-tuner-app/SKILL.md`.

Workflow:
1. Read `.codex/skills/vtsc-tuner-app/SKILL.md` before acting.
2. Consult `.codex/skills/vtsc-tuner-app/references/` for architecture, math, apply-chain contract, and changelog.
3. Treat `.codex/skills/vtsc-tuner-app/` as the source of truth. If the workflow changes, update the canonical skill first and keep this wrapper aligned.
4. Record every substantive change in `.codex/skills/vtsc-tuner-app/references/changelog.md` (reverse-chronological, dated).
