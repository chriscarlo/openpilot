---
name: docs-hygiene
description: Documentation workflow + artifact hygiene for this openpilot repo. Use when creating/updating docs (README.md, docs/chauffeur/*), writing debug/experiment notes, or capturing/organizing logs/traces/snapshots so artifacts don't end up as ambiguous "final-final-2" files in the repo root.
---

# Docs Hygiene

## Overview

Create reproducible documentation and keep generated artifacts organized and searchable.
This skill provides a repeatable folder layout, naming rules, and a helper script to scaffold a doc bundle (README + artifact dirs).

## Quick Start (new doc bundle)

Use the helper script to create a structured documentation folder (plus an untracked artifacts directory):

```sh
python3 .codex/skills/docs-hygiene/scripts/new_doc_bundle.py \
  --area mtsc \
  --kind debug \
  --slug plannerd-mtsc-investigation \
  --env wsl
```

Then fill in the generated `README.md` (problem statement, repro commands, environment, results, follow-ups) and put any raw logs in the printed `.cache/...` artifacts path.

## Workflow (doc + artifacts)

1) Decide what is **durable documentation** vs **raw artifacts**:
- Durable, shareable docs: commit under `docs/` (typically `docs/chauffeur/<area>/...`).
- Raw artifacts (logs, traces, dumps): keep untracked under `.cache/` unless explicitly small + sanitized.

2) Create a doc bundle with `scripts/new_doc_bundle.py` (recommended).
- Put the writeup in the generated `README.md`.
- Put raw logs/traces in the printed `.cache/...` directory (already gitignored).

3) Name things so they’re stable and searchable.
- Do **not** use `final`, `fixed`, `test`, `reverted`, or variants like `final-final-2`.
- Prefer identifiers: date/time, short slug, git SHA, environment (tici/wsl/pc), and a run number.
- See `references/artifact_naming.md` for concrete patterns and examples.

4) Redact/sanitize before committing any artifacts.
- Never commit private keys, large binaries, or personal drive logs.
- Scrub dongle IDs / route IDs / VINs / serials when in doubt; commit summaries over raw logs.

## Where things go (default)

- Debug/experiment writeups: `docs/chauffeur/<area>/{debug|experiments|notes}/...`
- Untracked run artifacts: `.cache/doc_artifacts/<area>/...`
- Avoid dumping artifacts into repo root.

## Resources

- `scripts/new_doc_bundle.py`: scaffolds a doc folder + README template + an untracked artifacts directory.
- `references/artifact_naming.md`: naming rules and examples for artifact files/folders.
