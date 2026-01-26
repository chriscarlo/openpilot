# VTSC Docs (Baseline Index) — Agent Instructions

This folder is the **baseline, repo-wide reference index** for **Vision Turn Speed Control (VTSC)**.

It is intentionally small and stable:
- It should answer “where is VTSC implemented?” in under a minute.
- It should point to deeper, experimental, or session-specific VTSC work elsewhere (not duplicate it).

## Directory Contents (expected)
- `docs/vtsc/README.md`: Entry point / orientation / quick links.
- `docs/vtsc/INVENTORY.md`: “Where is everything?” map of code, UI, Params, tools, docs, and tests.
- `docs/vtsc/OCCLUSION.md`: Occlusion logic deep dive (what/why + lead-bypass expectations).
- `docs/vtsc/TESTS.md`: Existing VTSC test suites + how to run + where to extend.
- `docs/vtsc/PARAMS.md`: VTSC-related Params keys, defaults source, and UI mapping.
- `docs/vtsc/TOOLING.md`: Debugging + analysis tooling (on-device + offline).
- `docs/vtsc/RLOGS.md`: Rlog fixture provenance + replay guidance (tici-sourced requirement).
- `docs/vtsc/GLOSSARY.md`: Shared terminology (kappa/curvature, psi, occlusion, etc.).
- `docs/vtsc/CHANGELOG.md`: Change log for this docs folder (keep it updated).

If you add new docs here, prefer adding one of:
- `docs/vtsc/<TOPIC>.md` for stable concepts (architecture, interfaces, invariants), or
- a new subfolder `docs/vtsc/<topic>/` only when the topic clearly grows beyond ~1 doc.

## What Future Agents Must Do

When VTSC-related code changes (or when asked to investigate VTSC):
1) **Update `docs/vtsc/INVENTORY.md`**
   - Add/remove/rename file pointers.
   - Note when logic moves between folders (e.g., `sunnypilot/` ↔ `selfdrive/`).
2) **Update `docs/vtsc/TESTS.md`**
   - Record new/modified test suites.
   - Record any new harnesses and the recommended place to put them.
3) **Update `docs/vtsc/PARAMS.md`**
   - Add new Params keys and/or adjust descriptions.
   - Defaults source of truth is `common/params_keys.h`.
4) **Update `docs/vtsc/TOOLING.md`**
   - Add any new scripts (watchers, analyzers, replay tools) and how to run them.
5) **Append to `docs/vtsc/CHANGELOG.md`**
   - Use a date-stamped entry.
   - Include what changed and why (brief, factual).

## What NOT To Put Here
- Large logs, dumps, rlog extracts, or one-off debug session artifacts.
  - Put those under the existing VTSC debug workspace: `docs/chauffeur/vtsc/` (it has its own `AGENTS.md` and conventions).
- Generated data (large `.jsonl`, `.tsv`, plots) unless it’s a tiny example (<~50 KB) that is genuinely reusable.

## Rlog Data Source Rule (for VTSC regressions)
When adding or referencing rlog-based VTSC test cases, ensure:
- rlogs originate from a real drive on **TICI / comma3x** (even if copied onto a dev laptop),
- the case folder records provenance (device, segment id, what bug/invariant it represents),
- and `docs/vtsc/RLOGS.md` + `docs/vtsc/INVENTORY.md` are updated accordingly.

## “Deep Dive” VTSC Work Lives Elsewhere
This repo already has a large VTSC work area with debug sessions, replays, and test benches:
- `docs/chauffeur/vtsc/`

This `docs/vtsc/` folder should link to it, but avoid copying it.
