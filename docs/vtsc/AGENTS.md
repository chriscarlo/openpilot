# docs/vtsc — Agent Instructions

Avoid vague AI-isms such as "clean" or "concrete" when describing work; describe the specific property instead.

## Non-obvious requirements (must follow)
- Treat `docs/vtsc/` as the stable VTSC index (links, inventories, invariants), not a dumping ground for one-off artifacts.
- Any VTSC code/doc movement must update `docs/vtsc/INVENTORY.md` and append to `docs/vtsc/CHANGELOG.md`.

## Landmines / gotchas (things that fail silently)
- Rlog-based VTSC cases have provenance constraints; follow `docs/vtsc/RLOGS.md` before adding fixtures or citing “golden” cases.

## Verification / definition of done
- After changes, confirm `docs/vtsc/INVENTORY.md` points at the new file locations (no dead links/path drift).
- Append a date-stamped entry to `docs/vtsc/CHANGELOG.md` describing what changed (brief + factual).

## Updating this file (drift policy)
- Prefer linking to the specific doc (`RLOGS.md`, `TESTS.md`, etc.) instead of duplicating procedures here.

## Needs human confirmation (temporary; keep very short)
- None.
