# docs/chauffeur/vtsc — Agent Instructions

Avoid vague AI-isms such as "clean" or "concrete" when describing work; describe the specific property instead.

## Non-obvious requirements (must follow)
- Prefer durable writeups under `docs/chauffeur/vtsc/` and keep raw/large artifacts untracked under `.cache/` unless explicitly requested.
- For debug sessions, create/extend a dated folder under `docs/chauffeur/vtsc/debug/` (existing convention: `debug_YYYY-MM-DD/`) and include a clear human summary (`NOTES.md` or equivalent).

## Landmines / gotchas (things that fail silently)
- On-device paths referenced in VTSC docs (`/data/media/0/...`, `/data/log/...`) vary by device/setup; confirm before hardcoding or assuming they exist.

## Verification / definition of done
- Any new/updated workflow doc includes: environment assumptions, exact commands, and expected outputs/paths.
- If you add/modify a script under `docs/chauffeur/vtsc/` or `tools/vtsc/`, run it at least once on a small input (or validate `--help`/usage).

## Updating this file (drift policy)
- Keep this file small; move detailed procedures into the specific doc they belong to.

## Needs human confirmation (temporary; keep very short)
- None.
