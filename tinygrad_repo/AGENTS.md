# tinygrad_repo — Agent Instructions

## Non-obvious requirements (must follow)
- `tinygrad_repo/` is vendored into this repo; keep changes minimal and tightly scoped.
- Do not mix functional changes with whitespace-only reformatting.

## Landmines / gotchas (things that fail silently)
- Repo-root `pytest` ignores `tinygrad_repo/` (see `pyproject.toml` `addopts --ignore=tinygrad_repo/`); run tinygrad tests explicitly.

## Verification / definition of done
- From repo root: `cd tinygrad_repo && pytest`
- Confirm `git diff` contains only intended tinygrad source changes (avoid accidental churn in vendored/generated files).

## Updating this file (drift policy)
- Keep this file focused on integration pitfalls with the parent repo, not tinygrad philosophy or architecture.

## Needs human confirmation (temporary; keep very short)
- Any additional upstream tinygrad formatting/test gates expected beyond `pytest`.
