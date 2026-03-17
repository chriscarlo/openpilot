# agnos-builder (placeholder) — Agent Instructions

Avoid vague AI-isms such as "clean" or "concrete" when describing work; describe the specific property instead.

## Non-obvious requirements (must follow)
- This `agnos-builder/` directory is a placeholder/warning guard; do not treat it as the canonical project.
- If you need the real builder sources, use `/projects/agnos/agnos-builder` (this path exists in this workspace).

## Landmines / gotchas (things that fail silently)
- Building/patching/deploying from this placeholder path can produce “successful” artifacts from the wrong codebase.

## Verification / definition of done
- Before producing build artifacts, confirm `pwd` is not under `.../openpilot/agnos-builder/`.
- Confirm `git diff` has no changes under `agnos-builder/` unless the user explicitly asked.

## Updating this file (drift policy)
- Add rules only after a real confusion incident involving this placeholder path.

## Needs human confirmation (temporary; keep very short)
- Owner/contact and intended update workflow for `/projects/agnos/agnos-builder`.
