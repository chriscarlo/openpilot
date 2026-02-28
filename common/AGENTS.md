# common/ — Agent Instructions

## Non-obvious requirements (must follow)
- Avoid editing or committing build artifacts that live in this tree (examples present here include `*.o`, `*.a`, `*.so`); make changes in the corresponding sources (`*.cc`, `*.h`, `*.py`, `*.pyx`) instead.

## Landmines / gotchas (things that fail silently)
- This directory mixes Python, C++, and bindings; verify you changed the actual implementation layer (not just a wrapper).

## Verification / definition of done
- Run the smallest relevant test: `pytest common/tests` (or a specific test file under `common/tests/`).
- Confirm `git diff` does not include object/library artifacts unless the user explicitly requested it.

## Updating this file (drift policy)
- Add a bullet only after a real failure (e.g., someone edited artifacts or missed the correct source layer).

## Needs human confirmation (temporary; keep very short)
- None.
