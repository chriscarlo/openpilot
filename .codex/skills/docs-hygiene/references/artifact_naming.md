# Artifact Naming + Layout (anti “final-final-2”)

Goal: make artifacts grep-able, comparable, and reproducible without guessing which “final” is final.

## Principles

- Prefer **stable identifiers** over adjectives.
- Encode “what/where/when” in the name: subsystem, date/time, env, git SHA, run number.
- Avoid judgement words: `final`, `fixed`, `test`, `reverted`, `new`, `old`, `latest`.

## Recommended patterns

Pick one and be consistent inside a folder:

1) **Run-numbered** (best when iterating quickly)
- `plannerd_run0001_wsl_2026-02-27T1117Z_a1b2c3d.log`
- `mtscd_run0004_tici_2026-02-27T1942Z_a1b2c3d.log`

2) **Timestamp-first** (best when sorting by time is primary)
- `2026-02-27T1117Z_wsl_a1b2c3d_plannerd_run0001.log`

## Folder layout

- Durable writeups (committed):
  - `docs/chauffeur/<area>/debug/debug_YYYY-MM-DD/<slug>/README.md`
  - `docs/chauffeur/<area>/experiments/experiment_YYYY-MM-DD/<slug>/README.md`
- Raw artifacts (untracked by default):
  - `.cache/doc_artifacts/<area>/<debug_YYYY-MM-DD|experiment_YYYY-MM-DD>/<slug>/...`

## Bad patterns (avoid)

- `*_final.log`, `*_final2.log`, `*_fixed.log`, `*_test.log`
- `no-shit-for-realizez-final-final-2.log`
- `latest.log` (meaningless once you create a new “latest”)

## If you must keep “final” semantics

Put “finalness” in the **README**, not the filename:

- In `README.md`, add: “Chosen run: run0004 (reason: …)”
- Leave artifacts named by stable identifiers.

