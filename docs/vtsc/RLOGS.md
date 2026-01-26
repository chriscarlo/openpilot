# VTSC Rlog Data (Fixtures + Replay)

This repo uses **rlog/qlog replay** as the closest practical approximation of “live” VTSC behavior without being on-road.

## Data provenance requirement (important)

For VTSC regressions (overslow / failure-to-recover / gating edge cases), rlogs should originate from:
- a real on-device drive on the target device hardware, and
- the actual production message pipeline running this fork.

In this repo/workspace, that means:
- **TICI / comma3x**.

Even if an rlog file lives in this git checkout on a dev laptop, treat it as **tici-sourced** if it was copied over from the device.

Why this matters:
- Simulator/desktop logs often differ in message timing and content (camera/model details, publish cadence, missing services).
- Those differences can mask the exact VTSC overslow/recovery issues we’re trying to reproduce.

## Where rlog fixtures live in this repo

Committed example rlogs (pulled from tici and checked into the repo as fixtures) live under:
- `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_00000085--f247b281ca--67.zst`
- `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/rlog_00000085--f247b281ca--80.zst`

There are also copies in the full-trace case folders:
- `docs/chauffeur/vtsc/fullTrace/cases/overslow_2025-09-05/.../rlog_*.zst`

## Replay entry points

Snapshot-only replay (fast, compact output):
- `python3 docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py <rlog.zst> --out .cache/vtsc_replay.jsonl --max-frames 500`

Method-level trace capture (slower, much richer output):
- `python3 docs/chauffeur/vtsc/fullTrace/full_trace_replay.py <rlog.zst> --out .cache/vtsc_full_trace.jsonl --max-frames 300`

Rlog-based regression tests (strict; run only when explicitly requested):
- `pytest -q -m regression docs/chauffeur/vtsc/fullTrace/tests/test_regressions_rlogs.py`

## Dependencies / portability

Rlog replay uses `LogReader` and requires:
- `pycapnp` (Python `capnp` module)
- zstd support for `.zst` logs

The regression tests in `docs/chauffeur/vtsc/fullTrace/tests/` skip automatically if `capnp` is not installed.

## Adding a new rlog-based VTSC case (process)

When you add a new “golden” VTSC case log:
1) Capture it on **tici / comma3x**.
2) Copy the segment from the device (`/data/media/0/realdata/<dongle>--<route>--<seg>/rlog.zst`) to your dev machine.
3) Store it under `docs/chauffeur/vtsc/cases/<case_name>/<segment_id>/rlog_<segment_id>.zst`.
4) Add a short `CASE_REPORT.md` describing:
   - what bug/invariant it represents,
   - how to reproduce it on-road,
   - and what a passing fix should change (qualitatively).
5) If you also generate full-trace artifacts, keep them under `docs/chauffeur/vtsc/fullTrace/cases/<case_name>/<segment_id>/`.
6) Update `docs/vtsc/INVENTORY.md` and `docs/vtsc/CHANGELOG.md`.

