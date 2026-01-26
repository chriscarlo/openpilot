# VTSC Tooling (Debugging + Analysis)

This doc lists practical tooling around VTSC: on-device monitoring, snapshot logging, and offline analysis.

## On-Device Telemetry Outputs

VTSC can emit two main debug streams:

1) `VTSCDBG` lines (rate-limited JSON in logs)
- Enable via Param: `VTSCVerboseDebug` (bool)
- Emitted by: `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
- Destination: cloudlog / swaglog (read via rlog/qlog or `/data/log/swaglog.*`)

2) Snapshot file (JSONL, rotating)
- Enable via Param: `VTSCWriteSnapshotFile` (bool)
- Path: `/data/media/0/VTSCDebug/vtsc_snapshots.jsonl`
- Rotation: controller rotates at ~512 KB (moves to `.1`)

## Live Watcher

Primary watcher:
- `tools/vtsc/vtsc_watch.py`

It can tail:
- snapshot JSONL file, and/or
- swaglog JSON lines containing `VTSCDBG { ... }`

Typical usage:
```bash
python tools/vtsc/vtsc_watch.py
python tools/vtsc/vtsc_watch.py --snapshots
python tools/vtsc/vtsc_watch.py --swaglog
```

## Quick Offline Scan: VTSC vs “Physics From Vision”

Script:
- `tools/vtsc/analyze_vtsc_vs_vision.py`

Purpose:
- Scans recent `rlog.zst` segments under `/data/media/0/realdata/*--*--*/`
- Flags times VTSC target is below both current speed and “physics-from-vision” speed proxy.

Note:
- For meaningful VTSC signal behavior, these rlogs should come from a real on-device drive.
  - In this repo/workspace that means **TICI / comma3x**.

## Larger Offline Analysis + Replay (Docs Workspace)

The repo contains extensive VTSC offroad tooling under:
- `docs/chauffeur/vtsc/`

Notable entry points:
- `docs/chauffeur/vtsc/analysis/analyze_snapshots.py` (post-drive snapshot analysis)
- `docs/chauffeur/vtsc/offroad/replay_vtsc_on_rlog.py` (offline replay on a recorded rlog)
- `docs/chauffeur/vtsc/fullTrace/full_trace_replay.py` (method-level trace capture)

Rlog provenance guidance:
- `docs/vtsc/RLOGS.md` (includes the tici-sourced requirement and where committed fixtures live)

If you are creating new offline analyzers, prefer placing scripts in:
- `tools/vtsc/` (small reusable scripts), or
- `docs/chauffeur/vtsc/...` (analysis/replay pipelines tied to cases/reports)

## Where To Put Debug Session Artifacts

Do not store large logs in `docs/vtsc/`.

Instead, follow the established VTSC debug folder conventions:
- `docs/chauffeur/vtsc/debug/` (see `docs/chauffeur/vtsc/AGENTS.md`)
