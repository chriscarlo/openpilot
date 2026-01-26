# VTSC Inventory (Code / UI / Params / Tools / Tests / Docs)

This document records **all known VTSC-related locations** in this repo, grouped by role.

If you add/rename/move VTSC code, update this file first.

## 1) Core Control Logic (Most “Contained” Folder)

Primary implementation lives in `sunnypilot/selfdrive/controls/lib/`:
- `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
  - Defines `VisionTurnController` and supporting logic (physics mapping, occlusion handling, debug snapshots).
  - Includes a steering-curvature fallback used only when vision confidence is SEVERE and model curvature is flat (prevents “fail open” into real curves).
- `sunnypilot/selfdrive/controls/lib/vision_turn_params.py`
  - Implements `update_vtsc_params(...)` (debounced Params refresh + clamping).
- `sunnypilot/selfdrive/controls/lib/vision_turn_controller_backup.py`
  - A legacy/backup copy; not referenced by planner codepaths (treat as historical reference only unless re-wired).

## 2) Integration (Planner + Messaging)

Planner integration:
- `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
  - Instantiates `VisionTurnController` as `self.v_tsc`.
  - Calls `self.v_tsc.update(...)`.
  - Publishes to `longitudinalPlanSP.visionTurnSpeedControl`.
- `selfdrive/controls/lib/longitudinal_planner.py`
  - Main longitudinal planner inherits `LongitudinalPlannerSP` and uses `update_v_cruise(...)`.

Messaging schema:
- `cereal/custom.capnp`
  - `LongitudinalPlanSP.VisionTurnSpeedControl` struct and `VisionTurnSpeedControlState` enum.

Process manager note:
- `system/manager/process_config.py`
  - MTSC publisher process is disabled (“publisher removed; VTSC now consumes map lookahead directly”).

## 3) Params (Keys + Defaults)

Defaults/source of truth:
- `common/params_keys.h`
  - Defines `VisionTurnSpeedControl*` keys (plus `VTSCVerboseDebug`, `VTSCWriteSnapshotFile`, `VTSC.*` arbitration knobs, etc.).

Runtime Params refresh:
- `sunnypilot/selfdrive/controls/lib/vision_turn_params.py`

## 4) UI (Offroad + Onroad)

Offroad settings entry point:
- `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal_panel.cc`
  - Adds “Vision Turn Speed Controller” toggle + settings submenu navigation.

Offroad VTSC settings panels:
- `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vision_turn_control_with_settings.cc`
- `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_settings_panel.cc`
- `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_*_panel.{cc,h}`
  - Includes panels for filtering, smoothing, anticipation, physics, vision occlusion, etc.

Onroad HUD:
- `selfdrive/ui/qt/onroad/hud.h`
- `selfdrive/ui/qt/onroad/hud.cc`
  - Contains VTSC HUD fields and a lateral-accel meter widget (`drawVisionTurnControl`).
  - Note: the draw call is currently commented out in `HudRenderer::draw(...)` (widget exists but is not drawn).

UI strings / translations:
- `selfdrive/ui/translations/main_*.ts`
  - Contains “Vision Turn Speed Controller” translation entries used by the UI.

UI build wiring:
- `selfdrive/ui/sunnypilot/SConscript`
  - Ensures the VTSC settings panel sources are compiled into the Sunnypilot UI build.

## 5) Tools (Debugging / Analysis)

Small dedicated scripts:
- `tools/vtsc/vtsc_watch.py`
  - Live tailing of `VTSCDBG` lines and/or snapshot JSONL file; emits compact flags.
- `tools/vtsc/analyze_vtsc_vs_vision.py`
  - Offline scan of recent rlogs to compare VTSC target velocity vs physics-from-vision.

## 6) Tests

Co-located scenario tests near the controller:
- `sunnypilot/selfdrive/controls/lib/tests/vtsc/`
  - `sunnypilot/selfdrive/controls/lib/tests/vtsc/harness.py`
  - `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py`
  - `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_sweep_matrix.py`
  - `sunnypilot/selfdrive/controls/lib/tests/vtsc/pipeline_harness.py`
  - `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_pipeline_integration.py`
  - `sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_flow.py`

Docs-area VTSC test benches (bigger surface area):
- `docs/chauffeur/vtsc/testing/` (many pytest suites + harnesses)
- `docs/chauffeur/vtsc/fullTrace/tests/` (rlog-based regression tests)

Committed rlog fixtures (tici-sourced, copied into repo for reproducibility):
- `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--67/rlog_00000085--f247b281ca--67.zst`
- `docs/chauffeur/vtsc/cases/overslow_2025-09-05/00000085--f247b281ca--80/rlog_00000085--f247b281ca--80.zst`

Minimal “smoke” style test script:
- `test_vtsc_minimal.py`

## 7) Documentation (Existing)

Large VTSC workspace (debug sessions, reports, replays, deeper docs):
- `docs/chauffeur/vtsc/`
  - Practical overview: `docs/chauffeur/vtsc/documentation/VTSC_Overview.md`
  - Param defaults table: `docs/chauffeur/vtsc/documentation/paramBaseline.md`
  - Tests overview: `docs/chauffeur/vtsc/tests_overview.md`

This baseline index folder:
- `docs/vtsc/TESTING_STRATEGY.md` (how to test VTSC end-to-end / detect pipeline deviation)
- `docs/vtsc/OCCLUSION.md` (occlusion logic deep dive + lead-bypass expectations)
- `docs/vtsc/RLOGS.md` (rlog fixture provenance + replay guidance; tici-sourced requirement)

Offroad UI design guide (menu schema):
- `docs/chauffeur/ui/bsg/offroad/vtsc_menu_framework.md`

Related roadmap doc (mapd/MTSC/VTS C interplay):
- `docs/chauffeur/MTSC_VTSC_MAPD_ROADMAP_2025-08-31.md`

## 8) Related (Not VTSC, but adjacent)

Map Turn Speed Control (MTSC) daemon (currently disabled by process manager):
- `sunnypilot/selfdrive/controls/mtsc/mtscd.py`

MTSC docs/tests (useful context when touching map lookahead):
- `docs/chauffeur/mtsc/`

## 9) Known Stale Reference

The file `docs/chauffeur/system/documentation/README.md` mentions a `docs/claude/` tree that does not exist in this checkout.
Treat that file as informational but potentially stale.
