---
name: vtsc-rally-copilot-hud
description: Edit, debug, troubleshoot, iterate, and polish the Sunnypilot VTSC “rally co-pilot” curve strip-map HUD overlay that renders upcoming curve geometry from map data (no full map). Use when adjusting the Qt onroad rendering/layout/styling, tuning the curvature threshold + fade behavior, or debugging the data pipeline (MapCurvatures/mapd to VisionTurnController to LongitudinalPlanSP.visionTurnSpeedControl to HudRendererSP::drawVTSCCoPilotCurve). Triggers include “curve stripmap”, “rally co-pilot HUD”, VTSC curve preview points, `VTSCRallyCoPilotHUDEnabled`, `curvePreviewValid`, and `curveMaxCurvature`.
---

# VTSC Rally Co-Pilot HUD

## Guardrails

- Reuse existing MTSC/VTSC/mapd math. Do not add a separate curve computation path in the HUD.
- Render actual map-derived curve geometry via `LongitudinalPlanSP.visionTurnSpeedControl.curvePreviewTiles` (legacy `curvePreviewPoints` remains debug/compat telemetry; no generic left/right placeholder icon).
- Do not remove or break the system readiness indicator (`HudRendererSP::drawSystemReadiness`).
- Chauffeur currently runs pfeifer mapd `v1.10.0` in production. Upstream `pfeiferj/openpilot-mapd/main` is `v2.x` and incompatible. For current HUD work, treat `MapCurvatures` / `MapHazard` Params and existing chauffeur messages as the source of truth unless the task explicitly includes a v2 migration.

## Quick Loop (Edit -> Build -> Verify)

1. Edit:
- HUD rendering/layout: `selfdrive/ui/sunnypilot/qt/onroad/hud.cc` (`HudRendererSP::drawVTSCCoPilotCurve`)
- HUD state: `selfdrive/ui/sunnypilot/qt/onroad/hud.h`
- VTSC preview producer: `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
- VTSC publisher: `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
- Capnp fields: `cereal/custom.capnp`

2. Build the smallest target:
- `scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/hud.o`

3. Verify the data is present (no guessing):
- `python3 .codex/skills/vtsc-rally-copilot-hud/scripts/monitor_vtsc_copilot.py`
- Expect: `curvePreviewValid=1`, `tiles>=1`, `tile_pts>=2`, and `curveMaxCurvature` above the HUD threshold

## “Why Isn’t It Showing?” Checklist

- Confirm `VTSCRallyCoPilotHUDEnabled` is enabled.
- Confirm the producer is publishing: `curvePreviewValid` true and `curvePreviewTiles` non-empty.
- Confirm HUD gating is passing: `curveMaxCurvature` is above `KAPPA_SHOW_MIN`/`KAPPA_HOLD_MIN` in `hud.cc`.
- Confirm fade-in isn’t stuck: `vtsc_copilot_alpha_` rises above `0.01`.
- Keep speed text readable: draw the speed last (after curve strokes/glow) so it always sits on top.

## Layout Knobs (HUD)

- Placement uses an “inner” rect that excludes `UI_BORDER_SIZE`. Adjust placement in `HudRendererSP::drawVTSCCoPilotCurve`.
- Move the overlay up/down by changing `bottom_safe` (smaller moves it down, larger moves it up).
- Change overall sizing/thickness/fonts by changing `kScale`.
- Keep speed on top by drawing it last.

## References

- Read `references/pipeline.md` when debugging “is the map preview data wrong, or is the HUD rendering wrong?” and when changing any capnp fields.

## Skill Maintenance

After any real VTSC HUD debug or polish session:
- compare new evidence against existing guidance before finishing
- correct stale bullets instead of appending contradictory notes
- keep exact file paths, thresholds, and verification commands current
- prefer one clear proven failure mode and one reliable validation command over speculative lists
- add new gotchas only for real, repeatable footguns
- keep only durable workflow guidance in `SKILL.md`; move detailed examples or reference material to `references/` or `scripts/`
- update `agents/openai.yaml` only if the skill’s scope or trigger wording changed materially

Treat this skill as a recursive kaizen loop:
- every real invocation should leave it more accurate, more actionable, more compact, or all three
- recursively self-improve by reconciling new facts with old guidance in the same pass so the next invocation starts smarter
- if new evidence proves an older bullet wrong, incomplete, or redundant, replace, tighten, or delete it instead of stacking another warning
- prefer editing and pruning over adding line after line; a shorter, sharper skill beats a longer noisier one
