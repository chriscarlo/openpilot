# VTSC Rally Co-Pilot HUD: Data + Render Pipeline Notes

## High-Level Flow

1. `mapd` computes upcoming curvature geometry from OSM and stores it in the `MapCurvatures` Param (JSON).
2. VTSC reads that map curvature input and builds both:
   - a legacy preview polyline in ego-frame (`curvePreviewPoints`) for compatibility/debugging
   - a per-turn tile list (`curvePreviewTiles`) with geometry normalized into each tile's entry-up frame for the HUD
3. The longitudinal planner publishes the preview fields on the `longitudinalPlanSP` service.
4. The Sunnypilot onroad HUD renders the published tile list (and only renders it; no new curve computation in HUD).

Keep the HUD “dumb”: if you need different geometry, fix VTSC/mapd, then publish different `curvePreviewPoints`.

## Concrete Producers/Consumers (Repo Paths)

- Capnp schema:
  - `cereal/custom.capnp`
  - Struct: `LongitudinalPlanSP.VisionTurnSpeedControl`
- VTSC map enrichment / preview producer:
  - `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
- Planner publisher:
  - `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
- HUD consumer:
  - `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`
  - Function: `HudRendererSP::drawVTSCCoPilotCurve`

## Capnp Fields Used By The HUD

Read-only from the HUD’s perspective:

- `curvePreviewValid` (bool)
- `curvePreviewTiles` (list): ordered nearest-first
  - `tileId`
  - `distanceM`
  - `timeToS`
  - `direction`
  - `severity`
  - `maxCurvature`
  - `advisorySpeedMps`
  - `points` (tile-local path points; `xFwdM`, `yLeftM`)
- Legacy/debug fields still available:
  - `curveDistanceM`
  - `curveTimeToS`
  - `curveMaxCurvature`
  - `curveDirection`
  - `curveSeverity`
  - `curvePreviewPoints`

## Quick Debug Commands

- Monitor live message fields:
  - `python3 .codex/skills/vtsc-rally-copilot-hud/scripts/monitor_vtsc_copilot.py`
- Confirm the param toggle is on:
  - `python3 -c 'from openpilot.common.params import Params; print(Params().get_bool(\"VTSCRallyCoPilotHUDEnabled\"))'`

## Common Failure Modes

- `curvePreviewValid=0`: VTSC isn’t producing preview geometry (often `MapCurvatures` missing/stale).
- `tiles=0`: legacy road polyline may still exist, but the HUD intentionally hides when there are no upcoming turn tiles.
- `tile_pts<2`: tile exists but is undersampled/invalid; fix the producer, not the HUD.
- HUD gating hides it: check `KAPPA_SHOW_MIN`/`KAPPA_HOLD_MIN` in `hud.cc`.
- Geometry is fine but “looks wrong”: verify coordinate convention:
  - `xFwdM` should be forward meters (>=0)
  - `yLeftM` should be positive-left, negative-right
