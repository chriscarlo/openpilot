# VTSC Rally Co-Pilot HUD: Data + Render Pipeline Notes

## High-Level Flow

1. `mapd` computes upcoming curvature geometry from OSM and stores it in the `MapCurvatures` Param (JSON).
2. VTSC reads that map curvature input and builds a *preview polyline* in ego-frame.
3. The longitudinal planner publishes the preview fields on the `longitudinalPlanSP` service.
4. The Sunnypilot onroad HUD renders the preview polyline (and only renders it; no new curve computation in HUD).

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
- `curveDistanceM` (float32): distance to curve-start marker (meters, forward)
- `curveTimeToS` (float32): estimated time until curve-start marker
- `curveMaxCurvature` (float32): used for “more than slight” gating (HUD-side)
- `curveDirection` (enum): optional metadata
- `curveSeverity` (enum): optional metadata
- `curvePreviewPoints` (list of points): ego-frame polyline points
  - Each point: `xFwdM` and `yLeftM`

## Quick Debug Commands

- Monitor live message fields:
  - `python3 .codex/skills/vtsc-rally-copilot-hud/scripts/monitor_vtsc_copilot.py`
- Confirm the param toggle is on:
  - `python3 -c 'from openpilot.common.params import Params; print(Params().get_bool(\"VTSCRallyCoPilotHUDEnabled\"))'`

## Common Failure Modes

- `curvePreviewValid=0`: VTSC isn’t producing preview geometry (often `MapCurvatures` missing/stale).
- `pts<3`: preview is “valid” but empty/undersampled; fix the producer, not the HUD.
- HUD gating hides it: check `KAPPA_SHOW_MIN`/`KAPPA_HOLD_MIN` in `hud.cc`.
- Geometry is fine but “looks wrong”: verify coordinate convention:
  - `xFwdM` should be forward meters (>=0)
  - `yLeftM` should be positive-left, negative-right

