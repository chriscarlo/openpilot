# VTSC Rally Co-Pilot Tile HUD Spec

## Purpose

Replace the current continuously-updating strip-map ribbon with a discrete queue of upcoming turn tiles. Each tile renders actual map-derived curve geometry, but the presentation is static per tile so the HUD reads as a stable turn list instead of a dancing preview.

## Environment assumptions

- Repo root: `/projects/chauffeur/data/openpilot`
- Producer path: `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
- Publisher path: `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
- HUD path: `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`
- Schema path: `cereal/custom.capnp`

## Design contract

- Adjectives: glanceable, stable, tactile
- Theme: dark translucent HUD surface with warm active-route emphasis
- Brand motif: stacked cards with a single warm-highlighted active tile and subdued queued tiles
- Curve rendering priority: pretty first, faithful enough second; preserve the turn's general shape, direction, and severity, but smooth aggressively to avoid noisy geometry

## Information hierarchy

- Primary goal: let the driver understand the sequence and shape of upcoming turns in one glance
- Primary focal region: the bottom-most active tile
- Secondary region: queued tiles above, compressed and de-emphasized
- Empty state: no widget when there are no upcoming tiles
- Loading/error state: same as empty; use existing alert systems for failures, not the turn widget

## Data contract

- Keep using VTSC/mapd as the only curve computation path.
- Publish a per-turn tile list on `LongitudinalPlanSP.VisionTurnSpeedControl`.
- Each tile carries:
  - distance/time to tile start
  - peak curvature
  - direction/severity
  - curve geometry points normalized into a tile-local frame
- Tile-local frame rules:
  - the tile entry point starts at the bottom center of the tile
  - the tile entry tangent is treated as "forward/up"
  - geometry is not cardinally oriented
  - once published, the tile geometry remains static until VTSC republishes a new tile list

## Tile behavior

- Stack order:
  - nearest tile sits at the bottom
  - farther tiles stack upward
- Horizon:
  - include tiles up to roughly `20 s` ahead, bounded by available map geometry
- Visual treatment:
  - bottom tile is largest and highest contrast
  - upper tiles shrink slightly and reduce opacity
  - active tile may include advisory speed text; queued tiles may omit it
  - the road glyph is solid white, with the first and last portions fading to transparent
  - the road glyph carries a soft outer glow or drop shadow for separation from the background
- Geometry treatment:
  - use actual map-derived geometry as input
  - apply aggressive smoothing/resampling before publishing tile-local points
  - preserve direction and broad severity even if the exact radius is stylized
- Animation:
  - while the active turn is in progress, keep the active tile stable
  - once the active tile is considered complete/apex passed, animate it falling off the bottom of the widget
  - simultaneously shift queued tiles down into their new resting positions
  - animation target: `180 ms` shift, `260 ms` drop-away

## Implementation plan

1. Add a tile list type to `cereal/custom.capnp`.
2. Extend VTSC map preview generation to segment multiple curve regions from the existing map curvature input.
3. Normalize each tile's geometry around its own entry heading before publishing.
4. Publish the tile list through `longitudinal_planner.py` while preserving existing single-preview fields for compatibility/debugging.
5. Replace the ribbon renderer in `HudRendererSP::drawVTSCCoPilotCurve` with stacked tile rendering and discrete transition state.

## Verification

- Targeted producer tests:
  - `pytest sunnypilot/selfdrive/controls/lib/tests/vtsc/test_map_curve_preview.py`
- Focused planner publish tests:
  - `pytest sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_flow.py`
- Fast HUD build:
  - `scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/hud.o`

## Expected outputs

- Producer tests confirm multiple upcoming curves are segmented into ordered tiles with stable local orientation.
- Planner tests confirm tile fields are published onto `longitudinalPlanSP`.
- HUD build succeeds with the new tile renderer.
