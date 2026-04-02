# Onroad HUD Reference

## Table of Contents

- Runtime selection and aliasing
- Active paint stack
- File ownership map
- State and message ownership
- Symptom-to-file guide
- Verification
- Footguns

## Runtime Selection and Aliasing

- The default chauffeur build defines `-DSUNNYPILOT` in `SConstruct`. Use `scons --stock-ui` only when the task explicitly requires stock-openpilot runtime behavior.
- `selfdrive/ui/SConscript` builds both the base `qt/onroad/*.cc` files and the extra `sunnypilot/qt/onroad/*.cc` files into the default UI binary.
- `selfdrive/ui/qt/home.h` remaps `OnroadWindow` to `OnroadWindowSP` when `SUNNYPILOT` is enabled.
- `selfdrive/ui/qt/onroad/onroad_home.h` remaps `AnnotatedCameraWidget` to `AnnotatedCameraWidgetSP`.
- `selfdrive/ui/qt/onroad/annotated_camera.h` remaps `ExperimentalButton`, `ModelRenderer`, and `HudRenderer` to SP variants.
- Because of those header macros, a file under `selfdrive/ui/qt/onroad/` can still instantiate SP classes in the default build. Verify the active path before assuming a file is stock-only.

## Active Paint Stack

1. `HomeWindow` resolves to `HomeWindowSP`, which owns the onroad window stack.
2. `OnroadWindow::OnroadWindow` creates `AnnotatedCameraWidget`, which resolves to `AnnotatedCameraWidgetSP` in the default build.
3. `AnnotatedCameraWidget::paintGL` draws the camera frame, then the model renderer, then the HUD renderer.
4. `AnnotatedCameraWidgetSP::paintGL` calls the base implementation and then paints its own `hud_sp` pass on top.
5. `HudRendererSP::draw` paints in this order:
- VTSC rally co-pilot curve
- `HudRenderer::draw()` base HUD widgets
- readiness tree
- RTI cards
- `OnroadAlerts` is a separate stacked widget above the camera surface. Changing HUD draw order alone does not move content above alerts.

## File Ownership Map

- Window and stacked layout:
- `selfdrive/ui/qt/home.h`
- `selfdrive/ui/sunnypilot/qt/home.h`
- `selfdrive/ui/qt/onroad/onroad_home.h`
- `selfdrive/ui/qt/onroad/onroad_home.cc`
- `selfdrive/ui/sunnypilot/qt/onroad/onroad_home.h`
- `selfdrive/ui/sunnypilot/qt/onroad/onroad_home.cc`

- Camera view and overlay orchestration:
- `selfdrive/ui/qt/onroad/annotated_camera.h`
- `selfdrive/ui/qt/onroad/annotated_camera.cc`
- `selfdrive/ui/sunnypilot/qt/onroad/annotated_camera.h`
- `selfdrive/ui/sunnypilot/qt/onroad/annotated_camera.cc`

- Base HUD widgets and text:
- `selfdrive/ui/qt/onroad/hud.h`
- `selfdrive/ui/qt/onroad/hud.cc`
- Key helpers: `drawSetSpeed`, `drawSpeedLimitSigns`, `drawUpcomingSpeedLimit`, `drawRoadName`, `drawVisionTurnControl`, `drawCurrentSpeed`, `drawSLCSourceBadge`

- SP HUD overlays:
- `selfdrive/ui/sunnypilot/qt/onroad/hud.h`
- `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`
- Key surfaces: readiness tree, RTI multi-threat cards, VTSC rally co-pilot curve

- Path, lane, lead, and blindspot visuals:
- `selfdrive/ui/qt/onroad/model.h`
- `selfdrive/ui/qt/onroad/model.cc`
- `selfdrive/ui/sunnypilot/qt/onroad/model.h`
- `selfdrive/ui/sunnypilot/qt/onroad/model.cc`
- SP additions: rainbow path and blindspot polygons

- Onroad buttons:
- `selfdrive/ui/qt/onroad/buttons.h`
- `selfdrive/ui/qt/onroad/buttons.cc`
- `selfdrive/ui/sunnypilot/qt/onroad/buttons.h`
- `selfdrive/ui/sunnypilot/qt/onroad/buttons.cc`
- SP addition: DEC split experimental button

- Alerts above the camera surface:
- `selfdrive/ui/qt/onroad/alerts.h`
- `selfdrive/ui/qt/onroad/alerts.cc`

- UI subscriptions and scene state:
- `selfdrive/ui/ui.h`
- `selfdrive/ui/ui.cc`
- `selfdrive/ui/sunnypilot/ui.h`
- `selfdrive/ui/sunnypilot/ui.cc`

## State and Message Ownership

- `UIState` owns the base scene, socket updates, and common subscriptions.
- `UIStateSP` extends the default `SubMaster` set with SP-specific streams, including:
- `modelManagerSP`
- `selfdriveStateSP`
- `longitudinalPlanSP`
- `backupManagerSP`
- `liveMapDataSP`
- `rtiStateSP`

- `HudRenderer::updateState` reads:
- `controlsState`
- `carState`
- `longitudinalPlanSP`
- `liveMapDataSP`

- `HudRendererSP::updateState` first calls the base HUD update, then reads:
- `selfdriveStateSP`
- `rtiStateSP`
- `longitudinalPlanSP`

- `ModelRenderer` depends on the scene transform plus:
- `modelV2`
- `radarState`
- `carState`
- `selfdriveState`

- `ExperimentalButtonSP::updateState` reads `longitudinalPlanSP.getDec()`.

- If the screen is wrong because the underlying message is wrong, fix the publisher path first. Do not mirror planner logic in the HUD.

## Symptom-to-File Guide

- Speed number, set-speed card, speed-limit signs, upcoming speed limit, or road-name banner look wrong:
- Start in `selfdrive/ui/qt/onroad/hud.cc`.

- RTI card order, RTI card layout, readiness tree spacing, readiness opacity, or VTSC rally strip-map look wrong:
- Start in `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`.

- Path fill, lane lines, lead chevrons, or blindspot shading look wrong:
- Start in `selfdrive/ui/qt/onroad/model.cc` or `selfdrive/ui/sunnypilot/qt/onroad/model.cc`.

- Experimental button or DEC split icon looks wrong:
- Start in `selfdrive/ui/qt/onroad/buttons.cc` or `selfdrive/ui/sunnypilot/qt/onroad/buttons.cc`.

- Overlay is hidden behind something, clipped strangely, or painted in the wrong order:
- Inspect `annotated_camera.cc`, `onroad_home.cc`, and the active `draw()` ordering before editing the individual widget.

- Onroad UI shows stale or missing data even though paint code looks reasonable:
- Inspect `selfdrive/ui/ui.cc`, `selfdrive/ui/sunnypilot/ui.cc`, and the source publisher before repainting.

## Verification

- Build the narrowest touched object first:
- `scons -j$(nproc) selfdrive/ui/qt/onroad/hud.o`
- `scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/hud.o`
- `scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/model.o`
- `scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/buttons.o`
- `scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/annotated_camera.o`

- If you change readiness semantics or SP state publishing, run the narrow test too:
- `pytest selfdrive/selfdrived/tests/test_subsystem_readiness.py`

- Use a broader UI build only after the narrow object build passes:
- `scons -j$(nproc) selfdrive/ui/ui`

## Footguns

- The default build uses the Sunnypilot alias path. Do not trust class names alone; read the aliasing headers.
- `AnnotatedCameraWidgetSP` adds an extra HUD paint pass after the base implementation. If a HUD change looks doubled, missing, or inconsistent, inspect both the aliased base member and `hud_sp`.
- `OnroadAlerts` lives in a stacked widget above the camera surface. Reordering HUD paint code does not place content above alerts.
- `selfdrive/ui/claude.md` and `selfdrive/ui/qt/onroad/claude.md` are placeholder text, not authoritative file maps.
- `selfdrive/ui/sunnypilot/qt/onroad/tests/test_threat_sorting.py` is only a narrow RTI sort smoke check. It does not validate full RTI HUD rendering or card stacking.
- Offroad settings panels under `selfdrive/ui/sunnypilot/qt/offroad/settings/` are not part of this onroad HUD surface even when they configure onroad behavior.
