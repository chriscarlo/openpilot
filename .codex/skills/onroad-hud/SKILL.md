---
name: onroad-hud
description: "Maintain and debug the onroad HUD shown on the tici while driving across stock openpilot Qt and Sunnypilot/Chauffeur overlays. Use when changing the on-screen speed or set-speed widgets, SLC/VTSC indicators, road-name or speed-limit banners, Experimental button behavior, path/lane/lead/blindspot overlays, RTI cards, readiness tree, or the onroad paint/layout chain under `selfdrive/ui/qt/onroad/`, `selfdrive/ui/sunnypilot/qt/onroad/`, `selfdrive/ui/ui.cc`, and `selfdrive/ui/sunnypilot/ui.cc`."
---

# Onroad HUD Maintainer

## Start Here

- Read [onroad-hud-reference.md](./references/onroad-hud-reference.md) before editing when you need the active runtime path, macro aliasing, file ownership, or message map.
- Treat the onroad display as four layers: container/window, camera/model renderer, base HUD widgets, and Sunnypilot overlays. Classify the symptom before choosing a file.
- In the default chauffeur build, `SConstruct` adds `-DSUNNYPILOT`. Read the aliasing headers first: `selfdrive/ui/qt/home.h`, `selfdrive/ui/qt/onroad/onroad_home.h`, and `selfdrive/ui/qt/onroad/annotated_camera.h`.

## Triage

1. Classify the surface.
- Speed, set speed, SLC signs, upcoming speed limit, road name, or the base VTSC widget: edit `selfdrive/ui/qt/onroad/hud.cc`.
- RTI cards, readiness tree, or the VTSC rally co-pilot strip-map: edit `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`.
- Lane/path/lead drawing or blindspot shading: edit `selfdrive/ui/qt/onroad/model.cc` or `selfdrive/ui/sunnypilot/qt/onroad/model.cc`.
- Experimental button behavior or DEC split iconography: edit `selfdrive/ui/qt/onroad/buttons.cc` or `selfdrive/ui/sunnypilot/qt/onroad/buttons.cc`.
- Paint order, overlay stacking, or camera/view container behavior: edit `selfdrive/ui/qt/onroad/annotated_camera.cc`, `selfdrive/ui/sunnypilot/qt/onroad/annotated_camera.cc`, or the corresponding `onroad_home.cc` file.
- Missing or stale data on screen: verify `selfdrive/ui/ui.cc` or `selfdrive/ui/sunnypilot/ui.cc` before touching paint code.

2. Verify the data owner.
- Base HUD state comes from `controlsState`, `carState`, `longitudinalPlanSP`, and `liveMapDataSP`.
- SP HUD adds `selfdriveStateSP` and `rtiStateSP`.
- Do not recompute planner or backend decisions in the HUD when a published field already exists.

3. Make the smallest layer-local change.
- Prefer editing the existing draw helper instead of adding more logic to top-level `draw()`.
- Preserve draw order unless the task is explicitly about stacking or occlusion.
- Keep offroad settings work out of this skill. If the change belongs under `selfdrive/ui/sunnypilot/qt/offroad/settings/`, switch to the relevant settings surface.
- For Sunnypilot Qt controls, prefer `*SP` widgets from `selfdrive/ui/sunnypilot/qt/widgets/controls.h`.

## Verify

- Build the narrowest touched object first, for example:
- `scons -j$(nproc) selfdrive/ui/qt/onroad/hud.o`
- `scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/hud.o`
- `scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/model.o`
- `scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/buttons.o`
- If the change affects published readiness or state semantics, run the matching test, such as `pytest selfdrive/selfdrived/tests/test_subsystem_readiness.py`.
- Inspect `git diff` and keep the change set limited to intended UI or publisher files.

## Pair With Narrow Skills

- Use `readiness-tree` when the task is specifically about subsystem colors, labels, or `selfdriveStateSP.subsystemStatuses`.
- Use `rti-tuner` when the task is specifically about RTI threat publishing, road matching, or slowdown behavior.
- Use `vtsc-rally-copilot-hud` when the task is specifically about the curve strip-map overlay and its capnp/path pipeline.

## Resources

- Read [onroad-hud-reference.md](./references/onroad-hud-reference.md) for the active runtime path, paint stack, file ownership map, state/message ownership, symptom-to-file guide, and HUD-specific footguns.

## Kaizen Loop

- Correct or replace stale guidance before adding new notes.
- Prefer rewriting, tightening, or reorganizing sections over append-only accumulation.
- Add new material only when no shorter correction, replacement example, or stronger file pointer captures the lesson.
- Revisit `agents/openai.yaml`, trigger wording, and the split between `SKILL.md` and `references/` when the HUD surface map changes materially.
- Rerun `python3 /home/chris/.codex/skills/.system/skill-creator/scripts/quick_validate.py /projects/chauffeur/data/openpilot/.codex/skills/onroad-hud` after skill edits.
