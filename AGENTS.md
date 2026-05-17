# Chauffeur Agent Instructions

## Non-obvious requirements (must follow)

- Project scope is the 2023 CAN-FD HDA2 Kia EV6; treat vehicle behavior as Hyundai code/logic unless the user explicitly expands scope.
- For object-hazard work, read `.codex/skills/object-hazard-live-monitor/SKILL.md` before changing code or judging live behavior.
- Keep object-hazard reviews on the actual pipeline: `sunnypilot/objectd/`, `sunnypilot/selfdrive/controls/lib/object_hazard_controller.py`, `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`, `selfdrive/controls/lib/longitudinal_planner.py`, `selfdrive/controls/plannerd.py`, `system/manager/process_config.py`, `cereal/custom.capnp`, and `cereal/services.py`.

## Landmines / gotchas (things that fail silently)

- `objectd` only starts when onroad, `CP.notCar` is false, and `ObjectHazardEnabled` is true; the default for `ObjectHazardEnabled` is `"1"` in `common/params_keys.h`.
- `objectd` defaults to `OBJECTD_BACKEND=snpe_gpu` and expects model assets at `.cache/objectd/yolo11n/model.dlc` plus `.cache/objectd/yolo11n/metadata.json` unless `OBJECTD_MODEL_*` env vars override the paths.
- A healthy detector publish is not enough: verify planner wiring from `objectHazardStateSP` through `LongitudinalPlannerSP.object_hazard`, `longitudinalPlanSP.objectHazardControl`, and main `longitudinalPlan.shouldStop`.
- Windows-only pytest failures from missing native/generated modules are not device evidence; report the exact missing import separately from code-health findings.

## Verification / definition of done

- For instruction-file edits: run `python .agents/skills/context-file-librarian/scripts/audit_context_files.py` and show `git diff -- AGENTS.md CLAUDE.md .claude/CLAUDE.md`.
- For object-hazard code review without device access: run `python -m compileall -q sunnypilot\objectd sunnypilot\selfdrive\controls\lib\object_hazard_controller.py`.
- Also run the focused tests when the local environment supports them: `python -m pytest sunnypilot/objectd/tests/test_process_registration.py sunnypilot/objectd/tests/test_path_association.py sunnypilot/selfdrive/controls/lib/tests/test_object_hazard_controller.py sunnypilot/selfdrive/controls/lib/tests/test_object_hazard_pipeline.py`.

## Updating this file (drift policy)

- Start small.
- Add a bullet only after observing a real agent/user failure that was not obvious from code/config.
- Remove a bullet once the underlying issue is fixed or becomes obvious in code/config.
- Keep commands and paths verifiable in this repo; move uncertain deployment facts to "Needs human confirmation".

## Needs human confirmation (temporary; keep very short)

- Before the next tici test, confirm the object-hazard model assets exist on device or that `OBJECTD_MODEL_PATH` and `OBJECTD_MODEL_METADATA` point to valid files; this repo does not track those assets, so re-check the external source or use a saved Qualcomm AI Hub export artifact.
