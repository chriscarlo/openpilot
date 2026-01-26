# VTSC (Vision Turn Speed Control) — Repo Index

This folder is a **baseline reference index** for the **Vision Turn Speed Control (VTSC)** feature.

If you are trying to answer any of these quickly:
- “Where is VTSC implemented?”
- “Where are the tests?”
- “Which Params keys control VTSC?”
- “How do I debug VTSC on-device?”

Start here, then jump into the referenced code/docs.

## Where VTSC Actually Lives (Code)

The VTSC feature is primarily implemented as a Python controller in:
- `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` (core logic)
- `sunnypilot/selfdrive/controls/lib/vision_turn_params.py` (Params refresh/decoding)

It is integrated into the longitudinal stack via:
- `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py` (calls `VisionTurnController.update()` and publishes `longitudinalPlanSP.visionTurnSpeedControl`)
- `selfdrive/controls/lib/longitudinal_planner.py` (main planner inherits `LongitudinalPlannerSP`)

Messaging schema:
- `cereal/custom.capnp` (`LongitudinalPlanSP.VisionTurnSpeedControl`)

## Where To Look Next

- For a full file-by-file map: `docs/vtsc/INVENTORY.md`
- For a walkthrough of occlusion logic + lead bypass: `docs/vtsc/OCCLUSION.md`
- For tests and harnesses: `docs/vtsc/TESTS.md`
- For how we emulate the live pipeline: `docs/vtsc/TESTING_STRATEGY.md`
- For rlog fixture provenance + replay guidance: `docs/vtsc/RLOGS.md`
- For Params keys and defaults: `docs/vtsc/PARAMS.md`
- For debugging + analysis tooling: `docs/vtsc/TOOLING.md`
- For terminology: `docs/vtsc/GLOSSARY.md`

## Existing “Deep Dive” VTSC Workspace

This repo already contains an extensive VTSC debug + analysis workspace:
- `docs/chauffeur/vtsc/`

That directory includes:
- deeper design/analysis docs,
- replay artifacts and offroad reports,
- larger test benches and traces,
- its own `AGENTS.md` describing conventions.

This `docs/vtsc/` folder is intentionally a small “index + pointers” layer above it.
