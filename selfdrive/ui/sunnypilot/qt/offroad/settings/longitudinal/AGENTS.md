# Longitudinal Offroad Settings (Qt) — Agent Instructions

Avoid vague AI-isms such as "clean" or "concrete" when describing work; describe the specific property instead.

## Non-obvious requirements (must follow)
- Prefer Sunnypilot `*SP` widgets from `selfdrive/ui/sunnypilot/qt/widgets/controls.h` (this directory already uses `ListWidgetSP`, `ScrollViewSP`, `OptionControlSP`, etc.).
- Do not edit or commit generated Qt/build artifacts in this directory (`moc_*.cc`, `*.o`); edit the corresponding `*.cc`/`*.h` sources.

## Landmines / gotchas (things that fail silently)
- UI layout changes can overflow on the target device even if they look fine on desktop; avoid introducing horizontal scrollbars and verify on the target UI when possible.

## Verification / definition of done
- Run a minimal build for UI changes: `scons -j$(nproc)` (or the smallest UI target if known).
- Confirm `git diff` does not include `moc_*.cc` / `*.o` churn.

## Updating this file (drift policy)
- Keep only correctness pitfalls here; move style specifics into `docs/chauffeur/ui/bsg/offroad/offroad_settings_bsg.md`.

## Needs human confirmation (temporary; keep very short)
- Any hard pixel-width constraints intended for these panels (if they exist, point to the code constant(s) that enforce them).
