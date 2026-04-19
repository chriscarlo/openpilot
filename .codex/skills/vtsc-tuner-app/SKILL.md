---
name: vtsc-tuner-app
description: Edit, debug, troubleshoot, iterate, and polish the standalone Rust/egui VTSC sigmoid tuner at `tools/vtsc_tuner/`. Use when changing the hero-plot interactions, the plain-English knobs, EQ bands, apply/commit/push/pull-to-tici pipeline, undo/redo, theming/scale, or when debugging WSLg-specific window crashes and detached-launch issues. Triggers include "vtsc tuner app", "vtsc sigmoid tuner", "knob widget", "apply chain", "Revert to baseline", "Pull on tici", and anything under `tools/vtsc_tuner/`.
---

# VTSC Sigmoid Tuner (desktop app)

A standalone Rust + egui GUI for shaping the VTSC sigmoid (`_physics_based_lateral_acceleration` in `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`) and its optional Q-curve multiplier. Not to be confused with the `vtsc-tuner` skill (that one analyses on-device VTSC interventions).

## Guardrails

- **Don't** confuse this tool with `tools/vtsc_tuner/` vs the deleted `tools/vtsc/curve_tuner.py`. The Python tkinter version was removed on 2026-04-18 — do not resurrect it.
- **Don't** introduce new features without keeping the math in sync with `vision_turn_controller.py` lines 818–823 and `vision_turn_params.py` lines 335–351 (clip ranges).
- **Don't** bypass the existing `plot::hero_plot` monotonic-v post-processing (see "Landmines"). The parametric `(v(κ), a(κ))` trace can fold back on itself; the fix is non-negotiable UX.
- **Don't** remove `event_loop_builder` that forces the X11 winit backend in `main.rs` — WSLg's wayland path crashes on corner-resize.
- **Don't** use the Claude Code Bash tool's `run_in_background: true` to launch the binary from this shell session — it severs the display connection. Launch with `nohup <bin> >log 2>&1 </dev/null & disown` and verify with `pgrep`.

## Location + stack

- Crate: `tools/vtsc_tuner/` (standalone, NOT part of the openpilot SCons build).
- Toolchain: stable Rust (1.89+). egui/eframe 0.33 + glow renderer + winit 0.30 (X11-only on Unix).
- Binary: `tools/vtsc_tuner/target/release/vtsc_tuner` (~10 MB).

## Quick loop (edit → build → run)

```bash
# 1. Edit any src file, then:
cd tools/vtsc_tuner && cargo build --release

# 2. Run (WSL/Linux).  NEVER use Claude Code's run_in_background.
nohup ./target/release/vtsc_tuner >/tmp/vtsc_tuner.log 2>&1 </dev/null &
disown
pgrep -af target/release/vtsc_tuner | grep -v bash   # confirm alive

# 3. Quick unit tests (sigmoid + params math, no GUI):
cargo test --bin vtsc_tuner
```

Release rebuilds are ~7–9 s incrementally; full builds ~30 s.

## Module map (src/)

| File | Responsibility |
|---|---|
| `main.rs` | `eframe::run_native` entry, diag-log-on-panic, winit X11 forcing |
| `app.rs` | `TunerApp` state, header/footer/side-panel layout, undo/redo, apply modal |
| `plot.rs` | Custom-painted hero graph: drag targets, baseline overlay, Q zones, monotonic-v post-process |
| `knob.rs` | Rotary knob widget; vertical-drag, scroll, shift=fine, dbl-click-reset; optional editable text |
| `params.rs` | `PlainKnobs` (tight/straight accel, transition mph, sharpness) ↔ `SigmoidParams` (A,B,C,D,MIN,MAX) |
| `sigmoid.rs` | Raw math: `SigmoidParams::eval`, `sample_curve`, `Band` + `apply_prepared_bands`, `bands_as_q_curve_points` |
| `apply.rs` | Background-thread apply chain: tune JSON → source patch → git → ssh tici |
| `io.rs` | Tune JSON (schema v1) save/load, `DeviceProfile` ssh target enum, `plan_push` |
| `theme.rs` | Dark palette, `pixels_per_point` scale, egui `Visuals` install |

All public symbols are minimal — internal helpers are private.

## The math (see `references/math.md` for derivations)

- Runtime model in `vision_turn_controller.py:890-912`:
  `a(κ) = A/(1 + exp(B·(κ−C))) + D`, clamped to `[MIN, MAX]`. Then `v = sqrt(a/κ)`.
- UI plots the parametric `(v(κ), a(κ))` with κ swept log-uniformly.
- **Plain knobs** map to raw params via `PlainKnobs::to_sigmoid` / `from_sigmoid` so the low-κ asymptote equals the "straight-road ceiling" and the high-κ asymptote equals the "tight-curve ceiling" (what-you-see-is-what-you-get collapse).
- **EQ bands** compose multiplicatively as Gaussian bumps in log-κ space with σ = 0.5/Q and gain in dB. `prepare_bands()` freezes each band's κ_centre against the base sigmoid so bands don't "chase their own tail" during sampling.
- **Monotonic-v** post-process in `plot.rs` holds `v` at the running max when the parametric trace would step backwards. This turns aggressive bands into vertical notches (parametric-EQ idiom) instead of visually-wrong folded curves. **Do not remove this.**

## Apply chain contract (`apply.rs`)

Four actions, each is a superset of the previous:

| Action | Writes tune JSON | Patches source | git commit | git push | ssh tici pull+reboot |
|---|---|---|---|---|---|
| Local | ✓ | ✓ | – | – | – |
| Commit | ✓ | ✓ | ✓ | – | – |
| Push | ✓ | ✓ | ✓ | ✓ | – |
| PullOnTici | ✓ | ✓ | ✓ | ✓ | ✓ |

- Tune JSON default path: `~/.config/vtsc_tuner/current.tune.json` (via `dirs::config_dir`).
- Source patch targets:
  - `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` — rewrites lines starting `PHYSICS_A = …` … `PHYSICS_MAX_LAT_ACCEL = …`.
  - `sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py` — rewrites `Q_CURVE_ENABLED` and replaces the entire `Q_CURVE_POINTS: list[tuple[float, float]] = [...]` block (possibly multi-line).
- Commit message: `vtsc: tune sigmoid via vtsc_tuner`.
- Worker runs on a std thread, events flow through an `mpsc::channel` collapsed by step id in `Task::poll`.
- Each step emits start (⏳ Running) then end (✓ Ok / ✗ Err) with plain-English text + raw command detail.

### SSH profile discovery (tici)

`apply::detect_ssid()` tries `iwgetid -r` then PowerShell `(Get-NetConnectionProfile | …).Name` (WSL fallback). If the SSID contains "comma", profile order is `commaCar → commaHome → commaAdb`; otherwise `commaHome → commaCar → commaAdb`. Each is probed with `ssh -o ConnectTimeout=2 -o BatchMode=yes <profile> true`. First responder wins.

Profiles must exist in `~/.ssh/config`:
- `commaHome` — home Wi-Fi (192.168.1.172)
- `commaCar` — car hotspot (192.168.0.229)
- `commaAdb` — USB via adb port-forward (127.0.0.1:2222)

Remote command: `cd /data/openpilot && git pull`, then a fire-and-forget `sudo reboot`. Note that the device tracks `chauffeur-dev4`; if the dev branch differs, the tici's `git pull` won't pick up the just-pushed change. Leave this as a user-facing caveat — don't try to auto-rebase / checkout on the device.

## Landmines (things that will bite you)

- **WSLg wayland crashes on corner-resize.** Winit's calloop event loop closes with `Io error: Broken pipe` three times then `WinitEventLoop(ExitFailure(1))`. Fixed by forcing X11 via `EventLoopBuilderExtX11::with_x11()` in `main.rs`. Don't remove.
- **Detached launch from Claude Code's Bash tool (`run_in_background: true` or certain `pkill`-prefaced scripts) breaks the display connection.** Foreground works, `nohup … & disown` after a non-destructive prelude works, `run_in_background` doesn't. Exit code 144 is the telltale.
- **Parametric (v, a) plot can fold back on itself** when a band's local slope exceeds `a/κ`. Monotonic-v post-process in `plot.rs` is the accepted workaround.
- **Side-panel width + pixels_per_point interaction.** `ctx.set_pixels_per_point(1.5)` in `theme::install`. Long knob labels were clipping; `knob.rs` now measures label width via `ctx.fonts_mut(|f| f.glyph_width(...))` and grows the knob footprint to fit. If you re-introduce single-line-label allocation, test with "Tight-Curve Ceiling" / "Straight-Road Ceiling".
- **Header subtitle gets truncated** on narrow windows. Currently shortened to "curvature → lat-accel". Keep it short or remove entirely.
- **`with_inner_size` is LOGICAL points** (post pixels_per_point). Defaults: 1300×850 logical = ~2600×1700 device-equivalent at ppp=1.5. Min 900×620 logical.
- **`bands_as_q_curve_points` returns points in log-uniform κ** (1e-5..1.0). Export converts band lat-accel gain to speed gain via `sqrt(ratio)` because `Q_CURVE_POINTS` is a speed multiplier in the Python runtime.
- **Undo/redo uses frame-debounced snapshots.** `maybe_capture()` pushes the pre-edit state only on the first frame a change is observed after an idle period — continuous drags become one undo step. `commit_snapshot()` forces an immediate push for discrete actions like "Revert to baseline".
- **Bands iterated by index** in panels so mutation during iteration works. Don't refactor to iter_mut without handling the selected-band resync on removal.
- **Widget hit-priority** in `plot::pick_drag_target`: inflection → steepness wings → band dots → rails. Keep this order or smaller/newer targets get shadowed.

## Verification / definition of done

Smallest check for any change:
```bash
cd tools/vtsc_tuner
cargo test --bin vtsc_tuner          # ~9 unit tests (sigmoid + params)
cargo build --release                # ~7 s incremental
# smoke-launch:
timeout 4 ./target/release/vtsc_tuner; echo "exit=$?"   # 143 = SIGTERM = alive
```

Before claiming "works":
1. Knob labels fit without clipping (try "Tight-Curve Ceiling", "Straight-Road Ceiling").
2. Resize the window by corner-drag. Must not crash.
3. Drag each on-graph handle: min rail, max rail, inflection, steepness wings, any band dot. All respond cleanly.
4. Shift-click on the curve adds a band; clicking the band dot selects; Q zone appears for the selected band only.
5. Ctrl+Z undoes last edit; Ctrl+Shift+Z redoes.
6. Revert-to-baseline is itself undoable with Ctrl+Z.
7. Apply → Local → run; verify `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` PHYSICS_* lines updated and (if bands) `vtsc_curve_tuning.py` Q_CURVE_POINTS rewritten.

## References

- `references/architecture.md` — module-level interaction diagrams + data flow.
- `references/math.md` — sigmoid derivations, plain-knob ↔ raw-params mapping, EQ band math, monotonic-v reasoning.
- `references/changelog.md` — dated entries for every substantive change to the tuner.
- `references/apply-chain.md` — per-step contract for the apply pipeline.

## Skill maintenance

After any real editing / debugging session on the tuner:
- Add a dated entry to `references/changelog.md`.
- If you hit a new repeatable footgun, add it to the "Landmines" section above and link supporting detail in a reference file.
- Correct stale bullets instead of appending contradictions.
- Keep exact file paths, line numbers, and verification commands current — stale landmarks are worse than no landmarks.
- Keep `SKILL.md` focused on durable workflow. Move derivations or longer lists into `references/`.
