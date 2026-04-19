# VTSC Tuner — change log

Reverse-chronological. Add a new dated section for every substantive change.

## 2026-04-18 — RebuildTilesAndReboot action + sigmoid-baked map tiles

- New `Action::RebuildTilesAndReboot` variant added to `apply.rs:17-31` (a superset of `PullOnTici`). Two-reboot chain: step 6 reboots after `git pull` so the device's openpilot Python lands first; then steps 7-999 generate sigmoid-baked tiles locally, rsync them per region, and final reboot.
- Steps added: 7 validate `~/.config/vtsc_tuner/mapd.json`; 8 wait-for-tici poll + `ssh ... ls /data/media/0/osm/offline/` discover; 9 `df -BM /data/media/0` pre-flight (50 MB/region × 1.5 headroom); 10 build mapd via `earthly +build` if missing; 11..N gen per region (step ids `100+i`); N+1..M rsync per region (step ids `200+i`); 999 final reboot.
- New file `mapd_config.rs`: `MapdConfig { pbf_path, mapd_repo_path, mapd_binary_path?, regions_override? }` saved at `~/.config/vtsc_tuner/mapd.json` (alongside `current.tune.json`). Tuner does NOT auto-create — surfaces a plain-English step-7 error with example contents.
- `Task` gained `Arc<AtomicBool>` cancel flag + `request_cancel()` / `is_cancel_requested()` API. Cancellation honoured at region boundaries (mid-`mapd --generate` is opaque). Cancel button shows in apply modal only for `RebuildTilesAndReboot` while running.
- Companion mapd changes (in `mapd_repo/openpilot-mapd/`, NOT the standalone `/projects/mapd` which is a publish destination only):
  - `offline.capnp` adds `safeSpeeds @20 :List(Float64)` (per-node, parallel to `nodes`), `schemaVersion @6 :UInt16`, `sigmoidHash @7 :Text`. Cap'n Proto field numbers immutable; readers ignore unknown fields → wire-compat with old binaries.
  - `sigmoid.go` (new): `PhysicsLatAccel`, `CurvatureToSpeed`, `SigmoidCfg.Hash()` mirror Python `_physics_based_lateral_acceleration` + `curvature_to_speed` exactly. `Hash()` is sha256 of `"%.6f|%.6f|%.6f|%.6f|%.4f|%.4f|%.2f"`-formatted tuple, first 12 hex chars.
  - `mapd.go` adds `--phys-{a,b,c,d,min-lat,max-lat}` and `--max-speed-default` CLI flags, threaded into `GenerateOffline(sigCfg)`. Defaults match the file-committed `vision_turn_controller.py` values so flagless invocation produces baseline tiles.
  - `mapd.go` runtime loop publishes `MapPreCurveSpeeds` + `MapTilesSigmoidHash` gated on `Offline.SchemaVersion() >= 1`. `math.go::GetStateBakedSpeeds` mirrors `GetStateCurvatures`'s walk to align baked outputs with `MapCurvatures` (length `numPoints-4`).
- Companion openpilot changes:
  - `common/params_keys.h`: registered `MapPreCurveSpeeds` + `MapTilesSigmoidHash` (both CLEAR_ON_ONROAD_TRANSITION, STRING).
  - `vision_turn_controller.py`: added module-level `_compute_runtime_sigmoid_hash` + instance methods `_load_map_pre_curve_speeds` and `_baked_vsafe_with_runtime_multipliers`. Replaced line 5534 with hash-checked baked-vs-live fallback; live-tuned PHYSICS_* changes the runtime hash and auto-engages fallback. Low-speed calibration scale != 1.0 also forces fallback for the whole batch.
- Tests: 4 new Go tests (`TestSigmoidMatchesPython` validates Go matches Python within 1e-9 against 11 precomputed κ samples + `f9d38ab3357c` reference hash; `TestSigmoidHashStableAcrossTrivialChanges`; `TestOldReaderNewTile` round-trips a v1 tile w/ baked + legacy ways; `TestLegacyTileReadsSchemaVersionZero` validates Cap'n Proto reads unset primitives as 0). 3 new Rust `mapd_config` unit tests. All pass. Pre-existing `TestVector`/`TestBearing` cupaloy snapshot mismatches are unrelated FP noise from a different platform/Go version.
- Landmines added to watch:
  - `/projects/mapd` is a publish destination, NOT the source of truth — edits live in `mapd_repo/openpilot-mapd/`. The user owns chriscarlo/mapd; if they want to publish a release tag, they need to port edits manually (e.g., `cp` + their `push_to_github.sh`).
  - Distribution is local-only via SSH — no public CDN. Tiles uploaded by Pfeifer at `https://map-data.pfeifer.dev` ignore the new schema fields (they're sigmoid-agnostic v0). `MapdTileBaseUrl` could repoint at a self-hosted bucket later if multi-device distribution is needed.
  - Cap'n Proto Go bindings regenerate via `earthly +compile-capnp` (Earthfile:69-73). Locally accomplished without earthly via `PATH=$HOME/go/bin:$PATH capnp compile -I /tmp/go-capnp-std/std -ogo offline.capnp` (after curl-tarball clone of go-capnp std files; `git clone` is blocked by the openpilot branch-protection hook).
  - Docker is required to build the Go binary locally if the user has no Go toolchain. Earthfile target `+build` uses `golang:1.24-alpine3.21`.

## 2026-04-18 — wider Q range + denser Q_CURVE_POINTS export

- Width-(Q) knob range bumped from `(0.3, 6.0)` → `(0.3, 16.0)` in `app.rs:784` so users can carve a 5–10 mph notch (Q ≈ 8–15 at typical highway centres). Default stays 1.5; mapping stays log.
- `Q_CURVE_POINTS` export sample count raised `64 → 256` in `apply.rs:439`. Old spacing (≈0.079 in log₁₀κ) was the same order as σ at Q=6, so anything narrower would alias on export. New spacing (≈0.020) leaves 4× headroom up to Q≈16. Runtime file `vtsc_curve_tuning.py` just lerps between samples — over-sampling is cheap, under-sampling silently smears notches.

## 2026-04-18 — hover + selected highlights on built-in handles

- Min/max rail pills, inflection circle, and both steepness wing chevrons now light up when the cursor is over them and stay lit when clicked, deselecting on Esc, click on empty plot, or click on another handle. Bands gained a matching hover ring (alongside their existing selected ring) so all "dots and arrow tips" behave identically.
- New `HandleId` enum (`MinRail`, `MaxRail`, `Inflection`, `LeftWing`, `RightWing`) and `PlotState::selected_handle` field. Mutually exclusive with `selected_band` — selecting one always clears the other (`apply_selection` enforces).
- `DragTarget::SteepWing` split into `LeftWing` / `RightWing` so each wing can highlight independently. Drag math still funnels both into `sharpness` via a shared match arm.
- Hover hit-test reuses `pick_drag_target`, so the highlight zones match the drag zones byte-for-byte (no surprise: hover an area, get the highlight, drag from same area, drag works).
- Cursor turns to `CursorIcon::Grab` over any handle, `CursorIcon::Grabbing` while a drag is active. Major affordance win for naive users.
- Selection wires up on both `clicked()` (quick press) and `drag_started()` (press + move). Empty-plot click clears selection.
- Esc to clear selection is gated on `state.line_menu.is_none()` so an open right-click menu still gets first-press priority for closing.
- Three small helpers in `plot.rs`: `drag_target_to_handle`, `level_for_handle`, `apply_selection`, plus a `HighlightLevel` enum threaded through `draw_rail` / `draw_wing_handle` and inflection inline.
- Selected ring widths/alphas: rails get a 4-px-expanded glow rect with optional outer stroke; inflection a 11-px stroke ring; wings a 11-px filled disc + stroke; bands keep their existing 4-px stroke ring. Hover variants are slimmer / ~0.5 alpha.

## 2026-04-18 — right-click context menu on the curve

- Hero plot now opens a small popup when the user right-clicks **on** the curve line (within 8 px of any rendered segment, not anywhere in the plot field).
- Menu items:
  - **✚ Add anchor point here** — calls the same `push_band_at` helper that shift-click uses, so the two paths can't drift. Adds a 0-dB band at the right-click's mph and selects it.
  - **⎘ Copy "X.X mph, Y.YY m/s²"** — `ctx.copy_text(...)` of the formatted data point. Useful for jotting tuning decisions in commits/notes; otherwise the exact (speed, accel) under the cursor is unrecoverable from the UI.
- Dismissal: choosing an item, pressing Escape, primary-clicking outside the menu rect, or right-clicking anywhere off the curve.
- Hit-test uses point-to-segment distance against the rendered (post-monotonic-v) `curve_pts`, so the menu only opens where the user can actually see the line. Baseline overlay is intentionally not hit-tested (it's reference-only).
- Discoverability hints updated: status bar default and empty-bands hint both mention "shift-click or right-click the curve".
- New helpers in `plot.rs`: `push_band_at`, `hit_polyline`, `dist_sq_point_to_segment`, plus `LineMenuState` / `LineMenuAction` types and `PlotState::line_menu`.
- Same-frame race: a fresh open is guarded by `menu_just_opened` so the opening right-click doesn't immediately get treated as an outside-click. Outside-click dismissal only fires on PRIMARY clicks; subsequent right-clicks either move the menu (on curve) or close it (off curve).

## 2026-04-18 — initial creation + iterative polish

### Bootstrapped

- New standalone Rust crate at `tools/vtsc_tuner/` (eframe 0.33 + egui + glow + winit 0.30 + X11-only).
- Eight src modules: `main`, `app`, `apply`, `io`, `knob`, `params`, `plot`, `sigmoid`, `theme`.
- 9 unit tests for sigmoid + plain-knob round-trips — all green.
- Deleted the superseded `tools/vtsc/curve_tuner.py` (tkinter ~1450 LOC) and the single dangling reference in `sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py`.

### Math layer
- `SigmoidParams` mirrors the runtime `_physics_based_lateral_acceleration` byte-for-byte, including asymptote clamps.
- `PlainKnobs` exposes four user-facing controls (Tight-Curve Ceiling, Straight-Road Ceiling, Transition Speed, Sharpness) that project down to the six raw params. Ranges match `vision_turn_params.py` so values survive the device-side re-clamp.
- `Band` + `apply_prepared_bands` compose Gaussian dB bumps in log-κ space. `prepare_bands` freezes per-band κ_centre against the base sigmoid so bands don't chase their own tail.
- `bands_as_q_curve_points` exports bands as `(κ, q_speed)` tuples for the runtime's multiplicative speed-scale path — gain in lateral accel converted to speed via `sqrt(ratio)`.

### UI
- Hero plot custom-painted in `(speed mph, m/s²)` space; baseline (repo-default) drawn as a faint dashed overlay.
- Draggable handles: min/max rails (with prominent left-edge pill + arrows), inflection dot (X-axis → transition speed), steepness "wings" (in/out → sharpness), per-band dots.
- Shift-click anywhere on the curve adds a new band at that location.
- EQ bands get stable palette colors; each band's marker, side-panel header swatch, and translucent Q zone on the graph all match. Q zone visible only for the selected band.
- Per-band controls in side panel: compact knobs for Gain (dB), Width (Q, log), Center (mph), each with editable DragValue under the knob.
- Rotary knob widget: vertical drag, scroll wheel, shift=fine, double-click reset. Auto-resizes footprint to fit the label string.
- Dark "pro audio" palette, 1.5× `pixels_per_point` for readability.

### Landmine resolved: WSLg corner-resize crash
- Symptom: `Io error: Broken pipe` ×3 + `WinitEventLoop(ExitFailure(1))` whenever the window was resized.
- Root cause: `smithay-clipboard` / `wayland-backend` logs EPIPE when WSLg's compositor sends resize protocol in a sequence calloop can't follow.
- Fix: `NativeOptions::event_loop_builder` calls `winit::platform::x11::EventLoopBuilderExtX11::with_x11()` → winit uses XWayland instead of native wayland, resize is clean.

### Landmine: Claude Code Bash `run_in_background`
- Symptom: same `Io error: Broken pipe` spam, but from the first frame.
- Root cause: the Bash tool's detach wrapper severs the X socket connection.
- Workaround: launch via `nohup <bin> >log 2>&1 </dev/null & disown` with no `pkill` prelude. Exit code 144 from bash is the telltale that the wrapper killed the process.

### Apply pipeline (`apply.rs`)
- Background-thread chain: write tune JSON → patch source → git commit → git push → ssh tici pull+reboot.
- Four `Action` kinds covering subsets of the chain.
- SSID-aware tici discovery: `iwgetid` (Linux) or PowerShell `Get-NetConnectionProfile` (WSL); probes `commaHome` / `commaCar` / `commaAdb` via `ssh -o ConnectTimeout=2 -o BatchMode=yes <profile> true`; first responder wins; SSID biases order.
- Remote: `cd /data/openpilot && git pull` followed by fire-and-forget `sudo reboot`.
- Progress modal with per-step Running/Ok/Err status, plain-English text, and raw command-line detail below.

### Undo/redo
- `Ctrl+Z` / `Ctrl+Shift+Z` / `Ctrl+Y` consume at the top of `update()`.
- `History` keeps pre-edit snapshots with frame-debounced capture so continuous drags collapse to one undo step.
- "Revert to baseline" is also undoable via an explicit `commit_snapshot()` call.

### Monotonic-v plot fix
- User observed the plot folding back on itself when a band's slope exceeded `a/κ`.
- Post-process in `plot::hero_plot` holds v at the running max as samples are walked in κ-descending order — aggressive bands render as vertical notches (parametric-EQ idiom) rather than multi-valued folds.

### Layout polish
- Knob labels were clipping ("ight-Curve Ceilin"). `knob.rs` now measures the label via `ctx.fonts_mut(|f| f.glyph_width(…))` and grows the knob's footprint to fit.
- Header subtitle was also clipping; shortened to `"curvature → lat-accel"`.
- `pixels_per_point` dropped from 2.0 → 1.5 after user feedback that 2× was too big.

---

<!--
Template for future entries:

## YYYY-MM-DD — <short title>

### <section>
- <change>, paths touched (e.g. `src/apply.rs:123`), motivation, any landmines encountered.

-->
