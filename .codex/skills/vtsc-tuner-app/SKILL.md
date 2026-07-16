---
name: vtsc-tuner-app
description: Edit, debug, troubleshoot, iterate, and polish the native SwiftUI macOS VTSC tuner at `tools/vtsc_tuner_mac/` or the legacy Rust/egui implementation at `tools/vtsc_tuner/`. Use when changing hero-plot interactions, plain-English knobs, EQ bands, the apply/commit/push/pull-to-tici pipeline, undo/redo, packaging, or either desktop implementation.
---

# VTSC Sigmoid Tuner (desktop app)

A standalone desktop GUI for shaping the VTSC sigmoid (`_physics_based_lateral_acceleration` in `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`) and its optional Q-curve multiplier. The primary Mac implementation is native SwiftUI; the Rust + egui implementation remains for WSL/Linux. Not to be confused with the `vtsc-tuner` skill (that one analyses on-device VTSC interventions).

## Guardrails

- **Don't** confuse this tool with `tools/vtsc_tuner/` vs the deleted `tools/vtsc/curve_tuner.py`. The Python tkinter version was removed on 2026-04-18 — do not resurrect it.
- **Don't** edit the wrong desktop implementation. Use `tools/vtsc_tuner_mac/` for native Mac work and `tools/vtsc_tuner/` for Rust/WSL work; keep their math, tune schema, plot hit priority, and apply actions compatible.
- **Don't** introduce new features without keeping the math in sync with `vision_turn_controller.py` lines 818–823 and `vision_turn_params.py` lines 335–351 (clip ranges).
- **Don't** bypass the existing `plot::hero_plot` monotonic-v post-processing (see "Landmines"). The parametric `(v(κ), a(κ))` trace can fold back on itself; the fix is non-negotiable UX.
- **Don't** bypass `VTSCMath.sampleCurve(... enforceMonotonicSpeed: true)` in the Swift plot either.
- **Don't** remove `event_loop_builder` that forces the X11 winit backend in `main.rs` — WSLg's wayland path crashes on corner-resize.
- **Don't** use the Claude Code Bash tool's `run_in_background: true` to launch the binary from this shell session — it severs the display connection. Launch with `nohup <bin> >log 2>&1 </dev/null & disown` and verify with `pgrep`.

## Location + stack

- Native Mac app: `tools/vtsc_tuner_mac/` (Swift 6, SwiftUI + AppKit + Foundation, macOS 14+, no third-party packages).
- Mac app bundle: `tools/vtsc_tuner_mac/dist/VTSC Tuner.app` after `bash scripts/build_app.sh`; the bundle is intentionally unsandboxed so it can run Git/SSH/rsync and edit the selected checkout.
- Crate: `tools/vtsc_tuner/` (standalone, NOT part of the openpilot SCons build).
- Toolchain: stable Rust (1.89+). egui/eframe 0.33 + glow renderer + winit 0.30 (X11-only on Unix).
- Binary: `tools/vtsc_tuner/target/release/vtsc_tuner` (~10 MB).

## Quick loop (edit → build → run)

Native macOS:

```bash
cd tools/vtsc_tuner_mac
swift test
bash scripts/build_app.sh
open "dist/VTSC Tuner.app"
```

The tune and mapd config live in `~/Library/Application Support/vtsc_tuner/`. Open `Package.swift` directly in Xcode; there is intentionally no generated `.xcodeproj`.

Legacy WSL/Linux:

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

## Native Mac module map

| Path | Responsibility |
|---|---|
| `Sources/VTSCTunerCore/Models.swift`, `VTSCMath.swift` | Rust-schema-compatible tune types, sigmoid/EQ math, monotonic sampling, Q export |
| `Sources/VTSCTunerCore/Repository.swift`, `TuneStore.swift` | Checkout discovery/validation, live baseline parsing, atomic source patching, Application Support persistence |
| `Sources/VTSCTunerCore/ApplyPipeline.swift`, `ProductionDeploymentSupport.swift`, `MapdReleaseArtifact.swift`, `TuneDeploymentIdentity.swift` | Async five-action pipeline, exact Git/release/tune identity, single-reboot tici deployment, postflight, and coherent rollback |
| `Sources/VTSCTunerCore/TileSetManifest.swift`, `TiciTileSetDeployment.swift`, `Map*.swift`, `TiciMapTileSync.swift` | Canonical tile-set build/validation, immutable generation activation/rollback, packed decode, exact bake math/hash, and preview cache sync |
| `Sources/VTSCTunerCore/SigmoidFitter.swift` | Deterministic bounded complete-curve fit from familiar targets to four base knobs plus canonical residual Q bands |
| `Sources/VTSCTunerApp/TunerSession.swift` | App state, grouped undo/redo, selected checkout, apply stream integration |
| `Sources/VTSCTunerApp/MapPreview*.swift`, `StrategicMapView.swift` | MapKit context/search/opening location, actual/proposed/delta mapd overlays, apex selection, persistent no-cap curve bank |
| `Sources/VTSCTunerApp/CurvePlotView.swift`, `RotaryKnob.swift`, `RootView.swift` | Canvas/AppKit plot interaction, controls, workspace chrome, sheets, status |
| `Helpers/VTSCTileDecoder/`, `scripts/build_tile_decoder.sh` | Native arm64 Cap'n Proto helper and pinned, checksum-verified Go build |
| `scripts/build_app.sh`, `Support/Info.plist` | Release build, helper packaging, `.app` ad-hoc signing and verification |

## Legacy Rust module map (`tools/vtsc_tuner/src/`)

| File | Responsibility |
|---|---|
| `main.rs` | `eframe::run_native` entry, diag-log-on-panic, winit X11 forcing |
| `app.rs` | `TunerApp` state, header/footer/side-panel layout, undo/redo, apply modal |
| `plot.rs` | Custom-painted hero graph: drag targets, baseline overlay, Q zones, monotonic-v post-process |
| `knob.rs` | Rotary knob widget; vertical-drag, scroll, shift=fine, dbl-click-reset; optional editable text |
| `params.rs` | `PlainKnobs` (tight/straight accel, transition mph, sharpness) ↔ `SigmoidParams` (A,B,C,D,MIN,MAX) |
| `sigmoid.rs` | Raw math: `SigmoidParams::eval`, `sample_curve`, `Band` + `apply_prepared_bands`, `bands_as_q_curve_points` |
| `apply.rs` | Background-thread apply chain: tune JSON → source patch → git → ssh tici → local bake → rsync |
| `mapd_config.rs` | `MapdConfig { pbf_path, mapd_repo_path, mapd_binary_path?, regions_override? }`; loaded from `~/.config/vtsc_tuner/mapd.json`; step 7 of RebuildTilesAndReboot |
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

## Apply chain contract

Five actions, each is a superset of the previous:

| Action | Tune JSON | Patch src | git commit | git push | exact tici deployment | canonical tile generation/activation |
|---|---|---|---|---|---|---|
| Local | ✓ | ✓ | – | – | – | – |
| Commit | ✓ | ✓ | ✓ | – | – | – |
| Push | ✓ | ✓ | ✓ | ✓ | – | – |
| PullOnTici | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| RebuildTilesAndReboot | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Both car-facing native actions preflight source authorities, a clean exact Git branch, the immutable ARM64 mapd release manifest, tici transport, `IsOffroad=1`, `IsOnroad=0`, and `MTSCLookaheadEnabled=0` before mutation. They run the complete host test gate, push and verify the exact origin commit, fast-forward the tici to that commit, install and verify the release plus its identity-keyed persistent cache, read back all six physics Params and exact Q source, and reboot once. Immediately before reboot the app durably records Linux `/proc/sys/kernel/random/boot_id`; install success requires a different valid post-reboot ID plus stable parked-state brackets and exact Git/Params/Q/mapd/cache/build/process/tile identity. GPS/profile/controller proof remains pending until the separate two-phase outdoor Resume action. A failed static post-reboot proof retains the pending journal and never auto-rolls back. Rebooted rollback uses its own separately captured pre-reboot ID and static-only identity snapshot; it never depends on liveMapData/GPS. Failures before the durable outdoor handoff invoke the rollback journal and recheck parked/offroad/kill-switch state before any rollback mutation or reboot.

All production transactions share one global owner flock in the authoritative deployment-journal directory. Lock order is always global owner, then per-journal resolution lock. The current app publishes its released-reader-unknown `preflightReserved` journal as the first durable namespace action under that owner, before peer detection, pruning, or unresolved scans; those operations exclude only that exact live reservation. Because released app versions do not honor the flock, a car-facing action refuses while another exact VTSC Tuner process is alive and repeats peer plus unresolved-journal checks immediately before mutation claim. It then freshly re-reads and requires the exact recorded device baseline—parked state, boot/Git/Params/Q/mapd build/cache/tile identity—before `mutationInProgress`. Ownership remains held through every mutation and durable rebooted-postflight handoff or rollback settlement, but is released during long outdoor postflight polling. Resume finalization, interrupted rollback recovery, and explicit pending-deployment abort reacquire global then journal ownership. Git fast-forward/reset and reboot also repeat exact parked/kill-switch checks inside the same remote shell immediately before mutation.

`PullOnTici` intentionally leaves tiles untouched; the runtime whole-curve estimator uses existing raw route geometry, so this is the correct first-road-test action. `RebuildTilesAndReboot` additionally preflights `mapd.json`, a prepared PBF, a native Darwin mapd generator, and the bundled decoder; generates into a clean tree; fully decodes and hashes every tile; persists `~/Library/Application Support/vtsc_tuner/map_tiles/sets/<tile-set-id>/`; stages it to an empty remote sibling; and atomically activates one immutable `offline/` plus embedded manifest while retaining the previous generation for rollback. It still reboots only once after every artifact is ready.

- Tune JSON default path is platform-native: `~/Library/Application Support/vtsc_tuner/current.tune.json` on Mac and the `dirs::config_dir()` equivalent on Rust platforms. The Mac app can import the legacy `~/.config/vtsc_tuner/current.tune.json` when the native file is absent.
- Source patch targets:
  - `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` — rewrites lines starting `PHYSICS_A = …` … `PHYSICS_MAX_LAT_ACCEL = …`.
  - `sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py` — rewrites `Q_CURVE_ENABLED` and replaces the entire `Q_CURVE_POINTS: list[tuple[float, float]] = [...]` block (possibly multi-line).
  - `common/params_keys.h` — keeps all six persistent-Param defaults identical to the source-rounded controller constants.
  - `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_physics_panel.cc` — keeps all six reset and ensure literals aligned and preserves the runtime's full parameter ranges.
- Commit message: `vtsc: tune sigmoid via vtsc_tuner`.
- Worker runs on a std thread, events flow through an `mpsc::channel` collapsed by step id in `Task::poll`.
- Each step emits start (⏳ Running) then end (✓ Ok / ✗ Err) with plain-English text + raw command detail.

### SSH profile discovery (tici)

The Mac app uses `networksetup` only as an ordering hint, then autonomously probes the same three SSH aliases with absolute `/usr/bin/ssh` paths. Finder's sparse `PATH` therefore does not affect Git/SSH/rsync.

`apply::detect_ssid()` tries `iwgetid -r` then PowerShell `(Get-NetConnectionProfile | …).Name` (WSL fallback). If the SSID contains "comma", profile order is `commaCar → commaHome → commaAdb`; otherwise `commaHome → commaCar → commaAdb`. Each is probed with `ssh -o ConnectTimeout=2 -o BatchMode=yes <profile> true`. First responder wins.

Profiles must exist in `~/.ssh/config`:
- `commaHome` — home Wi-Fi (192.168.1.172)
- `commaCar` — car hotspot (192.168.0.229)
- `commaAdb` — USB via adb port-forward (127.0.0.1:2222)

Remote sequence: fetch `chauffeur-exp01`, require the exact pushed object, and `git merge --ff-only <exact-head>`; stage the release outside the active path; verify SHA-256, Linux ARM64 ELF identity, release/build markers, capability, and `--build-info`; seed the identity-keyed persistent cache; atomically replace the disposable checkout binary; run `tools/vtsc/apply_physics_params.py` with all six source-rounded values; verify exact Q data; optionally activate one verified canonical tile generation; recheck parked/offroad/kill-switch state; then reboot once. **As of 2026-07-13 the user's tici tracks `chauffeur-exp01`**. If branch state differs, inspect it live rather than trusting static notes.

## Landmines (things that will bite you)

- **SwiftUI Canvas can draw beneath custom `safeAreaInset` chrome on macOS.** `GeometryProxy.safeAreaInsets.top` reflects the 34-point workspace strip here, but the custom 30-point bottom status bar has returned zero in live builds. The curve plot must keep explicit footer clearance in its internal mapper; outside `.padding` cannot rescue Canvas text already clipped at its own bounds.
- **A tile bake is not the final strategic command.** `Way.safeSpeeds` contains per-node physics values. Route matching, adjacent-way smoothing, bias/factor/Q modifiers, learned calibration, deceleration timing, and planner policy remain live. Label current/proposed map values as tile bakes and keep this boundary visible in the inspector.
- **Never fit calibration targets against raw per-node tile curvature.** Schema-v1 `Way.safeSpeeds` use an adjacent-three-node OSM circumcircle, but live mapd publishes an arc-length-weighted three-triplet value over five route nodes after its lane merge/split clamp. Calibration must carry lane/one-way boundaries into the route stencil, fit only the runtime-equivalent curvature, preserve raw curvature as diagnostics, and exclude unresolved, fork-ambiguous, or direction-dependent samples. Bump the stored estimator version whenever parity changes. Keep legacy `MapPreCurveSpeeds` disabled until tile and runtime curvature carry the same estimator version.
- **A fitted Q curve is invalid if runtime Params load a different sigmoid.** Before fitting or applying, require the controller constants, `params_keys.h` defaults, and offroad-panel reset/ensure literals to agree at source precision. Tici apply must migrate and read back all six persistent physics Params while offroad before reboot; changing checked-in defaults alone does not replace already-persistent device values.
- **Go JSON rejects NaN.** Real tiles can contain non-finite safe-speed/metadata values. The helper emits unavailable baked speeds as JSON `null`, zeroes non-finite scalar metadata, and rejects non-finite coordinates; the Swift store also preserves good visible neighbors when one tile cannot decode.
- **Do not merge tile generations during tici deployment.** A production set is immutable at `/data/media/0/osm/tile-generations/<tile-set-id>/` and contains one `offline/` tree plus its embedded manifest. Transfer only to an empty partial sibling, verify the complete manifest on-device, then atomically switch the active pointer and retain the previous pointer. Direct rsync into `/data/media/0/osm/offline` is forbidden. Preview-only cache sync uses its own staged local swap and must not be confused with production activation.
- **Calibration targets use the effective curve speed, not the raw tile bake.** Default desired mph and fitter predictions must use the same explicit modifier snapshot. The Mac fitter replaces current Q/EQ bands with a canonical generated residual set, so acceptance must update knobs and bands atomically; cancel/invalidate a fit whenever samples, knobs, or bands change. Never label source-default modifiers as live device Params.
- **Do not use the four-knob sigmoid as the calibration feasibility envelope.** The Mac proposal refits the bounded backbone so projected targets fit that base's real 0.5–1.5 Q authority, then solves a canonical broad Q=4 residual curve and scores the source-rounded 256-point runtime representation. Preserve weighted monotonic projection, final-base pointwise envelopes, the four-sigma `sqrt(10)` influence collar, dense complete-curve/backbone anchoring beyond that collar, raw-runtime reversal checks, high-acceleration acknowledgement, and the final input knob/band/anchor snapshot check at acceptance. The original bank anchor must persist across apply and relaunch for idempotence, and Curve Lab must expose generated centers above its legacy 90 mph viewport.
- **MapKit does not expose a road-label font control.** Keep Apple labels for geographic context, but render large app-owned labels from each decoded mapd way's `name` / `reference`; deduplicate them, use annotation collision priorities, and stop emitting them at wide zooms. Mapd tiles do not contain a searchable town/address index.
- **Cancelling a Swift search task does not reliably stop its `MKLocalSearch`.** Retain the active search, call `cancel()` when the query changes or clears, and guard results with both a generation token and the normalized current query so stale address results cannot move the map.
- **The Mac tile rebuild must execute a Darwin/Mach-O mapd binary.** An Earthly/Linux ELF output is rejected during preflight. Set `mapd_binary_path` in `~/Library/Application Support/vtsc_tuner/mapd.json` to a native Mac helper; the app deliberately stops before writing the tune or rebooting the tici when this prerequisite is missing.
- **An installed `.app` cannot find Chauffeur by walking up from its executable.** The Mac app persists the chosen checkout, supports `VTSC_REPO_ROOT` / `VTSC_TUNER_REPO`, checks `~/Documents/chauffeur`, and exposes File → Choose Chauffeur Repository. It validates `.git` plus both exact VTSC source targets before apply.
- **PBF must be pre-prepped with `osmium add-locations-to-ways` before `mapd --generate` produces real tiles.** Raw geofabrik extracts (`california-latest.osm.pbf`) only store node IDs on way refs — the osmpbf scanner reads way nodes with `Lat/Lon == 0`, every bbox check fails, and generate writes 26+ tiles each 55–57 bytes with zero ways. VTSC then takes the fallback path because `Way.safeSpeeds` is absent. Symptoms: `--generate` exits cleanly, logs say "Done Generating Offline Map" with zero "Writing Area" entries (or many but each output file ≤57 bytes). Fix: run `mapd_repo/openpilot-mapd/scripts/{filter_planet,add_locations}.sh` or equivalent, save the output as `ca_ready.osm.pbf`, and point `~/.config/vtsc_tuner/mapd.json` `pbf_path` at that — NOT the raw geofabrik file. Use `--platform=linux/amd64` on the docker invocation (without it WSL may run osmium under qemu, 7+ min instead of <1 min).
- **The old silent `mapd_installer.py` failure is fixed.** Terminal download failure raises, version Params are not advanced without a verified binary, and the installer restores from an identity-keyed persistent cache when offline. Current releases must match the configured SHA-256, Linux ARM64 ELF header, release/build markers, `MapWholeCurveProfile:whole-curve-v3` capability, and runtime `--build-info` before manager starts mapd. Inspect the release identity and persistent cache on failure; copying an arbitrary binary directly into the active checkout path is not a normal recovery path.
- **Cross-compile the device binary, do not qemu-build it.** `earthly +build-release` or `docker run --platform=linux/arm64 … go build` both emulate arm64 via qemu — 20+ min on a 1.3 GB PBF. Native amd64 Docker with `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -ldflags="-extldflags=-static -s -w"` produces the exact same binary in ~1 min. Verify with `/usr/bin/file build/mapd` → "ELF 64-bit LSB executable, ARM aarch64 … statically linked, stripped" and grep the binary strings for `sigmoid_hash` / `phys-min-lat` / `winding` to confirm new code was included.
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

Native Mac minimum:

```bash
cd tools/vtsc_tuner_mac
bash scripts/build_tile_decoder.sh --test
swift test
bash scripts/build_app.sh
codesign --verify --deep --strict --verbose=2 "dist/VTSC Tuner.app"
open -n "dist/VTSC Tuner.app"
```

Visually inspect the real app window after launch. Confirm the selected checkout baseline appears in the footer/plot, long knob labels fit, inspector and window resize cleanly, handles and band interactions work, Cmd-Z/Cmd-Shift-Z work, and the apply confirmation/progress sheets cannot close a running action. In Map Preview, confirm the real Application Support cache indexes, search is visible and moves the camera, the selected opening location survives relaunch, actual overlays render, a click snaps to a marked apex, all three color modes work, and more than five curves remain in one bank and reach Fit Review without applying or contacting the car.

Legacy Rust minimum:

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
