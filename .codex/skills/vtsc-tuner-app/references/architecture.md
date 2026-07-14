# VTSC Tuner — architecture

## Native macOS app (`tools/vtsc_tuner_mac/`)

The native port is a SwiftPM package that produces a SwiftUI/AppKit executable and a deterministic unsandboxed `.app` bundle. `VTSCTunerCore` owns value-semantic schema/math/repository/apply behavior; `VTSCTunerApp` owns presentation and input.

```text
Models + VTSCMath ──▶ Repository + TuneStore ──▶ ApplyPipeline + ProcessRunner
        │                         │                          │
        ├──▶ MapBakeMath + MapTileStore ◀── bundled Cap'n Proto helper
        │                 │
        └──▶ SigmoidFitter┴────────────▶ TunerSession ◀──────┘
                                               │
                   RootView + CurvePlotView + MapKit StrategicMapView
```

The Swift tune schema remains compatible with Rust schema v1. The plot preserves the 512-point log-κ sample, mandatory monotonic-speed post-process, and target hit priority. The app parses the selected checkout's live `PHYSICS_*` constants for its baseline and fails preflight if they disagree with the declared Params defaults or offroad-panel reset/ensure values.

The Mac `ApplyPipeline` is an actor with `AsyncStream<ApplyEvent>` progress. Source apply transactionally patches the controller constants, declared Params defaults, offroad-panel reset/ensure literals, and Q curve. Both tici actions preflight a clean exact dev/origin/device commit, parked/offroad/kill-switch state, the immutable Linux ARM64 mapd release, and a durable rollback journal before mutation. They fast-forward to the exact pushed commit, install and cache the verified release, read back all six Params and exact Q source, then reboot once and require a complete postflight. `pullOnTici` deliberately leaves tiles unchanged. Rebuild additionally generates and fully validates a durable canonical set, stages it outside the active tree, atomically activates one immutable generation, and retains the prior generation for rollback. See `tools/vtsc_tuner_mac/README.md` for build, bundle, config, and repository-selection details.

Map Preview treats MapKit as context/navigation only. A bundled native Go helper decodes mapd's packed Cap'n Proto `Offline` tiles into compact JSON, and Swift renders only that actual road geometry. `MapBakeMath` mirrors both schema-v1's raw three-node tile bake and mapd's live arc-length-weighted three-triplet curvature, including the 0.0015 merge/split correction applied before averaging. `MapRuntimeCurvatureResolver` stitches direction-feasible geometry across cached tile boundaries using estimator-v5's deterministic priority: a unique gentle exact name, then a unique gentle exact reference, then a uniquely least-curvature gentle partial reference; identity ties fail closed, and the fallback accepts only one physical continuation. It carries lane/one-way transition metadata into the stencil, also fails closed for direction-dependent bidirectional transitions, and supplies the effective curvature used for apex selection and calibration. The UI therefore distinguishes stored `Way.safeSpeeds`, a proposed raw per-node tile rebake, raw vertex diagnostics, runtime-equivalent curvature, and the later scenario-dependent strategic command.

Real-curve calibration stores raw and effective curvature provenance plus a user-entered desired effective speed. Schema-3 samples retain the raw vertex value for audit but only the runtime-equivalent value may reach `SigmoidFitter`; legacy or unresolved geometry is automatically re-resolved and remains ineligible until route context is complete. `SigmoidFitter` performs a deterministic bounded search over the four plain controls with an explicit penalty when the projected targets would exceed that base's real 0.5–1.5 Q authority, then synthesizes a canonical set of bounded Q=4 residual bands in log-curvature space and evaluates the result through the source-rounded 256-point runtime interpolation. Weighted monotonic projection turns inverted speed requests into a visible compromise; fixed-anchor clustering preserves resolution, while a separate sliding window reports local conflicts. The Gaussian residual's four-sigma influence collar extends by a `sqrt(10)` curvature ratio around the bank; beyond it, dense transition-aware guards require both the backbone and complete exported curve to remain within 2 mph of the bank's persisted checkout anchor. The exported result is also rejected if it introduces a meaningful tighter-curve/faster-command reversal. The review returns per-sample speed and implied-lateral-acceleration residuals, worst miss, and a high-acceleration acknowledgement gate. Acceptance replaces knobs and EQ bands together as one explicit undoable tune edit and checks the current knob/band/anchor snapshot again at the acceptance boundary; sample/tune or repository changes invalidate any in-flight or completed fit. The original bank anchor survives apply/relaunch for idempotent regeneration and is cleared with an empty bank. Curve Lab dynamically widens its legacy 0–90 mph domain so every generated center up to the runtime's 156.6 mph cap remains visible and editable. Predictions use checked-in source-default modifiers, not live device Params.

Production mapd now builds one ordered directional route shared by legacy curvature output and `whole-curve-v1`. It carries coordinates, cumulative distance, traversal/source provenance, way boundaries, merge/split context, and ambiguity across approximately 1,200 m, retaining predecessor context through current-way rollover. `MapWholeCurveProfile` publishes versioned signed curvature points and stable events—not final speeds—with route fingerprint/generation, confidence, flags, and freshness. The Python controller accepts only a fresh, finite, internally consistent, nearby matching profile, applies the current source-rounded sigmoid and Q at runtime with learned scaling excluded, and falls back to `MapCurvatures` on any rejection. The falling edge of `MTSCLookaheadEnabled` clears all map tail/hold/release state immediately.

Parked startup uses only a real `LastGPSPosition` restart anchor. `OsmMapData` writes shared memory on every fresh `updated/alive/valid/hasFix` message whose coordinates are finite, ranged, and not `0,0`; it persists the first accepted fix, then throttles flash writes to at least five minutes plus one kilometre, with a 30-minute stationary refresh. Go rejects malformed, stale, future-dated, or out-of-range persistent seeds. Deployment postflight reports missing/invalid real GPS as pending rather than substituting a route coordinate.

The controller deliberately rejects legacy `MapPreCurveSpeeds` until tile metadata carries an estimator version aligned with `MapCurvatures`. Schema-v1 tile speeds are derived from raw triplets, while current mapd publishes smoothed curvatures; pairing the two would reintroduce the same false slowdowns the calibration audit removes. Runtime therefore continues through live sigmoid evaluation at mapd's effective or whole-curve controlling curvature.

## Legacy Rust app (`tools/vtsc_tuner/`)

Module dependency graph (leaf → root):

```
sigmoid.rs ── no deps outside serde/std
params.rs  ── uses sigmoid
theme.rs   ── egui only
knob.rs    ── egui + theme
plot.rs    ── egui + sigmoid + params + theme
io.rs      ── sigmoid (types), dirs, chrono, serde_json
apply.rs   ── sigmoid + io  (spawns std::thread for chain)
app.rs     ── knob + params + plot + sigmoid + theme + apply + io + egui
main.rs    ── app + apply + winit platform ext
```

## TunerApp state (`app.rs`)

```
TunerApp
├── tune: Tune                (persisted to ~/.config/vtsc_tuner/current.tune.json)
│   ├── schema: u32 == 1
│   ├── created: DateTime<Local>
│   ├── note: String           (unused in v1 UI)
│   ├── params: SigmoidParams
│   └── bands: Vec<Band>
├── knobs: PlainKnobs          (derived; synced to tune.params via sync_params_from_knobs)
├── plot_state: PlotState
│   ├── hover_readout: Option<(f64, f64)>
│   ├── selected_band: Option<usize>
│   └── active_drag: DragTarget { None, MinRail, MaxRail, Inflection, SteepWing, Band(usize) }
├── advanced_open: bool
├── push_dialog: Option<PushDialog>   (legacy; dead code path — not wired to any button)
├── status_msg: StatusMessage         (4s timed toast at the footer)
├── history: History                  (undo/redo ring + debounce)
│   ├── past: Vec<Snapshot>
│   ├── future: Vec<Snapshot>
│   ├── idle_frames: u32
│   ├── last_seen: Snapshot
│   └── suppress_capture: bool        (true during restore to avoid double-push)
├── apply_task: Option<apply::Task>   (active chain, mpsc::Receiver owned here)
└── apply_pending: Option<apply::Action>   (between Apply-menu-click and Run-button-click)
```

### Snapshot / History

`Snapshot { knobs, bands }` — purely user-editable state. `History::new` starts with `idle_frames = HISTORY_DEBOUNCE_FRAMES` so the first edit immediately pushes a snapshot. Each frame `TunerApp::maybe_capture` runs AFTER all UI code — it detects transition "idle → changing" and pushes the previous (stable) snapshot. `restore` + `suppress_capture` flag prevents a restored snapshot from re-triggering a push.

Limit: `HISTORY_LIMIT = 200`. Older entries drop from the front of `past`.

## Frame lifecycle (`TunerApp::update`)

Order per frame:
1. `status_msg.tick()` — decrement toast TTL.
2. Input keyboard shortcuts (Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y) via `ctx.input_mut(|i| i.consume_key(...))`.
3. Header `TopBottomPanel::top` — title, Apply▾ menu, Advanced toggle, Revert, Load.
4. Footer `TopBottomPanel::bottom` — status message + raw params row.
5. Right `SidePanel::right("controls")` — macro knobs, EQ band list, (optional) Advanced block.
6. Central `CentralPanel` — baseline cloned, `plot::hero_plot` called.
7. `show_apply_modal(ctx)` — confirmation or progress modal.
8. `maybe_capture()` — snapshot diff + history push.
9. `ctx.request_repaint()` — keep knobs / drags glued.

## Data paths

### Tune file
- Write: `Tune::save_to(path)` → `serde_json::to_string_pretty` → `fs::write`.
- Read: `Tune::load_from(path)` → `fs::read_to_string` → `serde_json::from_str`.
- Auto-load on startup in `TunerApp::new`.

### Legacy Rust source patch (`apply::patch_source`)
1. `patch_physics` — line-by-line replace of `PHYSICS_A = …` … `PHYSICS_MAX_LAT_ACCEL = …`.
   - Reads file, applies `replace_constant` six times, writes back only if content changed.
   - `replace_constant` matches an unindented `NAME =` line; preserves any inline comment / newline tail via `split_value_and_tail`.
   - Precision: A/B/C/D at 6 decimals, MIN/MAX at 4.
2. `patch_q_curve` — single-pass: if `Q_CURVE_ENABLED` line, overwrite with `True`/`False`. If `Q_CURVE_POINTS` line, emit fresh multi-line list literal from `bands_as_q_curve_points(params, bands, 64)` — scientific κ + 4-decimal q. Swallows the original list body if it spans multiple lines.

### Git
- `git_commit`: `git add <touched paths>` per file, then `git commit -m "vtsc: tune sigmoid via vtsc_tuner"`. Any failure (non-zero) returns Err with stdout+stderr.
- `git_push`: `git push` on the current branch. Reports stdout+stderr regardless of success.

### SSH (tici)
- `pick_tici_profile`: SSID hint → ordered probe list → first `ssh … true` success.
- Legacy Rust `ssh_pull_and_reboot`: two separate `ssh` invocations — `cd /data/openpilot && git pull` (synchronous, result matters) then `sudo reboot` (fire-and-forget because the connection drops mid-command).
- Native Mac `ApplyPipeline`: preflight first, exact-push/fast-forward, install and cache the verified ARM64 mapd release, run the rollback-protected `tools/vtsc/apply_physics_params.py` with all six values, verify exact Q data, optionally atomically activate a canonical tile generation, then reboot once and require full postflight. Rollback freshly rechecks parked/offroad/kill-switch state before mutation.

## Hero plot data flow (`plot::hero_plot`)

```
knobs ──to_sigmoid──▶ params ──sample_curve──▶ samples (512 points, log κ)
                │                                │
                │                                └──monotonic-v post-process──▶ curve_pts
                │                                                                  │
bands + params ──prepare_bands──▶ prepped  (Vec<PreparedBand>)                     │
                                                                                   │
painter draws:                                                                     │
  1. baseline (dashed, TEXT_MUTED * 0.55) from `SigmoidParams::default()`          │
  2. glow underlay (thick, accent * 0.18)                                          ▼
  3. curve (accent, 2.5px)                                                    painter.add
  4. min/max rails + pill handles
  5. Q zone (only for selected band)
  6. band dots (colored)
  7. steepness wings + inflection circle
  8. hover crosshair + readout
```

Drag dispatch in `pick_drag_target` picks the highest-priority target under the pointer with `+2` / `+4` tolerance inflations (inflection gets `+2`, wings get `+4`).

## Apply modal (`app::show_apply_modal`)

State machine:
```
(no dialog) ──menu click──▶ apply_pending: Some(action), apply_task: None
                                    │
                                    ├──Cancel──▶ apply_pending = None
                                    └──Run─────▶ apply_task = Some(Task::spawn(...))
                                                 apply_pending = None
                                                        │
                                                 task.running == true
                                                        │
                                                 (mpsc events drain each frame via task.poll)
                                                        │
                                                 task.running == false (Done event received)
                                                        │
                                                 ──Close──▶ apply_task = None
                                                            status_msg set to summary
```

Events carry `id: u32` so start (Running) and end (Ok/Err) collapse into a single display row via `Task::poll`'s find-by-id.
