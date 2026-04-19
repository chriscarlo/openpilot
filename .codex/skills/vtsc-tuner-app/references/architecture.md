# VTSC Tuner — architecture

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

### Source patch (`apply::patch_source`)
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
- `ssh_pull_and_reboot`: two separate `ssh` invocations — `cd /data/openpilot && git pull` (synchronous, result matters) then `sudo reboot` (fire-and-forget because the connection drops mid-command). 200 ms sleep after reboot to ensure signal is sent.

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
