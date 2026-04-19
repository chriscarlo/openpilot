//! Top-level eframe app: layout, widget composition, file I/O glue.

use eframe::egui::{
  self, Align, Align2, CentralPanel, Color32, CornerRadius, Frame, Id, Layout, Margin, RichText,
  ScrollArea, SidePanel, Stroke, TopBottomPanel, Vec2, pos2, vec2,
};

use crate::apply;
use crate::io::{self, DeviceProfile, Tune};
use crate::knob::Knob;
use crate::params::{PlainKnobs, clip};
use crate::plot::{self, PlotState};
use crate::sigmoid::{Band, SigmoidParams};
use crate::theme;

pub struct TunerApp {
  tune: Tune,
  knobs: PlainKnobs,
  plot_state: PlotState,
  advanced_open: bool,
  push_dialog: Option<PushDialog>,
  status_msg: StatusMessage,
  history: History,
  /// Active "Apply" chain (write tune → patch source → optional commit / push
  /// / SSH-pull-on-tici).  Modal blocks the rest of the UI while it runs.
  apply_task: Option<apply::Task>,
  /// Pending "Apply <action>?" confirmation modal — held separately so the
  /// user can cancel before the worker thread is spawned.
  apply_pending: Option<apply::Action>,
}

/// Snapshot of everything the user can edit, used by undo / redo.
#[derive(Debug, Clone, PartialEq)]
struct Snapshot {
  knobs: PlainKnobs,
  bands: Vec<Band>,
}

const HISTORY_LIMIT: usize = 200;
const HISTORY_DEBOUNCE_FRAMES: u32 = 18; // ~0.3 s @ 60 fps

struct History {
  past: Vec<Snapshot>,
  future: Vec<Snapshot>,
  /// Frames since the last user-visible state change. We push the current
  /// state into `past` only after the change has settled, so a continuous
  /// drag collapses into one undo step instead of dozens.
  idle_frames: u32,
  /// Last snapshot we recorded so we can detect "did anything change?"
  /// without paying for a clone every frame.
  last_seen: Snapshot,
  /// True while we're applying an undo / redo so we don't immediately
  /// re-snapshot the state we just restored.
  suppress_capture: bool,
}

impl History {
  fn new(initial: Snapshot) -> Self {
    Self {
      past: Vec::new(),
      future: Vec::new(),
      // Start "settled" so the very first edit triggers a snapshot.
      idle_frames: HISTORY_DEBOUNCE_FRAMES,
      last_seen: initial,
      suppress_capture: false,
    }
  }
}

struct PushDialog {
  profile: DeviceProfile,
  commands: Vec<String>,
}

struct StatusMessage {
  text: String,
  accent: Color32,
  // Frames-remaining to display. Cheap; avoids a timer.
  ttl: u32,
}

impl Default for StatusMessage {
  fn default() -> Self {
    Self {
      text: String::from(
        "Ready. Drag curve anchors • shift-click or right-click the curve to add a band.",
      ),
      accent: theme::TEXT_MUTED,
      ttl: u32::MAX,
    }
  }
}

impl StatusMessage {
  fn set(&mut self, text: impl Into<String>, accent: Color32) {
    self.text = text.into();
    self.accent = accent;
    self.ttl = 240; // ~4 s at 60 fps
  }
  fn tick(&mut self) {
    if self.ttl != u32::MAX && self.ttl > 0 {
      self.ttl -= 1;
      if self.ttl == 0 {
        *self = Self::default();
      }
    }
  }
}

impl Default for TunerApp {
  fn default() -> Self {
    let tune = io::Tune::default();
    let knobs = PlainKnobs::from_sigmoid(&tune.params);
    let initial = Snapshot {
      knobs,
      bands: tune.bands.clone(),
    };
    Self {
      tune,
      knobs,
      plot_state: PlotState::default(),
      advanced_open: false,
      push_dialog: None,
      status_msg: StatusMessage::default(),
      history: History::new(initial),
      apply_task: None,
      apply_pending: None,
    }
  }
}

impl TunerApp {
  pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
    theme::install(&cc.egui_ctx);
    let mut app = Self::default();
    // Try to load the last saved tune; silently fall through on error.
    if let Ok(loaded) = Tune::load_from(&io::default_file()) {
      app.tune = loaded;
      app.knobs = PlainKnobs::from_sigmoid(&app.tune.params);
    }
    app.history = History::new(app.snapshot());
    app
  }

  fn sync_params_from_knobs(&mut self) {
    self.tune.params = self.knobs.to_sigmoid();
  }

  fn snapshot(&self) -> Snapshot {
    Snapshot {
      knobs: self.knobs,
      bands: self.tune.bands.clone(),
    }
  }

  fn restore(&mut self, snap: &Snapshot) {
    self.knobs = snap.knobs;
    self.tune.bands = snap.bands.clone();
    self.tune.params = self.knobs.to_sigmoid();
  }

  /// Capture an immediate snapshot, bypassing the idle-frame debounce.
  /// Use for discrete actions like "Revert to baseline" so a single Ctrl-Z
  /// undoes them.
  fn commit_snapshot(&mut self) {
    let snap = self.snapshot();
    if self.history.last_seen == snap {
      return;
    }
    self.history.past.push(self.history.last_seen.clone());
    if self.history.past.len() > HISTORY_LIMIT {
      let drop = self.history.past.len() - HISTORY_LIMIT;
      self.history.past.drain(0..drop);
    }
    self.history.future.clear();
    self.history.last_seen = snap;
    self.history.idle_frames = 0;
  }

  fn undo(&mut self) -> bool {
    if let Some(prev) = self.history.past.pop() {
      let cur = self.snapshot();
      self.history.future.push(cur);
      self.restore(&prev);
      self.history.last_seen = prev;
      self.history.suppress_capture = true;
      true
    } else {
      false
    }
  }

  fn redo(&mut self) -> bool {
    if let Some(next) = self.history.future.pop() {
      let cur = self.snapshot();
      self.history.past.push(cur);
      self.restore(&next);
      self.history.last_seen = next;
      self.history.suppress_capture = true;
      true
    } else {
      false
    }
  }

  /// Confirmation dialog (before run) and live progress modal (during / after).
  /// Held as a single method so the flow lives in one place.
  fn show_apply_modal(&mut self, ctx: &egui::Context) {
    // Drain worker events into the task before painting.
    if let Some(task) = self.apply_task.as_mut() {
      task.poll();
    }

    // ---- Confirmation phase ----
    if let Some(action) = self.apply_pending {
      if self.apply_task.is_some() {
        // Already running — clear the pending flag.
        self.apply_pending = None;
      } else {
        let mut keep_open = true;
        let mut launch = false;
        let mut cancel = false;
        egui::Window::new(format!("{}  ?", action.label()))
          .id(egui::Id::new("apply_confirm"))
          .collapsible(false)
          .resizable(false)
          .default_width(560.0)
          .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
          .open(&mut keep_open)
          .show(ctx, |ui| {
            ui.label(
              RichText::new(action.description())
                .color(theme::TEXT)
                .size(13.0),
            );
            ui.add_space(8.0);
            if matches!(
              action,
              apply::Action::Push | apply::Action::PullOnTici
            ) {
              ui.label(
                RichText::new(
                  "Pushes the *current* branch.  Tici only sees the new tune \
                   if its tracked branch matches what was just pushed.",
                )
                .color(theme::TEXT_MUTED)
                .size(11.0)
                .italics(),
              );
              ui.add_space(6.0);
            }
            if matches!(action, apply::Action::PullOnTici) {
              if let Some(ssid) = apply::detect_ssid() {
                ui.label(
                  RichText::new(format!("Active SSID: {ssid}"))
                    .color(theme::TEXT_DIM)
                    .size(11.0),
                );
              } else {
                ui.label(
                  RichText::new("Could not read SSID — will probe profiles in default order.")
                    .color(theme::TEXT_DIM)
                    .size(11.0),
                );
              }
              ui.add_space(6.0);
            }
            ui.horizontal(|ui| {
              if ui.button("Cancel").clicked() {
                cancel = true;
              }
              ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("Run").clicked() {
                  launch = true;
                }
              });
            });
          });
        if cancel || !keep_open {
          self.apply_pending = None;
        } else if launch {
          let repo = apply::detect_repo_root().unwrap_or_else(|| std::path::PathBuf::from("."));
          self.apply_task = Some(apply::Task::spawn(
            action,
            self.knobs.to_sigmoid(),
            self.tune.bands.clone(),
            repo,
            io::default_file(),
          ));
          self.apply_pending = None;
        }
      }
    }

    // ---- Progress / result phase ----
    if let Some(task) = self.apply_task.as_ref() {
      let action = task.action;
      let running = task.running;
      let success = task.success;
      let events = task.events.clone();
      let mut close = false;
      let mut keep_open = true;
      let title = if running {
        format!("{}  …", action.label())
      } else if success == Some(true) {
        format!("{}  ✓", action.label())
      } else {
        format!("{}  ✗", action.label())
      };
      egui::Window::new(title)
        .id(egui::Id::new("apply_progress"))
        .collapsible(false)
        .resizable(true)
        .default_width(640.0)
        .min_height(280.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .open(&mut keep_open)
        .show(ctx, |ui| {
          ScrollArea::vertical()
            .max_height(360.0)
            .auto_shrink([false, true])
            .show(ui, |ui| {
              for ev in &events {
                let (glyph, color) = match ev.status {
                  apply::StepStatus::Running => ("⏳", theme::TEXT_MUTED),
                  apply::StepStatus::Ok => ("✓", theme::ACCENT),
                  apply::StepStatus::Err => ("✗", theme::RAIL),
                };
                ui.horizontal(|ui| {
                  ui.label(RichText::new(glyph).color(color).strong().size(14.0));
                  ui.label(
                    RichText::new(&ev.text)
                      .color(theme::TEXT)
                      .size(13.0),
                  );
                  if matches!(ev.status, apply::StepStatus::Running) {
                    ui.add_space(6.0);
                    ui.spinner();
                  }
                });
                if !ev.detail.is_empty() {
                  Frame::NONE
                    .fill(theme::PANEL)
                    .inner_margin(Margin::symmetric(10, 6))
                    .corner_radius(CornerRadius::same(3))
                    .show(ui, |ui| {
                      ui.label(
                        RichText::new(&ev.detail)
                          .monospace()
                          .size(11.0)
                          .color(theme::TEXT_DIM),
                      );
                    });
                }
                ui.add_space(4.0);
              }
            });
          ui.add_space(8.0);
          ui.horizontal(|ui| {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
              if running {
                ui.add_enabled(false, egui::Button::new("Close (running…)"));
                // Cancel button — only meaningful for the long tile-rebuild
                // chain, but cheap to show for any running task. Cancellation
                // takes effect at the next region boundary in the worker.
                if matches!(action, apply::Action::RebuildTilesAndReboot) {
                  if let Some(t) = self.apply_task.as_ref() {
                    let label = if t.is_cancel_requested() {
                      "Cancelling…"
                    } else {
                      "Cancel"
                    };
                    let resp = ui.add_enabled(
                      !t.is_cancel_requested(),
                      egui::Button::new(label),
                    );
                    if resp.clicked() {
                      t.request_cancel();
                    }
                  }
                }
              } else if ui.button("Close").clicked() {
                close = true;
              }
            });
          });
        });
      if !running && (close || !keep_open) {
        // Surface a one-line result in the status bar before clearing.
        let summary = match success {
          Some(true) => format!("{} succeeded.", action.label()),
          Some(false) => format!("{} FAILED — see modal for details.", action.label()),
          None => format!("{} ended (status unknown).", action.label()),
        };
        let color = if success == Some(true) {
          theme::ACCENT
        } else {
          theme::RAIL
        };
        self.status_msg.set(summary, color);
        self.apply_task = None;
      }
    }
  }

  /// Called once per frame after UI processing.  Records ONE history entry per
  /// contiguous edit (a continuous drag collapses into one undo step) by
  /// pushing the pre-edit snapshot the first frame state changes after a
  /// period of stillness.
  fn maybe_capture(&mut self) {
    let cur = self.snapshot();
    let changed = cur != self.history.last_seen;

    if self.history.suppress_capture {
      // Just restored via undo / redo; let the state re-stabilise without
      // generating a new entry.
      if !changed {
        self.history.suppress_capture = false;
      }
      self.history.last_seen = cur;
      return;
    }

    if changed && self.history.idle_frames >= HISTORY_DEBOUNCE_FRAMES {
      // First frame of a new edit after a settled period — record the
      // pre-edit snapshot as the undo target.
      self.history.past.push(self.history.last_seen.clone());
      if self.history.past.len() > HISTORY_LIMIT {
        let drop = self.history.past.len() - HISTORY_LIMIT;
        self.history.past.drain(0..drop);
      }
      self.history.future.clear();
    }
    if changed {
      self.history.idle_frames = 0;
    } else {
      self.history.idle_frames = self.history.idle_frames.saturating_add(1);
    }
    self.history.last_seen = cur;
  }
}

impl eframe::App for TunerApp {
  fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
    self.status_msg.tick();

    // --------------------------------------------------------- shortcuts
    let (mut do_undo, mut do_redo) = (false, false);
    ctx.input_mut(|i| {
      let z = egui::Key::Z;
      let y = egui::Key::Y;
      // Ctrl+Z (no shift) → undo;  Ctrl+Shift+Z or Ctrl+Y → redo.
      if i.consume_key(egui::Modifiers::COMMAND, z) {
        do_undo = true;
      }
      if i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, z)
        || i.consume_key(egui::Modifiers::COMMAND, y)
      {
        do_redo = true;
      }
    });
    if do_undo {
      if self.undo() {
        self
          .status_msg
          .set("Undid last change.", theme::ACCENT);
      } else {
        self
          .status_msg
          .set("Nothing to undo.", theme::TEXT_MUTED);
      }
    }
    if do_redo {
      if self.redo() {
        self
          .status_msg
          .set("Redid change.", theme::ACCENT);
      } else {
        self
          .status_msg
          .set("Nothing to redo.", theme::TEXT_MUTED);
      }
    }

    // --------------------------------------------------------- header
    TopBottomPanel::top("header")
      .frame(
        Frame::NONE
          .fill(theme::BG)
          .inner_margin(Margin::symmetric(14, 10)),
      )
      .show(ctx, |ui| {
        ui.horizontal(|ui| {
          ui.label(
            RichText::new("VTSC Sigmoid Tuner")
              .color(theme::TEXT)
              .strong()
              .size(16.0),
          );
          ui.add_space(12.0);
          ui.label(
            RichText::new("curvature → lat-accel")
              .color(theme::TEXT_DIM)
              .size(11.0),
          );
          ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            // Apply ▾  dropdown (Local / Commit / Push / Pull-on-tici).
            ui.menu_button("Apply ▾", |ui| {
              for act in apply::Action::ALL {
                if ui
                  .button(act.label())
                  .on_hover_text(act.description())
                  .clicked()
                {
                  self.sync_params_from_knobs();
                  self.apply_pending = Some(act);
                  ui.close_menu();
                }
              }
            })
            .response
            .on_hover_text(
              "Write the tune locally, optionally commit / push, or push and \
               pull + reboot on a reachable tici device.",
            );
            ui.add_space(6.0);
            if ui
              .toggle_value(&mut self.advanced_open, "Advanced")
              .on_hover_text("Show raw A / B / C / D / MIN / MAX fields.")
              .changed()
            {}
            ui.add_space(6.0);
            if ui
              .button("Revert to baseline")
              .on_hover_text(
                "Restore the repo-default sigmoid and clear all bands.\n\
                 Undoable with Ctrl+Z.",
              )
              .clicked()
            {
              self.knobs = PlainKnobs::from_sigmoid(&SigmoidParams::default());
              self.tune.bands.clear();
              self.commit_snapshot();
              self
                .status_msg
                .set("Reverted to baseline.  Ctrl+Z to undo.", theme::ACCENT);
            }
            ui.add_space(6.0);
            if ui.button("Load").clicked() {
              match Tune::load_from(&io::default_file()) {
                Ok(t) => {
                  self.tune = t;
                  self.knobs = PlainKnobs::from_sigmoid(&self.tune.params);
                  self.status_msg.set("Loaded saved tune.", theme::ACCENT);
                }
                Err(e) => self
                  .status_msg
                  .set(format!("Load failed: {e}"), theme::RAIL),
              }
            }
          });
        });
      });

    // --------------------------------------------------------- footer
    TopBottomPanel::bottom("footer")
      .frame(
        Frame::NONE
          .fill(theme::BG)
          .inner_margin(Margin::symmetric(14, 8)),
      )
      .show(ctx, |ui| {
        ui.horizontal(|ui| {
          ui.label(
            RichText::new(&self.status_msg.text)
              .color(self.status_msg.accent)
              .size(11.0),
          );
          ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let raw = self.knobs.to_sigmoid();
            ui.label(
              RichText::new(format!(
                "A={:.3}  B={:>7.0}  C={:.5}  D={:.3}  MIN={:.2}  MAX={:.2}",
                raw.a, raw.b, raw.c, raw.d, raw.min_lat, raw.max_lat
              ))
              .color(theme::TEXT_DIM)
              .monospace()
              .size(11.0),
            );
          });
        });
      });

    // --------------------------------------------------------- right sidebar
    SidePanel::right("controls")
      .resizable(false)
      .exact_width(320.0)
      .frame(
        Frame::NONE
          .fill(theme::BG)
          .inner_margin(Margin::same(12)),
      )
      .show(ctx, |ui| {
        section_header(ui, "Curve Shape");
        ui.horizontal_wrapped(|ui| {
          ui.add_space(2.0);
          ui.add(
            Knob::new(
              "Tight-Curve Ceiling",
              &mut self.knobs.tight_curve_accel,
              clip::MIN_LAT,
              1.8,
            )
            .unit("m/s²")
            .precision(2),
          );
          ui.add(
            Knob::new(
              "Straight-Road Ceiling",
              &mut self.knobs.straight_road_accel,
              clip::MAX_LAT,
              4.478,
            )
            .unit("m/s²")
            .precision(2),
          );
        });
        ui.horizontal_wrapped(|ui| {
          ui.add_space(2.0);
          ui.add(
            Knob::new(
              "Transition Speed",
              &mut self.knobs.transition_speed_mph,
              clip::TRANSITION_MPH,
              48.0,
            )
            .unit("mph")
            .precision(1),
          );
          ui.add(
            Knob::new(
              "Sharpness",
              &mut self.knobs.sharpness,
              clip::SHARPNESS,
              8.0,
            )
            .unit("")
            .precision(1),
          );
        });

        ui.add_space(10.0);
        plain_hint(
          ui,
          "Drag a knob vertically. Shift-drag for fine tuning. \
           Double-click to restore default.",
        );

        ui.add_space(18.0);
        section_header(ui, "Local Shaping  (EQ bands)");
        bands_panel(
          ui,
          &mut self.tune.bands,
          &mut self.plot_state.selected_band,
          &mut self.status_msg,
        );

        if self.advanced_open {
          ui.add_space(18.0);
          section_header(ui, "Advanced  (raw parameters)");
          self.sync_params_from_knobs();
          advanced_panel(ui, &mut self.tune.params, &mut self.knobs);
        }
      });

    // --------------------------------------------------------- hero plot
    let baseline = SigmoidParams::default();
    CentralPanel::default()
      .frame(
        Frame::NONE
          .fill(theme::BG)
          .inner_margin(Margin::same(12)),
      )
      .show(ctx, |ui| {
        plot::hero_plot(
          ui,
          &mut self.knobs,
          &mut self.tune.bands,
          &baseline,
          &mut self.plot_state,
        );
      });

    // --------------------------------------------------------- apply modal
    self.show_apply_modal(ctx);

    // Snapshot for undo/redo after all UI ran for this frame.
    self.maybe_capture();

    // Continuous repaint while interacting so knobs / drags feel glued.
    ctx.request_repaint();
  }
}

// --------------------------------------------------------- helpers

fn section_header(ui: &mut egui::Ui, text: &str) {
  ui.label(
    RichText::new(text.to_uppercase())
      .color(theme::TEXT)
      .size(11.0)
      .strong(),
  );
  let rect = ui.available_rect_before_wrap();
  let y = rect.top() + 2.0;
  let stroke = Stroke::new(1.0, theme::SEPARATOR);
  ui.painter().line_segment(
    [pos2(rect.left(), y), pos2(rect.right(), y)],
    stroke,
  );
  ui.add_space(8.0);
}

fn plain_hint(ui: &mut egui::Ui, text: &str) {
  ui.label(
    RichText::new(text)
      .color(theme::TEXT_DIM)
      .size(11.0)
      .italics(),
  );
}

fn bands_panel(
  ui: &mut egui::Ui,
  bands: &mut Vec<Band>,
  selected: &mut Option<usize>,
  status: &mut StatusMessage,
) {
  if bands.is_empty() {
    plain_hint(
      ui,
      "Shift-click on the curve to add a band (or right-click the line for \
       a menu). Each band nudges a narrow speed range up or down without \
       changing the rest of the curve.",
    );
    return;
  }

  ScrollArea::vertical()
    .max_height(360.0)
    .auto_shrink([false; 2])
    .show(ui, |ui| {
      let mut remove_idx: Option<usize> = None;
      for idx in 0..bands.len() {
        let is_selected = *selected == Some(idx);
        let color = plot::band_color(idx);
        Frame::NONE
          .fill(if is_selected {
            theme::PANEL_HI
          } else {
            theme::PANEL
          })
          .inner_margin(Margin::same(8))
          .corner_radius(CornerRadius::same(5))
          .stroke(if is_selected {
            Stroke::new(1.5, color)
          } else {
            Stroke::new(1.0, theme::SEPARATOR)
          })
          .show(ui, |ui| {
            // Header — color swatch + name + select / enable / remove.
            ui.horizontal(|ui| {
              let (rect, _) = ui.allocate_exact_size(vec2(10.0, 16.0), egui::Sense::hover());
              ui.painter()
                .rect_filled(rect, CornerRadius::same(2), color);
              let label = format!("Band {}  @ {:.1} mph", idx + 1, bands[idx].center_speed_mph);
              if ui
                .selectable_label(is_selected, RichText::new(&label).color(theme::TEXT))
                .clicked()
              {
                *selected = if is_selected { None } else { Some(idx) };
              }
              ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.small_button("✕").on_hover_text("remove band").clicked() {
                  remove_idx = Some(idx);
                }
                ui.checkbox(&mut bands[idx].enabled, "")
                  .on_hover_text("bypass this band without deleting it");
              });
            });
            ui.add_space(4.0);
            // Three small editable knobs in a row.
            ui.horizontal_wrapped(|ui| {
              ui.add(
                Knob::new("Gain", &mut bands[idx].gain_db, (-12.0, 12.0), 0.0)
                  .unit("dB")
                  .precision(1)
                  .diameter(46.0)
                  .accent(color)
                  .editable_text(true),
              );
              ui.add(
                Knob::new("Width (Q)", &mut bands[idx].q, (0.3, 16.0), 1.5)
                  .precision(2)
                  .diameter(46.0)
                  .accent(color)
                  .editable_text(true)
                  .log(true),
              );
              ui.add(
                Knob::new(
                  "Center",
                  &mut bands[idx].center_speed_mph,
                  (5.0, 90.0),
                  45.0,
                )
                .unit("mph")
                .precision(1)
                .diameter(46.0)
                .accent(color)
                .editable_text(true),
              );
            });
          });
        ui.add_space(6.0);
      }
      if let Some(i) = remove_idx {
        bands.remove(i);
        if let Some(sel) = *selected {
          if sel == i {
            *selected = None;
          } else if sel > i {
            *selected = Some(sel - 1);
          }
        }
        status.set("Band removed.", theme::ACCENT);
      }
    });
}

fn advanced_panel(ui: &mut egui::Ui, params: &mut SigmoidParams, knobs: &mut PlainKnobs) {
  plain_hint(
    ui,
    "Editing raw parameters overrides the plain knobs above until you \
     touch a knob again.",
  );
  ui.add_space(6.0);
  let mut dirty = false;
  egui::Grid::new("advanced_grid")
    .num_columns(2)
    .spacing(vec2(10.0, 6.0))
    .show(ui, |ui| {
      dirty |= raw_row(ui, "A  (amplitude)", &mut params.a, -5.0..=-0.2);
      dirty |= raw_row(ui, "B  (steepness)", &mut params.b, -1.0e5..=-100.0);
      dirty |= raw_row(ui, "C  (centre κ)", &mut params.c, 1e-5..=0.1);
      dirty |= raw_row(ui, "D  (baseline)", &mut params.d, 2.0..=6.5);
      dirty |= raw_row(ui, "MIN  (floor)", &mut params.min_lat, 1.0..=3.0);
      dirty |= raw_row(ui, "MAX  (ceiling)", &mut params.max_lat, 2.0..=5.5);
    });
  if dirty {
    *knobs = PlainKnobs::from_sigmoid(params);
  }
}

fn raw_row(
  ui: &mut egui::Ui,
  label: &str,
  value: &mut f64,
  range: std::ops::RangeInclusive<f64>,
) -> bool {
  ui.label(RichText::new(label).monospace().size(11.0));
  let resp = ui.add(
    egui::DragValue::new(value)
      .range(range)
      .speed(0.001)
      .fixed_decimals(5),
  );
  ui.end_row();
  resp.changed()
}

