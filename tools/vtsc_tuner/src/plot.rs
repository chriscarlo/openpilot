//! Hero graph — custom-painted (speed mph, max lat accel m/s²) sigmoid view.
//!
//! Responsibilities:
//!   * render axes, grid, sigmoid, clamp rails, EQ bands
//!   * handle drag of clamp rails, inflection handle, band markers
//!   * expose hover readout and shift-click-to-add-band

use eframe::egui::{
  self, Align2, CornerRadius, FontId, Pos2, Rect, Response, Sense, Stroke, Ui, pos2, vec2,
};

use crate::params::{PlainKnobs, clip};
use crate::sigmoid::{Band, SigmoidParams, apply_prepared_bands, prepare_bands};
use crate::theme;

pub const X_RANGE_MPH: (f64, f64) = (0.0, 90.0);
pub const Y_RANGE_ACCEL: (f64, f64) = (0.0, 6.0);

pub struct PlotState {
  pub hover_readout: Option<(f64, f64)>,
  pub selected_band: Option<usize>,
  /// Selected built-in handle (rails / inflection / wings). Mutually exclusive
  /// with `selected_band` so a click on one always deselects the other.
  pub selected_handle: Option<HandleId>,
  /// Tracks which drag target (if any) is currently active so a new frame
  /// doesn't accidentally grab a neighbour.
  pub active_drag: DragTarget,
  /// Right-click context menu anchored on the curve line, if open.
  pub line_menu: Option<LineMenuState>,
}

impl Default for PlotState {
  fn default() -> Self {
    Self {
      hover_readout: None,
      selected_band: None,
      selected_handle: None,
      active_drag: DragTarget::None,
      line_menu: None,
    }
  }
}

/// Stable identity for the built-in (non-band) handles on the plot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleId {
  MinRail,
  MaxRail,
  Inflection,
  LeftWing,
  RightWing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HighlightLevel {
  None,
  Hover,
  Selected,
}

#[derive(Debug, Clone, Copy)]
pub struct LineMenuState {
  /// Screen-space anchor where the menu's top-left corner sits.
  pub screen_pos: Pos2,
  /// Data-space coordinates at the right-click point.
  pub mph: f64,
  pub accel: f64,
}

#[derive(Debug, Clone, Copy)]
enum LineMenuAction {
  AddBand,
  CopyValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragTarget {
  None,
  MinRail,
  MaxRail,
  Inflection,
  /// Steepness "wing" anchors flanking the inflection. Both sides drive the
  /// same `sharpness` value, but we track which one was grabbed so the
  /// hover/selected highlight paints the correct chevron.
  LeftWing,
  RightWing,
  Band(usize),
}

/// Linear map between (mph, m/s²) data and pixel space.
struct Mapper {
  rect: Rect,
}

impl Mapper {
  fn new(rect: Rect) -> Self {
    Self { rect }
  }
  fn w(&self) -> f32 {
    self.rect.width().max(1.0)
  }
  fn h(&self) -> f32 {
    self.rect.height().max(1.0)
  }
  fn x_to_px(&self, mph: f64) -> f32 {
    let (lo, hi) = X_RANGE_MPH;
    let t = ((mph - lo) / (hi - lo)) as f32;
    self.rect.left() + t * self.w()
  }
  fn y_to_px(&self, accel: f64) -> f32 {
    let (lo, hi) = Y_RANGE_ACCEL;
    let t = ((accel - lo) / (hi - lo)) as f32;
    self.rect.bottom() - t * self.h()
  }
  fn px_to_x(&self, px: f32) -> f64 {
    let (lo, hi) = X_RANGE_MPH;
    let t_raw = (px - self.rect.left()) / self.w();
    let t = if t_raw.is_finite() { t_raw.clamp(0.0, 1.0) } else { 0.0 };
    lo + (hi - lo) * (t as f64)
  }
  fn px_to_y(&self, py: f32) -> f64 {
    let (lo, hi) = Y_RANGE_ACCEL;
    let t_raw = (self.rect.bottom() - py) / self.h();
    let t = if t_raw.is_finite() { t_raw.clamp(0.0, 1.0) } else { 0.0 };
    lo + (hi - lo) * (t as f64)
  }
  fn data_to_pos(&self, mph: f64, accel: f64) -> Pos2 {
    pos2(self.x_to_px(mph), self.y_to_px(accel))
  }
}

pub fn hero_plot(
  ui: &mut Ui,
  knobs: &mut PlainKnobs,
  bands: &mut Vec<Band>,
  baseline: &SigmoidParams,
  state: &mut PlotState,
) -> Response {
  // Claim space via egui's normal layout path (more robust to resize races
  // than allocating an explicit rect). Fall back to a tiny placeholder
  // response if there's effectively no room — prevents degenerate geometry
  // from reaching the painter during resize frames.
  let available = ui.available_size_before_wrap();
  let size = egui::vec2(available.x.max(1.0), available.y.max(1.0));
  let (outer_rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
  if outer_rect.width() < 100.0 || outer_rect.height() < 80.0 {
    return response;
  }

  let axis_gutter_left = 48.0_f32;
  let axis_gutter_bottom = 28.0_f32;
  let plot_rect = Rect::from_min_max(
    pos2(outer_rect.left() + axis_gutter_left, outer_rect.top() + 12.0),
    pos2(
      outer_rect.right() - 12.0,
      outer_rect.bottom() - axis_gutter_bottom,
    ),
  );
  if plot_rect.width() < 50.0 || plot_rect.height() < 50.0 {
    return response;
  }

  let painter = ui.painter_at(outer_rect);
  let map = Mapper::new(plot_rect);

  // ---------------------------------------------------------- background
  painter.rect_filled(plot_rect, CornerRadius::same(6), theme::PANEL);
  painter.rect_stroke(
    plot_rect,
    CornerRadius::same(6),
    Stroke::new(1.0, theme::SEPARATOR),
    egui::StrokeKind::Outside,
  );

  draw_grid(&painter, &map);
  draw_axes(&painter, &map, plot_rect, outer_rect);

  // ---------------------------------------------------------- baseline overlay
  // Draw the repo-default sigmoid behind the live curve as a faint dashed
  // line so the user can see how far they've strayed at a glance.
  let baseline_samples = crate::sigmoid::sample_curve(baseline, &[], 256, 1e-5, 1.0);
  let baseline_pts: Vec<Pos2> = baseline_samples
    .iter()
    .filter(|s| s.speed_mph >= X_RANGE_MPH.0 - 2.0 && s.speed_mph <= X_RANGE_MPH.1 + 2.0)
    .map(|s| map.data_to_pos(s.speed_mph, s.a_lat))
    .collect();
  draw_dashed_polyline(
    &painter,
    &baseline_pts,
    7.0,
    5.0,
    Stroke::new(1.5, theme::TEXT_MUTED.gamma_multiply(0.55)),
  );

  // ---------------------------------------------------------- live sigmoid
  let params = knobs.to_sigmoid();
  let mut samples = crate::sigmoid::sample_curve(&params, bands, 512, 1e-5, 1.0);
  // The parametric (v(κ), a(κ)) trace becomes multi-valued in v whenever a
  // band's local slope exceeds a/κ — the curve folds back on itself. The
  // runtime never sees that (it always maps κ → a → v in one direction), so
  // we render the visualisation as the monotonic envelope: any speed that
  // would step backwards is held at the previous max-v, which causes a
  // sharp band to render as a vertical notch (the parametric-EQ idiom).
  let mut max_v = f64::NEG_INFINITY;
  for s in samples.iter_mut() {
    if s.speed_mph < max_v {
      s.speed_mph = max_v;
    } else {
      max_v = s.speed_mph;
    }
  }
  let curve_pts: Vec<Pos2> = samples
    .iter()
    .filter(|s| s.speed_mph >= X_RANGE_MPH.0 - 2.0 && s.speed_mph <= X_RANGE_MPH.1 + 2.0)
    .map(|s| map.data_to_pos(s.speed_mph, s.a_lat))
    .collect();
  if curve_pts.len() >= 2 {
    // soft glow behind the curve
    painter.add(egui::Shape::line(
      curve_pts.clone(),
      Stroke::new(6.0, theme::ACCENT.gamma_multiply(0.18)),
    ));
    painter.add(egui::Shape::line(
      curve_pts.clone(),
      Stroke::new(2.5, theme::ACCENT),
    ));
  }

  // ---------------------------------------------------------- interaction
  let pointer = response.hover_pos();
  let drag_start = response.drag_started();
  let dragging = response.dragged();
  let released = response.drag_stopped();
  let shift = ui.input(|i| i.modifiers.shift);
  let clicked = response.clicked();
  let secondary_clicked = response.secondary_clicked();

  let hit_tolerance = 10.0_f32;

  // What is the pointer over right now? Reuses pick_drag_target so the hover
  // hit zones match the drag hit zones exactly (no surprises).
  let hovered_target = pointer
    .map(|p| pick_drag_target(p, &params, knobs, bands, &map, hit_tolerance))
    .unwrap_or(DragTarget::None);
  let hovered_handle = drag_target_to_handle(hovered_target);
  let hovered_band = match hovered_target {
    DragTarget::Band(i) => Some(i),
    _ => None,
  };

  // Grab/grabbing cursor advertises the affordance over any handle.
  if !matches!(hovered_target, DragTarget::None) {
    let icon = if matches!(state.active_drag, DragTarget::None) {
      egui::CursorIcon::Grab
    } else {
      egui::CursorIcon::Grabbing
    };
    ui.ctx().set_cursor_icon(icon);
  }

  // ---------------------------------------------------------- clamp rails
  draw_rail(
    &painter,
    &map,
    params.min_lat,
    "Tight-Curve Ceiling",
    level_for_handle(state, hovered_handle, HandleId::MinRail),
  );
  draw_rail(
    &painter,
    &map,
    params.max_lat,
    "Straight-Road Ceiling",
    level_for_handle(state, hovered_handle, HandleId::MaxRail),
  );

  // Decide which target this drag is for on drag_started, and follow the
  // selection so highlight tracks the actively-dragged handle.
  if drag_start {
    if let Some(p) = pointer {
      let target = pick_drag_target(p, &params, knobs, bands, &map, hit_tolerance);
      state.active_drag = target;
      apply_selection(state, target);
    }
  }
  if released {
    state.active_drag = DragTarget::None;
  }

  // Apply drag deltas.
  if dragging {
    if let Some(p) = pointer {
      let new_y = map.px_to_y(p.y).clamp(Y_RANGE_ACCEL.0, Y_RANGE_ACCEL.1);
      let new_x = map.px_to_x(p.x).clamp(X_RANGE_MPH.0, X_RANGE_MPH.1);
      match state.active_drag {
        DragTarget::MinRail => {
          knobs.tight_curve_accel = new_y.clamp(clip::MIN_LAT.0, clip::MIN_LAT.1);
          if knobs.straight_road_accel < knobs.tight_curve_accel + clip::ABS_A.0 {
            knobs.straight_road_accel = knobs.tight_curve_accel + clip::ABS_A.0;
          }
        }
        DragTarget::MaxRail => {
          knobs.straight_road_accel = new_y.clamp(clip::MAX_LAT.0, clip::MAX_LAT.1);
          if knobs.tight_curve_accel > knobs.straight_road_accel - clip::ABS_A.0 {
            knobs.tight_curve_accel = knobs.straight_road_accel - clip::ABS_A.0;
          }
        }
        DragTarget::Inflection => {
          knobs.transition_speed_mph = new_x.clamp(clip::TRANSITION_MPH.0, clip::TRANSITION_MPH.1);
        }
        DragTarget::LeftWing | DragTarget::RightWing => {
          // Distance from inflection encodes sharpness: wider = gentler.
          let half_width = (new_x - knobs.transition_speed_mph).abs().max(0.5);
          knobs.sharpness = wing_half_width_to_sharpness(half_width)
            .clamp(clip::SHARPNESS.0, clip::SHARPNESS.1);
        }
        DragTarget::Band(idx) => {
          if let Some(b) = bands.get_mut(idx) {
            b.center_speed_mph = new_x.clamp(X_RANGE_MPH.0 + 1.0, X_RANGE_MPH.1 - 1.0);
            // Vertical drag sets gain_db so the band passes through the pointer's a_lat.
            let prepped = prepare_bands(&knobs.to_sigmoid(), &[Band {
              gain_db: 0.0,
              ..*b
            }]);
            if let Some(prep) = prepped.first() {
              let kappa = 10f64.powf(prep.log_kc);
              let base = knobs.to_sigmoid().eval(kappa);
              if base > 1e-6 {
                let target = new_y;
                let ratio = (target / base).max(1e-3);
                b.gain_db = (20.0 * ratio.log10()).clamp(-12.0, 12.0);
              }
            }
          }
        }
        DragTarget::None => {}
      }
    }
  }

  // Shift-click → add a band at the pointer; select it.
  if clicked && shift {
    if let Some(p) = pointer {
      if plot_rect.contains(p) {
        push_band_at(bands, map.px_to_x(p.x));
        state.selected_band = Some(bands.len() - 1);
        state.selected_handle = None;
      }
    }
  } else if clicked {
    // Plain click selects whichever handle / band the pointer is over;
    // a click on empty plot space deselects everything.
    if let Some(p) = pointer {
      let target = pick_drag_target(p, &params, knobs, bands, &map, hit_tolerance);
      if matches!(target, DragTarget::None) {
        if plot_rect.contains(p) {
          state.selected_band = None;
          state.selected_handle = None;
        }
      } else {
        apply_selection(state, target);
      }
    }
  }

  // Right-click ON the curve line opens a context menu. Right-click anywhere
  // else inside the plot dismisses an open menu (clicks outside the plot are
  // ignored — egui's `secondary_clicked()` only fires inside the response).
  let mut menu_just_opened = false;
  if secondary_clicked {
    if let Some(p) = pointer {
      if plot_rect.contains(p) && hit_polyline(p, &curve_pts, 8.0) {
        state.line_menu = Some(LineMenuState {
          screen_pos: p,
          mph: map.px_to_x(p.x),
          accel: map.px_to_y(p.y),
        });
        menu_just_opened = true;
      } else if state.line_menu.is_some() {
        state.line_menu = None;
      }
    }
  }

  // Escape clears the active selection — but only if no line context menu is
  // open, since the menu's render block (run later) wants Escape for itself.
  // Without this short-circuit, our consume_key here would steal the press
  // and the menu would refuse to close on the first Escape.
  if state.line_menu.is_none()
    && (state.selected_handle.is_some() || state.selected_band.is_some())
    && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
  {
    state.selected_handle = None;
    state.selected_band = None;
  }

  // ---------------------------------------------------------- band Q zone
  // Translucent zone painted UNDER the curve — only for the selected band
  // to keep the graph readable when many bands exist.
  let params = knobs.to_sigmoid();
  let prepped = prepare_bands(&params, bands);
  if let Some(sel_idx) = state.selected_band {
    if let Some(band) = bands.get(sel_idx) {
      if band.enabled {
        if let Some(prep) = prepped.get(prepped_index(bands, sel_idx)) {
          let color = band_color(sel_idx);
          draw_band_zone(&painter, &map, plot_rect, &params, prep, color);
        }
      }
    }
  }

  // ---------------------------------------------------------- band markers
  for (idx, band) in bands.iter().enumerate() {
    if !band.enabled {
      continue;
    }
    let kappa = if let Some(prep) = prepped.get(prepped_index(bands, idx)) {
      10f64.powf(prep.log_kc)
    } else {
      continue;
    };
    let base = params.eval(kappa);
    let composed = apply_prepared_bands(kappa, base, &prepped);
    let p = map.data_to_pos(band.center_speed_mph, composed);
    let level = if state.selected_band == Some(idx) {
      HighlightLevel::Selected
    } else if hovered_band == Some(idx) {
      HighlightLevel::Hover
    } else {
      HighlightLevel::None
    };
    let r = if level == HighlightLevel::Selected { 7.0 } else { 5.0 };
    let color = band_color(idx);
    painter.circle_filled(p, r, color);
    painter.circle_stroke(p, r + 1.0, Stroke::new(1.0, theme::BG));
    match level {
      HighlightLevel::Hover => {
        painter.circle_stroke(p, r + 3.0, Stroke::new(1.0, color.gamma_multiply(0.55)));
      }
      HighlightLevel::Selected => {
        painter.circle_stroke(p, r + 4.0, Stroke::new(1.5, color));
      }
      HighlightLevel::None => {}
    }
  }

  // ---------------------------------------------------------- inflection + steep wings
  let mid_y = 0.5 * (knobs.tight_curve_accel + knobs.straight_road_accel);
  let inflect_pos = map.data_to_pos(knobs.transition_speed_mph, mid_y);
  let wing_half = sharpness_to_wing_half_width(knobs.sharpness);
  let left_wing_x = (knobs.transition_speed_mph - wing_half).max(X_RANGE_MPH.0 + 1.0);
  let right_wing_x = (knobs.transition_speed_mph + wing_half).min(X_RANGE_MPH.1 - 1.0);
  let left_wing = map.data_to_pos(left_wing_x, mid_y);
  let right_wing = map.data_to_pos(right_wing_x, mid_y);
  let inflect_level = level_for_handle(state, hovered_handle, HandleId::Inflection);
  let left_wing_level = level_for_handle(state, hovered_handle, HandleId::LeftWing);
  let right_wing_level = level_for_handle(state, hovered_handle, HandleId::RightWing);
  // horizontal wing bar
  painter.line_segment(
    [left_wing, right_wing],
    Stroke::new(2.0, theme::ACCENT.gamma_multiply(0.6)),
  );
  // wing handles (chevrons) — glow ring scales with hover/selected level
  draw_wing_handle(&painter, left_wing, true, left_wing_level);
  draw_wing_handle(&painter, right_wing, false, right_wing_level);
  // inflection circle
  match inflect_level {
    HighlightLevel::Hover => {
      painter.circle_stroke(
        inflect_pos,
        11.0,
        Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.45)),
      );
    }
    HighlightLevel::Selected => {
      painter.circle_stroke(inflect_pos, 11.0, Stroke::new(1.5, theme::ACCENT));
    }
    HighlightLevel::None => {}
  }
  painter.circle_filled(inflect_pos, 6.0, theme::TEXT);
  painter.circle_stroke(inflect_pos, 7.0, Stroke::new(1.5, theme::ACCENT));
  // vertical guide at transition speed
  painter.line_segment(
    [
      pos2(inflect_pos.x, plot_rect.top()),
      pos2(inflect_pos.x, plot_rect.bottom()),
    ],
    Stroke::new(1.0, theme::SEPARATOR),
  );

  // ---------------------------------------------------------- hover readout
  if let Some(p) = pointer {
    if plot_rect.contains(p) {
      let speed = map.px_to_x(p.x);
      let accel = map.px_to_y(p.y);
      state.hover_readout = Some((speed, accel));
      painter.line_segment(
        [pos2(p.x, plot_rect.top()), pos2(p.x, plot_rect.bottom())],
        Stroke::new(1.0, theme::TEXT_DIM),
      );
      painter.line_segment(
        [pos2(plot_rect.left(), p.y), pos2(plot_rect.right(), p.y)],
        Stroke::new(1.0, theme::TEXT_DIM),
      );
      let label = format!("{:>5.1} mph   •   {:>4.2} m/s²", speed, accel);
      let text_pos = pos2(plot_rect.right() - 8.0, plot_rect.top() + 8.0);
      painter.rect_filled(
        Rect::from_two_pos(text_pos, pos2(text_pos.x - 170.0, text_pos.y + 22.0)),
        CornerRadius::same(4),
        theme::BG.gamma_multiply(0.9),
      );
      painter.text(
        text_pos - vec2(4.0, -4.0),
        Align2::RIGHT_TOP,
        label,
        FontId::monospace(12.0),
        theme::TEXT,
      );
    } else {
      state.hover_readout = None;
    }
  } else {
    state.hover_readout = None;
  }

  // Shift-click hint while holding shift over the plot
  if shift {
    if let Some(p) = pointer {
      if plot_rect.contains(p) {
        painter.text(
          pos2(p.x + 12.0, p.y - 4.0),
          Align2::LEFT_BOTTOM,
          "+ add band",
          FontId::proportional(11.0),
          theme::ACCENT_WARM,
        );
      }
    }
  }

  // ---------------------------------------------------------- line context menu
  // Lives in its own foreground Area so it floats above panels and grid.
  if let Some(menu) = state.line_menu {
    let mut chosen: Option<LineMenuAction> = None;
    let area_resp = egui::Area::new(egui::Id::new("vtsc_line_context_menu"))
      .order(egui::Order::Foreground)
      .fixed_pos(menu.screen_pos + vec2(2.0, 2.0))
      .show(ui.ctx(), |ui| {
        egui::Frame::popup(ui.style())
          .inner_margin(egui::Margin::symmetric(4, 4))
          .show(ui, |ui| {
            ui.set_min_width(240.0);
            let add = ui.add_sized(
              vec2(240.0, 24.0),
              egui::Button::new(
                egui::RichText::new("✚  Add anchor point here").size(12.0),
              )
              .frame(false),
            );
            if add.clicked() {
              chosen = Some(LineMenuAction::AddBand);
            }
            let copy_label = format!(
              "⎘  Copy “{:.1} mph,  {:.2} m/s²”",
              menu.mph, menu.accel
            );
            let copy = ui.add_sized(
              vec2(240.0, 24.0),
              egui::Button::new(egui::RichText::new(copy_label).size(12.0)).frame(false),
            );
            if copy.clicked() {
              chosen = Some(LineMenuAction::CopyValues);
            }
          });
      });

    let menu_rect = area_resp.response.rect;
    // consume_key so a plot-level Escape handler doesn't also fire on the
    // same press and clear the selection underneath.
    let escape = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
    let primary_clicked_now = ui.input(|i| i.pointer.primary_clicked());
    let click_pos = ui.input(|i| i.pointer.interact_pos());
    let outside_click = primary_clicked_now
      && !menu_just_opened
      && click_pos.map(|p| !menu_rect.contains(p)).unwrap_or(false);

    if let Some(action) = chosen {
      match action {
        LineMenuAction::AddBand => {
          push_band_at(bands, menu.mph);
          state.selected_band = Some(bands.len() - 1);
          state.selected_handle = None;
        }
        LineMenuAction::CopyValues => {
          ui.ctx()
            .copy_text(format!("{:.1} mph, {:.2} m/s²", menu.mph, menu.accel));
        }
      }
    }

    if chosen.is_some() || escape || outside_click {
      state.line_menu = None;
    }
  }

  response
}

/// Append a new band at `mph` (0-dB, default Q, enabled). Shared by shift-click
/// and the right-click context menu so the two paths cannot drift.
fn push_band_at(bands: &mut Vec<Band>, mph: f64) {
  bands.push(Band {
    center_speed_mph: mph.clamp(X_RANGE_MPH.0 + 1.0, X_RANGE_MPH.1 - 1.0),
    gain_db: 0.0,
    q: 1.5,
    enabled: true,
  });
}

/// Distance-to-polyline hit test. Returns true if `p` is within `tol` pixels
/// of any segment of `pts`. Used to scope the right-click menu to the curve
/// itself rather than the entire plot field.
fn hit_polyline(p: Pos2, pts: &[Pos2], tol: f32) -> bool {
  if pts.len() < 2 {
    return false;
  }
  let tol2 = tol * tol;
  for w in pts.windows(2) {
    if dist_sq_point_to_segment(p, w[0], w[1]) <= tol2 {
      return true;
    }
  }
  false
}

fn dist_sq_point_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
  let ab = b - a;
  let ap = p - a;
  let len2 = ab.length_sq();
  if len2 < 1e-6 {
    return ap.length_sq();
  }
  let t = (ap.dot(ab) / len2).clamp(0.0, 1.0);
  let proj = a + ab * t;
  (p - proj).length_sq()
}

fn prepared_count_before(bands: &[Band], idx: usize) -> usize {
  bands
    .iter()
    .take(idx)
    .filter(|b| b.enabled)
    .count()
}
fn prepped_index(bands: &[Band], idx: usize) -> usize {
  prepared_count_before(bands, idx)
}

/// Convert a drag target into the corresponding built-in HandleId, if any.
/// Returns None for `Band(_)` and `None` since those don't represent built-in
/// handles.
fn drag_target_to_handle(t: DragTarget) -> Option<HandleId> {
  match t {
    DragTarget::MinRail => Some(HandleId::MinRail),
    DragTarget::MaxRail => Some(HandleId::MaxRail),
    DragTarget::Inflection => Some(HandleId::Inflection),
    DragTarget::LeftWing => Some(HandleId::LeftWing),
    DragTarget::RightWing => Some(HandleId::RightWing),
    DragTarget::Band(_) | DragTarget::None => None,
  }
}

/// Resolve the highlight level for a built-in handle: Selected wins over
/// Hover wins over None. Centralised so every handle's draw call agrees.
fn level_for_handle(
  state: &PlotState,
  hovered: Option<HandleId>,
  id: HandleId,
) -> HighlightLevel {
  if state.selected_handle == Some(id) {
    HighlightLevel::Selected
  } else if hovered == Some(id) {
    HighlightLevel::Hover
  } else {
    HighlightLevel::None
  }
}

/// Set the selected handle/band based on a drag target, enforcing mutual
/// exclusion (selecting one always clears the other).
fn apply_selection(state: &mut PlotState, target: DragTarget) {
  match target {
    DragTarget::Band(i) => {
      state.selected_band = Some(i);
      state.selected_handle = None;
    }
    DragTarget::None => {}
    other => {
      if let Some(h) = drag_target_to_handle(other) {
        state.selected_handle = Some(h);
        state.selected_band = None;
      }
    }
  }
}

fn pick_drag_target(
  p: Pos2,
  params: &SigmoidParams,
  knobs: &PlainKnobs,
  bands: &[Band],
  map: &Mapper,
  tol: f32,
) -> DragTarget {
  let mid_y = 0.5 * (knobs.tight_curve_accel + knobs.straight_road_accel);
  let inflect_pos = map.data_to_pos(knobs.transition_speed_mph, mid_y);
  // Inflection wins over wings when targets overlap.
  if (p - inflect_pos).length() <= tol + 2.0 {
    return DragTarget::Inflection;
  }
  // Steepness wings.
  let wing_half = sharpness_to_wing_half_width(knobs.sharpness);
  for (sign, target) in &[(-1.0_f64, DragTarget::LeftWing), (1.0_f64, DragTarget::RightWing)] {
    let wx = (knobs.transition_speed_mph + sign * wing_half)
      .clamp(X_RANGE_MPH.0 + 1.0, X_RANGE_MPH.1 - 1.0);
    let wp = map.data_to_pos(wx, mid_y);
    if (p - wp).length() <= tol + 4.0 {
      return *target;
    }
  }
  // Band dots.
  let prepped = prepare_bands(params, bands);
  for (idx, band) in bands.iter().enumerate() {
    if !band.enabled {
      continue;
    }
    let Some(prep) = prepped.get(prepped_index(bands, idx)) else {
      continue;
    };
    let kappa = 10f64.powf(prep.log_kc);
    let base = params.eval(kappa);
    let composed = apply_prepared_bands(kappa, base, &prepped);
    let band_pos = map.data_to_pos(band.center_speed_mph, composed);
    if (p - band_pos).length() <= tol + 2.0 {
      return DragTarget::Band(idx);
    }
  }
  // Clamp rails — horizontal bands across the full plot width.
  let min_y_px = map.y_to_px(params.min_lat);
  let max_y_px = map.y_to_px(params.max_lat);
  if (p.y - min_y_px).abs() < tol {
    return DragTarget::MinRail;
  }
  if (p.y - max_y_px).abs() < tol {
    return DragTarget::MaxRail;
  }
  DragTarget::None
}

fn draw_grid(painter: &egui::Painter, map: &Mapper) {
  // Minor grid: every 5 mph and 0.5 m/s².
  for mph in (0..=90).step_by(5) {
    let x = map.x_to_px(mph as f64);
    let major = mph % 10 == 0;
    painter.line_segment(
      [pos2(x, map.rect.top()), pos2(x, map.rect.bottom())],
      Stroke::new(
        if major { 1.0 } else { 0.5 },
        if major {
          theme::GRID_MAJOR
        } else {
          theme::GRID
        },
      ),
    );
  }
  let mut a = 0.0_f64;
  while a <= Y_RANGE_ACCEL.1 + 1e-6 {
    let y = map.y_to_px(a);
    let major = ((a * 2.0).round() as i32) % 2 == 0; // every 1.0
    painter.line_segment(
      [pos2(map.rect.left(), y), pos2(map.rect.right(), y)],
      Stroke::new(
        if major { 1.0 } else { 0.5 },
        if major {
          theme::GRID_MAJOR
        } else {
          theme::GRID
        },
      ),
    );
    a += 0.5;
  }
}

fn draw_axes(painter: &egui::Painter, map: &Mapper, plot: Rect, _outer: Rect) {
  // X tick labels every 10 mph.
  for mph in (0..=90).step_by(10) {
    let x = map.x_to_px(mph as f64);
    painter.text(
      pos2(x, plot.bottom() + 4.0),
      Align2::CENTER_TOP,
      format!("{mph}"),
      FontId::monospace(11.0),
      theme::TEXT_MUTED,
    );
  }
  // Y tick labels every 1.0 m/s².
  let mut a = 0.0_f64;
  while a <= Y_RANGE_ACCEL.1 + 1e-6 {
    let y = map.y_to_px(a);
    painter.text(
      pos2(plot.left() - 8.0, y),
      Align2::RIGHT_CENTER,
      format!("{a:.0}"),
      FontId::monospace(11.0),
      theme::TEXT_MUTED,
    );
    a += 1.0;
  }
  // axis titles
  painter.text(
    pos2(plot.center().x, plot.bottom() + 20.0),
    Align2::CENTER_CENTER,
    "Speed  (mph)",
    FontId::proportional(11.0),
    theme::TEXT_MUTED,
  );
  painter.text(
    pos2(plot.left() - 38.0, plot.center().y),
    Align2::CENTER_CENTER,
    "m/s²",
    FontId::proportional(11.0),
    theme::TEXT_MUTED,
  );
}

fn draw_rail(
  painter: &egui::Painter,
  map: &Mapper,
  accel: f64,
  label: &str,
  highlight: HighlightLevel,
) {
  let y = map.y_to_px(accel);
  let left = map.rect.left();
  let right = map.rect.right();
  // Dashed line — draw short segments every 8 px.
  let dash_len = 7.0_f32;
  let gap = 5.0_f32;
  let mut x = left;
  while x < right {
    let x1 = (x + dash_len).min(right);
    painter.line_segment(
      [pos2(x, y), pos2(x1, y)],
      Stroke::new(1.0, theme::RAIL.gamma_multiply(0.85)),
    );
    x = x1 + gap;
  }
  painter.text(
    pos2(right - 6.0, y - 3.0),
    Align2::RIGHT_BOTTOM,
    format!("{label}  {accel:.2}"),
    FontId::proportional(11.0,),
    theme::RAIL.gamma_multiply(0.95),
  );
  // Prominent drag handle at the left edge — pill shape with vertical
  // double-arrow so the affordance is unmissable.
  let handle_center = pos2(left + 12.0, y);
  let pill = Rect::from_center_size(handle_center, vec2(18.0, 14.0));
  // Hover/selected glow — paint UNDER the pill so the pill renders crisply.
  match highlight {
    HighlightLevel::Hover => {
      let glow = pill.expand(3.0);
      painter.rect_filled(
        glow,
        CornerRadius::same(10),
        theme::RAIL.gamma_multiply(0.22),
      );
    }
    HighlightLevel::Selected => {
      let glow = pill.expand(4.0);
      painter.rect_filled(
        glow,
        CornerRadius::same(11),
        theme::RAIL.gamma_multiply(0.32),
      );
      painter.rect_stroke(
        glow,
        CornerRadius::same(11),
        Stroke::new(1.5, theme::RAIL),
        egui::StrokeKind::Outside,
      );
    }
    HighlightLevel::None => {}
  }
  painter.rect_filled(pill, CornerRadius::same(7), theme::RAIL);
  painter.rect_stroke(
    pill,
    CornerRadius::same(7),
    Stroke::new(1.0, theme::BG),
    egui::StrokeKind::Outside,
  );
  // Up/down arrow glyphs inside the pill
  let cx = handle_center.x;
  let arrow_color = theme::BG;
  // up arrow
  painter.add(egui::Shape::convex_polygon(
    vec![
      pos2(cx, y - 5.0),
      pos2(cx - 3.0, y - 1.5),
      pos2(cx + 3.0, y - 1.5),
    ],
    arrow_color,
    Stroke::NONE,
  ));
  // down arrow
  painter.add(egui::Shape::convex_polygon(
    vec![
      pos2(cx, y + 5.0),
      pos2(cx - 3.0, y + 1.5),
      pos2(cx + 3.0, y + 1.5),
    ],
    arrow_color,
    Stroke::NONE,
  ));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn draw_dashed_polyline(
  painter: &egui::Painter,
  pts: &[Pos2],
  dash: f32,
  gap: f32,
  stroke: Stroke,
) {
  if pts.len() < 2 {
    return;
  }
  let mut leftover = 0.0_f32; // negative = pen up, positive = pen down
  let mut pen_down = true;
  for w in pts.windows(2) {
    let a = w[0];
    let b = w[1];
    let v = b - a;
    let seg_len = v.length();
    if seg_len < 1e-3 {
      continue;
    }
    let dir = v / seg_len;
    let mut t = 0.0_f32;
    while t < seg_len {
      let span = if pen_down { dash } else { gap } - leftover;
      let next_t = (t + span).min(seg_len);
      if pen_down {
        painter.line_segment([a + dir * t, a + dir * next_t], stroke);
      }
      if next_t == t + span {
        pen_down = !pen_down;
        leftover = 0.0;
      } else {
        leftover = next_t - t;
      }
      t = next_t;
    }
  }
}

/// Stable per-band color so the marker, panel header, and Q zone all match.
pub fn band_color(idx: usize) -> egui::Color32 {
  // Curated palette tuned for the dark theme — distinct hues, similar
  // perceived brightness so no band visually dominates the others.
  const PALETTE: &[egui::Color32] = &[
    egui::Color32::from_rgb(0xf0, 0xa0, 0x4b), // amber
    egui::Color32::from_rgb(0x6b, 0xa6, 0xff), // cornflower
    egui::Color32::from_rgb(0xf0, 0x70, 0x90), // rose
    egui::Color32::from_rgb(0x9c, 0xd6, 0x5b), // lime
    egui::Color32::from_rgb(0xc0, 0x7a, 0xf0), // violet
    egui::Color32::from_rgb(0xf0, 0xd6, 0x4b), // gold
    egui::Color32::from_rgb(0x4b, 0xd6, 0xc1), // teal (matches accent — last resort)
  ];
  PALETTE[idx % PALETTE.len()]
}

pub fn sharpness_to_wing_half_width(sharpness: f64) -> f64 {
  // Empirical mapping: full half-width 32 mph at sharpness 0,
  // ~3 mph at sharpness 10.  Smooth exp decay → linear-feeling drag.
  30.0 * (-0.4 * sharpness).exp() + 2.0
}

pub fn wing_half_width_to_sharpness(half_mph: f64) -> f64 {
  // Inverse of the above; clamps the log argument so the user can't
  // produce a NaN by dragging the wing inside its lower bound.
  let arg = ((half_mph - 2.0).max(1e-3) / 30.0).max(1e-6);
  -arg.ln() / 0.4
}

fn draw_wing_handle(
  painter: &egui::Painter,
  p: Pos2,
  left_facing: bool,
  highlight: HighlightLevel,
) {
  // Highlight ring sits behind the chevron so the arrow always reads cleanly.
  match highlight {
    HighlightLevel::Hover => {
      painter.circle_filled(p, 9.0, theme::ACCENT.gamma_multiply(0.20));
    }
    HighlightLevel::Selected => {
      painter.circle_filled(p, 11.0, theme::ACCENT.gamma_multiply(0.28));
      painter.circle_stroke(p, 11.0, Stroke::new(1.5, theme::ACCENT));
    }
    HighlightLevel::None => {}
  }
  let arrow = if left_facing {
    [pos2(p.x - 6.0, p.y), pos2(p.x + 2.0, p.y - 5.0), pos2(p.x + 2.0, p.y + 5.0)]
  } else {
    [pos2(p.x + 6.0, p.y), pos2(p.x - 2.0, p.y - 5.0), pos2(p.x - 2.0, p.y + 5.0)]
  };
  painter.add(egui::Shape::convex_polygon(
    arrow.to_vec(),
    theme::ACCENT,
    Stroke::new(1.0, theme::BG),
  ));
}

/// Translucent vertical "bell" zone for a band — wider Q narrows the zone.
fn draw_band_zone(
  painter: &egui::Painter,
  map: &Mapper,
  plot: Rect,
  params: &SigmoidParams,
  prep: &crate::sigmoid::PreparedBand,
  color: egui::Color32,
) {
  // Convert the prepared band's log-κ centre + sigma into mph half-widths.
  // We sample the bell across log-κ, but visualise as a vertical translucent
  // slab spanning ±2σ around the centre in mph space (covers ~95% of the
  // bell).
  let kc = 10f64.powf(prep.log_kc);
  let center_a = params.eval(kc);
  if center_a < 1e-6 {
    return;
  }
  let v_center_mph = (center_a / kc).sqrt() * crate::sigmoid::MS_TO_MPH;
  // A 2σ swing in log-κ corresponds to a κ ratio of 10^(2σ); convert to
  // approximate mph delta via v ∝ 1/√κ at fixed a.
  let log_lo = prep.log_kc - 2.0 * prep.sigma;
  let log_hi = prep.log_kc + 2.0 * prep.sigma;
  let kappa_lo = 10f64.powf(log_lo);
  let kappa_hi = 10f64.powf(log_hi);
  let a_lo = params.eval(kappa_lo);
  let a_hi = params.eval(kappa_hi);
  let v_hi_mph = (a_lo / kappa_lo).sqrt() * crate::sigmoid::MS_TO_MPH; // small κ → high v
  let v_lo_mph = (a_hi / kappa_hi).sqrt() * crate::sigmoid::MS_TO_MPH;
  let left_x = map
    .x_to_px(v_lo_mph.min(v_hi_mph).max(X_RANGE_MPH.0))
    .max(plot.left());
  let right_x = map
    .x_to_px(v_lo_mph.max(v_hi_mph).min(X_RANGE_MPH.1))
    .min(plot.right());
  if right_x <= left_x + 0.5 {
    return;
  }
  let zone = Rect::from_min_max(pos2(left_x, plot.top()), pos2(right_x, plot.bottom()));
  painter.rect_filled(zone, CornerRadius::same(2), color.gamma_multiply(0.10));
  // Center tick on the curve
  let cx = map.x_to_px(v_center_mph);
  painter.line_segment(
    [pos2(cx, plot.top()), pos2(cx, plot.bottom())],
    Stroke::new(1.0, color.gamma_multiply(0.35)),
  );
}
