//! Rotary knob widget — vertical-drag to turn, scroll-wheel fine-tune,
//! double-click to restore the default.

use std::f32::consts::{PI, TAU};

use eframe::egui::{
  self, Align2, Color32, FontId, Response, Sense, Stroke, Ui, Vec2, Widget, pos2, vec2,
};

use crate::theme;

pub struct Knob<'a> {
  label: &'a str,
  value: &'a mut f64,
  range: (f64, f64),
  default: f64,
  /// Logarithmic mapping when true (useful for sharpness / steepness).
  log: bool,
  unit: &'a str,
  precision: usize,
  /// How many pixels of vertical drag spans the full range.
  drag_span_px: f32,
  diameter: f32,
  /// Color of the value arc + halo. Defaults to theme::ACCENT but EQ bands
  /// override this with `band_color(idx)` for visual coordination.
  accent: Color32,
  /// When true, the value text below the knob is replaced with an editable
  /// DragValue (click to type, drag to nudge).
  editable_text: bool,
}

impl<'a> Knob<'a> {
  pub fn new(label: &'a str, value: &'a mut f64, range: (f64, f64), default: f64) -> Self {
    Self {
      label,
      value,
      range,
      default,
      log: false,
      unit: "",
      precision: 2,
      drag_span_px: 200.0,
      diameter: 72.0,
      accent: theme::ACCENT,
      editable_text: false,
    }
  }
  #[allow(dead_code)] // reserved for non-linear knobs (e.g. raw B steepness)
  pub fn log(mut self, on: bool) -> Self {
    self.log = on;
    self
  }
  pub fn unit(mut self, u: &'a str) -> Self {
    self.unit = u;
    self
  }
  pub fn precision(mut self, p: usize) -> Self {
    self.precision = p;
    self
  }
  pub fn diameter(mut self, d: f32) -> Self {
    self.diameter = d;
    self
  }
  pub fn accent(mut self, c: Color32) -> Self {
    self.accent = c;
    self
  }
  pub fn editable_text(mut self, on: bool) -> Self {
    self.editable_text = on;
    self
  }

  fn to_normalized(&self, v: f64) -> f32 {
    let (lo, hi) = self.range;
    if self.log {
      let lo = lo.max(1e-9).log10();
      let hi = hi.max(lo + 1e-9).log10();
      let v = v.max(1e-9).log10();
      ((v - lo) / (hi - lo)).clamp(0.0, 1.0) as f32
    } else {
      ((v - lo) / (hi - lo)).clamp(0.0, 1.0) as f32
    }
  }

  fn from_normalized(&self, n: f32) -> f64 {
    let (lo, hi) = self.range;
    let n = n.clamp(0.0, 1.0) as f64;
    if self.log {
      let lo = lo.max(1e-9).log10();
      let hi = hi.max(lo + 1e-9).log10();
      10f64.powf(lo + (hi - lo) * n)
    } else {
      lo + (hi - lo) * n
    }
  }

  pub fn show(self, ui: &mut Ui) -> Response {
    let Knob {
      label,
      value,
      range,
      default,
      log,
      unit,
      precision,
      drag_span_px,
      diameter,
      accent,
      editable_text,
    } = self;
    let label_h = ui.text_style_height(&egui::TextStyle::Small);
    let label_font = FontId::proportional(11.0);
    // Measure the label so the knob's footprint is wide enough to print it
    // without clipping (otherwise long labels like "Tight-Curve Ceiling" get
    // chopped on both ends inside the rect's clip region).
    let label_width = ui.ctx().fonts_mut(|f| {
      let font_id = label_font.clone();
      let mut w = 0.0f32;
      for ch in label.chars() {
        w += f.glyph_width(&font_id, ch);
      }
      w
    });
    // Knob region — circle plus the label band above. The value text below is
    // either painted in-place or replaced by an editable DragValue depending
    // on `editable_text`.
    let value_band_h = if editable_text {
      ui.text_style_height(&egui::TextStyle::Body) + 6.0
    } else {
      label_h * 1.4
    };
    let knob_h = diameter + label_h + 14.0;
    let knob_w = (diameter + 12.0).max(label_width + 12.0);
    let desired = Vec2::new(knob_w, knob_h + value_band_h);
    let (outer_rect, _outer_resp) = ui.allocate_exact_size(desired, Sense::hover());
    let knob_rect = egui::Rect::from_min_size(
      outer_rect.min,
      egui::vec2(outer_rect.width(), knob_h),
    );
    let value_rect = egui::Rect::from_min_size(
      pos2(outer_rect.left(), knob_rect.bottom()),
      egui::vec2(outer_rect.width(), value_band_h),
    );

    // Manual interaction-rect since the outer allocation only sensed hover.
    let mut response = ui.interact(
      knob_rect,
      ui.id().with(("knob", label)),
      Sense::click_and_drag(),
    );
    let mut changed = false;
    let normalize = |v: f64| -> f32 {
      let (lo, hi) = range;
      if log {
        let lo = lo.max(1e-9).log10();
        let hi = hi.max(lo + 1e-9).log10();
        let v = v.max(1e-9).log10();
        ((v - lo) / (hi - lo)).clamp(0.0, 1.0) as f32
      } else {
        ((v - lo) / (hi - lo)).clamp(0.0, 1.0) as f32
      }
    };
    let denorm = |n: f32| -> f64 {
      let (lo, hi) = range;
      let n = n.clamp(0.0, 1.0) as f64;
      if log {
        let lo = lo.max(1e-9).log10();
        let hi = hi.max(lo + 1e-9).log10();
        10f64.powf(lo + (hi - lo) * n)
      } else {
        lo + (hi - lo) * n
      }
    };
    let mut norm = normalize(*value);

    // --- drag ---
    if response.dragged() {
      let dy = -response.drag_delta().y;
      let fine = ui.input(|i| i.modifiers.shift);
      let scale = if fine { 0.1 } else { 1.0 };
      norm = (norm + dy / drag_span_px * scale).clamp(0.0, 1.0);
      *value = denorm(norm);
      changed = true;
    }

    // --- scroll wheel (when hovered) ---
    if response.hovered() {
      let scroll = ui.input(|i| i.raw_scroll_delta.y);
      if scroll.abs() > 0.0 {
        let fine = ui.input(|i| i.modifiers.shift);
        let scale = if fine { 0.002 } else { 0.01 };
        norm = (norm + scroll * scale).clamp(0.0, 1.0);
        *value = denorm(norm);
        changed = true;
      }
    }

    // --- double-click to reset ---
    if response.double_clicked() {
      *value = default;
      norm = normalize(*value);
      changed = true;
    }

    // ---------- paint knob ----------
    let painter = ui.painter_at(knob_rect);
    let center = pos2(knob_rect.center().x, knob_rect.top() + label_h + 6.0 + diameter * 0.5);
    let radius = diameter * 0.5;
    painter.circle_filled(center, radius, theme::PANEL_HI);
    painter.circle_stroke(center, radius, Stroke::new(1.0, theme::SEPARATOR));
    let inner_r = radius - 6.0;
    painter.circle_filled(center, inner_r, theme::PANEL);
    let start_angle = -PI * 0.75;
    let sweep = (3.0 * PI / 2.0) * norm;
    // Faint full track first…
    draw_arc(
      &painter,
      center,
      inner_r - 3.0,
      start_angle - PI * 0.5,
      3.0 * PI / 2.0,
      Stroke::new(2.0, theme::SEPARATOR),
    );
    // …then the active sweep on top.
    draw_arc(
      &painter,
      center,
      inner_r - 3.0,
      start_angle - PI * 0.5,
      sweep,
      Stroke::new(3.0, accent),
    );
    let pointer_angle = start_angle + sweep - PI * 0.5;
    let p0 = center
      + vec2(
        pointer_angle.cos() * (inner_r - 16.0),
        pointer_angle.sin() * (inner_r - 16.0),
      );
    let p1 = center
      + vec2(
        pointer_angle.cos() * (inner_r - 4.0),
        pointer_angle.sin() * (inner_r - 4.0),
      );
    painter.line_segment([p0, p1], Stroke::new(2.5, theme::TEXT));
    painter.text(
      pos2(knob_rect.center().x, knob_rect.top()),
      Align2::CENTER_TOP,
      label,
      label_font.clone(),
      theme::TEXT_MUTED,
    );
    if response.hovered() || response.dragged() {
      painter.circle_stroke(center, radius + 1.5, Stroke::new(1.0, accent));
    }

    // ---------- value band ----------
    if editable_text {
      let mut child = ui.new_child(
        egui::UiBuilder::new()
          .max_rect(value_rect)
          .layout(egui::Layout::centered_and_justified(
            egui::Direction::TopDown,
          )),
      );
      // Explicit reborrow so `value` is still usable for the hover_text below.
      let mut dv = egui::DragValue::new(&mut *value)
        .range(range.0..=range.1)
        .speed((range.1 - range.0).abs() / 200.0)
        .fixed_decimals(precision);
      if !unit.is_empty() {
        dv = dv.suffix(format!(" {unit}"));
      }
      let dv_resp = child.add(dv);
      if dv_resp.changed() {
        changed = true;
      }
    } else {
      let value_text = format!(
        "{:.*}{}{}",
        precision,
        *value,
        if unit.is_empty() { "" } else { " " },
        unit
      );
      painter.text(
        pos2(
          value_rect.center().x,
          value_rect.top() + value_rect.height() * 0.5,
        ),
        Align2::CENTER_CENTER,
        value_text,
        FontId::monospace(12.0),
        theme::TEXT,
      );
    }

    if changed {
      response.mark_changed();
    }

    let final_value = *value;
    response.on_hover_text(format!(
      "{}: {:.6}{}{}\n\ndrag vertical • shift=fine • scroll=nudge • double-click=reset",
      label,
      final_value,
      if unit.is_empty() { "" } else { " " },
      unit
    ))
  }
}

/// Draw a 2D arc as a polyline. egui's Painter has no native arc primitive.
fn draw_arc(
  painter: &egui::Painter,
  center: egui::Pos2,
  radius: f32,
  start_angle: f32,
  sweep: f32,
  stroke: Stroke,
) {
  if sweep.abs() < 1e-3 {
    return;
  }
  let segments = (sweep.abs() / (TAU / 128.0)).ceil().max(2.0) as usize;
  let mut pts = Vec::with_capacity(segments + 1);
  for i in 0..=segments {
    let t = i as f32 / segments as f32;
    let a = start_angle + sweep * t;
    pts.push(center + vec2(a.cos() * radius, a.sin() * radius));
  }
  painter.add(egui::Shape::line(pts, stroke));
}

impl<'a> Widget for Knob<'a> {
  fn ui(self, ui: &mut Ui) -> Response {
    self.show(ui)
  }
}
