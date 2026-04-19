//! Dark, pro-audio palette. All colors live here so a future "light" theme
//! or palette tweak only touches one file.

use eframe::egui::{self, Color32, CornerRadius, Stroke};

pub const BG: Color32 = Color32::from_rgb(0x0f, 0x12, 0x17);
pub const PANEL: Color32 = Color32::from_rgb(0x1a, 0x1e, 0x26);
pub const PANEL_HI: Color32 = Color32::from_rgb(0x23, 0x28, 0x32);
pub const SEPARATOR: Color32 = Color32::from_rgb(0x2e, 0x34, 0x40);
pub const TEXT: Color32 = Color32::from_rgb(0xdf, 0xe4, 0xee);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x8a, 0x94, 0xa3);
pub const TEXT_DIM: Color32 = Color32::from_rgb(0x5a, 0x63, 0x73);

/// Primary accent — the sigmoid curve and active knob pointer.
pub const ACCENT: Color32 = Color32::from_rgb(0x4c, 0xd6, 0xc1);
/// Secondary accent — EQ bands, warnings, highlights.
pub const ACCENT_WARM: Color32 = Color32::from_rgb(0xf0, 0xa0, 0x4b);
/// Clamp rails — draw as a dashed, slightly warm red-orange.
pub const RAIL: Color32 = Color32::from_rgb(0xd6, 0x4f, 0x4f);

/// Grid on the plot. Very subtle; must not compete with the curve.
pub const GRID: Color32 = Color32::from_rgb(0x25, 0x2b, 0x36);
pub const GRID_MAJOR: Color32 = Color32::from_rgb(0x34, 0x3b, 0x48);

pub fn install(ctx: &egui::Context) {
  // 1.5× UI scale — readable but not oversized.  Keeps knob / hit-target
  // metrics in proportion with the text rather than the text floating on
  // small widgets.
  ctx.set_pixels_per_point(1.5);

  let mut visuals = egui::Visuals::dark();
  visuals.override_text_color = Some(TEXT);
  visuals.panel_fill = BG;
  visuals.window_fill = PANEL;
  visuals.extreme_bg_color = BG;
  visuals.faint_bg_color = PANEL;
  visuals.code_bg_color = PANEL_HI;
  visuals.widgets.noninteractive.bg_fill = PANEL;
  visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, SEPARATOR);
  visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_MUTED);
  visuals.widgets.inactive.bg_fill = PANEL_HI;
  visuals.widgets.inactive.weak_bg_fill = PANEL_HI;
  visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, SEPARATOR);
  visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
  visuals.widgets.hovered.bg_fill = SEPARATOR;
  visuals.widgets.hovered.weak_bg_fill = SEPARATOR;
  visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
  visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT);
  visuals.widgets.active.bg_fill = ACCENT;
  visuals.widgets.active.weak_bg_fill = ACCENT;
  visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
  visuals.widgets.active.fg_stroke = Stroke::new(1.0, BG);
  visuals.selection.bg_fill = ACCENT.gamma_multiply(0.4);
  visuals.selection.stroke = Stroke::new(1.0, ACCENT);
  visuals.widgets.noninteractive.corner_radius = CornerRadius::same(6);
  visuals.widgets.inactive.corner_radius = CornerRadius::same(6);
  visuals.widgets.hovered.corner_radius = CornerRadius::same(6);
  visuals.widgets.active.corner_radius = CornerRadius::same(6);
  ctx.set_visuals(visuals);

  let mut style: egui::Style = (*ctx.style()).clone();
  style.spacing.item_spacing = egui::vec2(10.0, 8.0);
  style.spacing.button_padding = egui::vec2(10.0, 6.0);
  style.spacing.menu_margin = egui::Margin::same(8);
  ctx.set_style(style);
}
