/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/hud.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <QPainterPath>
#include <QLinearGradient>
#include <QRadialGradient>
#include <QTransform>
#include <QTime>
#include <QMutexLocker>
#include <unordered_set>
#include <unordered_map>
#include <limits>

#include "common/params.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/ui.h"  // For UI_FREQ

// (Removed) legacy static polygons used by the deprecated single-threat widget icons

// Static color constants for performance
static const QColor kRtiColorWhite(255, 255, 255, 245);    // Default white for text/arrows
static const QColor kRtiColorNeonYellow(255, 255, 0, 255); // Neon yellow for on_same_road arrows

// Helper function to create neon color from base color
static QColor createNeonColor(const QColor& baseColor) {
  QColor neonColor = baseColor.toHsl();
  int h, s, l, a;
  neonColor.getHsl(&h, &s, &l, &a);
  // Neon effect: max saturation, medium-high lightness
  neonColor.setHsl(h, 255, 160, 255);
  return neonColor;
}

// Helper functions
// Distance comparator helpers
static inline bool compareByDistance(const RTIThreatInfo& a, const RTIThreatInfo& b) {
  return a.distance < b.distance;  // ascending (closest first)
}

static inline bool compareByDistancePtr(const RTIThreatInfo* a, const RTIThreatInfo* b) {
  return a->distance < b->distance;  // ascending (closest first)
}

static inline bool comparePolicePriority(const RTIThreatInfo* a, const RTIThreatInfo* b) {
  // Police on same road have highest priority, then by distance
  if (a->on_same_road != b->on_same_road) return a->on_same_road && !b->on_same_road;
  return a->distance < b->distance;
}

static inline double normalize180(double a) {
  // Bounds check for invalid values
  if (!std::isfinite(a)) return 0.0;
  if (std::abs(a) > 1e6) return 0.0;  // Sanity check for unreasonable angles
  
  // Use fmod for efficient normalization
  a = std::fmod(a + 180.0, 360.0);
  if (a < 0) a += 360.0;
  return a - 180.0;
}

// Numeric validation and conversion helpers
static inline int roundToInt(double value) {
  return static_cast<int>(std::lround(value));
}

static float readParamFloatClamped(Params &params, const char *key, float fallback, float lo, float hi) {
  try {
    const std::string raw = params.get(key);
    if (raw.empty()) {
      return fallback;
    }
    const float v = std::stof(raw);
    if (!std::isfinite(v)) {
      return fallback;
    }
    return std::clamp(v, lo, hi);
  } catch (const std::exception&) {
    return fallback;
  }
}



// Template function for pruning caches based on active IDs
template<typename CacheType>
static void pruneCacheByActiveIds(CacheType& cache, QMutex& mutex, const std::unordered_set<std::string>& active_ids) {
  QMutexLocker lock(&mutex);
  for (auto it = cache.begin(); it != cache.end(); ) {
    if (active_ids.find(it->first) == active_ids.end()) {
      it = cache.erase(it);
    } else {
      ++it;
    }
  }
}

// Centralized threat type mapping - must be defined before first use
struct ThreatTypeInfo {
  const char* text;
  QColor bg_color;
};

static const std::unordered_map<RTIThreatType, ThreatTypeInfo> kThreatTypeMap = {
  {cereal::RtiStateSP::ThreatType::POLICE,          {"POLICE", QColor(200, 30, 30)}},
  {cereal::RtiStateSP::ThreatType::POLICE_HIDING,   {"POLICE", QColor(200, 30, 30)}},
  {cereal::RtiStateSP::ThreatType::SPEED_TRAP,      {"CAMERA", QColor(40, 120, 230)}},
  {cereal::RtiStateSP::ThreatType::SPEED_CAMERA,    {"CAMERA", QColor(40, 120, 230)}},
  {cereal::RtiStateSP::ThreatType::ACCIDENT,        {"ACCIDENT", QColor(240, 140, 0)}},
  {cereal::RtiStateSP::ThreatType::JAM,             {"TRAFFIC", QColor(160, 80, 200)}},
  {cereal::RtiStateSP::ThreatType::CONSTRUCTION,    {"CONSTRUCTION", QColor(215, 130, 0)}},
  {cereal::RtiStateSP::ThreatType::HAZARD,          {"HAZARD", QColor(220, 200, 0)}},
  {cereal::RtiStateSP::ThreatType::SHOULDER_HAZARD, {"HAZARD", QColor(220, 200, 0)}},
  {cereal::RtiStateSP::ThreatType::ROAD_HAZARD,     {"HAZARD", QColor(220, 200, 0)}},
  {cereal::RtiStateSP::ThreatType::ROAD_CLOSED,     {"CLOSED", QColor(100, 100, 100)}},
};

static const ThreatTypeInfo kDefaultThreatInfo = {"ALERT", QColor(80, 80, 80)};

HudRendererSP::HudRendererSP() {
  // RTI state initialized with safe defaults
  // rti_enabled will be updated periodically in updateState()
}

void HudRendererSP::refreshVTSCCoPilotTuning() {
  VTSCCoPilotHudTuning tuning{};
  Params params;

  // Expert gate prevents stale unsafe values from silently applying.
  if (!params.getBool("VTSCExpertModeEnabled")) {
    vtsc_copilot_tuning_ = tuning;
    return;
  }

  tuning.curve_hold_new_dist_min_m = readParamFloatClamped(params, "VTSCHUD.CurveHoldNewDistMinM", tuning.curve_hold_new_dist_min_m, 0.0f, 250.0f);
  tuning.geometry_epsilon_m = readParamFloatClamped(params, "VTSCHUD.GeometryEpsilonM", tuning.geometry_epsilon_m, 0.001f, 1.0f);
  tuning.kappa_show_min = readParamFloatClamped(params, "VTSCHUD.KappaShowMin", tuning.kappa_show_min, 1e-6f, 0.05f);
  tuning.kappa_hold_min = readParamFloatClamped(params, "VTSCHUD.KappaHoldMin", tuning.kappa_hold_min, 1e-6f, 0.05f);
  tuning.kappa_hold_min = std::min(tuning.kappa_hold_min, tuning.kappa_show_min);
  tuning.fade_in_alpha = readParamFloatClamped(params, "VTSCHUD.FadeInAlpha", tuning.fade_in_alpha, 0.01f, 0.95f);
  tuning.fade_out_alpha = readParamFloatClamped(params, "VTSCHUD.FadeOutAlpha", tuning.fade_out_alpha, 0.01f, 0.95f);
  tuning.scale = readParamFloatClamped(params, "VTSCHUD.Scale", tuning.scale, 0.5f, 4.0f);
  tuning.bottom_safe_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.BottomSafePxAtScale1", tuning.bottom_safe_px_at_scale1, 0.0f, 120.0f);
  tuning.pad_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.PadPxAtScale1", tuning.pad_px_at_scale1, 2.0f, 80.0f);
  tuning.gap_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.GapPxAtScale1", tuning.gap_px_at_scale1, 0.0f, 80.0f);
  tuning.top_height_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.TopHeightPxAtScale1", tuning.top_height_px_at_scale1, 8.0f, 120.0f);
  tuning.bottom_height_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.BottomHeightPxAtScale1", tuning.bottom_height_px_at_scale1, 8.0f, 140.0f);
  tuning.min_curve_area_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.MinCurveAreaPxAtScale1", tuning.min_curve_area_px_at_scale1, 16.0f, 240.0f);
  tuning.road_main_width_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.RoadMainWidthPxAtScale1", tuning.road_main_width_px_at_scale1, 2.0f, 48.0f);
  tuning.glow_width_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.GlowWidthPxAtScale1", tuning.glow_width_px_at_scale1, 2.0f, 120.0f);
  tuning.outline_width_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.OutlineWidthPxAtScale1", tuning.outline_width_px_at_scale1, 1.0f, 100.0f);
  tuning.main_stroke_width_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.MainStrokeWidthPxAtScale1", tuning.main_stroke_width_px_at_scale1, 1.0f, 100.0f);
  tuning.distance_label_sep_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.DistanceLabelSepPxAtScale1", tuning.distance_label_sep_px_at_scale1, 0.0f, 120.0f);
  tuning.speed_font_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.SpeedFontPxAtScale1", tuning.speed_font_px_at_scale1, 8.0f, 80.0f);
  tuning.bottom_font_px_at_scale1 = readParamFloatClamped(params, "VTSCHUD.BottomFontPxAtScale1", tuning.bottom_font_px_at_scale1, 8.0f, 80.0f);

  vtsc_copilot_tuning_ = tuning;
}

void HudRendererSP::updateState(const UIState &s) {
  // Update base HUD state
  HudRenderer::updateState(s);
  
  // Update subsystem readiness from selfdriveStateSP
  if (s.sm && s.sm->valid("selfdriveStateSP") && s.sm->updated("selfdriveStateSP")) {
    try {
      const auto ss_sp = (*s.sm)["selfdriveStateSP"].getSelfdriveStateSP();
      all_systems_ready_ = ss_sp.getAllSystemsReady();
      auto statuses = ss_sp.getSubsystemStatuses();
      subsystem_statuses_.clear();
      subsystem_statuses_.reserve(statuses.size());
      for (const auto &st : statuses) {
        subsystem_statuses_.emplace_back(std::string(st.getName().cStr()), static_cast<int>(st.getStatus()));
      }
    } catch (const std::exception&) {
      // Keep previous state on error
    }
  }

  // Fade readiness column when engaged + all green
  float target_opacity = (status == STATUS_ENGAGED && all_systems_ready_) ? 0.3f : 1.0f;
  readiness_opacity_ += 0.05f * (target_opacity - readiness_opacity_);

  // Refresh params at 5 Hz for live tuning responsiveness without per-frame overhead.
  if (s.sm && s.sm->frame % 4 == 0) {
    rti_enabled = Params().getBool("RTIEnabled");  // Master switch
    rti_hud_enabled = Params().getBool("RTIHUDEnabled");  // HUD display switch
    vtsc_copilot_hud_enabled_ = Params().getBool("VTSCRallyCoPilotHUDEnabled");
    refreshVTSCCoPilotTuning();
  }
  
  // Update multiple threats only if RTI HUD is enabled
  if (rti_enabled && rti_hud_enabled) {
    updateRTIThreats(s);
  } else {
    // Clear threat data when disabled
    rti_threats.clear();
    rti_has_threat = false;
    rti_threat_ahead = false;
  }
  
  // Safe RTI message access with multiple layers of protection
  if (s.sm) {
    try {
      // Check for stale RTI data (20Hz message, timeout after 3 seconds)
      // Only check if the message is valid AND has been received at least once
      if (s.sm->valid("rtiStateSP")) {
        // Additional safety: check if rcv_frame > 0 to ensure message was actually received
        uint64_t rti_rcv_frame = s.sm->rcv_frame("rtiStateSP");
        if (rti_rcv_frame > 0 && (s.sm->frame - rti_rcv_frame) > 3 * UI_FREQ) {
          // Mark as stale/inactive but don't reset all data immediately
          rti_active = false;
        }
      }
      
      // Update RTI state from messages - only if enabled and message is valid and updated
      if (rti_enabled && rti_hud_enabled && s.sm->valid("rtiStateSP") && s.sm->updated("rtiStateSP")) {
        const auto rti_state = (*s.sm)["rtiStateSP"].getRtiStateSP();
        
        rti_threat_ahead = rti_state.getThreatAhead();
        
        // Validate and set threat distance with bounds checking
        float raw_distance = rti_state.getThreatDistanceM();
        rti_threat_distance = (std::isfinite(raw_distance) && raw_distance >= 0) ? raw_distance : 0.0;
        
        // Validate and set recommended speed with bounds checking
        float raw_speed = rti_state.getRecommendedSpeed();
        rti_recommended_speed = (std::isfinite(raw_speed) && raw_speed >= 0) ? raw_speed : 0.0;
        
        // Get threat details if available
        auto threats = rti_state.getThreats();
        if (threats.size() > 0) {
          // Use the first (closest) threat
          auto threat = threats[0];
          
          // Use threat type directly from capnp - no mapping needed
          rti_threat_type = threat.getType();
          rti_has_threat = true;
          
          // Get threat location for arrow calculation
          rti_threat_lat = threat.getLatitude();
          rti_threat_lon = threat.getLongitude();
          
          // Validate and set confidence with bounds checking
          float raw_confidence = threat.getConfidence();
          rti_threat_confidence = (std::isfinite(raw_confidence) && raw_confidence >= 0.0 && raw_confidence <= 1.0) ?
                                  raw_confidence : 0.0;
        } else {
          rti_has_threat = false;
          rti_threat_confidence = 0.0;
          rti_threat_lat = 0.0;
          rti_threat_lon = 0.0;
        }
      }
      
      // Check if RTI is actively controlling speed (from longitudinal planner)
      if (s.sm->valid("longitudinalPlanSP") && s.sm->updated("longitudinalPlanSP")) {
        // RTI is active if threat is ahead and we have a valid speed recommendation
        rti_active = rti_threat_ahead && rti_recommended_speed > 0;
      }
      
    } catch (const std::exception& e) {
      // Handle any exceptions from message access safely - reset to safe state
      rti_has_threat = false;
      rti_threat_ahead = false;
      rti_active = false;
    }
  }

  // ===== VTSC Rally Co-Pilot: always-on road preview =====
  if (!vtsc_copilot_hud_enabled_ || !s.sm) {
    vtsc_copilot_alpha_ = 0.0f;
    vtsc_copilot_curve_points_m_.clear();
    vtsc_copilot_tiles_.clear();
    vtsc_copilot_prev_tiles_.clear();
    vtsc_copilot_exiting_tile_active_ = false;
    vtsc_copilot_last_nonempty_tiles_time_valid_ = false;
    vtsc_copilot_branch_stubs_.clear();
    return;
  }

  // Cache v_ego for speed display and time-based metadata.
  try {
    if (s.sm->valid("carState")) {
      vtsc_copilot_v_ego_mps_ = (*s.sm)["carState"].getCarState().getVEgo();
    }
  } catch (const std::exception&) {}

  // Grab per-turn tiles from VTSC. The HUD should only render this published list.
  std::vector<VTSCCoPilotTileState> new_tiles;
  try {
    if (s.sm->valid("longitudinalPlanSP")) {
      const auto lp_sp = (*s.sm)["longitudinalPlanSP"].getLongitudinalPlanSP();
      const auto vtsc = lp_sp.getVisionTurnSpeedControl();

      if (vtsc.getCurvePreviewValid()) {
        auto tiles = vtsc.getCurvePreviewTiles();
        new_tiles.reserve(static_cast<size_t>(tiles.size()));
        for (const auto &tile : tiles) {
          VTSCCoPilotTileState state;
          state.id = tile.getTileId();
          state.distance_m = tile.getDistanceM();
          state.time_to_s = tile.getTimeToS();
          state.advisory_speed_mps = tile.getAdvisorySpeedMps();
          state.max_curvature = tile.getMaxCurvature();
          state.direction = static_cast<int>(tile.getDirection());
          state.severity = static_cast<int>(tile.getSeverity());

          auto pts = tile.getPoints();
          state.points_m.reserve(static_cast<size_t>(pts.size()));
          for (const auto &pt : pts) {
            state.points_m.emplace_back(pt.getXFwdM(), pt.getYLeftM());
          }
          if (state.points_m.size() >= 2) {
            new_tiles.push_back(std::move(state));
          }
        }
      }
    }
  } catch (const std::exception&) {}

  const auto tiles_now = std::chrono::steady_clock::now();
  if (!new_tiles.empty()) {
    vtsc_copilot_last_nonempty_tiles_time_ = tiles_now;
    vtsc_copilot_last_nonempty_tiles_time_valid_ = true;
  } else if (!vtsc_copilot_tiles_.empty() && vtsc_copilot_last_nonempty_tiles_time_valid_) {
    constexpr float kMissingTileHoldS = 0.35f;
    const float empty_dt = std::chrono::duration<float>(tiles_now - vtsc_copilot_last_nonempty_tiles_time_).count();
    if (empty_dt < kMissingTileHoldS) {
      new_tiles = vtsc_copilot_tiles_;
    }
  }

  auto same_tile_order = [](const auto &lhs, const auto &rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i].id != rhs[i].id) return false;
    }
    return true;
  };
  auto contains_tile_id = [](const auto &tiles, uint32_t id) {
    return std::any_of(tiles.begin(), tiles.end(), [id](const auto &tile) { return tile.id == id; });
  };

  if (!same_tile_order(new_tiles, vtsc_copilot_tiles_)) {
    const uint32_t previous_bottom_id = vtsc_copilot_tiles_.empty() ? 0 : vtsc_copilot_tiles_.front().id;
    if (previous_bottom_id != 0 && !contains_tile_id(new_tiles, previous_bottom_id)) {
      vtsc_copilot_exiting_tile_ = vtsc_copilot_tiles_.front();
      vtsc_copilot_exiting_tile_active_ = !vtsc_copilot_exiting_tile_.points_m.empty();
      vtsc_copilot_exit_anim_progress_ = 0.0f;
    }
    vtsc_copilot_prev_tiles_ = vtsc_copilot_tiles_;
    vtsc_copilot_tiles_ = std::move(new_tiles);
    vtsc_copilot_stack_anim_progress_ = vtsc_copilot_prev_tiles_.empty() ? 1.0f : 0.0f;
  } else {
    vtsc_copilot_tiles_ = std::move(new_tiles);
  }

  if (!vtsc_copilot_tiles_.empty()) {
    const auto &tile = vtsc_copilot_tiles_.front();
    vtsc_copilot_curve_distance_m_ = tile.distance_m;
    vtsc_copilot_curve_time_to_s_ = tile.time_to_s;
    vtsc_copilot_curve_kappa_max_ = tile.max_curvature;
    vtsc_copilot_curve_direction_ = tile.direction;
    vtsc_copilot_curve_severity_ = tile.severity;
    vtsc_target_speed_mps_ = tile.advisory_speed_mps;
  } else {
    vtsc_copilot_curve_distance_m_ = 0.0f;
    vtsc_copilot_curve_time_to_s_ = 0.0f;
    vtsc_copilot_curve_kappa_max_ = 0.0f;
    vtsc_copilot_curve_direction_ = 0;
    vtsc_copilot_curve_severity_ = 0;
    vtsc_target_speed_mps_ = 0.0f;
  }

  vtsc_copilot_alpha_ = (!vtsc_copilot_tiles_.empty() || vtsc_copilot_exiting_tile_active_) ? 1.0f : 0.0f;
}

void HudRendererSP::draw(QPainter &p, const QRect &surface_rect) {
  // Draw base HUD elements
  HudRenderer::draw(p, surface_rect);

  // Draw system readiness indicator
  drawSystemReadiness(p, surface_rect);

  // Draw VTSC rally co-pilot curve preview when enabled and visible (with fade)
  if (vtsc_copilot_hud_enabled_ && vtsc_copilot_alpha_ > 0.01f) {
    drawVTSCCoPilotCurve(p, surface_rect);
  }

  // Draw RTI widget when enabled (multi-threat only)
  if (rti_enabled && rti_hud_enabled) {
    drawRTIThreatIndicatorMulti(p, surface_rect);
  }
}


void HudRendererSP::drawSystemReadiness(QPainter &p, const QRect &surface_rect) {
  if (subsystem_statuses_.empty()) return;

  p.save();
  p.setRenderHint(QPainter::Antialiasing, true);

  const float opacity = readiness_opacity_;
  const bool show_labels = !all_systems_ready_;

  // Dot sizing and layout
  const int dot_r = 6;       // small dot radius (12px diameter)
  const int master_r = 10;   // master dot radius (20px diameter)
  const int spacing = 28;    // vertical spacing between dot centers
  const int master_gap = 8;  // extra gap before master dot
  const int label_gap = 6;   // gap between dot and label

  // Total column height: N subsystem dots + gap + master dot
  const int n = static_cast<int>(subsystem_statuses_.size());
  const int col_h = (n - 1) * spacing + 2 * dot_r + master_gap + 2 * master_r;
  const int x_center = 30;   // dot center x from left edge
  const int y_top = (surface_rect.height() - col_h) / 2;

  // Colors
  static const QColor kRed(0xFF, 0x33, 0x33);
  static const QColor kYellow(0xFF, 0xCC, 0x00);
  static const QColor kGreen(0x33, 0xCC, 0x33);

  auto colorForStatus = [&](int st) -> QColor {
    if (st == 0) return kRed;
    if (st == 1) return kYellow;
    return kGreen;
  };

  // Background pill
  const int pill_pad = 8;
  int pill_w = show_labels ? 80 : 36;
  QRect pill(x_center - pill_w / 2, y_top - pill_pad,
             pill_w, col_h + 2 * pill_pad);
  p.setPen(Qt::NoPen);
  p.setBrush(QColor(0, 0, 0, static_cast<int>(100 * opacity)));
  p.drawRoundedRect(pill, 12, 12);

  // Draw subsystem dots (bottom to top: index 0 at bottom)
  for (int i = 0; i < n; i++) {
    const auto &[name, st] = subsystem_statuses_[i];
    QColor c = colorForStatus(st);

    // Y position: first dot at bottom, last at top
    int y = y_top + col_h - 2 * master_r - master_gap - dot_r - i * spacing;

    // Glow (larger circle at reduced opacity)
    QColor glow = c;
    glow.setAlphaF(0.3 * opacity);
    p.setPen(Qt::NoPen);
    p.setBrush(glow);
    p.drawEllipse(QPoint(x_center, y), dot_r + 4, dot_r + 4);

    // Dot
    c.setAlphaF(opacity);
    p.setBrush(c);
    p.drawEllipse(QPoint(x_center, y), dot_r, dot_r);

    // Label (only when not all green)
    if (show_labels) {
      QFont lbl_font = InterFont(18, QFont::DemiBold);
      p.setFont(lbl_font);
      c.setAlphaF(0.9 * opacity);
      p.setPen(c);
      p.drawText(x_center + dot_r + label_gap, y + 5, QString::fromStdString(name));
    }
  }

  // Master dot at top
  int master_y = y_top + master_r;
  QColor master_c = all_systems_ready_ ? kGreen : kRed;

  // Pulse effect when not all green
  if (!all_systems_ready_) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    float pulse = 0.6f + 0.4f * static_cast<float>(std::sin(ms / 500.0 * M_PI));
    master_c.setAlphaF(pulse * opacity);
  } else {
    master_c.setAlphaF(opacity);
  }

  // Master glow
  QColor master_glow = master_c;
  master_glow.setAlphaF(0.3 * opacity);
  p.setPen(Qt::NoPen);
  p.setBrush(master_glow);
  p.drawEllipse(QPoint(x_center, master_y), master_r + 5, master_r + 5);

  // Master dot
  p.setBrush(master_c);
  p.drawEllipse(QPoint(x_center, master_y), master_r, master_r);

  // Master label
  if (show_labels) {
    QFont lbl_font = InterFont(18, QFont::Bold);
    p.setFont(lbl_font);
    master_c.setAlphaF(0.9 * opacity);
    p.setPen(master_c);
    p.drawText(x_center + master_r + label_gap, master_y + 5, "ALL");
  }

  p.restore();
}

QColor HudRendererSP::getRTIThreatColor(float distance) const {
  // Arrows are always white by default
  // Color changes happen based on on_same_road, not distance
  return kRtiColorWhite;
}



void HudRendererSP::updateRTIThreats(const UIState &s) {
  // Only clear and update threats when we have new data to avoid flicker
  if (s.sm && s.sm->valid("rtiStateSP") && s.sm->updated("rtiStateSP")) {
    rti_threats.clear();  // Clear only when we have new data
    
    const auto rti_state = (*s.sm)["rtiStateSP"].getRtiStateSP();
    auto threats = rti_state.getThreats();
    
    // Process all threats; display capping and prioritization occurs at draw time
    size_t num_threats = static_cast<size_t>(threats.size());
    for (size_t i = 0; i < num_threats; i++) {
      auto threat = threats[i];
      
      RTIThreatInfo info;
      if (threat.hasId()) {
        auto idtxt = threat.getId();
        info.id = std::string(idtxt.cStr());
      } else {
        info.id = std::to_string(i);
      }
      info.type = threat.getType();
      info.distance = threat.getDistance();
      info.latitude = threat.getLatitude();
      info.longitude = threat.getLongitude();
      info.has_location = threat.getHasLocation();  // Use pre-computed value from rtid
      info.direction = threat.getDirection();
      info.speed_limit_ms = threat.getSpeedLimitMs();
      info.on_same_road = threat.getOnSameRoad();
      info.is_causing_recommendation = threat.getIsCausingRecommendation();
      
      // Use pre-computed display angle from rtid
      info.relative_bearing = threat.getDisplayArrowAngle();
      
      rti_threats.push_back(info);
    }
    
    // Sort by distance in ASCENDING order (closest threat first)
    // This ensures the display shows:
    //   Top:    Closest threat (most urgent)
    //   Middle: Medium distance threats
    //   Bottom: Furthest threat (least urgent)
    std::sort(rti_threats.begin(), rti_threats.end(), compareByDistance);
    
    // Update legacy single-threat variables with closest threat
    if (!rti_threats.empty()) {
      rti_has_threat = true;
      rti_threat_distance = rti_threats[0].distance;
      rti_threat_type = rti_threats[0].type;
      rti_threat_lat = rti_threats[0].latitude;
      rti_threat_lon = rti_threats[0].longitude;
      rti_relative_bearing = rti_threats[0].relative_bearing;
      rti_direction = rti_threats[0].direction;
    } else {
      rti_has_threat = false;
    }

    // Prune smoothed caches to only keep active threats
    {
      std::unordered_set<std::string> live_ids;
      live_ids.reserve(rti_threats.size());
      for (const auto &t : rti_threats) live_ids.insert(t.id);
      
      pruneCacheByActiveIds(smoothed_angles_deg_, smoothed_angles_mutex_, live_ids);
      pruneCacheByActiveIds(smoothed_y_top_, smoothed_y_mutex_, live_ids);
    }
  }
}

void HudRendererSP::drawRTIThreatIndicatorMulti(QPainter &p, const QRect &surface_rect) {
  // Position to bottom-align with lateral accel widget
  const int bottom_margin = 15;
  const int left_margin = 15;

  // Transparent container; just a reserved area to stack mini-widgets
  // Increased width by 20% for better widget breathing room
  const int widget_width = 744;  // 620 * 1.2
  const int widget_height = 520;

  const int x_offset = left_margin;
  const int y_offset = surface_rect.height() - widget_height - bottom_margin;

  QRect rti_rect(x_offset, y_offset, widget_width, widget_height);

  if (!rti_threats.empty()) {
    const int outer_pad_y = 3;  // 3px top/bottom per widget => 6px total between
    const int padding_x = 12;
    const int padding_y = 8;
    // Bigger arrow for better readability and parity with legacy single
    const int arrow_sz = 48; // increased from ~38
    const int left_x = rti_rect.x() + 12;
    const int right_margin = 12;
    const int label_margin_px = 34;  // reserve space at bottom for Target label

    // Build selection with POLICE priority for display occupancy
    std::vector<const RTIThreatInfo*> police;
    std::vector<const RTIThreatInfo*> others;
    police.reserve(rti_threats.size());
    others.reserve(rti_threats.size());
    for (const auto &t : rti_threats) {
      bool is_police = (t.type == cereal::RtiStateSP::ThreatType::POLICE) ||
                       (t.type == cereal::RtiStateSP::ThreatType::POLICE_HIDING);
      (is_police ? police : others).push_back(&t);
    }
    // Sort police by same-road first, then distance asc
    std::sort(police.begin(), police.end(), comparePolicePriority);
    // Sort others by distance asc
    std::sort(others.begin(), others.end(), compareByDistancePtr);

    // Candidates in priority order: all POLICE (same-road then distance), then others (distance)
    std::vector<const RTIThreatInfo*> candidates;
    candidates.reserve(police.size() + others.size());
    for (auto *t : police) candidates.push_back(t);
    for (auto *t : others) candidates.push_back(t);

    // Use compact font; slightly larger for readability
    QFont row_font = InterFont(roundToInt(42 * 1.0), QFont::Normal);
    p.setFont(row_font);
    QFontMetrics fm(row_font);
    const int row_h = padding_y * 2 + std::max(arrow_sz, fm.height());

    // Dynamically select as many as fit within the block, reserving bottom margin
    std::vector<const RTIThreatInfo*> selected;
    selected.reserve(candidates.size());
    int y_cursor_fit = rti_rect.y() + rti_rect.height() - label_margin_px;
    for (auto *t : candidates) {
      int next_top = y_cursor_fit - outer_pad_y - row_h;
      if (next_top < rti_rect.y()) break;  // no more space
      selected.push_back(t);
      y_cursor_fit = next_top - outer_pad_y;
    }

    // Final ordering for display: ascending distance (closest first)
    std::sort(selected.begin(), selected.end(), compareByDistancePtr);

    // Build row objects with measured width/height per selection
    struct Row { const RTIThreatInfo* t; QString text; int w; int h; };
    std::vector<Row> rows;
    rows.reserve(selected.size());
    int max_natural_w = 0;
    for (auto *t : selected) {
      QString line = QString("%1 • %2").arg(getRTIThreatTextShort(t->type)).arg(formatDistance(t->distance));
      int text_w = fm.horizontalAdvance(line);
      // Natural width based on contents (used to compute uniform width below)
      int box_w_unscaled = padding_x + arrow_sz + 8 + text_w + padding_x;
      int box_w = box_w_unscaled;                                            // natural width
      int box_h = row_h;                                                      // uniform height per row
      max_natural_w = std::max(max_natural_w, box_w);
      rows.push_back({t, line, box_w, box_h});
    }

    // Change 4: enforce uniform width for all rows, clamped to container width
    int allowable_max_w = rti_rect.width() - (left_x - rti_rect.x()) - right_margin; // fit within transparent box
    // Add 20% more width for better breathing room
    int uniform_box_w = std::min(roundToInt(max_natural_w * 1.2), allowable_max_w);

    // Place from bottom-up with Y smoothing
    int y_cursor = rti_rect.y() + rti_rect.height() - label_margin_px;  // reserve space for bottom label
    for (int i = (int)rows.size() - 1; i >= 0; --i) {
      const auto &row = rows[i];
      // Colors
      QColor arrow_color = row.t->on_same_road ? kRtiColorNeonYellow : kRtiColorWhite;
      QColor bg_color = getRTIThreatBgColorByType(row.t->type);       // type-based hue (background)
      bg_color.setAlpha(115);
      QColor border_color(255, 255, 255, 90);                         // default border
      
      // Check if this threat is the one causing speed recommendation
      // This is now determined by threat_detector and passed through RTID
      bool is_active_threat = row.t->is_causing_recommendation;
      
      // Animate border for active threat
      if (is_active_threat) {
        // Create neon version of the threat color
        QColor neon_border = createNeonColor(bg_color);
        // Use time-based animation (toggles every ~500ms)
        static auto last_toggle = std::chrono::steady_clock::now();
        static bool pulse_state = false;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_toggle).count();
        if (elapsed >= 500) {
          pulse_state = !pulse_state;
          last_toggle = now;
        }
        if (pulse_state) {
          border_color = neon_border;
          border_color.setAlpha(240);
        } else {
          border_color.setAlpha(0);  // Transparent
        }
      }
      
      // Text is always white
      QColor text_color = kRtiColorWhite;

      // Target top position using outer breathing
      int target_top = y_cursor - outer_pad_y - row.h;
      // Initial position starts just above reserved label area for a clean slide-up
      double initial_top = rti_rect.y() + rti_rect.height() - label_margin_px + outer_pad_y;
      int draw_top = roundToInt(smoothYForThreat(row.t->id, target_top, initial_top));
      QRect box_rect(left_x, draw_top, uniform_box_w, row.h);

      // Update cursor for next (higher) row placement
      y_cursor = target_top - outer_pad_y;

      // Mini-widget background + thin border
      p.setPen(QPen(border_color, 2));
      p.setBrush(bg_color);
      p.drawRoundedRect(box_rect, 10, 10);

      // Use pre-computed arrow angle from rtid (bearing calculations done upstream)
      // The angle is already calculated with GPS precision or fallback logic as needed
      double arrow_angle = row.t->relative_bearing;
      
      // Apply smoothing for visual stability with adaptive rate
      arrow_angle = smoothAngleForThreat(row.t->id, arrow_angle);
      
      QRect arrow_rect(box_rect.x() + padding_x, box_rect.y() + (box_rect.height() - arrow_sz) / 2, arrow_sz, arrow_sz);
      drawRTIArrowCompact(p, arrow_rect, arrow_angle, arrow_color);

      // Draw text: always white
      p.setPen(text_color);
      int text_x = arrow_rect.right() + 8;
      QRect text_rect(text_x, box_rect.y(), box_rect.right() - text_x - 20, box_rect.height());
      p.drawText(text_rect, Qt::AlignVCenter | Qt::AlignLeft, row.text);
      // (Removed) previous same-road blinking dot per Change 3
    }

    if (rti_active && rti_recommended_speed > 0) {
      p.setFont(InterFont(42, QFont::Normal));
      p.setPen(QColor(255, 255, 255, 180));
      float rec_speed_display = rti_recommended_speed * (is_metric ? 3.6 : 2.237);
      QString speed_text = QString("Target: %1 %2").arg(static_cast<int>(rec_speed_display)).arg(is_metric ? "km/h" : "mph");
      p.drawText(rti_rect.adjusted(12, -34, -12, -6), Qt::AlignBottom | Qt::AlignLeft, speed_text);
    }
  } else {
    // No placeholder text when no threats
  }
}

void HudRendererSP::drawRTIArrowCompact(QPainter &p, const QRect &arrow_rect, 
                                        double relative_bearing, const QColor &color) {
  // Create smaller arrow for compact view
  const int size = arrow_rect.width();  // 32px for compact view
  
  if (!compact_arrow_cached || compact_arrow_size != size) {
    createCompactArrowPixmap(size);
  }
  
  p.save();
  
  // Apply rotation around center
  QTransform transform;
  transform.translate(arrow_rect.center().x(), arrow_rect.center().y());
  transform.rotate(relative_bearing);
  transform.translate(-size/2.0, -size/2.0);
  
  p.setTransform(transform, true);
  
  // Tint and draw the arrow
  QPixmap tinted_arrow = compact_arrow_pixmap;
  QPainter tint_painter(&tinted_arrow);
  tint_painter.setCompositionMode(QPainter::CompositionMode_SourceIn);
  tint_painter.fillRect(tinted_arrow.rect(), color);
  tint_painter.end();
  
  p.drawPixmap(0, 0, size, size, tinted_arrow);
  
  p.restore();
}

void HudRendererSP::createCompactArrowPixmap(int size) {
  compact_arrow_pixmap = loadPixmap("../assets/icons/arrow_simple.svg", QSize(size, size));
  compact_arrow_cached = !compact_arrow_pixmap.isNull();
  compact_arrow_size = size;
}

QString HudRendererSP::getRTIThreatTextShort(RTIThreatType type) const {
  auto it = kThreatTypeMap.find(type);
  return tr(it != kThreatTypeMap.end() ? it->second.text : kDefaultThreatInfo.text);
}

QString HudRendererSP::formatDistance(float distance_m) const {
  if (is_metric) {
    if (distance_m < 1000) {
      return QString("%1m").arg(static_cast<int>(distance_m));
    } else {
      return QString("%1km").arg(distance_m / 1000.0, 0, 'f', 1);
    }
  } else {
    float distance_yd = distance_m * 1.09361;  // Convert meters to yards
    // Use yards for distances under 500 yards (~0.28 miles), then miles beyond
    if (distance_yd < 500) {
      return QString("%1yd").arg(static_cast<int>(distance_yd));
    } else {
      float distance_mi = distance_yd / 1760.0;  // 1760 yards = 1 mile
      return QString("%1mi").arg(distance_mi, 0, 'f', 1);
    }
  }
}

void HudRendererSP::drawVTSCCoPilotNavArrow(QPainter &p, const QPointF &center, float size_px) const {
  p.save();
  p.setRenderHint(QPainter::Antialiasing, true);
  p.translate(center);

  const float s = std::max(14.0f, size_px);
  QPainterPath arrow;
  arrow.moveTo(0.0f, -1.15f * s);
  arrow.lineTo(0.58f * s, 0.34f * s);
  arrow.lineTo(0.18f * s, 0.18f * s);
  arrow.lineTo(0.0f, 1.08f * s);
  arrow.lineTo(-0.18f * s, 0.18f * s);
  arrow.lineTo(-0.58f * s, 0.34f * s);
  arrow.closeSubpath();

  QPainterPath shadow = arrow.translated(0.0f, 2.0f);
  p.setPen(Qt::NoPen);
  p.setBrush(QColor(0, 0, 0, 130));
  p.drawPath(shadow);

  QLinearGradient arrow_grad(QPointF(0.0f, -1.15f * s), QPointF(0.0f, 1.08f * s));
  arrow_grad.setColorAt(0.00, QColor(224, 245, 255, 250));
  arrow_grad.setColorAt(0.45, QColor(116, 198, 255, 245));
  arrow_grad.setColorAt(1.00, QColor(40, 126, 255, 238));
  p.setBrush(arrow_grad);
  p.setPen(QPen(QColor(255, 255, 255, 215), std::max(1.5f, 0.12f * s), Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  p.drawPath(arrow);

  p.setPen(QPen(QColor(255, 255, 255, 125), std::max(1.0f, 0.06f * s), Qt::SolidLine, Qt::RoundCap));
  p.drawLine(QPointF(0.0f, -0.55f * s), QPointF(0.0f, 0.58f * s));

  p.restore();
}

void HudRendererSP::drawCurveDirectionIcon(QPainter &p, const QRect &icon_rect, int direction) const {
  p.save();
  p.setRenderHint(QPainter::Antialiasing, true);

  // Unknown: draw a simple dot.
  if (direction != 1 && direction != 2) {
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(255, 255, 255, 220));
    p.drawEllipse(icon_rect.center(), icon_rect.width() / 8, icon_rect.height() / 8);
    p.restore();
    return;
  }

  // Build in a local coordinate system centered in icon_rect.
  p.translate(icon_rect.center());
  const bool is_right = (direction == 2);
  if (is_right) {
    p.scale(-1.0, 1.0);  // mirror horizontally for right curves
  }

  const float w = static_cast<float>(icon_rect.width());
  const float h = static_cast<float>(icon_rect.height());
  const float stroke = std::max(4.0f, std::min(w, h) * 0.12f);

  QPainterPath path;
  const QPointF p0(+0.35f * w, +0.30f * h);
  const QPointF c1(+0.10f * w, +0.05f * h);
  const QPointF c2(-0.15f * w, -0.05f * h);
  const QPointF p3(-0.25f * w, -0.32f * h);
  path.moveTo(p0);
  path.cubicTo(c1, c2, p3);

  p.setBrush(Qt::NoBrush);
  p.setPen(QPen(QColor(255, 255, 255, 235), stroke, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  p.drawPath(path);

  // Arrow head at the end of the curve.
  const QPointF tip = p3;
  const QPointF a(tip.x() + 0.18f * w, tip.y() + 0.03f * h);
  const QPointF b(tip.x() + 0.03f * w, tip.y() + 0.18f * h);
  QPainterPath arrow;
  arrow.moveTo(tip);
  arrow.lineTo(a);
  arrow.lineTo(b);
  arrow.closeSubpath();
  p.setPen(Qt::NoPen);
  p.setBrush(QColor(255, 255, 255, 235));
  p.drawPath(arrow);

  p.restore();
}

void HudRendererSP::drawVTSCCoPilotCurve(QPainter &p, const QRect &surface_rect) {
  if ((vtsc_copilot_tiles_.empty() && !vtsc_copilot_exiting_tile_active_) || vtsc_copilot_alpha_ < 0.01f) {
    return;
  }

  const QRect inner = surface_rect.adjusted(UI_BORDER_SIZE, UI_BORDER_SIZE, -UI_BORDER_SIZE, -UI_BORDER_SIZE);
  const QRect right_third(inner.left() + (2 * inner.width()) / 3, inner.top(), inner.width() / 3, inner.height());

  p.save();
  p.setRenderHint(QPainter::Antialiasing, true);

  const float kScale = std::clamp(vtsc_copilot_tuning_.scale * 0.525f, 0.70f, 1.70f);
  const auto now_tp = std::chrono::steady_clock::now();
  float dt = 0.0f;
  if (vtsc_copilot_last_draw_time_valid_) {
    dt = std::min(std::chrono::duration<float>(now_tp - vtsc_copilot_last_draw_time_).count(), 0.10f);
  }
  vtsc_copilot_last_draw_time_ = now_tp;
  vtsc_copilot_last_draw_time_valid_ = true;

  if (vtsc_copilot_stack_anim_progress_ < 1.0f) {
    vtsc_copilot_stack_anim_progress_ = std::min(1.0f, vtsc_copilot_stack_anim_progress_ + dt / 0.18f);
  }
  if (vtsc_copilot_exiting_tile_active_) {
    vtsc_copilot_exit_anim_progress_ = std::min(1.0f, vtsc_copilot_exit_anim_progress_ + dt / 0.26f);
    if (vtsc_copilot_exit_anim_progress_ >= 1.0f) {
      vtsc_copilot_exiting_tile_active_ = false;
    }
  }

  auto ease_out = [](float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const float inv = 1.0f - t;
    return 1.0f - inv * inv * inv;
  };
  auto ease_in = [](float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * t;
  };
  auto severity_text = [](int severity) -> QString {
    switch (severity) {
      case 3: return QStringLiteral("TIGHT");
      case 2: return QStringLiteral("MEDIUM");
      case 1: return QStringLiteral("GENTLE");
      default: return QStringLiteral("CURVE");
    }
  };
  auto direction_text = [](int direction) -> QString {
    switch (direction) {
      case 1: return QStringLiteral("LEFT");
      case 2: return QStringLiteral("RIGHT");
      default: return QStringLiteral("AHEAD");
    }
  };
  auto slot_for_tile = [&](uint32_t id) -> int {
    for (size_t i = 0; i < vtsc_copilot_prev_tiles_.size(); ++i) {
      if (vtsc_copilot_prev_tiles_[i].id == id) {
        return static_cast<int>(i);
      }
    }
    return -1;
  };
  auto speed_text = [&](float speed_mps) -> QString {
    const float display = std::max(0.0f, speed_mps) * (is_metric ? MS_TO_KPH : MS_TO_MPH);
    return QString::number(static_cast<int>(std::round(display)));
  };
  const QString speed_unit = is_metric ? QStringLiteral("km/h") : QStringLiteral("mph");

  const int bottom_safe = roundToInt(vtsc_copilot_tuning_.bottom_safe_px_at_scale1 * kScale) + roundToInt(12.0f * kScale);
  const float active_h = 228.0f * kScale;
  const float queued_h = 154.0f * kScale;
  const float stack_step = 96.0f * kScale;
  const float stack_w = std::min(static_cast<float>(right_third.width()) * 0.88f, 392.0f * kScale);
  const float stack_left = static_cast<float>(right_third.center().x()) - stack_w * 0.5f;
  const float active_bottom = static_cast<float>(inner.bottom()) - static_cast<float>(bottom_safe);
  const int visible_count = std::min<int>(4, static_cast<int>(vtsc_copilot_tiles_.size()));
  const float stack_top = active_bottom - active_h - std::max(0, visible_count - 1) * stack_step - 26.0f * kScale;
  const QRectF ambient_rect(stack_left - 24.0f * kScale, stack_top, stack_w + 48.0f * kScale,
                            active_h + std::max(0, visible_count - 1) * stack_step + 56.0f * kScale);

  QRadialGradient ambient_glow(ambient_rect.center(), std::max(ambient_rect.width(), ambient_rect.height()) * 0.58f);
  ambient_glow.setColorAt(0.00, QColor(245, 188, 92, 32));
  ambient_glow.setColorAt(0.34, QColor(16, 20, 26, 34));
  ambient_glow.setColorAt(1.00, QColor(0, 0, 0, 0));
  p.setPen(Qt::NoPen);
  p.setBrush(ambient_glow);
  p.drawEllipse(ambient_rect);

  auto draw_tile = [&](const VTSCCoPilotTileState &tile, const QRectF &rect, float opacity, bool active) {
    if (rect.width() < 40.0f || rect.height() < 40.0f || tile.points_m.size() < 2 || opacity <= 0.01f) {
      return;
    }

    p.save();
    p.setOpacity(opacity * vtsc_copilot_alpha_);

    const float radius = 22.0f * kScale;
    const float shadow_offset = (active ? 12.0f : 9.0f) * kScale;
    const float road_width = (active ? 15.5f : 12.0f) * kScale;
    const QColor accent = active ? QColor(245, 188, 92, 255) : QColor(178, 188, 202, 255);
    const QColor surface_top = active ? QColor(22, 28, 34, 238) : QColor(18, 23, 29, 224);
    const QColor surface_bottom = active ? QColor(10, 13, 18, 232) : QColor(8, 11, 16, 216);

    QPainterPath tile_path;
    tile_path.addRoundedRect(rect, radius, radius);

    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0, 0, 0, active ? 92 : 72));
    p.drawPath(tile_path.translated(0.0f, shadow_offset));

    QLinearGradient tile_grad(rect.topLeft(), rect.bottomLeft());
    tile_grad.setColorAt(0.00, surface_top);
    tile_grad.setColorAt(0.70, surface_bottom);
    tile_grad.setColorAt(1.00, QColor(5, 8, 12, active ? 238 : 220));
    p.setBrush(tile_grad);
    p.drawPath(tile_path);

    QPen border_pen(QColor(255, 255, 255, active ? 34 : 22), 1.4f * kScale);
    p.setBrush(Qt::NoBrush);
    p.setPen(border_pen);
    p.drawPath(tile_path);

    QLinearGradient accent_grad(rect.left(), rect.top(), rect.right(), rect.top());
    accent_grad.setColorAt(0.00, QColor(accent.red(), accent.green(), accent.blue(), active ? 0 : 0));
    accent_grad.setColorAt(0.22, QColor(accent.red(), accent.green(), accent.blue(), active ? 196 : 68));
    accent_grad.setColorAt(0.80, QColor(accent.red(), accent.green(), accent.blue(), active ? 128 : 28));
    accent_grad.setColorAt(1.00, QColor(accent.red(), accent.green(), accent.blue(), 0));
    QRectF accent_bar(rect.left() + 18.0f * kScale, rect.top() + 12.0f * kScale,
                      rect.width() - 36.0f * kScale, 3.0f * kScale);
    p.fillRect(accent_bar, accent_grad);

    QRectF glyph_rect = active
      ? rect.adjusted(22.0f * kScale, 26.0f * kScale, -22.0f * kScale, -70.0f * kScale)
      : rect.adjusted(20.0f * kScale, 24.0f * kScale, -20.0f * kScale, -34.0f * kScale);
    glyph_rect.setHeight(std::max(static_cast<float>(glyph_rect.height()), 28.0f * kScale));

    float fwd_max = 18.0f;
    float lat_max = 0.0f;
    for (const auto &pt : tile.points_m) {
      fwd_max = std::max(fwd_max, static_cast<float>(pt.x()));
      lat_max = std::max(lat_max, std::abs(static_cast<float>(pt.y())));
    }
    const float usable_h = std::max(static_cast<float>(glyph_rect.height()) - 8.0f * kScale, 16.0f * kScale);
    const float half_width = std::max(0.5f * static_cast<float>(glyph_rect.width()) - (road_width + 10.0f * kScale), 18.0f * kScale);
    const float meters_to_px_y = usable_h / std::max(18.0f, fwd_max);
    const float meters_to_px_x_fit = lat_max > 0.1f ? (half_width / lat_max) : meters_to_px_y;
    // Preserve turn severity by sharing a meter scale between forward and lateral axes.
    // A small lateral boost keeps gentle sweepers readable without turning them into hairpins.
    const float meters_to_px_x = std::min(meters_to_px_x_fit, meters_to_px_y * (active ? 1.18f : 1.12f));

    auto to_px = [&](const QPointF &pt) -> QPointF {
      const float x_px = static_cast<float>(glyph_rect.center().x()) - static_cast<float>(pt.y()) * meters_to_px_x;
      const float y_px = static_cast<float>(glyph_rect.bottom()) - std::clamp(static_cast<float>(pt.x()) * meters_to_px_y, 0.0f, usable_h);
      return QPointF(x_px, y_px);
    };

    std::vector<QPointF> glyph_pts;
    glyph_pts.reserve(tile.points_m.size());
    for (const auto &pt : tile.points_m) {
      glyph_pts.push_back(to_px(pt));
    }

    QPainterPath road_path;
    road_path.moveTo(glyph_pts.front());
    for (size_t i = 1; i < glyph_pts.size(); ++i) {
      road_path.lineTo(glyph_pts[i]);
    }

    QLinearGradient road_grad(glyph_pts.front(), glyph_pts.back());
    road_grad.setColorAt(0.00, QColor(255, 255, 255, 0));
    road_grad.setColorAt(0.14, QColor(255, 255, 255, active ? 238 : 212));
    road_grad.setColorAt(0.82, QColor(255, 255, 255, active ? 232 : 200));
    road_grad.setColorAt(1.00, QColor(255, 255, 255, 0));

    QLinearGradient glow_grad(glyph_pts.front(), glyph_pts.back());
    glow_grad.setColorAt(0.00, QColor(255, 255, 255, 0));
    glow_grad.setColorAt(0.16, QColor(255, 255, 255, active ? 74 : 52));
    glow_grad.setColorAt(0.84, QColor(255, 255, 255, active ? 56 : 40));
    glow_grad.setColorAt(1.00, QColor(255, 255, 255, 0));

    p.setPen(QPen(QColor(0, 0, 0, active ? 110 : 88), road_width + 9.0f * kScale, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.drawPath(road_path.translated(0.0f, 4.5f * kScale));
    p.setPen(QPen(QBrush(glow_grad), road_width + 7.0f * kScale, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.drawPath(road_path);
    p.setPen(QPen(QBrush(road_grad), road_width, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.drawPath(road_path);
    p.setPen(QPen(QColor(255, 255, 255, active ? 224 : 192), std::max(1.2f, 2.1f * kScale), Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.setOpacity(opacity * vtsc_copilot_alpha_ * (active ? 0.36f : 0.26f));
    p.drawPath(road_path);
    p.setOpacity(opacity * vtsc_copilot_alpha_);

    const QString title = QString("%1 %2").arg(severity_text(tile.severity), direction_text(tile.direction));
    const QString distance = formatDistance(std::max(0.0f, tile.distance_m));
    const QString eta = tile.time_to_s > 0.05f ? QString("%1s").arg(tile.time_to_s, 0, 'f', tile.time_to_s >= 10.0f ? 0 : 1)
                                               : QStringLiteral("NOW");

    p.setPen(QColor(accent.red(), accent.green(), accent.blue(), active ? 255 : 220));
    p.setFont(InterFont(roundToInt((active ? 20.0f : 17.0f) * kScale), QFont::DemiBold));
    p.drawText(QRectF(rect.left() + 22.0f * kScale, rect.top() + 18.0f * kScale,
                      rect.width() - 44.0f * kScale, 28.0f * kScale),
               Qt::AlignLeft | Qt::AlignVCenter, title);

    p.setPen(QColor(228, 234, 242, active ? 228 : 192));
    p.setFont(InterFont(roundToInt((active ? 18.0f : 16.0f) * kScale), QFont::DemiBold));
    p.drawText(QRectF(rect.left() + 22.0f * kScale, rect.top() + 18.0f * kScale,
                      rect.width() - 44.0f * kScale, 28.0f * kScale),
               Qt::AlignRight | Qt::AlignVCenter, eta);

    if (active) {
      p.setPen(QColor(232, 237, 244, 220));
      p.setFont(InterFont(roundToInt(18.0f * kScale), QFont::DemiBold));
      p.drawText(QRectF(rect.left() + 22.0f * kScale, rect.bottom() - 42.0f * kScale,
                        rect.width() * 0.48f, 22.0f * kScale),
                 Qt::AlignLeft | Qt::AlignVCenter, distance);

      p.setPen(QColor(255, 255, 255, 246));
      p.setFont(InterFont(roundToInt(34.0f * kScale), QFont::Bold));
      p.drawText(QRectF(rect.right() - 132.0f * kScale, rect.bottom() - 58.0f * kScale,
                        88.0f * kScale, 34.0f * kScale),
                 Qt::AlignRight | Qt::AlignVCenter, speed_text(tile.advisory_speed_mps));
      p.setPen(QColor(214, 221, 230, 200));
      p.setFont(InterFont(roundToInt(16.0f * kScale), QFont::DemiBold));
      p.drawText(QRectF(rect.right() - 126.0f * kScale, rect.bottom() - 31.0f * kScale,
                        112.0f * kScale, 18.0f * kScale),
                 Qt::AlignRight | Qt::AlignVCenter, speed_unit);
    } else {
      p.setPen(QColor(224, 229, 236, 176));
      p.setFont(InterFont(roundToInt(15.0f * kScale), QFont::DemiBold));
      p.drawText(QRectF(rect.left() + 22.0f * kScale, rect.bottom() - 30.0f * kScale,
                        rect.width() - 44.0f * kScale, 18.0f * kScale),
                 Qt::AlignLeft | Qt::AlignVCenter, distance);
    }

    p.restore();
  };

  const float shift_eased = ease_out(vtsc_copilot_stack_anim_progress_);
  for (size_t i = 0; i < vtsc_copilot_tiles_.size() && i < 4; ++i) {
    const auto &tile = vtsc_copilot_tiles_[i];
    const int previous_slot = slot_for_tile(tile.id);
    const float start_slot = previous_slot >= 0 ? static_cast<float>(previous_slot) : static_cast<float>(i) + 0.40f;
    const float display_slot = start_slot + (static_cast<float>(i) - start_slot) * shift_eased;
    const float depth = std::clamp(display_slot, 0.0f, 3.0f);
    const float width_scale = 1.0f - 0.06f * depth;
    const float height_scale = 1.0f - 0.08f * depth;
    const float tile_h = (i == 0 ? active_h : queued_h) * height_scale;
    const float tile_w = stack_w * width_scale;
    const float tile_bottom = active_bottom - display_slot * stack_step;
    const QRectF tile_rect(stack_left + 0.5f * (stack_w - tile_w), tile_bottom - tile_h, tile_w, tile_h);
    const float opacity = (i == 0 ? 1.0f : std::max(0.34f, 0.88f - 0.16f * depth));
    draw_tile(tile, tile_rect, opacity, i == 0);
  }

  if (vtsc_copilot_exiting_tile_active_ && vtsc_copilot_exiting_tile_.points_m.size() >= 2) {
    const float exit_t = ease_in(vtsc_copilot_exit_anim_progress_);
    const float drop_distance = 140.0f * kScale;
    const float tile_bottom = active_bottom + exit_t * drop_distance;
    const QRectF exit_rect(stack_left, tile_bottom - active_h, stack_w, active_h);

    p.save();
    p.translate(exit_rect.center());
    p.rotate(-3.5f * exit_t);
    p.translate(-exit_rect.center());
    draw_tile(vtsc_copilot_exiting_tile_, exit_rect, 1.0f - exit_t, true);
    p.restore();
  }

  p.restore();
}


QColor HudRendererSP::getRTIThreatBgColorByType(RTIThreatType type) const {
  auto it = kThreatTypeMap.find(type);
  return it != kThreatTypeMap.end() ? it->second.bg_color : kDefaultThreatInfo.bg_color;
}

static inline double shortestDelta(double from_deg, double to_deg) {
  return normalize180(to_deg - from_deg);
}

double HudRendererSP::smoothAngleForThreat(const std::string &id, double raw_angle_deg) const {
  QMutexLocker lock(&smoothed_angles_mutex_);
  double target = normalize180(raw_angle_deg);
  auto it = smoothed_angles_deg_.find(id);
  if (it == smoothed_angles_deg_.end()) {
    smoothed_angles_deg_.emplace(id, target);
    return target;
  }
  double current = it->second;
  double delta = shortestDelta(current, target);
  
  // Adaptive smoothing: faster for large changes, slower for small ones
  // This ensures 1-degree precision while maintaining smooth motion
  double alpha = 0.15;  // base smoothing factor
  if (std::abs(delta) > 45.0) {
    alpha = 0.3;  // faster response for large changes
  } else if (std::abs(delta) < 5.0) {
    alpha = 0.1;  // slower for fine adjustments (1-degree precision)
  }
  
  double next = normalize180(current + alpha * delta);
  
  // Snap to target if very close (within 1 degree) for precise pointing
  if (std::abs(delta) < 1.0) {
    next = target;
  }
  
  it->second = next;
  return next;
}

double HudRendererSP::smoothYForThreat(const std::string &id, double target_y, double initial_y) const {
  QMutexLocker lock(&smoothed_y_mutex_);
  auto it = smoothed_y_top_.find(id);
  if (it == smoothed_y_top_.end()) {
    smoothed_y_top_.emplace(id, initial_y);
    return initial_y;
  }
  double current = it->second;
  const double alpha = 0.25;  // vertical position smoothing factor
  double next = current + alpha * (target_y - current);
  it->second = next;
  return next;
}
