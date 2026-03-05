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
    return;
  }

  // Cache v_ego for ego-advance interpolation and speed-dependent lookahead.
  try {
    if (s.sm->valid("carState")) {
      vtsc_copilot_v_ego_mps_ = (*s.sm)["carState"].getCarState().getVEgo();
    }
  } catch (const std::exception&) {}

  // Grab road preview geometry — always accept updates, no kappa gating.
  try {
    if (s.sm->valid("longitudinalPlanSP")) {
      const auto lp_sp = (*s.sm)["longitudinalPlanSP"].getLongitudinalPlanSP();
      const auto vtsc = lp_sp.getVisionTurnSpeedControl();

      if (vtsc.getCurvePreviewValid()) {
        std::vector<QPointF> new_pts;
        auto pts = vtsc.getCurvePreviewPoints();
        new_pts.reserve(static_cast<size_t>(pts.size()));
        for (const auto &pt : pts) {
          new_pts.emplace_back(pt.getXFwdM(), pt.getYLeftM());
        }
        if (new_pts.size() >= 3) {
          // Detect whether geometry actually changed.  The producer throttles to
          // ~5 Hz but the planner re-publishes the same points at 20 Hz.  Resetting
          // ego-advance on every identical re-publish kills smooth interpolation.
          bool geom_changed = (new_pts.size() != vtsc_copilot_curve_points_m_.size());
          if (!geom_changed) {
            constexpr float geom_eps_m = 0.05f;
            const size_t mid = new_pts.size() / 2;
            const size_t last_idx = new_pts.size() - 1;
            for (size_t idx : {size_t(0), mid, last_idx}) {
              if (std::abs(new_pts[idx].x() - vtsc_copilot_curve_points_m_[idx].x()) > geom_eps_m ||
                  std::abs(new_pts[idx].y() - vtsc_copilot_curve_points_m_[idx].y()) > geom_eps_m) {
                geom_changed = true;
                break;
              }
            }
          }
          vtsc_copilot_curve_points_m_ = std::move(new_pts);
          if (geom_changed) {
            vtsc_copilot_ego_advance_m_ = 0.0f;
            vtsc_copilot_last_draw_time_valid_ = false;
          }
        }
      }
    }
  } catch (const std::exception&) {}

  // Always visible when we have road geometry. No fade — instant on/off.
  vtsc_copilot_alpha_ = (vtsc_copilot_curve_points_m_.size() >= 3) ? 1.0f : 0.0f;
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
  if (vtsc_copilot_curve_points_m_.size() < 2 || vtsc_copilot_alpha_ < 0.01f) {
    return;
  }

  const QRect inner = surface_rect.adjusted(UI_BORDER_SIZE, UI_BORDER_SIZE, -UI_BORDER_SIZE, -UI_BORDER_SIZE);
  const QRect right_third(inner.left() + (2 * inner.width()) / 3, inner.top(), inner.width() / 3, inner.height());

  p.save();
  p.setRenderHint(QPainter::Antialiasing, true);

  const float kScale = std::max(0.5f, vtsc_copilot_tuning_.scale);

  // Widget bounding box — right third, bottom-aligned.
  const int box_w = static_cast<int>(right_third.width() * 0.995f);
  const int box_h = std::min(static_cast<int>(inner.height() * 0.875f), 950);
  const int bottom_safe = static_cast<int>(vtsc_copilot_tuning_.bottom_safe_px_at_scale1 * kScale);
  const int box_left = right_third.center().x() - box_w / 2;
  const int box_top = std::max(inner.top(), inner.bottom() - bottom_safe - box_h + 1);
  const QRect box(box_left, box_top, box_w, box_h);

  // Full box for road rendering (no text labels).
  const int pad = static_cast<int>(vtsc_copilot_tuning_.pad_px_at_scale1 * kScale);
  const QRect curve_area(box.left() + pad, box.top() + pad, box.width() - 2 * pad, box.height() - 2 * pad);
  if (curve_area.height() < 40 || curve_area.width() < 40) {
    p.restore();
    return;
  }

  // 360-degree feathered halo backdrop.
  const float radius = 0.50f * std::min(static_cast<float>(box.width()), static_cast<float>(box.height()));
  QRadialGradient vignette(box.center(), radius);
  vignette.setColorAt(0.00, QColor::fromRgbF(0, 0, 0, 0.40));
  vignette.setColorAt(0.55, QColor::fromRgbF(0, 0, 0, 0.14));
  vignette.setColorAt(1.00, QColor::fromRgbF(0, 0, 0, 0.00));
  p.setPen(Qt::NoPen);
  p.setBrush(vignette);
  p.drawRect(box);

  // --- Ego-advance: smooth scrolling between producer updates ---
  {
    const auto now_tp = std::chrono::steady_clock::now();
    if (vtsc_copilot_last_draw_time_valid_) {
      const float dt = std::chrono::duration<float>(now_tp - vtsc_copilot_last_draw_time_).count();
      vtsc_copilot_ego_advance_m_ += vtsc_copilot_v_ego_mps_ * std::min(dt, 0.1f);
    }
    vtsc_copilot_last_draw_time_ = now_tp;
    vtsc_copilot_last_draw_time_valid_ = true;
  }
  const float ego_adv = vtsc_copilot_ego_advance_m_;

  // --- Speed-dependent lookahead ---
  constexpr float kMinSpeedMps = 8.94f;    // 20 mph
  constexpr float kMaxSpeedMps = 31.29f;   // 70 mph
  constexpr float kMinLookaheadS = 5.0f;
  constexpr float kMaxLookaheadS = 12.0f;
  constexpr float kMinLookaheadM = 20.0f;

  const float v_ego = vtsc_copilot_v_ego_mps_;
  const float t_speed = std::clamp((v_ego - kMinSpeedMps) / (kMaxSpeedMps - kMinSpeedMps), 0.0f, 1.0f);
  const float lookahead_s = kMinLookaheadS + t_speed * (kMaxLookaheadS - kMinLookaheadS);
  const float target_lookahead_m = std::max(kMinLookaheadM, lookahead_s * v_ego);

  constexpr float kLookaheadSmooth = 0.06f;
  vtsc_copilot_smoothed_lookahead_m_ += kLookaheadSmooth * (target_lookahead_m - vtsc_copilot_smoothed_lookahead_m_);
  const float lookahead_m = vtsc_copilot_smoothed_lookahead_m_;

  // --- Build ego-shifted, lookahead-clipped point set ---
  std::vector<QPointF> pts_m;
  pts_m.reserve(vtsc_copilot_curve_points_m_.size());
  QPointF prev_raw;
  bool have_prev = false;
  for (const auto &pt : vtsc_copilot_curve_points_m_) {
    const float xf_raw = static_cast<float>(pt.x());
    const float yl = static_cast<float>(pt.y());
    if (!std::isfinite(xf_raw) || !std::isfinite(yl)) { have_prev = false; continue; }
    const float xf = xf_raw - ego_adv;
    if (xf < 0.0f) {
      prev_raw = QPointF(xf_raw, yl);
      have_prev = true;
      continue;
    }
    if (xf > lookahead_m) {
      if (!pts_m.empty()) {
        const auto &last = pts_m.back();
        const float x_last = static_cast<float>(last.x());
        const float y_last = static_cast<float>(last.y());
        const float denom = xf - x_last;
        if (std::abs(denom) > 1e-6f) {
          const float frac = (lookahead_m - x_last) / denom;
          pts_m.emplace_back(lookahead_m, y_last + frac * (yl - y_last));
        }
      }
      break;
    }
    if (have_prev && pts_m.empty()) {
      const float x_prev = static_cast<float>(prev_raw.x()) - ego_adv;
      const float y_prev = static_cast<float>(prev_raw.y());
      const float denom = xf - x_prev;
      if (std::abs(denom) > 1e-6f) {
        const float frac = -x_prev / denom;
        pts_m.emplace_back(0.0f, y_prev + frac * (yl - y_prev));
      }
    }
    have_prev = false;
    pts_m.emplace_back(xf, yl);
  }

  if (pts_m.size() < 2) {
    p.restore();
    return;
  }

  // --- Auto-centering: smooth ego lateral offset to prevent drift at intersections ---
  const float ego_y_raw = static_cast<float>(pts_m.front().y());
  constexpr float kEgoLatSmooth = 0.12f;
  vtsc_copilot_smoothed_ego_y_left_ += kEgoLatSmooth * (ego_y_raw - vtsc_copilot_smoothed_ego_y_left_);
  const float center_y = vtsc_copilot_smoothed_ego_y_left_;

  // --- Scaling with auto-fit to prevent overflow ---
  // Road half-width in meters (must match the value used for edge computation below).
  constexpr float kRoadHalfWidthM = 3.0f;

  float x_max = 0.0f;
  float y_abs = 0.0f;
  for (const auto &pt : pts_m) {
    x_max = std::max(x_max, static_cast<float>(pt.x()));
    y_abs = std::max(y_abs, std::abs(static_cast<float>(pt.y()) - center_y));
  }
  x_max = std::max(10.0f, x_max);
  // Include road edge width in the lateral range so edges don't overflow.
  y_abs = std::max(1.0f, y_abs + kRoadHalfWidthM);

  const float fwd_scale = static_cast<float>(curve_area.height()) / x_max;
  // Lateral scale: proportional to forward for zoom, but clamped to prevent overflow.
  const float lat_scale_zoom = fwd_scale * 3.0f;
  const float lat_scale_fit = (0.45f * static_cast<float>(curve_area.width())) / y_abs;
  const float lat_scale = std::min(lat_scale_zoom, lat_scale_fit);

  auto toPx = [&](float x_fwd_m, float y_left_m) -> QPointF {
    const float t = std::clamp(x_fwd_m / x_max, 0.0f, 1.0f);
    const float persp = std::clamp(1.0f - 0.35f * t, 0.65f, 1.0f);
    // Center on smoothed ego lateral position — prevents drift at intersections.
    const float x_px = static_cast<float>(curve_area.center().x()) - (y_left_m - center_y) * lat_scale * persp;
    const float y_px = static_cast<float>(curve_area.bottom()) - x_fwd_m * fwd_scale;
    return QPointF(x_px, y_px);
  };

  // Convert centerline to pixel coordinates.
  std::vector<QPointF> px_pts;
  px_pts.reserve(pts_m.size());
  for (const auto &pt : pts_m) {
    px_pts.emplace_back(toPx(static_cast<float>(pt.x()), static_cast<float>(pt.y())));
  }

  p.setClipRect(box);

  // --- Road surface polygon: perspective-varying width, gradient fill ---
  // Render a filled road surface instead of a centerline-only stroke.
  // Edge offsets are computed in METER space and mapped through toPx so that
  // perspective is handled naturally — no self-intersection even on tight onramps.
  {
    // Road half-width in meters — must match kRoadHalfWidthM used in auto-fit above.
    // ~3m half = 6m total ≈ 1.5 lanes — visible but won't self-intersect
    // (even the tightest onramp has radius ~30m >> 3m).
    const float road_hw_m = kRoadHalfWidthM;

    std::vector<QPointF> left_edge_px, right_edge_px;
    left_edge_px.reserve(pts_m.size());
    right_edge_px.reserve(pts_m.size());

    for (size_t i = 0; i < pts_m.size(); i++) {
      // Tangent in meter space via central differences.
      QPointF tangent;
      if (i == 0) {
        tangent = (pts_m.size() > 1) ? (pts_m[1] - pts_m[0]) : QPointF(1.0, 0.0);
      } else if (i == pts_m.size() - 1) {
        tangent = pts_m[i] - pts_m[i - 1];
      } else {
        tangent = pts_m[i + 1] - pts_m[i - 1];
      }
      float len = std::sqrt(tangent.x() * tangent.x() + tangent.y() * tangent.y());
      if (len < 0.01f) len = 0.01f;

      // Left perpendicular in ego-local (x=fwd, y=left): (-dy, dx) / len.
      const float lnx = -tangent.y() / len;
      const float lny =  tangent.x() / len;

      const float xf = static_cast<float>(pts_m[i].x());
      const float yl = static_cast<float>(pts_m[i].y());

      left_edge_px.push_back(toPx(xf + lnx * road_hw_m, yl + lny * road_hw_m));
      right_edge_px.push_back(toPx(xf - lnx * road_hw_m, yl - lny * road_hw_m));
    }

    // Build the road surface as a raw polygon (no Catmull-Rom on edges —
    // splines overshoot catastrophically on tight bends).
    QPainterPath road_surface;
    if (!left_edge_px.empty()) {
      road_surface.moveTo(left_edge_px.front());
      for (size_t i = 1; i < left_edge_px.size(); i++) {
        road_surface.lineTo(left_edge_px[i]);
      }
      for (int i = static_cast<int>(right_edge_px.size()) - 1; i >= 0; --i) {
        road_surface.lineTo(right_edge_px[i]);
      }
      road_surface.closeSubpath();
    }

    // Fill with vertical gradient — opaque at ego, fading to transparent at horizon.
    QLinearGradient road_grad(
      QPointF(curve_area.center().x(), curve_area.bottom()),
      QPointF(curve_area.center().x(), curve_area.top())
    );
    road_grad.setColorAt(0.00, QColor(50, 55, 65, 140));
    road_grad.setColorAt(0.70, QColor(50, 55, 65, 70));
    road_grad.setColorAt(1.00, QColor(50, 55, 65, 0));
    p.setPen(Qt::NoPen);
    p.setBrush(road_grad);
    p.drawPath(road_surface);

    // Edge lines (raw polylines — no spline to avoid overshoot).
    QPainterPath left_line, right_line;
    if (!left_edge_px.empty()) {
      left_line.moveTo(left_edge_px.front());
      for (size_t i = 1; i < left_edge_px.size(); i++) left_line.lineTo(left_edge_px[i]);
    }
    if (!right_edge_px.empty()) {
      right_line.moveTo(right_edge_px.front());
      for (size_t i = 1; i < right_edge_px.size(); i++) right_line.lineTo(right_edge_px[i]);
    }

    // Edge glow.
    p.setPen(QPen(QColor(255, 255, 255, 12), 5.0 * kScale, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.setBrush(Qt::NoBrush);
    p.drawPath(left_line);
    p.drawPath(right_line);

    // Edge lines proper.
    p.setPen(QPen(QColor(255, 255, 255, 70), 1.5 * kScale, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.drawPath(left_line);
    p.drawPath(right_line);
  }

  // --- Ego dot: always at the bottom of the strip map ---
  if (!px_pts.empty()) {
    const QPointF dot_center = px_pts.front();
    const int kRoadMainWidth = std::max(1, static_cast<int>(vtsc_copilot_tuning_.road_main_width_px_at_scale1));
    const qreal dot_r = static_cast<qreal>(kRoadMainWidth * kScale) * 0.5;

    p.setPen(Qt::NoPen);
    p.setBrush(QColor(255, 59, 48, 50));
    p.drawEllipse(dot_center, dot_r + 3.0 * kScale, dot_r + 3.0 * kScale);

    p.setBrush(Qt::NoBrush);
    p.setPen(QPen(QColor(0, 0, 0, 200), 2.0 * kScale, Qt::SolidLine));
    p.drawEllipse(dot_center, dot_r, dot_r);

    p.setPen(Qt::NoPen);
    p.setBrush(QColor(255, 59, 48, 240));
    p.drawEllipse(dot_center, dot_r - 1.0 * kScale, dot_r - 1.0 * kScale);
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
