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
#include <QFontMetrics>
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
  tuning.road_width_px = readParamFloatClamped(params, "VTSCHUD.RoadWidthPx", tuning.road_width_px, 4.0f, 60.0f);
  tuning.glow_width_px = readParamFloatClamped(params, "VTSCHUD.GlowWidthPx", tuning.glow_width_px, 2.0f, 40.0f);
  tuning.speed_font_px = readParamFloatClamped(params, "VTSCHUD.SpeedFontPx", tuning.speed_font_px, 20.0f, 120.0f);
  tuning.unit_font_px = readParamFloatClamped(params, "VTSCHUD.UnitFontPx", tuning.unit_font_px, 10.0f, 60.0f);

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

  // ===== VTSC Rally Co-Pilot: single-curve full-panel preview =====
  if (!vtsc_copilot_hud_enabled_ || !s.sm) {
    vtsc_copilot_alpha_ = 0.0f;
    vtsc_copilot_curve_points_m_.clear();
    vtsc_copilot_tiles_.clear();
    vtsc_copilot_exiting_tile_active_ = false;
    vtsc_copilot_last_nonempty_tiles_time_valid_ = false;
    vtsc_prev_tile_id_ = 0;
    vtsc_prev_advisory_speed_mps_ = 0.0f;
    vtsc_speed_flash_active_ = false;
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

  // Hold previous tiles briefly when data goes empty to prevent flicker.
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

  // Detect front-tile change for transition animation.
  const uint32_t new_front_id = new_tiles.empty() ? 0 : new_tiles.front().id;
  const uint32_t old_front_id = vtsc_copilot_tiles_.empty() ? 0 : vtsc_copilot_tiles_.front().id;
  if (new_front_id != old_front_id) {
    // Previous front tile exits
    if (old_front_id != 0 && !vtsc_copilot_tiles_.front().points_m.empty()) {
      vtsc_copilot_exiting_tile_ = vtsc_copilot_tiles_.front();
      vtsc_copilot_exiting_tile_active_ = true;
      vtsc_copilot_exit_anim_progress_ = 0.0f;
    }
    // New front tile enters
    vtsc_copilot_enter_anim_progress_ = 0.0f;
    // Reset speed flash on tile change
    vtsc_prev_tile_id_ = new_front_id;
    vtsc_prev_advisory_speed_mps_ = new_tiles.empty() ? 0.0f : new_tiles.front().advisory_speed_mps;
    vtsc_speed_flash_active_ = false;
  }
  vtsc_copilot_tiles_ = std::move(new_tiles);

  // Update convenience state from front tile.
  if (!vtsc_copilot_tiles_.empty()) {
    const auto &tile = vtsc_copilot_tiles_.front();
    vtsc_copilot_curve_distance_m_ = tile.distance_m;
    vtsc_copilot_curve_time_to_s_ = tile.time_to_s;
    vtsc_copilot_curve_kappa_max_ = tile.max_curvature;
    vtsc_copilot_curve_direction_ = tile.direction;
    vtsc_copilot_curve_severity_ = tile.severity;
    vtsc_target_speed_mps_ = tile.advisory_speed_mps;

    // Detect mid-turn speed adjustment for flash animation.
    if (tile.id == vtsc_prev_tile_id_ &&
        std::abs(tile.advisory_speed_mps - vtsc_prev_advisory_speed_mps_) > 0.15f) {
      vtsc_speed_flash_is_increase_ = tile.advisory_speed_mps > vtsc_prev_advisory_speed_mps_;
      vtsc_speed_flash_active_ = true;
      vtsc_speed_flash_start_ = std::chrono::steady_clock::now();
    }
    vtsc_prev_advisory_speed_mps_ = tile.advisory_speed_mps;
    vtsc_prev_tile_id_ = tile.id;
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
  // Draw VTSC rally co-pilot curve FIRST so it renders behind speed/set-speed
  if (vtsc_copilot_hud_enabled_ && vtsc_copilot_alpha_ > 0.01f) {
    drawVTSCCoPilotCurve(p, surface_rect);
  }

  // Draw base HUD elements (speed, set speed, etc.) on top of curve
  HudRenderer::draw(p, surface_rect);

  // Keep the readiness tree beneath the RTI stack when both occupy the left edge.
  drawSystemReadiness(p, surface_rect);

  // Draw RTI widget last so its cards render on top of the readiness tree.
  if (rti_enabled && rti_hud_enabled) {
    drawRTIThreatIndicatorMulti(p, surface_rect);
  }
}


void HudRendererSP::drawSystemReadiness(QPainter &p, const QRect &surface_rect) {
  if (subsystem_statuses_.empty()) return;

  p.save();
  p.setRenderHint(QPainter::Antialiasing, true);

  const float opacity = std::max(readiness_opacity_, 0.65f);
  const bool show_labels = true;

  const QFont subsystem_font = InterFont(27, QFont::DemiBold);
  const QFont master_font = InterFont(27, QFont::Bold);
  const QFontMetrics subsystem_metrics(subsystem_font);
  const QFontMetrics master_metrics(master_font);

  // Dot sizing and layout
  const int dot_r = 6;
  const int master_r = 9;
  const int spacing = subsystem_metrics.height() + 4;
  const int master_gap = std::max(10, master_metrics.height() / 4);
  const int label_gap = 10;
  const int pill_pad = 10;
  const int pill_left = surface_rect.left() + 12;
  const int x_center = pill_left + 18;

  const int n = static_cast<int>(subsystem_statuses_.size());
  const int legacy_col_h = (n - 1) * spacing + 2 * dot_r + master_gap + 2 * master_r;
  const int master_y = (surface_rect.height() - legacy_col_h) / 2 + master_r;
  const int master_to_first_spacing = std::max(master_r + dot_r,
                                               (master_metrics.height() + subsystem_metrics.height()) / 2) + master_gap;
  const int first_subsystem_y = master_y + master_to_first_spacing;

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
  int max_label_w = master_metrics.horizontalAdvance(QStringLiteral("MSTR"));
  for (const auto &[name, st] : subsystem_statuses_) {
    (void)st;
    max_label_w = std::max(max_label_w, subsystem_metrics.horizontalAdvance(QString::fromStdString(name)));
  }
  const int label_x = x_center + master_r + label_gap;
  // Draw subsystem dots (bottom to top: index 0 at bottom) below the master row.
  for (int i = 0; i < n; i++) {
    const auto &[name, st] = subsystem_statuses_[i];
    QColor c = colorForStatus(st);

    int y = first_subsystem_y + (n - 1 - i) * spacing;

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
      p.setFont(subsystem_font);
      c.setAlphaF(0.9 * opacity);
      p.setPen(c);
      QRect text_rect(label_x, y - subsystem_metrics.height() / 2,
                      max_label_w + pill_pad, subsystem_metrics.height());
      p.drawText(text_rect, Qt::AlignVCenter | Qt::AlignLeft, QString::fromStdString(name));
    }
  }

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
    p.setFont(master_font);
    master_c.setAlphaF(0.9 * opacity);
    p.setPen(master_c);
    QRect text_rect(label_x, master_y - master_metrics.height() / 2,
                    max_label_w + pill_pad, master_metrics.height());
    p.drawText(text_rect, Qt::AlignVCenter | Qt::AlignLeft, QStringLiteral("MSTR"));
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

  // === Layout: right 33% of usable area, below steering wheel exclusion zone ===
  const QRect inner = surface_rect.adjusted(UI_BORDER_SIZE, UI_BORDER_SIZE, -UI_BORDER_SIZE, -UI_BORDER_SIZE);
  constexpr int kSteeringWheelClearance = 242;  // UI_BORDER_SIZE(30) + btn_size(192) + 20px padding
  constexpr int kBottomMargin = 40;
  constexpr int kSidePad = 30;
  const int panel_left = inner.left() + (2 * inner.width()) / 3;
  const QRect panel(panel_left, kSteeringWheelClearance,
                    inner.right() - panel_left - kSidePad,
                    inner.bottom() - kSteeringWheelClearance - kBottomMargin);
  if (panel.width() < 80 || panel.height() < 80) return;

  p.save();
  p.setRenderHint(QPainter::Antialiasing, true);

  // === Timing ===
  const auto now_tp = std::chrono::steady_clock::now();
  float dt = 0.0f;
  if (vtsc_copilot_last_draw_time_valid_) {
    dt = std::min(std::chrono::duration<float>(now_tp - vtsc_copilot_last_draw_time_).count(), 0.10f);
  }
  vtsc_copilot_last_draw_time_ = now_tp;
  vtsc_copilot_last_draw_time_valid_ = true;

  // Advance enter/exit animations.
  if (vtsc_copilot_enter_anim_progress_ < 1.0f) {
    vtsc_copilot_enter_anim_progress_ = std::min(1.0f, vtsc_copilot_enter_anim_progress_ + dt / 0.22f);
  }
  if (vtsc_copilot_exiting_tile_active_) {
    vtsc_copilot_exit_anim_progress_ = std::min(1.0f, vtsc_copilot_exit_anim_progress_ + dt / 0.26f);
    if (vtsc_copilot_exit_anim_progress_ >= 1.0f) {
      vtsc_copilot_exiting_tile_active_ = false;
    }
  }

  auto ease_out = [](float t) -> float {
    t = std::clamp(t, 0.0f, 1.0f);
    const float inv = 1.0f - t;
    return 1.0f - inv * inv * inv;
  };

  // === Rally HUD backdrop: full-height strip with an in-panel left-edge fade ===
  // Keep the right/top/bottom edges flush to the screen border, and fade the backdrop
  // out to transparent as it reaches the rally HUD's left edge.
  {
    const int fade_dist = std::min(static_cast<int>((UI_HEADER_HEIGHT / 2.5f) * 3.0f), panel.width());
    const int backdrop_left = panel.left();
    const int backdrop_right = surface_rect.right() + 1;
    const int backdrop_width = std::max(1, backdrop_right - backdrop_left);
    const int fade_end_x = std::min(backdrop_right, backdrop_left + std::max(1, fade_dist));
    QLinearGradient vignette(backdrop_left, 0, fade_end_x, 0);
    vignette.setColorAt(0.0, QColor::fromRgbF(0, 0, 0, 0));
    vignette.setColorAt(1.0, QColor::fromRgbF(0, 0, 0, 0.45));
    p.setPen(Qt::NoPen);
    p.setBrush(vignette);
    p.drawRect(QRect(backdrop_left, surface_rect.top(), backdrop_width, surface_rect.height()));
  }

  // === Pick which tile to display (single tile, apex-flip for linked curves) ===
  const VTSCCoPilotTileState *display_tile = nullptr;
  if (!vtsc_copilot_tiles_.empty()) {
    display_tile = &vtsc_copilot_tiles_.front();
    // Linked-curve apex flip: if we're inside the first curve (distanceM <= 0) and past
    // the approximate midpoint, and there's a next curve, flip to it.
    if (vtsc_copilot_tiles_.size() > 1 && display_tile->distance_m <= 0.0f) {
      float fwd_extent = 0.0f;
      for (const auto &pt : display_tile->points_m) {
        fwd_extent = std::max(fwd_extent, static_cast<float>(pt.x()));
      }
      if (fwd_extent > 0.1f && std::abs(display_tile->distance_m) > fwd_extent * 0.5f) {
        display_tile = &vtsc_copilot_tiles_[1];
      }
    }
  }

  // === Helper: draw a single curve segment in the panel ===
  const float road_w = vtsc_copilot_tuning_.road_width_px * 1.5f;
  const float glow_extra = vtsc_copilot_tuning_.glow_width_px * 1.5f;

  auto draw_curve = [&](const VTSCCoPilotTileState &tile, float opacity) {
    if (tile.points_m.size() < 2 || opacity <= 0.01f) return;

    p.save();
    p.setOpacity(opacity * vtsc_copilot_alpha_);

    // Compute geometry bounding box.
    float fwd_min = std::numeric_limits<float>::max(), fwd_max = 0.0f;
    float lat_min = std::numeric_limits<float>::max(), lat_max = std::numeric_limits<float>::lowest();
    for (const auto &pt : tile.points_m) {
      fwd_min = std::min(fwd_min, static_cast<float>(pt.x()));
      fwd_max = std::max(fwd_max, static_cast<float>(pt.x()));
      lat_min = std::min(lat_min, static_cast<float>(pt.y()));
      lat_max = std::max(lat_max, static_cast<float>(pt.y()));
    }
    const float fwd_range = std::max(fwd_max - fwd_min, 1.0f);
    const float lat_range = std::max(lat_max - lat_min, 0.1f);
    // Scale to fill panel with padding. Forward and lateral axes use separate scales:
    // the lateral axis gets a modest boost (up to 1.5x the forward scale) so that
    // gentle sweepers show visible curvature without distorting into hairpins.
    const float pad = road_w + glow_extra + 30.0f;
    const float usable_w = std::max(static_cast<float>(panel.width()) - 2.0f * pad, 40.0f);
    const float usable_h = std::max(static_cast<float>(panel.height()) - 2.0f * pad, 40.0f);
    const float scale_fwd = usable_h / fwd_range;
    const float scale_lat_fit = lat_range > 0.1f ? (usable_w / lat_range) : scale_fwd;
    const float m_to_px_fwd = scale_fwd;
    const float m_to_px_lat = std::min(scale_lat_fit, scale_fwd * 1.5f);

    // Center the curve geometry within the panel.
    const float rendered_w = lat_range * m_to_px_lat;
    const float offset_x = static_cast<float>(panel.center().x()) - rendered_w * 0.5f;
    const float offset_y = static_cast<float>(panel.bottom()) - pad;

    // yLeftM is positive-left: negate so left curves render on the left side of the glyph.
    auto to_px = [&](const QPointF &pt) -> QPointF {
      const float px_x = offset_x + (lat_max - static_cast<float>(pt.y())) * m_to_px_lat;
      const float px_y = offset_y - (static_cast<float>(pt.x()) - fwd_min) * m_to_px_fwd;
      return QPointF(px_x, px_y);
    };

    // Build pixel path.
    std::vector<QPointF> pts;
    pts.reserve(tile.points_m.size());
    for (const auto &pt : tile.points_m) {
      pts.push_back(to_px(pt));
    }

    auto smooth_pts = [](std::vector<QPointF> in) {
      if (in.size() < 3) return in;

      constexpr int kChaikinPasses = 3;
      for (int pass = 0; pass < kChaikinPasses; ++pass) {
        if (in.size() < 3) break;
        std::vector<QPointF> next;
        next.reserve(in.size() * 2);
        next.push_back(in.front());
        for (size_t i = 0; i + 1 < in.size(); ++i) {
          const QPointF &a = in[i];
          const QPointF &b = in[i + 1];
          const QPointF q = a * 0.75 + b * 0.25;
          const QPointF r = a * 0.25 + b * 0.75;
          next.push_back(q);
          next.push_back(r);
        }
        next.push_back(in.back());
        in.swap(next);
      }
      return in;
    };

    pts = smooth_pts(std::move(pts));

    QPainterPath road_path;
    road_path.moveTo(pts.front());
    if (pts.size() == 2) {
      road_path.lineTo(pts.back());
    } else {
      for (size_t i = 1; i + 1 < pts.size(); ++i) {
        const QPointF mid = (pts[i] + pts[i + 1]) * 0.5;
        road_path.quadTo(pts[i], mid);
      }
      road_path.quadTo(pts[pts.size() - 2], pts.back());
    }

    // Road gradient: fade at entry (bottom) and exit (top).
    QLinearGradient road_grad(pts.front(), pts.back());
    road_grad.setColorAt(0.00, QColor(255, 255, 255, 0));
    road_grad.setColorAt(0.10, QColor(255, 255, 255, 240));
    road_grad.setColorAt(0.85, QColor(255, 255, 255, 235));
    road_grad.setColorAt(1.00, QColor(255, 255, 255, 0));

    QLinearGradient glow_grad(pts.front(), pts.back());
    glow_grad.setColorAt(0.00, QColor(255, 255, 255, 0));
    glow_grad.setColorAt(0.12, QColor(255, 255, 255, 60));
    glow_grad.setColorAt(0.88, QColor(255, 255, 255, 50));
    glow_grad.setColorAt(1.00, QColor(255, 255, 255, 0));

    // 1) Glow
    p.setPen(QPen(QBrush(glow_grad), road_w + glow_extra, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.drawPath(road_path);

    // 2) Main road stroke
    p.setPen(QPen(QBrush(road_grad), road_w, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.drawPath(road_path);

    // 3) Bright center highlight
    p.setOpacity(opacity * vtsc_copilot_alpha_ * 0.30f);
    p.setPen(QPen(QColor(255, 255, 255, 210), std::max(1.5f, road_w * 0.14f), Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.drawPath(road_path);
    p.setOpacity(opacity * vtsc_copilot_alpha_);

    p.restore();
  };

  // === Draw exiting tile (cross-fade out) ===
  if (vtsc_copilot_exiting_tile_active_ && vtsc_copilot_exiting_tile_.points_m.size() >= 2) {
    const float exit_opacity = 1.0f - ease_out(vtsc_copilot_exit_anim_progress_);
    draw_curve(vtsc_copilot_exiting_tile_, exit_opacity);
  }

  // === Draw active tile (cross-fade in) ===
  if (display_tile && display_tile->points_m.size() >= 2) {
    const float enter_opacity = ease_out(vtsc_copilot_enter_anim_progress_);
    draw_curve(*display_tile, enter_opacity);

    // === Speed recommendation in lower-right of panel ===
    const float speed_mps = display_tile->advisory_speed_mps;
    const float display_speed = std::max(0.0f, speed_mps) * (is_metric ? MS_TO_KPH : MS_TO_MPH);
    const QString speed_str = QString::number(static_cast<int>(std::round(display_speed)));
    const QString unit_str = is_metric ? QStringLiteral("km/h") : QStringLiteral("mph");

    // Determine speed text color (white, or two-phase flash green/red).
    // Phase 1 (0–350ms): snap to bright flash color, hold at peak.
    // Phase 2 (350ms–1.35s): ease-out fade from flash color back to white.
    QColor speed_color(255, 255, 255, 240);
    if (vtsc_speed_flash_active_) {
      const float flash_elapsed = std::chrono::duration<float>(now_tp - vtsc_speed_flash_start_).count();
      constexpr float kPeakDuration = 0.35f;   // bright hold
      constexpr float kFadeDuration = 1.00f;    // fade back to white
      constexpr float kTotalDuration = kPeakDuration + kFadeDuration;
      if (flash_elapsed >= kTotalDuration) {
        vtsc_speed_flash_active_ = false;
      } else {
        const QColor flash_base = vtsc_speed_flash_is_increase_
          ? QColor(0, 220, 80, 240)   // green for bump
          : QColor(240, 60, 60, 240);  // red for drop
        if (flash_elapsed < kPeakDuration) {
          // Phase 1: hold at full flash color
          speed_color = flash_base;
        } else {
          // Phase 2: ease-out fade from flash color → white
          const float fade_t = ease_out((flash_elapsed - kPeakDuration) / kFadeDuration);
          speed_color = QColor(
            flash_base.red()   + static_cast<int>(fade_t * (255 - flash_base.red())),
            flash_base.green() + static_cast<int>(fade_t * (255 - flash_base.green())),
            flash_base.blue()  + static_cast<int>(fade_t * (255 - flash_base.blue())),
            240);
        }
      }
    }

    p.setOpacity(enter_opacity * vtsc_copilot_alpha_);
    const int speed_font = roundToInt(vtsc_copilot_tuning_.speed_font_px);
    const int unit_font = roundToInt(vtsc_copilot_tuning_.unit_font_px);
    const int text_right_pad = 24;
    const int text_bottom_pad = 20;

    // Speed number
    p.setPen(speed_color);
    p.setFont(InterFont(speed_font, QFont::Bold));
    QRectF speed_rect(panel.right() - 200.0f, panel.bottom() - text_bottom_pad - speed_font - unit_font - 6,
                      200.0f - text_right_pad, static_cast<float>(speed_font) + 4.0f);
    p.drawText(speed_rect, Qt::AlignRight | Qt::AlignBottom, speed_str);

    // Unit label
    p.setPen(QColor(210, 218, 228, 190));
    p.setFont(InterFont(unit_font, QFont::DemiBold));
    QRectF unit_rect(panel.right() - 200.0f, speed_rect.bottom() + 2.0f,
                     200.0f - text_right_pad, static_cast<float>(unit_font) + 4.0f);
    p.drawText(unit_rect, Qt::AlignRight | Qt::AlignTop, unit_str);
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
