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
#include <QTransform>
#include <QTime>
#include <QMutexLocker>
#include <QLinearGradient>
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

// Strip map constants
static constexpr float kMileM = 1609.34f;
static constexpr float kStripForwardM = 0.85f * kMileM;
static constexpr float kStripBehindM = 0.15f * kMileM;

HudRendererSP::HudRendererSP() {
  // RTI state initialized with safe defaults
  // rti_enabled will be updated periodically in updateState()
}

void HudRendererSP::updateState(const UIState &s) {
  // Update base HUD state
  HudRenderer::updateState(s);
  
  // Update RTI parameters more frequently (every 5 frames = 250ms) to reduce race condition
  if (s.sm && s.sm->frame % 5 == 0) {
    rti_enabled = Params().getBool("RTIEnabled");  // Master switch
    rti_hud_enabled = Params().getBool("RTIHUDEnabled");  // HUD display switch
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

  updateStripMap(s);
}

void HudRendererSP::draw(QPainter &p, const QRect &surface_rect) {
  // Draw base HUD elements
  HudRenderer::draw(p, surface_rect);

  drawStripMap(p, surface_rect);
  
  // Draw RTI widget when enabled (multi-threat only)
  if (rti_enabled && rti_hud_enabled) {
    drawRTIThreatIndicatorMulti(p, surface_rect);
  }

  drawCompassRose(p, surface_rect);
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

void HudRendererSP::updateStripMap(const UIState &s) {
  strip_map_scene_.valid = false;
  if (!s.sm) return;

  const SubMaster &sm = *(s.sm);
  if (sm.rcv_frame("liveMapDataSP") <= s.scene.started_frame) return;

  const auto live_map = sm["liveMapDataSP"].getLiveMapDataSP();
  if (!live_map.getRoadGeometryValid()) return;

  if (sm.rcv_frame("gpsLocationExternal") == 0) return;
  const auto gps = sm["gpsLocationExternal"].getGpsLocationExternal();

  const double lat = gps.getLatitude();
  const double lon = gps.getLongitude();
  if (!std::isfinite(lat) || !std::isfinite(lon)) return;
  if (std::abs(lat) < 1e-6 && std::abs(lon) < 1e-6) return;

  float heading_deg = gps.getBearingDeg();
  bool heading_ok = std::isfinite(heading_deg) && heading_deg >= 0.0f && heading_deg <= 360.0f;
  if (!heading_ok) {
    const float road_dir = live_map.getCurrentRoadSegment().getRoadDirection();
    if (std::isfinite(road_dir)) {
      heading_deg = road_dir;
      heading_ok = true;
    }
  }

  if (!heading_ok) {
    heading_deg = smoothed_heading_deg_;
  }

  if (!strip_heading_initialized_) {
    smoothed_heading_deg_ = heading_deg;
    strip_heading_initialized_ = true;
  } else {
    const float delta = static_cast<float>(normalize180(heading_deg - smoothed_heading_deg_));
    smoothed_heading_deg_ = static_cast<float>(normalize180(smoothed_heading_deg_ + delta * 0.2f));
  }

  strip_map_scene_ = build_strip_map_scene(live_map, lat, lon, smoothed_heading_deg_, kStripForwardM, kStripBehindM);
}

void HudRendererSP::drawStripMap(QPainter &p, const QRect &surface_rect) {
  if (!strip_map_scene_.valid) return;

  const int map_width = std::clamp(static_cast<int>(surface_rect.width() * 0.22f), 300, 420);
  const int map_height = std::clamp(static_cast<int>(surface_rect.height() * 0.55f), 520, 760);
  const int right_margin = 60;
  const int map_x = surface_rect.width() - right_margin - map_width;
  const int map_y = (surface_rect.height() - map_height) / 2;
  const QRect map_rect(map_x, map_y, map_width, map_height);

  const float span_m = strip_map_scene_.forward_m + strip_map_scene_.behind_m;
  if (span_m <= 1e-3f) return;

  const float scale = static_cast<float>(map_rect.height()) / span_m;
  const QPointF ego_anchor(map_rect.center().x(), map_rect.y() + map_rect.height() * 0.85f);

  constexpr double kPi = 3.14159265358979323846;
  const double heading_rad = strip_map_scene_.heading_deg * kPi / 180.0;
  const double cos_h = std::cos(-heading_rad);
  const double sin_h = std::sin(-heading_rad);

  auto map_point = [&](const QPointF &pt) -> QPointF {
    const double rx = pt.x() * cos_h - pt.y() * sin_h;
    const double ry = pt.x() * sin_h + pt.y() * cos_h;
    return QPointF(ego_anchor.x() + rx * scale, ego_anchor.y() - ry * scale);
  };

  p.save();
  p.setRenderHint(QPainter::Antialiasing);
  p.setClipRect(map_rect);

  for (const auto &poly : strip_map_scene_.polylines) {
    if (poly.points_m.size() < 2) continue;

    QPainterPath path;
    const QPointF first = map_point(poly.points_m.front());
    path.moveTo(first);
    for (size_t i = 1; i < poly.points_m.size(); ++i) {
      path.lineTo(map_point(poly.points_m[i]));
    }

    if (poly.is_current_road) {
      QPen shadow(QColor(0, 0, 0, 60), 12.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
      p.setPen(shadow);
      p.setBrush(Qt::NoBrush);
      p.drawPath(path);

      QPen road(QColor(245, 245, 245, 230), 8.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
      p.setPen(road);
      p.drawPath(path);
    } else if (poly.is_stub) {
      const QPointF last = map_point(poly.points_m.back());
      QLinearGradient grad(first, last);
      grad.setColorAt(0.0, QColor(245, 245, 245, 0));
      grad.setColorAt(0.2, QColor(245, 245, 245, 210));
      grad.setColorAt(0.8, QColor(245, 245, 245, 210));
      grad.setColorAt(1.0, QColor(245, 245, 245, 0));

      QPen stub(QBrush(grad), 4.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
      p.setPen(stub);
      p.setBrush(Qt::NoBrush);
      p.drawPath(path);
    }
  }

  // Ego arrow
  QPainterPath arrow;
  const double arrow_h = 26.0;
  const double arrow_w = 18.0;
  arrow.moveTo(ego_anchor.x(), ego_anchor.y() - arrow_h * 0.6);
  arrow.lineTo(ego_anchor.x() - arrow_w * 0.5, ego_anchor.y() + arrow_h * 0.4);
  arrow.lineTo(ego_anchor.x() + arrow_w * 0.5, ego_anchor.y() + arrow_h * 0.4);
  arrow.closeSubpath();

  QPainterPath shadow = arrow.translated(0.0, 2.0);
  p.setPen(Qt::NoPen);
  p.setBrush(QColor(0, 0, 0, 90));
  p.drawPath(shadow);

  p.setBrush(QColor(245, 245, 245, 245));
  p.drawPath(arrow);

  p.restore();
}

void HudRendererSP::drawCompassRose(QPainter &p, const QRect &surface_rect) {
  const int rose_size = 192;
  const QSize default_size(172, 204);
  const QSize set_speed_size = is_metric ? QSize(200, 204) : default_size;
  const int set_speed_x = 60 + (default_size.width() - set_speed_size.width()) / 2;
  const int set_speed_y = 45;
  const QRect set_speed_rect(QPoint(set_speed_x, set_speed_y), set_speed_size);

  const int center_x = set_speed_rect.center().x();
  const int bottom_margin = 25;
  int rose_y = surface_rect.height() - bottom_margin - rose_size;

  if (rti_enabled && rti_hud_enabled && !rti_threats.empty()) {
    const int widget_width = 744;
    const int widget_height = 520;
    const int left_margin = 15;
    const int bottom_margin_rti = 15;
    const QRect rti_rect(left_margin, surface_rect.height() - widget_height - bottom_margin_rti, widget_width, widget_height);
    const int above_rti = rti_rect.y() - 15 - rose_size;
    if (above_rti < rose_y) rose_y = above_rti;
  }

  if (rose_y < 0) return;
  const QRect rose_rect(center_x - rose_size / 2, rose_y, rose_size, rose_size);

  p.save();
  p.setRenderHint(QPainter::Antialiasing);

  const QRectF ring_rect = rose_rect.adjusted(6, 6, -6, -6);
  p.setPen(QPen(QColor(245, 245, 245, 180), 3));
  p.setBrush(Qt::NoBrush);
  p.drawEllipse(ring_rect);

  const QPointF c = ring_rect.center();
  const double ring_r = ring_rect.width() * 0.5;
  const double tip_ext = 8.0;

  auto draw_tip = [&](float angle_deg, float base_w, const QColor &color) {
    constexpr double kPi = 3.14159265358979323846;
    const double rad = (angle_deg - 90.0) * kPi / 180.0;
    const double dx = std::cos(rad);
    const double dy = std::sin(rad);
    const QPointF tip(c.x() + (ring_r + tip_ext) * dx, c.y() + (ring_r + tip_ext) * dy);
    const QPointF base_center(c.x() + ring_r * dx, c.y() + ring_r * dy);
    const QPointF perp(-dy, dx);

    QPainterPath tri;
    tri.moveTo(tip);
    tri.lineTo(base_center + perp * (base_w * 0.5));
    tri.lineTo(base_center - perp * (base_w * 0.5));
    tri.closeSubpath();

    p.setPen(Qt::NoPen);
    p.setBrush(color);
    p.drawPath(tri);
  };

  draw_tip(0.0f, 18.0f, QColor(245, 245, 245, 230));
  draw_tip(90.0f, 14.0f, QColor(245, 245, 245, 180));
  draw_tip(180.0f, 14.0f, QColor(245, 245, 245, 180));
  draw_tip(270.0f, 14.0f, QColor(245, 245, 245, 180));

  p.setPen(QPen(QColor(245, 245, 245, 140), 2));
  p.drawLine(QPointF(c.x(), c.y() - ring_r * 0.6), QPointF(c.x(), c.y() + ring_r * 0.6));
  p.drawLine(QPointF(c.x() - ring_r * 0.6, c.y()), QPointF(c.x() + ring_r * 0.6, c.y()));

  p.setPen(Qt::NoPen);
  p.setBrush(QColor(245, 245, 245, 200));
  p.drawEllipse(c, 3.0, 3.0);

  p.restore();
}
