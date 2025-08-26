/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/hud.h"

#include <algorithm>
#include <cmath>
#include <QPainterPath>
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
static const QColor kRtiColorCritical(255, 0, 0, 255);    // Red
static const QColor kRtiColorNear(255, 165, 0, 255);      // Orange
static const QColor kRtiColorNormal(255, 255, 0, 255);    // Yellow
static const QColor kRtiColorFar(150, 150, 150, 255);     // Gray

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

static inline int clampToByteRange(int value) {
  return std::min(255, std::max(0, value));
}

static inline int scaleAndClampByte(int value, double scale) {
  return clampToByteRange(roundToInt(value * scale));
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
}

void HudRendererSP::draw(QPainter &p, const QRect &surface_rect) {
  // Draw base HUD elements
  HudRenderer::draw(p, surface_rect);
  
  // Draw RTI widget when enabled (multi-threat only)
  if (rti_enabled && rti_hud_enabled) {
    drawRTIThreatIndicatorMulti(p, surface_rect);
  }
}


QColor HudRendererSP::getRTIThreatColor(float distance) const {
  if (distance < 100) {
    // Critical - bright red
    return kRtiColorCritical;
  } else if (distance < 300) {
    // Near - orange
    return kRtiColorNear;
  } else if (distance < 1000) {
    // Normal - yellow
    return kRtiColorNormal;
  } else {
    // Far - gray (shouldn't normally display)
    return kRtiColorFar;
  }
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
      QColor threat_color = getRTIThreatColor(row.t->distance);       // distance-based hue
      QColor bg_color = getRTIThreatBgColorByType(row.t->type);       // type-based hue (background)
      bg_color.setAlpha(115);
      QColor border_color(255, 255, 255, 90);                         // default border

      // For 'on same road', derive a brighter/saturated variant from the background hue,
      // and use it for both border and text to match the observed design.
      QColor bright_text_color = QColor(255, 255, 255, 245);          // default text (white)
      if (row.t->on_same_road) {
        QColor hsl = bg_color.toHsl();
        int h, s, l, a; hsl.getHsl(&h, &s, &l, &a);
        s = scaleAndClampByte(s, 1.25); // more saturated
        l = scaleAndClampByte(l, 1.10); // slightly lighter
        QColor bright; bright.setHsl(h, s, l, 255);
        border_color = bright;
        border_color.setAlpha(240);
        bright_text_color = bright;
        bright_text_color.setAlpha(245);
      }

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
      drawRTIArrowCompact(p, arrow_rect, arrow_angle, threat_color);

      // Draw text: baseline white, or brighter color if on same road
      p.setPen(bright_text_color);
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
    float distance_ft = distance_m * 3.28084;
    // Use feet for distances under 0.25 miles (1320 ft), then miles beyond
    if (distance_ft < 1320) {
      return QString("%1ft").arg(static_cast<int>(distance_ft));
    } else {
      float distance_mi = distance_ft / 5280.0;
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
