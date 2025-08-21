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
#include <limits>

#include "common/params.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/ui.h"  // For UI_FREQ

// Static polygons for threat icons to avoid per-frame creation
static const QPolygon police_car_body = QPolygon()
    << QPoint(10, 50) << QPoint(30, 30) << QPoint(70, 30)
    << QPoint(90, 50) << QPoint(90, 70) << QPoint(10, 70);

static const QPolygon accident_triangle = QPolygon()
    << QPoint(50, 20) << QPoint(80, 70) << QPoint(20, 70);

static const QPolygon construction_cone = QPolygon()
    << QPoint(40, 20) << QPoint(60, 20) << QPoint(70, 70) << QPoint(30, 70);

// Static color constants for performance
static const QColor kRtiColorCritical(255, 0, 0, 255);    // Red
static const QColor kRtiColorNear(255, 165, 0, 255);      // Orange
static const QColor kRtiColorNormal(255, 255, 0, 255);    // Yellow
static const QColor kRtiColorFar(150, 150, 150, 255);     // Gray

HudRendererSP::HudRendererSP() {
  // RTI state initialized with safe defaults
  // rti_enabled will be updated periodically in updateState()
  
  // Initialize cached font objects for performance
  // Increase base fonts ~50% for better readability in RTI widget
  threat_text_font = InterFont(53, QFont::DemiBold);   // was 35
  distance_font = InterFont(68, QFont::Bold);          // was 45
  speed_rec_font = InterFont(45, QFont::Normal);       // was 30
  icon_font_small = InterFont(60, QFont::Bold);        // was 40
  icon_font_large = InterFont(68, QFont::Bold);        // was 45
  
  // Create arrow pixmap once at startup
  createArrowPixmap();
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
      
      // Update ego GPS position and heading
      if (s.sm->valid("gpsLocationExternal") && s.sm->updated("gpsLocationExternal")) {
        const auto gps = (*s.sm)["gpsLocationExternal"].getGpsLocationExternal();
        ego_lat = gps.getLatitude();
        ego_lon = gps.getLongitude();
        ego_bearing = gps.getBearingDeg();  // True heading in degrees
        has_gps = true;
        
        // Calculate relative bearing to threat if we have both positions
        if (has_gps && rti_has_threat && 
            std::isfinite(rti_threat_lat) && std::isfinite(rti_threat_lon) &&
            std::isfinite(ego_lat) && std::isfinite(ego_lon)) {
          rti_relative_bearing = calculateRelativeBearing(
            ego_lat, ego_lon, rti_threat_lat, rti_threat_lon, ego_bearing
          );
        }
      }
      
    } catch (const std::exception& e) {
      // Handle any exceptions from message access safely - reset to safe state
      rti_has_threat = false;
      rti_threat_ahead = false;
      rti_active = false;
      has_gps = false;
    }
  }
}

void HudRendererSP::draw(QPainter &p, const QRect &surface_rect) {
  // Draw base HUD elements
  HudRenderer::draw(p, surface_rect);
  
  // Draw RTI widget when enabled
  // Shows placeholder when no threat, actual threat info when detected
  if (rti_enabled && rti_hud_enabled) {
    // Use multi-threat view
    drawRTIThreatIndicatorMulti(p, surface_rect);
  }
}

void HudRendererSP::drawRTIThreatIndicator(QPainter &p, const QRect &surface_rect) {
  // Position to bottom-align with lateral accel widget which is at (height - 72 - 15)
  // Lateral accel widget bottom = surface_rect.height() - 15
  // RTI widget should have same bottom position
  const int bottom_margin = 15;  // Same as lateral accel widget
  const int left_margin = 15;    // Same spacing from left as bottom
  
  // Increased size: enlarge widget to accommodate bigger fonts/arrows
  const int widget_width = 600;
  const int widget_height = 620;
  
  // Position with bottom alignment to lateral accel widget
  const int x_offset = left_margin;
  const int y_offset = surface_rect.height() - widget_height - bottom_margin;
  
  QRect rti_rect(x_offset, y_offset, widget_width, widget_height);
  
  // Determine widget state and colors
  // Show threats for situational awareness regardless of direction
  bool has_situational_threat = rti_has_threat;  // Always show threats for awareness
  bool has_active_threat = rti_threat_ahead && rti_has_threat;  // Only for urgent coloring
  QColor threat_color = has_active_threat ? getRTIThreatColor(rti_threat_distance) : QColor(150, 150, 150, 255);
  
  // Match header shade opacity (0.45 → 115 alpha) for consistency
  p.setPen(QPen(QColor(255, 255, 255, 75), 6));
  p.setBrush(QColor(0, 0, 0, 115));  // Changed from 166 to 115 to match header shade
  p.drawRoundedRect(rti_rect, 32, 32);
  
  // Draw threat-colored inner border if active
  if (has_active_threat) {
    p.setPen(QPen(threat_color, 3));
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(rti_rect.adjusted(3, 3, -3, -3), 29, 29);
  }
  
  if (has_situational_threat) {
    // Draw threat information for situational awareness - proportionally scaled
    QRect icon_rect(rti_rect.x() + 48, rti_rect.y() + 24, 120, 96);
    drawRTIThreatIcon(p, icon_rect, rti_threat_type);
    
    // Draw arrow and threat type text - increased font size (+50%)
    p.setFont(InterFont(63, QFont::Normal));
    p.setPen(threat_color);
    QString threat_text = getRTIThreatText(rti_threat_type);

    // Always draw directional arrow; compute bearing every frame when GPS available
    double arrow_angle = 0.0;
    if (has_gps && std::isfinite(rti_threat_lat) && std::isfinite(rti_threat_lon) &&
        std::isfinite(ego_lat) && std::isfinite(ego_lon)) {
      arrow_angle = calculateRelativeBearing(ego_lat, ego_lon, rti_threat_lat, rti_threat_lon, ego_bearing);
    } else {
      arrow_angle = angleForDirection(rti_direction);
    }
    // Draw arrow to the left of the threat text
    QRect arrow_rect(rti_rect.x() + 180, rti_rect.y() + 125, 96, 96);  // doubled from 48
    drawRTIArrow(p, arrow_rect, arrow_angle);
    // Draw threat text shifted to the right
    p.drawText(rti_rect.adjusted(50, 132, 0, 0), Qt::AlignTop | Qt::AlignHCenter, threat_text);
    
    // Draw distance
    p.setFont(distance_font);
    QString distance_text;
    if (is_metric) {
      if (rti_threat_distance < 1000) {
        distance_text = QString("%1m").arg(static_cast<int>(rti_threat_distance));
      } else {
        distance_text = QString("%1km").arg(rti_threat_distance / 1000.0, 0, 'f', 1);
      }
    } else {
      float distance_ft = rti_threat_distance * 3.28084;
      if (distance_ft < 1000) {
        distance_text = QString("%1ft").arg(static_cast<int>(distance_ft));
      } else {
        float distance_mi = distance_ft / 5280.0;
        distance_text = QString("%1mi").arg(distance_mi, 0, 'f', 1);
      }
    }
    p.drawText(rti_rect.adjusted(0, 186, 0, 0), Qt::AlignTop | Qt::AlignHCenter, distance_text);
    
    // Draw speed recommendation ONLY for active (ahead) threats
    if (has_active_threat && rti_active && std::abs(rti_recommended_speed - speed / (is_metric ? 3.6 : 2.237)) > 1.0) {
      p.setFont(speed_rec_font);
      p.setPen(QColor(255, 255, 255, 200));
      
      float rec_speed_display = rti_recommended_speed * (is_metric ? 3.6 : 2.237);
      QString speed_text = QString("↓ %1").arg(static_cast<int>(rec_speed_display));
      p.drawText(rti_rect.adjusted(0, 240, 0, 0), Qt::AlignTop | Qt::AlignHCenter, speed_text);
    }
  } else {
    // Draw placeholder same as header: top-left, same font/weight
    p.setFont(InterFont(59, QFont::DemiBold));  // +50%
    p.setPen(QColor(150, 150, 150, 200));
    p.drawText(rti_rect.adjusted(20, 15, -20, 0), Qt::AlignTop | Qt::AlignLeft, tr("RTI"));
  }
}

void HudRendererSP::drawRTIThreatIcon(QPainter &p, const QRect &icon_rect, RTIThreatType type) {
  // Save painter state
  p.save();
  
  // Set icon color
  QColor icon_color = getRTIThreatColor(rti_threat_distance);
  p.setPen(QPen(icon_color, 3));
  p.setBrush(Qt::NoBrush);
  
  // Draw simple vector icons based on threat type
  switch (type) {
    case cereal::RtiStateSP::ThreatType::POLICE:
    case cereal::RtiStateSP::ThreatType::POLICE_HIDING: {
      // Draw police car silhouette using cached polygon
      p.save();
      p.translate(icon_rect.topLeft());
      p.drawPolygon(police_car_body);
      
      // Draw light bar on top
      p.fillRect(35, 20, 30, 8, icon_color);
      p.restore();
      break;
    }
    
    case cereal::RtiStateSP::ThreatType::SPEED_TRAP:
    case cereal::RtiStateSP::ThreatType::SPEED_CAMERA: {
      // Draw camera icon
      p.drawRect(icon_rect.x() + 25, icon_rect.y() + 30, 50, 35);
      p.drawEllipse(QPoint(icon_rect.x() + 50, icon_rect.y() + 47), 12, 12);
      // Draw mount
      p.drawLine(icon_rect.x() + 50, icon_rect.y() + 65, icon_rect.x() + 50, icon_rect.y() + 75);
      break;
    }
    
    case cereal::RtiStateSP::ThreatType::ACCIDENT: {
      // Draw warning triangle using cached polygon
      p.save();
      p.translate(icon_rect.topLeft());
      p.drawPolygon(accident_triangle);
      p.restore();
      
      // Draw exclamation mark
      p.setFont(icon_font_small);
      p.drawText(icon_rect, Qt::AlignCenter, "!");
      break;
    }
    
    case cereal::RtiStateSP::ThreatType::CONSTRUCTION: {
      // Draw traffic cone using cached polygon
      p.save();
      p.translate(icon_rect.topLeft());
      p.drawPolygon(construction_cone);
      
      // Draw stripes
      p.drawLine(35, 35, 65, 35);
      p.drawLine(37, 50, 63, 50);
      p.restore();
      break;
    }
    
    case cereal::RtiStateSP::ThreatType::JAM: {
      // Draw multiple cars
      for (int i = 0; i < 3; i++) {
        int y_pos = icon_rect.y() + 20 + i * 20;
        p.drawRect(icon_rect.x() + 30, y_pos, 40, 15);
      }
      break;
    }
    
    default: {
      // Draw generic warning icon (circle with exclamation)
      p.drawEllipse(icon_rect.center(), 35, 35);
      p.setFont(icon_font_large);
      p.drawText(icon_rect, Qt::AlignCenter, "!");
      break;
    }
  }
  
  // Restore painter state
  p.restore();
}

QString HudRendererSP::getRTIThreatText(RTIThreatType type) const {
  switch (type) {
    case cereal::RtiStateSP::ThreatType::POLICE:
    case cereal::RtiStateSP::ThreatType::POLICE_HIDING:
      return tr("POLICE");
    case cereal::RtiStateSP::ThreatType::SPEED_TRAP:
    case cereal::RtiStateSP::ThreatType::SPEED_CAMERA:
      return tr("CAMERA");
    case cereal::RtiStateSP::ThreatType::ACCIDENT:
      return tr("ACCIDENT");
    case cereal::RtiStateSP::ThreatType::JAM:
      return tr("TRAFFIC");
    case cereal::RtiStateSP::ThreatType::CONSTRUCTION:
      return tr("WORK ZONE");
    case cereal::RtiStateSP::ThreatType::HAZARD:
    case cereal::RtiStateSP::ThreatType::SHOULDER_HAZARD:
      return tr("HAZARD");
    case cereal::RtiStateSP::ThreatType::ROAD_HAZARD:
      return tr("ROAD");
    case cereal::RtiStateSP::ThreatType::ROAD_CLOSED:
      return tr("CLOSED");
    default:
      return tr("ALERT");
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

void HudRendererSP::createArrowPixmap() {
  // Create arrow pointing up (0° = ahead)
  const int arrow_size = 96;  // doubled
  arrow_pixmap = QPixmap(arrow_size, arrow_size);
  arrow_pixmap.fill(Qt::transparent);
  
  QPainter painter(&arrow_pixmap);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setRenderHint(QPainter::SmoothPixmapTransform);
  
  // Create arrow path (pointing up)
  QPainterPath arrow;
  int center = arrow_size / 2;
  int arrow_length = arrow_size * 0.8;
  int arrow_width = arrow_size * 0.5;
  
  // Arrow tip (top)
  arrow.moveTo(center, center - arrow_length/2);
  
  // Right side of arrowhead
  arrow.lineTo(center + arrow_width/3, center - arrow_length/6);
  
  // Right side of shaft
  arrow.lineTo(center + arrow_width/6, center - arrow_length/6);
  arrow.lineTo(center + arrow_width/6, center + arrow_length/3);
  
  // Bottom of arrow
  arrow.lineTo(center - arrow_width/6, center + arrow_length/3);
  
  // Left side of shaft
  arrow.lineTo(center - arrow_width/6, center - arrow_length/6);
  
  // Left side of arrowhead
  arrow.lineTo(center - arrow_width/3, center - arrow_length/6);
  
  // Close path back to tip
  arrow.closeSubpath();
  
  // Fill with white (will be tinted when drawn)
  painter.fillPath(arrow, Qt::white);
  
  // Add border for better visibility
  painter.setPen(QPen(QColor(0, 0, 0, 100), 2));
  painter.drawPath(arrow);
  
  arrow_pixmap_cached = true;
}

void HudRendererSP::drawRTIArrow(QPainter &p, const QRect &arrow_rect, double relative_bearing) {
  if (!arrow_pixmap_cached) {
    createArrowPixmap();
  }
  
  p.save();
  
  // Set color based on threat distance
  QColor arrow_color = getRTIThreatColor(rti_threat_distance);
  
  // Apply rotation around center
  QTransform transform;
  transform.translate(arrow_rect.center().x(), arrow_rect.center().y());
  transform.rotate(relative_bearing);
  transform.translate(-arrow_rect.width()/2.0, -arrow_rect.height()/2.0);
  
  p.setTransform(transform, true);
  
  // Tint the arrow with threat color
  p.setCompositionMode(QPainter::CompositionMode_SourceOver);
  
  // Draw the arrow
  QPixmap tinted_arrow = arrow_pixmap;
  QPainter tint_painter(&tinted_arrow);
  tint_painter.setCompositionMode(QPainter::CompositionMode_SourceIn);
  tint_painter.fillRect(tinted_arrow.rect(), arrow_color);
  tint_painter.end();
  
  p.drawPixmap(0, 0, arrow_rect.width(), arrow_rect.height(), tinted_arrow);
  
  p.restore();
}

double HudRendererSP::calculateRelativeBearing(double ego_latitude, double ego_longitude, 
                                              double threat_latitude, double threat_longitude, 
                                              double ego_heading_deg) const {
  // Handle same location edge case
  if (std::abs(ego_latitude - threat_latitude) < 1e-9 && 
      std::abs(ego_longitude - threat_longitude) < 1e-9) {
    return 0.0;  // Default to ahead
  }
  
  // Earth radius in meters
  const double kEarthRadiusM = 6371000.0;
  
  // Convert to radians
  double phi1 = ego_latitude * M_PI / 180.0;
  double phi2 = threat_latitude * M_PI / 180.0;
  double lam1 = ego_longitude * M_PI / 180.0;
  double lam2 = threat_longitude * M_PI / 180.0;
  
  // Calculate differences
  double dphi = phi2 - phi1;
  double dlam = lam2 - lam1;
  
  // Local tangent plane approximation (accurate for < 10km)
  double avg_lat = (phi1 + phi2) / 2.0;
  double north = dphi * kEarthRadiusM;
  double east = dlam * kEarthRadiusM * std::cos(avg_lat);
  
  // Calculate world bearing (0° = north, clockwise positive)
  double world_bearing = std::atan2(east, north) * 180.0 / M_PI;
  if (world_bearing < 0) world_bearing += 360.0;
  
  // Calculate relative bearing
  double rel_bearing = world_bearing - ego_heading_deg;
  
  // Normalize to [-180, 180]
  while (rel_bearing > 180.0) rel_bearing -= 360.0;
  while (rel_bearing < -180.0) rel_bearing += 360.0;
  
  return rel_bearing;
}

void HudRendererSP::updateRTIThreats(const UIState &s) {
  rti_threats.clear();
  
  if (s.sm && s.sm->valid("rtiStateSP") && s.sm->updated("rtiStateSP")) {
    const auto rti_state = (*s.sm)["rtiStateSP"].getRtiStateSP();
    auto threats = rti_state.getThreats();
    
    // Process up to 4 threats, sorted by distance
    size_t num_threats = std::min(static_cast<size_t>(threats.size()), static_cast<size_t>(4));
    for (size_t i = 0; i < num_threats; i++) {
      auto threat = threats[i];
      
      RTIThreatInfo info;
      info.type = threat.getType();
      info.distance = threat.getDistance();
      info.latitude = threat.getLatitude();
      info.longitude = threat.getLongitude();
      info.has_location = std::isfinite(info.latitude) && std::isfinite(info.longitude);
      info.direction = threat.getDirection();
      
      // Do not cache relative bearing here; it will be computed per-frame in draw
      info.relative_bearing = std::numeric_limits<double>::quiet_NaN();
      
      rti_threats.push_back(info);
    }
    
    // Sort by distance in ASCENDING order (closest threat first)
    // This ensures the display shows:
    //   Top:    Closest threat (most urgent)
    //   Middle: Medium distance threats
    //   Bottom: Furthest threat (least urgent)
    std::sort(rti_threats.begin(), rti_threats.end(), 
              [](const RTIThreatInfo& a, const RTIThreatInfo& b) {
                return a.distance < b.distance;  // a < b means ascending (smallest/closest first)
              });
    
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
  }
}

void HudRendererSP::drawRTIThreatIndicatorMulti(QPainter &p, const QRect &surface_rect) {
  // Position to bottom-align with lateral accel widget
  const int bottom_margin = 15;
  const int left_margin = 15;
  
  // Expanded size to accommodate larger fonts/arrows
  const int widget_width = 620;   // slightly wider
  const int widget_height = 520;  // taller to fit 4 lines with 50% larger text
  
  const int x_offset = left_margin;
  const int y_offset = surface_rect.height() - widget_height - bottom_margin;
  
  QRect rti_rect(x_offset, y_offset, widget_width, widget_height);
  
  // Determine if any police threat is present to color the border red
  bool police_present = false;
  for (const auto &t : rti_threats) {
    if (t.type == cereal::RtiStateSP::ThreatType::POLICE ||
        t.type == cereal::RtiStateSP::ThreatType::POLICE_HIDING) {
      police_present = true;
      break;
    }
  }

  // Draw widget background with conditional border color
  p.setPen(QPen(police_present ? QColor(255, 0, 0, 200) : QColor(255, 255, 255, 75), 6));
  p.setBrush(QColor(0, 0, 0, 115));
  p.drawRoundedRect(rti_rect, 32, 32);
  
  // Check if we have active threats
  if (!rti_threats.empty()) {
    // Draw header (+50%)
    p.setFont(InterFont(59, QFont::DemiBold));
    p.setPen(QColor(255, 255, 255, 200));
    p.drawText(rti_rect.adjusted(20, 15, -20, 0), Qt::AlignTop | Qt::AlignLeft, "RTI");
    
    // Draw each threat on a single line in SORTED ORDER (closest to furthest)
    // Layout from top to bottom:
    //   Line 1 (top):    Closest/most urgent threat
    //   Line 2:          Second closest threat  
    //   Line 3:          Third closest threat
    //   Line 4 (bottom): Furthest threat (if 4 threats present)
    const int line_height = 84;  // ~55 * 1.5
    const int start_y = rti_rect.y() + 90;  // Start below bigger header
    const int max_threats = 4;  // Maximum threats to display
    
    int displayed = 0;
    for (const auto& threat : rti_threats) {  // Iterating in sorted order (closest first)
      if (displayed >= max_threats) break;
      
      int y_pos = start_y + (displayed * line_height);  // Each threat gets next line down
      
      // Get threat color based on distance
      QColor threat_color = getRTIThreatColor(threat.distance);
      
      // Draw compact single line: [arrow] TYPE • 0.5mi
      QRect line_rect(rti_rect.x() + 20, y_pos, widget_width - 40, line_height);
      
      // Draw arrow (larger; doubled). Compute bearing every frame using GPS when available.
      {
        double arrow_angle = 0.0;
        bool can_use_gps = has_gps && threat.has_location &&
                           std::isfinite(ego_lat) && std::isfinite(ego_lon) &&
                           std::isfinite(threat.latitude) && std::isfinite(threat.longitude);
        arrow_angle = can_use_gps ?
          calculateRelativeBearing(ego_lat, ego_lon, threat.latitude, threat.longitude, ego_bearing) :
          angleForDirection(threat.direction);
        QRect arrow_rect(line_rect.x(), line_rect.y() + 8, 64, 64);  // doubled from 32
        drawRTIArrowCompact(p, arrow_rect, arrow_angle, threat_color);
        line_rect.adjust(80, 0, 0, 0);  // shift text to accommodate larger arrow
      }

      // Draw threat type and distance on same line
      p.setFont(InterFont(54, QFont::Normal));  // +50%
      p.setPen(threat_color);
      
      QString threat_line = QString("%1 • %2")
        .arg(getRTIThreatTextShort(threat.type))
        .arg(formatDistance(threat.distance));
      
      p.drawText(line_rect, Qt::AlignVCenter | Qt::AlignLeft, threat_line);
      
      displayed++;
    }
    
    // If RTI is actively controlling speed, show recommendation at bottom
    if (rti_active && rti_recommended_speed > 0) {
      p.setFont(InterFont(48, QFont::Normal));  // +50%
      p.setPen(QColor(255, 255, 255, 180));
      
      float rec_speed_display = rti_recommended_speed * (is_metric ? 3.6 : 2.237);
      QString speed_text = QString("Target: %1 %2")
        .arg(static_cast<int>(rec_speed_display))
        .arg(is_metric ? "km/h" : "mph");
      
      p.drawText(rti_rect.adjusted(20, -40, -20, -10), 
                 Qt::AlignBottom | Qt::AlignLeft, speed_text);
    }
  } else {
    // Draw placeholder as top-left title identical to header style
    p.setFont(InterFont(59, QFont::DemiBold));  // +50%
    p.setPen(QColor(150, 150, 150, 200));
    p.drawText(rti_rect.adjusted(20, 15, -20, 0), Qt::AlignTop | Qt::AlignLeft, tr("RTI"));
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
  compact_arrow_pixmap = QPixmap(size, size);
  compact_arrow_pixmap.fill(Qt::transparent);
  
  QPainter painter(&compact_arrow_pixmap);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setRenderHint(QPainter::SmoothPixmapTransform);
  
  // Create arrow path (pointing up) - same proportions as main arrow
  QPainterPath arrow;
  int center = size / 2;
  int arrow_length = size * 0.8;
  int arrow_width = size * 0.5;
  
  // Arrow tip (top)
  arrow.moveTo(center, center - arrow_length/2);
  
  // Right side of arrowhead
  arrow.lineTo(center + arrow_width/3, center - arrow_length/6);
  
  // Right side of shaft
  arrow.lineTo(center + arrow_width/6, center - arrow_length/6);
  arrow.lineTo(center + arrow_width/6, center + arrow_length/3);
  
  // Bottom of arrow
  arrow.lineTo(center - arrow_width/6, center + arrow_length/3);
  
  // Left side of shaft
  arrow.lineTo(center - arrow_width/6, center - arrow_length/6);
  
  // Left side of arrowhead
  arrow.lineTo(center - arrow_width/3, center - arrow_length/6);
  
  // Close path back to tip
  arrow.closeSubpath();
  
  // Fill with white (will be tinted when drawn)
  painter.fillPath(arrow, Qt::white);
  
  // Add border for better visibility
  painter.setPen(QPen(QColor(0, 0, 0, 100), 2));
  painter.drawPath(arrow);
  
  compact_arrow_cached = true;
  compact_arrow_size = size;
}

QString HudRendererSP::getRTIThreatTextShort(RTIThreatType type) const {
  // Shorter labels for compact multi-threat view
  switch (type) {
    case cereal::RtiStateSP::ThreatType::POLICE:
    case cereal::RtiStateSP::ThreatType::POLICE_HIDING:
      return tr("POLICE");
    case cereal::RtiStateSP::ThreatType::SPEED_TRAP:
    case cereal::RtiStateSP::ThreatType::SPEED_CAMERA:
      return tr("CAMERA");
    case cereal::RtiStateSP::ThreatType::ACCIDENT:
      return tr("ACCIDENT");
    case cereal::RtiStateSP::ThreatType::JAM:
      return tr("TRAFFIC");
    case cereal::RtiStateSP::ThreatType::CONSTRUCTION:
      return tr("CONSTRUCTION");
    case cereal::RtiStateSP::ThreatType::HAZARD:
    case cereal::RtiStateSP::ThreatType::SHOULDER_HAZARD:
    case cereal::RtiStateSP::ThreatType::ROAD_HAZARD:
      return tr("HAZARD");
    case cereal::RtiStateSP::ThreatType::ROAD_CLOSED:
      return tr("CLOSED");
    default:
      return tr("ALERT");
  }
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
    if (distance_ft < 1000) {
      return QString("%1ft").arg(static_cast<int>(distance_ft));
    } else {
      float distance_mi = distance_ft / 5280.0;
      return QString("%1mi").arg(distance_mi, 0, 'f', 1);
    }
  }
}

double HudRendererSP::angleForDirection(cereal::RtiStateSP::Direction dir) const {
  using D = cereal::RtiStateSP::Direction;
  switch (dir) {
    case D::AHEAD:   return 0.0;    // forward
    case D::RIGHT:   return 90.0;   // right
    case D::BEHIND:  return 180.0;  // behind
    case D::LEFT:    return -90.0;  // left
    case D::UNKNOWN:
    default:         return 0.0;    // default to ahead
  }
}
