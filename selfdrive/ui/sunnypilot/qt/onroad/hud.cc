/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/hud.h"

#include <cmath>

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
  threat_text_font = InterFont(35, QFont::DemiBold);
  distance_font = InterFont(45, QFont::Bold);
  speed_rec_font = InterFont(30, QFont::Normal);
  icon_font_small = InterFont(40, QFont::Bold);
  icon_font_large = InterFont(45, QFont::Bold);
}

void HudRendererSP::updateState(const UIState &s) {
  // Update base HUD state
  HudRenderer::updateState(s);
  
  // Update RTI enabled parameter periodically (once per second)
  if (s.sm && s.sm->frame % UI_FREQ == 0) {
    rti_enabled = Params().getBool("RTIEnabled");
  }
  
  // Check for stale RTI data (1Hz message, timeout after 3 seconds)
  if (s.sm && (s.sm->frame - s.sm->rcv_frame("rtiStateSP")) > 3 * UI_FREQ) {
    rti_threat_ahead = false;
    rti_active = false;
    rti_threat_type = RTIThreatType::NONE;
  }
  
  // Update RTI state from messages
  if (s.sm && s.sm->updated("rtiStateSP")) {
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
      
      // Map threat type enum to our local enum using direct mapping
      auto threat_type_enum = threat.getType();
      switch (threat_type_enum) {
        case cereal::RtiStateSP::ThreatType::POLICE:
        case cereal::RtiStateSP::ThreatType::POLICE_HIDING:
          rti_threat_type = RTIThreatType::POLICE; break;
        case cereal::RtiStateSP::ThreatType::SPEED_TRAP:
        case cereal::RtiStateSP::ThreatType::SPEED_CAMERA:
          rti_threat_type = RTIThreatType::SPEED_TRAP; break;
        case cereal::RtiStateSP::ThreatType::ACCIDENT:
          rti_threat_type = RTIThreatType::ACCIDENT; break;
        case cereal::RtiStateSP::ThreatType::JAM:
          rti_threat_type = RTIThreatType::TRAFFIC_JAM; break;
        case cereal::RtiStateSP::ThreatType::CONSTRUCTION:
          rti_threat_type = RTIThreatType::CONSTRUCTION; break;
        case cereal::RtiStateSP::ThreatType::HAZARD:
        case cereal::RtiStateSP::ThreatType::SHOULDER_HAZARD:
          rti_threat_type = RTIThreatType::OBJECT_HAZARD; break;
        case cereal::RtiStateSP::ThreatType::ROAD_CLOSED:
          rti_threat_type = RTIThreatType::ROAD_CLOSED; break;
        case cereal::RtiStateSP::ThreatType::ROAD_HAZARD:
          rti_threat_type = RTIThreatType::ROAD_HAZARD; break;
        default:
          rti_threat_type = RTIThreatType::OTHER; break;
      }
      
      // Validate and set confidence with bounds checking
      float raw_confidence = threat.getConfidence();
      rti_threat_confidence = (std::isfinite(raw_confidence) && raw_confidence >= 0.0 && raw_confidence <= 1.0) ?
                              raw_confidence : 0.0;
    } else {
      rti_threat_type = RTIThreatType::NONE;
      rti_threat_confidence = 0.0;
    }
  }
  
  // Check if RTI is actively controlling speed (from longitudinal planner)
  if (s.sm && s.sm->updated("longitudinalPlanSP")) {
    // RTI is active if the controller is engaged and providing speed recommendations
    rti_active = rti_threat_ahead && rti_recommended_speed > 0;
  }
}

void HudRendererSP::draw(QPainter &p, const QRect &surface_rect) {
  // Draw base HUD elements
  HudRenderer::draw(p, surface_rect);
  
  // Draw RTI threat indicator if enabled and threat present
  if (rti_enabled && rti_threat_ahead) {
    drawRTIThreatIndicator(p, surface_rect);
  }
}

void HudRendererSP::drawRTIThreatIndicator(QPainter &p, const QRect &surface_rect) {
  // Position: bottom left corner of the display
  const int x_offset = 50;  // Left margin
  const int y_offset = surface_rect.height() - 290;  // Bottom margin (height + 50px from bottom)
  const int widget_width = 180;
  const int widget_height = 240;
  
  QRect rti_rect(x_offset, y_offset, widget_width, widget_height);
  
  // Get threat color based on distance
  QColor threat_color = getRTIThreatColor(rti_threat_distance);
  
  // Draw background box with semi-transparent fill
  p.setPen(QPen(threat_color, 3));
  p.setBrush(QColor(0, 0, 0, 150));
  p.drawRoundedRect(rti_rect, 20, 20);
  
  // Draw threat icon
  QRect icon_rect(rti_rect.x() + 40, rti_rect.y() + 20, 100, 80);
  drawRTIThreatIcon(p, icon_rect, rti_threat_type);
  
  // Draw threat type text
  p.setFont(threat_text_font);
  p.setPen(threat_color);
  QString threat_text = getRTIThreatText(rti_threat_type);
  p.drawText(rti_rect.adjusted(0, 110, 0, 0), Qt::AlignTop | Qt::AlignHCenter, threat_text);
  
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
  p.drawText(rti_rect.adjusted(0, 155, 0, 0), Qt::AlignTop | Qt::AlignHCenter, distance_text);
  
  // Draw speed recommendation if different from current
  if (rti_active && std::abs(rti_recommended_speed - speed / (is_metric ? 3.6 : 2.237)) > 1.0) {
    p.setFont(speed_rec_font);
    p.setPen(QColor(255, 255, 255, 200));
    
    float rec_speed_display = rti_recommended_speed * (is_metric ? 3.6 : 2.237);
    QString speed_text = QString("↓ %1").arg(static_cast<int>(rec_speed_display));
    p.drawText(rti_rect.adjusted(0, 200, 0, 0), Qt::AlignTop | Qt::AlignHCenter, speed_text);
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
    case RTIThreatType::POLICE: {
      // Draw police car silhouette using cached polygon
      p.save();
      p.translate(icon_rect.topLeft());
      p.drawPolygon(police_car_body);
      
      // Draw light bar on top
      p.fillRect(35, 20, 30, 8, icon_color);
      p.restore();
      break;
    }
    
    case RTIThreatType::SPEED_TRAP: {
      // Draw camera icon
      p.drawRect(icon_rect.x() + 25, icon_rect.y() + 30, 50, 35);
      p.drawEllipse(QPoint(icon_rect.x() + 50, icon_rect.y() + 47), 12, 12);
      // Draw mount
      p.drawLine(icon_rect.x() + 50, icon_rect.y() + 65, icon_rect.x() + 50, icon_rect.y() + 75);
      break;
    }
    
    case RTIThreatType::ACCIDENT: {
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
    
    case RTIThreatType::CONSTRUCTION: {
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
    
    case RTIThreatType::TRAFFIC_JAM: {
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
    case RTIThreatType::POLICE:
      return tr("POLICE");
    case RTIThreatType::SPEED_TRAP:
      return tr("CAMERA");
    case RTIThreatType::ACCIDENT:
      return tr("ACCIDENT");
    case RTIThreatType::TRAFFIC_JAM:
      return tr("TRAFFIC");
    case RTIThreatType::CONSTRUCTION:
      return tr("WORK ZONE");
    case RTIThreatType::OBJECT_HAZARD:
      return tr("HAZARD");
    case RTIThreatType::WEATHER_HAZARD:
      return tr("WEATHER");
    case RTIThreatType::ANIMAL_HAZARD:
      return tr("ANIMAL");
    case RTIThreatType::ROAD_CLOSED:
      return tr("CLOSED");
    case RTIThreatType::ROAD_HAZARD:
      return tr("ROAD");
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
