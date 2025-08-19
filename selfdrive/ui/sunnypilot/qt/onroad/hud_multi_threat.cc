/**
 * Multi-threat RTI display implementation
 * Shows up to 4 threats in a compact, single-line format with directional arrows
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/hud.h"
#include <algorithm>

void HudRendererSP::drawRTIThreatIndicatorMulti(QPainter &p, const QRect &surface_rect) {
  // Position to bottom-align with lateral accel widget
  const int bottom_margin = 15;
  const int left_margin = 15;
  
  // Expanded size to accommodate multiple threats
  const int widget_width = 580;  // Increased width
  const int widget_height = 280;  // Adjusted height for 4 single-line threats
  
  const int x_offset = left_margin;
  const int y_offset = surface_rect.height() - widget_height - bottom_margin;
  
  QRect rti_rect(x_offset, y_offset, widget_width, widget_height);
  
  // Draw widget background
  p.setPen(QPen(QColor(255, 255, 255, 75), 6));
  p.setBrush(QColor(0, 0, 0, 115));
  p.drawRoundedRect(rti_rect, 32, 32);
  
  // Check if we have active threats
  if (!rti_threats.empty()) {
    // Draw header
    p.setFont(InterFont(38, QFont::DemiBold));
    p.setPen(QColor(255, 255, 255, 200));
    p.drawText(rti_rect.adjusted(20, 15, -20, 0), Qt::AlignTop | Qt::AlignLeft, "RTI");
    
    // Draw each threat on a single line
    const int line_height = 55;  // Height per threat line
    const int start_y = rti_rect.y() + 60;  // Start below header
    const int max_threats = 4;  // Maximum threats to display
    
    int displayed = 0;
    for (const auto& threat : rti_threats) {
      if (displayed >= max_threats) break;
      
      int y_pos = start_y + (displayed * line_height);
      
      // Get threat color based on distance
      QColor threat_color = getRTIThreatColor(threat.distance);
      
      // Draw compact single line: [arrow] TYPE • 0.5mi
      QRect line_rect(rti_rect.x() + 20, y_pos, widget_width - 40, line_height);
      
      // Draw arrow (smaller for multi-threat view)
      if (has_gps && threat.has_location) {
        QRect arrow_rect(line_rect.x(), line_rect.y() + 8, 32, 32);  // Smaller arrow
        drawRTIArrowCompact(p, arrow_rect, threat.relative_bearing, threat_color);
        line_rect.adjust(40, 0, 0, 0);  // Shift text right
      }
      
      // Draw threat type and distance on same line
      p.setFont(InterFont(36, QFont::Normal));
      p.setPen(threat_color);
      
      QString threat_line = QString("%1 • %2")
        .arg(getRTIThreatTextShort(threat.type))
        .arg(formatDistance(threat.distance));
      
      p.drawText(line_rect, Qt::AlignVCenter | Qt::AlignLeft, threat_line);
      
      displayed++;
    }
    
    // If RTI is actively controlling speed, show recommendation at bottom
    if (rti_active && rti_recommended_speed > 0) {
      p.setFont(InterFont(32, QFont::Normal));
      p.setPen(QColor(255, 255, 255, 180));
      
      float rec_speed_display = rti_recommended_speed * (is_metric ? 3.6 : 2.237);
      QString speed_text = QString("Target: %1 %2")
        .arg(static_cast<int>(rec_speed_display))
        .arg(is_metric ? "km/h" : "mph");
      
      p.drawText(rti_rect.adjusted(20, -40, -20, -10), 
                 Qt::AlignBottom | Qt::AlignLeft, speed_text);
    }
  } else {
    // Draw placeholder when no threats
    p.setFont(InterFont(48, QFont::DemiBold));
    p.setPen(QColor(150, 150, 150, 200));
    p.drawText(rti_rect, Qt::AlignCenter, tr("RTI"));
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
  tint_painter.setCompositionMode(QPainter.CompositionMode_SourceIn);
  tint_painter.fillRect(tinted_arrow.rect(), color);
  tint_painter.end();
  
  p.drawPixmap(0, 0, size, size, tinted_arrow);
  
  p.restore();
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

// Update state to track multiple threats
void HudRendererSP::updateRTIThreats(const UIState &s) {
  rti_threats.clear();
  
  if (s.sm && s.sm->valid("rtiStateSP") && s.sm->updated("rtiStateSP")) {
    const auto rti_state = (*s.sm)["rtiStateSP"].getRtiStateSP();
    auto threats = rti_state.getThreats();
    
    // Process up to 4 threats, sorted by distance
    for (size_t i = 0; i < std::min(threats.size(), size_t(4)); i++) {
      auto threat = threats[i];
      
      RTIThreatInfo info;
      info.type = threat.getType();
      info.distance = threat.getDistance();
      info.latitude = threat.getLatitude();
      info.longitude = threat.getLongitude();
      info.has_location = std::isfinite(info.latitude) && std::isfinite(info.longitude);
      
      // Calculate relative bearing if we have GPS
      if (has_gps && info.has_location) {
        info.relative_bearing = calculateRelativeBearing(
          ego_lat, ego_lon, info.latitude, info.longitude, ego_bearing
        );
      } else {
        info.relative_bearing = 0.0;  // Default to ahead
      }
      
      rti_threats.push_back(info);
    }
    
    // Sort by distance (closest first)
    std::sort(rti_threats.begin(), rti_threats.end(), 
              [](const RTIThreatInfo& a, const RTIThreatInfo& b) {
                return a.distance < b.distance;
              });
  }
}