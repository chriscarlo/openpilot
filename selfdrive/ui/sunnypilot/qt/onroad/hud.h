/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QPainter>
#include <QString>
#include <vector>

#include "cereal/gen/cpp/custom.capnp.h"
#include "selfdrive/ui/qt/onroad/hud.h"

// Use cereal threat types directly instead of duplicating the enum
// This ensures we stay in sync with the capnp definitions
using RTIThreatType = cereal::RtiStateSP::ThreatType;

// Structure to hold info for each threat
struct RTIThreatInfo {
  RTIThreatType type;
  double latitude;
  double longitude;
  float distance;  // meters
  double relative_bearing;  // degrees
  bool has_location;
};

class HudRendererSP : public HudRenderer {
  Q_OBJECT

public:
  HudRendererSP();
  void updateState(const UIState &s) override;
  void draw(QPainter &p, const QRect &surface_rect) override;

protected:
  // RTI drawing methods
  void drawRTIThreatIndicator(QPainter &p, const QRect &surface_rect);
  void drawRTIThreatIndicatorMulti(QPainter &p, const QRect &surface_rect);
  void drawRTIThreatIcon(QPainter &p, const QRect &icon_rect, RTIThreatType type);
  void drawRTIArrow(QPainter &p, const QRect &arrow_rect, double relative_bearing);
  void drawRTIArrowCompact(QPainter &p, const QRect &arrow_rect, double relative_bearing, const QColor &color);
  QString getRTIThreatText(RTIThreatType type) const;
  QString getRTIThreatTextShort(RTIThreatType type) const;
  QString formatDistance(float distance_m) const;
  QColor getRTIThreatColor(float distance) const;
  double calculateRelativeBearing(double ego_latitude, double ego_longitude, 
                                 double threat_latitude, double threat_longitude, 
                                 double ego_heading_deg) const;
  void updateRTIThreats(const UIState &s);
  
  // RTI state variables
  bool rti_enabled = false;  // Master RTI enabled switch
  bool rti_hud_enabled = false;  // HUD display enabled switch
  bool rti_threat_ahead = false;
  float rti_recommended_speed = 0.0;  // m/s
  bool rti_active = false;  // RTI is actively controlling speed
  
  // Multiple threats support
  std::vector<RTIThreatInfo> rti_threats;
  
  // Legacy single threat variables (for backward compatibility)
  float rti_threat_distance = 0.0;  // meters
  RTIThreatType rti_threat_type = RTIThreatType::POLICE;  // Default to first enum value
  bool rti_has_threat = false;  // Whether we have a valid threat
  float rti_threat_confidence = 0.0;
  double rti_threat_lat = 0.0;
  double rti_threat_lon = 0.0;
  double rti_relative_bearing = 0.0;  // Relative bearing to threat in degrees
  
  // Ego position and heading  
  double ego_lat = 0.0;
  double ego_lon = 0.0;
  double ego_bearing = 0.0;  // True heading in degrees
  bool has_gps = false;
  
  // Cached font objects for performance
  QFont threat_text_font;
  QFont distance_font;
  QFont speed_rec_font;
  QFont icon_font_small;
  QFont icon_font_large;
  
  // Cached arrow pixmaps for performance
  QPixmap arrow_pixmap;
  bool arrow_pixmap_cached = false;
  QPixmap compact_arrow_pixmap;
  bool compact_arrow_cached = false;
  int compact_arrow_size = 0;
  void createArrowPixmap();
  void createCompactArrowPixmap(int size);
};
