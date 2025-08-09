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

#include "selfdrive/ui/qt/onroad/hud.h"

// RTI threat types matching the capnp enum
enum class RTIThreatType {
  NONE = 0,
  POLICE = 1,
  SPEED_TRAP = 2,
  ACCIDENT = 3,
  TRAFFIC_JAM = 4,
  CONSTRUCTION = 5,
  OBJECT_HAZARD = 6,
  WEATHER_HAZARD = 7,
  ANIMAL_HAZARD = 8,
  ROAD_CLOSED = 9,
  ROAD_HAZARD = 10,
  OTHER = 11,
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
  void drawRTIThreatIcon(QPainter &p, const QRect &icon_rect, RTIThreatType type);
  QString getRTIThreatText(RTIThreatType type) const;
  QColor getRTIThreatColor(float distance) const;
  
  // RTI state variables
  bool rti_enabled = false;
  bool rti_threat_ahead = false;
  float rti_threat_distance = 0.0;  // meters
  float rti_recommended_speed = 0.0;  // m/s
  RTIThreatType rti_threat_type = RTIThreatType::NONE;
  float rti_threat_confidence = 0.0;
  bool rti_active = false;  // RTI is actively controlling speed
  
  // Cached font objects for performance
  QFont threat_text_font;
  QFont distance_font;
  QFont speed_rec_font;
  QFont icon_font_small;
  QFont icon_font_large;
};
