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
  bool rti_enabled = false;  // Master RTI enabled switch
  bool rti_hud_enabled = false;  // HUD display enabled switch
  bool rti_threat_ahead = false;
  float rti_threat_distance = 0.0;  // meters
  float rti_recommended_speed = 0.0;  // m/s
  RTIThreatType rti_threat_type = RTIThreatType::POLICE;  // Default to first enum value
  bool rti_has_threat = false;  // Whether we have a valid threat
  float rti_threat_confidence = 0.0;
  bool rti_active = false;  // RTI is actively controlling speed
  
  // Cached font objects for performance
  QFont threat_text_font;
  QFont distance_font;
  QFont speed_rec_font;
  QFont icon_font_small;
  QFont icon_font_large;
};
