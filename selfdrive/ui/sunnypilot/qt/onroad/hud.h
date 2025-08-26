/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QPainter>
#include <unordered_map>
#include <string>
#include <QString>
#include <QMutex>
#include <vector>

#include "cereal/gen/cpp/custom.capnp.h"
#include "selfdrive/ui/qt/onroad/hud.h"

// Use cereal threat types directly instead of duplicating the enum
// This ensures we stay in sync with the capnp definitions
using RTIThreatType = cereal::RtiStateSP::ThreatType;

// Structure to hold info for each threat
struct RTIThreatInfo {
  std::string id;
  RTIThreatType type;
  double latitude;
  double longitude;
  float distance;  // meters
  double relative_bearing;  // degrees
  bool has_location;
  cereal::RtiStateSP::Direction direction;  // coarse direction fallback
  float speed_limit_ms;  // optional; provided by backend
  bool on_same_road;     // provided by backend (no inference)
};

class HudRendererSP : public HudRenderer {
  Q_OBJECT

public:
  HudRendererSP();
  void updateState(const UIState &s) override;
  void draw(QPainter &p, const QRect &surface_rect) override;

protected:
  // RTI drawing methods
  void drawRTIThreatIndicatorMulti(QPainter &p, const QRect &surface_rect);
  void drawRTIArrowCompact(QPainter &p, const QRect &arrow_rect, double relative_bearing, const QColor &color);
  QString getRTIThreatTextShort(RTIThreatType type) const;
  QString formatDistance(float distance_m) const;
  QColor getRTIThreatColor(float distance) const;
  QColor getRTIThreatBgColorByType(RTIThreatType type) const;
  void updateRTIThreats(const UIState &s);
  double smoothAngleForThreat(const std::string &id, double raw_angle_deg) const;
  double smoothYForThreat(const std::string &id, double target_y, double initial_y) const;
  
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
  cereal::RtiStateSP::Direction rti_direction = cereal::RtiStateSP::Direction::UNKNOWN;
  
  QPixmap compact_arrow_pixmap;
  bool compact_arrow_cached = false;
  int compact_arrow_size = 0;
  void createCompactArrowPixmap(int size);

  // Smoothed angles per threat id
  mutable std::unordered_map<std::string, double> smoothed_angles_deg_;
  mutable QMutex smoothed_angles_mutex_;

  // Smoothed Y positions (top of box) per threat id
  mutable std::unordered_map<std::string, double> smoothed_y_top_;
  mutable QMutex smoothed_y_mutex_;
};
