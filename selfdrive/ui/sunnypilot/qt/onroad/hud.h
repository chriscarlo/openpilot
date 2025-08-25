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
  double calculateRelativeBearing(double ego_latitude, double ego_longitude, 
                                 double threat_latitude, double threat_longitude, 
                                 double ego_heading_deg) const;
  void updateRTIThreats(const UIState &s);
  double angleForDirection(cereal::RtiStateSP::Direction dir) const;
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
  
  // Ego position and heading  
  double ego_lat = 0.0;
  double ego_lon = 0.0;
  double ego_bearing = 0.0;  // True heading in degrees
  bool has_gps = false;
  uint64_t last_gps_rcv_frame = 0;  // Track last GPS update frame for staleness check
  // Track GPS freshness transitions for logging and robustness
  bool gps_fresh_prev = false;
  uint64_t last_gps_log_frame = 0;  // rate-limit logs
  enum class GPSSource { UNKNOWN = 0, EXTERNAL = 1, INTERNAL = 2 };
  GPSSource last_gps_source = GPSSource::UNKNOWN;
  
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
  
  // Arrow update rate control (1Hz target for bearing updates)
  mutable std::unordered_map<std::string, uint64_t> arrow_last_update_frame_;
  mutable std::unordered_map<std::string, double> arrow_cached_angle_deg_;
  uint64_t current_frame_ = 0;  // Current frame number for update tracking
};
