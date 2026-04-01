/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <QPainter>
#include <unordered_map>
#include <string>
#include <QString>
#include <QMutex>
#include <vector>
#include <QPointF>

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
  bool is_causing_recommendation;  // true if this threat is causing speed recommendation
};

class HudRendererSP : public HudRenderer {
  Q_OBJECT

public:
  HudRendererSP();
  void updateState(const UIState &s) override;
  void draw(QPainter &p, const QRect &surface_rect) override;

protected:
  // VTSC Rally Co-Pilot curve preview (HUD-only rendering)
  void drawVTSCCoPilotCurve(QPainter &p, const QRect &surface_rect);
  void drawVTSCCoPilotNavArrow(QPainter &p, const QPointF &center, float size_px) const;
  void drawCurveDirectionIcon(QPainter &p, const QRect &icon_rect, int direction) const;
  void refreshVTSCCoPilotTuning();

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

  // System readiness HUD
  void drawSystemReadiness(QPainter &p, const QRect &surface_rect);
  std::vector<std::pair<std::string, int>> subsystem_statuses_;
  bool all_systems_ready_ = false;
  float readiness_opacity_ = 1.0f;  // fades when engaged + all green

  // Rally co-pilot curve preview state (fed by longitudinalPlanSP.visionTurnSpeedControl)
  bool vtsc_copilot_hud_enabled_ = false;
  bool vtsc_copilot_visible_ = false;
  bool vtsc_copilot_visible_prev_ = false;
  float vtsc_copilot_alpha_ = 0.0f;  // fade in/out for game-style HUD feel
  float vtsc_target_speed_mps_ = 0.0f;
  float vtsc_copilot_curve_distance_m_ = 0.0f;
  float vtsc_copilot_curve_time_to_s_ = 0.0f;
  float vtsc_copilot_curve_kappa_max_ = 0.0f;
  int vtsc_copilot_curve_direction_ = 0;  // TurnDirection (unknown=0, left=1, right=2)
  int vtsc_copilot_curve_severity_ = 0;   // CurveSeverity (unknown=0, gentle=1, medium=2, tight=3)
  std::vector<QPointF> vtsc_copilot_curve_points_m_;
  struct VTSCCoPilotTileState {
    uint32_t id = 0;
    float distance_m = 0.0f;
    float time_to_s = 0.0f;
    float advisory_speed_mps = 0.0f;
    float max_curvature = 0.0f;
    int direction = 0;
    int severity = 0;
    std::vector<QPointF> points_m;
  };
  std::vector<VTSCCoPilotTileState> vtsc_copilot_tiles_;
  VTSCCoPilotTileState vtsc_copilot_exiting_tile_;
  bool vtsc_copilot_exiting_tile_active_ = false;
  float vtsc_copilot_exit_anim_progress_ = 1.0f;
  float vtsc_copilot_enter_anim_progress_ = 1.0f;  // fade-in for new tile

  // Speed flash animation (green bump / red drop)
  float vtsc_prev_advisory_speed_mps_ = 0.0f;
  uint32_t vtsc_prev_tile_id_ = 0;
  std::chrono::steady_clock::time_point vtsc_speed_flash_start_{};
  bool vtsc_speed_flash_active_ = false;
  bool vtsc_speed_flash_is_increase_ = false;  // true=green, false=red

  // Ego-advance interpolation for smooth 60 Hz scrolling between 5 Hz producer updates.
  float vtsc_copilot_v_ego_mps_ = 0.0f;
  float vtsc_copilot_ego_advance_m_ = 0.0f;
  std::chrono::steady_clock::time_point vtsc_copilot_last_draw_time_{};
  bool vtsc_copilot_last_draw_time_valid_ = false;
  std::chrono::steady_clock::time_point vtsc_copilot_last_nonempty_tiles_time_{};
  bool vtsc_copilot_last_nonempty_tiles_time_valid_ = false;

  struct VTSCCoPilotHudTuning {
    float curve_hold_new_dist_min_m = 30.0f;
    float geometry_epsilon_m = 0.05f;
    float kappa_show_min = 1.1e-3f;
    float kappa_hold_min = 1.0e-3f;
    float fade_in_alpha = 0.22f;
    float fade_out_alpha = 0.12f;
    float road_width_px = 18.0f;
    float glow_width_px = 10.0f;
    float speed_font_px = 52.0f;
    float unit_font_px = 22.0f;
  };
  VTSCCoPilotHudTuning vtsc_copilot_tuning_;
};
