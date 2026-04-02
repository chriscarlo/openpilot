#pragma once

#include <chrono>

#include <QTimer>

#include "selfdrive/ui/qt/onroad/alerts.h"

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/qt/onroad/annotated_camera.h"
#define UIState UIStateSP
#define AnnotatedCameraWidget AnnotatedCameraWidgetSP
#else
#include "selfdrive/ui/qt/onroad/annotated_camera.h"
#endif

class OnroadWindow : public QWidget {
  Q_OBJECT

public:
  OnroadWindow(QWidget* parent = 0);

protected:
  void paintEvent(QPaintEvent *event) override;
  OnroadAlerts *alerts;
  AnnotatedCameraWidget *nvg;
  QColor bg = bg_colors[STATUS_DISENGAGED];
  QHBoxLayout* split;

  void drawTorqueMeters(QPainter &p) const;
  void drawTorqueMeterStrip(QPainter &p, const QRect &strip_rect, float fill_norm, bool active) const;

protected slots:
  virtual void offroadTransition(bool offroad);
  virtual void updateState(const UIState &s);
  void updateTorqueAnimation();

private:
  QTimer *torque_meter_timer = nullptr;
  std::chrono::steady_clock::time_point last_torque_animation_ts_{};
  bool last_torque_animation_ts_valid_ = false;
  float target_torque_norm_ = 0.0f;
  float display_torque_norm_ = 0.0f;
  int target_torque_side_ = 0;
  int display_torque_side_ = 0;
};
