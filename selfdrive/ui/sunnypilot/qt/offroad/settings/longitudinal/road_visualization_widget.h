/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
#include <QLabel>
#include <QVBoxLayout>
#include <QPropertyAnimation>

#include "common/params.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class RoadVisualizationWidget : public QWidget {
  Q_OBJECT

public:
  explicit RoadVisualizationWidget(QWidget *parent = nullptr);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  
  float getAggressiveness() const { return aggressiveness; }

private:
  void updateAggressiveness(float value);
  void drawRoad(QPainter &painter);
  void drawCar(QPainter &painter);
  void drawAnticipationMarker(QPainter &painter);
  void updateInfoText();
  
  float aggressiveness;
  float touch_position;  // 0.0 to 1.0 along the road
  bool is_dragging;
  
  // Road geometry
  const int road_start_y = 100;
  const int road_height = 400;
  const int road_width = 200;
  const int curve_start_y = 350;
  
  // Animation
  QPropertyAnimation *marker_animation;
  
  Params params;
};

class AnticipationDistancePanel : public QFrame {
  Q_OBJECT

public:
  explicit AnticipationDistancePanel(QWidget *parent = nullptr);

signals:
  void backPress();

private:
  void setupUI();
  
  RoadVisualizationWidget *road_widget;
  QLabel *info_label;
  QLabel *distance_label;
  QLabel *timing_label;
  QPushButton *back_btn;
};