/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/road_visualization_widget.h"
#include <QPainterPath>
#include <QHBoxLayout>
#include <QTimer>
#include <cmath>

RoadVisualizationWidget::RoadVisualizationWidget(QWidget *parent) : QWidget(parent) {
  setFixedSize(800, 600);
  setStyleSheet("background-color: #1C1C1C;");
  
  // Get current aggressiveness value
  std::string aggr_str = params.get("VisionTurnSpeedControlAggressiveness");
  aggressiveness = aggr_str.empty() ? 1.0f : std::stof(aggr_str);
  touch_position = 1.0 - (aggressiveness - 0.5) / 1.5; // Convert 0.5-2.0 to 1.0-0.0
  is_dragging = false;
  
  // Enable mouse tracking for hover effects
  setMouseTracking(true);
}

void RoadVisualizationWidget::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  
  drawRoad(painter);
  drawCar(painter);
  drawAnticipationMarker(painter);
}

void RoadVisualizationWidget::drawRoad(QPainter &painter) {
  int center_x = width() / 2;
  
  // Draw straight road section
  painter.fillRect(center_x - road_width/2, road_start_y, road_width, curve_start_y - road_start_y, QColor(80, 80, 80));
  
  // Draw road lines
  painter.setPen(QPen(Qt::white, 3, Qt::DashLine));
  painter.drawLine(center_x, road_start_y, center_x, curve_start_y);
  
  // Draw curve using QPainterPath
  QPainterPath curve_path;
  curve_path.moveTo(center_x - road_width/2, curve_start_y);
  
  // Create a smooth curve
  QPointF control1(center_x - road_width/2, curve_start_y + 100);
  QPointF control2(center_x + 50, curve_start_y + 150);
  QPointF end_point(center_x + 200, curve_start_y + 100);
  
  curve_path.cubicTo(control1, control2, end_point);
  curve_path.lineTo(center_x + 200, curve_start_y + 100 + road_width);
  
  QPointF control3(center_x + 50, curve_start_y + 150 + road_width);
  QPointF control4(center_x + road_width/2, curve_start_y + 100);
  QPointF start_return(center_x + road_width/2, curve_start_y);
  
  curve_path.cubicTo(control3, control4, start_return);
  curve_path.closeSubpath();
  
  painter.fillPath(curve_path, QColor(80, 80, 80));
  
  // Draw curve center line
  QPainterPath center_line;
  center_line.moveTo(center_x, curve_start_y);
  QPointF center_control1(center_x, curve_start_y + 100);
  QPointF center_control2(center_x + 100, curve_start_y + 125);
  QPointF center_end(center_x + 200, curve_start_y + 100 + road_width/2);
  center_line.cubicTo(center_control1, center_control2, center_end);
  
  painter.setPen(QPen(Qt::white, 3, Qt::DashLine));
  painter.drawPath(center_line);
  
  // Draw gradient overlay to show anticipation zone
  QLinearGradient gradient(center_x, road_start_y, center_x, curve_start_y);
  gradient.setColorAt(0, QColor(0, 255, 0, 0));
  gradient.setColorAt(1 - touch_position, QColor(255, 255, 0, 40));
  gradient.setColorAt(1, QColor(255, 0, 0, 60));
  
  painter.fillRect(center_x - road_width/2, road_start_y, road_width, curve_start_y - road_start_y, gradient);
}

void RoadVisualizationWidget::drawCar(QPainter &painter) {
  int center_x = width() / 2;
  int car_y = road_height - 50;
  int car_width = 60;
  int car_height = 100;
  
  // Draw car body
  painter.setBrush(QColor(41, 128, 185));
  painter.setPen(QPen(Qt::white, 2));
  painter.drawRoundedRect(center_x - car_width/2, car_y - car_height, car_width, car_height, 10, 10);
  
  // Draw windshield
  painter.setBrush(QColor(52, 152, 219));
  painter.drawRect(center_x - car_width/2 + 10, car_y - car_height + 20, car_width - 20, 30);
}

void RoadVisualizationWidget::drawAnticipationMarker(QPainter &painter) {
  int center_x = width() / 2;
  int marker_y = road_start_y + (curve_start_y - road_start_y) * touch_position;
  
  // Draw marker line
  painter.setPen(QPen(QColor(255, 193, 7), 4));
  painter.drawLine(center_x - road_width/2 - 20, marker_y, center_x + road_width/2 + 20, marker_y);
  
  // Draw marker handle
  QRect handle_rect(center_x + road_width/2 + 30, marker_y - 20, 40, 40);
  painter.setBrush(QColor(255, 193, 7));
  painter.setPen(QPen(Qt::white, 2));
  painter.drawEllipse(handle_rect);
  
  // Draw distance text
  float distance = 50 + (300 - 50) * (1 - touch_position) * aggressiveness;
  QString distance_text = QString("%1m").arg(static_cast<int>(distance));
  
  painter.setPen(Qt::white);
  painter.setFont(QFont("Inter", 18, QFont::Bold));
  painter.drawText(center_x - road_width/2 - 100, marker_y + 5, distance_text);
}

void RoadVisualizationWidget::mousePressEvent(QMouseEvent *event) {
  int click_y = event->pos().y();
  
  // Check if click is in the road area or near the handle
  if (click_y >= road_start_y && click_y <= curve_start_y) {
    is_dragging = true;
    touch_position = static_cast<float>(click_y - road_start_y) / (curve_start_y - road_start_y);
    touch_position = std::max(0.0f, std::min(1.0f, touch_position));
    update();
  }
}

void RoadVisualizationWidget::mouseMoveEvent(QMouseEvent *event) {
  if (is_dragging) {
    int click_y = event->pos().y();
    touch_position = static_cast<float>(click_y - road_start_y) / (curve_start_y - road_start_y);
    touch_position = std::max(0.0f, std::min(1.0f, touch_position));
    update();
  }
}

void RoadVisualizationWidget::mouseReleaseEvent(QMouseEvent *event) {
  if (is_dragging) {
    is_dragging = false;
    // Convert position to aggressiveness value
    float new_aggressiveness = 0.5 + (1.0 - touch_position) * 1.5;  // Convert 1.0-0.0 to 0.5-2.0
    updateAggressiveness(new_aggressiveness);
  }
}

void RoadVisualizationWidget::updateAggressiveness(float value) {
  aggressiveness = std::max(0.5f, std::min(2.0f, value));
  params.put("VisionTurnSpeedControlAggressiveness", std::to_string(aggressiveness));
  updateInfoText();
}

void RoadVisualizationWidget::updateInfoText() {
  // This will be connected to labels in the parent panel
}

// AnticipationDistancePanel implementation
AnticipationDistancePanel::AnticipationDistancePanel(QWidget *parent) : QFrame(parent) {
  setStyleSheet(R"(
    #back_btn {
      font-size: 50px;
      margin: 0px;
      padding: 15px;
      border-width: 0;
      border-radius: 30px;
      color: #dddddd;
      background-color: #393939;
    }
    #back_btn:pressed {
      background-color: #4a4a4a;
    }
  )");
  
  setupUI();
}

void AnticipationDistancePanel::setupUI() {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  
  // Header
  QHBoxLayout *header_layout = new QHBoxLayout();
  header_layout->setContentsMargins(50, 30, 50, 0);
  
  back_btn = new QPushButton("◀", this);
  back_btn->setObjectName("back_btn");
  back_btn->setFixedSize(90, 90);
  connect(back_btn, &QPushButton::clicked, this, &AnticipationDistancePanel::backPress);
  
  QLabel *title = new QLabel(tr("Pre-emptive Slowing Distance"));
  title->setStyleSheet("font-size: 48px; font-weight: 600; color: #FFFFFF;");
  title->setAlignment(Qt::AlignCenter);
  
  header_layout->addWidget(back_btn);
  header_layout->addWidget(title, 1, Qt::AlignCenter);
  header_layout->addSpacing(90);
  
  main_layout->addLayout(header_layout);
  main_layout->addSpacing(30);
  
  // Road visualization
  road_widget = new RoadVisualizationWidget(this);
  main_layout->addWidget(road_widget, 0, Qt::AlignCenter);
  
  // Info panel
  QWidget *info_panel = new QWidget(this);
  info_panel->setStyleSheet("background-color: #393939; border-radius: 20px;");
  QVBoxLayout *info_layout = new QVBoxLayout(info_panel);
  info_layout->setContentsMargins(30, 20, 30, 20);
  
  info_label = new QLabel(tr("Drag the yellow marker to adjust when the car starts slowing down for curves"));
  info_label->setStyleSheet("font-size: 28px; color: #FFFFFF;");
  info_label->setWordWrap(true);
  info_label->setAlignment(Qt::AlignCenter);
  
  distance_label = new QLabel();
  distance_label->setStyleSheet("font-size: 36px; font-weight: 600; color: #FFD700;");
  distance_label->setAlignment(Qt::AlignCenter);
  
  timing_label = new QLabel();
  timing_label->setStyleSheet("font-size: 24px; color: #CCCCCC;");
  timing_label->setAlignment(Qt::AlignCenter);
  
  info_layout->addWidget(info_label);
  info_layout->addSpacing(20);
  info_layout->addWidget(distance_label);
  info_layout->addWidget(timing_label);
  
  main_layout->addWidget(info_panel, 0, Qt::AlignCenter);
  main_layout->addStretch();
  
  // Update labels initially and connect to parameter changes
  auto updateLabels = [this]() {
    Params p;
    std::string aggr_str = p.get("VisionTurnSpeedControlAggressiveness");
    float aggressiveness = aggr_str.empty() ? 1.0f : std::stof(aggr_str);
    int percentage = static_cast<int>((aggressiveness - 0.5) / 1.5 * 100);
    distance_label->setText(QString("%1% aggressiveness").arg(percentage + 50));
    
    QString timing_text;
    if (aggressiveness < 0.8) {
      timing_text = tr("Later slowing (sportier)");
    } else if (aggressiveness > 1.3) {
      timing_text = tr("Earlier slowing (more conservative)");
    } else {
      timing_text = tr("Balanced timing");
    }
    timing_label->setText(timing_text);
  };
  
  updateLabels();
  
  // Update labels when parameters change
  QTimer *update_timer = new QTimer(this);
  connect(update_timer, &QTimer::timeout, updateLabels);
  update_timer->start(1000); // Update every second
}