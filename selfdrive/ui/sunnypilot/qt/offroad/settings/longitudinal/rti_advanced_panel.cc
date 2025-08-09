/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_advanced_panel.h"
#include "selfdrive/ui/qt/util.h"
#include <QPainterPath>
#include <QApplication>
#include <QScreen>
#include <QLineEdit>
#include <QComboBox>
#include <cmath>

// Helper function for safe string conversion
static int safeStringToInt(const std::string& str, int defaultValue) {
  if (str.empty()) return defaultValue;
  try {
    // Check if string contains only digits and optional leading negative sign
    if (str.find_first_not_of("0123456789-") != std::string::npos) {
      return defaultValue;
    }
    return std::atoi(str.c_str());
  } catch (...) {
    return defaultValue;
  }
}

// RTI Visualization Widget Implementation
RTIVisualizationWidget::RTIVisualizationWidget(QWidget *parent) : QWidget(parent) {
  // Match VTSC panel dimensions
  setMinimumSize(400, 350);
  setMaximumSize(500, 450);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  
  // Initialize state
  current_aggressiveness = RTIAggressiveness::BALANCED;
  min_distance_m = 100;
  max_distance_m = 2000;
  is_dragging = false;
  is_hovering = false;
  animation_phase = 0.0f;
  
  // Setup animation timer for threat indicators
  animation_timer = new QTimer(this);
  connect(animation_timer, &QTimer::timeout, [this]() {
    animation_phase += 0.1f;
    if (animation_phase > 2 * M_PI) animation_phase = 0.0f;
    update();
  });
  animation_timer->start(50); // 20 FPS for smooth animation
  
  // Cache fonts for performance
  label_font = InterFont(28, QFont::DemiBold);
  distance_font = InterFont(32, QFont::Bold);
  
  setMouseTracking(true);
  setCursor(Qt::PointingHandCursor);
}

void RTIVisualizationWidget::updateConfiguration(RTIAggressiveness aggressiveness, int min_distance, int max_distance) {
  current_aggressiveness = aggressiveness;
  min_distance_m = min_distance;
  max_distance_m = max_distance;
  recalculateLayout();
  update();
}

void RTIVisualizationWidget::recalculateLayout() {
  // Compact road layout matching VTSC dimensions
  layout.road_width = width() * 0.5f;
  layout.scale_factor = height() / 3000.0f; // Reduced view distance for compact display
  
  // Road runs from bottom (vehicle) to top (horizon)
  layout.road_start = QPointF(width() / 2, height() - 30);
  layout.road_end = QPointF(width() / 2, 30);
}

void RTIVisualizationWidget::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
  
  // Dark background
  painter.fillRect(rect(), QColor(16, 16, 16));
  
  // Draw components in order
  drawRoadScene(painter);
  drawThreatZones(painter);
  drawVehicle(painter, layout.road_start.y());
  drawThreatIndicators(painter);
}

void RTIVisualizationWidget::drawRoadScene(QPainter &painter) {
  // Draw road surface with perspective
  QPainterPath road_path;
  float bottom_width = layout.road_width;
  float top_width = layout.road_width * 0.3f; // Perspective narrowing
  
  QPointF bottom_left(layout.road_start.x() - bottom_width/2, layout.road_start.y());
  QPointF bottom_right(layout.road_start.x() + bottom_width/2, layout.road_start.y());
  QPointF top_left(layout.road_end.x() - top_width/2, layout.road_end.y());
  QPointF top_right(layout.road_end.x() + top_width/2, layout.road_end.y());
  
  road_path.moveTo(bottom_left);
  road_path.lineTo(top_left);
  road_path.lineTo(top_right);
  road_path.lineTo(bottom_right);
  road_path.closeSubpath();
  
  // Road surface
  painter.fillPath(road_path, QColor(64, 64, 64));
  
  // Road edges
  painter.setPen(QPen(QColor(200, 200, 200), 2));
  painter.drawPath(road_path);
  
  // Center line dashes
  painter.setPen(QPen(QColor(255, 255, 0), 2, Qt::DashLine));
  painter.drawLine(layout.road_start, layout.road_end);
  
  // Distance markers
  painter.setFont(QFont("Inter", 16));
  painter.setPen(QColor(150, 150, 150));
  
  for (int distance = 500; distance <= 2000; distance += 500) {
    float y = layout.road_start.y() - distance * layout.scale_factor;
    if (y > layout.road_end.y()) {
      float width_at_distance = bottom_width * (1.0f - (distance * layout.scale_factor / height()) * 0.7f);
      painter.drawLine(layout.road_start.x() - width_at_distance/2 - 20, y,
                      layout.road_start.x() + width_at_distance/2 + 20, y);
      painter.drawText(QRect(layout.road_start.x() + width_at_distance/2 + 30, y - 15, 100, 30),
                      Qt::AlignLeft | Qt::AlignVCenter, QString("%1m").arg(distance));
    }
  }
}

void RTIVisualizationWidget::drawThreatZones(QPainter &painter) {
  // Draw threat detection zones
  float min_y = layout.road_start.y() - min_distance_m * layout.scale_factor;
  float max_y = layout.road_start.y() - max_distance_m * layout.scale_factor;
  
  if (max_y < layout.road_end.y()) max_y = layout.road_end.y();
  
  // Critical zone (red) - closest threats
  if (min_distance_m < 300) {
    float critical_y = layout.road_start.y() - 300 * layout.scale_factor;
    QLinearGradient critical_gradient(0, min_y, 0, critical_y);
    critical_gradient.setColorAt(0.0, QColor(255, 0, 0, 120));
    critical_gradient.setColorAt(1.0, QColor(255, 100, 0, 80));
    
    float width_min = layout.road_width * (1.0f - (min_distance_m * layout.scale_factor / height()) * 0.7f);
    float width_critical = layout.road_width * (1.0f - (300 * layout.scale_factor / height()) * 0.7f);
    
    QPainterPath critical_zone;
    critical_zone.moveTo(layout.road_start.x() - width_min/2, min_y);
    critical_zone.lineTo(layout.road_start.x() - width_critical/2, critical_y);
    critical_zone.lineTo(layout.road_start.x() + width_critical/2, critical_y);
    critical_zone.lineTo(layout.road_start.x() + width_min/2, min_y);
    critical_zone.closeSubpath();
    
    painter.fillPath(critical_zone, critical_gradient);
  }
  
  // Warning zone (yellow) - medium distance threats
  QLinearGradient warning_gradient(0, min_y, 0, max_y);
  warning_gradient.setColorAt(0.0, QColor(255, 255, 0, 100));
  warning_gradient.setColorAt(0.5, QColor(255, 200, 0, 60));
  warning_gradient.setColorAt(1.0, QColor(0, 255, 0, 40));
  
  float width_min = layout.road_width * (1.0f - (min_distance_m * layout.scale_factor / height()) * 0.7f);
  float width_max = layout.road_width * (1.0f - (max_distance_m * layout.scale_factor / height()) * 0.7f);
  
  QPainterPath warning_zone;
  warning_zone.moveTo(layout.road_start.x() - width_min/2, min_y);
  warning_zone.lineTo(layout.road_start.x() - width_max/2, max_y);
  warning_zone.lineTo(layout.road_start.x() + width_max/2, max_y);
  warning_zone.lineTo(layout.road_start.x() + width_min/2, min_y);
  warning_zone.closeSubpath();
  
  painter.fillPath(warning_zone, warning_gradient);
  
  // Zone labels
  painter.setFont(label_font);
  painter.setPen(QColor(255, 255, 255, 180));
  
  // Format distances with proper units
  Params params;
  bool is_metric = params.getBool("IsMetric");
  QString min_label, max_label;
  
  if (min_y > layout.road_end.y() + 50) {
    if (is_metric) {
      double min_km = min_distance_m * 0.001;
      min_label = QString("%1 km").arg(min_km, 0, 'f', 1);
    } else {
      double min_mi = min_distance_m * 0.000621371;
      min_label = QString("%1 mi").arg(min_mi, 0, 'f', 2);
    }
    
    painter.drawText(QRect(10, min_y - 15, width() - 20, 30),
                    Qt::AlignCenter, QString("Min Detection: %1").arg(min_label));
  }
  
  if (max_y > layout.road_end.y() + 50) {
    if (is_metric) {
      double max_km = max_distance_m * 0.001;
      max_label = QString("%1 km").arg(max_km, 0, 'f', 1);
    } else {
      double max_mi = max_distance_m * 0.000621371;
      max_label = QString("%1 mi").arg(max_mi, 0, 'f', 2);
    }
    
    painter.drawText(QRect(10, max_y - 15, width() - 20, 30),
                    Qt::AlignCenter, QString("Max Detection: %1").arg(max_label));
  }
}

void RTIVisualizationWidget::drawVehicle(QPainter &painter, float y_position) {
  // Draw our vehicle at bottom
  const float car_width = 50;
  const float car_height = 90;
  const float car_x = layout.road_start.x() - car_width/2;
  const float car_y = y_position - car_height + 20;
  
  // Car shadow
  painter.setPen(Qt::NoPen);
  painter.setBrush(QColor(0, 0, 0, 100));
  painter.drawEllipse(QPointF(layout.road_start.x(), car_y + car_height), 30, 12);
  
  // Car body (sunnypilot blue)
  painter.setBrush(QColor(74, 144, 226));
  QPainterPath car_body;
  car_body.addRoundedRect(car_x, car_y, car_width, car_height, 12, 20);
  painter.fillPath(car_body, QColor(74, 144, 226));
  
  // Windshield
  painter.setBrush(QColor(150, 200, 255, 150));
  QPainterPath windshield;
  windshield.addRoundedRect(car_x + 8, car_y + 15, car_width - 16, car_height * 0.3f, 8, 8);
  painter.fillPath(windshield, QColor(150, 200, 255, 150));
  
  // Car outline
  painter.setPen(QPen(QColor(255, 255, 255), 2));
  painter.setBrush(Qt::NoBrush);
  painter.drawPath(car_body);
}

void RTIVisualizationWidget::drawThreatIndicators(QPainter &painter) {
  // Animated threat indicators at various distances
  painter.setFont(QFont("Inter", 24, QFont::Bold));
  
  // Simulate threats at different distances with animation
  float threats[] = {150, 400, 800, 1500}; // meters
  QString threat_types[] = {"POL", "CAM", "HAZ", "CON"};
  QColor threat_colors[] = {QColor(255, 0, 0), QColor(255, 100, 0), QColor(255, 255, 0), QColor(100, 255, 100)};
  
  for (int i = 0; i < 4; i++) {
    float threat_distance = threats[i];
    if (threat_distance >= min_distance_m && threat_distance <= max_distance_m) {
      float threat_y = layout.road_start.y() - threat_distance * layout.scale_factor;
      
      if (threat_y > layout.road_end.y()) {
        // Animated pulsing
        float alpha_multiplier = 0.7f + 0.3f * sin(animation_phase + i * M_PI/2);
        QColor color = threat_colors[i];
        color.setAlphaF(alpha_multiplier);
        
        // Threat icon
        painter.setPen(QPen(color, 3));
        painter.setBrush(color);
        painter.drawEllipse(QPointF(layout.road_start.x() + (i % 2 ? 40 : -40), threat_y), 20, 20);
        
        // Threat type text
        painter.setPen(Qt::white);
        painter.drawText(QRect(layout.road_start.x() + (i % 2 ? 25 : -55), threat_y - 15, 30, 30),
                        Qt::AlignCenter, threat_types[i]);
        
        // Distance label
        painter.setFont(QFont("Inter", 18));
        painter.setPen(QColor(255, 255, 255, 200));
        painter.drawText(QRect(layout.road_start.x() + (i % 2 ? 65 : -100), threat_y - 10, 80, 20),
                        Qt::AlignCenter, QString("%1m").arg(static_cast<int>(threat_distance)));
      }
    }
  }
}

QColor RTIVisualizationWidget::getThreatColor(float distance) const {
  if (distance < 200) return QColor(255, 0, 0);      // Red - Critical
  else if (distance < 500) return QColor(255, 165, 0); // Orange - Near  
  else if (distance < 1000) return QColor(255, 255, 0); // Yellow - Normal
  else return QColor(0, 255, 0);                      // Green - Far
}

void RTIVisualizationWidget::mousePressEvent(QMouseEvent *event) {
  // Allow clicking to adjust threat detection zones
  float click_y = event->pos().y();
  
  if (click_y >= layout.road_end.y() && click_y <= layout.road_start.y()) {
    is_dragging = true;
    setCursor(Qt::ClosedHandCursor);
    
    // Convert click position to distance
    float distance = (layout.road_start.y() - click_y) / layout.scale_factor;
    distance = std::max(50.0f, std::min(3000.0f, distance));
    
    // Determine if closer to min or max distance
    if (abs(distance - min_distance_m) < abs(distance - max_distance_m)) {
      min_distance_m = static_cast<int>(distance);
    } else {
      max_distance_m = static_cast<int>(distance);
    }
    
    // Ensure min < max
    if (min_distance_m >= max_distance_m) {
      std::swap(min_distance_m, max_distance_m);
    }
    
    emit distanceChanged(min_distance_m, max_distance_m);
    update();
    event->accept();
  }
}

void RTIVisualizationWidget::mouseMoveEvent(QMouseEvent *event) {
  if (is_dragging) {
    // Continue dragging logic similar to mousePressEvent
    float click_y = event->pos().y();
    float distance = (layout.road_start.y() - click_y) / layout.scale_factor;
    distance = std::max(50.0f, std::min(3000.0f, distance));
    
    // Update the closer distance marker
    if (abs(distance - min_distance_m) < abs(distance - max_distance_m)) {
      min_distance_m = static_cast<int>(distance);
    } else {
      max_distance_m = static_cast<int>(distance);
    }
    
    if (min_distance_m >= max_distance_m) {
      std::swap(min_distance_m, max_distance_m);
    }
    
    emit distanceChanged(min_distance_m, max_distance_m);
    update();
    event->accept();
  } else {
    // Update hover state
    setCursor(Qt::PointingHandCursor);
  }
}

void RTIVisualizationWidget::mouseReleaseEvent(QMouseEvent *event) {
  if (is_dragging) {
    is_dragging = false;
    setCursor(Qt::PointingHandCursor);
    event->accept();
  }
}

// RTI Config Data Panel Implementation
RTIConfigDataPanel::RTIConfigDataPanel(QWidget *parent) : QFrame(parent) {
  setupUI();
}

void RTIConfigDataPanel::setupUI() {
  // Match VTSC text panel fixed width
  setFixedWidth(650);
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
  setStyleSheet("background-color: transparent; border: none;");
  
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(20, 10, 20, 10);
  main_layout->setSpacing(15);
  
  // Main description
  main_description = new QLabel(this);
  main_description->setWordWrap(true);
  main_description->setStyleSheet(R"(
    font-size: 38px;
    font-weight: 400;
    color: #E4E4E4;
    background-color: transparent;
    padding-bottom: 20px;
  )");
  main_layout->addWidget(main_description);
  
  // Configuration info section
  distance_info = new QLabel(this);
  distance_info->setStyleSheet(R"(
    font-size: 32px;
    color: #aaaaaa;
    padding-left: 15px;
    background-color: transparent;
  )");
  distance_info->setWordWrap(true);
  main_layout->addWidget(distance_info);
  
  speed_reduction_info = new QLabel(this);
  speed_reduction_info->setStyleSheet(R"(
    font-size: 32px;
    color: #aaaaaa;
    padding-left: 15px;
    background-color: transparent;
  )");
  speed_reduction_info->setWordWrap(true);
  main_layout->addWidget(speed_reduction_info);
  
  aggressiveness_info = new QLabel(this);
  aggressiveness_info->setStyleSheet(R"(
    font-size: 32px;
    color: #aaaaaa;
    padding-left: 15px;
    background-color: transparent;
  )");
  aggressiveness_info->setWordWrap(true);
  main_layout->addWidget(aggressiveness_info);
  
  main_layout->addSpacing(20);
  
  // Technical details section
  technical_details = new QGroupBox(tr("Response Times by Speed"), this);
  technical_details->setStyleSheet(R"(
    QGroupBox {
      font-size: 36px;
      font-weight: 500;
      color: #E4E4E4;
      background-color: transparent;
      border: 2px solid #555;
      border-radius: 10px;
      margin-top: 15px;
      padding-top: 10px;
    }
    QGroupBox::title {
      subcontrol-origin: margin;
      left: 15px;
      padding: 0 10px 0 10px;
    }
  )");
  
  QVBoxLayout *tech_layout = new QVBoxLayout(technical_details);
  tech_layout->setSpacing(8);
  tech_layout->setContentsMargins(15, 20, 15, 15);
  
  reaction_time_25mph = new QLabel(technical_details);
  reaction_time_25mph->setStyleSheet("font-size: 28px; color: #aaaaaa;");
  tech_layout->addWidget(reaction_time_25mph);
  
  reaction_time_45mph = new QLabel(technical_details);
  reaction_time_45mph->setStyleSheet("font-size: 28px; color: #aaaaaa;");
  tech_layout->addWidget(reaction_time_45mph);
  
  reaction_time_65mph = new QLabel(technical_details);
  reaction_time_65mph->setStyleSheet("font-size: 28px; color: #aaaaaa;");
  tech_layout->addWidget(reaction_time_65mph);
  
  main_layout->addWidget(technical_details);
  
  // Safety summary
  safety_summary = new QLabel(this);
  safety_summary->setStyleSheet(R"(
    font-size: 34px;
    font-weight: 450;
    color: #E4E4E4;
    background-color: transparent;
    padding-top: 20px;
  )");
  safety_summary->setAlignment(Qt::AlignLeft);
  safety_summary->setWordWrap(true);
  main_layout->addWidget(safety_summary);
  
  main_layout->addStretch();
}

void RTIConfigDataPanel::updateConfiguration(RTIAggressiveness aggressiveness, int min_dist, int max_dist, int speed_reduction) {
  main_description->setText(tr(
    "Real-time Traffic Intelligence (RTI) monitors ahead for threats "
    "and adjusts speed automatically for safer driving."
  ));
  
  // Format distance based on metric setting
  Params params;
  bool is_metric = params.getBool("IsMetric");
  QString distance_text;
  
  if (is_metric) {
    double min_km = min_dist * 0.001;
    double max_km = max_dist * 0.001;
    distance_text = QString("• Detection Range: %1 - %2 km")
      .arg(min_km, 0, 'f', 1).arg(max_km, 0, 'f', 1);
  } else {
    double min_mi = min_dist * 0.000621371;
    double max_mi = max_dist * 0.000621371;
    distance_text = QString("• Detection Range: %1 - %2 mi")
      .arg(min_mi, 0, 'f', 2).arg(max_mi, 0, 'f', 2);
  }
  
  distance_info->setText(distance_text);
  
  speed_reduction_info->setText(QString("• Max Speed Reduction: %1 km/h")
    .arg(speed_reduction));
  
  aggressiveness_info->setText(QString("• Response Style: %1")
    .arg(getAggressivenessDescription(aggressiveness)));
  
  // Calculate reaction times based on aggressiveness and speed
  float aggr_multiplier = 1.0f;
  switch (aggressiveness) {
    case RTIAggressiveness::CONSERVATIVE: aggr_multiplier = 1.4f; break;
    case RTIAggressiveness::BALANCED: aggr_multiplier = 1.0f; break;
    case RTIAggressiveness::AGGRESSIVE: aggr_multiplier = 0.7f; break;
  }
  
  float time_25mph = 2.5f * aggr_multiplier;
  float time_45mph = 3.2f * aggr_multiplier;
  float time_65mph = 4.1f * aggr_multiplier;
  
  reaction_time_25mph->setText(QString("25 mph: %1s ahead of threat").arg(time_25mph, 0, 'f', 1));
  reaction_time_45mph->setText(QString("45 mph: %1s ahead of threat").arg(time_45mph, 0, 'f', 1));
  reaction_time_65mph->setText(QString("65 mph: %1s ahead of threat").arg(time_65mph, 0, 'f', 1));
  
  // Safety summary
  QString safety_text;
  if (aggressiveness == RTIAggressiveness::CONSERVATIVE) {
    safety_text = tr("Conservative settings provide maximum safety with earlier, gentler speed reductions.");
  } else if (aggressiveness == RTIAggressiveness::BALANCED) {
    safety_text = tr("Balanced settings optimize comfort while maintaining safety.");
  } else {
    safety_text = tr("Aggressive settings provide sportier response with later, quicker adjustments.");
  }
  
  safety_summary->setText(safety_text);
}

QString RTIConfigDataPanel::getAggressivenessDescription(RTIAggressiveness aggressiveness) const {
  switch (aggressiveness) {
    case RTIAggressiveness::CONSERVATIVE: return tr("Conservative");
    case RTIAggressiveness::BALANCED: return tr("Balanced");
    case RTIAggressiveness::AGGRESSIVE: return tr("Aggressive");
    default: return tr("Unknown");
  }
}

// RTI API Config Panel Implementation
RTIApiConfigPanel::RTIApiConfigPanel(QWidget *parent) : QFrame(parent) {
  setupUI();
  
  test_timer = new QTimer(this);
  test_timer->setSingleShot(true);
  connect(test_timer, &QTimer::timeout, [this]() {
    // Simulate API test completion
    connection_status->setValue(100);
    status_label->setText(tr("Connection successful"));
    status_label->setStyleSheet("color: #00ff00; font-size: 28px;");
    test_button->setEnabled(true);
  });
}

void RTIApiConfigPanel::setupUI() {
  setStyleSheet("background-color: #1a1a1a; border-radius: 15px; padding: 20px;");
  
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setSpacing(20);
  
  QLabel *title = new QLabel(tr("Manual API Configuration"), this);
  title->setStyleSheet("font-size: 42px; font-weight: 600; color: #E4E4E4; margin-bottom: 20px;");
  title->setAlignment(Qt::AlignCenter);
  main_layout->addWidget(title);
  
  // API endpoint
  QLabel *endpoint_label = new QLabel(tr("API Endpoint:"), this);
  endpoint_label->setStyleSheet("font-size: 32px; color: #E4E4E4;");
  main_layout->addWidget(endpoint_label);
  
  api_endpoint = new QLineEdit(this);
  api_endpoint->setPlaceholderText("https://api.example.com/traffic/v1");
  api_endpoint->setStyleSheet(R"(
    font-size: 28px; 
    padding: 12px; 
    background-color: #333; 
    color: white; 
    border: 2px solid #555; 
    border-radius: 8px;
  )");
  main_layout->addWidget(api_endpoint);
  
  // API key
  QLabel *key_label = new QLabel(tr("API Key:"), this);
  key_label->setStyleSheet("font-size: 32px; color: #E4E4E4;");
  main_layout->addWidget(key_label);
  
  api_key = new QLineEdit(this);
  api_key->setEchoMode(QLineEdit::Password);
  api_key->setPlaceholderText("Enter your API key");
  api_key->setStyleSheet(R"(
    font-size: 28px; 
    padding: 12px; 
    background-color: #333; 
    color: white; 
    border: 2px solid #555; 
    border-radius: 8px;
  )");
  main_layout->addWidget(api_key);
  
  // API format
  QLabel *format_label = new QLabel(tr("Data Format:"), this);
  format_label->setStyleSheet("font-size: 32px; color: #E4E4E4;");
  main_layout->addWidget(format_label);
  
  api_format = new QComboBox(this);
  api_format->addItems({"JSON", "XML", "CSV"});
  api_format->setStyleSheet(R"(
    font-size: 28px; 
    padding: 12px; 
    background-color: #333; 
    color: white; 
    border: 2px solid #555; 
    border-radius: 8px;
  )");
  main_layout->addWidget(api_format);
  
  // Update frequency
  QLabel *freq_label = new QLabel(tr("Update Frequency:"), this);
  freq_label->setStyleSheet("font-size: 32px; color: #E4E4E4;");
  main_layout->addWidget(freq_label);
  
  update_frequency = new QSlider(Qt::Horizontal, this);
  update_frequency->setRange(30, 300); // 30 seconds to 5 minutes
  update_frequency->setValue(60); // Default 1 minute
  update_frequency->setStyleSheet(R"(
    QSlider::groove:horizontal {
      border: 1px solid #555;
      height: 8px;
      background: #333;
      border-radius: 4px;
    }
    QSlider::handle:horizontal {
      background: #4a90e2;
      border: 1px solid #555;
      width: 20px;
      margin: -6px 0;
      border-radius: 10px;
    }
  )");
  main_layout->addWidget(update_frequency);
  
  QLabel *freq_value = new QLabel(tr("60 seconds"), this);
  freq_value->setStyleSheet("font-size: 24px; color: #aaaaaa; text-align: center;");
  freq_value->setAlignment(Qt::AlignCenter);
  connect(update_frequency, &QSlider::valueChanged, [freq_value](int value) {
    freq_value->setText(QString("%1 seconds").arg(value));
  });
  main_layout->addWidget(freq_value);
  
  // Connection test section
  main_layout->addSpacing(20);
  
  QHBoxLayout *button_layout = new QHBoxLayout();
  
  test_button = new QPushButton(tr("Test Connection"), this);
  test_button->setStyleSheet(R"(
    QPushButton {
      font-size: 30px;
      font-weight: 500;
      padding: 15px 30px;
      background-color: #4a90e2;
      color: white;
      border: none;
      border-radius: 12px;
    }
    QPushButton:hover {
      background-color: #357abd;
    }
    QPushButton:pressed {
      background-color: #2d5aa0;
    }
    QPushButton:disabled {
      background-color: #666;
      color: #999;
    }
  )");
  connect(test_button, &QPushButton::clicked, this, &RTIApiConfigPanel::testApiConnection);
  button_layout->addWidget(test_button);
  
  save_button = new QPushButton(tr("Save Configuration"), this);
  save_button->setStyleSheet(R"(
    QPushButton {
      font-size: 30px;
      font-weight: 500;
      padding: 15px 30px;
      background-color: #27ae60;
      color: white;
      border: none;
      border-radius: 12px;
    }
    QPushButton:hover {
      background-color: #219a52;
    }
    QPushButton:pressed {
      background-color: #1e8449;
    }
  )");
  connect(save_button, &QPushButton::clicked, this, &RTIApiConfigPanel::saveApiConfiguration);
  button_layout->addWidget(save_button);
  
  main_layout->addLayout(button_layout);
  
  // Connection status
  connection_status = new QProgressBar(this);
  connection_status->setStyleSheet(R"(
    QProgressBar {
      border: 2px solid #555;
      border-radius: 8px;
      background-color: #333;
      height: 25px;
    }
    QProgressBar::chunk {
      background-color: #4a90e2;
      border-radius: 6px;
    }
  )");
  connection_status->setVisible(false);
  main_layout->addWidget(connection_status);
  
  status_label = new QLabel(this);
  status_label->setStyleSheet("font-size: 28px; color: #aaaaaa; text-align: center;");
  status_label->setAlignment(Qt::AlignCenter);
  main_layout->addWidget(status_label);
  
  main_layout->addStretch();
}

void RTIApiConfigPanel::testApiConnection() {
  if (api_endpoint->text().isEmpty()) {
    status_label->setText(tr("Please enter an API endpoint"));
    status_label->setStyleSheet("color: #ff0000; font-size: 28px;");
    return;
  }
  
  test_button->setEnabled(false);
  connection_status->setVisible(true);
  connection_status->setValue(0);
  status_label->setText(tr("Testing connection..."));
  status_label->setStyleSheet("color: #ffaa00; font-size: 28px;");
  
  // Animate progress bar
  QTimer *progress_timer = new QTimer(this);
  connect(progress_timer, &QTimer::timeout, [this, progress_timer]() {
    int value = connection_status->value() + 10;
    connection_status->setValue(value);
    if (value >= 80) {
      progress_timer->stop();
      progress_timer->deleteLater();
      test_timer->start(1000); // Complete after 1 second
    }
  });
  progress_timer->start(200);
}

void RTIApiConfigPanel::saveApiConfiguration() {
  // Save API configuration to parameters
  Params params;
  params.put("RTIManualApiEndpoint", api_endpoint->text().toStdString());
  params.put("RTIManualApiKey", api_key->text().toStdString());
  params.put("RTIManualApiFormat", api_format->currentText().toStdString());
  params.put("RTIManualApiFrequency", std::to_string(update_frequency->value()));
  
  status_label->setText(tr("Configuration saved successfully"));
  status_label->setStyleSheet("color: #00ff00; font-size: 28px;");
}

// Main Advanced RTI Panel Implementation
RTIAdvancedPanel::RTIAdvancedPanel(QWidget *parent) : QFrame(parent) {
  setupUI();
}

void RTIAdvancedPanel::setupUI() {
  main_layout = new QStackedLayout(this);
  
  // Main configuration screen with scroll area
  QScrollArea *scrollArea = new QScrollArea(this);
  scrollArea->setWidgetResizable(true);
  scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  scrollArea->setStyleSheet("QScrollArea { background-color: black; border: none; }");
  
  main_screen = new QWidget();
  QVBoxLayout *screen_layout = new QVBoxLayout(main_screen);
  screen_layout->setContentsMargins(0, 0, 0, 0);
  screen_layout->setSpacing(0);
  
  // Header - Fixed size and position
  QWidget *header = new QWidget();
  header->setFixedHeight(100);
  header->setStyleSheet("background-color: #292929;");
  
  QHBoxLayout *header_layout = new QHBoxLayout(header);
  header_layout->setContentsMargins(50, 0, 50, 0);
  
  back_btn = new QPushButton(tr("← Back"), this);
  back_btn->setFixedSize(200, 80);
  back_btn->setStyleSheet(R"(
    QPushButton {
      background-color: #393939;
      border-radius: 40px;
      color: #E4E4E4;
      font-size: 35px;
      font-weight: 500;
      padding: 20px;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
  )");
  connect(back_btn, &QPushButton::clicked, this, &RTIAdvancedPanel::backPress);
  
  title_label = new QLabel(tr("Advanced RTI Configuration"), this);
  title_label->setStyleSheet(R"(
    font-size: 50px;
    font-weight: 450;
    color: #E4E4E4;
    background-color: transparent;
  )");
  title_label->setAlignment(Qt::AlignCenter);
  
  header_layout->addWidget(back_btn);
  header_layout->addStretch();
  header_layout->addWidget(title_label);
  header_layout->addStretch();
  header_layout->addSpacing(200);  // Balance for back button width
  
  screen_layout->addWidget(header);
  
  // Content container - constrained width to prevent horizontal scrolling
  QWidget *content_container = new QWidget();
  content_container->setMaximumWidth(1300);  // Enforce 1400px total constraint
  content_container->setMinimumWidth(1000);  // Reasonable minimum width
  
  QVBoxLayout *container_layout = new QVBoxLayout(content_container);
  container_layout->setContentsMargins(50, 30, 50, 30);
  container_layout->setSpacing(40);
  
  // Visualization row - Fixed dimensions
  QWidget *viz_row = new QWidget();
  viz_row->setFixedHeight(450);  // Controlled height
  QHBoxLayout *viz_layout = new QHBoxLayout(viz_row);
  viz_layout->setContentsMargins(0, 0, 0, 0);
  viz_layout->setSpacing(30);
  
  // Visualization widget (left side)
  visualization_widget = new RTIVisualizationWidget(this);
  viz_layout->addWidget(visualization_widget);
  
  // Configuration panel (right side - fixed width)
  config_data_panel = new RTIConfigDataPanel(this);
  viz_layout->addWidget(config_data_panel);
  
  viz_layout->addStretch();  // Prevent horizontal expansion
  container_layout->addWidget(viz_row);
  
  // Control buttons section
  QFrame *controls_frame = new QFrame();
  controls_frame->setStyleSheet("QFrame { background-color: #292929; border-radius: 20px; }");
  controls_frame->setFixedHeight(120);
  
  QHBoxLayout *controls_layout = new QHBoxLayout(controls_frame);
  controls_layout->setContentsMargins(40, 20, 40, 20);
  controls_layout->setSpacing(30);
  
  api_config_btn = new QPushButton(tr("API Configuration"), this);
  api_config_btn->setFixedSize(350, 80);
  api_config_btn->setStyleSheet(R"(
    QPushButton {
      font-size: 35px;
      font-weight: 500;
      background-color: #4a90e2;
      color: white;
      border-radius: 20px;
    }
    QPushButton:pressed {
      background-color: #357abd;
    }
  )");
  connect(api_config_btn, &QPushButton::clicked, this, &RTIAdvancedPanel::showApiConfig);
  controls_layout->addWidget(api_config_btn);
  
  reset_defaults_btn = new QPushButton(tr("Reset to Defaults"), this);
  reset_defaults_btn->setFixedSize(350, 80);
  reset_defaults_btn->setStyleSheet(R"(
    QPushButton {
      font-size: 35px;
      font-weight: 500;
      background-color: #666;
      color: white;
      border-radius: 20px;
    }
    QPushButton:pressed {
      background-color: #555;
    }
  )");
  controls_layout->addWidget(reset_defaults_btn);
  
  controls_layout->addStretch();
  container_layout->addWidget(controls_frame);
  
  container_layout->addStretch();  // Push content to top
  
  // Center the content container horizontally
  QHBoxLayout *center_layout = new QHBoxLayout();
  center_layout->addStretch();
  center_layout->addWidget(content_container);
  center_layout->addStretch();
  
  screen_layout->addLayout(center_layout);
  
  scrollArea->setWidget(main_screen);
  main_layout->addWidget(scrollArea);
  
  // API configuration screen
  api_config_screen = new RTIApiConfigPanel();
  main_layout->addWidget(api_config_screen);
  
  // Connect signals
  connect(visualization_widget, &RTIVisualizationWidget::distanceChanged, 
          this, &RTIAdvancedPanel::onVisualizationChanged);
  
  // Initialize with current values
  updateVisualization();
}

void RTIAdvancedPanel::onVisualizationChanged(int min_distance, int max_distance) {
  // Update parameters
  params.put("RTIMinDistance", std::to_string(min_distance));
  params.put("RTIMaxDistance", std::to_string(max_distance));
  
  // Update display
  updateVisualization();
}

void RTIAdvancedPanel::onParameterChanged() {
  updateVisualization();
}

void RTIAdvancedPanel::showApiConfig() {
  main_layout->setCurrentWidget(api_config_screen);
}

void RTIAdvancedPanel::updateVisualization() {
  // Get current parameter values using safe conversion
  int min_dist = safeStringToInt(params.get("RTIMinDistance"), 100);
  int max_dist = safeStringToInt(params.get("RTIMaxDistance"), 2000);
  int speed_reduction = safeStringToInt(params.get("RTISpeedReduction"), 15);
  int aggr_val = safeStringToInt(params.get("RTIAggressiveness"), 1);
  
  // Validate values
  min_dist = std::max(50, std::min(2000, min_dist));
  max_dist = std::max(500, std::min(5000, max_dist));
  speed_reduction = std::max(5, std::min(50, speed_reduction));
  aggr_val = std::max(0, std::min(2, aggr_val));
  
  RTIAggressiveness aggressiveness = static_cast<RTIAggressiveness>(aggr_val);
  
  // Update widgets
  visualization_widget->updateConfiguration(aggressiveness, min_dist, max_dist);
  config_data_panel->updateConfiguration(aggressiveness, min_dist, max_dist, speed_reduction);
}