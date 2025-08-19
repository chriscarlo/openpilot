/**
 * Professional Road Visualization Widget Implementation
 * World-class UX design for VTSC configuration
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/road_visualization_widget_v2.h"
#include <QPainterPath>
#include <QHBoxLayout>
#include <algorithm>
#include <cmath>

// ConfigurationDataPanel Implementation
ConfigurationDataPanel::ConfigurationDataPanel(QWidget *parent) : QFrame(parent) {
  setupUI();
}

void ConfigurationDataPanel::setupUI() {
  setFixedWidth(650);  // Optimized width for better space utilization
  // No background or borders - plain black
  setStyleSheet("background-color: transparent; border: none;");
  
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(20, 10, 20, 10);  // Reduced vertical margins  
  main_layout->setSpacing(12);
  
  // Main description at the TOP - what this setting does
  main_description = new QLabel(this);
  main_description->setWordWrap(true);
  main_description->setStyleSheet(R"(
    font-size: 42px;
    font-weight: 400;
    color: #E4E4E4;
    background-color: transparent;
    padding-bottom: 25px;
  )");
  main_layout->addWidget(main_description);
  
  main_layout->addSpacing(5);
  
  // Speed context examples section
  QLabel *examples_header = new QLabel(tr("Anticipation timing by speed:"), this);
  examples_header->setStyleSheet(R"(
    font-size: 45px;
    font-weight: 500;
    color: #E4E4E4;
    background-color: transparent;
  )");
  main_layout->addWidget(examples_header);
  
  // Speed examples
  residential_example = new QLabel(this);
  residential_example->setStyleSheet(R"(
    font-size: 35px;
    color: #aaaaaa;
    padding-left: 20px;
    background-color: transparent;
  )");
  residential_example->setWordWrap(true);
  
  urban_example = new QLabel(this);
  urban_example->setStyleSheet(R"(
    font-size: 35px;
    color: #aaaaaa;
    padding-left: 20px;
    background-color: transparent;
  )");
  urban_example->setWordWrap(true);
  
  highway_example = new QLabel(this);
  highway_example->setStyleSheet(R"(
    font-size: 35px;
    color: #aaaaaa;
    padding-left: 20px;
    background-color: transparent;
  )");
  highway_example->setWordWrap(true);
  
  main_layout->addWidget(residential_example);
  main_layout->addWidget(urban_example);
  main_layout->addWidget(highway_example);
  
  main_layout->addSpacing(20);
  
  // Technical details section
  QLabel *tech_header = new QLabel(tr("Configuration:"), this);
  tech_header->setStyleSheet(R"(
    font-size: 45px;
    font-weight: 500;
    color: #E4E4E4;
    background-color: transparent;
  )");
  main_layout->addWidget(tech_header);
  
  // Technical values
  multiplier_value = new QLabel(this);
  multiplier_value->setStyleSheet(R"(
    font-size: 35px;
    color: #aaaaaa;
    padding-left: 20px;
    background-color: transparent;
  )");
  
  timing_range = new QLabel(this);
  timing_range->setStyleSheet(R"(
    font-size: 35px;
    color: #aaaaaa;
    padding-left: 20px;
    background-color: transparent;
  )");
  timing_range->setWordWrap(true);
  
  main_layout->addWidget(multiplier_value);
  main_layout->addWidget(timing_range);
  
  main_layout->addSpacing(20);
  
  // Comfort description
  comfort_description = new QLabel(this);
  comfort_description->setStyleSheet(R"(
    font-size: 40px;
    font-weight: 450;
    color: #E4E4E4;
    background-color: transparent;
  )");
  comfort_description->setAlignment(Qt::AlignLeft);
  comfort_description->setWordWrap(true);
  
  main_layout->addWidget(comfort_description);
  main_layout->addStretch();
}

void ConfigurationDataPanel::updateConfiguration(float aggressiveness) {
  // Main description at top
  main_description->setText(tr(
    "Controls when the vehicle begins slowing for curves. "
    "Drag the car to adjust anticipation timing."
  ));
  
  // Calculate actual timing values based on the formula from vision_turn_controller.py
  float base_reaction = 1.185f;
  
  // Speed context examples - showing timing in seconds
  float residential_time = base_reaction * (15.0f/15.0f) * aggressiveness * 1.144f;  // 30mph example
  float urban_time = base_reaction * (22.0f/15.0f) * aggressiveness * 1.200f;       // 50mph example  
  float highway_time = base_reaction * (31.0f/15.0f) * aggressiveness * 1.384f;     // 70mph example
  
  residential_example->setText(QString("  30 mph: %1 seconds before curve")
    .arg(residential_time, 0, 'f', 1));
    
  urban_example->setText(QString("  50 mph: %1 seconds before curve")
    .arg(urban_time, 0, 'f', 1));
    
  highway_example->setText(QString("  70 mph: %1 seconds before curve")
    .arg(highway_time, 0, 'f', 1));
  
  // Technical details
  multiplier_value->setText(QString("  Settings: %1x multiplier").arg(aggressiveness, 0, 'f', 2));
  
  // Show percentage for clearer understanding
  int percentage = static_cast<int>((aggressiveness - 0.5) / 1.5 * 100);
  timing_range->setText(QString("  Anticipation level: %1%").arg(percentage));
  
  // Comfort level description
  QString comfort_text;
  if (aggressiveness < 0.7) {
    comfort_text = tr("Sportier - Later braking");
  } else if (aggressiveness < 1.3) {
    comfort_text = tr("Balanced - Optimal comfort");
  } else {
    comfort_text = tr("Conservative - Earlier braking");
  }
  comfort_description->setText(comfort_text);
}

// ProfessionalRoadWidget Implementation
ProfessionalRoadWidget::ProfessionalRoadWidget(QWidget *parent) : QWidget(parent) {
  // Optimized size to fit within available space without creating blank space above
  setMinimumSize(400, 500);
  setMaximumSize(700, 650);  // Reduced max height to fit better in panel
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);  // Changed to Preferred to prevent over-expansion
  
  // Get current aggressiveness
  std::string aggr_str = params.get("VisionTurnSpeedControlAggressiveness");
  aggressiveness = aggr_str.empty() ? 1.0f : std::stof(aggr_str);
  marker_position = 1.0f - ((aggressiveness - 0.5f) / 1.5f); // Inverted: bottom = more time
  
  is_dragging = false;
  is_hovering = false;
  
  // Enable mouse tracking
  setMouseTracking(true);
  setCursor(Qt::PointingHandCursor);
}

void ProfessionalRoadWidget::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  recalculateLayout();
}

void ProfessionalRoadWidget::recalculateLayout() {
  // Professional layout with proper perspective
  layout.road_width = width() * 0.5;
  layout.lane_width = layout.road_width * 0.45;
  
  // Road runs from bottom to top with perspective
  layout.road_start = QPointF(width() / 2, height() - 50);
  layout.road_end = QPointF(width() / 2, 100);
  
  // Curve parameters
  layout.curve_start = QPointF(width() / 2, height() * 0.3);
  layout.curve_apex = QPointF(width() * 0.75, height() * 0.15);
  layout.curve_end = QPointF(width() * 0.85, height() * 0.05);
  
  layout.perspective_factor = 0.4; // How much the road narrows at the top
}

QPointF ProfessionalRoadWidget::getPerspectivePoint(const QPointF &point, float distance_ratio) {
  // Apply perspective transformation
  float perspective_scale = 1.0f - (distance_ratio * layout.perspective_factor);
  float x_offset = (point.x() - width() / 2) * perspective_scale;
  return QPointF(width() / 2 + x_offset, point.y());
}

void ProfessionalRoadWidget::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
  
  // Black background
  painter.fillRect(rect(), Qt::black);
  
  // Draw in correct order
  drawProfessionalRoad(painter);
  drawAnticipationVisualization(painter);
  drawInteractiveMarker(painter);
}

void ProfessionalRoadWidget::drawProfessionalRoad(QPainter &painter) {
  // Draw road with perspective
  QPainterPath road_path;
  
  // Calculate road edges with perspective
  float bottom_width = layout.road_width;
  float top_width = layout.road_width * (1.0f - layout.perspective_factor);
  
  // Straight section
  QPointF bottom_left(layout.road_start.x() - bottom_width/2, layout.road_start.y());
  QPointF bottom_right(layout.road_start.x() + bottom_width/2, layout.road_start.y());
  QPointF curve_left(layout.curve_start.x() - top_width*0.7, layout.curve_start.y());
  QPointF curve_right(layout.curve_start.x() + top_width*0.7, layout.curve_start.y());
  
  road_path.moveTo(bottom_left);
  road_path.lineTo(curve_left);
  
  // Curve section with smooth bezier
  QPointF control1_left(curve_left.x(), curve_left.y() - 50);
  QPointF control2_left(layout.curve_apex.x() - top_width*0.4, layout.curve_apex.y() + 20);
  QPointF end_left(layout.curve_end.x() - top_width*0.3, layout.curve_end.y());
  
  road_path.cubicTo(control1_left, control2_left, end_left);
  
  // Right edge of curve
  QPointF end_right(layout.curve_end.x() + top_width*0.3, layout.curve_end.y());
  QPointF control2_right(layout.curve_apex.x() + top_width*0.4, layout.curve_apex.y() + 20);
  QPointF control1_right(curve_right.x(), curve_right.y() - 50);
  
  road_path.lineTo(end_right);
  road_path.cubicTo(control2_right, control1_right, curve_right);
  road_path.lineTo(bottom_right);
  road_path.closeSubpath();
  
  // Fill road surface
  painter.fillPath(road_path, theme.road_surface);
  
  // Draw road edges
  painter.setPen(QPen(theme.road_edge, 3));
  painter.drawPath(road_path);
  
  // Draw center line with dashes
  QPainterPath center_line;
  center_line.moveTo(layout.road_start);
  
  // Straight section center line
  for (float t = 0; t < 1.0; t += 0.05) {
    float y = layout.road_start.y() + t * (layout.curve_start.y() - layout.road_start.y());
    
    if (fmod(t * 20, 2) < 1) {  // Dashed pattern
      painter.setPen(QPen(theme.center_line, 2));
      painter.drawLine(QPointF(layout.road_start.x(), y),
                      QPointF(layout.road_start.x(), y - 10));
    }
  }
  
  // Curve center line
  QPainterPath curve_center;
  curve_center.moveTo(layout.curve_start);
  QPointF curve_control1(layout.curve_start.x(), layout.curve_start.y() - 50);
  QPointF curve_control2(layout.curve_apex.x(), layout.curve_apex.y() + 20);
  curve_center.cubicTo(curve_control1, curve_control2, layout.curve_end);
  
  painter.setPen(QPen(theme.center_line, 2, Qt::DashLine));
  painter.drawPath(curve_center);
}

void ProfessionalRoadWidget::drawAnticipationVisualization(QPainter &painter) {
  // Only draw if marker is not at the starting position
  if (marker_position <= 0.01) return;
  
  // Calculate anticipation zone based on marker position
  float zone_start_y = layout.road_start.y() + marker_position * (layout.curve_start.y() - layout.road_start.y());
  float zone_end_y = layout.curve_start.y();
  
  // Create gradient that transitions from red to green based on distance
  QLinearGradient zone_gradient(0, zone_end_y, 0, zone_start_y);
  
  // Color progression from curve (red) to far away (green)
  zone_gradient.setColorAt(0.0, QColor(255, 0, 0, 100));      // Red at curve
  zone_gradient.setColorAt(0.3, QColor(255, 127, 0, 80));     // Orange
  zone_gradient.setColorAt(0.6, QColor(255, 255, 0, 60));     // Yellow
  zone_gradient.setColorAt(1.0, QColor(0, 255, 0, 40));       // Green far away
  
  // Draw anticipation zone with perspective
  QPainterPath zone_path;
  
  float start_width = layout.road_width * 0.8 * (1.0f - layout.perspective_factor * marker_position);
  float end_width = layout.road_width * 0.8 * (1.0f - layout.perspective_factor);
  
  zone_path.moveTo(layout.road_start.x() - start_width/2, zone_start_y);
  zone_path.lineTo(layout.road_start.x() - end_width/2, zone_end_y);
  zone_path.lineTo(layout.road_start.x() + end_width/2, zone_end_y);
  zone_path.lineTo(layout.road_start.x() + start_width/2, zone_start_y);
  zone_path.closeSubpath();
  
  painter.fillPath(zone_path, zone_gradient);
  
  // Draw time grid lines
  painter.setPen(QPen(QColor(255, 255, 255, 30), 1));
  QFont time_font("Inter", 30);
  painter.setFont(time_font);
  
  // Draw time markers at 1s intervals
  float total_time = (1.0f - marker_position) * 5.0f;  // Inverted: more time when further from curve
  int num_marks = static_cast<int>(total_time);
  
  for (int i = 1; i <= num_marks && i <= 5; i++) {
    float t = 1.0f - (i / 5.0f);  // Inverted to match the new time logic
    float y = layout.road_start.y() + t * (layout.curve_start.y() - layout.road_start.y());
    
    // Draw line
    painter.drawLine(0, y, width(), y);
    
    // Draw time label
    painter.setPen(QColor(255, 255, 255, 100));
    painter.drawText(QRect(width() - 100, y - 15, 80, 30), 
                    Qt::AlignRight | Qt::AlignVCenter, 
                    QString("%1s").arg(i));
    painter.setPen(QPen(QColor(255, 255, 255, 30), 1));
  }
}

void ProfessionalRoadWidget::drawVehicle(QPainter &painter, float y_position) {
  // Draw stylized vehicle at the marker position
  const float car_width = 60;
  const float car_height = 100;
  const float car_x = layout.road_start.x() - car_width/2;
  const float car_y = y_position - car_height/2;
  
  // Car shadow
  painter.setPen(Qt::NoPen);
  painter.setBrush(QColor(0, 0, 0, 80));
  painter.drawEllipse(QPointF(layout.road_start.x(), car_y + car_height - 10), 40, 15);
  
  // Car body
  painter.setBrush(theme.car_body);
  QPainterPath car_body;
  car_body.addRoundedRect(car_x, car_y, car_width, car_height, 15, 25);
  painter.fillPath(car_body, theme.car_body);
  
  // Windshield
  QPainterPath windshield;
  windshield.moveTo(car_x + 10, car_y + 25);
  windshield.lineTo(car_x + 15, car_y + 15);
  windshield.lineTo(car_x + car_width - 15, car_y + 15);
  windshield.lineTo(car_x + car_width - 10, car_y + 25);
  windshield.closeSubpath();
  painter.fillPath(windshield, theme.car_glass);
  
  // Headlights
  painter.setBrush(QColor(255, 255, 200, 150));
  painter.drawEllipse(QPointF(car_x + 15, car_y + 10), 8, 6);
  painter.drawEllipse(QPointF(car_x + car_width - 15, car_y + 10), 8, 6);
  
  // Car outline
  painter.setPen(QPen(theme.car_detail, 2));
  painter.setBrush(Qt::NoBrush);
  painter.drawPath(car_body);
}

void ProfessionalRoadWidget::drawInteractiveMarker(QPainter &painter) {
  float marker_y = layout.road_start.y() + marker_position * (layout.curve_start.y() - layout.road_start.y());
  
  // Calculate road width at marker position
  float width_ratio = marker_position;
  float road_width_at_marker = layout.road_width * (1.0f - layout.perspective_factor * width_ratio);
  
  // Draw marker line across road
  QColor marker_color = is_dragging ? QColor(255, 255, 255, 200) : QColor(255, 255, 255, 120);
  
  // Main marker line
  painter.setPen(QPen(marker_color, 3, Qt::DashLine));
  painter.drawLine(layout.road_start.x() - road_width_at_marker/2 - 30, marker_y,
                  layout.road_start.x() + road_width_at_marker/2 + 30, marker_y);
  
  // Draw the car at this position
  drawVehicle(painter, marker_y);
  
  // Time label - inverted so dragging away from curve increases time
  float time_seconds = (1.0f - marker_position) * 5.0f;  // Max 5 seconds range
  QString time_text = QString("%1s").arg(time_seconds, 0, 'f', 1);
  
  QFont font("Inter", 50, QFont::DemiBold);
  painter.setFont(font);
  painter.setPen(Qt::white);
  
  // Draw time in larger font next to the car
  QRect text_rect(layout.road_start.x() + road_width_at_marker/2 + 40, 
                  marker_y - 25, 150, 50);
  painter.drawText(text_rect, Qt::AlignLeft | Qt::AlignVCenter, time_text);
}

void ProfessionalRoadWidget::mousePressEvent(QMouseEvent *event) {
  float click_y = event->pos().y();
  
  // Allow clicking anywhere in the road area to position the marker
  bool in_road_area = click_y >= layout.curve_start.y() - 50 && click_y <= layout.road_start.y() + 50;
  
  if (in_road_area) {
    is_dragging = true;
    setCursor(Qt::ClosedHandCursor);
    
    // Immediately move marker to click position for responsive interaction
    float road_height = layout.road_start.y() - layout.curve_start.y();
    marker_position = (layout.road_start.y() - click_y) / road_height;
    marker_position = std::max(0.0f, std::min(1.0f, marker_position));
    
    update();
    event->accept();  // Prevent scroll propagation
  } else {
    event->ignore();  // Allow parent to handle scrolling
    QWidget::mousePressEvent(event);
  }
}

void ProfessionalRoadWidget::mouseMoveEvent(QMouseEvent *event) {
  QPointF pos = event->pos();
  
  if (is_dragging) {
    float click_y = pos.y();
    
    // Direct marker positioning for smooth dragging
    float road_height = layout.road_start.y() - layout.curve_start.y();
    marker_position = (layout.road_start.y() - click_y) / road_height;
    marker_position = std::max(0.0f, std::min(1.0f, marker_position));
    update();
    event->accept();  // Prevent scroll propagation during drag
  } else {
    // Check for hover near the car/marker
    float marker_y = layout.road_start.y() + marker_position * (layout.curve_start.y() - layout.road_start.y());
    float distance = std::abs(pos.y() - marker_y);
    
    bool was_hovering = is_hovering;
    is_hovering = distance < 60;  // Larger hover area for the car
    
    if (is_hovering != was_hovering) {
      update();
    }
    
    setCursor(is_hovering ? Qt::OpenHandCursor : Qt::PointingHandCursor);
    event->ignore();  // Allow parent to handle scrolling
    QWidget::mouseMoveEvent(event);
  }
}

void ProfessionalRoadWidget::mouseReleaseEvent(QMouseEvent *event) {
  if (is_dragging) {
    is_dragging = false;
    setCursor(is_hovering ? Qt::OpenHandCursor : Qt::PointingHandCursor);
    
    // Convert position to aggressiveness (inverted)
    float new_aggressiveness = 0.5f + (1.0f - marker_position) * 1.5f;
    updateAggressiveness(new_aggressiveness);
    
    event->accept();  // Prevent scroll propagation
  } else {
    event->ignore();  // Allow parent to handle scrolling
    QWidget::mouseReleaseEvent(event);
  }
}

void ProfessionalRoadWidget::updateAggressiveness(float value) {
  aggressiveness = std::max(0.5f, std::min(2.0f, value));
  params.put("VisionTurnSpeedControlAggressiveness", std::to_string(aggressiveness));
  emit aggressivenessChanged(aggressiveness);
}

// AnticipationConfigPanel Implementation
AnticipationConfigPanel::AnticipationConfigPanel(QWidget *parent) : QFrame(parent) {
  setupUI();
  applyProfessionalStyling();
}

void AnticipationConfigPanel::setupUI() {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);
  
  // Header
  QWidget *header = new QWidget(this);
  header->setFixedHeight(100);
  header->setStyleSheet("background-color: #292929;");
  
  QHBoxLayout *header_layout = new QHBoxLayout(header);
  header_layout->setContentsMargins(30, 0, 30, 0);
  
  // Back button (consistent with other panels)
  back_btn = new QPushButton(tr("← Back"), this);
  back_btn->setObjectName("back_btn");
  back_btn->setFixedSize(200, 100);
  back_btn->setStyleSheet(R"(
    QPushButton#back_btn {
      background-color: #393939;
      border-radius: 50px;
      color: #E4E4E4;
      font-size: 35px;
      font-weight: 500;
      padding: 20px;
    }
    QPushButton#back_btn:pressed {
      background-color: #4a4a4a;
    }
  )");
  connect(back_btn, &QPushButton::clicked, this, &AnticipationConfigPanel::backPress);
  
  // Title
  title_label = new QLabel(tr("Curve Anticipation Settings"), this);
  title_label->setStyleSheet(R"(
    font-size: 50px;
    font-weight: 450;
    color: #E4E4E4;
    background-color: transparent;
  )");
  
  header_layout->addWidget(back_btn);
  header_layout->addStretch();
  header_layout->addWidget(title_label);
  header_layout->addStretch();
  header_layout->addSpacing(200);  // Balance for wider back button
  
  main_layout->addWidget(header);
  
  // Content area
  QWidget *content = new QWidget(this);
  QHBoxLayout *content_layout = new QHBoxLayout(content);
  content_layout->setContentsMargins(15, 0, 15, 15);  // ZERO top margin - fixes alignment issue
  content_layout->setSpacing(20);
  
  // Road visualization (left side)
  road_widget = new ProfessionalRoadWidget(this);
  content_layout->addWidget(road_widget, 1);
  
  // Data panel (right side)
  data_panel = new ConfigurationDataPanel(this);
  content_layout->addWidget(data_panel);
  
  main_layout->addWidget(content);
  
  // Connect road widget to data panel
  connect(road_widget, &ProfessionalRoadWidget::aggressivenessChanged,
          data_panel, &ConfigurationDataPanel::updateConfiguration);
  
  // Initial update
  std::string aggr_str = Params().get("VisionTurnSpeedControlAggressiveness");
  float current_aggr = aggr_str.empty() ? 1.0f : std::stof(aggr_str);
  data_panel->updateConfiguration(current_aggr);
}

void AnticipationConfigPanel::applyProfessionalStyling() {
  setStyleSheet("background-color: black;");
}