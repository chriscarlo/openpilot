/**
 * Professional Road Visualization Widget for VTSC Configuration
 * Setup screen for configuring pre-emptive curve slowing behavior
 */

#pragma once

#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
#include <QLabel>
#include <QVBoxLayout>
#include <QPropertyAnimation>
#include <QGraphicsDropShadowEffect>
#include <QLinearGradient>
#include <QRadialGradient>

#include "common/params.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class ConfigurationDataPanel : public QFrame {
  Q_OBJECT

public:
  explicit ConfigurationDataPanel(QWidget *parent = nullptr);
  void updateConfiguration(float aggressiveness);

private:
  void setupUI();
  
  // Primary explanation
  QLabel *main_description;
  
  // Speed context examples
  QLabel *residential_example;
  QLabel *urban_example; 
  QLabel *highway_example;
  
  // Technical details
  QLabel *multiplier_value;
  QLabel *timing_range;
  QLabel *comfort_description;
  
};

class ProfessionalRoadWidget : public QWidget {
  Q_OBJECT

public:
  explicit ProfessionalRoadWidget(QWidget *parent = nullptr);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

signals:
  void aggressivenessChanged(float value);

private:
  void updateAggressiveness(float value);
  void drawProfessionalRoad(QPainter &painter);
  void drawVehicle(QPainter &painter, float y_position);
  void drawAnticipationVisualization(QPainter &painter);
  void drawInteractiveMarker(QPainter &painter);
  
  // Core state
  float aggressiveness;
  float marker_position;  // 0.0 (bottom) to 1.0 (curve start)
  bool is_dragging;
  bool is_hovering;
  
  // Professional visual constants
  struct Theme {
    // Road colors
    QColor road_surface = QColor(55, 55, 65);
    QColor road_edge = QColor(255, 255, 255, 200);
    QColor center_line = QColor(255, 193, 7, 180);
    
    // Anticipation zone
    QColor zone_early = QColor(255, 152, 0, 60);    // Orange - early braking
    QColor zone_balanced = QColor(255, 193, 7, 50); // Yellow - balanced
    QColor zone_late = QColor(76, 175, 80, 40);     // Green - late braking
    
    // Vehicle
    QColor car_body = QColor(66, 165, 245);
    QColor car_glass = QColor(33, 150, 243, 180);
    QColor car_detail = QColor(255, 255, 255, 200);
    
    // Marker
    QColor marker_normal = QColor(255, 193, 7);
    QColor marker_hover = QColor(255, 214, 0);
    QColor marker_active = QColor(255, 235, 59);
  } theme;
  
  // Layout calculations
  struct Layout {
    int road_width;
    int lane_width;
    QPointF road_start;
    QPointF road_end;
    QPointF curve_start;
    QPointF curve_apex;
    QPointF curve_end;
    float perspective_factor;
  } layout;
  
  void recalculateLayout();
  QPointF getPerspectivePoint(const QPointF &point, float distance_ratio);
  
  Params params;
};

class AnticipationConfigPanel : public QFrame {
  Q_OBJECT

public:
  explicit AnticipationConfigPanel(QWidget *parent = nullptr);

signals:
  void backPress();

private:
  void setupUI();
  void applyProfessionalStyling();
  
  ProfessionalRoadWidget *road_widget;
  ConfigurationDataPanel *data_panel;
  QPushButton *back_btn;
  QLabel *title_label;
  
};