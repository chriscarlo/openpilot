/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QWidget>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QStackedLayout>
#include <QPainter>
#include <QPaintEvent>
#include <QMouseEvent>
#include <QSlider>
#include <QTimer>
#include <QProgressBar>
#include <QGroupBox>
#include <QComboBox>
#include <QLineEdit>

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_control.h"

// Interactive RTI Threat Zone Visualization Widget
class RTIVisualizationWidget : public QWidget {
  Q_OBJECT

public:
  RTIVisualizationWidget(QWidget *parent = nullptr);
  void updateConfiguration(RTIAggressiveness aggressiveness, int min_distance, int max_distance);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

signals:
  void distanceChanged(int min_distance, int max_distance);

private:
  struct ThreatZone {
    float min_distance;
    float max_distance;
    QColor color;
    QString label;
  };

  struct VisualizationLayout {
    QPointF road_start;
    QPointF road_end;
    float road_width;
    float scale_factor;
  };

  void recalculateLayout();
  void drawRoadScene(QPainter &painter);
  void drawThreatZones(QPainter &painter);
  void drawVehicle(QPainter &painter, float y_position);
  void drawThreatIndicators(QPainter &painter);
  QColor getThreatColor(float distance) const;

  VisualizationLayout layout;
  RTIAggressiveness current_aggressiveness;
  int min_distance_m;
  int max_distance_m;
  bool is_dragging;
  bool is_hovering;
  QTimer *animation_timer;
  float animation_phase;

  // Cached rendering objects for performance
  QFont label_font;
  QFont distance_font;
};

// RTI Configuration Data Panel with real-time updates
class RTIConfigDataPanel : public QFrame {
  Q_OBJECT

public:
  RTIConfigDataPanel(QWidget *parent = nullptr);
  void updateConfiguration(RTIAggressiveness aggressiveness, int min_dist, int max_dist, int speed_reduction);

private:
  void setupUI();
  QString getAggressivenessDescription(RTIAggressiveness aggressiveness) const;
  
  QLabel *main_description;
  QLabel *distance_info;
  QLabel *speed_reduction_info;
  QLabel *aggressiveness_info;
  QLabel *safety_summary;
  QGroupBox *technical_details;
  QLabel *reaction_time_25mph;
  QLabel *reaction_time_45mph;
  QLabel *reaction_time_65mph;
};

// Advanced API Configuration Panel for Manual data sources
class RTIApiConfigPanel : public QFrame {
  Q_OBJECT

public:
  RTIApiConfigPanel(QWidget *parent = nullptr);

private slots:
  void testApiConnection();
  void saveApiConfiguration();

private:
  void setupUI();
  
  QLineEdit *api_endpoint;
  QLineEdit *api_key;
  QComboBox *api_format;
  QSlider *update_frequency;
  QPushButton *test_button;
  QPushButton *save_button;
  QProgressBar *connection_status;
  QLabel *status_label;
  
  QTimer *test_timer;
};

// Main Advanced RTI Configuration Panel
class RTIAdvancedPanel : public QFrame {
  Q_OBJECT

public:
  RTIAdvancedPanel(QWidget *parent = nullptr);

signals:
  void backPress();

private slots:
  void onVisualizationChanged(int min_distance, int max_distance);
  void onParameterChanged();
  void showApiConfig();

private:
  void setupUI();
  void updateVisualization();
  
  QStackedLayout *main_layout;
  QWidget *main_screen;
  RTIApiConfigPanel *api_config_screen;
  
  QPushButton *back_btn;
  QLabel *title_label;
  
  RTIVisualizationWidget *visualization_widget;
  RTIConfigDataPanel *config_data_panel;
  
  // Quick configuration controls
  QSlider *aggressiveness_slider;
  QSlider *min_distance_slider;  
  QSlider *max_distance_slider;
  QSlider *speed_reduction_slider;
  
  QPushButton *api_config_btn;
  QPushButton *reset_defaults_btn;
  
  Params params;
};