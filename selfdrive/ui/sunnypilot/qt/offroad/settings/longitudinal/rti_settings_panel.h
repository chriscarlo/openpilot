/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QComboBox>
#include <QSlider>
#include <QScrollArea>
#include <QLabel>
#include <fstream>
#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_control.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class RTISettingsPanel : public QStackedWidget {
  Q_OBJECT

public:
  RTISettingsPanel(QWidget *parent = nullptr);
  void refresh();
  void showEvent(QShowEvent *event) override;

signals:
  void backPress();
  void advancedSettingsRequested();

private:
  Params params;
  QFrame *subPanelFrame;
  
  // UI controls
  QComboBox *rti_source_combo;
  QComboBox *rti_filter_combo;
  QComboBox *rti_aggr_combo;
  QSlider *rti_min_slider;
  QSlider *rti_max_slider;
  QSlider *rti_speed_slider;
  ToggleSP *rti_hud_toggle;
  ToggleSP *rti_audio_toggle;
  
  void loadWazeApiKey();
  
  // Metric/Imperial conversion constants
  static constexpr double METERS_TO_MILES = 0.000621371;
  static constexpr double MILES_TO_METERS = 1609.344;
  static constexpr double METERS_TO_KM = 0.001;
  static constexpr double KM_TO_METERS = 1000.0;
  
  // Slider increments for each unit system
  static constexpr double IMPERIAL_INCREMENT_MI = 0.25;  // 0.25 miles
  static constexpr double METRIC_INCREMENT_KM = 0.5;     // 0.5 km
  
  // Helper methods for unit conversion and slider management
  bool isMetricSystem();
  void configureDistanceSliders();
  void validateAndMigrateParameters();
  QString formatDistanceLabel(int meters_value, bool is_minimum);
  int snapToValidIncrement(int meters);

  static QString sourceDescription(RTISourceType type = RTISourceType::DISABLED) {
    QString disabled_str = tr("⦿ Disabled: RTI system is completely disabled");
    QString waze_str = tr("⦿ Waze: Uses Waze crowd-sourced traffic data");
    QString tomtom_str = tr("⦿ TomTom: Uses TomTom professional traffic data");
    QString inrix_str = tr("⦿ INRIX: Uses INRIX real-time traffic intelligence");
    QString manual_str = tr("⦿ Manual API: Custom API endpoint configuration");

    switch (type) {
      case RTISourceType::WAZE:
        waze_str = "<font color='white'><b>" + waze_str + "</b></font>";
        break;
      case RTISourceType::TOMTOM:
        tomtom_str = "<font color='white'><b>" + tomtom_str + "</b></font>";
        break;
      case RTISourceType::INRIX:
        inrix_str = "<font color='white'><b>" + inrix_str + "</b></font>";
        break;
      case RTISourceType::MANUAL_API:
        manual_str = "<font color='white'><b>" + manual_str + "</b></font>";
        break;
      default:
        disabled_str = "<font color='white'><b>" + disabled_str + "</b></font>";
        break;
    }

    return QString("%1<br>%2<br>%3<br>%4<br>%5")
        .arg(disabled_str)
        .arg(waze_str)
        .arg(tomtom_str)
        .arg(inrix_str)
        .arg(manual_str);
  }

  static QString threatFilterDescription(RTIThreatFilter type = RTIThreatFilter::ALL) {
    QString all_str = tr("⦿ All Threats: Police, speed traps, accidents, hazards");
    QString police_str = tr("⦿ Police Only: Only police and speed enforcement");
    QString speed_str = tr("⦿ Speed Only: Only speed cameras and traps");
    QString hazards_str = tr("⦿ Hazards Only: Only accidents and road hazards");
    QString custom_str = tr("⦿ Custom Filter: User-defined threat selection");

    switch (type) {
      case RTIThreatFilter::POLICE_ONLY:
        police_str = "<font color='white'><b>" + police_str + "</b></font>";
        break;
      case RTIThreatFilter::SPEED_ONLY:
        speed_str = "<font color='white'><b>" + speed_str + "</b></font>";
        break;
      case RTIThreatFilter::HAZARDS_ONLY:
        hazards_str = "<font color='white'><b>" + hazards_str + "</b></font>";
        break;
      case RTIThreatFilter::CUSTOM:
        custom_str = "<font color='white'><b>" + custom_str + "</b></font>";
        break;
      default:
        all_str = "<font color='white'><b>" + all_str + "</b></font>";
        break;
    }

    return QString("%1<br>%2<br>%3<br>%4<br>%5")
        .arg(all_str)
        .arg(police_str)
        .arg(speed_str)
        .arg(hazards_str)
        .arg(custom_str);
  }

  static QString aggressivenessDescription(RTIAggressiveness type = RTIAggressiveness::BALANCED) {
    QString conservative_str = tr("⦿ Conservative: Gentle speed reduction, longer distances");
    QString balanced_str = tr("⦿ Balanced: Moderate speed reduction, balanced approach");
    QString aggressive_str = tr("⦿ Aggressive: Rapid speed reduction, shorter distances");

    switch (type) {
      case RTIAggressiveness::CONSERVATIVE:
        conservative_str = "<font color='white'><b>" + conservative_str + "</b></font>";
        break;
      case RTIAggressiveness::AGGRESSIVE:
        aggressive_str = "<font color='white'><b>" + aggressive_str + "</b></font>";
        break;
      default:
        balanced_str = "<font color='white'><b>" + balanced_str + "</b></font>";
        break;
    }

    return QString("%1<br>%2<br>%3")
        .arg(conservative_str)
        .arg(balanced_str)
        .arg(aggressive_str);
  }
};