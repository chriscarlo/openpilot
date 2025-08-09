/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.h"

RTISettingsPanel::RTISettingsPanel(QWidget *parent) : QStackedWidget(parent) {
  subPanelFrame = new QFrame();
  QVBoxLayout *subPanelLayout = new QVBoxLayout(subPanelFrame);
  subPanelLayout->setContentsMargins(0, 0, 0, 0);
  subPanelLayout->setSpacing(0);

  // Back button
  PanelBackButton *back = new PanelBackButton(tr("Back"));
  connect(back, &QPushButton::clicked, [=]() { emit backPress(); });
  subPanelLayout->addWidget(back, 0, Qt::AlignLeft);

  subPanelLayout->addSpacing(20);

  ListWidgetSP *list = new ListWidgetSP(this, true);

  // RTI Data Source Selection
  std::vector<QString> rti_source_texts{
    RTISourceTypeText[static_cast<int>(RTISourceType::DISABLED)],
    RTISourceTypeText[static_cast<int>(RTISourceType::WAZE)],
    RTISourceTypeText[static_cast<int>(RTISourceType::TOMTOM)],
    RTISourceTypeText[static_cast<int>(RTISourceType::INRIX)],
    RTISourceTypeText[static_cast<int>(RTISourceType::MANUAL_API)]
  };
  rti_source_setting = new ButtonParamControlSP(
    "RTIDataSource",
    tr("Data Source"),
    "",
    "",
    rti_source_texts,
    400);
  rti_source_setting->showDescription();
  list->addItem(rti_source_setting);

  // RTI Threat Filter
  std::vector<QString> rti_threat_filter_texts{
    RTIThreatFilterText[static_cast<int>(RTIThreatFilter::ALL)],
    RTIThreatFilterText[static_cast<int>(RTIThreatFilter::POLICE_ONLY)],
    RTIThreatFilterText[static_cast<int>(RTIThreatFilter::SPEED_ONLY)],
    RTIThreatFilterText[static_cast<int>(RTIThreatFilter::HAZARDS_ONLY)],
    RTIThreatFilterText[static_cast<int>(RTIThreatFilter::CUSTOM)]
  };
  rti_threat_filter_setting = new ButtonParamControlSP(
    "RTIThreatFilter",
    tr("Threat Filter"),
    "",
    "",
    rti_threat_filter_texts,
    400);
  rti_threat_filter_setting->showDescription();
  list->addItem(rti_threat_filter_setting);

  // RTI Aggressiveness
  std::vector<QString> rti_aggressiveness_texts{
    RTIAggressivenessText[static_cast<int>(RTIAggressiveness::CONSERVATIVE)],
    RTIAggressivenessText[static_cast<int>(RTIAggressiveness::BALANCED)],
    RTIAggressivenessText[static_cast<int>(RTIAggressiveness::AGGRESSIVE)]
  };
  rti_aggressiveness_setting = new ButtonParamControlSP(
    "RTIAggressiveness",
    tr("Response Aggressiveness"),
    "",
    "",
    rti_aggressiveness_texts,
    400);
  rti_aggressiveness_setting->showDescription();
  list->addItem(rti_aggressiveness_setting);

  // Distance controls frame
  QFrame *distanceFrame = new QFrame(this);
  QVBoxLayout *distanceLayout = new QVBoxLayout(distanceFrame);

  // Minimum activation distance
  rti_min_distance = new OptionControlSP(
    "RTIMinDistance",
    tr("Minimum Distance"),
    tr("Minimum distance to threat for RTI activation"),
    "",
    {50, 2000}  // 50m to 2000m
  );
  distanceLayout->addWidget(rti_min_distance);

  // Maximum activation distance
  rti_max_distance = new OptionControlSP(
    "RTIMaxDistance",
    tr("Maximum Distance"),
    tr("Maximum distance to threat for RTI activation"),
    "",
    {500, 5000}  // 500m to 5000m
  );
  distanceLayout->addWidget(rti_max_distance);

  list->addItem(distanceFrame);

  // Speed reduction control
  rti_speed_reduction = new OptionControlSP(
    "RTISpeedReduction",
    tr("Max Speed Reduction"),
    tr("Maximum speed reduction when threat detected (km/h)"),
    "",
    {5, 50}  // 5 to 50 km/h reduction
  );
  list->addItem(rti_speed_reduction);

  // HUD Display toggle
  rti_hud_enabled = new ParamControlSP("RTIHUDEnabled",
    tr("HUD Display"),
    tr("Show RTI threat information on the heads-up display"),
    "../assets/offroad/icon_shell.png");
  list->addItem(rti_hud_enabled);

  // Audio alerts toggle
  rti_audio_alerts = new ParamControlSP("RTIAudioAlerts",
    tr("Audio Alerts"),
    tr("Play audio alerts when threats are detected"),
    "../assets/offroad/icon_shell.png");
  list->addItem(rti_audio_alerts);

  // Advanced settings button
  rti_advanced_button = new PushButtonSP(tr("Advanced Configuration"));
  rti_advanced_button->setStyleSheet(R"(
    PushButtonSP {
      font-size: 30px;
      font-weight: 500;
      padding: 15px 30px;
      background-color: #4a90e2;
      color: white;
      border-radius: 12px;
      margin: 10px;
    }
    PushButtonSP:pressed {
      background-color: #357abd;
    }
  )");
  list->addItem(rti_advanced_button);

  // Connect signals for dynamic updates
  connect(rti_source_setting, &ButtonParamControlSP::buttonClicked, this, &RTISettingsPanel::refresh);
  connect(rti_threat_filter_setting, &ButtonParamControlSP::buttonClicked, this, &RTISettingsPanel::refresh);
  connect(rti_aggressiveness_setting, &ButtonParamControlSP::buttonClicked, this, &RTISettingsPanel::refresh);
  connect(rti_min_distance, &OptionControlSP::updateLabels, this, &RTISettingsPanel::refresh);
  connect(rti_max_distance, &OptionControlSP::updateLabels, this, &RTISettingsPanel::refresh);
  connect(rti_speed_reduction, &OptionControlSP::updateLabels, this, &RTISettingsPanel::refresh);
  
  // Connect advanced settings button
  connect(rti_advanced_button, &QPushButton::clicked, [=]() {
    emit advancedSettingsRequested();
  });

  refresh();
  subPanelLayout->addWidget(list);
  addWidget(subPanelFrame);
  setCurrentWidget(subPanelFrame);
}

// Safe string to int conversion with validation
static int safeStringToInt(const std::string& str, int defaultValue = 0) {
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

void RTISettingsPanel::refresh() {
  // Safe enum conversion with bounds checking and string validation
  int source_val = safeStringToInt(params.get("RTIDataSource"), 0);
  RTISourceType source_type = (source_val >= 0 && source_val <= 4) ? 
    static_cast<RTISourceType>(source_val) : RTISourceType::DISABLED;
  
  int filter_val = safeStringToInt(params.get("RTIThreatFilter"), 0);
  RTIThreatFilter threat_filter = (filter_val >= 0 && filter_val <= 4) ? 
    static_cast<RTIThreatFilter>(filter_val) : RTIThreatFilter::ALL;
  
  int aggr_val = safeStringToInt(params.get("RTIAggressiveness"), 1);
  RTIAggressiveness aggressiveness = (aggr_val >= 0 && aggr_val <= 2) ? 
    static_cast<RTIAggressiveness>(aggr_val) : RTIAggressiveness::BALANCED;

  // Update descriptions with current selections
  rti_source_setting->setDescription(sourceDescription(source_type));
  rti_threat_filter_setting->setDescription(threatFilterDescription(threat_filter));
  rti_aggressiveness_setting->setDescription(aggressivenessDescription(aggressiveness));

  // Update distance labels with units and safe validation
  int min_dist_val = safeStringToInt(params.get("RTIMinDistance"), 100);
  min_dist_val = std::max(50, std::min(2000, min_dist_val)); // Clamp to valid range
  QString minDistanceLabel = QString::number(min_dist_val) + "m";
  
  int max_dist_val = safeStringToInt(params.get("RTIMaxDistance"), 2000);
  max_dist_val = std::max(500, std::min(5000, max_dist_val)); // Clamp to valid range
  QString maxDistanceLabel = QString::number(max_dist_val) + "m";
  
  int speed_red_val = safeStringToInt(params.get("RTISpeedReduction"), 15);
  speed_red_val = std::max(5, std::min(50, speed_red_val)); // Clamp to valid range
  QString speedReductionLabel = QString::number(speed_red_val) + " km/h";

  rti_min_distance->setLabel(minDistanceLabel);
  rti_max_distance->setLabel(maxDistanceLabel);
  rti_speed_reduction->setLabel(speedReductionLabel);

  // Show/hide distance controls based on source selection
  bool sourceEnabled = (source_type != RTISourceType::DISABLED);
  rti_min_distance->setVisible(sourceEnabled);
  rti_max_distance->setVisible(sourceEnabled);
  rti_speed_reduction->setVisible(sourceEnabled);
  rti_hud_enabled->setVisible(sourceEnabled);
  rti_audio_alerts->setVisible(sourceEnabled);

  // Show descriptions
  if (sourceEnabled) {
    rti_min_distance->showDescription();
    rti_max_distance->showDescription();
    rti_speed_reduction->showDescription();
  }
}

void RTISettingsPanel::showEvent(QShowEvent *event) {
  refresh();
  rti_source_setting->showDescription();
  rti_threat_filter_setting->showDescription();
  rti_aggressiveness_setting->showDescription();
}