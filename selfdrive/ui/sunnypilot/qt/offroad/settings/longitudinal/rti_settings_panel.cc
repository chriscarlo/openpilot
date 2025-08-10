/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.h"
#include "selfdrive/ui/sunnypilot/qt/util/numeric_utils.h"

// Helper methods to eliminate code duplication
static QPair<QFrame*, QVBoxLayout*> createSettingsFrame() {
  QFrame *frame = new QFrame();
  frame->setStyleSheet("QFrame { background-color: #292929; border-radius: 20px; padding: 25px; }");
  QVBoxLayout *layout = new QVBoxLayout(frame);
  return qMakePair(frame, layout);
}

static QLabel* createSectionLabel(const QString &text, QVBoxLayout *layout) {
  QLabel *label = new QLabel(text);
  label->setStyleSheet("font-size: 40px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  layout->addWidget(label);
  return label;
}

static QHBoxLayout* createToggleRow(const QString &text, ToggleSP *toggle, const QString &param, Params &params, QVBoxLayout *parentLayout) {
  QHBoxLayout *layout = new QHBoxLayout();
  QLabel *label = new QLabel(text);
  label->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  layout->addWidget(label);
  layout->addStretch();
  
  toggle->setFixedSize(150, 80);
  toggle->setChecked(params.getBool(param.toStdString()));
  QObject::connect(toggle, &ToggleSP::stateChanged, [&params, param](bool checked) {
    params.putBool(param.toStdString(), checked);
  });
  layout->addWidget(toggle);
  parentLayout->addLayout(layout);
  return layout;
}


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

  // Create a scroll area for vertical scrolling only
  QScrollArea *scrollArea = new QScrollArea(this);
  scrollArea->setWidgetResizable(true);
  scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  scrollArea->setStyleSheet("QScrollArea { background-color: transparent; border: none; }");
  
  QWidget *scrollWidget = new QWidget();
  QVBoxLayout *scrollLayout = new QVBoxLayout(scrollWidget);
  scrollLayout->setContentsMargins(50, 20, 50, 20);
  
  // Enforce maximum width constraint to prevent horizontal scrolling
  scrollWidget->setMaximumWidth(1300); // Leave 100px margin from 1400px limit
  scrollLayout->setSpacing(30);

  // Title
  QLabel *title = new QLabel(tr("Real-time Traffic Intelligence"));
  title->setStyleSheet("font-size: 48px; font-weight: 600; color: #E4E4E4; padding-bottom: 10px;");
  title->setAlignment(Qt::AlignCenter);
  scrollLayout->addWidget(title);
  
  // Description
  QLabel *description = new QLabel(tr("Monitor traffic ahead and adjust speed automatically for safer driving"));
  description->setStyleSheet("font-size: 32px; color: #999999; padding-bottom: 30px;");
  description->setWordWrap(true);
  description->setAlignment(Qt::AlignCenter);
  scrollLayout->addWidget(description);
  
  // Data Source Dropdown
  QPair<QFrame*, QVBoxLayout*> sourcePair = createSettingsFrame();
  QFrame *sourceFrame = sourcePair.first;
  QVBoxLayout *sourceLayout = sourcePair.second;
  createSectionLabel(tr("Data Source"), sourceLayout);
  
  rti_source_combo = new QComboBox();
  rti_source_combo->setStyleSheet(R"(
    QComboBox {
      font-size: 36px;
      padding: 20px;
      background-color: #393939;
      color: white;
      border: 2px solid #555;
      border-radius: 15px;
      min-height: 60px;
    }
    QComboBox QAbstractItemView {
      font-size: 36px;
      background-color: #393939;
      selection-background-color: #4a90e2;
      border: 2px solid #555;
      border-radius: 5px;
      padding: 10px;
      outline: none;
    }
    QComboBox QAbstractItemView::item {
      padding: 8px;
      border: none;
    }
  )");
  
  // Fix popup positioning issue with QStackedWidget + QScrollArea
  // Ensure popup uses correct parent for coordinate calculations
  auto *rti_source_view = rti_source_combo->view();
  rti_source_view->setParent(nullptr);
  rti_source_view->setWindowFlags(Qt::Popup | Qt::FramelessWindowHint | Qt::NoDropShadowWindowHint);
  
  rti_source_combo->addItem(tr("Disabled"));
  rti_source_combo->addItem(tr("Waze"));
  // Commented out non-functional data sources for now
  // rti_source_combo->addItem(tr("TomTom"));
  // rti_source_combo->addItem(tr("INRIX"));
  // rti_source_combo->addItem(tr("Manual API"));
  
  int source_val = SunnypilotUtils::safeStringToInt(params.get("RTIDataSource"), 0);
  rti_source_combo->setCurrentIndex(source_val);
  
  connect(rti_source_combo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
    params.put("RTIDataSource", std::to_string(index));
    if (index == 1) { // Waze selected
      loadWazeApiKey();
    }
    refresh();
  });
  
  sourceLayout->addWidget(rti_source_combo);
  scrollLayout->addWidget(sourceFrame);

  // Threat Filter Dropdown
  QPair<QFrame*, QVBoxLayout*> filterPair = createSettingsFrame();
  QFrame *filterFrame = filterPair.first;
  QVBoxLayout *filterLayout = filterPair.second;
  createSectionLabel(tr("Threat Filter"), filterLayout);
  
  rti_filter_combo = new QComboBox();
  rti_filter_combo->setStyleSheet(rti_source_combo->styleSheet());
  
  // Fix popup positioning issue with QStackedWidget + QScrollArea
  auto *rti_filter_view = rti_filter_combo->view();
  rti_filter_view->setParent(nullptr);
  rti_filter_view->setWindowFlags(Qt::Popup | Qt::FramelessWindowHint | Qt::NoDropShadowWindowHint);
  
  rti_filter_combo->addItem(tr("All Threats"));
  rti_filter_combo->addItem(tr("Police Only"));
  rti_filter_combo->addItem(tr("Speed Cameras Only"));
  rti_filter_combo->addItem(tr("Hazards Only"));
  rti_filter_combo->addItem(tr("Custom"));
  
  int filter_val = SunnypilotUtils::safeStringToInt(params.get("RTIThreatFilter"), 0);
  rti_filter_combo->setCurrentIndex(filter_val);
  
  connect(rti_filter_combo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
    params.put("RTIThreatFilter", std::to_string(index));
    refresh();
  });
  
  filterLayout->addWidget(rti_filter_combo);
  scrollLayout->addWidget(filterFrame);

  // Response Style Dropdown
  QPair<QFrame*, QVBoxLayout*> aggrPair = createSettingsFrame();
  QFrame *aggrFrame = aggrPair.first;
  QVBoxLayout *aggrLayout = aggrPair.second;
  createSectionLabel(tr("Response Style"), aggrLayout);
  
  rti_aggr_combo = new QComboBox();
  rti_aggr_combo->setStyleSheet(rti_source_combo->styleSheet());
  
  // Fix popup positioning issue with QStackedWidget + QScrollArea
  auto *rti_aggr_view = rti_aggr_combo->view();
  rti_aggr_view->setParent(nullptr);
  rti_aggr_view->setWindowFlags(Qt::Popup | Qt::FramelessWindowHint | Qt::NoDropShadowWindowHint);
  
  rti_aggr_combo->addItem(tr("Conservative - Early, gentle braking"));
  rti_aggr_combo->addItem(tr("Balanced - Optimal comfort"));
  rti_aggr_combo->addItem(tr("Aggressive - Later, quicker response"));
  
  int aggr_val = SunnypilotUtils::safeStringToInt(params.get("RTIAggressiveness"), 1);
  rti_aggr_combo->setCurrentIndex(aggr_val);
  
  connect(rti_aggr_combo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
    params.put("RTIAggressiveness", std::to_string(index));
    refresh();
  });
  
  aggrLayout->addWidget(rti_aggr_combo);
  scrollLayout->addWidget(aggrFrame);

  // Distance Settings
  QPair<QFrame*, QVBoxLayout*> distancePair = createSettingsFrame();
  QFrame *distanceFrame = distancePair.first;
  QVBoxLayout *distanceLayout = distancePair.second;
  createSectionLabel(tr("Detection Range"), distanceLayout);
  
  // Min distance slider
  QLabel *minDistLabel = new QLabel(tr("Minimum: 0.5 km"));
  minDistLabel->setStyleSheet("font-size: 32px; color: #999999;");
  distanceLayout->addWidget(minDistLabel);
  
  rti_min_slider = new QSlider(Qt::Horizontal);
  rti_min_slider->setStyleSheet(R"(
    QSlider::groove:horizontal {
      height: 10px;
      background: #393939;
      border-radius: 5px;
    }
    QSlider::handle:horizontal {
      width: 40px;
      height: 40px;
      background: #4a90e2;
      border-radius: 20px;
      margin: -15px 0;
    }
    QSlider::sub-page:horizontal {
      background: #4a90e2;
      border-radius: 5px;
    }
  )");
  connect(rti_min_slider, &QSlider::valueChanged, [this, minDistLabel](int value) {
    minDistLabel->setText(formatDistanceLabel(value, true));
    params.put("RTIMinDistance", std::to_string(value));
  });
  distanceLayout->addWidget(rti_min_slider);
  
  distanceLayout->addSpacing(20);
  
  // Max distance slider
  QLabel *maxDistLabel = new QLabel(tr("Maximum: 2.0 km"));
  maxDistLabel->setStyleSheet("font-size: 32px; color: #999999;");
  distanceLayout->addWidget(maxDistLabel);
  
  rti_max_slider = new QSlider(Qt::Horizontal);
  rti_max_slider->setStyleSheet(rti_min_slider->styleSheet());
  connect(rti_max_slider, &QSlider::valueChanged, [this, maxDistLabel](int value) {
    maxDistLabel->setText(formatDistanceLabel(value, false));
    params.put("RTIMaxDistance", std::to_string(value));
  });
  distanceLayout->addWidget(rti_max_slider);
  
  scrollLayout->addWidget(distanceFrame);
  
  // Speed Reduction
  QPair<QFrame*, QVBoxLayout*> speedPair = createSettingsFrame();
  QFrame *speedFrame = speedPair.first;
  QVBoxLayout *speedLayout = speedPair.second;
  createSectionLabel(tr("Speed Reduction"), speedLayout);
  
  QLabel *speedLabel = new QLabel(tr("Max reduction: 15 km/h"));
  speedLabel->setStyleSheet("font-size: 32px; color: #999999;");
  speedLayout->addWidget(speedLabel);
  
  rti_speed_slider = new QSlider(Qt::Horizontal);
  rti_speed_slider->setRange(5, 50);
  rti_speed_slider->setSingleStep(5);
  rti_speed_slider->setValue(SunnypilotUtils::safeStringToInt(params.get("RTISpeedReduction"), 15));
  rti_speed_slider->setStyleSheet(rti_min_slider->styleSheet());
  connect(rti_speed_slider, &QSlider::valueChanged, [this, speedLabel](int value) {
    speedLabel->setText(QString(tr("Max reduction: %1 km/h")).arg(value));
    params.put("RTISpeedReduction", std::to_string(value));
  });
  speedLayout->addWidget(rti_speed_slider);
  
  scrollLayout->addWidget(speedFrame);

  // Visual & Audio Settings
  QPair<QFrame*, QVBoxLayout*> alertsPair = createSettingsFrame();
  QFrame *alertsFrame = alertsPair.first;
  QVBoxLayout *alertsLayout = alertsPair.second;
  createSectionLabel(tr("Alerts & Display"), alertsLayout);
  
  // HUD toggle
  rti_hud_toggle = new ToggleSP();
  createToggleRow(tr("HUD Display"), rti_hud_toggle, "RTIHUDEnabled", params, alertsLayout);
  
  alertsLayout->addSpacing(15);
  
  // Audio toggle
  rti_audio_toggle = new ToggleSP();
  createToggleRow(tr("Audio Alerts"), rti_audio_toggle, "RTIAudioAlerts", params, alertsLayout);
  
  scrollLayout->addWidget(alertsFrame);

  // Advanced settings button - commented out until other data sources are implemented
  // QPushButton *advanced_btn = new QPushButton(tr("Advanced Settings"));
  // advanced_btn->setFixedHeight(100);
  // advanced_btn->setStyleSheet(R"(
  //   QPushButton {
  //     font-size: 38px;
  //     font-weight: 500;
  //     background-color: #4a90e2;
  //     color: white;
  //     border-radius: 20px;
  //     margin: 20px 0;
  //   }
  //   QPushButton:pressed {
  //     background-color: #357abd;
  //   }
  // )");
  // connect(advanced_btn, &QPushButton::clicked, [=]() {
  //   emit advancedSettingsRequested();
  // });
  // scrollLayout->addWidget(advanced_btn);
  
  scrollLayout->addStretch();
  
  scrollArea->setWidget(scrollWidget);
  subPanelLayout->addWidget(scrollArea);
  
  // Configure sliders with proper unit-aware settings
  configureDistanceSliders();
  
  refresh();
  addWidget(subPanelFrame);
  setCurrentWidget(subPanelFrame);
}

void RTISettingsPanel::loadWazeApiKey() {
  // Try to load API key from various locations in priority order
  const std::vector<std::string> api_key_paths = {
    "/data/persist/rapidapi_key",
    "/persist/rapidapi_key",
    "/data/openpilot/persist/rapidapi_key",
    "/data/openpilot/rapidapi_key"
  };
  
  std::string api_key;
  
  // First check environment variables
  const char* env_key = std::getenv("RAPIDAPI_KEY");
  if (env_key && strlen(env_key) > 0) {
    api_key = env_key;
  } else {
    // Try each file location
    for (const auto& path : api_key_paths) {
      std::ifstream file(path);
      if (file.is_open()) {
        std::getline(file, api_key);
        file.close();
        // Trim whitespace
        api_key.erase(0, api_key.find_first_not_of(" \n\r\t"));
        api_key.erase(api_key.find_last_not_of(" \n\r\t") + 1);
        if (!api_key.empty()) {
          break;
        }
      }
    }
  }
  
  // If we found a key, configure Waze parameters
  if (!api_key.empty()) {
    params.put("RTIManualApiKey", api_key);
    params.put("RTIManualApiEndpoint", "https://waze.p.rapidapi.com/alerts-and-jams");
    params.put("RTIManualApiFormat", "waze_rapid");
    params.put("RTIApiConfigured", "1");
  } else {
    // No API key found - mark as not configured
    params.put("RTIApiConfigured", "0");
  }
}

void RTISettingsPanel::refresh() {
  // Check if metric setting changed and reconfigure sliders if needed
  static bool last_metric_state = isMetricSystem();
  bool current_metric_state = isMetricSystem();
  
  if (current_metric_state != last_metric_state) {
    // Metric setting changed - reconfigure sliders
    configureDistanceSliders();
    last_metric_state = current_metric_state;
  }
  
  // Check if RTI is enabled
  int source_val = SunnypilotUtils::safeStringToInt(params.get("RTIDataSource"), 0);
  bool sourceEnabled = (source_val != 0);
  
  // Enable/disable controls based on source selection
  if (rti_filter_combo) rti_filter_combo->setEnabled(sourceEnabled);
  if (rti_aggr_combo) rti_aggr_combo->setEnabled(sourceEnabled);
  if (rti_min_slider) rti_min_slider->setEnabled(sourceEnabled);
  if (rti_max_slider) rti_max_slider->setEnabled(sourceEnabled);
  if (rti_speed_slider) rti_speed_slider->setEnabled(sourceEnabled);
  if (rti_hud_toggle) rti_hud_toggle->setEnabled(sourceEnabled);
  if (rti_audio_toggle) rti_audio_toggle->setEnabled(sourceEnabled);
}

void RTISettingsPanel::showEvent(QShowEvent *event) {
  // Validate parameters and update sliders when panel is shown
  validateAndMigrateParameters();
  configureDistanceSliders();
  refresh();
}

// Helper method to check if system is in metric mode
bool RTISettingsPanel::isMetricSystem() {
  return params.getBool("IsMetric");
}

// Configure distance sliders based on metric/imperial setting
void RTISettingsPanel::configureDistanceSliders() {
  const bool is_metric = isMetricSystem();
  
  int min_range_m, max_range_m, step_m;
  int default_min, default_max;
  
  if (is_metric) {
    // Metric: 0.5km - 5km in 0.5km increments
    min_range_m = static_cast<int>(METRIC_INCREMENT_KM * KM_TO_METERS);  // 500m
    max_range_m = static_cast<int>(5.0 * KM_TO_METERS);                // 5000m
    step_m = min_range_m;                                               // 500m
    default_min = 500;
    default_max = 2000;
  } else {
    // Imperial: 0.25mi - 2mi in 0.25mi increments
    min_range_m = static_cast<int>(IMPERIAL_INCREMENT_MI * MILES_TO_METERS);  // ~402m
    max_range_m = static_cast<int>(2.0 * MILES_TO_METERS);                   // ~3219m
    step_m = min_range_m;                                                     // ~402m
    default_min = min_range_m;
    default_max = static_cast<int>(1.0 * MILES_TO_METERS);  // 1 mile
  }
  
  rti_min_slider->setRange(min_range_m, max_range_m);
  rti_min_slider->setSingleStep(step_m);
  rti_max_slider->setRange(min_range_m, max_range_m);
  rti_max_slider->setSingleStep(step_m);
  
  // Update current values from params, snapping to valid increments
  int current_min = SunnypilotUtils::safeStringToInt(params.get("RTIMinDistance"), default_min);
  int current_max = SunnypilotUtils::safeStringToInt(params.get("RTIMaxDistance"), default_max);
  
  // Snap to nearest valid increment and enforce range limits
  current_min = snapToValidIncrement(current_min);
  current_max = snapToValidIncrement(current_max);
  
  // Ensure min < max
  if (current_min >= current_max) {
    current_min = default_min;
    current_max = default_max;
  }
  
  rti_min_slider->setValue(current_min);
  rti_max_slider->setValue(current_max);
}

// Validate parameters and migrate if needed
void RTISettingsPanel::validateAndMigrateParameters() {
  // Get current values
  int current_min = SunnypilotUtils::safeStringToInt(params.get("RTIMinDistance"), 500);
  int current_max = SunnypilotUtils::safeStringToInt(params.get("RTIMaxDistance"), 2000);
  
  // Ensure they meet new constraints
  int valid_min = snapToValidIncrement(current_min);
  int valid_max = snapToValidIncrement(current_max);
  
  const bool is_metric = isMetricSystem();
  const int min_limit = is_metric ? 500 : static_cast<int>(0.25 * MILES_TO_METERS);
  const int max_limit = is_metric ? 5000 : static_cast<int>(2.0 * MILES_TO_METERS);
  
  // Enforce range limits
  valid_min = std::max(min_limit, std::min(max_limit, valid_min));
  valid_max = std::max(min_limit, std::min(max_limit, valid_max));
  
  // Ensure min < max
  if (valid_min >= valid_max) {
    valid_min = is_metric ? 500 : static_cast<int>(0.25 * MILES_TO_METERS);
    valid_max = is_metric ? 1000 : static_cast<int>(0.5 * MILES_TO_METERS);
  }
  
  // Update params if values changed
  if (valid_min != current_min) {
    params.put("RTIMinDistance", std::to_string(valid_min));
  }
  if (valid_max != current_max) {
    params.put("RTIMaxDistance", std::to_string(valid_max));
  }
}

// Format distance labels with appropriate units
QString RTISettingsPanel::formatDistanceLabel(int meters_value, bool is_minimum) {
  const bool is_metric = isMetricSystem();
  const QString prefix = is_minimum ? tr("Minimum: ") : tr("Maximum: ");
  
  if (is_metric) {
    const double km = meters_value * METERS_TO_KM;
    return QString("%1%2 km").arg(prefix).arg(km, 0, 'f', 1);
  } else {
    const double miles = meters_value * METERS_TO_MILES;
    return QString("%1%2 mi").arg(prefix).arg(miles, 0, 'f', 2);
  }
}

// Snap meter values to valid increments based on current unit system
int RTISettingsPanel::snapToValidIncrement(int meters) {
  const bool is_metric = isMetricSystem();
  
  if (is_metric) {
    // Round to nearest 0.5km (500m)
    return ((meters + 250) / 500) * 500;
  } else {
    // Round to nearest 0.25mi (≈402m)
    const int increment_m = static_cast<int>(IMPERIAL_INCREMENT_MI * MILES_TO_METERS);
    return ((meters + increment_m/2) / increment_m) * increment_m;
  }
}