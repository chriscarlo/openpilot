/**
 * RTI Settings Panel Implementation - Clean Redesign
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.h"
#include <QScrollArea>
#include <cmath>

// ============================================================================
// RTIRangeControl Implementation
// ============================================================================

RTIRangeControl::RTIRangeControl(const QString &title, const QString &description,
                                 const QString &paramKey, float minVal, float maxVal, 
                                 float step, float defaultVal, const QString &units,
                                 QWidget *parent) 
  : QFrame(parent), paramKey(paramKey), units(units), 
    minValue(minVal), maxValue(maxVal), stepSize(step), defaultValue(defaultVal) {
  
  // Main layout
  QVBoxLayout *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(0, 0, 0, 0);
  
  // Title
  QLabel *titleLabel = new QLabel(title);
  titleLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4;");
  mainLayout->addWidget(titleLabel);
  
  // Description
  if (!description.isEmpty()) {
    QLabel *descLabel = new QLabel(description);
    descLabel->setWordWrap(true);
    descLabel->setStyleSheet("font-size: 32px; color: #999999; margin-top: 5px; margin-bottom: 15px;");
    mainLayout->addWidget(descLabel);
  }
  
  // Control layout
  QHBoxLayout *controlLayout = new QHBoxLayout();
  controlLayout->setSpacing(20);
  
  // Minus button
  minusBtn = new QPushButton("-");
  minusBtn->setFixedSize(100, 100);
  minusBtn->setStyleSheet(R"(
    QPushButton {
      font-size: 60px;
      font-weight: 500;
      border-radius: 50px;
      background-color: #393939;
      color: #E4E4E4;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
    QPushButton:disabled {
      background-color: #2a2a2a;
      color: #666666;
    }
  )");
  controlLayout->addWidget(minusBtn);
  
  // Value display
  QVBoxLayout *valueLayout = new QVBoxLayout();
  valueLayout->setAlignment(Qt::AlignCenter);
  
  valueLabel = new QLabel("0.00");
  valueLabel->setAlignment(Qt::AlignCenter);
  valueLabel->setStyleSheet("font-size: 70px; font-weight: 500; color: #FFFFFF;");
  valueLabel->setFixedWidth(300);
  valueLayout->addWidget(valueLabel);
  
  statusLabel = new QLabel(tr("(Default)"));
  statusLabel->setAlignment(Qt::AlignCenter);
  statusLabel->setStyleSheet("font-size: 32px; color: #999999;");
  valueLayout->addWidget(statusLabel);
  
  controlLayout->addLayout(valueLayout);
  
  // Plus button
  plusBtn = new QPushButton("+");
  plusBtn->setFixedSize(100, 100);
  plusBtn->setStyleSheet(minusBtn->styleSheet());
  controlLayout->addWidget(plusBtn);
  
  controlLayout->addStretch();
  
  // Reset button
  resetBtn = new QPushButton(tr("Reset"));
  resetBtn->setFixedSize(150, 80);
  resetBtn->setStyleSheet(R"(
    QPushButton {
      font-size: 35px;
      font-weight: 500;
      border-radius: 20px;
      background-color: #393939;
      color: #E4E4E4;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
    QPushButton:disabled {
      background-color: #2a2a2a;
      color: #666666;
    }
  )");
  controlLayout->addWidget(resetBtn);
  
  mainLayout->addLayout(controlLayout);
  
  // Load current value from params (stored in meters, display in miles)
  QString storedValue = QString::fromStdString(params.get(paramKey.toStdString()));
  if (storedValue.isEmpty()) {
    currentValue = defaultValue;
  } else {
    // Convert from meters to miles for display
    float meters = storedValue.toFloat();
    currentValue = meters * METERS_TO_MILES;
  }
  
  // Connect signals
  connect(minusBtn, &QPushButton::clicked, this, &RTIRangeControl::decrement);
  connect(plusBtn, &QPushButton::clicked, this, &RTIRangeControl::increment);
  connect(resetBtn, &QPushButton::clicked, this, &RTIRangeControl::reset);
  
  updateLabels();
}

void RTIRangeControl::updateLabels() {
  // Display value with units
  valueLabel->setText(QString::number(currentValue, 'f', 2) + " " + units);
  
  bool isDefault = std::abs(currentValue - defaultValue) < 0.01f;
  if (isDefault) {
    statusLabel->setText(tr("(Default)"));
    statusLabel->setStyleSheet("font-size: 32px; color: #999999;");
  } else {
    statusLabel->setText(tr("(Modified)"));
    statusLabel->setStyleSheet("font-size: 32px; color: #FFC107;");
  }
  
  // Update button states
  minusBtn->setEnabled(currentValue > minValue);
  plusBtn->setEnabled(currentValue < maxValue);
  resetBtn->setEnabled(!isDefault);
}

void RTIRangeControl::increment() {
  currentValue = std::min(currentValue + stepSize, maxValue);
  // Convert miles to meters for storage
  float meters = currentValue * MILES_TO_METERS;
  params.put(paramKey.toStdString(), QString::number(meters, 'f', 0).toStdString());
  updateLabels();
}

void RTIRangeControl::decrement() {
  currentValue = std::max(currentValue - stepSize, minValue);
  // Convert miles to meters for storage
  float meters = currentValue * MILES_TO_METERS;
  params.put(paramKey.toStdString(), QString::number(meters, 'f', 0).toStdString());
  updateLabels();
}

void RTIRangeControl::reset() {
  currentValue = defaultValue;
  float meters = currentValue * MILES_TO_METERS;
  params.put(paramKey.toStdString(), QString::number(meters, 'f', 0).toStdString());
  updateLabels();
}

// ============================================================================
// RTISpeedReductionControl Implementation
// ============================================================================

RTISpeedReductionControl::RTISpeedReductionControl(QWidget *parent) : QFrame(parent) {
  QVBoxLayout *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(0, 0, 0, 0);
  
  // Title
  QLabel *titleLabel = new QLabel(tr("Speed Reduction"));
  titleLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4;");
  mainLayout->addWidget(titleLabel);
  
  // Description
  QLabel *descLabel = new QLabel(tr("How much to reduce speed when approaching threats"));
  descLabel->setWordWrap(true);
  descLabel->setStyleSheet("font-size: 32px; color: #999999; margin-top: 5px; margin-bottom: 15px;");
  mainLayout->addWidget(descLabel);
  
  // Mode selector
  QHBoxLayout *modeLayout = new QHBoxLayout();
  
  modeSelector = new QComboBox();
  modeSelector->setStyleSheet(R"(
    QComboBox {
      font-size: 36px;
      padding: 15px;
      background-color: #393939;
      color: white;
      border: 2px solid #555;
      border-radius: 15px;
      min-height: 60px;
    }
    QComboBox::drop-down {
      width: 50px;
      border: none;
    }
    QComboBox::down-arrow {
      image: none;
      border-left: 10px solid transparent;
      border-right: 10px solid transparent;
      border-top: 15px solid #E4E4E4;
      margin-right: 10px;
    }
    QComboBox QAbstractItemView {
      font-size: 36px;
      background-color: #393939;
      selection-background-color: #4a90e2;
      border: 2px solid #555;
      padding: 10px;
    }
  )");
  
  modeSelector->addItem(tr("Posted Speed Limit"));
  modeSelector->addItem(tr("Custom"));
  
  // Load current setting
  QString speedMode = QString::fromStdString(params.get("RTISpeedReductionMode"));
  if (speedMode == "custom") {
    modeSelector->setCurrentIndex(1);
  } else {
    modeSelector->setCurrentIndex(0);
  }
  
  connect(modeSelector, QOverload<int>::of(&QComboBox::currentIndexChanged), 
          this, &RTISpeedReductionControl::updateMode);
  
  modeLayout->addWidget(modeSelector);
  modeLayout->addStretch();
  mainLayout->addLayout(modeLayout);
  
  // Custom speed control (initially hidden)
  customFrame = new QFrame();
  customFrame->setStyleSheet("background-color: transparent;");
  QVBoxLayout *customLayout = new QVBoxLayout(customFrame);
  customLayout->setContentsMargins(0, 20, 0, 0);
  
  // Custom speed control layout
  QHBoxLayout *speedControlLayout = new QHBoxLayout();
  speedControlLayout->setSpacing(20);
  
  // Minus button
  speedMinusBtn = new QPushButton("-");
  speedMinusBtn->setFixedSize(100, 100);
  speedMinusBtn->setStyleSheet(R"(
    QPushButton {
      font-size: 60px;
      font-weight: 500;
      border-radius: 50px;
      background-color: #393939;
      color: #E4E4E4;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
    QPushButton:disabled {
      background-color: #2a2a2a;
      color: #666666;
    }
  )");
  speedControlLayout->addWidget(speedMinusBtn);
  
  // Value display
  QVBoxLayout *speedValueLayout = new QVBoxLayout();
  speedValueLayout->setAlignment(Qt::AlignCenter);
  
  speedValueLabel = new QLabel("15 mph");
  speedValueLabel->setAlignment(Qt::AlignCenter);
  speedValueLabel->setStyleSheet("font-size: 70px; font-weight: 500; color: #FFFFFF;");
  speedValueLabel->setFixedWidth(300);
  speedValueLayout->addWidget(speedValueLabel);
  
  speedStatusLabel = new QLabel(tr("(Default)"));
  speedStatusLabel->setAlignment(Qt::AlignCenter);
  speedStatusLabel->setStyleSheet("font-size: 32px; color: #999999;");
  speedValueLayout->addWidget(speedStatusLabel);
  
  speedControlLayout->addLayout(speedValueLayout);
  
  // Plus button
  speedPlusBtn = new QPushButton("+");
  speedPlusBtn->setFixedSize(100, 100);
  speedPlusBtn->setStyleSheet(speedMinusBtn->styleSheet());
  speedControlLayout->addWidget(speedPlusBtn);
  
  speedControlLayout->addStretch();
  
  // Reset button
  speedResetBtn = new QPushButton(tr("Reset"));
  speedResetBtn->setFixedSize(150, 80);
  speedResetBtn->setStyleSheet(R"(
    QPushButton {
      font-size: 35px;
      font-weight: 500;
      border-radius: 20px;
      background-color: #393939;
      color: #E4E4E4;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
    QPushButton:disabled {
      background-color: #2a2a2a;
      color: #666666;
    }
  )");
  speedControlLayout->addWidget(speedResetBtn);
  
  customLayout->addLayout(speedControlLayout);
  customFrame->setLayout(customLayout);
  
  // Load custom speed value (stored in km/h, display in mph)
  QString storedSpeed = QString::fromStdString(params.get("RTISpeedReduction"));
  if (storedSpeed.isEmpty()) {
    customSpeed = 10; // Default 10 mph
  } else {
    float kmh = storedSpeed.toFloat();
    customSpeed = static_cast<int>(kmh * KMH_TO_MPH);
  }
  
  // Connect custom speed controls
  connect(speedMinusBtn, &QPushButton::clicked, this, &RTISpeedReductionControl::decrementSpeed);
  connect(speedPlusBtn, &QPushButton::clicked, this, &RTISpeedReductionControl::incrementSpeed);
  connect(speedResetBtn, &QPushButton::clicked, this, &RTISpeedReductionControl::resetSpeed);
  
  updateCustomValue();
  
  mainLayout->addWidget(customFrame);
  
  // Show/hide custom frame based on current selection
  customFrame->setVisible(modeSelector->currentIndex() == 1);
}

void RTISpeedReductionControl::updateMode(int index) {
  if (index == 0) {
    params.put("RTISpeedReductionMode", "posted");
    customFrame->setVisible(false);
  } else {
    params.put("RTISpeedReductionMode", "custom");
    customFrame->setVisible(true);
  }
}

void RTISpeedReductionControl::updateCustomValue() {
  speedValueLabel->setText(QString::number(customSpeed) + " mph");
  
  bool isDefault = (customSpeed == 10);
  if (isDefault) {
    speedStatusLabel->setText(tr("(Default)"));
    speedStatusLabel->setStyleSheet("font-size: 32px; color: #999999;");
  } else {
    speedStatusLabel->setText(tr("(Modified)"));
    speedStatusLabel->setStyleSheet("font-size: 32px; color: #FFC107;");
  }
  
  speedMinusBtn->setEnabled(customSpeed > 0);
  speedPlusBtn->setEnabled(customSpeed < 30);
  speedResetBtn->setEnabled(!isDefault);
}

void RTISpeedReductionControl::incrementSpeed() {
  customSpeed = std::min(customSpeed + 1, 30);
  float kmh = customSpeed * MPH_TO_KMH;
  params.put("RTISpeedReduction", QString::number(kmh, 'f', 0).toStdString());
  updateCustomValue();
}

void RTISpeedReductionControl::decrementSpeed() {
  customSpeed = std::max(customSpeed - 1, 0);
  float kmh = customSpeed * MPH_TO_KMH;
  params.put("RTISpeedReduction", QString::number(kmh, 'f', 0).toStdString());
  updateCustomValue();
}

void RTISpeedReductionControl::resetSpeed() {
  customSpeed = 10;
  float kmh = customSpeed * MPH_TO_KMH;
  params.put("RTISpeedReduction", QString::number(kmh, 'f', 0).toStdString());
  updateCustomValue();
}

// ============================================================================
// RTISettingsPanel Implementation
// ============================================================================

RTISettingsPanel::RTISettingsPanel(QWidget *parent) : QFrame(parent) {
  setupMainLayout();
  loadWazeApiKey();
}

void RTISettingsPanel::setupMainLayout() {
  QVBoxLayout *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(0, 0, 0, 0);
  mainLayout->setSpacing(0);
  
  // Back button
  PanelBackButton *backBtn = new PanelBackButton(tr("Back"));
  connect(backBtn, &QPushButton::clicked, this, &RTISettingsPanel::backPress);
  mainLayout->addWidget(backBtn, 0, Qt::AlignLeft);
  
  mainLayout->addSpacing(20);
  
  // Create scroll area
  QScrollArea *scrollArea = new QScrollArea(this);
  scrollArea->setWidgetResizable(true);
  scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  scrollArea->setStyleSheet("QScrollArea { background-color: transparent; border: none; }");
  
  QWidget *scrollWidget = new QWidget();
  scrollWidget->setMaximumWidth(1300); // Prevent horizontal scrolling
  QVBoxLayout *scrollLayout = new QVBoxLayout(scrollWidget);
  scrollLayout->setContentsMargins(50, 20, 50, 20);
  scrollLayout->setSpacing(30);
  
  // Title
  QLabel *title = new QLabel(tr("Real-time Traffic Intelligence"));
  title->setStyleSheet("font-size: 50px; font-weight: 600; color: #E4E4E4; padding-bottom: 10px;");
  title->setAlignment(Qt::AlignCenter);
  scrollLayout->addWidget(title);
  
  // Description
  QLabel *description = new QLabel(tr("RTI uses Waze traffic data to automatically adjust your speed for safer driving"));
  description->setStyleSheet("font-size: 34px; color: #999999; padding-bottom: 30px;");
  description->setWordWrap(true);
  description->setAlignment(Qt::AlignCenter);
  scrollLayout->addWidget(description);
  
  // Threat Filter Section
  QFrame *filterFrame = createSectionFrame();
  QVBoxLayout *filterLayout = new QVBoxLayout(filterFrame);
  
  QLabel *filterLabel = new QLabel(tr("Threat Filter"));
  filterLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  filterLayout->addWidget(filterLabel);
  
  threatFilterCombo = new QComboBox();
  threatFilterCombo->setStyleSheet(R"(
    QComboBox {
      font-size: 36px;
      padding: 15px;
      background-color: #393939;
      color: white;
      border: 2px solid #555;
      border-radius: 15px;
      min-height: 60px;
    }
    QComboBox::drop-down {
      width: 50px;
      border: none;
    }
    QComboBox::down-arrow {
      image: none;
      border-left: 10px solid transparent;
      border-right: 10px solid transparent;
      border-top: 15px solid #E4E4E4;
      margin-right: 10px;
    }
    QComboBox QAbstractItemView {
      font-size: 36px;
      background-color: #393939;
      selection-background-color: #4a90e2;
      border: 2px solid #555;
      padding: 10px;
    }
  )");
  
  threatFilterCombo->addItem(tr("All Threats"));
  threatFilterCombo->addItem(tr("Police Only"));
  threatFilterCombo->addItem(tr("Speed Cameras Only"));
  threatFilterCombo->addItem(tr("Hazards Only"));
  threatFilterCombo->addItem(tr("Custom"));
  
  int filterVal = QString::fromStdString(params.get("RTIThreatFilter")).toInt();
  threatFilterCombo->setCurrentIndex(filterVal);
  
  connect(threatFilterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
    params.put("RTIThreatFilter", std::to_string(index));
  });
  
  filterLayout->addWidget(threatFilterCombo);
  scrollLayout->addWidget(filterFrame);
  
  // Detection Range Section
  QFrame *rangeFrame = createSectionFrame();
  QVBoxLayout *rangeLayout = new QVBoxLayout(rangeFrame);
  
  QLabel *rangeLabel = new QLabel(tr("Detection Range"));
  rangeLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  rangeLayout->addWidget(rangeLabel);
  
  // Min distance control
  minDistanceControl = new RTIRangeControl(
    tr("Minimum Distance"),
    tr("Start monitoring threats at this distance"),
    "RTIMinDistance",
    0.0f, 2.0f, 0.25f, 0.25f, "mi",
    this
  );
  rangeLayout->addWidget(minDistanceControl);
  
  rangeLayout->addSpacing(20);
  
  // Max distance control
  maxDistanceControl = new RTIRangeControl(
    tr("Maximum Distance"),
    tr("Maximum distance to look ahead for threats"),
    "RTIMaxDistance",
    0.25f, 2.0f, 0.25f, 1.5f, "mi",
    this
  );
  rangeLayout->addWidget(maxDistanceControl);
  
  scrollLayout->addWidget(rangeFrame);
  
  // Speed Reduction Section
  QFrame *speedFrame = createSectionFrame();
  QVBoxLayout *speedLayout = new QVBoxLayout(speedFrame);
  
  speedReductionControl = new RTISpeedReductionControl(this);
  speedLayout->addWidget(speedReductionControl);
  
  scrollLayout->addWidget(speedFrame);
  
  // Alerts & Display Section
  QFrame *alertsFrame = createSectionFrame();
  QVBoxLayout *alertsLayout = new QVBoxLayout(alertsFrame);
  
  QLabel *alertsLabel = new QLabel(tr("Alerts & Display"));
  alertsLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  alertsLayout->addWidget(alertsLabel);
  
  // HUD Display toggle
  QHBoxLayout *hudLayout = new QHBoxLayout();
  QLabel *hudLabel = new QLabel(tr("Show HUD Display"));
  hudLabel->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  hudLayout->addWidget(hudLabel);
  hudLayout->addStretch();
  
  hudToggle = new ToggleSP();
  hudToggle->setFixedSize(150, 80);
  hudToggle->setChecked(params.getBool("RTIHUDEnabled"));
  connect(hudToggle, &ToggleSP::stateChanged, [this](bool checked) {
    params.putBool("RTIHUDEnabled", checked);
  });
  hudLayout->addWidget(hudToggle);
  alertsLayout->addLayout(hudLayout);
  
  alertsLayout->addSpacing(15);
  
  // Audio Alerts toggle
  QHBoxLayout *audioLayout = new QHBoxLayout();
  QLabel *audioLabel = new QLabel(tr("Audio Alerts"));
  audioLabel->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  audioLayout->addWidget(audioLabel);
  audioLayout->addStretch();
  
  audioToggle = new ToggleSP();
  audioToggle->setFixedSize(150, 80);
  audioToggle->setChecked(params.getBool("RTIAudioAlerts"));
  connect(audioToggle, &ToggleSP::stateChanged, [this](bool checked) {
    params.putBool("RTIAudioAlerts", checked);
  });
  audioLayout->addWidget(audioToggle);
  alertsLayout->addLayout(audioLayout);
  
  scrollLayout->addWidget(alertsFrame);
  
  // Add stretch at the end
  scrollLayout->addStretch();
  
  scrollArea->setWidget(scrollWidget);
  mainLayout->addWidget(scrollArea);
}

QFrame* RTISettingsPanel::createSectionFrame() {
  QFrame *frame = new QFrame();
  frame->setStyleSheet("QFrame { background-color: #292929; border-radius: 20px; padding: 25px; }");
  return frame;
}

void RTISettingsPanel::loadWazeApiKey() {
  // Try to load the Waze API key from various locations
  std::vector<std::string> apiKeyPaths = {
    "/persist/rapidapi_key",
    "/data/persist/rapidapi_key",
    "/data/openpilot/persist/rapidapi_key"
  };
  
  std::string apiKey;
  
  // First check environment variable
  const char* envKey = std::getenv("RAPIDAPI_KEY");
  if (envKey && strlen(envKey) > 0) {
    apiKey = envKey;
  } else {
    // Try each file location
    for (const auto& path : apiKeyPaths) {
      std::ifstream file(path);
      if (file.is_open()) {
        std::getline(file, apiKey);
        file.close();
        // Trim whitespace
        apiKey.erase(0, apiKey.find_first_not_of(" \n\r\t"));
        apiKey.erase(apiKey.find_last_not_of(" \n\r\t") + 1);
        if (!apiKey.empty()) {
          break;
        }
      }
    }
  }
  
  // Store the API key and configure Waze endpoint
  if (!apiKey.empty()) {
    params.put("RTIManualApiKey", apiKey);
    params.put("RTIManualApiEndpoint", "https://waze.p.rapidapi.com/alerts-and-jams");
    params.put("RTIManualApiFormat", "waze_rapid");
    // Force Waze as the data source
    params.put("RTIDataSource", "1");
  }
}