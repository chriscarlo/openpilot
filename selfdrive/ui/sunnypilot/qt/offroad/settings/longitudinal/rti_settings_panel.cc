/**
 * RTI Settings Panel Implementation - Clean Redesign
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.h"
#include <QScrollArea>
#include <cmath>
#include "common/util.h"

// Unit conversion constants
static constexpr float MILES_TO_METERS = 1609.344f;
static constexpr float METERS_TO_MILES = 0.000621371f;
static constexpr float MPH_TO_KMH = 1.60934f;
static constexpr float KMH_TO_MPH = 0.621371f;

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
  minusBtn->setFocusPolicy(Qt::NoFocus);  // Prevent focus stealing during touch scrolling
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
  plusBtn->setFocusPolicy(Qt::NoFocus);  // Prevent focus stealing during touch scrolling
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
  resetBtn->setFocusPolicy(Qt::NoFocus);  // Prevent focus stealing during touch scrolling
  controlLayout->addWidget(resetBtn);
  
  mainLayout->addLayout(controlLayout);
  
  // Load current value from params (stored in meters, display in miles)
  reloadFromParams();

  // Connect signals
  connect(minusBtn, &QPushButton::clicked, this, &RTIRangeControl::decrement);
  connect(plusBtn, &QPushButton::clicked, this, &RTIRangeControl::increment);
  connect(resetBtn, &QPushButton::clicked, this, &RTIRangeControl::reset);
}

void RTIRangeControl::reloadFromParams() {
  QString storedValue = QString::fromStdString(params.get(paramKey.toStdString()));
  if (storedValue.isEmpty()) {
    currentValue = defaultValue;
  } else {
    // Convert from meters to miles for display
    float meters = storedValue.toFloat();
    currentValue = meters * METERS_TO_MILES;
  }
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
// IntRangeControl Implementation (raw integer, no unit conversion)
// ============================================================================

IntRangeControl::IntRangeControl(const QString &title, const QString &description,
                                 const QString &paramKey, int minVal, int maxVal,
                                 int step, int defaultVal, const QString &units,
                                 QWidget *parent)
  : QFrame(parent), paramKey(paramKey), units(units),
    minValue(minVal), maxValue(maxVal), stepSize(step), defaultValue(defaultVal) {

  QVBoxLayout *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(0, 0, 0, 0);

  QLabel *titleLabel = new QLabel(title);
  titleLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4;");
  mainLayout->addWidget(titleLabel);

  if (!description.isEmpty()) {
    QLabel *descLabel = new QLabel(description);
    descLabel->setWordWrap(true);
    descLabel->setStyleSheet("font-size: 32px; color: #999999; margin-top: 5px; margin-bottom: 15px;");
    mainLayout->addWidget(descLabel);
  }

  QHBoxLayout *controlLayout = new QHBoxLayout();
  controlLayout->setSpacing(20);

  minusBtn = new QPushButton("-");
  minusBtn->setFixedSize(100, 100);
  minusBtn->setStyleSheet(R"(
    QPushButton { font-size: 60px; font-weight: 500; border-radius: 50px; background-color: #393939; color: #E4E4E4; }
    QPushButton:pressed { background-color: #4a4a4a; }
    QPushButton:disabled { background-color: #2a2a2a; color: #666666; }
  )");
  minusBtn->setFocusPolicy(Qt::NoFocus);
  controlLayout->addWidget(minusBtn);

  QVBoxLayout *valueLayout = new QVBoxLayout();
  valueLayout->setAlignment(Qt::AlignCenter);

  valueLabel = new QLabel("0");
  valueLabel->setAlignment(Qt::AlignCenter);
  valueLabel->setStyleSheet("font-size: 70px; font-weight: 500; color: #FFFFFF;");
  valueLabel->setFixedWidth(300);
  valueLayout->addWidget(valueLabel);

  statusLabel = new QLabel(tr("(Default)"));
  statusLabel->setAlignment(Qt::AlignCenter);
  statusLabel->setStyleSheet("font-size: 32px; color: #999999;");
  valueLayout->addWidget(statusLabel);

  controlLayout->addLayout(valueLayout);

  plusBtn = new QPushButton("+");
  plusBtn->setFixedSize(100, 100);
  plusBtn->setStyleSheet(minusBtn->styleSheet());
  plusBtn->setFocusPolicy(Qt::NoFocus);
  controlLayout->addWidget(plusBtn);

  controlLayout->addStretch();

  resetBtn = new QPushButton(tr("Reset"));
  resetBtn->setFixedSize(150, 80);
  resetBtn->setStyleSheet(R"(
    QPushButton { font-size: 35px; font-weight: 500; border-radius: 20px; background-color: #393939; color: #E4E4E4; }
    QPushButton:pressed { background-color: #4a4a4a; }
    QPushButton:disabled { background-color: #2a2a2a; color: #666666; }
  )");
  resetBtn->setFocusPolicy(Qt::NoFocus);
  controlLayout->addWidget(resetBtn);

  mainLayout->addLayout(controlLayout);

  reloadFromParams();

  connect(minusBtn, &QPushButton::clicked, this, &IntRangeControl::decrement);
  connect(plusBtn, &QPushButton::clicked, this, &IntRangeControl::increment);
  connect(resetBtn, &QPushButton::clicked, this, &IntRangeControl::reset);
}

void IntRangeControl::reloadFromParams() {
  QString storedValue = QString::fromStdString(params.get(paramKey.toStdString()));
  if (storedValue.isEmpty()) {
    currentValue = defaultValue;
  } else {
    currentValue = storedValue.toInt();
  }
  updateLabels();
}

void IntRangeControl::updateLabels() {
  valueLabel->setText(QString::number(currentValue) + " " + units);

  bool isDefault = (currentValue == defaultValue);
  if (isDefault) {
    statusLabel->setText(tr("(Default)"));
    statusLabel->setStyleSheet("font-size: 32px; color: #999999;");
  } else {
    statusLabel->setText(tr("(Modified)"));
    statusLabel->setStyleSheet("font-size: 32px; color: #FFC107;");
  }

  minusBtn->setEnabled(currentValue > minValue);
  plusBtn->setEnabled(currentValue < maxValue);
  resetBtn->setEnabled(!isDefault);
}

void IntRangeControl::increment() {
  currentValue = std::min(currentValue + stepSize, maxValue);
  params.put(paramKey.toStdString(), std::to_string(currentValue));
  updateLabels();
}

void IntRangeControl::decrement() {
  currentValue = std::max(currentValue - stepSize, minValue);
  params.put(paramKey.toStdString(), std::to_string(currentValue));
  updateLabels();
}

void IntRangeControl::reset() {
  currentValue = defaultValue;
  params.put(paramKey.toStdString(), std::to_string(currentValue));
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
  
  // Mode selector carousel
  modeSelector = new RTISpeedModeCarousel(this);
  connect(modeSelector, &HorizontalCarousel::currentIndexChanged, 
          this, &RTISpeedReductionControl::updateMode);
  mainLayout->addWidget(modeSelector);
  
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
  speedMinusBtn->setFocusPolicy(Qt::NoFocus);  // Prevent focus stealing during touch scrolling
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
  speedPlusBtn->setFocusPolicy(Qt::NoFocus);  // Prevent focus stealing during touch scrolling
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
  speedResetBtn->setFocusPolicy(Qt::NoFocus);  // Prevent focus stealing during touch scrolling
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
  mainLayout->setContentsMargins(50, 20, 50, 20);
  mainLayout->setSpacing(30);
  
  // Back button
  PanelBackButton *backBtn = new PanelBackButton(tr("Back"));
  connect(backBtn, &QPushButton::clicked, this, &RTISettingsPanel::backPress);
  mainLayout->addWidget(backBtn, 0, Qt::AlignLeft);
  
  mainLayout->addSpacing(20);
  
  // Title
  QLabel *title = new QLabel(tr("Real-time Traffic Intelligence"));
  title->setStyleSheet("font-size: 50px; font-weight: 600; color: #E4E4E4; padding-bottom: 10px;");
  title->setAlignment(Qt::AlignCenter);
  mainLayout->addWidget(title);
  
  // Description
  QLabel *description = new QLabel(tr("RTI uses Waze traffic data to automatically adjust your speed for safer driving"));
  description->setStyleSheet("font-size: 34px; color: #999999; padding-bottom: 30px;");
  description->setWordWrap(true);
  description->setAlignment(Qt::AlignCenter);
  mainLayout->addWidget(description);
  
  // Threat Filter Section
  QFrame *filterFrame = createSectionFrame();
  QVBoxLayout *filterLayout = new QVBoxLayout(filterFrame);
  
  QLabel *filterLabel = new QLabel(tr("Threat Filter"));
  filterLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  filterLayout->addWidget(filterLabel);
  
  QLabel *filterDesc = new QLabel(tr("Select which types of traffic alerts to monitor"));
  filterDesc->setStyleSheet("font-size: 32px; color: #999999; padding-bottom: 20px;");
  filterDesc->setWordWrap(true);
  filterLayout->addWidget(filterDesc);
  
  // Use the new carousel widget
  threatFilterCarousel = new RTIThreatFilterCarousel(this);
  filterLayout->addWidget(threatFilterCarousel);
  mainLayout->addWidget(filterFrame);
  
  // Detection & Response Settings Section
  QFrame *rangeFrame = createSectionFrame();
  QVBoxLayout *rangeLayout = new QVBoxLayout(rangeFrame);
  
  QLabel *rangeLabel = new QLabel(tr("Detection & Response Settings"));
  rangeLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  rangeLayout->addWidget(rangeLabel);
  
  // Detection Radius - for HUD awareness display
  detectionRadiusControl = new RTIRangeControl(
    tr("Detection Radius"),
    tr("Display threats within this radius around your vehicle for situational awareness (360° coverage)"),
    "RTIDetectionRadius",
    0.25f, 5.0f, 0.25f, 3.0f, "mi",  // Default now 3.0 miles
    this
  );
  rangeLayout->addWidget(detectionRadiusControl);
  
  rangeLayout->addSpacing(20);
  
  // Forward Slowdown Range - when to start slowing for threats ahead
  forwardSlowdownControl = new RTIRangeControl(
    tr("Forward Slowdown Distance"),
    tr("Begin slowing when approaching a threat ahead on your route at this distance"),
    "RTIForwardSlowdownRange",
    0.0f, 2.0f, 0.25f, 0.75f, "mi",
    this
  );
  rangeLayout->addWidget(forwardSlowdownControl);
  
  rangeLayout->addSpacing(20);
  
  // Resume Speed Distance - when to resume normal speed after passing
  resumeSpeedControl = new RTIRangeControl(
    tr("Resume Speed Distance"),
    tr("Resume normal cruise speed after passing a threat by this distance"),
    "RTIResumeSpeedDistance",
    0.0f, 2.0f, 0.25f, 0.75f, "mi",
    this
  );
  rangeLayout->addWidget(resumeSpeedControl);

  rangeLayout->addSpacing(20);

  QLabel *dedupeLabel = new QLabel(tr("Duplicate Alert Merging"));
  dedupeLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-top: 10px; padding-bottom: 15px;");
  rangeLayout->addWidget(dedupeLabel);

  duplicateCollapseControl = new RTIRangeControl(
    tr("General Merge Radius"),
    tr("Merge nearby duplicate reports of the same hazard on your current road into one displayed alert"),
    "RTIDuplicateCollapseRadius",
    0.02f, 0.50f, 0.01f, 110.0f * METERS_TO_MILES, "mi",
    this
  );
  rangeLayout->addWidget(duplicateCollapseControl);

  rangeLayout->addSpacing(20);

  policeCollapseControl = new RTIRangeControl(
    tr("Police Merge Radius"),
    tr("Use a larger merge radius for police reports (for example multiple reports about one officer)"),
    "RTIPoliceCollapseRadius",
    0.02f, 0.75f, 0.01f, 140.0f * METERS_TO_MILES, "mi",
    this
  );
  rangeLayout->addWidget(policeCollapseControl);
  
  mainLayout->addWidget(rangeFrame);
  
  // Speed Reduction Section
  QFrame *speedFrame = createSectionFrame();
  QVBoxLayout *speedLayout = new QVBoxLayout(speedFrame);
  
  speedReductionControl = new RTISpeedReductionControl(this);
  speedLayout->addWidget(speedReductionControl);
  
  mainLayout->addWidget(speedFrame);
  
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
  // Initialize from Params: Toggle uses its own `on` state, not QAbstractButton's checked state
  {
    bool hud_on = params.getBool("RTIHUDEnabled");
    if (hudToggle->on != hud_on) {
      hudToggle->togglePosition();
    }
  }
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
  // Initialize from Params (see note above about Toggle's internal state)
  {
    bool audio_on = params.getBool("RTIAudioAlerts");
    if (audioToggle->on != audio_on) {
      audioToggle->togglePosition();
    }
  }
  connect(audioToggle, &ToggleSP::stateChanged, [this](bool checked) {
    params.putBool("RTIAudioAlerts", checked);
  });
  audioLayout->addWidget(audioToggle);
  alertsLayout->addLayout(audioLayout);
  
  mainLayout->addWidget(alertsFrame);

  // ================================================================
  // Inclement Weather Section
  // ================================================================
  QFrame *weatherFrame = createSectionFrame();
  QVBoxLayout *weatherLayout = new QVBoxLayout(weatherFrame);

  QLabel *weatherSectionLabel = new QLabel(tr("Inclement Weather"));
  weatherSectionLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 5px;");
  weatherLayout->addWidget(weatherSectionLabel);

  QLabel *weatherDesc = new QLabel(tr("Automatically reduce cruise speed based on real-time precipitation data"));
  weatherDesc->setWordWrap(true);
  weatherDesc->setStyleSheet("font-size: 32px; color: #999999; padding-bottom: 20px;");
  weatherLayout->addWidget(weatherDesc);

  // Master toggle
  QHBoxLayout *weatherToggleLayout = new QHBoxLayout();
  QLabel *weatherToggleLabel = new QLabel(tr("Weather-Aware Speed Control"));
  weatherToggleLabel->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  weatherToggleLayout->addWidget(weatherToggleLabel);
  weatherToggleLayout->addStretch();

  weatherToggle = new ToggleSP();
  weatherToggle->setFixedSize(150, 80);
  {
    bool weather_on = params.getBool("WeatherAwareControlEnabled");
    if (weatherToggle->on != weather_on) {
      weatherToggle->togglePosition();
    }
  }
  weatherToggleLayout->addWidget(weatherToggle);
  weatherLayout->addLayout(weatherToggleLayout);

  weatherLayout->addSpacing(20);

  // Speed reduction controls (disabled when toggle is OFF)
  weatherControlsFrame = new QFrame();
  weatherControlsFrame->setStyleSheet("background-color: transparent;");
  QVBoxLayout *weatherControlsLayout = new QVBoxLayout(weatherControlsFrame);
  weatherControlsLayout->setContentsMargins(0, 0, 0, 0);

  weatherLightControl = new IntRangeControl(
    tr("Light Rain Reduction"),
    tr("Speed reduction during light rain or drizzle"),
    "WeatherSpeedReductionLight",
    0, 20, 1, 5, "mph",
    this
  );
  weatherControlsLayout->addWidget(weatherLightControl);
  weatherControlsLayout->addSpacing(15);

  weatherModerateControl = new IntRangeControl(
    tr("Moderate Rain Reduction"),
    tr("Speed reduction during moderate rain or showers"),
    "WeatherSpeedReductionModerate",
    0, 25, 1, 10, "mph",
    this
  );
  weatherControlsLayout->addWidget(weatherModerateControl);
  weatherControlsLayout->addSpacing(15);

  weatherHeavyControl = new IntRangeControl(
    tr("Heavy Rain Reduction"),
    tr("Speed reduction during heavy rain, storms, or freezing rain"),
    "WeatherSpeedReductionHeavy",
    0, 30, 1, 15, "mph",
    this
  );
  weatherControlsLayout->addWidget(weatherHeavyControl);

  weatherLayout->addWidget(weatherControlsFrame);
  weatherControlsFrame->setVisible(params.getBool("WeatherAwareControlEnabled"));

  connect(weatherToggle, &ToggleSP::stateChanged, [this](bool checked) {
    params.putBool("WeatherAwareControlEnabled", checked);
    weatherControlsFrame->setVisible(checked);
  });

  mainLayout->addWidget(weatherFrame);

  // ================================================================
  // Weather Overlay Section
  // ================================================================
  QFrame *weatherOverlayFrame = createSectionFrame();
  QVBoxLayout *weatherOverlayLayout = new QVBoxLayout(weatherOverlayFrame);

  QLabel *weatherOverlaySectionLabel = new QLabel(tr("Weather Overlay"));
  weatherOverlaySectionLabel->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 5px;");
  weatherOverlayLayout->addWidget(weatherOverlaySectionLabel);

  QLabel *weatherOverlayDesc = new QLabel(tr("North-up rain and snow wash on the driving HUD. It appears when precipitation is inside the selected range."));
  weatherOverlayDesc->setWordWrap(true);
  weatherOverlayDesc->setStyleSheet("font-size: 32px; color: #999999; padding-bottom: 20px;");
  weatherOverlayLayout->addWidget(weatherOverlayDesc);

  QHBoxLayout *weatherOverlayToggleLayout = new QHBoxLayout();
  QLabel *weatherOverlayToggleLabel = new QLabel(tr("Show Onroad Overlay"));
  weatherOverlayToggleLabel->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  weatherOverlayToggleLayout->addWidget(weatherOverlayToggleLabel);
  weatherOverlayToggleLayout->addStretch();

  weatherOverlayToggle = new ToggleSP();
  weatherOverlayToggle->setFixedSize(150, 80);
  {
    bool overlay_on = params.getBool("WeatherOverlayEnabled");
    if (weatherOverlayToggle->on != overlay_on) {
      weatherOverlayToggle->togglePosition();
    }
  }
  weatherOverlayToggleLayout->addWidget(weatherOverlayToggle);
  weatherOverlayLayout->addLayout(weatherOverlayToggleLayout);

  weatherOverlayLayout->addSpacing(20);

  weatherOverlayControlsFrame = new QFrame();
  weatherOverlayControlsFrame->setStyleSheet("background-color: transparent;");
  QVBoxLayout *weatherOverlayControlsLayout = new QVBoxLayout(weatherOverlayControlsFrame);
  weatherOverlayControlsLayout->setContentsMargins(0, 0, 0, 0);

  QHBoxLayout *weatherOverlayForceLayout = new QHBoxLayout();
  QLabel *weatherOverlayForceLabel = new QLabel(tr("Force Visible"));
  weatherOverlayForceLabel->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  weatherOverlayForceLayout->addWidget(weatherOverlayForceLabel);
  weatherOverlayForceLayout->addStretch();

  weatherOverlayForceToggle = new ToggleSP();
  weatherOverlayForceToggle->setFixedSize(150, 80);
  {
    bool force_on = params.getBool("WeatherOverlayForceVisible");
    if (weatherOverlayForceToggle->on != force_on) {
      weatherOverlayForceToggle->togglePosition();
    }
  }
  weatherOverlayForceLayout->addWidget(weatherOverlayForceToggle);
  weatherOverlayControlsLayout->addLayout(weatherOverlayForceLayout);
  weatherOverlayControlsLayout->addSpacing(20);

  weatherOverlayRainOpacityControl = new IntRangeControl(
    tr("Rain Opacity"),
    tr("Overall rain-layer alpha applied on the HUD wash"),
    "WeatherOverlayRainOpacity",
    0, 100, 2, 38, "%",
    this
  );
  weatherOverlayControlsLayout->addWidget(weatherOverlayRainOpacityControl);
  weatherOverlayControlsLayout->addSpacing(15);

  weatherOverlaySnowOpacityControl = new IntRangeControl(
    tr("Snow Opacity"),
    tr("Overall snow-layer alpha applied on the HUD wash"),
    "WeatherOverlaySnowOpacity",
    0, 100, 2, 44, "%",
    this
  );
  weatherOverlayControlsLayout->addWidget(weatherOverlaySnowOpacityControl);
  weatherOverlayControlsLayout->addSpacing(15);

  weatherOverlayRangeControl = new IntRangeControl(
    tr("Overlay Range"),
    tr("Show the overlay when precipitation exists anywhere inside this look-ahead distance"),
    "WeatherOverlayRangeKm",
    3, 40, 1, 12, "km",
    this
  );
  weatherOverlayControlsLayout->addWidget(weatherOverlayRangeControl);
  weatherOverlayControlsLayout->addSpacing(15);

  weatherOverlayZoomControl = new IntRangeControl(
    tr("Map Zoom"),
    tr("Tile zoom used to sample the weather layers"),
    "WeatherOverlayZoomLevel",
    6, 13, 1, 10, "z",
    this
  );
  weatherOverlayControlsLayout->addWidget(weatherOverlayZoomControl);
  weatherOverlayControlsLayout->addSpacing(15);

  weatherOverlayRefreshControl = new IntRangeControl(
    tr("Refresh Cadence"),
    tr("How often to refresh the weather tiles from the provider"),
    "WeatherOverlayRefreshSeconds",
    30, 600, 30, 120, "s",
    this
  );
  weatherOverlayControlsLayout->addWidget(weatherOverlayRefreshControl);

  weatherOverlayLayout->addWidget(weatherOverlayControlsFrame);
  weatherOverlayControlsFrame->setVisible(params.getBool("WeatherOverlayEnabled"));

  connect(weatherOverlayToggle, &ToggleSP::stateChanged, [this](bool checked) {
    params.putBool("WeatherOverlayEnabled", checked);
    weatherOverlayControlsFrame->setVisible(checked);
  });
  connect(weatherOverlayForceToggle, &ToggleSP::stateChanged, [this](bool checked) {
    params.putBool("WeatherOverlayForceVisible", checked);
  });

  mainLayout->addWidget(weatherOverlayFrame);

  // Add stretch at the end
  mainLayout->addStretch();
}

QFrame* RTISettingsPanel::createSectionFrame() {
  QFrame *frame = new QFrame();
  frame->setStyleSheet("QFrame { background-color: #292929; border-radius: 20px; padding: 25px; }");
  return frame;
}

void RTISettingsPanel::showEvent(QShowEvent *event) {
  QFrame::showEvent(event);
  // Refresh toggles from Params whenever the panel is shown
  if (hudToggle) {
    bool hud_on = params.getBool("RTIHUDEnabled");
    if (hudToggle->on != hud_on) hudToggle->togglePosition();
  }
  if (audioToggle) {
    bool audio_on = params.getBool("RTIAudioAlerts");
    if (audioToggle->on != audio_on) audioToggle->togglePosition();
  }
  if (weatherToggle) {
    bool weather_on = params.getBool("WeatherAwareControlEnabled");
    if (weatherToggle->on != weather_on) weatherToggle->togglePosition();
    if (weatherControlsFrame) weatherControlsFrame->setVisible(weather_on);
  }
  if (weatherOverlayToggle) {
    bool overlay_on = params.getBool("WeatherOverlayEnabled");
    if (weatherOverlayToggle->on != overlay_on) weatherOverlayToggle->togglePosition();
    if (weatherOverlayControlsFrame) weatherOverlayControlsFrame->setVisible(overlay_on);
  }
  if (weatherOverlayForceToggle) {
    bool force_on = params.getBool("WeatherOverlayForceVisible");
    if (weatherOverlayForceToggle->on != force_on) weatherOverlayForceToggle->togglePosition();
  }
  if (detectionRadiusControl) detectionRadiusControl->reloadFromParams();
  if (forwardSlowdownControl) forwardSlowdownControl->reloadFromParams();
  if (resumeSpeedControl) resumeSpeedControl->reloadFromParams();
  if (duplicateCollapseControl) duplicateCollapseControl->reloadFromParams();
  if (policeCollapseControl) policeCollapseControl->reloadFromParams();
  if (weatherLightControl) weatherLightControl->reloadFromParams();
  if (weatherModerateControl) weatherModerateControl->reloadFromParams();
  if (weatherHeavyControl) weatherHeavyControl->reloadFromParams();
  if (weatherOverlayRainOpacityControl) weatherOverlayRainOpacityControl->reloadFromParams();
  if (weatherOverlaySnowOpacityControl) weatherOverlaySnowOpacityControl->reloadFromParams();
  if (weatherOverlayRangeControl) weatherOverlayRangeControl->reloadFromParams();
  if (weatherOverlayZoomControl) weatherOverlayZoomControl->reloadFromParams();
  if (weatherOverlayRefreshControl) weatherOverlayRefreshControl->reloadFromParams();
}

void RTISettingsPanel::loadWazeApiKey() {
  std::string apiKey;

  // First check environment variables (new provider first, then compatibility aliases)
  const char* envKey = std::getenv("OPENWEBNINJA_API_KEY");
  if (!envKey || strlen(envKey) == 0) {
    envKey = std::getenv("RTI_API_KEY");
  }
  if (!envKey || strlen(envKey) == 0) {
    envKey = std::getenv("WAZE_API_KEY");
  }
  if (!envKey || strlen(envKey) == 0) {
    envKey = std::getenv("RAPIDAPI_KEY");
  }

  if (envKey && strlen(envKey) > 0) {
    apiKey = envKey;
  } else {
    // File lookup order:
    // 1) TICI persistent storage
    // 2) Dev environment fallback
    const std::vector<std::string> keyPaths = {
      "/persist/openwebninja_waze_api_key",
      "/projects/chauffeur/persist/openwebninja_waze_api_key",
    };

    for (const std::string& apiKeyPath : keyPaths) {
      std::ifstream file(apiKeyPath);
      if (!file.is_open()) {
        continue;
      }
      std::string line;
      if (std::getline(file, line) && !line.empty()) {
        apiKey = line;
        break;
      }
    }
  }
  
  // Store the API key and configure Waze endpoint
  if (!apiKey.empty()) {
    params.put("RTIManualApiKey", apiKey);
    params.put("RTIManualApiEndpoint", "https://api.openwebninja.com/waze/alerts-and-jams");
    params.put("RTIManualApiFormat", "waze_openwebninja");
    // Force Waze as the data source
    params.put("RTIDataSource", "1");
  }
}
