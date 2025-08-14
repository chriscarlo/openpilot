/**
 * RTI Settings Panel - Clean Redesign
 * Complete settings interface for Real-time Traffic Intelligence
 */

#pragma once

#include <QWidget>
#include <QFrame>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QScrollArea>
#include <fstream>
#include <vector>
#include <string>

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/horizontal_carousel.h"
#include "common/params.h"

// Range control widget based on Live Steering Ratio pattern
class RTIRangeControl : public QFrame {
  Q_OBJECT
  
public:
  RTIRangeControl(const QString &title, const QString &description,
                  const QString &paramKey, float minVal, float maxVal, 
                  float step, float defaultVal, const QString &units,
                  QWidget *parent = nullptr);
  
private:
  void updateLabels();
  void increment();
  void decrement();
  void reset();
  
  Params params;
  QString paramKey;
  QString units;
  float currentValue;
  float defaultValue;
  float minValue;
  float maxValue;
  float stepSize;
  
  QLabel *valueLabel;
  QLabel *statusLabel;
  QPushButton *minusBtn;
  QPushButton *plusBtn;
  QPushButton *resetBtn;
};

// Speed reduction control with Posted/Custom selector
class RTISpeedReductionControl : public QFrame {
  Q_OBJECT
  
public:
  RTISpeedReductionControl(QWidget *parent = nullptr);
  
private:
  void updateMode(int index);
  void updateCustomValue();
  void incrementSpeed();
  void decrementSpeed();
  void resetSpeed();
  
  Params params;
  RTISpeedModeCarousel *modeSelector;
  QFrame *customFrame;
  RTIRangeControl *customSpeedControl;
  
  int customSpeed; // in mph
  QLabel *speedValueLabel;
  QLabel *speedStatusLabel;
  QPushButton *speedMinusBtn;
  QPushButton *speedPlusBtn;
  QPushButton *speedResetBtn;
};

// Main settings panel
class RTISettingsPanel : public QFrame {
  Q_OBJECT
  
public:
  RTISettingsPanel(QWidget *parent = nullptr);
  
signals:
  void backPress();
  
private:
  void loadWazeApiKey();
  void setupMainLayout();
  QFrame* createSectionFrame();
  
  Params params;
  
  // UI controls
  RTIThreatFilterCarousel *threatFilterCarousel;
  RTIRangeControl *detectionRadiusControl;
  RTIRangeControl *forwardSlowdownControl;
  RTIRangeControl *resumeSpeedControl;
  RTISpeedReductionControl *speedReductionControl;
  ToggleSP *hudToggle;
  ToggleSP *audioToggle;
  
  // Unit conversion helpers
  static constexpr float MILES_TO_METERS = 1609.344f;
  static constexpr float METERS_TO_MILES = 0.000621371f;
  static constexpr float MPH_TO_KMH = 1.60934f;
  static constexpr float KMH_TO_MPH = 0.621371f;
};