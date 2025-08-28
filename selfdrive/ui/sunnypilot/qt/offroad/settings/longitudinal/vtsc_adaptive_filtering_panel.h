/**
 * VTSC Adaptive Braking & Filtering Panel
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class VTSCAdaptiveFilteringPanel : public QWidget {
  Q_OBJECT

public:
  explicit VTSCAdaptiveFilteringPanel(QWidget *parent = nullptr);

protected:
  void showEvent(QShowEvent *event) override;

signals:
  void backPress();

private:
  void showAllDescriptions();
  void addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
                       float min, float max, float step, const QString &unit_suffix = "",
                       bool advanced = false);

  Params params;
  ListWidgetSP *list_ = nullptr;

  // Basic filtering
  OptionControlSP *filterAlpha = nullptr;         // VisionTurnSpeedControlFilterAlpha
  OptionControlSP *hysteresis = nullptr;          // VisionTurnSpeedControlHysteresisThreshold
  OptionControlSP *safetyBias = nullptr;          // VisionTurnSpeedControlSafetyBias

  // Comfort/adaptive (advanced)
  OptionControlSP *comfortDecel = nullptr;        // VisionTurnSpeedControlComfortDecelLimit
  OptionControlSP *comfortJerk = nullptr;         // VisionTurnSpeedControlComfortJerkLimit
  OptionControlSP *maxAdaptiveDecel = nullptr;    // VisionTurnSpeedControlMaxAdaptiveDecel
  OptionControlSP *maxAdaptiveJerk = nullptr;     // VisionTurnSpeedControlMaxAdaptiveJerk
};

