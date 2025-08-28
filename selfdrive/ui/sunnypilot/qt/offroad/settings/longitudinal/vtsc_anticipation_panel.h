/**
 * VTSC Anticipation & Overshoot Panel (BSG-aligned)
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class VTSCAnticipationPanel : public QWidget {
  Q_OBJECT

public:
  explicit VTSCAnticipationPanel(QWidget *parent = nullptr);

protected:
  void showEvent(QShowEvent *event) override;

signals:
  void backPress();

private:
  void showAllDescriptions();
  void addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
                       float min, float max, float step, const QString &unit_suffix = "",
                       bool advanced = false);
  void addIntControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
                     int min, int max, int step, const QString &unit_suffix = "",
                     bool advanced = false);

  Params params;
  ListWidgetSP *list_ = nullptr;

  // Controls
  OptionControlSP *planningDecelLimit = nullptr;      // VisionTurnSpeedControlPlanningDecelLimit
  OptionControlSP *overshootSafetyMargin = nullptr;   // VisionTurnSpeedControlOvershootSafetyMargin
  OptionControlSP *overshootMinDistance = nullptr;    // VisionTurnSpeedControlOvershootMinDistance
  OptionControlSP *anticipationTargetReduction = nullptr; // VisionTurnSpeedControlAnticipationTargetReduction
};
