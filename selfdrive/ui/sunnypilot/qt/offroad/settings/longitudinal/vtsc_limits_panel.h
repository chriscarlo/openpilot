/**
 * VTSC Limits Panel
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class VTSCLimitsPanel : public QWidget {
  Q_OBJECT

public:
  explicit VTSCLimitsPanel(QWidget *parent = nullptr);

protected:
  void showEvent(QShowEvent *event) override;

signals:
  void backPress();

private:
  void showAllDescriptions();
  void addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
                       float min, float max, float step, const QString &unit_suffix = "");

  Params params;
  ListWidgetSP *list_ = nullptr;

  OptionControlSP *maxSpeed = nullptr;        // VisionTurnSpeedControlMaxSpeed
  OptionControlSP *minOperatingSpeed = nullptr; // VisionTurnSpeedControlMinOperatingSpeed
};

