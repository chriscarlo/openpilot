/**
 * VTSC Curve Detection Panel
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class VTSCCurveDetectionPanel : public QWidget {
  Q_OBJECT

public:
  explicit VTSCCurveDetectionPanel(QWidget *parent = nullptr);

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

  OptionControlSP *curvEMA = nullptr;           // VisionTurnSpeedControlCurvatureEMAFactor
  ButtonControlSP *apexThresholdEdit = nullptr; // VisionTurnSpeedControlApexThreshold
  ButtonControlSP *apexProminenceEdit = nullptr;// VisionTurnSpeedControlApexProminence
  OptionControlSP *apexHysteresisTime = nullptr;// VisionTurnSpeedControlApexHysteresisTime
  OptionControlSP *apexMetersPerIndex = nullptr;// VisionTurnSpeedControlApexMetersPerIndex
  OptionControlSP *apexNearIndex = nullptr;     // VisionTurnSpeedControlApexNearIndex
};

