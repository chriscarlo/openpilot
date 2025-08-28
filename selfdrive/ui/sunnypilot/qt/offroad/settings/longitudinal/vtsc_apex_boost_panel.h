/**
 * VTSC Apex & Exit Boost Panel
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class VTSCApexBoostPanel : public QWidget {
  Q_OBJECT

public:
  explicit VTSCApexBoostPanel(QWidget *parent = nullptr);

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

  // Boost behavior
  OptionControlSP *apexBoostDistance = nullptr;   // VisionTurnSpeedControlApexBoostDistance
  OptionControlSP *apexBoostFactor = nullptr;     // VisionTurnSpeedControlApexBoostFactor
  OptionControlSP *apexBoostMinLat = nullptr;     // VisionTurnSpeedControlApexBoostMinLatAccel
  OptionControlSP *apexBoostCenter = nullptr;     // VisionTurnSpeedControlApexBoostCenter
  OptionControlSP *apexBoostWidth = nullptr;      // VisionTurnSpeedControlApexBoostWidth
  OptionControlSP *boostCurvScale = nullptr;      // VisionTurnSpeedControlBoostSafetyCurvatureScale

  // Detection (advanced; freeform for small thresholds)
  ButtonControlSP *apexThresholdEdit = nullptr;   // VisionTurnSpeedControlApexThreshold
  ButtonControlSP *apexProminenceEdit = nullptr;  // VisionTurnSpeedControlApexProminence
  OptionControlSP *apexHysteresisTime = nullptr;  // VisionTurnSpeedControlApexHysteresisTime
  OptionControlSP *apexMetersPerIndex = nullptr;  // VisionTurnSpeedControlApexMetersPerIndex
  OptionControlSP *apexNearIndex = nullptr;       // VisionTurnSpeedControlApexNearIndex
};

