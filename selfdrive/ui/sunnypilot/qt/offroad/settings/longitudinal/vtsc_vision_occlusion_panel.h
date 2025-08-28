/**
 * VTSC Vision Occlusion Panel
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class VTSCVisionOcclusionPanel : public QWidget {
  Q_OBJECT

public:
  explicit VTSCVisionOcclusionPanel(QWidget *parent = nullptr);

protected:
  void showEvent(QShowEvent *event) override;

signals:
  void backPress();

private:
  void showAllDescriptions();
  void addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
                       float min, float max, float step);

  Params params;
  ListWidgetSP *list_ = nullptr;

  OptionControlSP *confAlpha = nullptr;  // VisionTurnSpeedControlVisionConfAlpha
  OptionControlSP *confGood = nullptr;   // VisionTurnSpeedControlVisionConfGoodThreshold
  OptionControlSP *confBad = nullptr;    // VisionTurnSpeedControlVisionConfBadThreshold
};

