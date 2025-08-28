/**
 * VTSC Curve Physics Panel (Advanced)
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class VTSCPhysicsPanel : public QWidget {
  Q_OBJECT

public:
  explicit VTSCPhysicsPanel(QWidget *parent = nullptr);

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

  // Phys sigmoid core
  OptionControlSP *physBaseline = nullptr;   // VisionTurnSpeedControlPhysicsBaseline
  OptionControlSP *physAmplitude = nullptr;  // VisionTurnSpeedControlPhysicsAmplitude
  OptionControlSP *physMinLat = nullptr;     // VisionTurnSpeedControlPhysicsMinLatAccel
  OptionControlSP *physMaxLat = nullptr;     // VisionTurnSpeedControlPhysicsMaxLatAccel

  // Freeform inputs
  ButtonControlSP *physCenterEdit = nullptr;    // VisionTurnSpeedControlPhysicsCenter
  ButtonControlSP *physSteepnessEdit = nullptr; // VisionTurnSpeedControlPhysicsSteepness
};

