/**
 * VTSC Physics Internals Panel (Sigmoid Center/Steepness)
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class VTSCPhysicsInternalsPanel : public QWidget {
  Q_OBJECT

public:
  explicit VTSCPhysicsInternalsPanel(QWidget *parent = nullptr);

protected:
  void showEvent(QShowEvent *event) override;

signals:
  void backPress();

private:
  ListWidgetSP *list_ = nullptr;
};

