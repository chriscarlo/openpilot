/**
 * RTI Control Widget - Clean Redesign
 * Real-time Traffic Intelligence main toggle control
 */

#pragma once

#include <QWidget>
#include <QFrame>
#include <QVBoxLayout>
#include <QPushButton>
#include <QShowEvent>

#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "common/params.h"

class RTIControl : public AbstractControlSP {
  Q_OBJECT

public:
  RTIControl(QWidget *parent = nullptr);
  void refresh();

protected:
  void showEvent(QShowEvent *event) override;

signals:
  void settingsClicked();

private:
  Params params;
  ToggleSP *toggle;
  QPushButton *settingsBtn;
  
  void updateState(bool enabled);
};