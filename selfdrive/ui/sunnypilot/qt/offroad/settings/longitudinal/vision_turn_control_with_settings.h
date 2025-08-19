/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QHBoxLayout>
#include <QPushButton>
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class VisionTurnControlWithSettings : public AbstractControlSP {
  Q_OBJECT

public:
  VisionTurnControlWithSettings(const QString &param, const QString &title, const QString &desc, 
                                const QString &icon, QWidget *parent = nullptr);

  void refresh();  // Public method to refresh toggle state

signals:
  void settingsClicked();
  void toggleFlipped(bool state);

protected:
  void showEvent(QShowEvent *event) override;

private:
  void setupSettingsButton();
  
  QPushButton *settings_btn;
  ToggleSP *toggle;
  QString param_name;
  Params params;
};