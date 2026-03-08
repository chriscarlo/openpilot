/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QPushButton>

#include "common/params.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class VibeTuningControl : public AbstractControlSP {
  Q_OBJECT

public:
  explicit VibeTuningControl(QWidget *parent = nullptr);
  void refresh();
  void setSettingsEnabled(bool enabled);

signals:
  void settingsClicked();

protected:
  void showEvent(QShowEvent *event) override;

private:
  QPushButton *settings_btn = nullptr;
  bool settings_enabled = true;
};
