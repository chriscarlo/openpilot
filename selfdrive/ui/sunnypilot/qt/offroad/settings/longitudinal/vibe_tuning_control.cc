/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vibe_tuning_control.h"

#include <QHBoxLayout>

VibeTuningControl::VibeTuningControl(QWidget *parent) : AbstractControlSP(
  tr("Vibe Personality Tuning"),
  tr("Open per-mode tuning banks for follow distance, braking floors, and acceleration anchors. Defaults match the current Vibe tables and only apply when the corresponding Vibe toggle is enabled."),
  "",
  parent
) {
  settings_btn = new QPushButton(this);
  settings_btn->setFixedSize(120, 120);
  settings_btn->setStyleSheet(R"(
    QPushButton {
      background-color: #393939;
      border-radius: 60px;
      font-size: 63px;
      font-weight: 500;
      border: 2px solid #696969;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
    QPushButton:disabled {
      background-color: #2d2d2d;
      border-color: #444444;
      color: #696969;
    }
  )");
  settings_btn->setText("⚙");

  QWidget *controls_container = new QWidget(this);
  QHBoxLayout *controls_layout = new QHBoxLayout(controls_container);
  controls_layout->setContentsMargins(0, 0, 0, 0);
  controls_layout->setSpacing(40);
  controls_layout->addWidget(settings_btn);
  hlayout->addWidget(controls_container);

  connect(settings_btn, &QPushButton::clicked, this, &VibeTuningControl::settingsClicked);
}

void VibeTuningControl::showEvent(QShowEvent *event) {
  refresh();
  AbstractControlSP::showEvent(event);
}

void VibeTuningControl::refresh() {
  if (settings_btn) {
    settings_btn->setEnabled(settings_enabled);
  }
}

void VibeTuningControl::setSettingsEnabled(bool enabled) {
  settings_enabled = enabled;
  refresh();
}
