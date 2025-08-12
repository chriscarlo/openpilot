/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/slc/speed_limit_control.h"
#include <QHBoxLayout>
#include <QShowEvent>

SpeedLimitControl::SpeedLimitControl(QWidget *parent) : AbstractControlSP(
  tr("Speed Limit Control (SLC)"),
  tr("When you engage ACC, you will be prompted to set the cruising speed to the speed limit of the road adjusted by the Offset and Source Policy specified, or the current driving speed. "
    "The maximum cruising speed will always be the MAX set speed."),
  "",
  parent
) {
  // Create toggle
  toggle = new ToggleSP(this);
  toggle->setFixedSize(150, 100);
  
  // Setup settings button
  setupSettingsButton();
  
  // Create a container for settings button and toggle (swapped order)
  QWidget *controls_container = new QWidget(this);
  QHBoxLayout *controls_layout = new QHBoxLayout(controls_container);
  controls_layout->setContentsMargins(0, 0, 0, 0);
  controls_layout->setSpacing(30);  // Increased from 20px to 30px (50% increase)
  
  // Add settings button first, then toggle (swapped positions)
  controls_layout->addWidget(settings_btn);
  controls_layout->addWidget(toggle);
  
  hlayout->addWidget(controls_container);
  
  // Connect toggle
  QObject::connect(toggle, &Toggle::stateChanged, this, [this](bool state) {
    params.putBool("SpeedLimitControl", state);
    emit toggleFlipped(state);
    refresh();
  });
}

void SpeedLimitControl::showEvent(QShowEvent *event) {
  refresh();
  AbstractControlSP::showEvent(event);
}

void SpeedLimitControl::setupSettingsButton() {
  settings_btn = new QPushButton(this);
  settings_btn->setObjectName("slc_settings_btn");
  settings_btn->setFixedSize(120, 120);  // 20% larger than original
  
  // Style the button with a gear icon
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
  
  settings_btn->setText("⚙");  // Gear emoji
  
  QObject::connect(settings_btn, &QPushButton::clicked, this, &SpeedLimitControl::settingsClicked);
}

void SpeedLimitControl::refresh() {
  if (!toggle || !settings_btn) {
    return;
  }
  
  bool enabled = params.getBool("SpeedLimitControl");
  if (enabled != toggle->on) {
    toggle->togglePosition();
  }
  settings_btn->setEnabled(enabled);
}
