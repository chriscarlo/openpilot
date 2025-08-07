/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vision_turn_control_with_settings.h"
#include <QShowEvent>

VisionTurnControlWithSettings::VisionTurnControlWithSettings(const QString &param, const QString &title, 
                                                             const QString &desc, const QString &icon, 
                                                             QWidget *parent)
  : AbstractControlSP(title, desc, icon, parent), param_name(param) {
  
  // Create the toggle
  toggle = new ToggleSP(this);
  toggle->setFixedSize(150, 100);
  
  // Setup settings button
  setupSettingsButton();
  
  // Create a container for toggle and settings button
  QWidget *controls_container = new QWidget(this);
  QHBoxLayout *controls_layout = new QHBoxLayout(controls_container);
  controls_layout->setContentsMargins(0, 0, 0, 0);
  controls_layout->setSpacing(20);
  
  controls_layout->addWidget(toggle);
  controls_layout->addWidget(settings_btn);
  
  hlayout->addWidget(controls_container);
  
  // Connect toggle
  QObject::connect(toggle, &Toggle::stateChanged, this, [this](bool state) {
    params.putBool(param_name.toStdString(), state);
    emit toggleFlipped(state);
    refresh();
  });
  
  // Initial state
  refresh();
}

void VisionTurnControlWithSettings::showEvent(QShowEvent *event) {
  refresh();  // Refresh state when widget is shown
  AbstractControlSP::showEvent(event);
}

void VisionTurnControlWithSettings::setupSettingsButton() {
  settings_btn = new QPushButton(this);
  settings_btn->setObjectName("vtsc_settings_btn");
  settings_btn->setFixedSize(100, 100);
  
  // Style the button with a gear icon
  settings_btn->setStyleSheet(R"(
    QPushButton {
      background-color: #393939;
      border-radius: 50px;
      font-size: 40px;
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
  
  QObject::connect(settings_btn, &QPushButton::clicked, this, &VisionTurnControlWithSettings::settingsClicked);
}

void VisionTurnControlWithSettings::refresh() {
  bool enabled = params.getBool(param_name.toStdString());
  if (enabled != toggle->on) {
    toggle->togglePosition();
  }
  settings_btn->setEnabled(enabled);
}