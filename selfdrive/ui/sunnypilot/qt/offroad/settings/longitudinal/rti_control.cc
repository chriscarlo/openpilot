/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_control.h"

RTIControl::RTIControl(const QString &param, const QString &title, const QString &desc, const QString &icon, QWidget *parent)
    : ExpandableToggleRow(param, title, desc, icon, parent) {

  auto *rtiFrame = new QFrame(this);
  auto *rtiFrameLayout = new QVBoxLayout();
  rtiFrame->setLayout(rtiFrameLayout);
  rtiFrameLayout->setSpacing(0);
  rtiFrameLayout->setContentsMargins(0, 0, 0, 0);

  rtiSettings = new PushButtonSP(tr("Customize RTI"));
  rtiFrameLayout->addWidget(rtiSettings);
  connect(rtiSettings, &QPushButton::clicked, [&]() {
    emit rtiSettingsButtonClicked();
  });
  addItem(rtiFrame);
}