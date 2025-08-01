/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/lateral/live_steer_ratio.h"

#include <cereal/gen/cpp/car.capnp.h>
#include <capnp/message.h>

#include "cereal/messaging/messaging.h"
#include "common/util.h"

LiveSteerRatioControl::LiveSteerRatioControl(QWidget *parent) : QFrame(parent) {
  // Main layout
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 25, 50, 25);
  
  // Title and description
  QLabel *titleLabel = new QLabel(tr("Live Steering Ratio"));
  titleLabel->setStyleSheet("font-size: 50px; font-weight: 400;");
  main_layout->addWidget(titleLabel);
  
  QLabel *descLabel = new QLabel(tr("Manually override the steering ratio. Set to 0 to use the vehicle's default value. Adjust this if the steering feels too sensitive or too slow."));
  descLabel->setWordWrap(true);
  descLabel->setStyleSheet("font-size: 36px; color: #999999; margin-top: 10px; margin-bottom: 20px;");
  main_layout->addWidget(descLabel);
  
  // Control layout
  QHBoxLayout *control_layout = new QHBoxLayout();
  control_layout->setSpacing(20);
  
  // Minus button
  minusBtn = new QPushButton("-");
  minusBtn->setFixedSize(100, 100);
  minusBtn->setStyleSheet(R"(
    QPushButton {
      font-size: 60px;
      font-weight: 500;
      border-radius: 50px;
      background-color: #393939;
      color: #E4E4E4;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
  )");
  control_layout->addWidget(minusBtn);
  
  // Value display
  QVBoxLayout *value_layout = new QVBoxLayout();
  value_layout->setAlignment(Qt::AlignCenter);
  
  valueLabel = new QLabel("0.00");
  valueLabel->setAlignment(Qt::AlignCenter);
  valueLabel->setStyleSheet("font-size: 70px; font-weight: 500; color: #FFFFFF;");
  valueLabel->setFixedWidth(250);
  value_layout->addWidget(valueLabel);
  
  QLabel *statusLabel = new QLabel(tr("(Default)"));
  statusLabel->setAlignment(Qt::AlignCenter);
  statusLabel->setStyleSheet("font-size: 32px; color: #999999;");
  statusLabel->setObjectName("statusLabel");
  value_layout->addWidget(statusLabel);
  
  control_layout->addLayout(value_layout);
  
  // Plus button
  plusBtn = new QPushButton("+");
  plusBtn->setFixedSize(100, 100);
  plusBtn->setStyleSheet(R"(
    QPushButton {
      font-size: 60px;
      font-weight: 500;
      border-radius: 50px;
      background-color: #393939;
      color: #E4E4E4;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
  )");
  control_layout->addWidget(plusBtn);
  
  control_layout->addStretch();
  
  // Reset button
  resetBtn = new QPushButton(tr("Reset"));
  resetBtn->setFixedSize(150, 80);
  resetBtn->setStyleSheet(R"(
    QPushButton {
      font-size: 35px;
      font-weight: 500;
      border-radius: 20px;
      background-color: #393939;
      color: #E4E4E4;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
  )");
  control_layout->addWidget(resetBtn);
  
  main_layout->addLayout(control_layout);
  
  // Get default value
  defaultValue = getDefaultSteerRatio();
  
  // Load current value
  QString storedValue = QString::fromStdString(params.get("LiveSteerRatio"));
  if (storedValue.isEmpty()) {
    currentValue = 0.0f;
  } else {
    currentValue = storedValue.toFloat();
  }
  
  // Connect signals
  connect(minusBtn, &QPushButton::clicked, this, &LiveSteerRatioControl::decrement);
  connect(plusBtn, &QPushButton::clicked, this, &LiveSteerRatioControl::increment);
  connect(resetBtn, &QPushButton::clicked, this, &LiveSteerRatioControl::reset);
  
  // Update display
  updateLabels();
}

void LiveSteerRatioControl::updateLabels() {
  if (currentValue == 0.0f) {
    valueLabel->setText(QString::number(defaultValue, 'f', 2));
    findChild<QLabel*>("statusLabel")->setText(tr("(Default)"));
    findChild<QLabel*>("statusLabel")->setStyleSheet("font-size: 32px; color: #999999;");
  } else {
    valueLabel->setText(QString::number(currentValue, 'f', 2));
    findChild<QLabel*>("statusLabel")->setText(tr("(Modified)"));
    findChild<QLabel*>("statusLabel")->setStyleSheet("font-size: 32px; color: #FFC107;");
  }
  
  // Update button states
  minusBtn->setEnabled(currentValue == 0.0f ? defaultValue > MIN_VALUE : currentValue > MIN_VALUE);
  plusBtn->setEnabled(currentValue == 0.0f ? defaultValue < MAX_VALUE : currentValue < MAX_VALUE);
  resetBtn->setEnabled(currentValue != 0.0f);
}

void LiveSteerRatioControl::increment() {
  if (currentValue == 0.0f) {
    currentValue = defaultValue;
  }
  currentValue = std::min(currentValue + STEP, MAX_VALUE);
  params.put("LiveSteerRatio", QString::number(currentValue, 'f', 2).toStdString());
  updateLabels();
}

void LiveSteerRatioControl::decrement() {
  if (currentValue == 0.0f) {
    currentValue = defaultValue;
  }
  currentValue = std::max(currentValue - STEP, MIN_VALUE);
  params.put("LiveSteerRatio", QString::number(currentValue, 'f', 2).toStdString());
  updateLabels();
}

void LiveSteerRatioControl::reset() {
  currentValue = 0.0f;
  params.put("LiveSteerRatio", "0");
  updateLabels();
}

float LiveSteerRatioControl::getDefaultSteerRatio() {
  // Try to get the default from CarParams
  auto cp_bytes = params.get("CarParams");
  if (!cp_bytes.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();
    return CP.getSteerRatio();
  }
  
  // Fallback to a reasonable default
  return 13.43f;
}