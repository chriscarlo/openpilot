/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "common/params.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class LiveSteerRatioControl : public QFrame {
  Q_OBJECT

public:
  explicit LiveSteerRatioControl(QWidget *parent = nullptr);

private:
  void updateLabels();
  void increment();
  void decrement();
  void reset();
  
  float getDefaultSteerRatio();
  
  Params params;
  QLabel *valueLabel;
  QPushButton *minusBtn;
  QPushButton *plusBtn;
  QPushButton *resetBtn;
  
  float currentValue;
  float defaultValue;
  const float MIN_VALUE = 5.0f;
  const float MAX_VALUE = 25.0f;
  const float STEP = 0.01f;
};