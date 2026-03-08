/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QFrame>
#include <QShowEvent>
#include <QStackedLayout>

class VibeTuningPanel : public QFrame {
  Q_OBJECT

public:
  explicit VibeTuningPanel(QWidget *parent = nullptr);

signals:
  void backPress();

protected:
  void showEvent(QShowEvent *event) override;

private:
  void setupUi();

  QStackedLayout *stacked_layout = nullptr;
  QWidget *hub_screen = nullptr;
};
