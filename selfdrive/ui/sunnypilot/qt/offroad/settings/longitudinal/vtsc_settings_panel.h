/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QWidget>
#include <QStackedLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QFrame>
#include <QLabel>

#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class VTSCIconButton : public QPushButton {
  Q_OBJECT

public:
  VTSCIconButton(const QString &icon_path, const QString &text, QWidget *parent = nullptr);
  
signals:
  void buttonClicked();
};

class VTSCSettingsPanel : public QFrame {
  Q_OBJECT

public:
  explicit VTSCSettingsPanel(QWidget *parent = nullptr);

signals:
  void backPress();
  void anticipationSettingsClicked();
  
private:
  void setupUI();
  void createIconGrid();
  
  QStackedLayout *main_layout;
  QWidget *icon_grid_screen;
  QPushButton *back_btn;
};