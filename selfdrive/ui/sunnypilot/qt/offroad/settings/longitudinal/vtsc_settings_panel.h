/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QWidget>
#include <QFrame>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QShowEvent>

#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/horizontal_carousel.h"

class VTSCSettingsPanel : public QFrame {
  Q_OBJECT

public:
  explicit VTSCSettingsPanel(QWidget *parent = nullptr);
  
protected:
  void showEvent(QShowEvent *event) override;

signals:
  void backPress();
  
private:
  void setupUI();
  QFrame* createSectionFrame();

  // Persisted control handles for refresh
  ToggleSP *mapTog_ = nullptr;
  HorizontalCarousel *mapStrategyCarousel_ = nullptr;
  ToggleSP *bypassTog_ = nullptr;
  ToggleSP *dbgTog_ = nullptr;
  ToggleSP *recorderTog_ = nullptr;
  QLabel *curvePhaseValLabel_ = nullptr;
  QLabel *curvePhaseStatusLabel_ = nullptr;
  QLabel *overshootPhaseValLabel_ = nullptr;
  QLabel *overshootPhaseStatusLabel_ = nullptr;
  QLabel *apexExitValLabel_ = nullptr;
  QLabel *apexExitStatusLabel_ = nullptr;
  QLabel *headValLabel_ = nullptr;
  QLabel *headStatusLabel_ = nullptr;
};
