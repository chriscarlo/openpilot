/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

// Must come before annotated_camera.h: that header defines
// `ExperimentalButton -> ExperimentalButtonSP` (and friends) after its own
// includes, and this header must be parsed with those macros inactive.
#include "selfdrive/ui/sunnypilot/qt/onroad/buttons.h"

#include "selfdrive/ui/qt/onroad/annotated_camera.h"
#include "selfdrive/ui/sunnypilot/qt/onroad/hud.h"

class AnnotatedCameraWidgetSP : public AnnotatedCameraWidget {
  Q_OBJECT

public:
  explicit AnnotatedCameraWidgetSP(VisionStreamType type, QWidget *parent = nullptr);
  void updateState(const UIState &s) override;
  void paintGL() override;

private:
  HudRendererSP hud_sp;
  LongitudinalFlagButtonSP *flag_btn = nullptr;
};
