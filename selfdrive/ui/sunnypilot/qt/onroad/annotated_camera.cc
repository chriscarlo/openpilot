/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/annotated_camera.h"
#include <QPainter>

AnnotatedCameraWidgetSP::AnnotatedCameraWidgetSP(VisionStreamType type, QWidget *parent)
    : AnnotatedCameraWidget(type, parent) {
}

void AnnotatedCameraWidgetSP::updateState(const UIState &s) {
  AnnotatedCameraWidget::updateState(s);
  // Also update the SP HUD state (only if scene is started)
  if (s.scene.started) {
    hud_sp.updateState(s);
  }
}

void AnnotatedCameraWidgetSP::paintGL() {
  // First draw everything from the base class (model, base HUD, etc.)
  AnnotatedCameraWidget::paintGL();
  
  // Only draw SP HUD if we have a valid UI state
  UIState *s = uiState();
  if (s && s->scene.started) {
    // Then draw the SP HUD on top (which includes RTI widget)
    // This draws RTI and any other SP-specific HUD elements
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(Qt::NoPen);
    
    hud_sp.draw(painter, rect());
  }
}
