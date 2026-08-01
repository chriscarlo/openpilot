/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/annotated_camera.h"

#include <QBoxLayout>
#include <QPainter>

#include "common/swaglog.h"

AnnotatedCameraWidgetSP::AnnotatedCameraWidgetSP(VisionStreamType type, QWidget *parent)
    : AnnotatedCameraWidget(type, parent) {
  // Bottom-center longitudinal incident flag button.
  //
  // AnnotatedCameraWidget::main_layout is private, so reach the same QVBoxLayout
  // through QWidget::layout() -- it was installed with `new QVBoxLayout(this)`
  // in the base ctor, which sets it as this widget's layout. The base already
  // added the experimental button with AlignTop | AlignRight; the stretch pushes
  // the flag button to the bottom edge of the layout's content box.
  //
  // Layout-managed on purpose: no setGeometry, no resizeEvent. The button is NOT
  // raised above the alert banner -- OnroadWindow's alerts widget is raised and
  // carries WA_TransparentForMouseEvents, so taps still reach the button while
  // safety-critical alert text stays on top.
  if (QBoxLayout *box = qobject_cast<QBoxLayout *>(layout())) {
    flag_btn = new LongitudinalFlagButtonSP(this);
    box->addStretch(1);
    box->addWidget(flag_btn, 0, Qt::AlignBottom | Qt::AlignHCenter);
  } else {
    // Never silently ship a HUD with no flag button: if the base class ever stops
    // installing a QBoxLayout (or installs a non-box layout), the feature would
    // vanish with zero on-device evidence. Make the failure reach rlog.
    LOGE("longitudinal flag button DISABLED: AnnotatedCameraWidget::layout() is %s, not a QBoxLayout",
         layout() == nullptr ? "null" : layout()->metaObject()->className());
  }
}

void AnnotatedCameraWidgetSP::updateState(const UIState &s) {
  AnnotatedCameraWidget::updateState(s);
  // Also update the SP HUD state
  hud_sp.updateState(s);
}

void AnnotatedCameraWidgetSP::paintGL() {
  // First draw everything from the base class (model, base HUD, etc.)
  AnnotatedCameraWidget::paintGL();
  
  // Then draw the SP HUD on top (which includes RTI widget)
  // This draws RTI and any other SP-specific HUD elements
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setPen(Qt::NoPen);
  
  hud_sp.draw(painter, rect());
}
