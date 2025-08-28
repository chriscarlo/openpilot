/**
 * VTSC Physics Internals Panel (Sigmoid Center/Steepness)
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_physics_internals_panel.h"

VTSCPhysicsInternalsPanel::VTSCPhysicsInternalsPanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  list_ = new ListWidgetSP(this, false);
  ScrollViewSP *scroll = new ScrollViewSP(list_, this);
  main_layout->addWidget(scroll);

  PanelBackButton *back_btn = new PanelBackButton();
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCPhysicsInternalsPanel::backPress);
  list_->addItem(back_btn);

  // Center
  auto center_btn = new ButtonControlSP(QObject::tr("Sigmoid Center (1/m)"), QObject::tr("Edit"),
                                       QObject::tr("Curvature where the transition centers (e.g., 0.004778)."), this, true);
  QObject::connect(center_btn, &ButtonControlSP::clicked, [=]() {
    Params p;
    QString cur = QString::fromStdString(p.get("VisionTurnSpeedControlPhysicsCenter"));
    QString val = InputDialog::getText(QObject::tr("Sigmoid Center"), nullptr, QObject::tr("Enter float (1/m), e.g., 0.004778"), false, -1, cur.isEmpty() ? "0.004778" : cur);
    if (!val.isEmpty()) p.put("VisionTurnSpeedControlPhysicsCenter", val.toStdString());
  });
  list_->addItem(center_btn);

  // Steepness
  auto steep_btn = new ButtonControlSP(QObject::tr("Sigmoid Steepness (neg)"), QObject::tr("Edit"),
                                      QObject::tr("Transition steepness (negative), e.g., -2000.0"), this, true);
  QObject::connect(steep_btn, &ButtonControlSP::clicked, [=]() {
    Params p;
    QString cur = QString::fromStdString(p.get("VisionTurnSpeedControlPhysicsSteepness"));
    QString val = InputDialog::getText(QObject::tr("Sigmoid Steepness"), nullptr, QObject::tr("Enter negative float, e.g., -2000.0"), false, -1, cur.isEmpty() ? "-2000.0" : cur);
    if (!val.isEmpty()) p.put("VisionTurnSpeedControlPhysicsSteepness", val.toStdString());
  });
  list_->addItem(steep_btn);
}

void VTSCPhysicsInternalsPanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
}

