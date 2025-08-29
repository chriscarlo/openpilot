/**
 * VTSC Apex & Exit Boost Panel
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_apex_boost_panel.h"

VTSCApexBoostPanel::VTSCApexBoostPanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  list_ = new ListWidgetSP(this, false);
  ScrollViewSP *scroll = new ScrollViewSP(list_, this);
  main_layout->addWidget(scroll);

  PanelBackButton *back_btn = new PanelBackButton();
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCApexBoostPanel::backPress);
  list_->addItem(back_btn);

  QPushButton *reset_btn = new QPushButton(tr("Reset to Defaults"));
  reset_btn->setStyleSheet(R"(
    QPushButton { border-radius: 20px; font-size: 45px; font-weight: 500; height: 120px; margin: 20px 40px; color: #FFFFFF; background-color: #393939; }
    QPushButton:pressed { background-color: #4a4a4a; }
  )");
  list_->addItem(reset_btn);

  QObject::connect(reset_btn, &QPushButton::clicked, [=]() {
    params.put("VisionTurnSpeedControlApexBoostDistance", "50.0");
    params.put("VisionTurnSpeedControlApexBoostFactor", "0.10");
    params.put("VisionTurnSpeedControlApexBoostMinLatAccel", "1.00");
    params.put("VisionTurnSpeedControlApexBoostCenter", "2.00");
    params.put("VisionTurnSpeedControlApexBoostWidth", "0.50");
    params.put("VisionTurnSpeedControlBoostSafetyCurvatureScale", "0.70");
    params.put("VisionTurnSpeedControlApexThreshold", "0.00005");
    params.put("VisionTurnSpeedControlApexProminence", "0.0001");
    params.put("VisionTurnSpeedControlApexHysteresisTime", "2.0");
    params.put("VisionTurnSpeedControlApexMetersPerIndex", "2.0");
    params.put("VisionTurnSpeedControlApexNearIndex", "3");
  });

  // Initialize defaults if unset
  auto ensure = [&](const char *k, const char *v){ if (QString::fromStdString(params.get(k)).isEmpty()) params.put(k, v); };
  ensure("VisionTurnSpeedControlApexBoostDistance", "50.0");
  ensure("VisionTurnSpeedControlApexBoostFactor", "0.10");
  ensure("VisionTurnSpeedControlApexBoostMinLatAccel", "1.00");
  ensure("VisionTurnSpeedControlApexBoostCenter", "2.00");
  ensure("VisionTurnSpeedControlApexBoostWidth", "0.50");
  ensure("VisionTurnSpeedControlBoostSafetyCurvatureScale", "0.70");

  addFloatControl(apexBoostDistance, "VisionTurnSpeedControlApexBoostDistance",
                  tr("Boost Distance"), tr("Apply exit boost within this distance past the apex. Larger values keep boost longer; too large can feel pushy."),
                  0.0f, 300.0f, 5.0f, tr("m"));

  addFloatControl(apexBoostFactor, "VisionTurnSpeedControlApexBoostFactor",
                  tr("Boost Factor"), tr("How much to multiply the base target on exit (sigmoid-based). Start small (0.05–0.15) to avoid overshoot."),
                  0.0f, 0.5f, 0.01f, tr("×"));

  addFloatControl(apexBoostMinLat, "VisionTurnSpeedControlApexBoostMinLatAccel",
                  tr("Min Lateral Accel"), tr("Only boost when actual lateral accel exceeds this (avoid boosting on straights)."),
                  0.0f, 5.0f, 0.10f, tr("m/s²"));

  addFloatControl(apexBoostCenter, "VisionTurnSpeedControlApexBoostCenter",
                  tr("Boost Center"), tr("Lateral accel where boost reaches mid-strength."),
                  0.0f, 5.0f, 0.10f, tr("m/s²"));

  addFloatControl(apexBoostWidth, "VisionTurnSpeedControlApexBoostWidth",
                  tr("Boost Width"), tr("How quickly boost ramps with lateral accel. Wider = smoother, narrower = punchier."),
                  0.05f, 5.0f, 0.05f, tr("m/s²"));

  addFloatControl(boostCurvScale, "VisionTurnSpeedControlBoostSafetyCurvatureScale",
                  tr("Safety Curvature Scale"), tr("Safety scaling on curvature when computing physics max during boost. Lower is safer."),
                  0.50f, 1.00f, 0.05f);

  // Detection controls moved to Curve Detection panel
}

void VTSCApexBoostPanel::addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
                                         float min, float max, float step, const QString &unit_suffix, bool advanced) {
  const int per = std::max(1, static_cast<int>(std::round(step * 100.0f)));
  ptr = new OptionControlSP(param, title, desc, "../assets/offroad/icon_blank.png",
                            {static_cast<int>(std::round(min * 100.0f)), static_cast<int>(std::round(max * 100.0f))},
                            per, false, nullptr, true, advanced);
  QWidget *container = new QWidget();
  QHBoxLayout *layout = new QHBoxLayout(container);
  layout->setContentsMargins(40, 0, 0, 0);
  layout->setAlignment(Qt::AlignLeft);
  layout->addWidget(ptr, 0, Qt::AlignLeft);
  QObject::connect(ptr, &OptionControlSP::updateLabels, [=]() {
    const auto val = QString::fromStdString(params.get(param));
    ptr->setLabel(val + (unit_suffix.isEmpty() ? "" : (" " + unit_suffix)));
  });
  const auto initial = QString::fromStdString(params.get(param));
  ptr->setLabel(initial + (unit_suffix.isEmpty() ? "" : (" " + unit_suffix)));
  list_->addItem(container);
}

void VTSCApexBoostPanel::addIntControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
                                       int min, int max, int step, const QString &unit_suffix, bool advanced) {
  ptr = new OptionControlSP(param, title, desc, "../assets/offroad/icon_blank.png",
                            {min, max}, step, false, nullptr, false, advanced);
  QWidget *container = new QWidget();
  QHBoxLayout *layout = new QHBoxLayout(container);
  layout->setContentsMargins(40, 0, 0, 0);
  layout->setAlignment(Qt::AlignLeft);
  layout->addWidget(ptr, 0, Qt::AlignLeft);
  QObject::connect(ptr, &OptionControlSP::updateLabels, [=]() {
    const auto val = QString::fromStdString(params.get(param));
    ptr->setLabel(val + (unit_suffix.isEmpty() ? "" : (" " + unit_suffix)));
  });
  const auto initial = QString::fromStdString(params.get(param));
  ptr->setLabel(initial + (unit_suffix.isEmpty() ? "" : (" " + unit_suffix)));
  list_->addItem(container);
}

void VTSCApexBoostPanel::showAllDescriptions() {
  if (apexBoostDistance) apexBoostDistance->showDescription();
  if (apexBoostFactor) apexBoostFactor->showDescription();
  if (apexBoostMinLat) apexBoostMinLat->showDescription();
  if (apexBoostCenter) apexBoostCenter->showDescription();
  if (apexBoostWidth) apexBoostWidth->showDescription();
  if (boostCurvScale) boostCurvScale->showDescription();
  if (apexHysteresisTime) apexHysteresisTime->showDescription();
  if (apexMetersPerIndex) apexMetersPerIndex->showDescription();
  if (apexNearIndex) apexNearIndex->showDescription();
}

void VTSCApexBoostPanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  showAllDescriptions();
}
