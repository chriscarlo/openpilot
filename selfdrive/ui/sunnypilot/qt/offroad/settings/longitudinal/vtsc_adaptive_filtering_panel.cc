/**
 * VTSC Adaptive Braking & Filtering Panel
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_adaptive_filtering_panel.h"

VTSCAdaptiveFilteringPanel::VTSCAdaptiveFilteringPanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  list_ = new ListWidgetSP(this, false);
  ScrollViewSP *scroll = new ScrollViewSP(list_, this);
  main_layout->addWidget(scroll);

  PanelBackButton *back_btn = new PanelBackButton();
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCAdaptiveFilteringPanel::backPress);
  list_->addItem(back_btn);

  QPushButton *reset_btn = new QPushButton(tr("Reset to Defaults"));
  reset_btn->setStyleSheet(R"(
    QPushButton { border-radius: 20px; font-size: 45px; font-weight: 500; height: 120px; margin: 20px 40px; color: #FFFFFF; background-color: #393939; }
    QPushButton:pressed { background-color: #4a4a4a; }
  )");
  list_->addItem(reset_btn);

  QObject::connect(reset_btn, &QPushButton::clicked, [=]() {
    params.put("VisionTurnSpeedControlFilterAlpha", "0.30");
    params.put("VisionTurnSpeedControlHysteresisThreshold", "0.20");
    params.put("VisionTurnSpeedControlSafetyBias", "0.10");
    params.put("VisionTurnSpeedControlComfortDecelLimit", "-1.47");
    params.put("VisionTurnSpeedControlComfortJerkLimit", "-2.00");
    params.put("VisionTurnSpeedControlMaxAdaptiveDecel", "-6.00");
    params.put("VisionTurnSpeedControlMaxAdaptiveJerk", "-6.00");
    emit filterAlpha->updateLabels();
    emit hysteresis->updateLabels();
    emit safetyBias->updateLabels();
    emit comfortDecel->updateLabels();
    emit comfortJerk->updateLabels();
    emit maxAdaptiveDecel->updateLabels();
    emit maxAdaptiveJerk->updateLabels();
  });

  // Initialize defaults if unset
  if (QString::fromStdString(params.get("VisionTurnSpeedControlFilterAlpha")).isEmpty()) params.put("VisionTurnSpeedControlFilterAlpha", "0.30");
  if (QString::fromStdString(params.get("VisionTurnSpeedControlHysteresisThreshold")).isEmpty()) params.put("VisionTurnSpeedControlHysteresisThreshold", "0.20");
  if (QString::fromStdString(params.get("VisionTurnSpeedControlSafetyBias")).isEmpty()) params.put("VisionTurnSpeedControlSafetyBias", "0.10");

  addFloatControl(filterAlpha, "VisionTurnSpeedControlFilterAlpha",
                  tr("Filter Alpha"), tr("Smoothing on decel demand. Higher reacts faster; lower is calmer but can delay reaching target."),
                  0.10f, 0.90f, 0.05f);

  addFloatControl(hysteresis, "VisionTurnSpeedControlHysteresisThreshold",
                  tr("Hysteresis"), tr("How far below the comfort limit decel must drop to leave adaptive mode. Higher reduces flapping; too high can linger."),
                  0.10f, 0.50f, 0.05f);

  addFloatControl(safetyBias, "VisionTurnSpeedControlSafetyBias",
                  tr("Safety Bias"), tr("Multiply physics decel to ensure target is reached before apex. Higher is safer but slower."),
                  0.00f, 0.50f, 0.05f);

  addFloatControl(comfortDecel, "VisionTurnSpeedControlComfortDecelLimit",
                  tr("Comfort Decel Limit"), tr("Primary braking strength in normal operation (negative). More negative = stronger braking."),
                  -3.00f, -1.00f, 0.05f, tr("m/s²"), true);

  addFloatControl(comfortJerk, "VisionTurnSpeedControlComfortJerkLimit",
                  tr("Comfort Jerk Limit"), tr("How quickly braking can change (negative). Higher magnitude feels snappier."),
                  -4.00f, -1.00f, 0.05f, tr("m/s³"), true);

  addFloatControl(maxAdaptiveDecel, "VisionTurnSpeedControlMaxAdaptiveDecel",
                  tr("Max Adaptive Decel"), tr("Hard floor for emergency decel (negative). Larger magnitude = stronger emergency braking."),
                  -9.00f, -3.00f, 0.10f, tr("m/s²"), true);

  addFloatControl(maxAdaptiveJerk, "VisionTurnSpeedControlMaxAdaptiveJerk",
                  tr("Max Adaptive Jerk"), tr("Hard floor on how quickly emergency braking can ramp (negative)."),
                  -10.00f, -3.00f, 0.10f, tr("m/s³"), true);
}

void VTSCAdaptiveFilteringPanel::addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
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

void VTSCAdaptiveFilteringPanel::showAllDescriptions() {
  if (filterAlpha) filterAlpha->showDescription();
  if (hysteresis) hysteresis->showDescription();
  if (safetyBias) safetyBias->showDescription();
  if (comfortDecel) comfortDecel->showDescription();
  if (comfortJerk) comfortJerk->showDescription();
  if (maxAdaptiveDecel) maxAdaptiveDecel->showDescription();
  if (maxAdaptiveJerk) maxAdaptiveJerk->showDescription();
}

void VTSCAdaptiveFilteringPanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  showAllDescriptions();
}
