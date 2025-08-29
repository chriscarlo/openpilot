/**
 * VTSC Smoothing Limits Panel
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_smoothing_limits_panel.h"

VTSCSmoothingLimitsPanel::VTSCSmoothingLimitsPanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  list_ = new ListWidgetSP(this, false);
  ScrollViewSP *scroll = new ScrollViewSP(list_, this);
  main_layout->addWidget(scroll);

  PanelBackButton *back_btn = new PanelBackButton();
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCSmoothingLimitsPanel::backPress);
  list_->addItem(back_btn);

  QPushButton *reset_btn = new QPushButton(tr("Reset to Defaults"));
  reset_btn->setStyleSheet(R"(
    QPushButton { border-radius: 20px; font-size: 45px; font-weight: 500; height: 120px; margin: 20px 40px; color: #FFFFFF; background-color: #393939; }
    QPushButton:pressed { background-color: #4a4a4a; }
  )");
  list_->addItem(reset_btn);

  QObject::connect(reset_btn, &QPushButton::clicked, [=]() {
    params.put("VisionTurnSpeedControlSmoothingMaxDecel", "3.50");
    params.put("VisionTurnSpeedControlSmoothingMaxJerk", "6.00");
    params.put("VisionTurnSpeedControlAccelToDecelRatio", "1.30");
    params.put("VisionTurnSpeedControlJerkAccelMultiplier", "2.00");
    emit smoothingMaxDecel->updateLabels();
    emit smoothingMaxJerk->updateLabels();
    emit accelToDecelRatio->updateLabels();
    emit jerkAccelMultiplier->updateLabels();
  });

  // Initialize defaults if unset
  if (QString::fromStdString(params.get("VisionTurnSpeedControlSmoothingMaxDecel")).isEmpty()) params.put("VisionTurnSpeedControlSmoothingMaxDecel", "3.50");
  if (QString::fromStdString(params.get("VisionTurnSpeedControlSmoothingMaxJerk")).isEmpty()) params.put("VisionTurnSpeedControlSmoothingMaxJerk", "6.00");
  if (QString::fromStdString(params.get("VisionTurnSpeedControlAccelToDecelRatio")).isEmpty()) params.put("VisionTurnSpeedControlAccelToDecelRatio", "1.30");
  if (QString::fromStdString(params.get("VisionTurnSpeedControlJerkAccelMultiplier")).isEmpty()) params.put("VisionTurnSpeedControlJerkAccelMultiplier", "2.00");

  addFloatControl(smoothingMaxDecel, "VisionTurnSpeedControlSmoothingMaxDecel",
                  tr("Max Decel (smoothing)"), tr("Clamp how quickly target accel can decrease per update. Lower feels smoother; too low can miss targets."),
                  1.0f, 7.0f, 0.10f, tr("m/s²"));

  addFloatControl(smoothingMaxJerk, "VisionTurnSpeedControlSmoothingMaxJerk",
                  tr("Max Jerk (smoothing)"), tr("Clamp on how quickly target accel may change. Higher is snappier; lower is gentler."),
                  1.0f, 12.0f, 0.10f, tr("m/s³"));

  addFloatControl(accelToDecelRatio, "VisionTurnSpeedControlAccelToDecelRatio",
                  tr("Accel to Decel Ratio"), tr("Positive accel cap relative to decel cap. >1 lets you speed up faster than you slow down."),
                  1.0f, 1.6f, 0.05f);

  addFloatControl(jerkAccelMultiplier, "VisionTurnSpeedControlJerkAccelMultiplier",
                  tr("Jerk Accel Multiplier"), tr("Ratio of positive to negative jerk (how quickly you can ramp accel after a curve)."),
                  1.0f, 3.0f, 0.10f);
}

void VTSCSmoothingLimitsPanel::addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
                                               float min, float max, float step, const QString &unit_suffix) {
  const int per = std::max(1, static_cast<int>(std::round(step * 100.0f)));
  ptr = new OptionControlSP(param, title, desc, "../assets/offroad/icon_blank.png",
                            {static_cast<int>(std::round(min * 100.0f)), static_cast<int>(std::round(max * 100.0f))},
                            per, false, nullptr, true);
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

void VTSCSmoothingLimitsPanel::showAllDescriptions() {
  if (smoothingMaxDecel) smoothingMaxDecel->showDescription();
  if (smoothingMaxJerk) smoothingMaxJerk->showDescription();
  if (accelToDecelRatio) accelToDecelRatio->showDescription();
  if (jerkAccelMultiplier) jerkAccelMultiplier->showDescription();
}

void VTSCSmoothingLimitsPanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  showAllDescriptions();
}
