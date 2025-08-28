/**
 * VTSC Limits Panel
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_limits_panel.h"

VTSCLimitsPanel::VTSCLimitsPanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  list_ = new ListWidgetSP(this, false);
  ScrollViewSP *scroll = new ScrollViewSP(list_, this);
  main_layout->addWidget(scroll);

  PanelBackButton *back_btn = new PanelBackButton();
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCLimitsPanel::backPress);
  list_->addItem(back_btn);

  QPushButton *reset_btn = new QPushButton(tr("Reset to Defaults"));
  reset_btn->setStyleSheet(R"(
    QPushButton { border-radius: 20px; font-size: 45px; font-weight: 500; height: 120px; margin: 20px 40px; color: #FFFFFF; background-color: #393939; }
    QPushButton:pressed { background-color: #4a4a4a; }
  )");
  list_->addItem(reset_btn);

  QObject::connect(reset_btn, &QPushButton::clicked, [=]() {
    params.put("VisionTurnSpeedControlMaxSpeed", "70.00");
    params.put("VisionTurnSpeedControlMinOperatingSpeed", "2.24");
    emit maxSpeed->updateLabels();
    emit minOperatingSpeed->updateLabels();
  });

  addFloatControl(maxSpeed, "VisionTurnSpeedControlMaxSpeed",
                  tr("Straight-Road Ceiling"), tr("Speed ceiling when curvature is near zero (m/s)."),
                  10.0f, 90.0f, 1.0f, tr("m/s"));

  addFloatControl(minOperatingSpeed, "VisionTurnSpeedControlMinOperatingSpeed",
                  tr("Min Operating Speed"), tr("Floor for speed clamps to avoid low-speed fighting (m/s)."),
                  0.5f, 10.0f, 0.10f, tr("m/s"));
}

void VTSCLimitsPanel::addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
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

void VTSCLimitsPanel::showAllDescriptions() {
  if (maxSpeed) maxSpeed->showDescription();
  if (minOperatingSpeed) minOperatingSpeed->showDescription();
}

void VTSCLimitsPanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  showAllDescriptions();
}

