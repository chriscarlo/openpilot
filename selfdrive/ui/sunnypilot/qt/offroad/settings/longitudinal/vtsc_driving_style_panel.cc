/**
 * VTSC Driving Style Panel
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_driving_style_panel.h"

VTSCDrivingStylePanel::VTSCDrivingStylePanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  list_ = new ListWidgetSP(this, false);
  ScrollViewSP *scroll = new ScrollViewSP(list_, this);
  main_layout->addWidget(scroll);

  // Back button
  PanelBackButton *back_btn = new PanelBackButton();
  back_btn->setStyleSheet(R"(
    QPushButton#back_btn {
      border: 4px solid #393939 !important;
      border-radius: 30px;
      font-size: 50px;
      margin: 0px;
      padding: 10px;
      color: #dddddd;
      background-color: #393939;
    }
    QPushButton#back_btn:pressed { background-color: #4a4a4a; }
  )");
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCDrivingStylePanel::backPress);
  list_->addItem(back_btn);

  // Reset button
  QPushButton *reset_btn = new QPushButton(tr("Reset to Defaults"));
  reset_btn->setStyleSheet(R"(
    QPushButton {
      border-radius: 20px;
      font-size: 45px;
      font-weight: 500;
      height: 120px;
      margin: 20px 40px 20px 40px;
      color: #FFFFFF;
      background-color: #393939;
    }
    QPushButton:pressed { background-color: #4a4a4a; }
  )");
  list_->addItem(reset_btn);

  QObject::connect(reset_btn, &QPushButton::clicked, [=]() {
    params.put("VisionTurnSpeedControlAggressiveness", "1.00");
    params.put("VisionTurnSpeedControlFixedLeadTimeSeconds", "0.00");
    params.put("VisionTurnSpeedControlLowSpeedSpeedBiasMph", "0.00");
    params.put("VisionTurnSpeedControlLowSpeedBiasEndMph", "50.00");
    params.put("VisionTurnSpeedControlSpeedIncreaseFactor", "1.00");

    emit aggressiveness->updateLabels();
    emit fixedLeadTime->updateLabels();
    emit lowSpeedBiasMph->updateLabels();
    emit lowSpeedBiasEndMph->updateLabels();
    emit speedIncreaseFactor->updateLabels();
  });

  addFloatControl(
    aggressiveness,
    "VisionTurnSpeedControlAggressiveness",
    tr("Anticipation Aggressiveness"),
    tr("How early to start slowing for curves. Higher = earlier/more conservative."),
    0.50f, 2.00f, 0.05f, tr("×")
  );

  addFloatControl(
    fixedLeadTime,
    "VisionTurnSpeedControlFixedLeadTimeSeconds",
    tr("Fixed Lead Time"),
    tr("Override dynamic timing with a fixed time buffer before curves. 0 disables."),
    0.0f, 10.0f, 0.10f, tr("s")
  );

  addFloatControl(
    lowSpeedBiasMph,
    "VisionTurnSpeedControlLowSpeedSpeedBiasMph",
    tr("Low-Speed Speed Bias"),
    tr("Add/subtract mph from physics target under the end speed."),
    -5.0f, 5.0f, 0.10f, tr("mph")
  );

  addFloatControl(
    lowSpeedBiasEndMph,
    "VisionTurnSpeedControlLowSpeedBiasEndMph",
    tr("Low-Speed Bias End"),
    tr("Bias tapers to zero by this speed."),
    10.0f, 80.0f, 1.0f, tr("mph")
  );

  addFloatControl(
    speedIncreaseFactor,
    "VisionTurnSpeedControlSpeedIncreaseFactor",
    tr("Global Speed Bias"),
    tr("Multiply physics target speed globally. Keep near 1.0."),
    0.50f, 1.50f, 0.05f, tr("×")
  );
}

void VTSCDrivingStylePanel::addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
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

void VTSCDrivingStylePanel::addIntControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
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

void VTSCDrivingStylePanel::showAllDescriptions() {
  if (aggressiveness) aggressiveness->showDescription();
  if (fixedLeadTime) fixedLeadTime->showDescription();
  if (lowSpeedBiasMph) lowSpeedBiasMph->showDescription();
  if (lowSpeedBiasEndMph) lowSpeedBiasEndMph->showDescription();
  if (speedIncreaseFactor) speedIncreaseFactor->showDescription();
}

void VTSCDrivingStylePanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  showAllDescriptions();
}

