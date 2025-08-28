/**
 * VTSC Anticipation & Overshoot Panel (BSG-aligned)
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_anticipation_panel.h"

VTSCAnticipationPanel::VTSCAnticipationPanel(QWidget *parent) : QWidget(parent) {
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
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCAnticipationPanel::backPress);
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
    params.put("VisionTurnSpeedControlPlanningDecelLimit", "3.50");
    params.put("VisionTurnSpeedControlOvershootSafetyMargin", "1.20");
    params.put("VisionTurnSpeedControlOvershootMinDistance", "10");
    params.put("VisionTurnSpeedControlAnticipationTargetReduction", "0.95");

    emit planningDecelLimit->updateLabels();
    emit overshootSafetyMargin->updateLabels();
    emit overshootMinDistance->updateLabels();
    emit anticipationTargetReduction->updateLabels();
  });

  // Controls
  addFloatControl(
    planningDecelLimit,
    "VisionTurnSpeedControlPlanningDecelLimit",
    tr("Planning Deceleration Limit"),
    tr("Assumed decel used to decide when to start slowing. Higher starts later."),
    1.0f, 7.0f, 0.10f, tr("m/s²")
  );

  addFloatControl(
    overshootSafetyMargin,
    "VisionTurnSpeedControlOvershootSafetyMargin",
    tr("Overshoot Safety Margin"),
    tr("Extra distance margin to ensure target speed is reached before the curve."),
    1.00f, 1.50f, 0.05f, tr("×")
  );

  addIntControl(
    overshootMinDistance,
    "VisionTurnSpeedControlOvershootMinDistance",
    tr("Overshoot Minimum Distance"),
    tr("Minimum distance considered for anticipatory deceleration."),
    5, 200, 5, tr("m")
  );

  addFloatControl(
    anticipationTargetReduction,
    "VisionTurnSpeedControlAnticipationTargetReduction",
    tr("Anticipation Target Reduction"),
    tr("Slightly reduce target speed during decel to hit target before apex."),
    0.90f, 1.00f, 0.01f, tr("×")
  );
}

void VTSCAnticipationPanel::addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
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

void VTSCAnticipationPanel::addIntControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
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

void VTSCAnticipationPanel::showAllDescriptions() {
  if (planningDecelLimit) planningDecelLimit->showDescription();
  if (overshootSafetyMargin) overshootSafetyMargin->showDescription();
  if (overshootMinDistance) overshootMinDistance->showDescription();
  if (anticipationTargetReduction) anticipationTargetReduction->showDescription();
}

void VTSCAnticipationPanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  showAllDescriptions();
}
