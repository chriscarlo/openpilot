/**
 * VTSC Curve Physics Panel (Advanced)
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_physics_panel.h"

VTSCPhysicsPanel::VTSCPhysicsPanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  list_ = new ListWidgetSP(this, false);
  ScrollViewSP *scroll = new ScrollViewSP(list_, this);
  main_layout->addWidget(scroll);

  PanelBackButton *back_btn = new PanelBackButton();
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCPhysicsPanel::backPress);
  list_->addItem(back_btn);

  QPushButton *reset_btn = new QPushButton(tr("Reset to Defaults"));
  reset_btn->setStyleSheet(R"(
    QPushButton { border-radius: 20px; font-size: 45px; font-weight: 500; height: 120px; margin: 20px 40px; color: #FFFFFF; background-color: #393939; }
    QPushButton:pressed { background-color: #4a4a4a; }
  )");
  list_->addItem(reset_btn);

  QObject::connect(reset_btn, &QPushButton::clicked, [=]() {
    params.put("VisionTurnSpeedControlPhysicsBaseline", "4.107103");
    params.put("VisionTurnSpeedControlPhysicsAmplitude", "-1.658965");
    params.put("VisionTurnSpeedControlPhysicsMinLatAccel", "2.4481");
    params.put("VisionTurnSpeedControlPhysicsMaxLatAccel", "4.1071");
    params.put("VisionTurnSpeedControlPhysicsCenter", "0.005397");
    params.put("VisionTurnSpeedControlPhysicsSteepness", "-1395.055546");
    emit physBaseline->updateLabels();
    emit physAmplitude->updateLabels();
    emit physMinLat->updateLabels();
    emit physMaxLat->updateLabels();
  });

  // Initialize defaults if unset
  auto ensure = [&](const char *k, const char *v){ if (QString::fromStdString(params.get(k)).isEmpty()) params.put(k, v); };
  ensure("VisionTurnSpeedControlPhysicsBaseline", "4.107103");
  ensure("VisionTurnSpeedControlPhysicsAmplitude", "-1.658965");
  ensure("VisionTurnSpeedControlPhysicsMinLatAccel", "2.4481");
  ensure("VisionTurnSpeedControlPhysicsMaxLatAccel", "4.1071");
  ensure("VisionTurnSpeedControlPhysicsCenter", "0.005397");
  ensure("VisionTurnSpeedControlPhysicsSteepness", "-1395.055546");

  addFloatControl(physBaseline, "VisionTurnSpeedControlPhysicsBaseline",
                  tr("Baseline Lat Accel"), tr("Baseline lateral accel on easy curves."),
                  2.0f, 6.5f, 0.05f, tr("m/s²"));
  addFloatControl(physAmplitude, "VisionTurnSpeedControlPhysicsAmplitude",
                  tr("Amplitude (neg)"), tr("Depth of reduction as curvature tightens (negative)."),
                  -5.00f, -0.20f, 0.05f, tr("m/s²"));
  addFloatControl(physMinLat, "VisionTurnSpeedControlPhysicsMinLatAccel",
                  tr("Min Lat Accel"), tr("Floor clamp for very tight curves."),
                  1.0f, 3.0f, 0.05f, tr("m/s²"));
  addFloatControl(physMaxLat, "VisionTurnSpeedControlPhysicsMaxLatAccel",
                  tr("Max Lat Accel"), tr("Ceiling clamp for straight/easy curves."),
                  2.0f, 5.5f, 0.05f, tr("m/s²"));

  // Sigmoid internals moved to Physics Internals panel
}

void VTSCPhysicsPanel::addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
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

void VTSCPhysicsPanel::showAllDescriptions() {
  if (physBaseline) physBaseline->showDescription();
  if (physAmplitude) physAmplitude->showDescription();
  if (physMinLat) physMinLat->showDescription();
  if (physMaxLat) physMaxLat->showDescription();
}

void VTSCPhysicsPanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  showAllDescriptions();
}
