/**
 * VTSC Vision Occlusion Panel
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_vision_occlusion_panel.h"

VTSCVisionOcclusionPanel::VTSCVisionOcclusionPanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  list_ = new ListWidgetSP(this, false);
  ScrollViewSP *scroll = new ScrollViewSP(list_, this);
  main_layout->addWidget(scroll);

  PanelBackButton *back_btn = new PanelBackButton();
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCVisionOcclusionPanel::backPress);
  list_->addItem(back_btn);

  QPushButton *reset_btn = new QPushButton(tr("Reset to Defaults"));
  reset_btn->setStyleSheet(R"(
    QPushButton { border-radius: 20px; font-size: 45px; font-weight: 500; height: 120px; margin: 20px 40px; color: #FFFFFF; background-color: #393939; }
    QPushButton:pressed { background-color: #4a4a4a; }
  )");
  list_->addItem(reset_btn);

  QObject::connect(reset_btn, &QPushButton::clicked, [=]() {
    params.put("VisionTurnSpeedControlVisionConfAlpha", "0.28");
    params.put("VisionTurnSpeedControlVisionConfGoodThreshold", "0.70");
    params.put("VisionTurnSpeedControlVisionConfBadThreshold", "0.65");
    emit confAlpha->updateLabels();
    emit confGood->updateLabels();
    emit confBad->updateLabels();
  });

  // Initialize defaults if unset to avoid empty labels and odd first steps
  if (QString::fromStdString(params.get("VisionTurnSpeedControlVisionConfAlpha")).isEmpty()) params.put("VisionTurnSpeedControlVisionConfAlpha", "0.28");
  if (QString::fromStdString(params.get("VisionTurnSpeedControlVisionConfGoodThreshold")).isEmpty()) params.put("VisionTurnSpeedControlVisionConfGoodThreshold", "0.70");
  if (QString::fromStdString(params.get("VisionTurnSpeedControlVisionConfBadThreshold")).isEmpty()) params.put("VisionTurnSpeedControlVisionConfBadThreshold", "0.65");

  addFloatControl(confAlpha, "VisionTurnSpeedControlVisionConfAlpha",
                  tr("Confidence EMA Alpha"), tr("How quickly VTSC trusts confidence changes. Higher exits occlusion faster but can chatter; lower holds longer and is more stable."),
                  0.01f, 0.90f, 0.01f);
  addFloatControl(confGood, "VisionTurnSpeedControlVisionConfGoodThreshold",
                  tr("Good Threshold"), tr("Confidence to exit occlusion and resume updates. Lower exits sooner; too low can accept noisy vision."),
                  0.50f, 0.99f, 0.01f);
  addFloatControl(confBad, "VisionTurnSpeedControlVisionConfBadThreshold",
                  tr("Bad Threshold"), tr("Confidence to enter occlusion and hold curvature. Higher enters earlier; too high can freeze too often."),
                  0.10f, 0.90f, 0.01f);
}

void VTSCVisionOcclusionPanel::addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
                                               float min, float max, float step) {
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
    ptr->setLabel(val);
  });
  const auto initial = QString::fromStdString(params.get(param));
  ptr->setLabel(initial);
  list_->addItem(container);
}

void VTSCVisionOcclusionPanel::showAllDescriptions() {
  if (confAlpha) confAlpha->showDescription();
  if (confGood) confGood->showDescription();
  if (confBad) confBad->showDescription();
}

void VTSCVisionOcclusionPanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  showAllDescriptions();
}
