/**
 * VTSC Curve Detection Panel
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_curve_detection_panel.h"

VTSCCurveDetectionPanel::VTSCCurveDetectionPanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  list_ = new ListWidgetSP(this, false);
  ScrollViewSP *scroll = new ScrollViewSP(list_, this);
  main_layout->addWidget(scroll);

  PanelBackButton *back_btn = new PanelBackButton();
  QObject::connect(back_btn, &QPushButton::clicked, this, &VTSCCurveDetectionPanel::backPress);
  list_->addItem(back_btn);

  QPushButton *reset_btn = new QPushButton(tr("Reset to Defaults"));
  reset_btn->setStyleSheet(R"(
    QPushButton { border-radius: 20px; font-size: 45px; font-weight: 500; height: 120px; margin: 20px 40px; color: #FFFFFF; background-color: #393939; }
    QPushButton:pressed { background-color: #4a4a4a; }
  )");
  list_->addItem(reset_btn);

  QObject::connect(reset_btn, &QPushButton::clicked, [=]() {
    params.put("VisionTurnSpeedControlCurvatureEMAFactor", "0.30");
    params.put("VisionTurnSpeedControlApexThreshold", "0.00005");
    params.put("VisionTurnSpeedControlApexProminence", "0.0001");
    params.put("VisionTurnSpeedControlApexHysteresisTime", "2.0");
    params.put("VisionTurnSpeedControlApexMetersPerIndex", "2.0");
    params.put("VisionTurnSpeedControlApexNearIndex", "3");
    emit curvEMA->updateLabels();
    emit apexHysteresisTime->updateLabels();
    emit apexMetersPerIndex->updateLabels();
    emit apexNearIndex->updateLabels();
  });

  addFloatControl(curvEMA, "VisionTurnSpeedControlCurvatureEMAFactor",
                  tr("Curvature EMA"), tr("Smoothing ratio for predicted curvature (0=smooth, 1=fast)."),
                  0.10f, 0.50f, 0.05f);

  // Advanced small thresholds via edit dialogs for precision
  apexThresholdEdit = new ButtonControlSP(tr("Apex Threshold"), tr("Edit"),
                                          tr("Minimum curvature to consider as apex (e.g., 5e-5)."), this, true);
  QObject::connect(apexThresholdEdit, &ButtonControlSP::clicked, [=]() {
    QString cur = QString::fromStdString(params.get("VisionTurnSpeedControlApexThreshold"));
    QString val = InputDialog::getText(tr("Apex Threshold"), nullptr, tr("Enter a small float (e.g., 5e-5)"), false, -1, cur.isEmpty() ? "5e-5" : cur);
    if (!val.isEmpty()) params.put("VisionTurnSpeedControlApexThreshold", val.toStdString());
  });
  list_->addItem(apexThresholdEdit);

  apexProminenceEdit = new ButtonControlSP(tr("Apex Prominence"), tr("Edit"),
                                           tr("Minimum peak prominence (e.g., 1e-4)."), this, true);
  QObject::connect(apexProminenceEdit, &ButtonControlSP::clicked, [=]() {
    QString cur = QString::fromStdString(params.get("VisionTurnSpeedControlApexProminence"));
    QString val = InputDialog::getText(tr("Apex Prominence"), nullptr, tr("Enter a small float (e.g., 1e-4)"), false, -1, cur.isEmpty() ? "1e-4" : cur);
    if (!val.isEmpty()) params.put("VisionTurnSpeedControlApexProminence", val.toStdString());
  });
  list_->addItem(apexProminenceEdit);

  addFloatControl(apexHysteresisTime, "VisionTurnSpeedControlApexHysteresisTime",
                  tr("Apex Cooldown"), tr("Don’t re-detect the same apex too soon."),
                  0.1f, 10.0f, 0.10f, tr("s"), true);

  addFloatControl(apexMetersPerIndex, "VisionTurnSpeedControlApexMetersPerIndex",
                  tr("Meters per Index"), tr("Approximate spacing between trajectory points."),
                  0.5f, 5.0f, 0.10f, tr("m"), true);

  addIntControl(apexNearIndex, "VisionTurnSpeedControlApexNearIndex",
                tr("Apex Near Index"), tr("Indices window to detect close/past apex."),
                1, 10, 1, tr("idx"), true);
}

void VTSCCurveDetectionPanel::addFloatControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
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

void VTSCCurveDetectionPanel::addIntControl(OptionControlSP *&ptr, const char *param, const QString &title, const QString &desc,
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

void VTSCCurveDetectionPanel::showAllDescriptions() {
  if (curvEMA) curvEMA->showDescription();
  if (apexHysteresisTime) apexHysteresisTime->showDescription();
  if (apexMetersPerIndex) apexMetersPerIndex->showDescription();
  if (apexNearIndex) apexNearIndex->showDescription();
}

void VTSCCurveDetectionPanel::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  showAllDescriptions();
}

