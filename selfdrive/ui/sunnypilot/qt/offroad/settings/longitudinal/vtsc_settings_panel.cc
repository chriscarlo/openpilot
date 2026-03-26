/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_settings_panel.h"
#include <QPushButton>
#include <algorithm>
#include <cmath>

namespace {

std::string vtscStrategyParamFromIndex(int index) {
  return index == 1 ? "strategic" : "advisory";
}

int vtscStrategyIndexFromParam(const std::string &value) {
  return value == "advisory" ? 0 : 1;
}

constexpr const char *kLowSpeedLearningEnabledParam = "VisionTurnSpeedControlLowSpeedLearningEnabled";

}  // namespace

// Local helper: create a section card with shared style
QFrame* VTSCSettingsPanel::createSectionFrame() {
  QFrame *frame = new QFrame();
  frame->setStyleSheet("QFrame { background-color: #292929; border-radius: 20px; padding: 25px; }");
  return frame;
}

VTSCSettingsPanel::VTSCSettingsPanel(QWidget *parent) : QFrame(parent) {
  setupUI();
}

void VTSCSettingsPanel::showEvent(QShowEvent *event) {
  QFrame::showEvent(event);
  // Refresh toggles from Params whenever panel is shown
  Params p;
  if (mapTog_) {
    bool on = p.getBool("MTSCLookaheadEnabled");
    if (mapTog_->on != on) mapTog_->togglePosition();
  }
  if (mapStrategyCarousel_) {
    const int idx = vtscStrategyIndexFromParam(p.get("VTSCMapStrategy"));
    if (mapStrategyCarousel_->currentIndex() != idx) {
      mapStrategyCarousel_->setCurrentIndex(idx, false);
    }
    mapStrategyCarousel_->setEnabled(p.getBool("MTSCLookaheadEnabled"));
  }
  if (bypassTog_) {
    bool on = p.getBool("VisionTurnSpeedControlOcclBypassWithLead");
    if (bypassTog_->on != on) bypassTog_->togglePosition();
  }
  if (dbgTog_) {
    bool on = p.getBool("VTSCVerboseDebug");
    if (dbgTog_->on != on) dbgTog_->togglePosition();
  }
  if (recorderTog_) {
    bool on = p.getBool("VTSCInterventionRecorderEnabled");
    if (recorderTog_->on != on) recorderTog_->togglePosition();
  }
  if (learnTog_) {
    bool on = p.getBool(kLowSpeedLearningEnabledParam);
    if (learnTog_->on != on) learnTog_->togglePosition();
  }
  auto refreshPhaseLabel = [&p](QLabel *valLabel, QLabel *statusLabel, const char *key) {
    if (!valLabel || !statusLabel) return;
    auto clampPhase = [](float x) { return std::max(-3.0f, std::min(3.0f, x)); };
    float v = 0.0f;
    QString s = QString::fromStdString(p.get(key));
    if (!s.isEmpty()) v = s.toFloat();
    v = clampPhase(v);
    valLabel->setText(QString::number(v, 'f', 2) + " s");
    bool isDefault = std::abs(v) < 0.001f;
    statusLabel->setText(isDefault ? tr("(Default)") : tr("(Modified)"));
    statusLabel->setStyleSheet(isDefault ? "font-size: 32px; color: #999999;" : "font-size: 32px; color: #FFC107;");
  };
  refreshPhaseLabel(curvePhaseValLabel_, curvePhaseStatusLabel_, "VisionTurnSpeedControlCurvePhaseOffsetS");
  refreshPhaseLabel(overshootPhaseValLabel_, overshootPhaseStatusLabel_, "VisionTurnSpeedControlOvershootPhaseOffsetS");
  refreshPhaseLabel(apexExitValLabel_, apexExitStatusLabel_, "VisionTurnSpeedControlApexExitPhaseOffsetS");
  if (headValLabel_ && headStatusLabel_) {
    auto clamp = [](float x){ return std::max(0.5f, std::min(5.0f, x)); };
    float v = 3.0f;
    QString s = QString::fromStdString(p.get("VisionTurnSpeedControlOcclBypassHeadwayS"));
    if (!s.isEmpty()) v = s.toFloat();
    v = clamp(v);
    headValLabel_->setText(QString::number(v, 'f', 2) + " s");
    bool isDefault = std::abs(v - 3.0f) < 0.001f;
    headStatusLabel_->setText(isDefault ? tr("(Default)") : tr("(Modified)"));
    headStatusLabel_->setStyleSheet(isDefault ? "font-size: 32px; color: #999999;" : "font-size: 32px; color: #FFC107;");
  }
}

void VTSCSettingsPanel::setupUI() {
  QVBoxLayout *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(50, 20, 50, 20);
  mainLayout->setSpacing(30);

  // Back button
  PanelBackButton *backBtn = new PanelBackButton(tr("Back"));
  connect(backBtn, &QPushButton::clicked, this, &VTSCSettingsPanel::backPress);
  mainLayout->addWidget(backBtn, 0, Qt::AlignLeft);

  mainLayout->addSpacing(20);

  // Title & description
  QLabel *title = new QLabel(tr("Vision Turn Speed Control"));
  title->setStyleSheet("font-size: 50px; font-weight: 600; color: #E4E4E4; padding-bottom: 10px;");
  title->setAlignment(Qt::AlignCenter);
  mainLayout->addWidget(title);

  QLabel *desc = new QLabel(tr("VTSC uses vision curvature — and optional map lookahead — to slow smoothly before curves."));
  desc->setStyleSheet("font-size: 34px; color: #999999; padding-bottom: 30px;");
  desc->setWordWrap(true);
  desc->setAlignment(Qt::AlignCenter);
  mainLayout->addWidget(desc);

  // Section: Map Data (MTSC)
  QFrame *mapFrame = createSectionFrame();
  QVBoxLayout *mapLayout = new QVBoxLayout(mapFrame);

  QLabel *mapTitle = new QLabel(tr("Map Data (MTSC)"));
  mapTitle->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  mapLayout->addWidget(mapTitle);

  // No category description per concise style

  // Row: Map Lookahead for VTSC
  QHBoxLayout *mapRow = new QHBoxLayout();
  QLabel *mapLbl = new QLabel(tr("Map Lookahead for VTSC"));
  mapLbl->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  mapRow->addWidget(mapLbl);
  mapRow->addStretch();
  mapTog_ = new ToggleSP();
  mapTog_->setFixedSize(150, 80);
  {
    Params p; bool on = p.getBool("MTSCLookaheadEnabled");
    if (mapTog_->on != on) mapTog_->togglePosition();
  }
  QObject::connect(mapTog_, &ToggleSP::stateChanged, [](bool s){ Params().putBool("MTSCLookaheadEnabled", s); });
  mapRow->addWidget(mapTog_);
  mapLayout->addLayout(mapRow);
  // Map lookahead helper (expanded)
  {
    QLabel *mapHelp = new QLabel(tr(
      "When ON, VTSC reads upcoming road curvature from offline map data to plan earlier, smoother slowing for turns. "
      "This only adds anticipation; it never raises speed above physics or your cruise setpoint.\n"
      ""
    ));
    mapHelp->setStyleSheet("font-size: 32px; color: #999999; padding-left: 10px; padding-bottom: 5px;");
    mapHelp->setWordWrap(true);
    mapLayout->addWidget(mapHelp);
  }

  QLabel *strategyTitle = new QLabel(tr("Map Planning Strategy"));
  strategyTitle->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4;");
  mapLayout->addWidget(strategyTitle);

  QLabel *strategyHelp = new QLabel(tr(
    "Advisory keeps the current map tail behavior. Strategic is experimental: map owns long-horizon timing, "
    "while vision earns the right to relax once the curve is clearly understood."
  ));
  strategyHelp->setStyleSheet("font-size: 32px; color: #999999; padding-left: 10px; padding-bottom: 10px;");
  strategyHelp->setWordWrap(true);
  mapLayout->addWidget(strategyHelp);

  mapStrategyCarousel_ = new HorizontalCarousel({tr("Advisory"), tr("Strategic")}, vtscStrategyIndexFromParam(Params().get("VTSCMapStrategy")), this);
  mapStrategyCarousel_->setEnabled(Params().getBool("MTSCLookaheadEnabled"));
  QObject::connect(mapStrategyCarousel_, &HorizontalCarousel::currentIndexChanged, [](int index) {
    Params().put("VTSCMapStrategy", vtscStrategyParamFromIndex(index));
  });
  QObject::connect(mapTog_, &ToggleSP::stateChanged, [this](bool enabled) {
    if (mapStrategyCarousel_) mapStrategyCarousel_->setEnabled(enabled);
  });
  mapLayout->addWidget(mapStrategyCarousel_);

  // Row: Rally co-pilot HUD curve preview
  QHBoxLayout *copilotRow = new QHBoxLayout();
  QLabel *copilotLbl = new QLabel(tr("Rally Co-Pilot Curve Preview (HUD)"));
  copilotLbl->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  copilotRow->addWidget(copilotLbl);
  copilotRow->addStretch();
  ToggleSP *copilotTog = new ToggleSP();
  copilotTog->setFixedSize(150, 80);
  {
    Params p; bool on = p.getBool("VTSCRallyCoPilotHUDEnabled");
    if (copilotTog->on != on) copilotTog->togglePosition();
  }
  QObject::connect(copilotTog, &ToggleSP::stateChanged, [](bool s){ Params().putBool("VTSCRallyCoPilotHUDEnabled", s); });
  copilotRow->addWidget(copilotTog);
  mapLayout->addLayout(copilotRow);
  {
    QLabel *copilotHelp = new QLabel(tr(
      "When ON, the onroad HUD shows a strip-map preview of the next curve (shape, distance, time-to-curve) "
      "using offline map data, plus the VTSC recommended speed. The overlay is available even when not engaged, "
      "and only appears for curves above a mild curvature threshold (ignores slight bends)."
    ));
    copilotHelp->setStyleSheet("font-size: 32px; color: #999999; padding-left: 10px; padding-bottom: 5px;");
    copilotHelp->setWordWrap(true);
    mapLayout->addWidget(copilotHelp);
  }
  mainLayout->addWidget(mapFrame);

  // Section: Timing Alignment
  QFrame *timingFrame = createSectionFrame();
  QVBoxLayout *timingLayout = new QVBoxLayout(timingFrame);

  QLabel *timingTitle = new QLabel(tr("Timing Alignment"));
  timingTitle->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  timingLayout->addWidget(timingTitle);

  const QString circleButtonStyle = R"(
    QPushButton { font-size: 60px; font-weight: 500; border-radius: 50px; background-color: #393939; color: #E4E4E4; }
    QPushButton:pressed { background-color: #4a4a4a; }
    QPushButton:disabled { background-color: #2a2a2a; color: #666666; }
  )";
  const QString resetButtonStyle = R"(
    QPushButton { font-size: 35px; font-weight: 500; border-radius: 20px; background-color: #393939; color: #E4E4E4; }
    QPushButton:pressed { background-color: #4a4a4a; }
    QPushButton:disabled { background-color: #2a2a2a; color: #666666; }
  )";

  auto addTimingControl = [&](const QString &title, const char *paramKey, const QString &help, QLabel *&valueLabel, QLabel *&statusLabel) {
    QLabel *controlTitle = new QLabel(title);
    controlTitle->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4;");
    timingLayout->addWidget(controlTitle);

    QHBoxLayout *row = new QHBoxLayout();

    QPushButton *minusBtn = new QPushButton("-");
    minusBtn->setFixedSize(100, 100);
    minusBtn->setStyleSheet(circleButtonStyle);
    minusBtn->setFocusPolicy(Qt::NoFocus);
    row->addWidget(minusBtn);

    QVBoxLayout *valueLayout = new QVBoxLayout();
    valueLabel = new QLabel("0.00 s");
    valueLabel->setAlignment(Qt::AlignCenter);
    valueLabel->setFixedWidth(300);
    valueLabel->setStyleSheet("font-size: 70px; font-weight: 500; color: #FFFFFF;");
    valueLayout->addWidget(valueLabel);
    statusLabel = new QLabel(tr("(Default)"));
    statusLabel->setAlignment(Qt::AlignCenter);
    statusLabel->setStyleSheet("font-size: 32px; color: #999999;");
    valueLayout->addWidget(statusLabel);
    row->addLayout(valueLayout);

    QPushButton *plusBtn = new QPushButton("+");
    plusBtn->setFixedSize(100, 100);
    plusBtn->setStyleSheet(circleButtonStyle);
    plusBtn->setFocusPolicy(Qt::NoFocus);
    row->addWidget(plusBtn);

    row->addStretch();

    QPushButton *resetBtn = new QPushButton(tr("Reset"));
    resetBtn->setFixedSize(150, 80);
    resetBtn->setStyleSheet(resetButtonStyle);
    resetBtn->setFocusPolicy(Qt::NoFocus);
    row->addWidget(resetBtn);

    auto clampPhase = [](float x) { return std::max(-3.0f, std::min(3.0f, x)); };
    auto readValue = [paramKey, clampPhase]() -> float {
      Params p;
      float v = 0.0f;
      QString s = QString::fromStdString(p.get(paramKey));
      if (!s.isEmpty()) v = s.toFloat();
      return clampPhase(v);
    };
    auto writeValue = [paramKey](float v) {
      Params().put(paramKey, QString::number(v, 'f', 2).toStdString());
    };
    auto updateLabels = [valueLabel, statusLabel, minusBtn, plusBtn, resetBtn](float v) {
      valueLabel->setText(QString::number(v, 'f', 2) + " s");
      bool isDefault = std::abs(v) < 0.001f;
      statusLabel->setText(isDefault ? QObject::tr("(Default)") : QObject::tr("(Modified)"));
      statusLabel->setStyleSheet(isDefault ? "font-size: 32px; color: #999999;" : "font-size: 32px; color: #FFC107;");
      minusBtn->setEnabled(v > -3.0f);
      plusBtn->setEnabled(v < 3.0f);
      resetBtn->setEnabled(!isDefault);
    };

    updateLabels(readValue());
    QObject::connect(minusBtn, &QPushButton::clicked, [readValue, writeValue, updateLabels, clampPhase]() {
      float v = clampPhase(readValue() - 0.10f);
      writeValue(v);
      updateLabels(v);
    });
    QObject::connect(plusBtn, &QPushButton::clicked, [readValue, writeValue, updateLabels, clampPhase]() {
      float v = clampPhase(readValue() + 0.10f);
      writeValue(v);
      updateLabels(v);
    });
    QObject::connect(resetBtn, &QPushButton::clicked, [writeValue, updateLabels]() {
      float v = 0.0f;
      writeValue(v);
      updateLabels(v);
    });

    timingLayout->addLayout(row);

    QLabel *helpLabel = new QLabel(help);
    helpLabel->setStyleSheet("font-size: 32px; color: #999999; padding-left: 10px; padding-bottom: 10px;");
    helpLabel->setWordWrap(true);
    timingLayout->addWidget(helpLabel);
  };

  addTimingControl(
    tr("Curve Entry Timing (seconds)"),
    "VisionTurnSpeedControlCurvePhaseOffsetS",
    tr("This setting adjusts how soon or how late VTSC starts interpreting upcoming curvature for turn entry. "
       "0 keeps default timing, lower values begin slowing sooner, and higher values begin slowing later."),
    curvePhaseValLabel_, curvePhaseStatusLabel_);

  addTimingControl(
    tr("Overshoot Braking Timing (seconds)"),
    "VisionTurnSpeedControlOvershootPhaseOffsetS",
    tr("This setting adjusts how soon or how late stronger braking is requested when a tighter section ahead needs extra slowdown. "
       "0 keeps default timing, lower values start braking earlier, and higher values start braking later."),
    overshootPhaseValLabel_, overshootPhaseStatusLabel_);

  addTimingControl(
    tr("Apex Exit Timing (seconds)"),
    "VisionTurnSpeedControlApexExitPhaseOffsetS",
    tr("This setting adjusts how soon or how late the car begins to accelerate when an apex is detected. "
       "0 is the default at-apex behavior, lower values move this target ahead of the apex, and larger values move it to after the apex."),
    apexExitValLabel_, apexExitStatusLabel_);

  mainLayout->addWidget(timingFrame);

  // Section: Low-Speed Learning
  QFrame *learningFrame = createSectionFrame();
  QVBoxLayout *learningLayout = new QVBoxLayout(learningFrame);

  QLabel *learningTitle = new QLabel(tr("Low-Speed Learning"));
  learningTitle->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  learningLayout->addWidget(learningTitle);

  QHBoxLayout *learnToggleRow = new QHBoxLayout();
  QLabel *learnToggleLbl = new QLabel(tr("Persistent Learned State"));
  learnToggleLbl->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  learnToggleRow->addWidget(learnToggleLbl);
  learnToggleRow->addStretch();
  learnTog_ = new ToggleSP();
  learnTog_->setFixedSize(150, 80);
  {
    Params p;
    bool on = p.getBool(kLowSpeedLearningEnabledParam);
    if (learnTog_->on != on) learnTog_->togglePosition();
  }
  learnToggleRow->addWidget(learnTog_);
  learningLayout->addLayout(learnToggleRow);

  QLabel *learnToggleHelp = new QLabel(tr(
    "When ON, VTSC uses and updates saved learned adjustments across drives. "
    "Driver accel overrides are learned by curve shape, and autonomous saturation or steering-margin learning stays internal. "
    "When OFF, VTSC ignores the saved learned state and leaves the base curve profile unchanged."
  ));
  learnToggleHelp->setStyleSheet("font-size: 32px; color: #999999; padding-left: 10px; padding-bottom: 10px;");
  learnToggleHelp->setWordWrap(true);
  learningLayout->addWidget(learnToggleHelp);

  QObject::connect(learnTog_, &ToggleSP::stateChanged, [](bool s) {
    Params().putBool(kLowSpeedLearningEnabledParam, s);
  });

  mainLayout->addWidget(learningFrame);

  // Section: Lead Vehicle Bypass
  QFrame *leadFrame = createSectionFrame();
  QVBoxLayout *visLayout = new QVBoxLayout(leadFrame);

  QLabel *visTitle = new QLabel(tr("Lead Vehicle Bypass"));
  visTitle->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  visLayout->addWidget(visTitle);

  // No category description per concise style

  // Row: Occlusion Bypass When Following Lead
  QHBoxLayout *bypassRow = new QHBoxLayout();
  QLabel *bypassLbl = new QLabel(tr("Occlusion Bypass When Following Lead"));
  bypassLbl->setStyleSheet("font-size: 36px; color: #E4E4E4;");
  bypassRow->addWidget(bypassLbl);
  bypassRow->addStretch();
  bypassTog_ = new ToggleSP();
  bypassTog_->setFixedSize(150, 80);
  {
    Params p; bool on = p.getBool("VisionTurnSpeedControlOcclBypassWithLead");
    if (bypassTog_->on != on) bypassTog_->togglePosition();
  }
  QObject::connect(bypassTog_, &ToggleSP::stateChanged, [](bool s){ Params().putBool("VisionTurnSpeedControlOcclBypassWithLead", s); });
  bypassRow->addWidget(bypassTog_);
  visLayout->addLayout(bypassRow);
  // Bypass helper (expanded, plain language)
  {
    QLabel *bypassHelp = new QLabel(tr(
      "When ON, VTSC relaxes its occlusion safeguards while you are following a tracked lead within the threshold below. "
      "In practice this avoids extra slowing (\"crawl\") behind a lead when the camera view is momentarily blocked.\n"
      ""
    ));
    bypassHelp->setStyleSheet("font-size: 32px; color: #999999; padding-left: 10px; padding-bottom: 5px;");
    bypassHelp->setWordWrap(true);
    visLayout->addWidget(bypassHelp);
  }

  // Headway control title (placed above control per pattern: Title -> Control -> Description)
  QLabel *headTitle = new QLabel(tr("Lead Bypass Headway (seconds)"));
  headTitle->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4;");
  visLayout->addWidget(headTitle);
  QHBoxLayout *headRow = new QHBoxLayout();

  // Minus
  QPushButton *hMinus = new QPushButton("-");
  hMinus->setFixedSize(100, 100);
  hMinus->setStyleSheet(R"(
    QPushButton { font-size: 60px; font-weight: 500; border-radius: 50px; background-color: #393939; color: #E4E4E4; }
    QPushButton:pressed { background-color: #4a4a4a; }
    QPushButton:disabled { background-color: #2a2a2a; color: #666666; }
  )");
  hMinus->setFocusPolicy(Qt::NoFocus);
  headRow->addWidget(hMinus);

  // Value + status
  QVBoxLayout *hValLay = new QVBoxLayout();
  headValLabel_ = new QLabel("3.00 s");
  headValLabel_->setAlignment(Qt::AlignCenter);
  headValLabel_->setFixedWidth(300);
  headValLabel_->setStyleSheet("font-size: 70px; font-weight: 500; color: #FFFFFF;");
  hValLay->addWidget(headValLabel_);
  headStatusLabel_ = new QLabel(tr("(Default)"));
  headStatusLabel_->setAlignment(Qt::AlignCenter);
  headStatusLabel_->setStyleSheet("font-size: 32px; color: #999999;");
  hValLay->addWidget(headStatusLabel_);
  headRow->addLayout(hValLay);

  // Plus
  QPushButton *hPlus = new QPushButton("+");
  hPlus->setFixedSize(100, 100);
  hPlus->setStyleSheet(hMinus->styleSheet());
  hPlus->setFocusPolicy(Qt::NoFocus);
  headRow->addWidget(hPlus);

  headRow->addStretch();

  // Reset
  QPushButton *hReset = new QPushButton(tr("Reset"));
  hReset->setFixedSize(150, 80);
  hReset->setStyleSheet(R"(
    QPushButton { font-size: 35px; font-weight: 500; border-radius: 20px; background-color: #393939; color: #E4E4E4; }
    QPushButton:pressed { background-color: #4a4a4a; }
    QPushButton:disabled { background-color: #2a2a2a; color: #666666; }
  )");
  hReset->setFocusPolicy(Qt::NoFocus);
  headRow->addWidget(hReset);

  // Load, bind, and wire events
  auto clamp = [](float x){ return std::max(0.5f, std::min(5.0f, x)); };
  auto updateLabels = [this, hMinus, hPlus, hReset](float v){
    headValLabel_->setText(QString::number(v, 'f', 2) + " s");
    bool isDefault = std::abs(v - 3.0f) < 0.001f;
    headStatusLabel_->setText(isDefault ? QObject::tr("(Default)") : QObject::tr("(Modified)"));
    headStatusLabel_->setStyleSheet(isDefault ? "font-size: 32px; color: #999999;" : "font-size: 32px; color: #FFC107;");
    hMinus->setEnabled(v > 0.5f);
    hPlus->setEnabled(v < 5.0f);
    hReset->setEnabled(!isDefault);
  };
  auto readHeadVal = [clamp]() -> float {
    Params p; QString s = QString::fromStdString(p.get("VisionTurnSpeedControlOcclBypassHeadwayS"));
    float v = 3.0f;
    if (!s.isEmpty()) v = s.toFloat();
    return clamp(v);
  };
  updateLabels(readHeadVal());
  QObject::connect(hMinus, &QPushButton::clicked, [updateLabels, clamp, readHeadVal]() {
    float v = clamp(readHeadVal() - 0.10f);
    Params().put("VisionTurnSpeedControlOcclBypassHeadwayS", QString::number(v, 'f', 2).toStdString());
    updateLabels(v);
  });
  QObject::connect(hPlus, &QPushButton::clicked, [updateLabels, clamp, readHeadVal]() {
    float v = clamp(readHeadVal() + 0.10f);
    Params().put("VisionTurnSpeedControlOcclBypassHeadwayS", QString::number(v, 'f', 2).toStdString());
    updateLabels(v);
  });
  QObject::connect(hReset, &QPushButton::clicked, [updateLabels]() {
    float v = 3.0f;
    Params().put("VisionTurnSpeedControlOcclBypassHeadwayS", QString::number(v, 'f', 2).toStdString());
    updateLabels(v);
  });

  visLayout->addLayout(headRow);
  // Description: placed below control
  {
    QLabel *headExplain = new QLabel(tr(
      "Time headway is distance to the lead divided by your speed.\n"
      "Bypass activates when time headway ≤ the value above (for example, 40 m at 20 m/s ≈ 2.0 s)."
    ));
    headExplain->setStyleSheet("font-size: 32px; color: #999999; padding-left: 10px; padding-bottom: 10px;");
    headExplain->setWordWrap(true);
    visLayout->addWidget(headExplain);
  }
  mainLayout->addWidget(leadFrame);

  // Section: Developer Options
  QFrame *devFrame = createSectionFrame();
  QVBoxLayout *devLayout = new QVBoxLayout(devFrame);

  QLabel *devTitle = new QLabel(tr("Developer Options"));
  devTitle->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  devLayout->addWidget(devTitle);

  // No category description per concise style

  // Row: Verbose debug logging
  {
    QHBoxLayout *dbgRow = new QHBoxLayout();
    QLabel *dbgLbl = new QLabel(tr("Verbose VTSC Debug Logging"));
    dbgLbl->setStyleSheet("font-size: 36px; color: #E4E4E4;");
    dbgRow->addWidget(dbgLbl);
    dbgRow->addStretch();
    dbgTog_ = new ToggleSP();
    dbgTog_->setFixedSize(150, 80);
    {
      Params p; bool on = p.getBool("VTSCVerboseDebug");
      if (dbgTog_->on != on) dbgTog_->togglePosition();
    }
    QObject::connect(dbgTog_, &ToggleSP::stateChanged, [](bool s){ Params().putBool("VTSCVerboseDebug", s); });
    dbgRow->addWidget(dbgTog_);
    devLayout->addLayout(dbgRow);
  }

  // Row: Write onroad VTSC snapshots to file (JSONL)
  {
    QHBoxLayout *fileRow = new QHBoxLayout();
    QLabel *fileLbl = new QLabel(tr("Write Onroad VTSC Snapshots (JSONL)"));
    fileLbl->setStyleSheet("font-size: 36px; color: #E4E4E4;");
    fileRow->addWidget(fileLbl);
    fileRow->addStretch();
    ToggleSP *fileTog = new ToggleSP();
    fileTog->setFixedSize(150, 80);
    {
      Params p; bool on = p.getBool("VTSCWriteSnapshotFile");
      if (fileTog->on != on) fileTog->togglePosition();
    }
    QObject::connect(fileTog, &ToggleSP::stateChanged, [](bool s){ Params().putBool("VTSCWriteSnapshotFile", s); });
    fileRow->addWidget(fileTog);
    devLayout->addLayout(fileRow);

    QLabel *fileHelp = new QLabel(tr(
      "When ON, VTSC writes compact JSON lines with key decisions to /data/media/0/VTSCDebug/vtsc_snapshots.jsonl.\n"
      "Keep this OFF unless debugging real-world discrepancies; file rotates automatically (small size)."
    ));
    fileHelp->setStyleSheet("font-size: 32px; color: #999999; padding-left: 10px; padding-bottom: 5px;");
    fileHelp->setWordWrap(true);
    devLayout->addWidget(fileHelp);
  }

  // Row: VTSC intervention recorder (driver gas/brake interventions while VTSC is limiting)
  {
    QHBoxLayout *recRow = new QHBoxLayout();
    QLabel *recLbl = new QLabel(tr("Record VTSC Driver Interventions"));
    recLbl->setStyleSheet("font-size: 36px; color: #E4E4E4;");
    recRow->addWidget(recLbl);
    recRow->addStretch();
    recorderTog_ = new ToggleSP();
    recorderTog_->setFixedSize(150, 80);
    {
      Params p; bool on = p.getBool("VTSCInterventionRecorderEnabled");
      if (recorderTog_->on != on) recorderTog_->togglePosition();
    }
    QObject::connect(recorderTog_, &ToggleSP::stateChanged, [](bool s){ Params().putBool("VTSCInterventionRecorderEnabled", s); });
    recRow->addWidget(recorderTog_);
    devLayout->addLayout(recRow);

    QLabel *recHelp = new QLabel(tr(
      "When ON, an onroad background recorder captures gas/brake interventions while VTSC is the limiting source.\n"
      "Bundles are written to /data/media/0/VTSCTuner/events and auto-pruned by total size."
    ));
    recHelp->setStyleSheet("font-size: 32px; color: #999999; padding-left: 10px; padding-bottom: 5px;");
    recHelp->setWordWrap(true);
    devLayout->addWidget(recHelp);
  }
  mainLayout->addWidget(devFrame);

  mainLayout->addStretch();
}
