/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_settings_panel.h"
#include <QPushButton>

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
  if (bypassTog_) {
    bool on = p.getBool("VisionTurnSpeedControlOcclBypassWithLead");
    if (bypassTog_->on != on) bypassTog_->togglePosition();
  }
  if (dbgTog_) {
    bool on = p.getBool("VTSCVerboseDebug");
    if (dbgTog_->on != on) dbgTog_->togglePosition();
  }
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
  mainLayout->addWidget(mapFrame);

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
  mainLayout->addWidget(devFrame);

  mainLayout->addStretch();
}
