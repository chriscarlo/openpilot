/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_settings_panel.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"
#include <QVBoxLayout>

VTSCIconButton::VTSCIconButton(const QString &icon_path, const QString &text, QWidget *parent)
  : QPushButton(parent) {
  // Compact tile size to reduce perceived gaps
  setFixedSize(160, 160);
  setObjectName("vtsc_icon_button");
  
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->setAlignment(Qt::AlignCenter);
  layout->setSpacing(4);
  layout->setContentsMargins(0, 0, 0, 0);
  
  // Icon label (using emoji for now)
  QLabel *icon_label = new QLabel(this);
  icon_label->setText(icon_path);  // This will be an emoji
  icon_label->setAlignment(Qt::AlignCenter);
  icon_label->setStyleSheet("font-size: 48px;");
  
  // Text label
  QLabel *text_label = new QLabel(text, this);
  text_label->setAlignment(Qt::AlignCenter);
  text_label->setWordWrap(true);
  text_label->setStyleSheet("font-size: 20px; color: #FFFFFF;");
  
  layout->addWidget(icon_label);
  layout->addWidget(text_label);
  
  setStyleSheet(R"(
    QPushButton#vtsc_icon_button {
      background-color: #393939;
      border-radius: 20px;
      border: 3px solid #696969;
    }
    QPushButton#vtsc_icon_button:pressed {
      background-color: #4a4a4a;
      border-color: #FFFFFF;
    }
    QPushButton#vtsc_icon_button:disabled {
      background-color: #2d2d2d;
      border-color: #444444;
    }
  )");
  
  connect(this, &QPushButton::clicked, this, &VTSCIconButton::buttonClicked);
}

VTSCSettingsPanel::VTSCSettingsPanel(QWidget *parent) : QFrame(parent) {
  setStyleSheet(R"(
    #back_btn {
      font-size: 50px;
      margin: 0px;
      padding: 15px;
      border-width: 0;
      border-radius: 30px;
      color: #dddddd;
      background-color: #393939;
    }
    #back_btn:pressed {
      background-color: #4a4a4a;
    }
  )");
  
  main_layout = new QStackedLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);
  setupUI();
}

void VTSCSettingsPanel::setupUI() {
  // Offroad panels use ScrollViewSP + ListWidgetSP; mirror that to avoid panel-sized gaps
  auto *list = new ListWidgetSP(this, false);
  auto *scroll = new ScrollViewSP(list, this);
  main_layout->addWidget(scroll);

  // Header with back button and title
  QWidget *header = new QWidget(this);
  QHBoxLayout *header_layout = new QHBoxLayout(header);
  header_layout->setContentsMargins(16, 8, 16, 8);
  header_layout->setSpacing(12);

  back_btn = new QPushButton("◀", header);
  back_btn->setObjectName("back_btn");
  back_btn->setFixedSize(90, 90);
  connect(back_btn, &QPushButton::clicked, this, &VTSCSettingsPanel::backPress);

  QLabel *title = new QLabel(tr("Vision Turn Speed Control Settings"), header);
  title->setStyleSheet("font-size: 48px; font-weight: 600; color: #FFFFFF;");
  title->setAlignment(Qt::AlignCenter);
  header_layout->addWidget(back_btn);
  header_layout->addWidget(title, 1, Qt::AlignCenter);

  list->addItem(header);

  // Icon grid directly under header
  QWidget *grid_container = createIconGrid();
  list->addItem(grid_container);
}

QWidget* VTSCSettingsPanel::createIconGrid() {
  QWidget *grid_container = new QWidget();
  QGridLayout *grid_layout = new QGridLayout(grid_container);
  grid_container->setContentsMargins(0, 0, 0, 0);
  grid_layout->setContentsMargins(0, 0, 0, 0);
  grid_layout->setHorizontalSpacing(12);
  grid_layout->setVerticalSpacing(6);
  
  // Create icon buttons
  struct IconInfo {
    QString emoji;
    QString text;
    bool enabled;
  };
  
  std::vector<IconInfo> icons = {
    {"🧭", tr("Driving\nStyle"), true},             // 0
    {"🛣️", tr("Anticipation\n& Overshoot"), true},   // 1
    {"🔎", tr("Curve\nDetection"), true},           // 2
    {"🧪", tr("Adaptive\nFiltering"), true},        // 3
    {"🎚️", tr("Smoothing\nLimits"), true},          // 4
    {"🏁", tr("Apex &\nExit Boost"), true},         // 5
    {"👁️", tr("Vision\nOcclusion"), true},         // 6
    {"🚦", tr("Limits"), true},                      // 7
    {"📐", tr("Curve\nPhysics"), true},             // 8 (Advanced)
    {"⚙️", tr("Physics\nInternals"), true},        // 9 (Advanced)
  };
  
  int row = 0, col = 0;
  const int columns = 5;  // target 5-wide grid for tighter layout
  for (size_t i = 0; i < icons.size(); ++i) {
    VTSCIconButton *btn = new VTSCIconButton(icons[i].emoji, icons[i].text, grid_container);
    btn->setEnabled(icons[i].enabled);
    
    switch (i) {
      case 0: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::drivingStyleClicked); break;
      case 1: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::anticipationSettingsClicked); break;
      case 2: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::curveDetectionClicked); break;
      case 3: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::adaptiveFilteringClicked); break;
      case 4: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::smoothingLimitsClicked); break;
      case 5: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::apexBoostClicked); break;
      case 6: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::visionOcclusionClicked); break;
      case 7: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::limitsClicked); break;
      case 8: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::physicsClicked); break;
      case 9: connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::physicsInternalsClicked); break;
    }
    
    grid_layout->addWidget(btn, row, col, Qt::AlignTop);
    col++;
    if (col >= columns) { col = 0; row++; }
  }
  
  return grid_container;
}
