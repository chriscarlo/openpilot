/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_settings_panel.h"
#include <QVBoxLayout>

VTSCIconButton::VTSCIconButton(const QString &icon_path, const QString &text, QWidget *parent)
  : QPushButton(parent) {
  
  setFixedSize(200, 200);
  setObjectName("vtsc_icon_button");
  
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->setAlignment(Qt::AlignCenter);
  layout->setSpacing(10);
  
  // Icon label (using emoji for now)
  QLabel *icon_label = new QLabel(this);
  icon_label->setText(icon_path);  // This will be an emoji
  icon_label->setAlignment(Qt::AlignCenter);
  icon_label->setStyleSheet("font-size: 60px;");
  
  // Text label
  QLabel *text_label = new QLabel(text, this);
  text_label->setAlignment(Qt::AlignCenter);
  text_label->setWordWrap(true);
  text_label->setStyleSheet("font-size: 24px; color: #FFFFFF;");
  
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
  setupUI();
}

void VTSCSettingsPanel::setupUI() {
  icon_grid_screen = new QWidget(this);
  QVBoxLayout *main_vlayout = new QVBoxLayout(icon_grid_screen);
  main_vlayout->setContentsMargins(50, 30, 50, 30);
  
  // Header with back button and title
  QHBoxLayout *header_layout = new QHBoxLayout();
  
  back_btn = new QPushButton("◀", this);
  back_btn->setObjectName("back_btn");
  back_btn->setFixedSize(90, 90);
  connect(back_btn, &QPushButton::clicked, this, &VTSCSettingsPanel::backPress);
  
  QLabel *title = new QLabel(tr("Vision Turn Speed Control Settings"));
  title->setStyleSheet("font-size: 48px; font-weight: 600; color: #FFFFFF;");
  title->setAlignment(Qt::AlignCenter);
  
  header_layout->addWidget(back_btn);
  header_layout->addWidget(title, 1, Qt::AlignCenter);
  header_layout->addSpacing(90); // Balance for back button
  
  main_vlayout->addLayout(header_layout);
  main_vlayout->addSpacing(30);
  
  // Create centered layout for the grid
  QHBoxLayout *center_layout = new QHBoxLayout();
  center_layout->addStretch();  // Add stretch on left to center the grid
  
  // Create the icon grid
  QWidget *grid_container = createIconGrid();
  center_layout->addWidget(grid_container);
  
  center_layout->addStretch();  // Add stretch on right to center the grid
  
  main_vlayout->addLayout(center_layout);
  main_vlayout->addStretch();  // Push everything to the top
  
  main_layout->addWidget(icon_grid_screen);
}

QWidget* VTSCSettingsPanel::createIconGrid() {
  QWidget *grid_container = new QWidget();
  QGridLayout *grid_layout = new QGridLayout(grid_container);
  grid_layout->setSpacing(30);
  
  // Create icon buttons
  struct IconInfo {
    QString emoji;
    QString text;
    bool enabled;
  };
  
  std::vector<IconInfo> icons = {
    {"🛣️", tr("Anticipation\nDistance"), true},
    {"🔧", tr("Placeholder"), false},
    {"📊", tr("Placeholder"), false},
    {"🎯", tr("Placeholder"), false},
    {"⚡", tr("Placeholder"), false},
    {"🔍", tr("Placeholder"), false},
    {"📐", tr("Placeholder"), false},
    {"🚦", tr("Placeholder"), false},
    {"🏁", tr("Placeholder"), false},
  };
  
  int row = 0, col = 0;
  for (size_t i = 0; i < icons.size(); ++i) {
    VTSCIconButton *btn = new VTSCIconButton(icons[i].emoji, icons[i].text, grid_container);
    btn->setEnabled(icons[i].enabled);
    
    if (i == 0) {  // Anticipation Distance button
      connect(btn, &VTSCIconButton::buttonClicked, this, &VTSCSettingsPanel::anticipationSettingsClicked);
    }
    
    grid_layout->addWidget(btn, row, col);
    
    col++;
    if (col >= 3) {  // 3 columns
      col = 0;
      row++;
    }
  }
  
  return grid_container;
}