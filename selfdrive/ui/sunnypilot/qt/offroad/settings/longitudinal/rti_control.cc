/**
 * RTI Control Widget Implementation - Clean Redesign
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_control.h"
#include <QHBoxLayout>
#include <QLabel>

RTIControl::RTIControl(QWidget *parent) : AbstractControlSP(
  tr("Real-time Traffic Intelligence"),
  tr("Monitor traffic ahead and adjust speed automatically"),
  "",
  parent
) {
  // Create toggle
  toggle = new ToggleSP();
  toggle->setFixedSize(150, 100);  // Match other controls
  
  // Connect toggle state changes
  connect(toggle, &ToggleSP::stateChanged, this, &RTIControl::updateState);
  
  // Create settings button with gear icon
  settingsBtn = new QPushButton();
  settingsBtn->setFixedSize(120, 120);  // 20% larger than original 100x100
  settingsBtn->setEnabled(params.getBool("RTIEnabled"));
  settingsBtn->setStyleSheet(R"(
    QPushButton {
      background-color: #393939;
      border-radius: 60px;
      font-size: 63px;
      font-weight: 500;
      border: 2px solid #696969;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
    QPushButton:disabled {
      background-color: #2d2d2d;
      border-color: #444444;
      color: #696969;
    }
  )");
  settingsBtn->setText("⚙");  // Gear emoji
  
  connect(settingsBtn, &QPushButton::clicked, this, &RTIControl::settingsClicked);
  
  // Create a container for settings button and toggle (swapped order)
  QWidget *controls_container = new QWidget(this);
  QHBoxLayout *controls_layout = new QHBoxLayout(controls_container);
  controls_layout->setContentsMargins(0, 0, 0, 0);
  controls_layout->setSpacing(30);  // Increased from 20px to 30px (50% increase)
  
  // Add settings button first, then toggle (swapped positions)
  controls_layout->addWidget(settingsBtn);
  controls_layout->addWidget(toggle);
  
  hlayout->addWidget(controls_container);
}

void RTIControl::showEvent(QShowEvent *event) {
  refresh();
  AbstractControlSP::showEvent(event);
}

void RTIControl::refresh() {
  if (!toggle || !settingsBtn) {
    return;
  }
  
  bool enabled = params.getBool("RTIEnabled");
  if (enabled != toggle->on) {
    toggle->togglePosition();
  }
  settingsBtn->setEnabled(enabled);
}

void RTIControl::updateState(bool enabled) {
  params.putBool("RTIEnabled", enabled);
  settingsBtn->setEnabled(enabled);
}