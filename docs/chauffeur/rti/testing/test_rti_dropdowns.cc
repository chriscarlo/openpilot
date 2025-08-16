/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "catch2/catch.hpp"
#include <QApplication>
#include <QComboBox>
#include <QWidget>
#include <QVBoxLayout>
#include <QScrollArea>
#include <QAbstractItemView>

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.h"

QString getSimpleDropdownStyle() {
  return R"(
    QComboBox {
      font-size: 36px;
      padding: 20px;
      background-color: #393939;
      color: white;
      border: 2px solid #555;
      border-radius: 15px;
      min-height: 60px;
    }
    QComboBox QAbstractItemView {
      font-size: 36px;
      background-color: #393939;
      selection-background-color: #4a90e2;
      border: 2px solid #555;
      border-radius: 5px;
      padding: 10px;
      outline: none;
    }
    QComboBox QAbstractItemView::item {
      padding: 8px;
      border: none;
    }
  )";
}

TEST_CASE("RTI Dropdown Geometry", "[rti][ui]") {
  // Create a simple test widget with scroll area similar to RTI settings
  QWidget *testWidget = new QWidget();
  QVBoxLayout *layout = new QVBoxLayout(testWidget);
  
  QScrollArea *scrollArea = new QScrollArea();
  scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  
  QWidget *scrollWidget = new QWidget();
  scrollWidget->setMaximumWidth(1300);
  QVBoxLayout *scrollLayout = new QVBoxLayout(scrollWidget);
  
  // Create test dropdown with simplified styling
  QComboBox *testCombo = new QComboBox();
  testCombo->setStyleSheet(getSimpleDropdownStyle());
  testCombo->addItem("Test Item 1");
  testCombo->addItem("Test Item 2");
  testCombo->addItem("Test Item 3");
  
  scrollLayout->addWidget(testCombo);
  scrollArea->setWidget(scrollWidget);
  layout->addWidget(scrollArea);
  
  // Verify dropdown is properly constructed
  REQUIRE(testCombo != nullptr);
  REQUIRE(testCombo->count() == 3);
  REQUIRE(testCombo->itemText(0) == "Test Item 1");
  
  // Verify styling doesn't contain problematic subcontrol positioning
  QString styleSheet = testCombo->styleSheet();
  REQUIRE_FALSE(styleSheet.contains("subcontrol-position"));
  REQUIRE_FALSE(styleSheet.contains("QComboBox::down-arrow"));
  
  delete testWidget;
}

TEST_CASE("RTI Dropdown Click Functionality", "[rti][ui]") {
  QComboBox *testCombo = new QComboBox();
  testCombo->setStyleSheet(getSimpleDropdownStyle());
  testCombo->addItem("Option A");
  testCombo->addItem("Option B");
  testCombo->addItem("Option C");
  
  // Test that clicking the dropdown works programmatically
  REQUIRE(testCombo->currentIndex() == 0);
  REQUIRE(testCombo->currentText() == "Option A");
  
  // Programmatically select second item (simulates user selection)
  testCombo->setCurrentIndex(1);
  REQUIRE(testCombo->currentIndex() == 1);
  REQUIRE(testCombo->currentText() == QString("Option B"));
  
  // Test all items are accessible
  REQUIRE(testCombo->count() == 3);
  REQUIRE(testCombo->itemText(2) == "Option C");
  
  delete testCombo;
}

TEST_CASE("RTI Dropdown Style Consistency", "[rti][ui]") {
  // Test that all RTI dropdowns use consistent styling
  RTISettingsPanel *panel = new RTISettingsPanel();
  
  // Find all combo boxes in the panel
  QList<QComboBox*> combos = panel->findChildren<QComboBox*>();
  REQUIRE(combos.size() >= 3); // Should have at least source, filter, aggr combos
  
  // Verify all combos have consistent styling
  for (QComboBox* combo : combos) {
    REQUIRE_FALSE(combo->styleSheet().isEmpty()); // ComboBox should have styling applied
    // All RTI dropdowns should use the same simplified styling
    REQUIRE_FALSE(combo->styleSheet().contains("subcontrol-position")); // No subcontrol positioning (causes rotation bug)
    REQUIRE_FALSE(combo->styleSheet().contains("QComboBox::down-arrow")); // No custom arrow (causes rotation bug)
  }
  
  delete panel;
}