/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/expandable_row.h"

enum class RTISourceType {
  DISABLED,
  WAZE,
  TOMTOM,
  INRIX,
  MANUAL_API,
};

inline const char *RTISourceTypeText[]{
  QT_TR_NOOP("Disabled"),
  QT_TR_NOOP("Waze"),
  QT_TR_NOOP("TomTom"),
  QT_TR_NOOP("INRIX"),
  QT_TR_NOOP("Manual API"),
};

enum class RTIThreatFilter {
  ALL,
  POLICE_ONLY,
  SPEED_ONLY,
  HAZARDS_ONLY,
  CUSTOM,
};

inline const char *RTIThreatFilterText[]{
  QT_TR_NOOP("All Threats"),
  QT_TR_NOOP("Police Only"),
  QT_TR_NOOP("Speed Only"),
  QT_TR_NOOP("Hazards Only"),
  QT_TR_NOOP("Custom Filter"),
};

enum class RTIAggressiveness {
  CONSERVATIVE,
  BALANCED,
  AGGRESSIVE,
};

inline const char *RTIAggressivenessText[]{
  QT_TR_NOOP("Conservative"),
  QT_TR_NOOP("Balanced"),
  QT_TR_NOOP("Aggressive"),
};

class RTIControl : public ExpandableToggleRow {
  Q_OBJECT

public:
  RTIControl(const QString &param, const QString &title, const QString &desc, const QString &icon, QWidget *parent = nullptr);

signals:
  void rtiSettingsButtonClicked();

private:
  Params params;
  PushButtonSP *rtiSettings;
};