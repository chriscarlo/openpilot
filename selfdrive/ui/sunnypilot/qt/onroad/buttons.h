/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <cstdint>

#include <QPixmap>

#include "selfdrive/ui/qt/onroad/buttons.h"

// Longitudinal incident flag button geometry.
//
// Deliberately SEPARATE from the shared btn_size/img_size in
// selfdrive/ui/qt/onroad/buttons.h (192/144) -- ExperimentalButton depends on
// those and must not change. 128 px was chosen so the bottom-center button
// clears most of the ModelRenderer lead-status readout band (the 150 px-wide
// dRel / speed / TTC text drawn under the lead chevron, which pins to the
// bottom center of the surface for a close lead) while still being a ~15 mm
// thumb target on the tici's 2160x1080 panel.
constexpr int kFlagBtnSize = 128;
constexpr int kFlagIconSize = (kFlagBtnSize / 4) * 3;  // 96

// Retro capture window advertised in the LONGFLAG cloudlog payload.
constexpr double kFlagRetroPreS = 20.0;
constexpr double kFlagRetroPostS = 6.0;

// Two taps closer together than this are one incident; taps farther apart are
// DELIBERATELY two separate incidents.
constexpr uint64_t kFlagDebounceNs = 1000ULL * 1000ULL * 1000ULL;  // 1.0 s
constexpr int kFlagFlashMs = 1000;                                 // just-fired dim, ms

class ExperimentalButtonSP : public ExperimentalButton {
  Q_OBJECT

public:
  explicit ExperimentalButtonSP(QWidget *parent = nullptr);
  void updateState(const UIState &s) override;

private:
  void drawButton(QPainter &p) override;

  bool dynamic_experimental_control;
  int dec_mpc_mode;
};

/**
 * Bottom-center "flag this longitudinal incident" button.
 *
 * One tap, no category UI -- the category is derived offline from the
 * preserved segment. Publishes the existing "bookmarkButton" cereal event via
 * the process-wide sendBookmark() helper (see selfdrive/ui/ui.h: msgq evicts
 * the older publisher of an endpoint, and the sidebar flag button publishes the
 * same event from this process), then emits ONE cloudlog line so the retro
 * window and press sequence land in rlog. No new cereal message, no new
 * service, no new capnp id.
 */
class LongitudinalFlagButtonSP : public QPushButton {
  Q_OBJECT

public:
  explicit LongitudinalFlagButtonSP(QWidget *parent = nullptr);

private:
  void paintEvent(QPaintEvent *event) override;
  void fire();

  QPixmap flag_img;
  uint64_t last_fire_mono_ns = 0;  // 0 == never fired; nanos_since_boot()
  uint32_t seq = 0;                // monotonic press counter, logged in the payload
  bool just_fired = false;         // drives the post-tap dim
};
