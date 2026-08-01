/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/buttons.h"

#include <cstdio>

#include <QPainter>
#include <QTimer>

#include "common/swaglog.h"
#include "common/timing.h"
#include "selfdrive/ui/qt/util.h"

ExperimentalButtonSP::ExperimentalButtonSP(QWidget *parent) : ExperimentalButton(parent) {
  QObject::disconnect(uiState(), &UIState::uiUpdate, this, &ExperimentalButton::updateState);
  QObject::connect(uiState(), &UIState::uiUpdate, this, &ExperimentalButtonSP::updateState);
}

void ExperimentalButtonSP::updateState(const UIState &s) {
  ExperimentalButton::updateState(s);
  const auto long_plan_sp = (*s.sm)["longitudinalPlanSP"].getLongitudinalPlanSP();

  int mode = int(long_plan_sp.getDec().getState());
  if ((long_plan_sp.getDec().getActive() != dynamic_experimental_control) || (mode != dec_mpc_mode)) {
    dynamic_experimental_control = long_plan_sp.getDec().getActive();
    dec_mpc_mode = mode;
    update();
  }
}

void ExperimentalButtonSP::drawButton(QPainter &p) {
  if (dynamic_experimental_control) {
    QPixmap left_half = engage_img.copy(0, 0, engage_img.width() / 2, engage_img.height());
    QPixmap right_half = experimental_img.copy(experimental_img.width() / 2, 0, experimental_img.width() / 2, experimental_img.height());

    QPixmap combined_img(engage_img.width(), engage_img.height());
    combined_img.fill(Qt::transparent);

    QPainter combined_painter(&combined_img);

    combined_painter.setOpacity(dec_mpc_mode == 1 ? 0.1 : 1.0);
    combined_painter.drawPixmap(0, 0, left_half);

    combined_painter.setOpacity(dec_mpc_mode == 1 ? 1.0 : 0.1);
    combined_painter.drawPixmap(engage_img.width() / 2, 0, right_half);

    combined_painter.end();

    drawIcon(p, QPoint(btn_size / 2, btn_size / 2), combined_img, QColor(0, 0, 0, 166), (isDown() || !engageable) ? 0.6 : 1.0);
  } else {
    ExperimentalButton::drawButton(p);
  }
}

// ---------------------------------------------------------------------------
// LongitudinalFlagButtonSP
// ---------------------------------------------------------------------------

namespace {

// Same compositing contract as the shared drawIcon() in
// selfdrive/ui/qt/onroad/buttons.cc (bg brush dictates the ellipse opacity, the
// pixmap is then drawn at `opacity`), but the radius comes from kFlagBtnSize
// instead of the file-scope btn_size. drawIcon() hard-codes btn_size / 2 == 96,
// which would paint a 192 px circle into this 128 px widget and get clipped to
// a blob. Modifying drawIcon() is not an option: ExperimentalButton depends on
// its current behaviour and that file is outside this change.
void drawFlagIcon(QPainter &p, const QPoint &center, const QPixmap &img, const QBrush &bg, float opacity) {
  p.setRenderHint(QPainter::Antialiasing);
  p.setOpacity(1.0);  // bg dictates opacity of ellipse
  p.setPen(Qt::NoPen);
  p.setBrush(bg);
  p.drawEllipse(center, kFlagBtnSize / 2, kFlagBtnSize / 2);
  p.setOpacity(opacity);
  p.drawPixmap(center - QPoint(img.width() / 2, img.height() / 2), img);
  p.setOpacity(1.0);
}

// --- Geometry tripwires ----------------------------------------------------
//
// The flag button is layout-managed (AlignBottom | AlignHCenter inside
// AnnotatedCameraWidget::main_layout), so nothing here is used at runtime --
// these exist purely to break the build if someone moves the HUD panels into
// the button's footprint.
//
// Onroad chain (all verified in-tree, not assumed):
//   - HomeWindow::offroadTransition hides the sidebar onroad
//     (selfdrive/ui/qt/home.cc), so OnroadWindow spans the full 2160x1080
//     panel (DEVICE_SCREEN_SIZE, selfdrive/ui/qt/qt_window.h).
//   - OnroadWindow's QVBoxLayout uses margin UI_BORDER_SIZE, so the
//     AnnotatedCameraWidget surface is 2100 x 1020 and that rect() is what gets
//     handed to HudRendererSP::draw as `surface_rect`.
//   - AnnotatedCameraWidget's own QVBoxLayout also uses margin UI_BORDER_SIZE,
//     so laid-out children live in a 2040 x 960 content box at (30, 30).
constexpr int kSurfaceW = 2160 - 2 * UI_BORDER_SIZE;              // 2100
constexpr int kContentW = kSurfaceW - 2 * UI_BORDER_SIZE;         // 2040
constexpr int kFlagBtnLeft = UI_BORDER_SIZE + (kContentW - kFlagBtnSize) / 2;  // 986
constexpr int kFlagBtnRightExcl = kFlagBtnLeft + kFlagBtnSize;                 // 1114

// HudRendererSP::drawRTIThreatIndicatorMulti: left_margin 15, widget_width 744
// (selfdrive/ui/sunnypilot/qt/onroad/hud.cc). Cards are clipped to that box.
constexpr int kRtiCardsRightExcl = 15 + 744;  // 759

// HudRendererSP::drawVTSCCoPilotCurve: inner = surface_rect.adjusted(30, 30, -30, -30),
// panel_left = inner.left() + (2 * inner.width()) / 3 (same file).
constexpr int kVtscPanelLeft = UI_BORDER_SIZE + (2 * kContentW) / 3;  // 1390

static_assert(kFlagBtnLeft > kRtiCardsRightExcl,
              "128 px flag button overlaps the RTI threat cards; move the button or shrink the cards");
static_assert(kFlagBtnRightExcl < kVtscPanelLeft,
              "128 px flag button overlaps the VTSC co-pilot panel; move the button or move the panel");

}  // namespace

LongitudinalFlagButtonSP::LongitudinalFlagButtonSP(QWidget *parent) : QPushButton(parent) {
  setFixedSize(kFlagBtnSize, kFlagBtnSize);
  flag_img = loadPixmap("../assets/images/button_flag.png", {kFlagIconSize, kFlagIconSize});
  QObject::connect(this, &QPushButton::clicked, this, &LongitudinalFlagButtonSP::fire);
}

void LongitudinalFlagButtonSP::fire() {
  const uint64_t now_ns = nanos_since_boot();
  if (last_fire_mono_ns != 0 && (now_ns - last_fire_mono_ns) < kFlagDebounceNs) {
    return;
  }
  last_fire_mono_ns = now_ns;
  seq++;

  // Shared publisher -- see selfdrive/ui/ui.h. Never construct a PubMaster here.
  sendBookmark();

  // One line, minimal payload: everything else about this moment is already in
  // rlog. loggerd preserves the segment off "bookmarkButton" itself, so this
  // line is for offline correlation (which button, which press, what window).
  char buf[160];
  snprintf(buf, sizeof(buf), "{\"src\":\"hud\",\"seq\":%u,\"monoNs\":%llu,\"preS\":%.1f,\"postS\":%.1f}",
           seq, (unsigned long long)now_ns, kFlagRetroPreS, kFlagRetroPostS);
  LOG("LONGFLAG %s", buf);

  just_fired = true;
  update();
  // Single-shot restore. Guarded on seq so a tap landing exactly on the
  // debounce boundary can't have its own dim cancelled by the previous timer.
  QTimer::singleShot(kFlagFlashMs, this, [this, fired_seq = seq]() {
    if (seq == fired_seq) {
      just_fired = false;
      update();
    }
  });
}

void LongitudinalFlagButtonSP::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  const float opacity = just_fired ? 0.35f : (isDown() ? 0.6f : 1.0f);
  drawFlagIcon(p, QPoint(kFlagBtnSize / 2, kFlagBtnSize / 2), flag_img, QColor(0, 0, 0, 166), opacity);
}
