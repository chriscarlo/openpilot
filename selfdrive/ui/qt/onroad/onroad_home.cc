#include "selfdrive/ui/qt/onroad/onroad_home.h"

#include <algorithm>
#include <cmath>

#include <QPainter>
#include <QLinearGradient>
#include <QStackedLayout>

#include "selfdrive/ui/qt/util.h"

namespace {

constexpr int kTorqueAnimationHz = 60;
constexpr float kTorqueRiseResponseHz = 18.0f;
constexpr float kTorqueFallResponseHz = 12.0f;
constexpr float kTorqueMagnitudeEpsilon = 0.0025f;
constexpr int kTorqueStubHeightPx = 10;
constexpr int kTorqueHeadFeatherPx = 12;

const QColor kTorqueGreen(0x33, 0xCC, 0x33, 220);
const QColor kTorqueYellow(0xFF, 0xCC, 0x00, 228);
const QColor kTorqueOrange(0xFF, 0x99, 0x1A, 236);
const QColor kTorqueRed(0xFF, 0x33, 0x33, 242);

int torqueSideFromValue(float torque) {
  if (torque > kTorqueMagnitudeEpsilon) return 1;
  if (torque < -kTorqueMagnitudeEpsilon) return -1;
  return 0;
}

float smoothingAlpha(float response_hz, float dt) {
  return 1.0f - std::exp(-response_hz * std::max(dt, 0.0f));
}

}  // namespace

OnroadWindow::OnroadWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout  = new QVBoxLayout(this);
  main_layout->setMargin(UI_BORDER_SIZE);
  QStackedLayout *stacked_layout = new QStackedLayout;
  stacked_layout->setStackingMode(QStackedLayout::StackAll);
  main_layout->addLayout(stacked_layout);

  nvg = new AnnotatedCameraWidget(VISION_STREAM_ROAD, this);

  QWidget * split_wrapper = new QWidget;
  split = new QHBoxLayout(split_wrapper);
  split->setContentsMargins(0, 0, 0, 0);
  split->setSpacing(0);
  split->addWidget(nvg);

  if (getenv("DUAL_CAMERA_VIEW")) {
    CameraWidget *arCam = new CameraWidget("camerad", VISION_STREAM_ROAD, this);
    split->insertWidget(0, arCam);
  }

  stacked_layout->addWidget(split_wrapper);

  alerts = new OnroadAlerts(this);
  alerts->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  stacked_layout->addWidget(alerts);

  // setup stacking order
  alerts->raise();

  setAttribute(Qt::WA_OpaquePaintEvent);

  torque_meter_timer = new QTimer(this);
  torque_meter_timer->setTimerType(Qt::PreciseTimer);
  torque_meter_timer->setInterval(1000 / kTorqueAnimationHz);
  QObject::connect(torque_meter_timer, &QTimer::timeout, this, &OnroadWindow::updateTorqueAnimation);
  torque_meter_timer->start();

  // We handle the connection of the signals on the derived class
#ifndef SUNNYPILOT
  QObject::connect(uiState(), &UIState::uiUpdate, this, &OnroadWindow::updateState);
  QObject::connect(uiState(), &UIState::offroadTransition, this, &OnroadWindow::offroadTransition);
#endif
}

void OnroadWindow::updateState(const UIState &s) {
  if (!s.scene.started) {
    return;
  }

  alerts->updateState(s);
  nvg->updateState(s);

  QColor bgColor = bg_colors[s.status];
  if (bg != bgColor) {
    // repaint border
    bg = bgColor;
    update();
  }

  if (!s.sm) {
    target_torque_norm_ = 0.0f;
    target_torque_side_ = 0;
    return;
  }

  const SubMaster &sm = *s.sm;
  const bool car_output_ready = sm.rcv_frame("carOutput") >= s.scene.started_frame && sm.valid("carOutput");
  const bool car_control_ready = sm.rcv_frame("carControl") >= s.scene.started_frame && sm.valid("carControl");

  const bool lat_active = car_control_ready && sm["carControl"].getCarControl().getLatActive();
  float torque_output = car_output_ready ? sm["carOutput"].getCarOutput().getActuatorsOutput().getTorque() : 0.0f;
  if (!std::isfinite(torque_output) || !lat_active) {
    torque_output = 0.0f;
  }

  target_torque_norm_ = std::clamp(std::abs(torque_output), 0.0f, 1.0f);
  target_torque_side_ = lat_active ? torqueSideFromValue(torque_output) : 0;
  if (target_torque_norm_ <= kTorqueMagnitudeEpsilon) {
    target_torque_norm_ = 0.0f;
    target_torque_side_ = 0;
  }
}

void OnroadWindow::offroadTransition(bool offroad) {
  alerts->clear();

  if (offroad) {
    target_torque_norm_ = 0.0f;
    display_torque_norm_ = 0.0f;
    target_torque_side_ = 0;
    display_torque_side_ = 0;
    last_torque_animation_ts_valid_ = false;
    update();
  }
}

void OnroadWindow::updateTorqueAnimation() {
  const auto now = std::chrono::steady_clock::now();
  const float dt = last_torque_animation_ts_valid_
    ? std::min(std::chrono::duration<float>(now - last_torque_animation_ts_).count(), 0.10f)
    : (1.0f / static_cast<float>(kTorqueAnimationHz));
  last_torque_animation_ts_ = now;
  last_torque_animation_ts_valid_ = true;

  const float prev_display_torque = display_torque_norm_;
  const int prev_display_side = display_torque_side_;

  const float response_hz = (target_torque_norm_ > display_torque_norm_) ? kTorqueRiseResponseHz : kTorqueFallResponseHz;
  display_torque_norm_ += (target_torque_norm_ - display_torque_norm_) * smoothingAlpha(response_hz, dt);

  if (target_torque_side_ != 0) {
    display_torque_side_ = target_torque_side_;
  } else if (display_torque_norm_ <= kTorqueMagnitudeEpsilon) {
    display_torque_side_ = 0;
  }

  if (target_torque_norm_ <= kTorqueMagnitudeEpsilon && display_torque_norm_ <= kTorqueMagnitudeEpsilon) {
    display_torque_norm_ = 0.0f;
    display_torque_side_ = 0;
  }

  if (std::abs(display_torque_norm_ - prev_display_torque) > 1e-4f || display_torque_side_ != prev_display_side) {
    update();
  }
}

void OnroadWindow::drawTorqueMeters(QPainter &p) const {
  const int strip_height = height() - (2 * UI_BORDER_SIZE);
  if (strip_height <= 0 || width() <= (2 * UI_BORDER_SIZE)) {
    return;
  }

  const QRect left_strip(0, UI_BORDER_SIZE, UI_BORDER_SIZE, strip_height);
  const QRect right_strip(width() - UI_BORDER_SIZE, UI_BORDER_SIZE, UI_BORDER_SIZE, strip_height);

  const bool left_active = display_torque_side_ > 0 && display_torque_norm_ > kTorqueMagnitudeEpsilon;
  const bool right_active = display_torque_side_ < 0 && display_torque_norm_ > kTorqueMagnitudeEpsilon;

  drawTorqueMeterStrip(p, left_strip, left_active ? display_torque_norm_ : 0.0f, left_active);
  drawTorqueMeterStrip(p, right_strip, right_active ? display_torque_norm_ : 0.0f, right_active);
}

void OnroadWindow::drawTorqueMeterStrip(QPainter &p, const QRect &strip_rect, float fill_norm, bool active) const {
  if (!strip_rect.isValid() || strip_rect.width() <= 0 || strip_rect.height() <= 0) {
    return;
  }

  const int stub_height = std::clamp(strip_rect.height() / 22, kTorqueStubHeightPx, strip_rect.height());

  if (active && fill_norm > 0.0f) {
    const int fill_height = std::clamp(static_cast<int>(std::lround(fill_norm * strip_rect.height())), stub_height, strip_rect.height());
    QRect fill_rect(strip_rect.left(), strip_rect.bottom() - fill_height + 1, strip_rect.width(), fill_height);

    QLinearGradient fill_grad(strip_rect.center().x(), strip_rect.top(), strip_rect.center().x(), strip_rect.bottom() + 1);
    fill_grad.setColorAt(0.00, kTorqueRed);
    fill_grad.setColorAt(0.10, kTorqueRed);
    fill_grad.setColorAt(0.24, kTorqueOrange);
    fill_grad.setColorAt(0.46, kTorqueYellow);
    fill_grad.setColorAt(1.00, kTorqueGreen);
    p.fillRect(fill_rect, fill_grad);

    const int head_feather = std::min({kTorqueHeadFeatherPx, fill_rect.height(), strip_rect.height()});
    if (head_feather > 1 && fill_rect.top() > strip_rect.top()) {
      QRect head_rect(strip_rect.left(), fill_rect.top(), strip_rect.width(), head_feather);
      QColor head_color = (fill_norm >= 0.90f) ? kTorqueRed :
                          (fill_norm >= 0.65f) ? kTorqueOrange :
                          (fill_norm >= 0.35f) ? kTorqueYellow : kTorqueGreen;
      QColor head_fade = head_color;
      head_color.setAlpha(140);
      head_fade.setAlpha(0);
      QLinearGradient head_grad(strip_rect.center().x(), head_rect.top(), strip_rect.center().x(), head_rect.bottom());
      head_grad.setColorAt(0.0, head_fade);
      head_grad.setColorAt(1.0, head_color);
      p.fillRect(head_rect, head_grad);
    }
  }

  QRect stub_rect(strip_rect.left(), strip_rect.bottom() - stub_height + 1, strip_rect.width(), stub_height);
  QColor stub_color = kTorqueGreen;
  stub_color.setAlpha(active ? 172 : 112);
  p.fillRect(stub_rect, stub_color);
}

void OnroadWindow::paintEvent(QPaintEvent *event) {
  (void)event;
  QPainter p(this);
  p.fillRect(rect(), QColor(bg.red(), bg.green(), bg.blue(), 255));
  drawTorqueMeters(p);
}
