/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/ui.h"

#include "common/watchdog.h"

void UIStateSP::updateStatus() {
  UIState::updateStatus();
}

UIStateSP::UIStateSP(QObject *parent) : UIState(parent) {
  sm = std::make_unique<SubMaster>(std::vector<const char*>{
    "modelV2", "controlsState", "liveCalibration", "radarState", "deviceState",
    "pandaStates", "carParams", "driverMonitoringState", "carState", "driverStateV2",
    "wideRoadCameraState", "managerState", "selfdriveState", "longitudinalPlan",
    "modelManagerSP", "selfdriveStateSP", "longitudinalPlanSP", "backupManagerSP", "carControl", "carOutput", "liveMapDataSP",
    "rtiStateSP",  // RTI (Realtime Traffic Intelligence) state with pre-computed threat display data
    "weatherOverlaySP",  // Heading-up precipitation wash for the SP onroad HUD (rendered north-up, rotated at draw time)
    // GPS subscriptions: weather overlay rotates the north-up radar PNG to the
    // ego compass heading, which is only sourced reliably from the GPS receiver.
    // livePose.orientationNED.z is *not* a compass heading — locationd's pose_kf
    // has no absolute-yaw observation, so its yaw drifts from an arbitrary zero.
    "gpsLocation",
    "gpsLocationExternal",
  });

  // update timer
  timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &UIStateSP::update);
  timer->start(1000 / UI_FREQ);
}

// This method overrides completely the update method from the parent class intentionally.
void UIStateSP::update() {
  update_sockets(this);
  update_state(this);
  updateStatus();

  // Force onroad mode if environment variable is set (dev only)
  if (getenv("FORCE_ONROAD_UI") && !scene.started) {
    scene.started = true;
    scene.started_frame = sm->frame;
  }

  if (sm->frame % UI_FREQ == 0) {
    watchdog_kick(nanos_since_boot());
  }
  emit uiUpdate(*this);
}

DeviceSP::DeviceSP(QObject *parent) : Device(parent) {
  QObject::connect(uiStateSP(), &UIStateSP::uiUpdate, this, &DeviceSP::update);
  QObject::connect(this, &Device::displayPowerChanged, this, &DeviceSP::handleDisplayPowerChanged);
}

UIStateSP *uiStateSP() {
  static UIStateSP ui_state;
  return &ui_state;
}

void UIStateSP::setSunnylinkRoles(const std::vector<RoleModel>& roles) {
  sunnylinkRoles = roles;
  emit sunnylinkRolesChanged(roles);
}

void UIStateSP::setSunnylinkDeviceUsers(const std::vector<UserModel>& users) {
  sunnylinkUsers = users;
  emit sunnylinkDeviceUsersChanged(users);
}

DeviceSP *deviceSP() {
  static DeviceSP _device;
  return &_device;
}

void DeviceSP::handleDisplayPowerChanged(bool on) {
  // if enabled, trigger offroad mode when device goes to sleep
  if (params.get("DeviceBootMode") == "1" && not on) {
    params.putBool("OffroadMode", true);
  }
}
