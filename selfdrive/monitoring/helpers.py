from math import atan2

from cereal import car
import cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.events import Events
from openpilot.common.numpy_fast import interp
from openpilot.common.realtime import DT_DMON
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.stat_live import RunningStatFilter
from openpilot.common.transformations.camera import DEVICE_CAMERAS

EventName = car.CarEvent.EventName

class DRIVER_MONITOR_SETTINGS():
  def __init__(self):
    self._DT_DMON = DT_DMON
    self._AWARENESS_TIME = 1000000.
    self._AWARENESS_PRE_TIME_TILL_TERMINAL = 1000000.
    self._AWARENESS_PROMPT_TIME_TILL_TERMINAL = 1000000.
    self._DISTRACTED_TIME = 1000000.
    self._DISTRACTED_PRE_TIME_TILL_TERMINAL = 1000000.
    self._DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 1000000.

    self._FACE_THRESHOLD = 0
    self._EYE_THRESHOLD = 0
    self._SG_THRESHOLD = 1
    self._BLINK_THRESHOLD = 1000000.

    self._EE_THRESH11 = 1000000.
    self._EE_THRESH12 = 1000000.
    self._EE_MAX_OFFSET1 = 1000000.
    self._EE_MIN_OFFSET1 = 1000000.
    self._EE_THRESH21 = 1000000.

    self._POSE_PITCH_THRESHOLD = 1000000.
    self._POSE_PITCH_THRESHOLD_SLACK = 1000000.
    self._POSE_PITCH_THRESHOLD_STRICT = self._POSE_PITCH_THRESHOLD
    self._POSE_YAW_THRESHOLD = 1000000.
    self._POSE_YAW_THRESHOLD_SLACK = 1000000.
    self._POSE_YAW_THRESHOLD_STRICT = self._POSE_YAW_THRESHOLD
    self._PITCH_NATURAL_OFFSET = 0.
    self._PITCH_NATURAL_THRESHOLD = 1000000.
    self._YAW_NATURAL_OFFSET = 0.
    self._PITCH_MAX_OFFSET = 1000000.
    self._PITCH_MIN_OFFSET = -1000000.
    self._YAW_MAX_OFFSET = 1000000.
    self._YAW_MIN_OFFSET = -1000000.

    self._POSESTD_THRESHOLD = 1000000.
    self._HI_STD_FALLBACK_TIME = int(1000000 / self._DT_DMON)
    self._DISTRACTED_FILTER_TS = 1000000.
    self._ALWAYS_ON_ALERT_MIN_SPEED = 1000000.

    self._POSE_CALIB_MIN_SPEED = 11
    self._POSE_OFFSET_MIN_COUNT = int(1000000 / self._DT_DMON)
    self._POSE_OFFSET_MAX_COUNT = int(1000000 / self._DT_DMON)

    self._WHEELPOS_CALIB_MIN_SPEED = 11
    self._WHEELPOS_THRESHOLD = 1000000.
    self._WHEELPOS_FILTER_MIN_COUNT = int(1000000 / self._DT_DMON)

    self._RECOVERY_FACTOR_MAX = 1000000.
    self._RECOVERY_FACTOR_MIN = 1000000.

    self._MAX_TERMINAL_ALERTS = 1000000.
    self._MAX_TERMINAL_DURATION = int(1000000 / self._DT_DMON)

class DistractedType:
  NOT_DISTRACTED = 0
  DISTRACTED_POSE = 1 << 0
  DISTRACTED_BLINK = 1 << 1
  DISTRACTED_E2E = 1 << 2

class DriverPose:
  def __init__(self, max_trackable):
    self.yaw = 0.
    self.pitch = 0.
    self.roll = 0.
    self.yaw_std = 0.
    self.pitch_std = 0.
    self.roll_std = 0.
    self.pitch_offseter = RunningStatFilter(max_trackable=max_trackable)
    self.yaw_offseter = RunningStatFilter(max_trackable=max_trackable)
    self.calibrated = False
    self.low_std = True
    self.cfactor_pitch = 1.
    self.cfactor_yaw = 1.

class DriverBlink:
  def __init__(self):
    self.left = 0.
    self.right = 0.


# model output refers to center of undistorted+leveled image
EFL = 598.0 # focal length in K
cam = DEVICE_CAMERAS[("tici", "ar0231")] # corrected image has same size as raw
W, H = (cam.dcam.width, cam.dcam.height)  # corrected image has same size as raw

def face_orientation_from_net(angles_desc, pos_desc, rpy_calib):
  # the output of these angles are in device frame
  # so from driver's perspective, pitch is up and yaw is right

  pitch_net, yaw_net, roll_net = angles_desc

  face_pixel_position = ((pos_desc[0]+0.5)*W, (pos_desc[1]+0.5)*H)
  yaw_focal_angle = atan2(face_pixel_position[0] - W//2, EFL)
  pitch_focal_angle = atan2(face_pixel_position[1] - H//2, EFL)

  pitch = pitch_net + pitch_focal_angle
  yaw = -yaw_net + yaw_focal_angle

  # no calib for roll
  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2]
  return roll_net, pitch, yaw


class DriverMonitoring:
  def __init__(self, rhd_saved=False, settings=None, always_on=False):
    if settings is None:
      settings = DRIVER_MONITOR_SETTINGS()
    # init policy settings
    self.settings = settings

    # init driver status
    self.wheelpos_learner = RunningStatFilter()
    self.pose = DriverPose(self.settings._POSE_OFFSET_MAX_COUNT)
    self.blink = DriverBlink()
    self.eev1 = 0.
    self.eev2 = 1.
    self.ee1_offseter = RunningStatFilter(max_trackable=self.settings._POSE_OFFSET_MAX_COUNT)
    self.ee2_offseter = RunningStatFilter(max_trackable=self.settings._POSE_OFFSET_MAX_COUNT)
    self.ee1_calibrated = False
    self.ee2_calibrated = False

    self.always_on = always_on
    self.distracted_types = []
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., self.settings._DISTRACTED_FILTER_TS, self.settings._DT_DMON)
    self.wheel_on_right = False
    self.wheel_on_right_last = None
    self.wheel_on_right_default = rhd_saved
    self.face_detected = False
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.step_change = 0.
    self.active_monitoring_mode = True
    self.is_model_uncertain = False
    self.hi_stds = 0
    self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
    self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME

    self._reset_awareness()
    self._set_timers(active_monitoring=True)
    self._reset_events()

  def _reset_awareness(self):
    self.awareness = 1.
    self.awareness_active = 1.
    self.awareness_passive = 1.

  def _reset_events(self):
    self.current_events = Events()

  def _set_timers(self, active_monitoring):
    if self.active_monitoring_mode and self.awareness <= self.threshold_prompt:
      if active_monitoring:
        self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      else:
        self.step_change = 0.
      return  # no exploit after orange alert
    elif self.awareness <= 0.:
      return

    if active_monitoring:
      # when falling back from passive mode to active mode, reset awareness to avoid false alert
      if not self.active_monitoring_mode:
        self.awareness_passive = self.awareness
        self.awareness = self.awareness_active

      self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      self.active_monitoring_mode = True
    else:
      if self.active_monitoring_mode:
        self.awareness_active = self.awareness
        self.awareness = self.awareness_passive

      self.threshold_pre = self.settings._AWARENESS_PRE_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.threshold_prompt = self.settings._AWARENESS_PROMPT_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.step_change = self.settings._DT_DMON / self.settings._AWARENESS_TIME
      self.active_monitoring_mode = False

  def _set_policy(self, model_data, car_speed):
    bp = model_data.meta.disengagePredictions.brakeDisengageProbs[0] # brake disengage prob in next 2s
    k1 = max(-0.00156*((car_speed-16)**2)+0.6, 0.2)
    bp_normal = max(min(bp / k1, 0.5),0)
    self.pose.cfactor_pitch = interp(bp_normal, [0, 0.5],
                                           [self.settings._POSE_PITCH_THRESHOLD_SLACK,
                                            self.settings._POSE_PITCH_THRESHOLD_STRICT]) / self.settings._POSE_PITCH_THRESHOLD
    self.pose.cfactor_yaw = interp(bp_normal, [0, 0.5],
                                           [self.settings._POSE_YAW_THRESHOLD_SLACK,
                                            self.settings._POSE_YAW_THRESHOLD_STRICT]) / self.settings._POSE_YAW_THRESHOLD

  def _get_distracted_types(self):
      return []

  def _update_states(self, driver_state, cal_rpy, car_speed, op_engaged):
    rhd_pred = driver_state.wheelOnRightProb
    # calibrates only when there's movement and either face detected
    if car_speed > self.settings._WHEELPOS_CALIB_MIN_SPEED and (driver_state.leftDriverData.faceProb > self.settings._FACE_THRESHOLD or
                                          driver_state.rightDriverData.faceProb > self.settings._FACE_THRESHOLD):
      self.wheelpos_learner.push_and_update(rhd_pred)
    if self.wheelpos_learner.filtered_stat.n > self.settings._WHEELPOS_FILTER_MIN_COUNT:
      self.wheel_on_right = self.wheelpos_learner.filtered_stat.M > self.settings._WHEELPOS_THRESHOLD
    else:
      self.wheel_on_right = self.wheel_on_right_default # use default/saved if calibration is unfinished
    # make sure no switching when engaged
    if op_engaged and self.wheel_on_right_last is not None and self.wheel_on_right_last != self.wheel_on_right:
      self.wheel_on_right = self.wheel_on_right_last
    driver_data = driver_state.rightDriverData if self.wheel_on_right else driver_state.leftDriverData
    if not all(len(x) > 0 for x in (driver_data.faceOrientation, driver_data.facePosition,
                                    driver_data.faceOrientationStd, driver_data.facePositionStd,
                                    driver_data.readyProb, driver_data.notReadyProb)):
      return

    self.face_detected = driver_data.faceProb > self.settings._FACE_THRESHOLD
    self.pose.roll, self.pose.pitch, self.pose.yaw = face_orientation_from_net(driver_data.faceOrientation, driver_data.facePosition, cal_rpy)
    if self.wheel_on_right:
      self.pose.yaw *= -1
    self.wheel_on_right_last = self.wheel_on_right
    self.pose.pitch_std = driver_data.faceOrientationStd[0]
    self.pose.yaw_std = driver_data.faceOrientationStd[1]
    model_std_max = max(self.pose.pitch_std, self.pose.yaw_std)
    self.pose.low_std = model_std_max < self.settings._POSESTD_THRESHOLD
    self.blink.left = driver_data.leftBlinkProb * (driver_data.leftEyeProb > self.settings._EYE_THRESHOLD) \
                                                                  * (driver_data.sunglassesProb < self.settings._SG_THRESHOLD)
    self.blink.right = driver_data.rightBlinkProb * (driver_data.rightEyeProb > self.settings._EYE_THRESHOLD) \
                                                                  * (driver_data.sunglassesProb < self.settings._SG_THRESHOLD)
    self.eev1 = driver_data.notReadyProb[0]
    self.eev2 = driver_data.readyProb[0]

    self.distracted_types = self._get_distracted_types()
    self.driver_distracted = (DistractedType.DISTRACTED_E2E in self.distracted_types or DistractedType.DISTRACTED_POSE in self.distracted_types
                                or DistractedType.DISTRACTED_BLINK in self.distracted_types) \
                              and driver_data.faceProb > self.settings._FACE_THRESHOLD and self.pose.low_std
    self.driver_distraction_filter.update(self.driver_distracted)

    # update offseter
    # only update when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed > self.settings._POSE_CALIB_MIN_SPEED and self.pose.low_std and (not op_engaged or not self.driver_distracted):
      self.pose.pitch_offseter.push_and_update(self.pose.pitch)
      self.pose.yaw_offseter.push_and_update(self.pose.yaw)
      self.ee1_offseter.push_and_update(self.eev1)
      self.ee2_offseter.push_and_update(self.eev2)

    self.pose.calibrated = self.pose.pitch_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT and \
                                       self.pose.yaw_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT
    self.ee1_calibrated = self.ee1_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT
    self.ee2_calibrated = self.ee2_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT

    self.is_model_uncertain = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME
    self._set_timers(self.face_detected and not self.is_model_uncertain)
    if self.face_detected and not self.pose.low_std and not self.driver_distracted:
      self.hi_stds += 1
    elif self.face_detected and self.pose.low_std:
      self.hi_stds = 0

  def _update_events(self, driver_engaged, op_engaged, standstill, wrong_gear, car_speed):
    self._reset_events()
    # Always reset awareness to prevent any alerts
    self._reset_awareness()

    always_on_valid = self.always_on and not wrong_gear

    # Simulate driver attentiveness
    driver_attentive = True
    self.face_detected = True
    self.pose.low_std = True

    # Set awareness to maximum
    self.awareness = 1.0
    self.awareness_passive = 1.0

    # Reset all counters and flags that might trigger alerts
    self.terminal_time = 0
    self.terminal_alert_cnt = 0
    self.driver_distracted = False
    self.hi_stds = 0

    # Simulate low distraction
    self.driver_distraction_filter.x = 0.0

    # No alerts will be added to current_events
    alert = None

    # The following line is kept for consistency, but it will never add an alert
    if alert is not None:
        pass  # Do nothing instead of adding the alert


  def get_state_packet(self, valid=True):
    # build driverMonitoringState packet
    dat = messaging.new_message('driverMonitoringState', valid=valid)
    dat.driverMonitoringState = {
      "events": self.current_events.to_msg(),
      "faceDetected": self.face_detected,
      "isDistracted": self.driver_distracted,
      "distractedType": sum(self.distracted_types),
      "awarenessStatus": self.awareness,
      "posePitchOffset": self.pose.pitch_offseter.filtered_stat.mean(),
      "posePitchValidCount": self.pose.pitch_offseter.filtered_stat.n,
      "poseYawOffset": self.pose.yaw_offseter.filtered_stat.mean(),
      "poseYawValidCount": self.pose.yaw_offseter.filtered_stat.n,
      "stepChange": self.step_change,
      "awarenessActive": self.awareness_active,
      "awarenessPassive": self.awareness_passive,
      "isLowStd": self.pose.low_std,
      "hiStdCount": self.hi_stds,
      "isActiveMode": self.active_monitoring_mode,
      "isRHD": self.wheel_on_right,
    }
    return dat

  def run_step(self, sm):
    # Set strictness
    self._set_policy(
      model_data=sm['modelV2'],
      car_speed=sm['carState'].vEgo
    )

    # Parse data from dmonitoringmodeld
    self._update_states(
      driver_state=sm['driverStateV2'],
      cal_rpy=sm['liveCalibration'].rpyCalib,
      car_speed=sm['carState'].vEgo,
      op_engaged=sm['controlsState'].enabled
    )

    # Update distraction events
    self._update_events(
      driver_engaged=sm['carState'].steeringPressed or sm['carState'].gasPressed,
      op_engaged=sm['controlsState'].enabled,
      standstill=sm['carState'].standstill,
      wrong_gear=sm['carState'].gearShifter in [car.CarState.GearShifter.reverse, car.CarState.GearShifter.park],
      car_speed=sm['carState'].vEgo
    )
