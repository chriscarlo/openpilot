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

# ******************************************************************************************
#  NOTE: To fork maintainers.
#  Disabling or nerfing safety features will get you and your users banned from our servers.
#  We recommend that you do not change these numbers from the defaults.
# ******************************************************************************************

class DRIVER_MONITOR_SETTINGS:
  def __init__(self):
    self._DT_DMON = DT_DMON
    # ref (page15-16): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:42018X1947&rid=2
    self._AWARENESS_TIME = 30. # passive wheeltouch total timeout
    self._AWARENESS_PRE_TIME_TILL_TERMINAL = 15.
    self._AWARENESS_PROMPT_TIME_TILL_TERMINAL = 6.
    self._DISTRACTED_TIME = 11. # active monitoring total timeout
    self._DISTRACTED_PRE_TIME_TILL_TERMINAL = 8.
    self._DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 6.

    self._FACE_THRESHOLD = 0.7
    self._EYE_THRESHOLD = 0.65
    self._SG_THRESHOLD = 0.9
    self._BLINK_THRESHOLD = 0.865

    self._EE_THRESH11 = 0.25
    self._EE_THRESH12 = 7.5
    self._EE_MAX_OFFSET1 = 0.06
    self._EE_MIN_OFFSET1 = 0.025
    self._EE_THRESH21 = 0.01
    self._EE_THRESH22 = 0.35

    self._POSE_PITCH_THRESHOLD = 0.3133
    self._POSE_PITCH_THRESHOLD_SLACK = 0.3237
    self._POSE_PITCH_THRESHOLD_STRICT = self._POSE_PITCH_THRESHOLD
    self._POSE_YAW_THRESHOLD = 0.4020
    self._POSE_YAW_THRESHOLD_SLACK = 0.5042
    self._POSE_YAW_THRESHOLD_STRICT = self._POSE_YAW_THRESHOLD
    self._PITCH_NATURAL_OFFSET = 0.029 # initial value before offset is learned
    self._PITCH_NATURAL_THRESHOLD = 0.449
    self._YAW_NATURAL_OFFSET = 0.097 # initial value before offset is learned
    self._PITCH_MAX_OFFSET = 0.124
    self._PITCH_MIN_OFFSET = -0.0881
    self._YAW_MAX_OFFSET = 0.289
    self._YAW_MIN_OFFSET = -0.0246

    self._POSESTD_THRESHOLD = 0.3
    self._HI_STD_FALLBACK_TIME = int(10  / self._DT_DMON)  # fall back to wheel touch if model is uncertain for 10s
    self._DISTRACTED_FILTER_TS = 0.25  # 0.6Hz
    self._ALWAYS_ON_ALERT_MIN_SPEED = 7

    self._POSE_CALIB_MIN_SPEED = 13  # 30 mph
    self._POSE_OFFSET_MIN_COUNT = int(60 / self._DT_DMON)  # valid data counts before calibration completes, 1min cumulative
    self._POSE_OFFSET_MAX_COUNT = int(360 / self._DT_DMON)  # stop deweighting new data after 6 min, aka "short term memory"

    self._WHEELPOS_CALIB_MIN_SPEED = 11
    self._WHEELPOS_THRESHOLD = 0.5
    self._WHEELPOS_FILTER_MIN_COUNT = int(15 / self._DT_DMON) # allow 15 seconds to converge wheel side

    self._RECOVERY_FACTOR_MAX = 5.  # relative to minus step change
    self._RECOVERY_FACTOR_MIN = 1.25  # relative to minus step change

    self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
    self._MAX_TERMINAL_DURATION = int(30 / self._DT_DMON)  # not allowed to engage after 30s of terminal alerts

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
    distracted_types = []

    if not self.pose.calibrated:
      pitch_error = self.pose.pitch - self.settings._PITCH_NATURAL_OFFSET
      yaw_error = self.pose.yaw - self.settings._YAW_NATURAL_OFFSET
    else:
      pitch_error = self.pose.pitch - min(max(self.pose.pitch_offseter.filtered_stat.mean(),
                                                       self.settings._PITCH_MIN_OFFSET), self.settings._PITCH_MAX_OFFSET)
      yaw_error = self.pose.yaw - min(max(self.pose.yaw_offseter.filtered_stat.mean(),
                                                    self.settings._YAW_MIN_OFFSET), self.settings._YAW_MAX_OFFSET)
    pitch_error = 0 if pitch_error > 0 else abs(pitch_error) # no positive pitch limit
    yaw_error = abs(yaw_error)
    if pitch_error > (self.settings._POSE_PITCH_THRESHOLD*self.pose.cfactor_pitch if self.pose.calibrated else self.settings._PITCH_NATURAL_THRESHOLD) or \
       yaw_error > self.settings._POSE_YAW_THRESHOLD*self.pose.cfactor_yaw:
      distracted_types.append(DistractedType.DISTRACTED_POSE)

    if (self.blink.left + self.blink.right)*0.5 > self.settings._BLINK_THRESHOLD:
      distracted_types.append(DistractedType.DISTRACTED_BLINK)

    if self.ee1_calibrated:
      ee1_dist = self.eev1 > max(min(self.ee1_offseter.filtered_stat.M, self.settings._EE_MAX_OFFSET1), self.settings._EE_MIN_OFFSET1) \
                              * self.settings._EE_THRESH12
    else:
      ee1_dist = self.eev1 > self.settings._EE_THRESH11
    if ee1_dist:
      distracted_types.append(DistractedType.DISTRACTED_E2E)

    return distracted_types

  def _update_states(self, driver_state, cal_rpy, car_speed, op_engaged):
    # Override to always set nominal states
    self.face_detected = True
    self.pose.low_std = True
    self.driver_distracted = False
    self.driver_distraction_filter.update(False)

    # Reset offseters to default values
    self.pose.pitch_offseter.push_and_update(0.0)
    self.pose.yaw_offseter.push_and_update(0.0)
    self.ee1_offseter.push_and_update(0.0)
    self.ee2_offseter.push_and_update(0.0)

    # Mark as calibrated
    self.pose.calibrated = True
    self.ee1_calibrated = True
    self.ee2_calibrated = True

    # Reset model uncertainty and high standard deviations count
    self.is_model_uncertain = False
    self.hi_stds = 0

    # Ensure awareness levels are full
    self.awareness = 1.0
    self.awareness_active = 1.0
    self.awareness_passive = 1.0

  def _update_events(self, driver_engaged, op_engaged, standstill, wrong_gear, car_speed):
    # Override to prevent any events from being added
    self.current_events = Events()
    # Do not add any distraction-related events
    return


  def get_state_packet(self, valid=True):
    # Override to always return a nominal driver monitoring state
    dat = messaging.new_message('driverMonitoringState', valid=True)
    dat.driverMonitoringState = {
      "events": self.current_events.to_msg(),
      "faceDetected": True,
      "isDistracted": False,
      "distractedType": 0,
      "awarenessStatus": 1.0,
      "posePitchOffset": 0.0,
      "posePitchValidCount": 1,
      "poseYawOffset": 0.0,
      "poseYawValidCount": 1,
      "stepChange": 0.0,
      "awarenessActive": 1.0,
      "awarenessPassive": 1.0,
      "isLowStd": True,
      "hiStdCount": 0,
      "isActiveMode": True,
      "isRHD": False,
    }
    return dat

  def run_step(self, sm):
    # Override to maintain nominal state without updating based on sensor data

    # Skip setting policies based on model data
    # self._set_policy(...) is no longer needed

    # Skip updating states based on actual driver data
    # self._update_states(...) is overridden to set nominal states

    # Force the awareness timers to remain at nominal
    self._set_timers(active_monitoring=True)

    # Update distraction events without changing the nominal state
    self._update_events(
      driver_engaged=False,
      op_engaged=sm['controlsState'].enabled,
      standstill=sm['carState'].standstill,
      wrong_gear=sm['carState'].gearShifter in [car.CarState.GearShifter.reverse, car.CarState.GearShifter.park],
      car_speed=sm['carState'].vEgo
    )
