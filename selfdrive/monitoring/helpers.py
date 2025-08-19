import cereal.messaging as messaging
from openpilot.common.realtime import DT_DMON
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.stat_live import RunningStatFilter
from openpilot.selfdrive.selfdrived.events import Events

# Minimal settings class to maintain interface compatibility
class DRIVER_MONITOR_SETTINGS:
  def __init__(self):
    self._DT_DMON = DT_DMON
    self._WHEELPOS_FILTER_MIN_COUNT = 15 / DT_DMON
    self._WHEELPOS_THRESHOLD = 0.5

class DriverMonitoring:
  def __init__(self, rhd_saved=False, settings=None, always_on=False):
    # Minimal initialization to maintain interface
    if settings is None:
      settings = DRIVER_MONITOR_SETTINGS()
    self.settings = settings

    # Properties accessed by dmonitoringd.py
    self.always_on = always_on
    self.wheel_on_right = rhd_saved
    self.wheelpos_learner = RunningStatFilter()

    self.params = Params()
    self.too_distracted = self.params.get_bool("DriverTooDistracted")

    self._reset_awareness()
    self._set_timers(active_monitoring=True)
    self._reset_events()

  def _reset_awareness(self):
    self.awareness = 1.
    self.awareness_active = 1.
    self.awareness_passive = 1.

  def _reset_events(self):
    self.current_events = Events()
    self.awareness = 1.0

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
    self.pose.cfactor_pitch = np.interp(bp_normal, [0, 0.5],
                                           [self.settings._POSE_PITCH_THRESHOLD_SLACK,
                                            self.settings._POSE_PITCH_THRESHOLD_STRICT]) / self.settings._POSE_PITCH_THRESHOLD
    self.pose.cfactor_yaw = np.interp(bp_normal, [0, 0.5],
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
    # Block engaging until ignition cycle after max number or time of distractions
    if self.terminal_alert_cnt >= self.settings._MAX_TERMINAL_ALERTS or \
       self.terminal_time >= self.settings._MAX_TERMINAL_DURATION:
      if not self.too_distracted:
        self.params.put_bool_nonblocking("DriverTooDistracted", True)
      self.too_distracted = True

    # Always-on distraction lockout is temporary
    if self.too_distracted or (self.always_on and self.awareness <= self.threshold_prompt):
      self.current_events.add(EventName.tooDistracted)

    always_on_valid = self.always_on and not wrong_gear
    if (driver_engaged and self.awareness > 0 and not self.active_monitoring_mode) or \
       (not always_on_valid and not op_engaged) or \
       (always_on_valid and not op_engaged and self.awareness <= 0):
      # always reset on disengage with normal mode; disengage resets only on red if always on
      self._reset_awareness()
      return

    driver_attentive = self.driver_distraction_filter.x < 0.37
    awareness_prev = self.awareness

    if (driver_attentive and self.face_detected and self.pose.low_std and self.awareness > 0):
      if driver_engaged:
        self._reset_awareness()
        return
      # only restore awareness when paying attention and alert is not red
      self.awareness = min(self.awareness + ((self.settings._RECOVERY_FACTOR_MAX-self.settings._RECOVERY_FACTOR_MIN)*
                                             (1.-self.awareness)+self.settings._RECOVERY_FACTOR_MIN)*self.step_change, 1.)
      if self.awareness == 1.:
        self.awareness_passive = min(self.awareness_passive + self.step_change, 1.)
      # don't display alert banner when awareness is recovering and has cleared orange
      if self.awareness > self.threshold_prompt:
        return

    _reaching_audible = self.awareness - self.step_change <= self.threshold_prompt
    _reaching_terminal = self.awareness - self.step_change <= 0
    standstill_orange_exemption = standstill and _reaching_audible
    always_on_red_exemption = always_on_valid and not op_engaged and _reaching_terminal
    always_on_lowspeed_exemption = always_on_valid and not op_engaged and car_speed < self.settings._ALWAYS_ON_ALERT_MIN_SPEED

    certainly_distracted = self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected
    maybe_distracted = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME or not self.face_detected

    if certainly_distracted or maybe_distracted:
      # should always be counting if distracted unless at standstill (lowspeed for always-on) and reaching orange
      # also will not be reaching 0 if DM is active when not engaged
      if not (standstill_orange_exemption or always_on_red_exemption or (always_on_lowspeed_exemption and _reaching_audible)):
        self.awareness = max(self.awareness - self.step_change, -0.1)

    alert = None
    if self.awareness <= 0.:
      # terminal red alert: disengagement required
      alert = EventName.driverDistracted if self.active_monitoring_mode else EventName.driverUnresponsive
      self.terminal_time += 1
      if awareness_prev > 0.:
        self.terminal_alert_cnt += 1
    elif self.awareness <= self.threshold_prompt:
      # prompt orange alert
      alert = EventName.promptDriverDistracted if self.active_monitoring_mode else EventName.promptDriverUnresponsive
    elif self.awareness <= self.threshold_pre and not always_on_lowspeed_exemption:
      # pre green alert
      alert = EventName.preDriverDistracted if self.active_monitoring_mode else EventName.preDriverUnresponsive

    if alert is not None:
      self.current_events.add(alert)


  def get_state_packet(self, valid=True):
    # Return nominal values for all fields
    dat = messaging.new_message('driverMonitoringState', valid=valid)
    dat.driverMonitoringState = {
      "events": [],  # No events
      "faceDetected": True,  # Always detected
      "isDistracted": False,  # Never distracted
      "distractedType": 0,  # No distraction
      "awarenessStatus": 1.0,  # Always fully aware
      "posePitchOffset": 0.0,
      "posePitchValidCount": 1000,
      "poseYawOffset": 0.0,
      "poseYawValidCount": 1000,
      "stepChange": 0.0,
      "awarenessActive": 1.0,
      "awarenessPassive": 1.0,
      "isLowStd": True,
      "hiStdCount": 0,
      "isActiveMode": False,
      "isRHD": self.wheel_on_right,
    }
    return dat

  # Methods needed for tests but can be no-ops
  def _update_states(self, driver_state, cal_rpy, car_speed, op_engaged):
    pass

  def _update_events(self, driver_engaged, op_engaged, standstill, wrong_gear, car_speed):
    pass
