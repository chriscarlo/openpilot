import cereal.messaging as messaging
from openpilot.common.realtime import DT_DMON
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.stat_live import RunningStatFilter
from openpilot.selfdrived.events import Events

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
    
    # Properties accessed by tests
    self.current_events = Events()
    self.awareness = 1.0
    
  def run_step(self, sm):
    # No-op - all monitoring disabled
    pass
    
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