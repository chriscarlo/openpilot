from types import MethodType, SimpleNamespace

import numpy as np

from cereal import messaging
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner


class _SubMaster(dict):
  def __init__(self, *, model_time_ns: int, radar_time_ns: int, service_times=None) -> None:
    radar = messaging.new_message("radarState").radarState
    super().__init__({"radarState": radar})
    self.logMonoTime = {"modelV2": model_time_ns, "radarState": radar_time_ns}
    self.logMonoTime.update(service_times or {})

  def all_checks(self, service_list=None) -> bool:
    return True


class _PubMaster:
  def __init__(self) -> None:
    self.messages = {}

  def send(self, service: str, msg) -> None:
    self.messages[service] = msg.as_reader()


def _planner(*, effective_cruise_mps: float = 23.246079) -> LongitudinalPlanner:
  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.mpc = SimpleNamespace(solve_time=0.001, source="cruise")
  planner.v_desired_trajectory = np.array([1.0])
  planner.a_desired_trajectory = np.array([0.0])
  planner.j_desired_trajectory = np.array([0.0])
  planner.fcw = False
  planner.output_a_target = 0.2
  planner.output_should_stop = False
  planner.allow_throttle = True
  planner.effective_v_cruise_mps = effective_cruise_mps
  return planner


def _publish_empty_longitudinal_plan_sp(self, sm, pm) -> None:
  pm.send("longitudinalPlanSP", messaging.new_message("longitudinalPlanSP"))


def test_publish_preserves_replay_inputs_and_processing_delay_units() -> None:
  planner = _planner()
  planner.publish_longitudinal_plan_sp = MethodType(lambda self, sm, pm: None, planner)

  model_time_ns = 123_456_789
  radar_time_ns = 120_000_000
  sm = _SubMaster(model_time_ns=model_time_ns, radar_time_ns=radar_time_ns)
  pm = _PubMaster()

  planner.publish(sm, pm)

  msg = pm.messages["longitudinalPlan"]
  plan = msg.longitudinalPlan
  assert plan.modelMonoTime == model_time_ns
  assert plan.radarStateMonoTimeDEPRECATED == radar_time_ns
  assert plan.vCruiseDEPRECATED == np.float32(planner.effective_v_cruise_mps)
  assert plan.processingDelay == np.float32((msg.logMonoTime - model_time_ns) / 1e9)


def test_longitudinal_plan_sp_replay_contract_records_exact_submaster_clocks() -> None:
  planner = _planner(effective_cruise_mps=27.125)
  planner.publish_longitudinal_plan_sp = MethodType(_publish_empty_longitudinal_plan_sp, planner)

  service_times = {
    "carState": 5_000_000_001,
    "carControl": 5_000_000_002,
    "controlsState": 5_000_000_003,
    "selfdriveState": 5_000_000_004,
    "liveParameters": 5_000_000_005,
    "liveMapDataSP": 5_000_000_006,
    "carStateSP": 5_000_000_007,
    "rtiStateSP": 5_000_000_008,
    "objectHazardStateSP": 5_000_000_009,
    "gpsLocation": 5_000_000_010,
    "gpsLocationExternal": 5_000_000_011,
  }
  model_time_ns = 5_000_000_012
  radar_time_ns = 5_000_000_013
  sm = _SubMaster(model_time_ns=model_time_ns, radar_time_ns=radar_time_ns, service_times=service_times)
  pm = _PubMaster()

  planner.publish(sm, pm)

  replay = pm.messages["longitudinalPlanSP"].longitudinalPlanSP.replayInputs
  plan = pm.messages["longitudinalPlan"]
  assert replay.valid
  assert replay.version == 1
  assert replay.longitudinalPlanMonoTimeNs == plan.logMonoTime
  assert replay.effectiveCruiseMps == np.float32(planner.effective_v_cruise_mps)
  assert replay.radarStateMonoTimeNs == radar_time_ns
  assert replay.modelV2MonoTimeNs == model_time_ns
  assert replay.carStateMonoTimeNs == service_times["carState"]
  assert replay.carControlMonoTimeNs == service_times["carControl"]
  assert replay.controlsStateMonoTimeNs == service_times["controlsState"]
  assert replay.selfdriveStateMonoTimeNs == service_times["selfdriveState"]
  assert replay.liveParametersMonoTimeNs == service_times["liveParameters"]
  assert replay.liveMapDataSPMonoTimeNs == service_times["liveMapDataSP"]
  assert replay.carStateSPMonoTimeNs == service_times["carStateSP"]
  assert replay.rtiStateSPMonoTimeNs == service_times["rtiStateSP"]
  assert replay.objectHazardStateSPMonoTimeNs == service_times["objectHazardStateSP"]
  assert replay.gpsLocationMonoTimeNs == service_times["gpsLocation"]
  assert replay.gpsLocationExternalMonoTimeNs == service_times["gpsLocationExternal"]


def test_longitudinal_plan_sp_replay_contract_zeros_absent_optional_clocks() -> None:
  planner = _planner()
  planner.publish_longitudinal_plan_sp = MethodType(_publish_empty_longitudinal_plan_sp, planner)
  sm = _SubMaster(model_time_ns=1_000, radar_time_ns=2_000)
  pm = _PubMaster()

  planner.publish(sm, pm)

  replay = pm.messages["longitudinalPlanSP"].longitudinalPlanSP.replayInputs
  assert replay.valid
  assert replay.version == 1
  assert replay.liveMapDataSPMonoTimeNs == 0
  assert replay.carStateSPMonoTimeNs == 0
  assert replay.rtiStateSPMonoTimeNs == 0
  assert replay.objectHazardStateSPMonoTimeNs == 0
  assert replay.gpsLocationMonoTimeNs == 0
  assert replay.gpsLocationExternalMonoTimeNs == 0
