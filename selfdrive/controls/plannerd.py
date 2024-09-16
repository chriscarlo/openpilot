#!/usr/bin/env python3
from cereal import car
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
import cereal.messaging as messaging

from openpilot.selfdrive.frogpilot.controls.lib.frogpilot_variables import FrogPilotVariables

def publish_ui_plan(sm, pm, longitudinal_planner):
  ui_send = messaging.new_message('uiPlan')
  ui_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'modelV2'])
  uiPlan = ui_send.uiPlan
  uiPlan.frameId = sm['modelV2'].frameId
  uiPlan.position.x = list(sm['modelV2'].position.x)
  uiPlan.position.y = list(sm['modelV2'].position.y)
  uiPlan.position.z = list(sm['modelV2'].position.z)
  uiPlan.accel = longitudinal_planner.a_desired_trajectory_full.tolist()
  pm.send('uiPlan', ui_send)

def plannerd_thread():
  config_realtime_process(5, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  with car.CarParams.from_bytes(params.get("CarParams", block=True)) as msg:
    CP = msg
  cloudlog.info("plannerd got CarParams: %s", CP.carName)

  longitudinal_planner = LongitudinalPlanner(CP)
  pm = messaging.PubMaster(['longitudinalPlan', 'uiPlan'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'radarState', 'modelV2', 'frogpilotCarControl', 'frogpilotCarState', 'frogpilotPlan'],
                           poll='modelV2', ignore_avg_freq=['radarState'])

  # FrogPilot variables
  frogpilot_toggles = FrogPilotVariables.toggles
  FrogPilotVariables.update_frogpilot_params()

  clairvoyant_driver = frogpilot_toggles.clairvoyant_driver
  clairvoyant_driver_v2 = frogpilot_toggles.clairvoyant_driver_v2
  tomb_raider = frogpilot_toggles.tomb_raider
  e2e_longitudinal_model = clairvoyant_driver or clairvoyant_driver_v2 or frogpilot_toggles.secretgoodopenpilot_model or tomb_raider
  radarless_model = frogpilot_toggles.radarless_model

  update_toggles = False

  while True:
    sm.update()
    if sm.updated['modelV2']:
      longitudinal_planner.update(clairvoyant_driver, clairvoyant_driver_v2, e2e_longitudinal_model, radarless_model, sm, tomb_raider, frogpilot_toggles)
      longitudinal_planner.publish(e2e_longitudinal_model, sm, pm)
      publish_ui_plan(sm, pm, longitudinal_planner)

    # Update FrogPilot parameters
    if FrogPilotVariables.toggles_updated:
      update_toggles = True
    elif update_toggles:
      FrogPilotVariables.update_frogpilot_params()
      update_toggles = False

def main():
  plannerd_thread()


if __name__ == "__main__":
  main()
