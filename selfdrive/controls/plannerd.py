#!/usr/bin/env python3
import time

from cereal import car
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.planner_lag_debug import (
  SPAN_DRIVER_ASSISTANCE_PUBLISH,
  end_span,
  start_span,
)
import cereal.messaging as messaging


def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("plannerd got CarParams: %s", CP.brand)

  gps_location_service = get_gps_location_service(params)

  ldw = LaneDepartureWarning()
  longitudinal_planner = LongitudinalPlanner(CP)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance', 'longitudinalPlanSP'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'liveParameters', 'radarState', 'modelV2', 'selfdriveState',
                            'liveMapDataSP', 'carStateSP', 'rtiStateSP', gps_location_service],
                           poll='modelV2')

  while True:
    sm.update()
    if sm.updated['modelV2']:
      loop_t0 = time.monotonic()
      longitudinal_planner.planner_lag_debug.begin_cycle(frame=int(sm.frame), model_logmono_ns=int(sm.logMonoTime['modelV2']))
      try:
        longitudinal_planner.update(sm)
        longitudinal_planner.publish(sm, pm)

        driver_assist_span = start_span(SPAN_DRIVER_ASSISTANCE_PUBLISH)
        try:
          ldw.update(sm.frame, sm['modelV2'], sm['carState'], sm['carControl'])
          msg = messaging.new_message('driverAssistance')
          msg.valid = sm.all_checks(['carControl', 'modelV2', 'liveParameters'])
          msg.driverAssistance.leftLaneDeparture = ldw.left
          msg.driverAssistance.rightLaneDeparture = ldw.right
          pm.send('driverAssistance', msg)
        finally:
          end_span(driver_assist_span)
      finally:
        loop_end_s = time.monotonic()
        longitudinal_planner.planner_lag_debug.finish_cycle(
          planner_loop_dt_s=max(0.0, loop_end_s - loop_t0),
          publish_end_s=loop_end_s,
        )


if __name__ == "__main__":
  main()
