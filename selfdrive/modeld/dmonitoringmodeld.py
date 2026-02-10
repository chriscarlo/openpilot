#!/usr/bin/env python3
"""
Privacy-first stub for driver monitoring model output.

This branch intentionally disables any meaningful driver monitoring. The original
`dmonitoringmodeld` runs a neural network on the driver camera stream and publishes
`driverStateV2`. For privacy, we publish a stable, nominal `driverStateV2` packet
at 20Hz without subscribing to the driver camera or running inference.
"""

import time

from cereal import messaging
from cereal.messaging import PubMaster
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog


PROCESS_NAME = "selfdrive.modeld.dmonitoringmodeld"


def _fill_driver_data(builder) -> None:
  # Match `cereal/log.capnp` DriverStateV2.DriverData field sizes.
  builder.faceOrientation = [0.0, 0.0, 0.0]
  builder.faceOrientationStd = [0.0, 0.0, 0.0]
  builder.facePosition = [0.0, 0.0]
  builder.facePositionStd = [0.0, 0.0]
  builder.faceProb = 1.0
  builder.leftEyeProb = 1.0
  builder.rightEyeProb = 1.0
  builder.leftBlinkProb = 0.0
  builder.rightBlinkProb = 0.0
  builder.sunglassesProb = 0.0
  builder.occludedProb = 0.0
  builder.readyProb = [1.0, 1.0, 1.0, 1.0]
  builder.notReadyProb = [0.0, 0.0]


def _get_driverstate_packet(frame_id: int, exec_time: float):
  msg = messaging.new_message("driverStateV2", valid=True)
  ds = msg.driverStateV2
  ds.frameId = frame_id
  ds.modelExecutionTime = float(exec_time)
  ds.gpuExecutionTime = 0.0
  ds.poorVisionProb = 0.0
  ds.wheelOnRightProb = 0.0
  ds.rawPredictions = b""
  _fill_driver_data(ds.leftDriverData)
  _fill_driver_data(ds.rightDriverData)
  return msg


def main() -> None:
  # Keep scheduling sane, but the work is intentionally minimal.
  config_realtime_process(7, 5)
  pm = PubMaster(["driverStateV2"])

  cloudlog.warning("dmonitoringmodeld privacy stub active (no camera, no ML inference)")

  frame_id = 0
  period = 1.0 / 20.0
  next_t = time.monotonic()

  while True:
    t0 = time.perf_counter()
    pm.send("driverStateV2", _get_driverstate_packet(frame_id, time.perf_counter() - t0))
    frame_id = (frame_id + 1) & 0xFFFFFFFF

    next_t += period
    sleep_t = next_t - time.monotonic()
    if sleep_t > 0:
      time.sleep(sleep_t)
    else:
      # If we're late (e.g., system load), resync rather than accumulating drift.
      next_t = time.monotonic()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("dmonitoringmodeld got SIGINT")
