"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params


def get_lat_delay(params: Params, stock_lat_delay: float) -> float:
  # UI semantics:
  # - When `LagdToggle` (Live Learning Steer Delay) is ON, use the live learner output
  #   published on the `liveDelay` stream.
  # - When it is OFF, use the user-selected software delay (`LagdToggleDelay`) + vehicle
  #   actuator delay, which is cached to `LagdValueCache` by `LagdToggle.update()` in
  #   `selfdrive/locationd/lagd.py`.
  #
  # The cache lives in Params so it can be shared by controlsd/modeld/torqued without
  # re-reading CarParams in each process.
  if not params.get_bool("LagdToggle"):
    try:
      cached = float(params.get("LagdValueCache", return_default=True))
      # Fallback to the liveDelay stream if the cache is missing/uninitialized.
      if cached > 0.05:
        return cached
    except Exception:
      pass

  return stock_lat_delay
