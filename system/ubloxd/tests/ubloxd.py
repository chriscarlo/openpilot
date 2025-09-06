#!/usr/bin/env python3
# type: ignore

# from openpilot.selfdrive.locationd.test import ublox  # Module no longer exists
import struct

baudrate = 460800
rate = 100  # send new data every 100ms


def configure_ublox(dev) -> None:
  """
  Deprecated test helper. The original ublox module no longer exists.
  Keep a reference payload structure for context, but do not execute.
  """
  # NOTE: Disabled to avoid import/runtime errors in test discovery.
  # If ublox support returns, re-enable and update accordingly.
  if False:  # pragma: no cover
    payload = struct.pack(
      '<HHIBBBBBBBBBBH6BBB2BH4B3BB',
      0, (1 << 10), 0, 0, 0,
      0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0,
      0, 0, 0, 0,
    )
    # dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAVX5, payload)
    # dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAV5)
    # dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAVX5)
    # dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_ODO)
    # dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_ITMF)
  return


if __name__ == "__main__":
  class Device:
    def write(self, s):
      d = '"{}"s'.format(''.join(f'\\x{b:02X}' for b in s))
      print(f"    if (!send_with_ack({d})) continue;")

  # dev = ublox.UBlox(Device(), baudrate=baudrate)  # ublox module no longer exists
  # configure_ublox(dev)
