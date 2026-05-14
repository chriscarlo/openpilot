import os
from typing import cast

from openpilot.system.hardware.base import HardwareBase
from openpilot.system.hardware.pc.hardware import Pc

TICI = os.path.isfile('/TICI')
AGNOS = os.path.isfile('/AGNOS')
PC = not TICI


if TICI:
  from openpilot.system.hardware.tici.hardware import Tici
  HARDWARE = cast(HardwareBase, Tici())
else:
  HARDWARE = cast(HardwareBase, Pc())
