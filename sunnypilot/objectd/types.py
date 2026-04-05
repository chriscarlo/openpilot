from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Detection:
  class_name: str
  confidence: float
  x_min: float
  y_min: float
  x_max: float
  y_max: float
  on_path: bool = False
  distance_m: float = math.inf
  footpoint_x: float = 0.0
  footpoint_y: float = 0.0

  def with_path_state(self, *, on_path: bool, distance_m: float, footpoint_x: float, footpoint_y: float) -> "Detection":
    return replace(self, on_path=on_path, distance_m=distance_m, footpoint_x=footpoint_x, footpoint_y=footpoint_y)
