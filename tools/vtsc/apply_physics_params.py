#!/usr/bin/env python3
"""Install one complete VTSC sigmoid into persistent Params while offroad."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping

from openpilot.common.params import Params


PHYSICS_KEYS = {
  "VisionTurnSpeedControlPhysicsAmplitude": (-5.0, -0.2),
  "VisionTurnSpeedControlPhysicsSteepness": (-100_000.0, -100.0),
  "VisionTurnSpeedControlPhysicsCenter": (1.0e-5, 0.1),
  "VisionTurnSpeedControlPhysicsBaseline": (2.0, 6.5),
  "VisionTurnSpeedControlPhysicsMinLatAccel": (1.0, 3.0),
  "VisionTurnSpeedControlPhysicsMaxLatAccel": (2.0, 5.5),
}


def install_physics_params(params: Params, requested: Mapping[str, float]) -> dict[str, float]:
  """Write and verify all six values, rolling back the complete set on failure."""
  if not params.get_bool("IsOffroad"):
    raise RuntimeError("refusing to change VTSC physics Params unless the tici is offroad")
  if set(requested) != set(PHYSICS_KEYS):
    raise ValueError("the VTSC physics update must contain exactly all six parameters")

  normalized: dict[str, float] = {}
  for key, (lower, upper) in PHYSICS_KEYS.items():
    value = float(requested[key])
    if not math.isfinite(value) or not lower <= value <= upper:
      raise ValueError(f"{key}={value!r} is outside [{lower}, {upper}]")
    normalized[key] = value
  if normalized["VisionTurnSpeedControlPhysicsMinLatAccel"] > normalized["VisionTurnSpeedControlPhysicsMaxLatAccel"]:
    raise ValueError("minimum lateral acceleration exceeds maximum lateral acceleration")

  previous = {key: params.get(key) for key in PHYSICS_KEYS}
  try:
    for key, value in normalized.items():
      params.put(key, value)
    actual = {key: float(params.get(key)) for key in PHYSICS_KEYS}
    mismatches = {
      key: (normalized[key], actual[key])
      for key in PHYSICS_KEYS
      if not math.isclose(normalized[key], actual[key], rel_tol=0.0, abs_tol=1.0e-12)
    }
    if mismatches:
      raise RuntimeError(f"VTSC physics Params failed read-back verification: {mismatches}")
    return actual
  except Exception:
    for key, value in previous.items():
      if value is None:
        params.remove(key)
      else:
        params.put(key, value)
    raise


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--amplitude", required=True, type=float)
  parser.add_argument("--steepness", required=True, type=float)
  parser.add_argument("--center", required=True, type=float)
  parser.add_argument("--baseline", required=True, type=float)
  parser.add_argument("--min-lat", required=True, type=float)
  parser.add_argument("--max-lat", required=True, type=float)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  requested = {
    "VisionTurnSpeedControlPhysicsAmplitude": args.amplitude,
    "VisionTurnSpeedControlPhysicsSteepness": args.steepness,
    "VisionTurnSpeedControlPhysicsCenter": args.center,
    "VisionTurnSpeedControlPhysicsBaseline": args.baseline,
    "VisionTurnSpeedControlPhysicsMinLatAccel": args.min_lat,
    "VisionTurnSpeedControlPhysicsMaxLatAccel": args.max_lat,
  }
  actual = install_physics_params(Params(), requested)
  print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
  main()
