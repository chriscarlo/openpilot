#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
from unittest.mock import MagicMock, patch

# Local import of the controller under test
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import (
  VisionTurnController,
  curvature_to_speed,
)


@dataclass
class Step:
  # One simulation step of inputs
  curvature: float           # model curvature (1/m)
  confidence: float          # model vision confidence [0..1]
  lead_d_rel_m: Optional[float] = None  # if provided, simulates a lead at this distance


def _mk_sm(curvature: float, v_pred: float, confidence: float, lead_d_rel_m: Optional[float]):
  """Create a minimal SM stub with modelV2 and optional radarState.leadOne."""
  model = SimpleNamespace(
    orientationRate=SimpleNamespace(z=[curvature] * 33),
    velocity=SimpleNamespace(x=[v_pred] * 33),
    laneLineProbs=[confidence] * 4,
  )

  # Optional simple radarState lead stub
  if lead_d_rel_m is not None:
    lead_one = SimpleNamespace(status=True, dRel=float(lead_d_rel_m))
    radar_state = SimpleNamespace(leadOne=lead_one)
    valid = {'modelV2': True, 'radarState': True}
  else:
    radar_state = None
    valid = {'modelV2': True}

  class SM:
    def __init__(self, model, radar_state, valid):
      self.valid = valid
      self._data = {
        'modelV2': model,
        'carState': SimpleNamespace(gasPressed=False),
      }
      if radar_state is not None:
        self._data['radarState'] = radar_state
    def __getitem__(self, key):
      return self._data.get(key)

  return SM(model, radar_state, valid)


def mk_vtsc_with_params(
  aggressiveness: float = 1.0,
  alpha: float = 0.3,
  hysteresis: float = 0.2,
  safety_bias: float = 0.1,
) -> VisionTurnController:
  """Instantiate VisionTurnController with Params patched to specified values."""
  class MockCP:  # minimal car params
    pass
  with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
    mp = MagicMock()
    mp.get_bool.return_value = True
    def _get(key: str):
      if key.endswith('Aggressiveness'):
        return str(aggressiveness).encode()
      if key.endswith('FilterAlpha'):
        return str(alpha).encode()
      if key.endswith('HysteresisThreshold'):
        return str(hysteresis).encode()
      if key.endswith('SafetyBias'):
        return str(safety_bias).encode()
      return None
    mp.get.side_effect = _get
    MockParams.return_value = mp
    return VisionTurnController(MockCP())


def simulate_sequence(
  steps: Iterable[Step],
  vtsc: Optional[VisionTurnController] = None,
  v0_mps: float = 25.0,
  v_cruise_mps: float = 30.0,
  dt: float = 0.05,
) -> Dict[str, Any]:
  """Run a simple time-stepped simulation and return the final debug snapshot.

  - Uses VTSC.update on each step with a synthetic clock.
  - Integrates acceleration to update v_ego.
  - Returns the controller snapshot_debug_state() from the final step.
  """
  ctrl = vtsc or mk_vtsc_with_params()
  v_ego = float(v0_mps)
  a_ego = 0.0
  t = 0.0

  for st in steps:
    sm = _mk_sm(st.curvature, v_ego, st.confidence, st.lead_d_rel_m)
    # Patch time used inside controller to advance deterministically
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: t), \
         patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: t):
      ctrl.update(sm, True, v_ego, a_ego, v_cruise_mps)

    # Integrate acceleration to update speed for next step
    a_cmd = float(ctrl.a_target)
    v_ego = max(0.0, v_ego + a_cmd * dt)
    a_ego = a_cmd
    t += dt

  # Final snapshot for assertions; augment with last a_target for sign checks
  snap = ctrl.snapshot_debug_state() or {}
  try:
    snap['a_last'] = float(ctrl.a_target)
  except Exception:
    snap['a_last'] = 0.0
  return snap
