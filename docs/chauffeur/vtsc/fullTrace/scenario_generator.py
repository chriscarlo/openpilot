# -*- coding: utf-8 -*-
"""
Synthetic scenario generator for VTSC tests.

Produces deterministic sequences (modelV2-ish stubs + (v_ego, a_ego, dt)) that
exercise VTSC gating paths without modifying the controller.

Usage (tests import):
  from docs.chauffeur.vtsc.fullTrace.scenario_generator import (
      ScenarioStep, scenario_straight_long_visibility, scenario_gradual_occlusion crest, run_sequence
  )
"""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Iterator, List, Optional, Tuple

from opendbc.car.common.conversions import Conversions as CV
from unittest.mock import patch

# Controller to call update() directly in tests
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController

@dataclass
class ScenarioStep:
  curvature: float      # representative curvature (1/m) across the horizon
  confidence: float     # path confidence [0..1]
  v_ego: float          # m/s
  a_ego: float = 0.0    # m/s^2
  dt: float = 0.05      # seconds
  lead_d_rel_m: Optional[float] = None  # reserved; not used by VTSC here

def _mk_sm(curvature: float, v_pred: float, confidence: float):
  """Minimal modelV2 stub with evenly-filled arrays of length 33."""
  horizon_len = 33
  model = SimpleNamespace(
    orientationRate=SimpleNamespace(z=[float(curvature)] * horizon_len),
    velocity=SimpleNamespace(x=[float(v_pred)] * horizon_len),
    laneLineProbs=[float(confidence)] * 4,
  )
  class SM:
    def __init__(self, m):
      self.valid = {'modelV2': True}
      self._data = {'modelV2': m, 'carState': SimpleNamespace(gasPressed=False)}
    def __getitem__(self, k):
      return self._data.get(k)
  return SM(model)

# ---------- canned scenarios ---------- #

def scenario_straight_long_visibility(n: int = 120, v_mps: float = 30.0, conf: float = 0.95) -> List[ScenarioStep]:
  """Straight road, high confidence, long vis => fail-open should be active or occlusion not winning."""
  # curvature well below FREEWAY_CURV_EPS
  curv = 1e-6
  return [ScenarioStep(curvature=curv, confidence=conf, v_ego=v_mps, a_ego=0.0, dt=0.05) for _ in range(n)]

def scenario_gentle_turn_no_occlusion(n: int = 120, v_mps: float = 22.5, conf: float = 0.92) -> List[ScenarioStep]:
  """Mild curvature without occlusion toggles."""
  curv = 0.0020
  return [ScenarioStep(curvature=curv, confidence=conf, v_ego=v_mps, a_ego=0.0, dt=0.05) for _ in range(n)]

def scenario_gradual_occlusion_crest_then_recover(n: int = 160, v_mps: float = 25.0) -> List[ScenarioStep]:
  """Confidence drifts below bad threshold, stays a bit, then recovers above good."""
  steps: List[ScenarioStep] = []
  # Start good, then drift down, hold, recover
  up = 40; down = 40; hold = 40; rec = 40
  def lin(a,b,k):
    return a + (b-a)*k
  # gentle curve to make PSI meaningful
  curv = 0.0030
  # Good -> Bad
  for i in range(up):
    steps.append(ScenarioStep(curvature=curv, confidence=lin(0.9, 0.60, i/(up-1)), v_ego=v_mps))
  # hold bad
  for _ in range(hold):
    steps.append(ScenarioStep(curvature=curv, confidence=0.60, v_ego=v_mps))
  # Bad -> Good
  for i in range(rec):
    steps.append(ScenarioStep(curvature=curv, confidence=lin(0.60, 0.9, i/(rec-1)), v_ego=v_mps))
  return steps[:n]

# ---------- runner ---------- #

def run_sequence(steps: Iterable[ScenarioStep],
                 vtsc: Optional[VisionTurnController] = None,
                 v_cruise_mps: float = 40.0) -> List[dict]:
  """
  Feed a synthetic sequence through VTSC.update(). Returns list of snapshot dicts
  (each is VisionTurnController.snapshot_debug_state()) for assertions.
  Uses a deterministic synthetic clock via patch to stabilize dwell timers.
  """
  ctrl = vtsc or VisionTurnController(type("CP", (), {})())
  ctrl._is_enabled = True
  ctrl._op_enabled = True
  ctrl._gas_pressed = False
  # Stub Params so unit tests don't require device keys
  class _StubParams:
    def __init__(self):
      self._d = {}
    def get(self, k: str):
      return self._d.get(k, None)
    def get_bool(self, k: str) -> bool:
      return False
  try:
    ctrl._params = _StubParams()
    ctrl._mem_params = ctrl._params
  except Exception:
    pass

  out: List[dict] = []
  t = 0.0
  for st in steps:
    sm = _mk_sm(st.curvature, st.v_ego, st.confidence)
    # Patch controller time with deterministic clock
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: t), \
         patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: t):
      ctrl.update(sm, True, float(st.v_ego), float(st.a_ego), float(v_cruise_mps))
    snap = ctrl.snapshot_debug_state() or {}
    out.append(snap)
    t += float(st.dt)
  return out
