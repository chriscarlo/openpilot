#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_response_model import (
  DEFAULT_CRUISE_MAX_ACCEL,
  DEFAULT_CRUISE_MIN_ACCEL,
  build_cruise_response_model,
  clip_cruise_speed_profile,
)


@dataclass
class Step:
  curvature: float                 # 1/m (abs curvature used by VTSC internally)
  confidence: float                # [0..1] laneLineProbs mean
  v_ego: float                     # m/s
  a_ego: float                     # m/s^2
  v_cruise: float                  # m/s (cruise setpoint already in m/s)
  long_active: bool = True
  lead_d_rel_m: Optional[float] = None


class FakeSubMaster:
  """Minimal SubMaster-like shim for unit/integration tests.

  VTSC and the Sunnypilot longitudinal planner expect:
  - `sm.valid` dict for gating some message accesses (notably modelV2/radarState)
  - `sm[...]` to return message-like objects with attributes
  """

  def __init__(self, data: Dict[str, Any], valid: Dict[str, bool],
               alive: Optional[Dict[str, bool]] = None,
               recv_time: Optional[Dict[str, float]] = None):
    self._data = dict(data)
    self.valid = dict(valid)
    self.alive = {k: self.valid.get(k, False) for k in self._data} if alive is None else dict(alive)
    self.recv_time = {k: 0.0 for k in self._data} if recv_time is None else dict(recv_time)

  def __getitem__(self, key: str) -> Any:
    return self._data[key]

  # SubMaster compatibility: publisher helpers sometimes call all_checks(...)
  def all_checks(self, service_list: Optional[list[str]] = None) -> bool:
    return True


def make_model_v2(*, curvature: float, curvature_ahead: float | None = None, v_pred: float, confidence: float, n: int = 33) -> Any:
  """Create a lightweight modelV2-like object with fields VTSC consumes.

  VTSC uses:
  - orientationRate.z (yaw rate rad/s) and velocity.x (m/s) to compute curvature
  - laneLineProbs (mean) as a proxy for "vision confidence"

  The main longitudinal planner may also look at:
  - position.x / velocity.x / acceleration.x (length == ModelConstants.IDX_N)
  - meta.disengagePredictions.gasPressProbs
  """
  n = int(n)
  v_pred = float(max(0.1, v_pred))
  k = float(max(0.0, curvature))
  k_ahead = float(max(0.0, curvature_ahead)) if curvature_ahead is not None else k
  yaw_rate = k * v_pred  # rad/s (since k = yaw_rate / v)
  yaw_rate_ahead = k_ahead * v_pred

  # VTSC expects arrays of at least 3 points; it will cap to N_POINTS internally.
  orientation_rate = SimpleNamespace(z=[yaw_rate] + [yaw_rate_ahead] * (n - 1))
  velocity = SimpleNamespace(x=[v_pred] * n)

  idx_n = int(ModelConstants.IDX_N)
  t_idxs = np.array(ModelConstants.T_IDXS, dtype=float)
  position = SimpleNamespace(x=[float(v_pred * t) for t in t_idxs[:idx_n]])
  acceleration = SimpleNamespace(x=[0.0] * idx_n)

  # Default: "gas not pressed" probabilities ~1.0 (allow throttle)
  disengage_preds = SimpleNamespace(gasPressProbs=[1.0] * 6)
  meta = SimpleNamespace(disengagePredictions=disengage_preds)

  return SimpleNamespace(
    orientationRate=orientation_rate,
    velocity=velocity,
    position=position,
    acceleration=acceleration,
    laneLineProbs=[float(confidence)] * 4,
    meta=meta,
    action=SimpleNamespace(desiredAcceleration=0.0, shouldStop=False),
  )


def make_radar_state(*, lead_d_rel_m: Optional[float]) -> Any:
  if lead_d_rel_m is None:
    return SimpleNamespace(leadOne=SimpleNamespace(status=False, dRel=1e9))
  return SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=float(lead_d_rel_m)))


def make_car_state(*, gas_pressed: bool = False, left_blinker: bool = False, right_blinker: bool = False) -> Any:
  return SimpleNamespace(
    gasPressed=bool(gas_pressed),
    leftBlinker=bool(left_blinker),
    rightBlinker=bool(right_blinker),
  )


def make_car_control(*, long_active: bool = True, pitch_rad: float = 0.0) -> Any:
  # The longitudinal planner reads orientationNED[1] as pitch (radians).
  return SimpleNamespace(longActive=bool(long_active), orientationNED=[0.0, float(pitch_rad), 0.0])


def install_fake_long_mpc(*, module_name: str = 'openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc') -> None:
  """Install a pure-Python fake long MPC module in sys.modules.

  This enables importing `openpilot.selfdrive.controls.lib.longitudinal_planner` in test environments
  where the Acados-generated `c_generated_code/` is not built.
  """

  class FakeLongitudinalMpc:
    def __init__(self, dt: float = 0.05, CP=None):
      self.dt = float(dt)
      self.CP = CP
      self.mode = 'acc'
      self.solve_time = 0.0
      self.crash_cnt = 0
      self.source = 0
      self.v_solution = np.zeros(len(ModelConstants.T_IDXS), dtype=float)
      self.a_solution = np.zeros(len(ModelConstants.T_IDXS), dtype=float)
      self.j_solution = np.zeros(len(ModelConstants.T_IDXS) - 1, dtype=float)
      self._v0 = 0.0
      self._a0 = 0.0
      self.last_v_cruise = None
      # Internal cruise envelope diagnostics (helps debug "VTSC requested X but MPC can't instantly follow")
      self.last_v_lower = None
      self.last_v_upper = None
      self.last_v_cruise_clipped = None

    def set_weights(self, prev_accel_constraint: bool, personality=None) -> None:
      pass

    def set_cur_state(self, v: float, a: float) -> None:
      self._v0 = float(v)
      self._a0 = float(a)

    def get_cruise_response_model(self, v_ego: float, *, actuation_delay_s: float = 0.0,
                                  planner_accel_limits: tuple[float, float] | None = None):
      planner_min = DEFAULT_CRUISE_MIN_ACCEL
      planner_max = DEFAULT_CRUISE_MAX_ACCEL
      if planner_accel_limits is not None:
        planner_min = float(planner_accel_limits[0])
        planner_max = float(planner_accel_limits[1])
      return build_cruise_response_model(
        min_accel_mps2=DEFAULT_CRUISE_MIN_ACCEL,
        max_accel_mps2=DEFAULT_CRUISE_MAX_ACCEL,
        actuation_delay_s=actuation_delay_s,
        planner_output_min_accel_mps2=planner_min,
        planner_output_max_accel_mps2=planner_max,
      )

    def update(self, radar_state, v_cruise: float, x, v, a, j, personality=None) -> None:
      self.last_v_cruise = float(v_cruise)
      t = np.array(ModelConstants.T_IDXS, dtype=float)
      # Mimic the real long MPC "cruise obstacle" envelope clipping (ACC mode) at a high level.
      # This does not attempt to reproduce the full solver; it only exposes the key confounder:
      # v_cruise is clipped by accel/decel envelopes, so a sharp VTSC cap step-down is softened.
      response_model = self.get_cruise_response_model(self._v0)
      v_lower, v_upper, v_cruise_clipped = clip_cruise_speed_profile(
        v_ego=self._v0,
        v_cruise=v_cruise,
        t_idxs=t,
        response_model=response_model,
      )
      self.last_v_lower = v_lower
      self.last_v_upper = v_upper
      self.last_v_cruise_clipped = v_cruise_clipped
      horizon = float(max(1.0, t[-1]))
      # Constant-accel plan to reach v_cruise by horizon; stable and predictable for tests.
      a_cmd = float(np.clip((float(v_cruise) - self._v0) / horizon, -4.0, 2.0))
      self.v_solution = np.maximum(0.0, self._v0 + a_cmd * t)
      self.a_solution = np.full_like(self.v_solution, a_cmd)
      self.j_solution = np.zeros(len(t) - 1, dtype=float)
      self.solve_time = 0.0
      self.crash_cnt = 0
      self.source = 0

  fake = types.ModuleType(module_name)
  fake.LongitudinalMpc = FakeLongitudinalMpc
  fake.T_IDXS = list(ModelConstants.T_IDXS)

  def get_low_speed_launch_follow_max_accel(_v_ego, _lead, _t_follow, base_max_accel: float) -> float:
    return float(base_max_accel)

  fake.get_low_speed_launch_follow_max_accel = get_low_speed_launch_follow_max_accel
  sys.modules[module_name] = fake


def import_longitudinal_planner_sp():
  mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner')
  return mod


def import_longitudinal_planner_with_fake_mpc():
  install_fake_long_mpc()
  # Ensure we import the planner after fake MPC is installed.
  sys.modules.pop('openpilot.selfdrive.controls.lib.longitudinal_planner', None)
  mod = importlib.import_module('openpilot.selfdrive.controls.lib.longitudinal_planner')
  return mod


def run_vtsc_min_of_sources(
  planner_sp,
  *,
  steps: Iterable[Step],
  dt: float = 0.05,
  start_t: float = 0.0,
  integrate_ego: bool = False,
) -> list[dict[str, Any]]:
  """Drive `LongitudinalPlannerSP.update_v_cruise` with synthetic messages.

  Returns a list of per-step dicts with:
  - `t`: simulated time
  - `vtsc_v_turn`: `planner_sp.v_tsc.v_turn`
  - `vtsc_active`: `planner_sp.v_tsc.is_active`
  - `v_cruise_final`: return value of update_v_cruise
  """
  # Patch time used inside VTSC deterministically (both module import paths).
  # NOTE: VTSC uses `time.time()` for dwell and reacq logic.
  vtc_mod = importlib.import_module('openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller')

  t = float(start_t)
  v_ego_sim: float | None = None
  a_ego_sim: float | None = None
  out: list[dict[str, Any]] = []

  for st in steps:
    if integrate_ego:
      if v_ego_sim is None:
        v_ego_sim = float(st.v_ego)
        a_ego_sim = float(st.a_ego)
      v_ego = float(v_ego_sim)
      a_ego = float(a_ego_sim)
    else:
      v_ego = float(st.v_ego)
      a_ego = float(st.a_ego)

    model = make_model_v2(curvature=st.curvature, v_pred=v_ego, confidence=st.confidence)
    sm = FakeSubMaster(
      data={
        'modelV2': model,
        'radarState': make_radar_state(lead_d_rel_m=st.lead_d_rel_m),
        'carState': make_car_state(gas_pressed=False),
        'carControl': make_car_control(long_active=st.long_active),
      },
      valid={'modelV2': True, 'radarState': True},
    )

    # NOTE: We patch time via `unittest.mock.patch` here to avoid relying on monkeypatch fixture.
    from unittest.mock import patch
    with patch.object(vtc_mod.time, 'time', lambda: t), patch.object(vtc_mod.time, 'monotonic', lambda: t):
      v_cruise_final = float(planner_sp.update_v_cruise(sm, v_ego, a_ego, float(st.v_cruise)))

    out.append({
      't': t,
      'v_ego': v_ego,
      'a_ego': a_ego,
      'vtsc_v_turn': float(getattr(planner_sp.v_tsc, 'v_turn', 0.0)),
      'vtsc_active': bool(getattr(planner_sp.v_tsc, 'is_active', False)),
      'v_cruise_final': v_cruise_final,
    })

    if integrate_ego:
      try:
        a_ego_sim = float(getattr(planner_sp.v_tsc, 'a_target', 0.0))
      except Exception:
        a_ego_sim = 0.0
      v_ego_sim = max(0.0, float(v_ego_sim) + float(a_ego_sim) * float(dt))

    t += float(dt)

  return out
