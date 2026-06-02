import os
import sys

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_response_model import DEFAULT_COMFORT_BRAKE
from openpilot.selfdrive.modeld.constants import index_function


if sys.platform != "win32":
  raise ImportError("windows_acados_stub is only for Windows-native tests")


class AcadosOcpSolverCython:
  def __init__(self, model_name, solver_type, n):
    self.N = int(n)
    self.t_idxs = np.array([index_function(idx, max_val=10.0, max_idx=self.N) for idx in range(self.N + 1)])
    self.x = np.zeros((self.N + 1, 3))
    self.u = np.zeros((self.N, 1))
    self.yref = np.zeros((self.N + 1, 6))
    self.params = np.zeros((self.N + 1, 6))
    self.uh = np.tile(np.full(4, 1e4), (self.N, 1))
    self.x0 = np.zeros(3)

  def reset(self):
    self.x[:] = 0.0
    self.u[:] = 0.0

  def cost_set(self, stage, field, value):
    pass

  def set(self, stage, field, value):
    arr = np.asarray(value, dtype=float)
    if field == "x":
      self.x[stage, :arr.shape[0]] = arr
    elif field == "u" and stage < self.N:
      self.u[stage, :arr.shape[0]] = arr
    elif field == "yref":
      self.yref[stage, :arr.shape[0]] = arr
    elif field == "p":
      self.params[stage, :arr.shape[0]] = arr

  def constraints_set(self, stage, field, value):
    if stage == 0 and field in ("lbx", "ubx"):
      self.x0 = np.asarray(value, dtype=float)
    elif field == "uh" and stage < self.N:
      arr = np.asarray(value, dtype=float)
      self.uh[stage, :arr.shape[0]] = arr

  def solve(self):
    self.x[0] = self.x0
    for i in range(self.N):
      dt = max(1e-3, float(self.t_idxs[i + 1] - self.t_idxs[i]))
      x_i, v_i, a_i = [float(v) for v in self.x[i]]
      p = self.params[i]
      a_min = float(p[0]) if np.isfinite(p[0]) else -3.5
      a_max = float(p[1]) if np.isfinite(p[1]) else 2.0
      obstacle = float(p[2]) if p[2] else 1e9
      t_follow = max(0.5, float(p[4]) if p[4] else 1.45)
      danger_factor = max(0.0, float(p[5]) if p[5] else 0.75)
      v_upper = float(self.uh[i, 0]) if self.uh[i, 0] else 1e4

      y_v = float(self.yref[i, 2])
      if y_v > 0.01:
        speed_error = y_v - v_i
      else:
        desired_dist_comfort = (
          self._safe_obstacle_distance(v_i, t_follow)
          if self._use_safe_obstacle_geometry()
          else 6.0 + max(v_i, 0.0) * t_follow
        )
        gap = obstacle - x_i - desired_dist_comfort
        speed_error = 0.06 * gap

      a_cmd = np.clip(a_i + 0.35 * speed_error, a_min, a_max)
      if obstacle < 1e8:
        danger_distance = (
          danger_factor * self._safe_obstacle_distance(v_i, t_follow)
          if self._use_safe_obstacle_geometry()
          else 4.5 + max(v_i, 0.0) * 0.75
        )
        danger_gap = obstacle - x_i - danger_distance
        if danger_gap < 0.0:
          a_cmd = min(float(a_cmd), max(a_min, -1.5 + 0.08 * danger_gap))
      if v_i + float(a_cmd) * dt > v_upper:
        a_cmd = min(float(a_cmd), max(a_min, (v_upper - v_i) / dt))

      self.u[i, 0] = (float(a_cmd) - a_i) / dt
      v_next = min(v_upper, max(0.0, v_i + float(a_cmd) * dt))
      x_next = x_i + v_i * dt + 0.5 * float(a_cmd) * dt * dt
      self.x[i + 1] = [x_next, v_next, float(a_cmd)]
    return 0

  @staticmethod
  def _safe_obstacle_distance(v_ego, t_follow):
    v_ego = max(float(v_ego), 0.0)
    return (v_ego * v_ego) / (2.0 * DEFAULT_COMFORT_BRAKE) + t_follow * v_ego + 6.0

  @staticmethod
  def _use_safe_obstacle_geometry():
    return os.environ.get("OPENPILOT_TEST_MPC_SAFE_OBSTACLE_GEOMETRY") == "1"

  def get_stats(self, name):
    return np.array([0.0])

  def get(self, stage, field):
    if field == "x":
      return self.x[stage]
    if field == "u":
      return self.u[stage]
    raise KeyError(field)
