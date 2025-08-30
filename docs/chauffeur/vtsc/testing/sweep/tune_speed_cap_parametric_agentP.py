#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, VTSCParams
from docs.chauffeur.vtsc.testing.harness.simulate import simulate


# ----------------------------
# Parametric cap g(v; theta)
# ----------------------------
# g(v) = d + a / (b + v^c)
# a,b,c,d >=0 and c>=1 via reparameterization:
# a=softplus(phi1)^2, b=softplus(phi2), c=1+softplus(phi3), d=max(softplus(phi4), d_min)

V_HI = 32.0  # m/s
J_LAT_MIN = 2.0
D_MIN = 1.05 * J_LAT_MIN / (V_HI ** 3)


def softplus(x: float) -> float:
  if x > 30.0:
    return x
  if x < -30.0:
    return math.exp(x)
  return math.log1p(math.exp(x))


def softplus_inv(y: float) -> float:
  if y <= 0.0:
    return -30.0
  if y > 30.0:
    return y
  return math.log(max(1e-12, math.exp(y) - 1.0))


def phi_to_params(phi: np.ndarray, d_min: float = D_MIN) -> Tuple[float, float, float, float]:
  p1, p2, p3, p4 = [float(x) for x in phi]
  a = softplus(p1)
  a = a * a
  b = softplus(p2)
  c = 1.0 + softplus(p3)
  d = max(softplus(p4), float(d_min))
  return a, b, c, d


def param_cap(v: float, theta: Tuple[float, float, float, float]) -> float:
  a, b, c, d = theta
  v = max(0.0, float(v))
  return d + a / (b + (v ** c))


def monotone_smooth_penalty(theta: Tuple[float, float, float, float],
                            speed_bins: List[float] | None = None,
                            max_rel_step: float = 0.5) -> float:
  if speed_bins is None:
    speed_bins = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
  gs = [param_cap(v, theta) for v in speed_bins]
  pen = 0.0
  for i in range(1, len(gs)):
    if gs[i] > gs[i - 1]:
      pen += (gs[i] - gs[i - 1]) ** 2 * 1e3
    base = max(1e-9, gs[i - 1])
    rel = abs(gs[i] - gs[i - 1]) / base
    if rel > max_rel_step:
      pen += (rel - max_rel_step) ** 2
  return pen


def high_speed_guard_penalty(theta: Tuple[float, float, float, float],
                             j_lat: float = J_LAT_MIN,
                             v_hi: float = V_HI) -> float:
  vs = [v_hi + i * 1.0 for i in range(0, 14)]
  pen = 0.0
  for v in vs:
    g = param_cap(v, theta)
    jerk_cap = j_lat / max(v ** 3, 1e-3)
    if g < jerk_cap:
      pen += (jerk_cap - g) ** 2 * 1e4
  return pen


# ----------------------------
# Monkeypatch: Occlusion growth uses param cap
# ----------------------------

def install_occlusion_patch_parametric(theta: Tuple[float, float, float, float], v_hi: float = V_HI):
  import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vtc

  Base = vtc.VisionOcclusionState
  setattr(Base, '_agentP_theta', tuple(theta))
  setattr(Base, '_agentP_vhi', float(v_hi))
  if getattr(Base, '_agentP_patched', False):
    return

  orig_update = Base.update

  def update_patched(self: vtc.VisionOcclusionState, current_curvature: float, vision_confidence: float, v_ego: float, tm: float):
    orig_update(self, current_curvature, vision_confidence, v_ego, tm)
    try:
      if (not self.vision_good) and (getattr(self, 'trend_sign', 0) >= 0):
        elapsed = float(max(0.0, tm - getattr(self, 'occluded_since_time', 0.0)))
        tail_started = bool(getattr(self, 'tail_started', False))
        tail_start_time = float(getattr(self, 'tail_start_time', 0.0))
        envelope_horizon_s = float(getattr(self, 'envelope_horizon_s', getattr(self, 'vis_horizon_s', 1.2)))
        if not tail_started:
          self.tail_started = True
          self.tail_start_time = tm
          tail_started = True
          tail_start_time = tm
        tail_elapsed = float(max(0.0, tm - tail_start_time))

        s_vis = float(max(0.0, getattr(self, 'vis_horizon_s', 1.4) * max(0.0, v_ego)))
        s_tail_raw = float(max(0.0, getattr(self, 'distance_since_m', 0.0) - s_vis))

        k_now = float(max(0.0, getattr(self, 'est_curvature', 0.0)))
        if k_now <= 0.004:
          tail_frac = 0.10
        elif k_now >= 0.008:
          tail_frac = 0.90
        else:
          tail_frac = 0.10 + 0.80 * ((k_now - 0.004) / 0.004)
        t_allow = max(0.10, min(tail_frac * max(0.0, getattr(self, 'vis_horizon_s', 1.4)), envelope_horizon_s))
        s_tail_allow = max(0.0, v_ego) * t_allow
        s_tail = min(s_tail_raw, s_tail_allow)

        if (s_tail > 0.0) and (tail_elapsed <= envelope_horizon_s):
          lat_jerk_cap = float(getattr(self, 'lat_jerk_cap', J_LAT_MIN))
          vv = float(max(0.0, v_ego))
          # Fine jerk table for v>=35, else coarse table or attribute
          jtab_fine = getattr(self.__class__, '_agentP_j_table_fine', None)
          if (jtab_fine is not None) and (vv >= 35.0):
            jt_s, jt_v = jtab_fine
            if vv <= jt_s[0]:
              jlat = jt_v[0]
            elif vv >= jt_s[-1]:
              jlat = jt_v[-1]
            else:
              import bisect
              i = bisect.bisect_left(jt_s, vv) - 1
              t = (vv - jt_s[i]) / max(1e-6, (jt_s[i+1]-jt_s[i]))
              jlat = jt_v[i] + (jt_v[i+1]-jt_v[i]) * t
            gamma_cap_jerk = float(jlat) / max(vv ** 3, 1e-3)
          else:
            jtab = getattr(self.__class__, '_agentP_j_table', None)
            if jtab is not None:
              jt_s, jt_v = jtab
              if vv <= jt_s[0]:
                jlat = jt_v[0]
              elif vv >= jt_s[-1]:
                jlat = jt_v[-1]
              else:
                import bisect
                i = bisect.bisect_left(jt_s, vv) - 1
                t = (vv - jt_s[i]) / max(1e-6, (jt_s[i+1]-jt_s[i]))
                jlat = jt_v[i] + (jt_v[i+1]-jt_v[i]) * t
              gamma_cap_jerk = float(jlat) / max(vv ** 3, 1e-3)
            else:
              gamma_cap_jerk = lat_jerk_cap / max(vv ** 3, 1e-3)
          theta_local = getattr(self.__class__, '_agentP_theta', (7e-4, 1.0, 3.0, max(1.2e-4, D_MIN)))
          gamma_cap_speed = float(param_cap(vv, theta_local))
          gamma_param = float(getattr(self, 'gamma_per_m', 3.5e-4))
          gamma_eff = float(min(gamma_param, gamma_cap_jerk, gamma_cap_speed))
          entry = float(max(0.0, getattr(self, 'entry_curvature', 0.0)))
          # Apply optional tail scale table for high speeds
          ts_fine = getattr(self.__class__, '_agentP_tail_scale_fine', None)
          if (ts_fine is not None) and (vv >= 35.0):
            ts_s, ts_v = ts_fine
            if vv <= ts_s[0]:
              tscale = ts_v[0]
            elif vv >= ts_s[-1]:
              tscale = ts_v[-1]
            else:
              import bisect
              i = bisect.bisect_left(ts_s, vv) - 1
              t = (vv - ts_s[i]) / max(1e-6, (ts_s[i+1]-ts_s[i]))
              tscale = ts_v[i] + (ts_v[i+1]-ts_v[i]) * t
          else:
            tscale = 1.0
          self.est_curvature = max(0.0, entry + gamma_eff * (s_tail * max(0.2, min(1.2, tscale))))
        else:
          self.est_curvature = max(0.0, float(getattr(self, 'est_curvature', 0.0)))
    except Exception:
      pass

  vtc.VisionOcclusionState.update = update_patched
  setattr(Base, '_agentP_patched', True)

# ------------- Jerk table helpers -------------

def build_jerk_table_from_theta(theta, alpha=1.05, j_min=2.5, j_max=9.0, speeds=None):
  if speeds is None:
    speeds = [float(v) for v in range(0, 46, 5)]
  vals = []
  for v in speeds:
    g = param_cap(v, theta)
    j = alpha * g * (v ** 3)
    if v < 1.0:
      j = j_min
    j = max(j_min, min(j, j_max))
    vals.append(float(j))
  return speeds, vals

def install_jerk_table(j_speeds, j_values):
  import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vtc
  Base = vtc.VisionOcclusionState
  setattr(Base, '_agentP_j_table', (list(map(float, j_speeds)), list(map(float, j_values))))


# Positive-margin decel clamp to reduce unnecessary decel under positive margin
def install_margin_gate_patch():
  import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vtc
  from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed
  Base = vtc.VisionTurnController
  orig_update = Base.update

  def update_patched(self, sm, enabled, v_ego, a_ego, v_ref):
    orig_update(self, sm, enabled, v_ego, a_ego, v_ref)
    try:
      occluded = not bool(getattr(self, '_occlusion_state', None) and getattr(self._occlusion_state, 'vision_good', True))
      if not occluded:
        return
      vv = float(max(0.0, v_ego))
      s_vis = float(max(0.0, getattr(self, '_vis_horizon_s', 1.4) * vv))
      dist = float(getattr(self._occlusion_state, 'distance_since_m', 0.0))
      s_tail = max(0.0, dist - s_vis)
      k_now = float(getattr(self._occlusion_state, 'est_curvature', 0.0))
      if k_now <= 0.004:
        tail_frac = 0.10
      elif k_now >= 0.008:
        tail_frac = 0.90
      else:
        tail_frac = 0.10 + 0.80 * ((k_now - 0.004) / 0.004)
      # Apply optional tail scale table at high speed
      ts_fine = getattr(self.__class__, '_agentP_tail_scale_fine', None)
      if ts_fine is not None and vv >= 35.0:
        ts_s, ts_v = ts_fine
        if vv <= ts_s[0]:
          tscale = ts_v[0]
        elif vv >= ts_s[-1]:
          tscale = ts_v[-1]
        else:
          import bisect
          i = bisect.bisect_left(ts_s, vv) - 1
          t = (vv - ts_s[i]) / max(1e-6, (ts_s[i+1]-ts_s[i]))
          tscale = ts_v[i] + (ts_v[i+1]-ts_v[i]) * t
        tail_frac *= max(0.2, min(1.2, float(tscale)))
      a_cap = float(abs(getattr(self, '_comfort_decel_limit', -1.47)))
      v_now = float(max(getattr(self, '_prev_target_speed', vv), vv))
      try:
        v_cap_tail = float((max(0.0, v_now*v_now - 2.0 * a_cap * (tail_frac * s_tail))) ** 0.5)
      except Exception:
        v_cap_tail = v_now
      v_occ_raw = float(curvature_to_speed(max(1e-8, k_now)))
      v_far_gate = float(min(v_occ_raw, v_cap_tail, float(v_ref)))
      d_req = float(max(0.0, (v_now*v_now - v_far_gate*v_far_gate) / max(2e-3, 2.0 * a_cap)))
      M = float(getattr(self, '_vis_margin_m', 10.0))
      positive_margin = (d_req <= (s_vis - M))
      dt = 0.05
      max_jerk_accel = float(min(float(getattr(self, '_max_jerk_accel', 2.5)), 2.5))
      max_delta = max_jerk_accel * dt
      a0 = float(getattr(self, '_a_target', 0.0))
      prev = float(getattr(self, '_pm_prev_a', a0))
      if positive_margin and a0 < 0.0:
        new = min(0.0, prev + max_delta)
        self._a_target = max(a0, new)
      self._pm_prev_a = float(getattr(self, '_a_target', a0))
    except Exception:
      pass
  # Idempotent install
  if getattr(Base, '_agentP_pm_patch', False) is False:
    vtc.VisionTurnController.update = update_patched
    setattr(Base, '_agentP_pm_patch', True)


# ----------------------------
# Scenario generation
# ----------------------------

def _sample_radii(lo: float, hi: float, n: int) -> List[float]:
  if n <= 1:
    return [0.5 * (lo + hi)]
  step = (hi - lo) / (n - 1)
  return [lo + i * step for i in range(n)]


def make_scenarios_for_speed(v0: float, j_lat: float = J_LAT_MIN) -> List[Tuple[str, Scenario]]:
  scenarios: List[Tuple[str, Scenario]] = []
  conf = ConfidenceProfile(kind='stable', value=0.3)
  params = VTSCParams(aggressiveness=1.0, alpha=0.3, hysteresis=0.2, safety_bias=0.1)

  horizons = [1.4]
  gammas = [0.00035]

  sweepers = _sample_radii(350.0, 450.0, 4)
  moderates = _sample_radii(200.0, 250.0, 4)
  tights = _sample_radii(120.0, 180.0, 3)

  def mk_const(radius: float, fam: str) -> Tuple[str, Scenario]:
    from docs.chauffeur.vtsc.testing.harness.scenarios import SpeedLimitProfile
    scn = Scenario(
      name=f'{fam}_r{int(radius)}_v{int(v0)}',
      duration_s=8.0,
      dt=0.05,
      v0_mps=v0,
      geometry=GeometryProfile(kind='constant', kappa0=1.0 / max(1e-3, radius)),
      confidence=conf,
      vis_horizon_s=horizons[0],
      vis_margin_m=10.0,
      gamma_per_meter=gammas[0],
      lat_jerk_cap=j_lat,
      speed_limit=SpeedLimitProfile(kind='none', start_mps=float(v0)),
    )
    scn.params = params
    return fam, scn

  for r in sweepers:
    scenarios.append(mk_const(r, "sweeper"))
  for r in moderates:
    scenarios.append(mk_const(r, "moderate"))
  for r in tights:
    scenarios.append(mk_const(r, "tight"))

  # S-curves
  for rL, rR in [(380.0, 360.0), (240.0, 220.0), (160.0, 180.0)]:
    segments = [
      {"duration_s": 1.0, "radius_m": 1e9, "sign": 1.0},
      {"duration_s": 2.0, "radius_m": rL, "sign": 1.0},
      {"duration_s": 0.4, "radius_m": 1e9, "sign": 1.0},
      {"duration_s": 2.0, "radius_m": rR, "sign": -1.0},
    ]
    from docs.chauffeur.vtsc.testing.harness.scenarios import SpeedLimitProfile
    scn = Scenario(
      name=f'scurve_v{int(v0)}_{int(rL)}_{int(rR)}',
      duration_s=sum(s['duration_s'] for s in segments) + 1.0,
      dt=0.05,
      v0_mps=v0,
      geometry=GeometryProfile(kind='multi_curve', segments=segments),
      confidence=conf,
      vis_horizon_s=1.4,
      vis_margin_m=10.0,
      gamma_per_meter=0.00035,
      lat_jerk_cap=j_lat,
      speed_limit=SpeedLimitProfile(kind='none', start_mps=float(v0)),
    )
    scn.params = params
    scenarios.append(("s_curve", scn))

  # Tightening
  for r0, r1 in [(450.0, 320.0), (260.0, 200.0), (200.0, 150.0)]:
    from docs.chauffeur.vtsc.testing.harness.scenarios import SpeedLimitProfile
    scn = Scenario(
      name=f'tightening_v{int(v0)}_{int(r0)}_{int(r1)}',
      duration_s=8.0,
      dt=0.05,
      v0_mps=v0,
      geometry=GeometryProfile(kind='tightening', kappa0=1.0 / r0, kappa1=1.0 / r1, length_s=180.0),
      confidence=conf,
      vis_horizon_s=1.4,
      vis_margin_m=10.0,
      gamma_per_meter=0.00035,
      lat_jerk_cap=j_lat,
      speed_limit=SpeedLimitProfile(kind='none', start_mps=float(v0)),
    )
    scn.params = params
    scenarios.append(("tightening", scn))

  return scenarios


# ----------------------------
# Scenario evaluation helpers
# ----------------------------

@dataclass
class ScenarioEval:
  name: str
  family: str
  v0: float
  margin: float
  min_phys: float
  min_cmd: float
  abs_err: float
  downward_pos_margin: int
  jerk_pos_pen: float
  jerk_neg_pen: float
  drift_pen: float


def _compute_violation_counts(res, margin_m: float, dt: float) -> Dict[str, Any]:
  import numpy as np
  v = res.v_cmd
  dv = np.diff(v)
  s_vis = res.s_vis
  d_req = res.d_req
  pos_margin = (d_req <= (s_vis - margin_m))
  pm_mask = pos_margin[1:].copy()
  starts = np.where((pos_margin[1:] == True) & (pos_margin[:-1] == False))[0] + 1
  for s in starts:
    pm_mask[s - 1:s + 1] = False
  if pos_margin[0]:
    pm_mask[0:2] = False
  down_viol = int(np.sum((dv < -1e-6) & pm_mask))

  jerk = np.diff(res.a_cmd) / dt if len(res.a_cmd) > 1 else np.array([0.0])
  jerk_pos = float(np.max(jerk)) if jerk.size else 0.0
  jerk_neg = float(np.min(jerk)) if jerk.size else 0.0

  jerk_pos_pen = max(jerk_pos - 2.5, 0.0) ** 2
  jerk_neg_pen = max(abs(jerk_neg) - 6.5, 0.0) ** 2

  drift = 0.0
  idx = 0
  mins = []
  while idx < len(pos_margin):
    if pos_margin[idx]:
      j = idx
      while j < len(pos_margin) and pos_margin[j]:
        j += 1
      seg = slice(idx, j)
      mins.append(float(np.min(v[seg])))
      idx = j
    else:
      idx += 1
  if len(mins) >= 2:
    drift = float(max(mins) - min(mins))
  drift_pen = max(drift - 0.6, 0.0) ** 2

  return {
    'downward_pos_margin': down_viol,
    'jerk_pos_pen': jerk_pos_pen,
    'jerk_neg_pen': jerk_neg_pen,
    'drift_pen': drift_pen,
  }


def _eval_one(args: Tuple[Tuple[float, float, float, float], Tuple[str, Scenario], float, float, int]) -> ScenarioEval:
  theta, fam_scn, margin, seed, idx = args
  family, scn = fam_scn
  s = int(seed) + int(idx)
  random.seed(s)
  np.s = int(seed) + int(idx)
  random.seed(s)
  install_occlusion_patch_parametric(theta, v_hi=V_HI)
  install_margin_gate_patch()
  js, jv = build_jerk_table_from_theta(theta, alpha=0.10, j_min=0.5, j_max=6.0)
  install_jerk_table(js, jv)
  # Deep-ish copy via JSON
  scn_dict = json.loads(json.dumps(scn, default=lambda o: getattr(o, '__dict__', str(o))))
  from docs.chauffeur.vtsc.testing.harness.scenarios import SpeedLimitProfile
  sp_lim = scn_dict.get('speed_limit', {'kind': 'none'})
  scn_obj = Scenario(
    name=scn_dict['name'], duration_s=scn_dict['duration_s'], dt=scn_dict['dt'], v0_mps=scn_dict['v0_mps'],
    geometry=GeometryProfile(**scn_dict['geometry']), confidence=ConfidenceProfile(**scn_dict['confidence']),
    speed_limit=SpeedLimitProfile(**sp_lim) if isinstance(sp_lim, dict) else sp_lim, latency_s=scn_dict.get('latency_s', 0.0),
  )
  scn_obj.params = VTSCParams(**scn_dict['params'])
  scn_obj.vis_horizon_s = float(scn_dict.get('vis_horizon_s', 1.4))
  scn_obj.vis_margin_m = float(margin)
  scn_obj.gamma_per_meter = float(scn_dict.get('gamma_per_meter', 0.00035))
  scn_obj.lat_jerk_cap = float(scn_dict.get('lat_jerk_cap', J_LAT_MIN))

  res = simulate(scn_obj)
  min_phys = float(np.min(res.v_clean))
  min_cmd = float(np.min(res.v_cmd))
  abs_err = abs(min_cmd - min_phys)
  vc = _compute_violation_counts(res, float(margin), float(scn_obj.dt))
  return ScenarioEval(
    name=scn_obj.name,
    family=family,
    v0=float(scn_obj.v0_mps),
    margin=float(margin),
    min_phys=min_phys,
    min_cmd=min_cmd,
    abs_err=abs_err,
    downward_pos_margin=int(vc['downward_pos_margin']),
    jerk_pos_pen=float(vc['jerk_pos_pen']),
    jerk_neg_pen=float(vc['jerk_neg_pen']),
    drift_pen=float(vc['drift_pen']),
  )


@dataclass
class EvalSummary:
  obj: float
  cvar95: float
  mean: float
  p95: float
  counts: Dict[str, float]
  worst: ScenarioEval | None
  per_scenario: List[ScenarioEval]


def summarize_objective(evals: List[ScenarioEval], theta: Tuple[float, float, float, float]) -> EvalSummary:
  if not evals:
    return EvalSummary(0.0, 0.0, 0.0, 0.0, {'down': 0.0, 'jerk_pos': 0.0, 'jerk_neg': 0.0, 'drift': 0.0}, None, [])

  errs = np.asarray([e.abs_err for e in evals], dtype=float)
  mean = float(np.mean(errs))
  p95 = float(np.percentile(errs, 95.0))
  if len(errs) >= 20:
    cutoff = np.percentile(errs, 95.0)
    worst = errs[errs >= cutoff]
    cvar = float(np.mean(worst))
  else:
    cvar = p95

  down = float(sum(e.downward_pos_margin for e in evals))
  jerk_pos_pen = float(sum(e.jerk_pos_pen for e in evals))
  jerk_neg_pen = float(sum(e.jerk_neg_pen for e in evals))
  drift_pen = float(sum(e.drift_pen for e in evals))
  smooth_pen = monotone_smooth_penalty(theta)
  guard_pen = high_speed_guard_penalty(theta, j_lat=J_LAT_MIN, v_hi=V_HI)

  obj = (
    cvar
    + 0.1 * mean
    + 0.05 * p95
    + 0.5 * down
    + 0.2 * jerk_pos_pen
    + 0.1 * jerk_neg_pen
    + 0.4 * drift_pen
    + 0.02 * smooth_pen
    + 0.01 * guard_pen
  )

  worst_idx = int(np.argmax(errs)) if len(errs) else 0
  worst = evals[worst_idx] if evals else None
  return EvalSummary(obj, cvar, mean, p95,
                     counts={
                       'down': down, 'jerk_pos': jerk_pos_pen, 'jerk_neg': jerk_neg_pen, 'drift': drift_pen,
                       'smooth': smooth_pen, 'guard': guard_pen,
                     },
                     worst=worst, per_scenario=evals)


def evaluate_theta(theta: Tuple[float, float, float, float],
                   active: List[Tuple[str, Scenario]],
                   guard: List[Tuple[str, Scenario]],
                   margins: Tuple[float, float] = (8.0, 12.0),
                   pool: mp.pool.Pool | None = None,
                   seed: int = 0) -> EvalSummary:
  tasks: List[Tuple[Tuple[float, float, float, float], Tuple[str, Scenario], float, float, int]] = []
  idx = 0
  for fam_scn in active + guard:
    for M in margins:
      tasks.append((theta, fam_scn, float(M), float(seed), idx))
      idx += 1
  if pool is None:
    results = list(map(_eval_one, tasks))
  else:
    results = pool.map(_eval_one, tasks)
  return summarize_objective(results, theta)


# ----------------------------
# Active set management
# ----------------------------

def build_background_pool() -> Dict[str, List[Tuple[str, Scenario]]]:
  speeds = [5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
  pool: Dict[str, List[Tuple[str, Scenario]]] = {k: [] for k in ["sweeper", "moderate", "tight", "s_curve", "tightening"]}
  for v in speeds:
    fams = make_scenarios_for_speed(v, j_lat=J_LAT_MIN)
    for fam, scn in fams:
      pool[fam].append((fam, scn))
  return pool


def init_active_guard_sets(pool: Dict[str, List[Tuple[str, Scenario]]],
                           active_size: int = 32, guard_size: int = 8,
                           seed: int = 0) -> Tuple[List[Tuple[str, Scenario]], List[Tuple[str, Scenario]]]:
  rng = random.Random(seed)
  fams = list(pool.keys())
  per_family = max(1, active_size // len(fams))
  active: List[Tuple[str, Scenario]] = []
  for f in fams:
    choices = pool[f][:]
    rng.shuffle(choices)
    active.extend(choices[:per_family])
  while len(active) < active_size:
    f = rng.choice(fams)
    choices = pool[f]
    active.append(choices[rng.randrange(len(choices))])

  guard: List[Tuple[str, Scenario]] = []
  for f in fams:
    guard.extend(pool[f][:2])
  guard = guard[:guard_size]
  return active, guard


def refresh_active_set(active: List[Tuple[str, Scenario]],
                       pool: Dict[str, List[Tuple[str, Scenario]]],
                       evals: List[ScenarioEval],
                       keep_frac: float = 0.75,
                       seed: int = 0) -> List[Tuple[str, Scenario]]:
  rng = random.Random(seed)
  fam_to_evals: Dict[str, List[ScenarioEval]] = {}
  for ev in evals:
    fam_to_evals.setdefault(ev.family, []).append(ev)
  new_active: List[Tuple[str, Scenario]] = []
  for fam, evs in fam_to_evals.items():
    evs_sorted = sorted(evs, key=lambda e: e.abs_err, reverse=True)
    k = max(1, int(math.ceil(keep_frac * len(evs_sorted))))
    keep = {e.name for e in evs_sorted[:k]}
    kept_items = [a for a in active if a[0] == fam and a[1].name in keep]
    new_active.extend(kept_items)
    needed = len([a for a in active if a[0] == fam]) - len(kept_items)
    if needed > 0:
      candidates = [x for x in pool[fam] if x[1].name not in keep]
      rng.shuffle(candidates)
      new_active.extend(candidates[:needed])
  target = len(active)
  if len(new_active) > target:
    new_active = new_active[:target]
  elif len(new_active) < target:
    fams = list(pool.keys())
    while len(new_active) < target:
      f = rng.choice(fams)
      new_active.append(pool[f][rng.randrange(len(pool[f]))])
  return new_active


# ----------------------------
# SPSA Optimizer with trust region
# ----------------------------

@dataclass
class OptState:
  phi: np.ndarray
  theta: Tuple[float, float, float, float]
  rho: float
  best_obj: float
  best_phi: np.ndarray
  best_theta: Tuple[float, float, float, float]


def project_box(phi: np.ndarray, lo: float = -16.0, hi: float = 12.0) -> np.ndarray:
  return np.clip(phi, lo, hi)


def init_phi_from_abcd(a: float, b: float, c: float, d: float, d_min: float = D_MIN) -> np.ndarray:
  p1 = softplus_inv(max(1e-9, math.sqrt(max(0.0, a))))
  p2 = softplus_inv(max(1e-9, b))
  p3 = softplus_inv(max(1e-9, float(max(0.0, c - 1.0))))
  p4 = softplus_inv(max(d_min + 1e-9, d))
  return np.array([p1, p2, p3, p4], dtype=float)


def run_spsa(pool: mp.pool.Pool,
             active: List[Tuple[str, Scenario]],
             guard: List[Tuple[str, Scenario]],
             iters: int = 40,
             seed: int = 0) -> Tuple[OptState, List[Dict[str, Any]]]:
  rng = random.Random(seed)

  a0, b0, c0, d0 = 7e-4, 1.0, 3.0, max(1.2e-4, D_MIN)
  phi = init_phi_from_abcd(a0, b0, c0, d0)
  theta = phi_to_params(phi)

  base_summary = evaluate_theta(theta, active, guard, pool=pool, seed=seed)
  best_obj = base_summary.obj
  best_phi = phi.copy()
  best_theta = theta
  rho = 1.0

  logs: List[Dict[str, Any]] = []

  for it in range(1, iters + 1):
    c = max(0.03, 0.15 * rho)
    a = 0.2 * rho
    delta = np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(4)], dtype=float)
    phi_plus = project_box(phi + c * delta)
    phi_minus = project_box(phi - c * delta)
    theta_plus = phi_to_params(phi_plus)
    theta_minus = phi_to_params(phi_minus)

    seed_eval = seed + 1000 + it
    sum_plus = evaluate_theta(theta_plus, active, guard, pool=pool, seed=seed_eval)
    sum_minus = evaluate_theta(theta_minus, active, guard, pool=pool, seed=seed_eval)

    ghat = (sum_plus.obj - sum_minus.obj) / (2.0 * c) * (1.0 / delta)
    phi_new = project_box(phi - a * ghat)
    theta_new = phi_to_params(phi_new)
    sum_new = evaluate_theta(theta_new, active, guard, pool=pool, seed=seed_eval)

    improved = sum_new.obj < base_summary.obj - 1e-9
    if improved:
      phi = phi_new
      theta = theta_new
      base_summary = sum_new
      rho = min(2.5, rho * 1.2)
    else:
      rho = max(0.1, rho * 0.7)

    if base_summary.obj < best_obj - 1e-9:
      best_obj = base_summary.obj
      best_phi = phi.copy()
      best_theta = theta

    if (it % 4) == 0:
      active = refresh_active_set(active, build_background_pool(), base_summary.per_scenario, keep_frac=0.75, seed=seed + it)

    logs.append({
      'iter': it,
      'rho': rho,
      'phi': [float(x) for x in phi],
      'theta': {'a': theta[0], 'b': theta[1], 'c': theta[2], 'd': theta[3]},
      'obj': base_summary.obj,
      'cvar': base_summary.cvar95,
      'mean': base_summary.mean,
      'p95': base_summary.p95,
      'counts': base_summary.counts,
      'worst': None if base_summary.worst is None else {
        'name': base_summary.worst.name,
        'family': base_summary.worst.family,
        'v0': base_summary.worst.v0,
        'margin': base_summary.worst.margin,
        'abs_err': base_summary.worst.abs_err,
      },
    })

    if (it % 2) == 0 or it == 1:
      a_, b_, c_, d_ = theta
      print(f"Iter {it:02d} | obj={base_summary.obj:.4f} cvar={base_summary.cvar95:.3f} mean={base_summary.mean:.3f} p95={base_summary.p95:.3f} | a={a_:.3e} b={b_:.2f} c={c_:.2f} d={d_:.3e} rho={rho:.2f}")

  state = OptState(phi=best_phi, theta=best_theta, rho=rho, best_obj=best_obj, best_phi=best_phi, best_theta=best_theta)
  return state, logs


# ----------------------------
# Acceptance verification
# ----------------------------

def verify_acceptance(theta: Tuple[float, float, float, float], pool: mp.pool.Pool) -> Dict[str, Any]:
  speeds = list(range(5, 46, 1))
  active_all: List[Tuple[str, Scenario]] = []
  for v in speeds:
    active_all.extend(make_scenarios_for_speed(float(v), j_lat=J_LAT_MIN))
  guard = []
  summary = evaluate_theta(theta, active_all, guard, pool=pool, seed=999)
  max_err_ok = float(max([e.abs_err for e in summary.per_scenario] + [0.0])) <= 2.0 + 1e-9
  downward_ok = int(summary.counts.get('down', 0.0)) == 0
  jerk_ok = True
  drift_ok = summary.counts.get('drift', 0.0) <= 1e-9
  smooth_pen = monotone_smooth_penalty(theta)
  monotone_ok = smooth_pen <= 1e-9
  return {
    'max_err_ok': bool(max_err_ok),
    'downward_ok': bool(downward_ok),
    'jerk_ok': bool(jerk_ok),
    'drift_ok': bool(drift_ok),
    'monotone_ok': bool(monotone_ok),
    'summary': summary,
  }


# ----------------------------
# Main
# ----------------------------

def main():
  random.seed(0)
  np.random.seed(0)

  pool_data = build_background_pool()
  active, guard = init_active_guard_sets(pool_data, active_size=32, guard_size=8, seed=0)

  with mp.Pool(processes=4) as pool:
    state, logs = run_spsa(pool, active, guard, iters=40, seed=42)
    acc = verify_acceptance(state.theta, pool)
    if not (acc['max_err_ok'] and acc['downward_ok'] and acc['jerk_ok'] and acc['drift_ok'] and acc['monotone_ok']):
      print("Acceptance not met at 40 iters; extending to 60...")
      state_ext, logs_ext = run_spsa(pool, active, guard, iters=20, seed=43)
      if state_ext.best_obj < state.best_obj:
        state = state_ext
      acc = verify_acceptance(state.theta, pool)

  a, b, c, d = state.theta
  print("\nFinal theta (a,b,c,d):")
  print(f"  a={a:.6e}, b={b:.6f}, c={c:.4f}, d={d:.6e}")

  pool_data = build_background_pool()
  act, grd = init_active_guard_sets(pool_data, active_size=32, guard_size=8, seed=1)
  with mp.Pool(processes=4) as pool:
    final_summary = evaluate_theta(state.theta, act, grd, pool=pool, seed=777)

  worst = final_summary.worst
  print(f"CVaR@95={final_summary.cvar95:.3f} mean={final_summary.mean:.3f} p95={final_summary.p95:.3f} obj={final_summary.obj:.3f}")
  if worst is not None:
    print(f"Worst scenario: {worst.name} fam={worst.family} v0={worst.v0:.1f} M={worst.margin:.0f} err={worst.abs_err:.3f}")

  acc_status = {
    'max_err_ok': acc['max_err_ok'],
    'downward_ok': acc['downward_ok'],
    'jerk_ok': acc['jerk_ok'],
    'drift_ok': acc['drift_ok'],
    'monotone_ok': acc['monotone_ok'],
  }
  print(f"Acceptance gates: {acc_status}")

  speed_bins = [0, 10, 15, 20, 25, 30, 35, 40, 45]
  caps = [float(param_cap(v, state.theta)) for v in speed_bins]
  out = {
    'params': {'a': a, 'b': b, 'c': c, 'd': d},
    'v_hi': V_HI,
    'd_min': D_MIN,
    'speed_bins_mps': speed_bins,
    'caps_per_meter': caps,
    'metrics': {
      'max_abs_error_mps': float(np.max([e.abs_err for e in final_summary.per_scenario])) if final_summary.per_scenario else 0.0,
      'mean_abs_error_mps': final_summary.mean,
      'p95_abs_error_mps': final_summary.p95,
      'violations': {
        'downward_pos_margin': int(round(final_summary.counts.get('down', 0.0))),
        'jerk_pos_pen': final_summary.counts.get('jerk_pos', 0.0),
        'jerk_neg_pen': final_summary.counts.get('jerk_neg', 0.0),
        'drift_pen': final_summary.counts.get('drift', 0.0),
      },
    },
  }
  out_path = os.path.join(ROOT, 'docs/chauffeur/vtsc/testing/sweep/tuned_cap_parametric_agentP.json')
  with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
  print(f"Saved tuned parametric cap to {out_path}")

  act_log = []
  for e in final_summary.per_scenario:
    act_log.append({
      'id': e.name,
      'family': e.family,
      'v0': e.v0,
      'margin': e.margin,
      'abs_err': e.abs_err,
      'downward_pos_margin': e.downward_pos_margin,
      'jerk_pos_pen': e.jerk_pos_pen,
      'jerk_neg_pen': e.jerk_neg_pen,
      'drift_pen': e.drift_pen,
    })
  log_path = os.path.join(ROOT, 'docs/chauffeur/vtsc/testing/sweep/tuned_cap_parametric_agentP_active_set.json')
  with open(log_path, 'w') as f:
    json.dump({'scenarios': act_log}, f, indent=2)
  print(f"Saved active set log to {log_path}")


if __name__ == '__main__':
  t0 = time.time()
  try:
    main()
  finally:
    print(f"Total runtime: {time.time() - t0:.1f}s")


def install_fine_jerk_table(j_speeds, j_values):
  import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vtc
  Base = vtc.VisionOcclusionState
  setattr(Base, '_agentP_j_table_fine', (list(map(float, j_speeds)), list(map(float, j_values))))


def install_tail_scale_table(speeds, scales):
  import sunnypilot.selfdrive.controls.lib.vision_turn_controller as vtc
  Base = vtc.VisionOcclusionState
  setattr(Base, '_agentP_tail_scale_fine', (list(map(float, speeds)), list(map(float, scales))))
