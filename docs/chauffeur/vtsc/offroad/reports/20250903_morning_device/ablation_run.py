#!/usr/bin/env python3
import os, sys, math, json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, SpeedLimitProfile, VTSCParams
from docs.chauffeur.vtsc.testing.harness.metrics import compute_metrics
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController, curvature_to_speed
from opendbc.car.common.conversions import Conversions as CV

@dataclass
class SimResult:
    t: np.ndarray
    v_cmd: np.ndarray
    v_clean: np.ndarray
    occluded: np.ndarray


def _mk_sm(curvature: float, v_pred: float, confidence: float):
    model = SimpleNamespace()
    model.orientationRate = SimpleNamespace(z=[curvature] * 33)
    model.velocity = SimpleNamespace(x=[v_pred] * 33)
    model.laneLineProbs = [confidence] * 4
    class SM:
        def __init__(self, model):
            self.valid = {'modelV2': True}
            self._data = {
                'modelV2': model,
                'carState': SimpleNamespace(gasPressed=False),
            }
        def __getitem__(self, key):
            return self._data.get(key)
    return SM(model)


def make_controller(params_overrides: Dict[str, float]) -> VisionTurnController:
    class MockCP: pass
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
        mp = MagicMock()
        mp.get_bool.return_value = True
        def _get(key: str):
            # Default returns None (use controller defaults), unless overridden here
            if key in params_overrides:
                return str(params_overrides[key]).encode()
            return None
        mp.get.side_effect = _get
        MockParams.return_value = mp
        return VisionTurnController(MockCP())


def simulate_hidden_turn(params_overrides: Dict[str, float]) -> SimResult:
    # Build the same hidden-turn scenario as integration test
    v0_mph = 50.0
    v0_mps = v0_mph * CV.MPH_TO_MS
    v_safe_mph = 25.0
    v_safe_mps = v_safe_mph * CV.MPH_TO_MS

    # Find curvature k such that curvature_to_speed(k) ≈ v_safe_mps
    lo, hi = 1e-5, 0.05
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v_mid = curvature_to_speed(mid)
        if v_mid > v_safe_mps:
            lo = mid
        else:
            hi = mid
    k_tight = 0.5 * (lo + hi)
    r_tight = 1.0 / max(1e-6, k_tight)

    straight_s = 2.0
    bend_s = 6.0
    segments = [
        {"duration_s": straight_s, "radius_m": 1e9, "sign": 1.0},
        {"duration_s": bend_s,     "radius_m": r_tight, "sign": 1.0},
    ]

    delta_psi = 0.524
    s_to_30deg = delta_psi / max(1e-6, k_tight)
    t_to_30deg = s_to_30deg / max(1e-6, v0_mps)
    t_occ_start = straight_s + t_to_30deg

    # Time base
    dt = 0.05
    T = int(round((straight_s + bend_s + 1.0) / dt))
    t = np.array([i*dt for i in range(T)])

    # Generate inputs
    kappa = np.zeros_like(t)
    # Straight in then constant radius
    for i,ti in enumerate(t):
        if ti <= straight_s:
            kappa[i] = 0.0
        else:
            kappa[i] = 1.0 / r_tight
    conf = np.ones_like(t) * 0.9
    # Drop confidence after FOV edge time
    for i,ti in enumerate(t):
        if ti >= t_occ_start:
            conf[i] = 0.3
    v_limit = np.ones_like(t) * v0_mps

    vtsc = make_controller(params_overrides)

    v_cmd=[]; v_clean=[]; occluded=[]
    v_ego = float(v0_mps); a_ego=0.0

    for i,ti in enumerate(t):
        sm=_mk_sm(kappa[i], v_ego, conf[i])
        # patch time for controller internal timers
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: float(ti)), \
             patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: float(ti)):
            vtsc.update(sm, True, v_ego, a_ego, v_limit[i])
        a = float(vtsc.a_target)
        v_ego = max(0.0, v_ego + a*dt)
        a_ego = a
        v_cmd.append(v_ego)
        v_clean.append(float(min(curvature_to_speed(max(1e-8,kappa[i])), v_limit[i])))
        occluded.append(not vtsc._occlusion_state.vision_good)

    return SimResult(t=np.array(t), v_cmd=np.array(v_cmd), v_clean=np.array(v_clean), occluded=np.array(occluded,dtype=bool))


def metric_for(sim: SimResult, t_occ_start: float, dt: float=0.05) -> Dict[str, Any]:
    occ_idx = int(round(t_occ_start / dt))
    chk_idx = min(len(sim.t)-1, occ_idx + int(round(4.5 / dt)))
    v0 = sim.v_cmd[0]
    v_chk = sim.v_cmd[chk_idx]
    # time to first ≥0.8 m/s reduction from v0 after onset
    tt = ''
    for k in range(occ_idx, min(len(sim.v_cmd), occ_idx+int(round(10.0/dt)))):
      if sim.v_cmd[k] <= v0 - 0.8:
        tt = (sim.t[k] - sim.t[occ_idx]); break
    return {
        'v0_mps': float(v0),
        'v_chk_mps': float(v_chk),
        'drop_ge0p8_s': (float(tt) if tt!='' else None),
        'pass_4p5s_rule': bool(v_chk <= v0 - 0.8),
    }


def main():
    # Baseline scenario check index time
    v0_mph = 50.0
    v0_mps = v0_mph * CV.MPH_TO_MS
    # Compute k for ~25 mph safe; reuse from simulate_hidden_turn
    sim0 = simulate_hidden_turn(params_overrides={})
    # Recompute t_occ_start same as inside simulate
    # replicate compute
    # It is robust to duplicate - we compute again
    # 25 mph target
    v_safe_mps = 25.0 * CV.MPH_TO_MS
    lo, hi = 1e-5, 0.05
    for _ in range(60):
        mid = 0.5 * (lo + hi); v_mid = curvature_to_speed(mid)
        if v_mid > v_safe_mps: lo = mid
        else: hi = mid
    k_tight = 0.5*(lo+hi)
    delta_psi = 0.524
    t_occ_start = 2.0 + (delta_psi / k_tight) / max(1e-6, v0_mps)

    cases = [
      ("baseline", {}),
      ("psi_margin_9deg", {"VisionTurnSpeedControlPsiMarginRad": math.radians(9.0)}),
      ("pretrigger_1.0s", {"VisionTurnSpeedControlFOVPretriggerTimeS": 1.0}),
      ("onset_boost_8", {"VisionTurnSpeedControlFOVOnsetBoostFrames": 8.0}),
      ("disable_overshoot_window", {"VisionTurnSpeedControlFOVOvershootFrames": 0.0}),
    ]

    out_lines=[]
    for name,ov in cases:
      sim = simulate_hidden_turn(params_overrides=ov)
      m = metric_for(sim, t_occ_start)
      out_lines.append(f"{name}: v_chk={m['v_chk_mps']:.2f} m/s, drop_ge0.8_time={m['drop_ge0p8_s']}")
    out_path=os.path.join(os.path.dirname(__file__), 'ablation_summary.txt')
    with open(out_path,'w') as f:
      f.write('\n'.join(out_lines) + '\n')
    print('WROTE', out_path)

if __name__=='__main__':
    main()
