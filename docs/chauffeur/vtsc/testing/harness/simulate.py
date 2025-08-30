#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController, curvature_to_speed
from .scenarios import Scenario
from .generator import generate_timebase, generate_curvature, generate_confidence, generate_speed_limit, find_apex_index
from .metrics import compute_metrics


@dataclass
class SimResult:
    t: np.ndarray
    kappa: np.ndarray
    conf: np.ndarray
    v_clean: np.ndarray
    v_cmd: np.ndarray
    a_cmd: np.ndarray
    occluded: np.ndarray
    metrics: Dict[str, Any]
    # Visibility barrier helpers
    s_vis: np.ndarray
    d_req: np.ndarray
    v_vis: np.ndarray
    v_bound: np.ndarray


def _mk_sm(curvature: float, v_pred: float, confidence: float):
    # Provide minimal modelV2/carState for VTSC.update
    model = SimpleNamespace()
    # Fill horizon with constant values for simplicity
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


def _mk_vtsc_with_params(aggr: float, alpha: float, hyst: float, bias: float,
                         vis_horizon_s: float | None = None,
                         vis_margin_m: float | None = None,
                         gamma_per_m: float | None = None,
                         lat_jerk_cap: float | None = None) -> VisionTurnController:
    class MockCP: pass
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
        mp = MagicMock()
        mp.get_bool.return_value = True
        def _get(key: str):
            if key.endswith("Aggressiveness"):
                return str(aggr).encode()
            if key.endswith("FilterAlpha"):
                return str(alpha).encode()
            if key.endswith("HysteresisThreshold"):
                return str(hyst).encode()
            if key.endswith("SafetyBias"):
                return str(bias).encode()
            if key.endswith("VisHorizonS") and vis_horizon_s is not None:
                return str(vis_horizon_s).encode()
            if key.endswith("VisMarginM") and vis_margin_m is not None:
                return str(vis_margin_m).encode()
            if key.endswith("GammaPerMeter") and gamma_per_m is not None:
                return str(gamma_per_m).encode()
            if key.endswith("LatJerkCap"):
                # None means 'no cap' in tests; send sentinel -1 to controller
                val = -1.0 if lat_jerk_cap is None else float(lat_jerk_cap)
                return str(val).encode()
            # Allow defaults otherwise
            return None
        mp.get.side_effect = _get
        MockParams.return_value = mp
        return VisionTurnController(MockCP())


def simulate(scn: Scenario, alpha_bump_on_reacq: Optional[float] = None) -> SimResult:
    # Time, inputs
    t = generate_timebase(scn)
    kappa = generate_curvature(scn, t)
    conf = generate_confidence(scn, t)
    v_limit = generate_speed_limit(scn, t)
    dt = scn.dt

    vtsc = _mk_vtsc_with_params(
        scn.params.aggressiveness,
        scn.params.alpha,
        scn.params.hysteresis,
        scn.params.safety_bias,
        scn.vis_horizon_s,
        scn.vis_margin_m,
        scn.gamma_per_meter,
        scn.lat_jerk_cap,
    )

    # Latency buffer for curvature
    latency_steps = int(round(scn.latency_s / dt))
    kappa_buf = [kappa[0]] * max(1, latency_steps)

    v_cmd: List[float] = []
    a_cmd: List[float] = []
    occluded: List[bool] = []
    v_clean: List[float] = []

    v_ego = float(scn.v0_mps)
    a_ego = 0.0

    # Track reacquisition event for optional alpha bump
    was_good = conf[0] >= 0.75
    # Synthetic clock for time-based logic inside VTSC
    sim_time = 0.0

    # Arrays for barrier diagnostics
    s_vis_arr: List[float] = []
    d_req_arr: List[float] = []
    v_vis_arr: List[float] = []
    v_bound_arr: List[float] = []

    for i in range(len(t)):
        # Latency injected curvature
        k_in = kappa_buf[0] if latency_steps > 0 else kappa[i]
        if latency_steps > 0:
            kappa_buf.append(kappa[i])
            kappa_buf.pop(0)

        c_in = conf[i]
        vref = v_limit[i]

        sm = _mk_sm(k_in, v_ego, c_in)
        # Advance synthetic time (20 Hz)
        sim_time = float(t[i])
        # Patch time.time() and time.monotonic() inside the controller to use synthetic clock
        # Capture prev raw target before update for d_req alignment
        prev_raw_target = float(getattr(vtsc, '_prev_target_speed', v_ego))
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: sim_time), \
             patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: sim_time):
            vtsc.update(sm, True, v_ego, a_ego, vref)

        # Optional alpha bump on reacquisition
        is_good = vtsc._occlusion_state.vision_good
        if alpha_bump_on_reacq is not None and (not was_good) and is_good:
            vtsc._filter_alpha = float(alpha_bump_on_reacq)
        was_good = is_good

        # Log signals
        a = float(vtsc.a_target)
        a_cmd.append(a)
        occluded.append(not is_good)
        v_clean.append(float(min(curvature_to_speed(kappa[i]), vref)))

        # Compute barrier helpers for assertions
        try:
            v_vis_val = float(curvature_to_speed(max(1e-8, float(vtsc._occlusion_state.last_valid_curvature))))
            v_occ_val = float(curvature_to_speed(max(1e-8, float(vtsc._occlusion_state.est_curvature))))
        except Exception:
            v_vis_val = float(v_ego)
            v_occ_val = float(v_ego)
        base_target = float(min(vref, curvature_to_speed(max(1e-8, float(kappa[i])))))
        v_near_val = float(min(base_target, v_vis_val, vref))
        # Tail-aware far bound to mirror controller
        s_vis_val = float(max(0.0, getattr(vtsc, '_vis_horizon_s', 1.4) * max(0.0, v_ego)))
        try:
            dist_since = float(getattr(vtsc._occlusion_state, 'distance_since_m', 0.0))
        except Exception:
            dist_since = 0.0
        s_tail = max(0.0, dist_since - s_vis_val)
        # Curvature-aware fraction mapping: 0.10 below 0.004, 0.70 at 0.008
        try:
            k_now = float(getattr(vtsc._occlusion_state, 'est_curvature', 0.0))
        except Exception:
            k_now = 0.0
        if k_now <= 0.004:
            tail_frac = 0.10
        elif k_now >= 0.008:
            tail_frac = 0.90
        else:
            tail_frac = 0.10 + 0.80 * ((k_now - 0.004) / 0.004)
        a_cap = float(abs(getattr(vtsc, '_comfort_decel_limit', -1.47)))
        v_now = float(max(prev_raw_target, v_ego))
        try:
            v_cap_tail = float(np.sqrt(max(0.0, v_now * v_now - 2.0 * a_cap * (tail_frac * s_tail))))
        except Exception:
            v_cap_tail = v_now
        v_far_val = float(min(v_occ_val, v_cap_tail, vref))
        d_req_val = float(max(0.0, (v_now * v_now - v_far_val * v_far_val) / max(2e-3, 2.0 * a_cap)))
        v_bound_val = float(min(v_near_val, v_far_val))
        v_vis_arr.append(v_vis_val)
        v_bound_arr.append(v_bound_val)
        s_vis_arr.append(s_vis_val)
        d_req_arr.append(d_req_val)

        # Integrate acceleration to get commanded speed following sim
        v_ego = max(0.0, v_ego + a * dt)
        a_ego = a
        v_cmd.append(v_ego)

    # Compute apex index for post-apex metrics window
    apex_idx = find_apex_index(np.asarray(kappa))
    metrics = compute_metrics(t, v_cmd, v_clean, a_cmd, conf, occluded, apex_idx=apex_idx)
    return SimResult(
        t=np.asarray(t),
        kappa=np.asarray(kappa),
        conf=np.asarray(conf),
        v_clean=np.asarray(v_clean),
        v_cmd=np.asarray(v_cmd),
        a_cmd=np.asarray(a_cmd),
        occluded=np.asarray(occluded),
        metrics=metrics,
        s_vis=np.asarray(s_vis_arr),
        d_req=np.asarray(d_req_arr),
        v_vis=np.asarray(v_vis_arr),
        v_bound=np.asarray(v_bound_arr),
    )
