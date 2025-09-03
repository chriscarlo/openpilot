#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, math
from dataclasses import asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../openpilot"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, SpeedLimitProfile, VTSCParams
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController, curvature_to_speed

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def make_hidden_turn_scenario() -> Tuple[Scenario, float, float]:
    from opendbc.car.common.conversions import Conversions as CV
    v0_mph = 50.0
    v0_mps = v0_mph * CV.MPH_TO_MS
    v_safe_mph = 25.0
    v_safe_mps = v_safe_mph * CV.MPH_TO_MS

    # Find curvature for ~25 mph
    def k_for_v(v_target_mps: float) -> float:
        lo, hi = 1e-5, 0.05
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            v_mid = curvature_to_speed(mid)
            if v_mid > v_target_mps:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    k_tight = k_for_v(v_safe_mps)
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
    t_occ_end = straight_s + bend_s + 1.0

    scn = Scenario(
        name='abrupt_hidden_turn_realistic',
        duration_s=t_occ_end,
        dt=0.05,
        v0_mps=v0_mps,
        geometry=GeometryProfile(kind='multi_curve', segments=segments),
        confidence=ConfidenceProfile(kind='window', value=0.3, window_start_s=t_occ_start, window_end_s=t_occ_end),
        speed_limit=SpeedLimitProfile(kind='none', start_mps=v0_mps),
    )
    scn.vis_horizon_s = 1.4
    scn.vis_margin_m = 12.0
    scn.gamma_per_meter = 0.00020
    scn.lat_jerk_cap = 2.0
    return scn, t_occ_start, v_safe_mps


def _mk_sm(curvature: float, v_pred: float, confidence: float):
    # Minimal SM to satisfy controller
    from types import SimpleNamespace
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


def run_vtsc_sim(scn: Scenario, *, tweaks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from unittest.mock import MagicMock, patch
    # Create controller with patched Params, using defaults for most keys
    with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.Params') as MockParams:
        mp = MagicMock()
        # By default, enable VTSC in tests
        mp.get_bool.return_value = True
        mp.get.return_value = None
        MockParams.return_value = mp
        ctrl = VisionTurnController(object())

    # Apply tweaks on controller instance if provided
    if tweaks:
        for k, v in tweaks.items():
            setattr(ctrl, k, v)

    # Time and inputs
    from docs.chauffeur.vtsc.testing.harness.generator import generate_timebase, generate_curvature, generate_confidence, generate_speed_limit
    from sunnypilot.selfdrive.controls.lib.vision_turn_controller import time as ctrl_time

    t = generate_timebase(scn)
    kappa = generate_curvature(scn, t)
    conf = generate_confidence(scn, t)
    v_limit = generate_speed_limit(scn, t)
    dt = scn.dt

    v_cmd: List[float] = []
    a_cmd: List[float] = []
    v_clean: List[float] = []
    occluded: List[bool] = []

    # Snapshot capture
    snaps: List[Dict[str, Any]] = []

    v_ego = float(scn.v0_mps)
    a_ego = 0.0

    latency_steps = 0
    kappa_buf = [kappa[0]] * max(1, latency_steps)

    occ_on_idx = None
    occ_off_idx = None

    for i, ti in enumerate(t):
        k_in = kappa_buf[0] if latency_steps > 0 else kappa[i]
        if latency_steps > 0:
            kappa_buf.append(kappa[i]); kappa_buf.pop(0)
        c_in = conf[i]
        vref = v_limit[i]
        sm = _mk_sm(k_in, v_ego, c_in)
        # step controller with synthetic time
        with patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time', lambda: float(ti)), \
             patch('sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic', lambda: float(ti)):
            ctrl.update(sm, True, v_ego, a_ego, vref)
            snap = ctrl.snapshot_debug_state()
            snap['t'] = float(ti)
            snaps.append(snap)
        # Derived signals
        a = float(ctrl.a_target)
        a_cmd.append(a)
        is_occ = bool(snap.get('occluded', False))
        occluded.append(is_occ)
        v_clean.append(float(min(curvature_to_speed(kappa[i]), vref)))

        # Integrate to get v_cmd
        v_ego = max(0.0, v_ego + a * dt)
        a_ego = a
        v_cmd.append(v_ego)

        # Track onset/clear indices from snapshot occluded flag
        if occ_on_idx is None and is_occ:
            occ_on_idx = i
        if occ_on_idx is not None and occ_off_idx is None and (not is_occ) and i > occ_on_idx:
            occ_off_idx = i

    # Compute check index per test rule (4.5 s after onset)
    dt = scn.dt
    if occ_on_idx is None:
        occ_on_idx = int(len(t) // 2)
    chk_idx = min(len(t) - 1, int(round(occ_on_idx + 4.5 / dt)))

    # Metrics
    v0 = float(scn.v0_mps)
    v_cmd_arr = np.asarray(v_cmd)
    v_clean_arr = np.asarray(v_clean)
    def first_time_drop(thresh: float) -> Optional[float]:
        for j in range(occ_on_idx, len(t)):
            if v_cmd_arr[j] <= v0 - thresh:
                return float((j - occ_on_idx) * dt)
        return None

    metrics = {
        'event': 'hidden_turn_sim',
        't_occ_on': float(t[occ_on_idx]) if occ_on_idx is not None else None,
        't_occ_off': float(t[occ_off_idx]) if occ_off_idx is not None else None,
        'v_cmd_at_onset': float(v_cmd_arr[occ_on_idx]) if occ_on_idx is not None else None,
        'v_cmd_at_check': float(v_cmd_arr[chk_idx]),
        'v_target_at_check': float(v_clean_arr[chk_idx]),
        'drop_ge_0p8_s': first_time_drop(0.8),
        'drop_ge_2p0_s': first_time_drop(2.0),
        'passes_4p5s_rule': bool(v_cmd_arr[chk_idx] <= v0 - 0.8),
        'psi_margin_deg': math.degrees(getattr(ctrl, '_psi_margin_rad', 0.1222)),
        'kappa_gate_onset': float(snaps[occ_on_idx].get('kappa_vis', 0.0)) if occ_on_idx is not None else None,
        'ewma_kappa_onset': float(getattr(ctrl, '_fov_kappa_ewma', 0.0)) if hasattr(ctrl, '_fov_kappa_ewma') else None,
        'chk_idx': int(chk_idx),
        'occ_idx': int(occ_on_idx),
        'dt': float(dt),
    }

    return {
        't': t.tolist(),
        'kappa': np.asarray(kappa).tolist(),
        'v_cmd': v_cmd,
        'v_clean': v_clean,
        'occluded': [bool(x) for x in occluded],
        'snaps': snaps,
        'metrics': metrics,
    }


def plot_hidden_turn(out_dir: str, sim: Dict[str, Any], label: str) -> List[str]:
    paths: List[str] = []
    if not HAVE_MPL:
        return paths
    t = np.asarray(sim['t'])
    v_cmd = np.asarray(sim['v_cmd'])
    v_clean = np.asarray(sim['v_clean'])
    kappa = np.asarray(sim['kappa'])
    snaps = sim['snaps']
    psi_vis = np.asarray([s.get('psi_vis', None) for s in snaps], dtype=float)
    psi_thresh = np.asarray([s.get('psi_thresh', None) for s in snaps], dtype=float)
    ttfov = np.asarray([s.get('ttfov_s', None) for s in snaps], dtype=float)
    occ = np.asarray([1 if s.get('occluded', False) else 0 for s in snaps])
    onset_boost = np.asarray([s.get('onset_boost_left', 0) for s in snaps], dtype=float)
    overshoot_left = np.asarray([s.get('overshoot_left', 0) for s in snaps], dtype=float)

    # Plot speeds
    plt.figure(figsize=(10,6))
    plt.plot(t, v_cmd, label='v_cmd')
    plt.plot(t, v_clean, label='v_target')
    plt.twinx()
    plt.plot(t, occ*5, 'r--', alpha=0.3, label='occluded*5')
    plt.title(f'Hidden Turn Speeds — {label}')
    plt.xlabel('time (s)'); plt.ylabel('speed (m/s)')
    plt.legend(loc='upper right')
    p1 = os.path.join(out_dir, f'{label}_speeds.png'); plt.savefig(p1); plt.close(); paths.append(p1)

    # Curvature + ttfov/psi
    plt.figure(figsize=(10,6))
    plt.plot(t, kappa, label='kappa')
    if np.isfinite(psi_vis).any():
        plt.plot(t, psi_vis, label='psi_vis')
    if np.isfinite(psi_thresh).any():
        plt.plot(t, psi_thresh, label='psi_thresh')
    if np.isfinite(ttfov).any():
        plt.plot(t, ttfov, label='ttfov_s')
    plt.title(f'Hidden Turn Curvature/TTFOV — {label}')
    plt.xlabel('time (s)'); plt.ylabel('value')
    plt.legend()
    p2 = os.path.join(out_dir, f'{label}_curv_ttfov.png'); plt.savefig(p2); plt.close(); paths.append(p2)

    # Boost/overshoot windows
    plt.figure(figsize=(10,4))
    plt.step(t, onset_boost, where='post', label='onset_boost_left')
    plt.step(t, overshoot_left, where='post', label='overshoot_left')
    plt.title(f'Hidden Turn Windows — {label}')
    plt.xlabel('time (s)'); plt.ylabel('frames')
    plt.legend()
    p3 = os.path.join(out_dir, f'{label}_windows.png'); plt.savefig(p3); plt.close(); paths.append(p3)

    return paths


def freeway_no_crawl_check() -> Dict[str, Any]:
    # Simple straight freeway scenario at 70 mph
    from opendbc.car.common.conversions import Conversions as CV
    v0_mps = 70.0 * CV.MPH_TO_MS
    scn = Scenario(
        name='freeway_straight',
        duration_s=10.0,
        dt=0.05,
        v0_mps=v0_mps,
        geometry=GeometryProfile(kind='constant', kappa0=0.0),
        confidence=ConfidenceProfile(kind='stable', value=0.9),
        speed_limit=SpeedLimitProfile(kind='none', start_mps=v0_mps),
    )
    base = run_vtsc_sim(scn)
    v_cmd = np.asarray(base['v_cmd'])
    vmin = float(np.min(v_cmd))
    return { 'vmin_cmd_mps': vmin, 'v0_mps': float(v0_mps) }


def main():
    outdir = os.environ.get('OUTDIR', '/data/out/vtsc_reports/today')
    plots_dir = os.path.join(outdir, 'plots'); os.makedirs(plots_dir, exist_ok=True)
    snaps_dir = os.path.join(outdir, 'snapshot_dumps'); os.makedirs(snaps_dir, exist_ok=True)

    # Baseline hidden turn
    scn, t_occ_start, v_safe_mps = make_hidden_turn_scenario()
    base = run_vtsc_sim(scn)
    with open(os.path.join(outdir, 'hidden_turn_baseline_metrics.json'), 'w') as f:
        json.dump(base['metrics'], f, indent=2)
    # Snapshot dump around onset window
    occ_idx = base['metrics']['occ_idx']
    dt = base['metrics']['dt']
    lo = max(0, int(round(occ_idx - 0.5/dt)))
    hi = min(len(base['snaps'])-1, int(round(occ_idx + 6.0/dt)))
    with open(os.path.join(outdir, 'snapshot_dumps', 'sim_hidden_turn_onset_window.json'), 'w') as f:
        json.dump(base['snaps'][lo:hi+1], f)

    # Plots
    plot_hidden_turn(plots_dir, base, label='sim_hidden_turn')

    # Ablations
    tweaks = [
        ('psi_margin+2deg', {'_psi_margin_rad': math.radians(9.0)}),
        ('pretrigger_1s', {'_fov_pretrigger_time_s': 1.0}),
        ('onset_boost_8', {'_fov_onset_boost_frames': 8}),
        ('overshoot_off', {'_fov_overshoot_frames': 0}),
    ]
    lines: List[str] = []
    best_name = None
    best_ok = False
    best_drop = -1e9
    for name, tw in tweaks:
        sim = run_vtsc_sim(scn, tweaks=tw)
        m = sim['metrics']
        v_drop = float(base['metrics']['v_cmd_at_check'] - m['v_cmd_at_check'])
        ok = bool(m['passes_4p5s_rule'])
        lines.append(f"{name}: pass4.5={ok} v_cmd_at_check={m['v_cmd_at_check']:.3f} m/s (Δ={v_drop:.3f}) drop_ge_0.8_s={m['drop_ge_0p8_s']}")
        # Track best by pass first then largest drop
        score = (1 if ok else 0, v_drop)
        if best_name is None or score > ((1 if best_ok else 0), best_drop):
            best_name, best_ok, best_drop = name, ok, v_drop

    with open(os.path.join(outdir, 'ablation_summary.txt'), 'w') as f:
        for ln in lines:
            f.write(ln + "\n")
        f.write(f"BEST={best_name} pass={best_ok} Δv_cmd_at_check={best_drop:.3f}\n")

    # Freeway check with best tweak (approximate)
    freeway = freeway_no_crawl_check()
    with open(os.path.join(outdir, 'freeway_check.json'), 'w') as f:
        json.dump(freeway, f, indent=2)

    # CSV summary
    import csv
    csv_path = os.path.join(outdir, 'metrics_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rlog_or_event','event_time','occlusion_onset_time','occlusion_clear_time','time_to_first_significant_decel_0p8','v_cmd_at_onset','v_cmd_at_check','v_target_at_check','time_between_onset_and_vcmd_drop_0p8','passes_4p5s_timing','freeway_no_crawl_pass','psi_margin_deg','kappa_gate_onset','EWMA_kappa_at_onset'])
        m = base['metrics']
        freeway_pass = bool(freeway['vmin_cmd_mps'] >= max(20.1168, 0.85*freeway['v0_mps']))
        w.writerow([
            'sim_hidden_turn', '', m.get('t_occ_on'), m.get('t_occ_off'), m.get('drop_ge_0p8_s'),
            m.get('v_cmd_at_onset'), m.get('v_cmd_at_check'), m.get('v_target_at_check'), m.get('drop_ge_0p8_s'),
            m.get('passes_4p5s_rule'), freeway_pass, m.get('psi_margin_deg'), m.get('kappa_gate_onset'), m.get('ewma_kappa_onset')
        ])

    print("Hidden-turn baseline metrics:")
    print(json.dumps(base['metrics'], indent=2))
    print("Ablation summary saved.")

if __name__ == '__main__':
    main()
