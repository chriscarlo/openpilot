#!/usr/bin/env python3
"""
Consecutive occluded turns with distance-aware barrier and reachable-raise.

Two variants:
- Highway sweeper chain: three bends, physics-only min ≈ 65 mph (r ≈ 320–360 m), mild S-S-R with 0.3–0.6 s straights.
- Mountain chain: three bends, physics-only min ≈ 40–50 mph (r ≈ 120–220 m), with 0.3–0.6 s straights.

For each variant, sweep barrier params:
  VisHorizonS ∈ {1.0, 1.4, 1.6}; VisMarginM ∈ {8, 12} m; GammaPerMeter ∈ {0.00020, 0.00030, 0.00035}.

Assertions per run:
- No downward motion with positive margin
- Reachable-raise is actually used (mean positive dv/dt ≥ 0.2 m/s² under positive margin and v_vis > v_now)
- Overslow drift per bend within caps (≤ 0.4 m/s highway, ≤ 0.6 m/s mountain)
- Jerk caps and invariants remain in bounds
"""

import sys
import os
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, VTSCParams
from docs.chauffeur.vtsc.testing.harness.simulate import simulate
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed


class TestMultiOccludedCurves(unittest.TestCase):
    def _run_chain(self, variant: str):
        # Build segments from radii (m) and straights (s)
        if variant == 'highway':
            # ~65 mph physics-only chain: radii 320–360 m, mild S-S-R
            bends_r = [340.0, 360.0, 320.0]
            straights = [0.5, 0.4, 0.3]
            start_v_mph = 72.0
            drift_cap = 0.4  # m/s
        elif variant == 'mountain':
            # ~40–50 mph physics-only chain: radii 120–220 m
            bends_r = [180.0, 150.0, 210.0]
            straights = [0.6, 0.5, 0.3]
            start_v_mph = 55.0
            drift_cap = 0.6  # m/s
        else:
            raise AssertionError('unknown variant')

        # Build piecewise segments: straight-in, bend1, straight, bend2, straight, bend3
        segments = []
        segments.append({"duration_s": 1.0, "radius_m": 1e9, "sign": 1.0})  # long straight
        signs = [1.0, -1.0, 1.0]  # S-S-R pattern approximation
        for i, r in enumerate(bends_r):
            segments.append({"duration_s": 3.0, "radius_m": r, "sign": signs[i]})
            if i < len(straights):
                segments.append({"duration_s": straights[i], "radius_m": 1e9, "sign": 1.0})

        # Calc physics-only mins for info
        k_list = []
        for seg in segments:
            if seg.get('radius_m', 1e9) < 1e8:
                k = abs(1.0 / max(1e-6, seg['radius_m']))
                k_list.append(k)
        phys_speeds = [curvature_to_speed(k) for k in k_list]
        min_phys = float(min(phys_speeds)) if phys_speeds else 35.0

        # Barrier parameter sweeps
        horizons = [1.0, 1.4, 1.6]
        margins = [8.0, 12.0]
        gammas = [0.00020, 0.00030, 0.00035]

        # Keep confidence < good throughout to exercise barrier (true occlusion)
        conf = ConfidenceProfile(kind='stable', value=0.3)

        # Run sweeps (use lat jerk cap for highway only)
        for H in horizons:
            for M in margins:
                for G in gammas:
                    cap = 2.0 if variant == 'highway' else None
                    scn = Scenario(
                        name=f'{variant}_chain',
                        duration_s= sum(s['duration_s'] for s in segments) + 0.5,
                        dt=0.05,
                        v0_mps=start_v_mph*0.44704,
                        geometry=GeometryProfile(kind='multi_curve', segments=segments),
                        confidence=conf,
                        vis_horizon_s=H,
                        vis_margin_m=M,
                        gamma_per_meter=G,
                        lat_jerk_cap=cap,
                    )
                    # Params: keep hysteresis at 0.15 to avoid masking recoveries
                    scn.params = VTSCParams(aggressiveness=1.0, alpha=0.3, hysteresis=0.15, safety_bias=0.1)

                    res = simulate(scn)
                    t = res.t; v = res.v_cmd; a = res.a_cmd
                    s_vis = res.s_vis; d_req = res.d_req
                    v_vis = res.v_vis; v_bound = res.v_bound
                    v_near = np.minimum(res.v_clean, v_vis)

                    # Positive-margin mask
                    pos_margin = (d_req <= (s_vis - M))

                    # 1) No downward with positive margin
                    dv = np.diff(v)
                    # Ignore first 2 ticks of each newly-entered positive-margin segment to allow jerk-limited neutralization
                    pm = pos_margin.copy()
                    starts = np.where((pm.astype(int)[1:] == 1) & (pm.astype(int)[:-1] == 0))[0] + 1
                    pm_mask = pm[1:].copy()
                    for s in starts:
                        pm_mask[s-1:s+1] = False  # drop first 2 samples inside the segment
                    if pm[0]:
                        pm_mask[0:2] = False  # also drop first 2 global samples if starting inside pos-margin
                    down_viol = int(np.sum((dv < -1e-6) & pm_mask))

                    # 2) Reachable-raise actually used: mean positive dv/dt under conditions
                    # Only consider windows where we're actually below the clean/physics bound (i.e., overslow)
                    cond = pm_mask & (v[:-1] < (res.v_clean[1:] - 0.2))
                    pos_dv_dt = (np.maximum(dv, 0.0) / scn.dt)[cond]
                    mean_pos_accel = float(np.mean(pos_dv_dt)) if pos_dv_dt.size > 0 else None

                    # 3) Overslow drift per bend in positive-margin windows
                    # Build masks per bend indices
                    # Map time spans using cumulative durations
                    idxs = [0]
                    acc = 0.0
                    for seg in segments:
                        acc += seg['duration_s']
                        idxs.append(int(round(acc / scn.dt)))
                    # bend segments are at indices 1,3,5 in segments
                    bend_windows = []
                    for bi, seg_idx in enumerate([1,3,5]):
                        start = int(round(sum(s['duration_s'] for s in segments[:seg_idx]) / scn.dt))
                        end = int(round(sum(s['duration_s'] for s in segments[:seg_idx+1]) / scn.dt))
                        mask = np.zeros_like(v, dtype=bool)
                        mask[start:end] = True
                        mask &= pos_margin
                        bend_windows.append(mask)

                    bend_mins = []
                    valid_bends = True
                    for mask in bend_windows:
                        vals = v[mask]
                        if vals.size < 3:
                            valid_bends = False
                            break
                        bend_mins.append(float(np.min(vals)))
                    overslow_drift = float(max(bend_mins) - min(bend_mins)) if valid_bends and bend_mins else 0.0

                    # 4) Jerk caps and invariants
                    m = res.metrics
                    jerk_pos = float(m['jerk_pos']); jerk_neg = float(m['jerk_neg'])

                    # Print summary for this sweep point
                    cap_str = 'cap=off' if cap is None else f'cap={cap:.1f}'
                    print(f"[{variant}] {cap_str} H={H:.2f} M={M:.1f} G={G:.5f} | min_phys={min_phys*2.237:.1f}mph | min_cmd={np.min(v)*2.237:.1f}mph | drift={overslow_drift:.3f}m/s | down_viol={down_viol} | jerk+={jerk_pos:.2f} jerk-={jerk_neg:.2f}")

                    # Assertions
                    self.assertEqual(down_viol, 0, msg=f"downward with positive margin for {variant} H={H} M={M} G={G} {cap_str}")
                    # Mean positive accel threshold applies only if windows exist
                    if mean_pos_accel is not None and np.any(cond):
                        self.assertGreaterEqual(mean_pos_accel, 0.2, msg=f"reachable-raise too flat: {mean_pos_accel:.2f} for {variant} H={H} M={M} G={G} {cap_str}")
                    # Overslow drift cap
                    if valid_bends:
                        self.assertLessEqual(overslow_drift, drift_cap, msg=f"overslow drift {overslow_drift:.2f} > cap {drift_cap} for {variant} {cap_str}")
                    # Jerk caps: must not exceed configured clamp (~2.5 m/s^3 positive), and reasonable negative bound
                    self.assertLessEqual(jerk_pos, 2.6)
                    self.assertGreaterEqual(jerk_neg, -6.5)

                    # 5) Within ±1 m/s of physics min (cap ON only)
                    self.assertLessEqual(abs(np.min(v) - min_phys), 1.0, msg=f"min_cmd vs min_phys out of ±1m/s for {variant} {cap_str}")

                    # 6) Track near bound when margin positive after 0.5s grace
                    # Identify positive-margin segment starts
                    starts = np.where((pos_margin[1:] == True) & (pos_margin[:-1] == False))[0] + 1
                    for sidx in starts:
                        eidx = sidx
                        while eidx < len(pos_margin) and pos_margin[eidx]:
                            eidx += 1
                        # Require at least 0.5s window
                        if eidx - sidx >= int(round(0.5 / scn.dt)):
                            k = sidx + int(round(0.5 / scn.dt))
                            self.assertLessEqual(abs(v[k] - v_near[k]), 0.2, msg=f"near-bound tracking off by {abs(v[k]-v_near[k]):.2f} for {variant} {cap_str}")

    def test_chains_sweep(self):
        self._run_chain('highway')
        self._run_chain('mountain')


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiOccludedCurves)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
