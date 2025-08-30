#!/usr/bin/env python3
"""
FOV-based occlusion fairness test for multi-curve chains.

Baseline assumptions:
- Flat road; camera optical axis aligned with vehicle tangent
- Camera horizontal FOV = 60° (±30° half-angle)
- Occlusion distance limited by FOV and local curvature: s_vis_fov = (π/6) / |kappa|
- No terrain/traffic/vegetation occluders (pure geometry)

This mirrors the structure of test_multi_occluded_curves.py but replaces
the positive-margin horizon with the FOV-limited horizon above.

We report physics-only minima, commanded minima, and basic invariants
using the FOV-derived visibility distance.
"""

import sys
import os
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, VTSCParams
from docs.chauffeur.vtsc.testing.harness.simulate import simulate
from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed, COMFORT_DECEL_LIMIT


class TestMultiOccludedCurvesFOV(unittest.TestCase):
    FOV_HALF_RAD = np.pi / 6.0  # 30° half-angle
    LANE_WIDTH_M = 11.0 * 0.3048  # 3.3528 m
    SHOULDER_WIDTH_M = 8.0 * 0.3048  # 2.4384 m (assumed)

    def _run_chain_fov(self, variant: str):
        if variant == 'mountain':
            bends_r = [180.0, 150.0, 210.0]
            straights = [0.6, 0.5, 0.3]
            start_v_mph = 55.0
        elif variant == 'highway':
            bends_r = [340.0, 360.0, 320.0]
            straights = [0.5, 0.4, 0.3]
            start_v_mph = 72.0
        else:
            raise AssertionError('unknown variant')

        # Piecewise segments
        segments = []
        segments.append({"duration_s": 1.0, "radius_m": 1e9, "sign": 1.0})
        signs = [1.0, -1.0, 1.0]
        for i, r in enumerate(bends_r):
            segments.append({"duration_s": 3.0, "radius_m": r, "sign": signs[i]})
            if i < len(straights):
                segments.append({"duration_s": straights[i], "radius_m": 1e9, "sign": 1.0})

        # Scenario (confidence fixed low to exercise occlusion)
        scn = Scenario(
            name=f'{variant}_chain_fov',
            duration_s=sum(s['duration_s'] for s in segments) + 0.5,
            dt=0.05,
            v0_mps=start_v_mph * 0.44704,
            geometry=GeometryProfile(kind='multi_curve', segments=segments),
            confidence=ConfidenceProfile(kind='stable', value=0.3),
            vis_horizon_s=1.0,  # not used in FOV gating below
            vis_margin_m=8.0,   # nominal margin
            gamma_per_meter=0.00020,
            lat_jerk_cap=2.0,
        )
        scn.params = VTSCParams(aggressiveness=1.0, alpha=0.3, hysteresis=0.15, safety_bias=0.1)

        # Run simulation once
        res = simulate(scn)
        t = res.t; kappa = res.kappa; v = res.v_cmd; v_clean = res.v_clean

        # Physics-only minima for info
        abs_k = []
        for seg in segments:
            if seg.get('radius_m', 1e9) < 1e8:
                abs_k.append(abs(1.0 / max(1e-6, seg['radius_m'])))
        phys_speeds = [curvature_to_speed(k) for k in abs_k]
        min_phys = float(min(phys_speeds)) if phys_speeds else 35.0

        # FOV-based visibility distance per step: s_vis_fov = θ / |k|
        k_eps = 1e-8
        s_vis_fov = np.where(np.abs(kappa) > k_eps, self.FOV_HALF_RAD / np.abs(kappa), 1e6)
        # Distance required to slow to local physics bound with comfort decel
        a_cap = abs(float(COMFORT_DECEL_LIMIT))
        v_now = v  # treat current commanded speed as the active setpoint
        v_ph = v_clean  # physics bound at this step (min of curvature vs limit)
        d_req = np.maximum(0.0, (v_now*v_now - v_ph*v_ph) / np.maximum(2e-3, 2.0*a_cap))

        # Positive-margin mask using FOV-based horizon and nominal margin
        M = 8.0
        pos_margin = d_req <= (s_vis_fov - M)

        # Compute per-bend minima within positive-margin windows
        idxs = [0]; acc = 0.0
        for seg in segments:
            acc += seg['duration_s']
            idxs.append(int(round(acc / scn.dt)))
        bend_windows = []
        for seg_idx in [1,3,5]:
            start = int(round(sum(s['duration_s'] for s in segments[:seg_idx]) / scn.dt))
            end = int(round(sum(s['duration_s'] for s in segments[:seg_idx+1]) / scn.dt))
            mask = np.zeros_like(v, dtype=bool)
            mask[start:end] = True
            mask &= pos_margin
            bend_windows.append(mask)
        bend_mins = []
        for mask in bend_windows:
            vals = v[mask]
            if vals.size:
                bend_mins.append(float(np.min(vals)))
        drift = float(max(bend_mins) - min(bend_mins)) if bend_mins else 0.0

        # Report summary
        print(f"[{variant}/FOV] min_phys={min_phys*2.237:.1f}mph | min_cmd={np.min(v)*2.237:.1f}mph | drift={drift:.3f}m/s | pos_margin_coverage={(np.mean(pos_margin)*100):.1f}%")

        # 500ms physics vs command readout
        conv = 2.2369362920544
        step = int(round(0.5 / scn.dt))
        print("time_s, physics_mph, controller_mph")
        for i in range(0, len(t), step):
            ts = float(t[i])
            vp = float(v_clean[i]) * conv
            vc = float(v[i]) * conv
            print(f"{ts:5.2f}, {vp:7.2f}, {vc:7.2f}")

    def _run_chain_fov_walls(self, variant: str):
        # geometry construction same as FOV-only
        if variant == 'mountain':
            bends_r = [180.0, 150.0, 210.0]
            straights = [0.6, 0.5, 0.3]
            start_v_mph = 55.0
        elif variant == 'highway':
            bends_r = [340.0, 360.0, 320.0]
            straights = [0.5, 0.4, 0.3]
            start_v_mph = 72.0
        else:
            raise AssertionError('unknown variant')

        segments = []
        segments.append({"duration_s": 1.0, "radius_m": 1e9, "sign": 1.0})
        signs = [1.0, -1.0, 1.0]
        for i, r in enumerate(bends_r):
            segments.append({"duration_s": 3.0, "radius_m": r, "sign": signs[i]})
            if i < len(straights):
                segments.append({"duration_s": straights[i], "radius_m": 1e9, "sign": 1.0})

        scn = Scenario(
            name=f'{variant}_chain_fov_walls',
            duration_s=sum(s['duration_s'] for s in segments) + 0.5,
            dt=0.05,
            v0_mps=start_v_mph * 0.44704,
            geometry=GeometryProfile(kind='multi_curve', segments=segments),
            confidence=ConfidenceProfile(kind='stable', value=0.3),
            vis_horizon_s=1.0,
            vis_margin_m=8.0,
            gamma_per_meter=0.00020,
            lat_jerk_cap=2.0,
        )
        scn.params = VTSCParams(aggressiveness=1.0, alpha=0.3, hysteresis=0.15, safety_bias=0.1)

        res = simulate(scn)
        t = res.t; kappa = res.kappa; v = res.v_cmd; v_clean = res.v_clean

        # Physics-only minima for info
        abs_k = []
        for seg in segments:
            if seg.get('radius_m', 1e9) < 1e8:
                abs_k.append(abs(1.0 / max(1e-6, seg['radius_m'])))
        phys_speeds = [curvature_to_speed(k) for k in abs_k]
        min_phys = float(min(phys_speeds)) if phys_speeds else 35.0

        # FOV-only horizon
        k_eps = 1e-8
        s_vis_fov = np.where(np.abs(kappa) > k_eps, self.FOV_HALF_RAD / np.abs(kappa), 1e6)

        # Wall-limited horizon (approximate circle geometry)
        w_lane = self.LANE_WIDTH_M
        w_sh = self.SHOULDER_WIDTH_M
        s_vis_wall = np.zeros_like(kappa, dtype=float)
        for i, k in enumerate(kappa):
            kk = float(abs(k))
            if kk < k_eps:
                s_vis_wall[i] = 1e6
                continue
            R_c = 1.0 / kk
            # Radii
            R_wall = max(1e-6, R_c - (w_lane + w_sh))
            # Camera/lane center radius differs by curvature sign
            R_cam = R_c - (w_lane / 2.0) if k > 0 else R_c + (w_lane / 2.0)
            if R_cam <= R_wall + 1e-6:
                s_vis_wall[i] = 1e6
                continue
            # Angle to tangency, approximate visible arc length along centerline
            ratio = np.clip(R_wall / max(R_cam, 1e-6), -1.0, 1.0)
            phi = float(np.arccos(ratio))
            s_vis_wall[i] = R_c * phi

        s_vis = np.minimum(s_vis_fov, s_vis_wall)

        # Positive-margin evaluation
        a_cap = abs(float(COMFORT_DECEL_LIMIT))
        v_now = v
        v_ph = v_clean
        d_req = np.maximum(0.0, (v_now*v_now - v_ph*v_ph) / max(2e-3, 2.0*a_cap))
        M = 8.0
        pos_margin = d_req <= (s_vis - M)

        # Per-bend minima (under margin)
        bend_windows = []
        for seg_idx in [1,3,5]:
            start = int(round(sum(s['duration_s'] for s in segments[:seg_idx]) / scn.dt))
            end = int(round(sum(s['duration_s'] for s in segments[:seg_idx+1]) / scn.dt))
            mask = np.zeros_like(v, dtype=bool)
            mask[start:end] = True
            mask &= pos_margin
            bend_windows.append(mask)
        bend_mins = []
        for mask in bend_windows:
            vals = v[mask]
            if vals.size:
                bend_mins.append(float(np.min(vals)))
        drift = float(max(bend_mins) - min(bend_mins)) if bend_mins else 0.0

        print(f"[{variant}/FOV+WALL] min_phys={min_phys*2.237:.1f}mph | min_cmd={np.min(v)*2.237:.1f}mph | drift={drift:.3f}m/s | pos_margin_coverage={(np.mean(pos_margin)*100):.1f}%")

        # 500ms physics vs command readout
        conv = 2.2369362920544
        step = int(round(0.5 / scn.dt))
        print("time_s, physics_mph, controller_mph")
        for i in range(0, len(t), step):
            ts = float(t[i])
            vp = float(v_clean[i]) * conv
            vc = float(v[i]) * conv
            print(f"{ts:5.2f}, {vp:7.2f}, {vc:7.2f}")

        # Basic invariants (sanity, not strict)
        self.assertGreater(len(t), 0)
        self.assertTrue(np.isfinite(min_phys))

    def test_mountain_chain_fov(self):
        self._run_chain_fov('mountain')

    def test_mountain_chain_fov_with_walls(self):
        self._run_chain_fov_walls('mountain')


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiOccludedCurvesFOV)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
