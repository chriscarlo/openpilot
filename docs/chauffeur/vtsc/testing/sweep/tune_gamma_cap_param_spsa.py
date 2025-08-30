#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric tuner for speed-based gamma cap used by the Vision Turn Speed Controller (VTSC).

Drop this file at:
  docs/chauffeur/vtsc/testing/sweep/tune_gamma_cap_param_spsa.py

Key features
------------
- Runtime-only monkey patch: wraps VisionOcclusionState.update to inject a parametric gamma cap g(v; θ)
- Parametric family: g(v; θ) = d + a / (b + v^c), with positivity/monotonicity via softplus reparameterization
- High-speed inertial guard: d >= d_min = s * (J_lat_min / v_hi^3)
- Objective: CVaR@95 of |min(v_cmd) - min(v_clean)| + penalties (PM downward events, jerk, overslow drift, smoothness)
- Optimizer: SPSA with a tiny L∞ trust region, active-set refresh, early stopping + verification
- No external deps beyond numpy/stdlib. Optional parallelism (ProcessPoolExecutor).

CLI
---
  --band {low,mid,high,vhigh,full}  speed band for scenario generation (default: full)
  --iters N                         SPSA iterations (default: 45)
  --vhi FLOAT                       high-speed threshold for inertial guard (default: 32.0)
  --pm-clamp                        enable optional jerk-safe PM clamp (off by default)
  --save PATH                       output JSON path (default: docs/chauffeur/vtsc/testing/sweep/tuned_gamma_param_spsa.json)
  --seed INT                        random seed (default: 123)
  --quick-verify                    run a reduced verifier at the end
  --full-verify                     run the full verifier across 5..45 m/s, all families/margins

Acceptance target
-----------------
Max |min_cmd − min_phys| ≤ 1–2 m/s; penalties must be zero.

This script follows the repo integration points and harness API described in the VTSC docs.
"""

from __future__ import annotations
import os
import sys
import math
import json
import time
import argparse
import random
import itertools
import contextlib
import importlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# --------------------------------------------------------------------------------------
# PATH injection (repo root)
# --------------------------------------------------------------------------------------

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Harness and controller imports (per provided integration points)
try:
    from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, VTSCParams
    from docs.chauffeur.vtsc.testing.harness.simulate import simulate  # returns SimResult with fields used below
    from sunnypilot.selfdrive.controls.lib.vision_turn_controller import curvature_to_speed  # noqa: F401 (ensures module loaded)
except Exception as e:
    # Helpful message if the repo layout differs
    raise ImportError(
        f"Failed to import harness/controller modules. Ensure this file is placed at:\n"
        f"  docs/chauffeur/vtsc/testing/sweep/tune_gamma_cap_param_spsa.py\n"
        f"and the repo root is correctly inferred.\nOriginal error: {e}"
    )

# --------------------------------------------------------------------------------------
# Utilities: numerics & mapping from free params φ to θ = (a,b,c,d)
# --------------------------------------------------------------------------------------

def softplus(x: np.ndarray | float) -> np.ndarray | float:
    """
    Numerically-stable softplus for scalars/arrays.
    softplus(x) = log(1 + exp(x)) but stable for large magnitude x.
    """
    x = np.asarray(x)
    # log1p(exp(-|x|)) + max(x, 0) is stable
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


@dataclass
class Theta:
    a: float
    b: float
    c: float
    d: float
    d_min: float
    v_hi: float

    def to_dict(self) -> Dict[str, float]:
        return {"a": float(self.a), "b": float(self.b), "c": float(self.c), "d": float(self.d),
                "d_min": float(self.d_min), "v_hi": float(self.v_hi)}


def phi_to_theta(phi: np.ndarray, v_hi: float = 32.0, s: float = 1.05, J_lat_min: float = 2.0) -> Theta:
    """
    Map unconstrained φ (R^4) to constrained θ (a,b,c,d) ensuring positivity and monotonicity.
    g(v; θ) = d + a / (b + v^c) with a,b,c,d ≥ 0 and c ≥ 1; d ≥ d_min.

    High-speed inertial guard:
      d_min = s * (J_lat_min / v_hi^3)
    """
    if len(phi) != 4:
        raise ValueError("phi must be length 4")
    phi = np.asarray(phi, dtype=float)
    # Scale parameters so g(v) sits in ~1e-6..3e-4 range to actually act below gamma_per_meter
    a_scale = 3.0e-4
    b_scale = 10.0
    d_scale = 2.0e-5

    a = a_scale * (softplus(phi[0]) ** 2)
    b = b_scale * softplus(phi[1])
    c = 1.0 + softplus(phi[2])
    d_min = s * (J_lat_min / max(v_hi, 1e-3) ** 3)
    d = float(d_min + d_scale * softplus(phi[3]))
    return Theta(a=float(a), b=float(b), c=float(c), d=d, d_min=float(d_min), v_hi=float(v_hi))


def gamma_cap_speed_param(v: np.ndarray | float, th: Theta) -> np.ndarray | float:
    """
    g(v; θ) = d + a / (b + v^c)
    Monotone decreasing in v. Supports scalar or vector v.
    """
    v = np.asarray(v, dtype=float)
    # Avoid negative or NaN speeds
    v = np.maximum(v, 0.0)
    return th.d + th.a / (th.b + np.power(v, th.c))


# --------------------------------------------------------------------------------------
# VTSC monkey patching: wrap VisionOcclusionState.update and temporarily override gamma_cap_speed
# --------------------------------------------------------------------------------------

class _VTSCMonkeyPatch:
    """
    Installs a runtime-only patch around VisionOcclusionState.update that temporarily overrides
    gamma_cap_speed(v) with the parametric g(v; θ). The patch is active for the duration of each
    update call and otherwise leaves the controller fully unchanged.

    Note: We do NOT modify master files; this is pure runtime monkey-patching.
    """

    def __init__(self, phi: np.ndarray, v_hi: float = 32.0, jerk_cap_guard: bool = False):
        self._mod = importlib.import_module("sunnypilot.selfdrive.controls.lib.vision_turn_controller")
        self._orig_update = getattr(self._mod.VisionOcclusionState, "update")
        self._had_gamma_cap = hasattr(self._mod, "gamma_cap_speed")
        self._orig_gamma_cap_fn = getattr(self._mod, "gamma_cap_speed", None)

        self._phi = np.asarray(phi, dtype=float).copy()
        self._theta = phi_to_theta(self._phi, v_hi=v_hi)
        self._jerk_cap_guard = bool(jerk_cap_guard)

        # Install the wrapper method
        def _wrapped_update(this, *args, **kwargs):
            """
            Wrapper that temporarily replaces gamma_cap_speed(v) during this update call.
            We purposely *do not* attempt to reimplement update; we rely on the original logic
            for state transitions/dwell, visible horizon, tail windows, etc.

            If --pm-clamp is enabled via a separate install, that patch is applied in an outer
            wrapper after this function returns (best-effort, jerk-safe).
            """
            # Define the parametric cap callable bound to current θ
            def _param_gamma_cap(v):
                return float(gamma_cap_speed_param(v, self._theta)) if np.ndim(v) == 0 else gamma_cap_speed_param(v, self._theta)

            # Call original update first to preserve all controller logic
            out = self._orig_update(this, *args, **kwargs)

            # Replace only the tail growth gamma cap with our parametric g(v;θ)
            try:
                # Unpack args (current_curvature, vision_confidence, v_ego, tm)
                if len(args) >= 4:
                    current_curvature = float(args[0])
                    vision_confidence = float(args[1])
                    v_ego = float(args[2])
                    tm = float(args[3])
                else:
                    # kwargs fallback (best-effort)
                    current_curvature = float(kwargs.get('current_curvature', 0.0))
                    vision_confidence = float(kwargs.get('vision_confidence', 1.0))
                    v_ego = float(kwargs.get('v_ego', 0.0))
                    tm = float(kwargs.get('tm', 0.0))

                # Conditions to apply growth-only override (occluded, monotonic mode, growth trend)
                if (not getattr(this, 'vision_good', True)) and getattr(this, 'mode_monotonic', True) and (getattr(this, 'trend_sign', 0) >= 0):
                    # Visible horizon and tail distance window similar to controller
                    s_vis = max(0.0, float(getattr(this, 'vis_horizon_s', 1.2)) * max(0.0, v_ego))
                    dist_since = float(getattr(this, 'distance_since_m', 0.0))
                    s_tail_raw = max(0.0, dist_since - s_vis)

                    # Tail window allowance based on curvature and speed
                    entry_k = float(max(0.0, getattr(this, 'entry_curvature', 0.0)))
                    est_k = float(max(0.0, getattr(this, 'est_curvature', entry_k)))
                    k_now_for_window = max(entry_k, est_k)
                    if k_now_for_window <= 0.0035:
                        t_allow = 0.10
                    else:
                        t_allow = 1.20
                    t_allow = max(0.10, min(t_allow, float(getattr(this, 'envelope_horizon_s', 1.2))))
                    s_tail_allow = max(0.0, v_ego) * t_allow
                    s_tail = min(s_tail_raw, s_tail_allow)

                    # Only apply inside tail window
                    if s_tail > 0.0:
                        # Jerk-based cap
                        v_cap_speed = 22.0
                        eff_v = min(max(0.0, v_ego), v_cap_speed)
                        lat_jerk_cap = float(getattr(this, 'lat_jerk_cap', 2.0))
                        gamma_cap_jerk = lat_jerk_cap / max(eff_v**3, 1e-3)
                        # Parametric speed cap
                        gamma_cap_spd = gamma_cap_speed_param(eff_v, self._theta)
                        # Net gamma (respect original gamma_per_m limit)
                        gamma_eff = float(min(max(0.0, getattr(this, 'gamma_per_m', 5e-4)), gamma_cap_jerk, gamma_cap_spd))
                        # Apply growth from entry curvature
                        est_new = max(0.0, entry_k + gamma_eff * s_tail)
                        try:
                            setattr(this, 'est_curvature', est_new)
                        except Exception:
                            pass

            except Exception:
                # Be conservative: never let patch errors break controller update
                return out

            return out

        # Bind the wrapper as a method
        setattr(self._mod.VisionOcclusionState, "update", _wrapped_update)

    def update_phi(self, phi: np.ndarray, v_hi: Optional[float] = None):
        """Update φ (and θ) for subsequent update calls without reinstalling the patch."""
        self._phi = np.asarray(phi, dtype=float).copy()
        if v_hi is None:
            v_hi = self._theta.v_hi
        self._theta = phi_to_theta(self._phi, v_hi=v_hi)

    def uninstall(self):
        """Restore the original VisionOcclusionState.update and gamma_cap_speed (if any)."""
        try:
            setattr(self._mod.VisionOcclusionState, "update", self._orig_update)
        except Exception:
            pass
        if self._had_gamma_cap:
            try:
                setattr(self._mod, "gamma_cap_speed", self._orig_gamma_cap_fn)
            except Exception:
                pass


def install_occlusion_patch(phi: np.ndarray, v_hi: float = 32.0, jerk_cap_guard: bool = False) -> _VTSCMonkeyPatch:
    """
    Convenience function to install the occlusion patch. Returns a handle providing update_phi(...) and uninstall().
    """
    return _VTSCMonkeyPatch(phi=phi, v_hi=v_hi, jerk_cap_guard=jerk_cap_guard)


# --------------------------------------------------------------------------------------
# Optional PM clamp (off by default)
#   Best-effort, jerk-safe post-update clamp to avoid negative accel when PM=true.
#   It's conservative and only attempts to modify common attribute names.
# --------------------------------------------------------------------------------------

class _PMClampPatch:
    def __init__(self, dt: float = 0.05, jerk_limit_pos: float = 2.5):
        self._mod = importlib.import_module("sunnypilot.selfdrive.controls.lib.vision_turn_controller")
        self._orig_update = getattr(self._mod.VisionOcclusionState, "update")
        self._dt = float(dt)
        self._jerk_plus = float(jerk_limit_pos)

        def _wrapped(this, *args, **kwargs):
            # Call original update first
            out = self._orig_update(this, *args, **kwargs)

            # Best-effort retrieval of accel target attribute on state
            a_names = ["a_target", "a_vtsc", "a_cmd", "accel_cmd", "accel_target"]
            a_val = None
            for nm in a_names:
                if hasattr(this, nm):
                    try:
                        a_val = float(getattr(this, nm))
                        a_attr = nm
                        break
                    except Exception:
                        continue

            # Best-effort detection of positive margin (PM) on the state (mirror harness logic when possible)
            is_pm = False
            for nm_dreq in ["d_req", "distance_required_m", "dist_req"]:
                for nm_svis in ["s_vis", "visible_distance_m", "s_visible"]:
                    if hasattr(this, nm_dreq) and hasattr(this, nm_svis) and hasattr(this, "vis_margin_m"):
                        try:
                            d_req = float(getattr(this, nm_dreq))
                            s_vis = float(getattr(this, nm_svis))
                            M = float(getattr(this, "vis_margin_m"))
                            if d_req <= (s_vis - M):
                                is_pm = True
                        except Exception:
                            pass

            # If PM and negative accel, jerk-safe ramp toward 0
            if is_pm and (a_val is not None) and (a_val < 0.0):
                a_new = min(a_val + self._jerk_plus * self._dt, 0.0)
                try:
                    setattr(this, a_attr, a_new)
                except Exception:
                    pass

            return out

        setattr(self._mod.VisionOcclusionState, "update", _wrapped)

    def uninstall(self):
        try:
            setattr(self._mod.VisionOcclusionState, "update", self._orig_update)
        except Exception:
            pass


def install_margin_gate_patch(dt: float = 0.05, jerk_limit_pos: float = 2.5) -> _PMClampPatch:
    """
    Optional jerk-safe PM clamp wrapper (OFF by default). Best-effort implementation; if state fields
    differ, it will no-op. Prefer leaving this disabled unless you explicitly need testing of PM downward
    suppression in the controller path.
    """
    return _PMClampPatch(dt=dt, jerk_limit_pos=jerk_limit_pos)


# --------------------------------------------------------------------------------------
# Scenario spec and builders (programmatic generation across families & margins)
# --------------------------------------------------------------------------------------

@dataclass
class ScenarioSpec:
    """A picklable, repo-agnostic description from which we can reconstruct a harness Scenario."""
    family: str                 # 'const_sweeper' | 'const_moderate' | 'const_tight' | 's_curve' | 'tightening'
    v0_mps: float
    dt: float
    duration_s: float
    vis_horizon_s: float
    vis_margin_m: float         # in {8.0, 12.0}
    gamma_per_meter: float
    lat_jerk_cap: float         # 2.0 .. 2.5
    # Geometry parameters (store radii/curvatures generically)
    # For 'const_*': radius_m
    # For 's_curve': r1_m, r2_m, straight_m
    # For 'tightening': r0_m, r1_m, length_m
    radius_m: Optional[float] = None
    r1_m: Optional[float] = None
    r2_m: Optional[float] = None
    straight_m: Optional[float] = None
    r0_m: Optional[float] = None
    length_m: Optional[float] = None


def _call_first_available(obj, candidates: List[Tuple[str, Dict[str, Any]]]):
    """Deprecated fallback helper (kept for compatibility)."""
    for name, kwargs in candidates:
        if hasattr(obj, name):
            fn = getattr(obj, name)
            try:
                return fn(**kwargs)
            except Exception:
                continue
    raise AttributeError(f"No suitable constructor found among: {[n for n, _ in candidates]}")


def _build_geometry(profile: GeometryProfile, spec: ScenarioSpec) -> Any:
    """Construct a GeometryProfile using the harness dataclass API."""
    if spec.family.startswith("const_"):
        R = float(spec.radius_m)
        kappa = 1.0 / max(R, 1e-3)
        return GeometryProfile(kind='constant', kappa0=float(kappa))

    if spec.family == "s_curve":
        R1 = float(spec.r1_m); R2 = float(spec.r2_m)
        S = float(spec.straight_m)
        k1 = 1.0 / max(R1, 1e-3)
        k2 = 1.0 / max(R2, 1e-3)
        # generator uses kappa0 for first bend (positive) and -abs(kappa1) for second
        return GeometryProfile(kind='s_curve', kappa0=float(k1), kappa1=float(k2), mid_straight_s=float(S))

    if spec.family == "tightening":
        R0 = float(spec.r0_m); R1 = float(spec.r1_m)
        k0 = 1.0 / max(R0, 1e-3)
        k1 = 1.0 / max(R1, 1e-3)
        # length_s is not directly used by generator for 'tightening', but store for completeness
        L = float(spec.length_m) if spec.length_m is not None else 150.0
        return GeometryProfile(kind='tightening', kappa0=float(k0), kappa1=float(k1), length_s=float(L))

    raise ValueError(f"Unknown family: {spec.family}")


def _build_scenario(spec: ScenarioSpec) -> Scenario:
    """Reconstruct a Scenario from ScenarioSpec using the harness dataclasses."""
    geom = _build_geometry(GeometryProfile, spec)
    # Induce an occlusion window covering most of the scenario to exercise VTSC
    w_start = 0.5
    w_end = max(1.0, float(spec.duration_s) - 0.5)
    confidence = ConfidenceProfile(kind='window', value=0.4, window_start_s=w_start, window_end_s=w_end)
    vtsc = VTSCParams(aggressiveness=1.0, alpha=0.3, hysteresis=0.15, safety_bias=0.1)

    name = f"{spec.family}_v{int(round(spec.v0_mps))}_M{int(round(spec.vis_margin_m))}"
    return Scenario(
        name=name,
        duration_s=float(spec.duration_s),
        dt=float(spec.dt),
        v0_mps=float(spec.v0_mps),
        geometry=geom,
        confidence=confidence,
        params=vtsc,
        vis_horizon_s=float(spec.vis_horizon_s),
        vis_margin_m=float(spec.vis_margin_m),
        gamma_per_meter=float(spec.gamma_per_meter),
        lat_jerk_cap=float(spec.lat_jerk_cap),
    )


def _band_to_speed_range(band: str) -> Tuple[float, float]:
    band = band.strip().lower()
    if band == "low":
        return 5.0, 15.0
    if band == "mid":
        return 15.0, 25.0
    if band == "high":
        return 25.0, 35.0
    if band == "vhigh":
        return 35.0, 45.0
    return 5.0, 45.0  # default full


def make_scenarios_for_speed(v0: float, dt: float = 0.05) -> List[Tuple[str, ScenarioSpec]]:
    """
    Produce a set of ScenarioSpec across families for a given start speed v0 (m/s).
    Families:
      - const_sweeper (R=350..450 m)
      - const_moderate (R=200..250 m)
      - const_tight (R=120..180 m)
      - s_curve (two bends with short straight)
      - tightening (kappa0→kappa1 over a length)
    Each returned tuple is (family, spec).
    """
    rng = random.Random(int((v0 * 1000) % 2**31))
    # radii with slight jitter for coverage
    R_sweeper = rng.uniform(350, 450)
    R_moderate = rng.uniform(200, 250)
    R_tight = rng.uniform(120, 180)

    # S-curve: moderate -> tight with a short straight section
    R1_sc = rng.uniform(220, 280)
    R2_sc = rng.uniform(140, 180)
    straight_m = rng.uniform(45, 70)

    # Tightening: from sweeper-ish to moderate/tight over some length
    R0_ti = rng.uniform(380, 450)
    R1_ti = rng.uniform(160, 220)
    length_m = rng.uniform(120, 180)

    # Common sim config
    duration_s = rng.uniform(8.0, 10.0)
    vis_horizon_s = 1.4
    gamma_per_meter = 0.00035
    lat_jerk_cap = rng.uniform(2.0, 2.5)
    margins = [8.0, 12.0]

    specs: List[Tuple[str, ScenarioSpec]] = []
    for M in margins:
        specs.extend([
            ("const_sweeper", ScenarioSpec("const_sweeper", v0, dt, duration_s, vis_horizon_s, M,
                                           gamma_per_meter, lat_jerk_cap, radius_m=R_sweeper)),
            ("const_moderate", ScenarioSpec("const_moderate", v0, dt, duration_s, vis_horizon_s, M,
                                            gamma_per_meter, lat_jerk_cap, radius_m=R_moderate)),
            ("const_tight", ScenarioSpec("const_tight", v0, dt, duration_s, vis_horizon_s, M,
                                         gamma_per_meter, lat_jerk_cap, radius_m=R_tight)),
            ("s_curve", ScenarioSpec("s_curve", v0, dt, duration_s, vis_horizon_s, M,
                                     gamma_per_meter, lat_jerk_cap, r1_m=R1_sc, r2_m=R2_sc, straight_m=straight_m)),
            ("tightening", ScenarioSpec("tightening", v0, dt, duration_s, vis_horizon_s, M,
                                        gamma_per_meter, lat_jerk_cap, r0_m=R0_ti, r1_m=R1_ti, length_m=length_m)),
        ])
    return specs


def build_initial_sets(band: str, active_target: int = 32, guard_target: int = 8, seed: int = 123) -> Tuple[List[ScenarioSpec], List[ScenarioSpec], List[ScenarioSpec]]:
    """
    Build active set A (~32) and guard set G (~8), plus a background pool B for refreshing.
    """
    rng = random.Random(seed)
    v_lo, v_hi = _band_to_speed_range(band)
    # Sample a representative spread of starting speeds within the band
    speeds = []
    if (v_hi - v_lo) <= 12.0:
        # narrow band: denser sampling
        n = 8
    else:
        n = 12
    for i in range(n):
        s = v_lo + (i + 0.5) * (v_hi - v_lo) / n
        speeds.append(s)

    # Build pool across speeds and families; shuffle for variety
    pool: List[ScenarioSpec] = []
    for v0 in speeds:
        fam_specs = make_scenarios_for_speed(v0)
        for _, sp in fam_specs:
            pool.append(sp)
    rng.shuffle(pool)

    # Active set: take first active_target but ensure per-family coverage
    def family(sp: ScenarioSpec) -> str:
        return sp.family

    fam_counts: Dict[str, int] = {}
    A: List[ScenarioSpec] = []
    min_per_family = 6  # ~5 families * 6 ≈ 30
    for sp in pool:
        f = family(sp)
        cnt = fam_counts.get(f, 0)
        if cnt < min_per_family or len(A) < active_target:
            A.append(sp)
            fam_counts[f] = cnt + 1
        if len(A) >= active_target:
            break

    # Guard set: next guard_target ensuring diversity
    G: List[ScenarioSpec] = []
    fam_counts_guard: Dict[str, int] = {}
    for sp in pool[len(A):]:
        f = family(sp)
        cnt = fam_counts_guard.get(f, 0)
        if cnt < 2 or len(G) < guard_target:
            G.append(sp)
            fam_counts_guard[f] = cnt + 1
        if len(G) >= guard_target:
            break

    # Background pool for refresh (rest)
    B: List[ScenarioSpec] = pool[len(A)+len(G):]
    return A, G, B


# --------------------------------------------------------------------------------------
# Loss/metrics helpers
# --------------------------------------------------------------------------------------

def _pm_diff_mask(pos_margin: np.ndarray) -> np.ndarray:
    """
    Build boolean mask aligned with np.diff(v_cmd):
      - For each positive-margin (PM) window, ignore the first two samples.
      - Mask length = len(v_cmd)-1 (align with diff).
    """
    pos_margin = np.asarray(pos_margin, dtype=bool)
    n = len(pos_margin)
    if n <= 2:
        return np.zeros(max(n - 1, 0), dtype=bool)
    mask = np.zeros(n - 1, dtype=bool)
    in_win = False
    count_in = 0
    for i in range(n):
        pm = pos_margin[i]
        if pm and not in_win:
            in_win = True
            count_in = 1
        elif pm and in_win:
            count_in += 1
        elif not pm and in_win:
            in_win = False
            count_in = 0
        # mask aligns to diff: mark position i-1 (transition into sample i)
        if i >= 1:
            if in_win and count_in > 2:
                mask[i - 1] = True
    return mask


@dataclass
class ScenarioEval:
    error_abs: float
    down_pm_count: int
    jerk_pos_excess: float
    jerk_neg_excess: float
    drift_excess: float
    jerk_pos: float
    jerk_neg: float
    drift_pm: float


def evaluate_one_scenario(scn: Scenario) -> ScenarioEval:
    """
    Run harness simulate(scn) and compute metrics for the loss:
      - error_abs = |min(v_cmd) - min(v_clean)|
      - PM downward events count (strictly negative dv) under PM mask
      - jerk exceedances: max(0, jerk_pos - 2.5), max(0, -6.5 - jerk_neg)
      - overslow drift within PM windows: span of minima; penalize exceedance over 0.6 m/s
    """
    res = simulate(scn)

    # Basic arrays
    v_cmd = np.asarray(res.v_cmd, dtype=float)
    v_clean = np.asarray(res.v_clean, dtype=float)
    min_cmd = float(np.min(v_cmd)) if v_cmd.size else 0.0
    min_phys = float(np.min(v_clean)) if v_clean.size else 0.0
    error_abs = float(abs(min_cmd - min_phys))

    # PM mask based on harness convention
    # pos_margin = (d_req <= (s_vis - M))
    M = getattr(scn, "vis_margin_m", 8.0)
    d_req = np.asarray(getattr(res, "d_req"), dtype=float)
    s_vis = np.asarray(getattr(res, "s_vis"), dtype=float)
    pos_margin = (d_req <= (s_vis - M))

    pm_mask = _pm_diff_mask(pos_margin)
    dv = np.diff(v_cmd)
    down_pm_count = int(np.sum((dv < -1e-6) & pm_mask))

    # jerk metrics
    jerk_pos = float(res.metrics.get("jerk_pos", 0.0))
    jerk_neg = float(res.metrics.get("jerk_neg", 0.0))
    jerk_pos_excess = float(max(0.0, jerk_pos - 2.5))
    jerk_neg_excess = float(max(0.0, (-6.5) - jerk_neg))

    # overslow drift: span of minima across PM windows of v_cmd
    # Identify PM windows and compute minimum inside each window
    drift_pm = 0.0
    mins = []
    in_win = False
    start = 0
    for i, pm in enumerate(pos_margin):
        if pm and not in_win:
            in_win = True
            start = i
        elif not pm and in_win:
            in_win = False
            s = start
            e = i
            if e > s:
                mins.append(np.min(v_cmd[s:e]))
    if in_win:
        # trailing window
        s = start
        e = len(pos_margin)
        if e > s:
            mins.append(np.min(v_cmd[s:e]))
    if len(mins) >= 2:
        drift_pm = float(np.max(mins) - np.min(mins))
    drift_excess = float(max(0.0, drift_pm - 0.6))

    return ScenarioEval(
        error_abs=error_abs,
        down_pm_count=down_pm_count,
        jerk_pos_excess=jerk_pos_excess,
        jerk_neg_excess=jerk_neg_excess,
        drift_excess=drift_excess,
        jerk_pos=jerk_pos,
        jerk_neg=jerk_neg,
        drift_pm=drift_pm,
    )


def cvar_at_alpha(errors: List[float], alpha: float = 0.95) -> float:
    """CVaR@alpha for non-negative errors (list)."""
    if not errors:
        return 0.0
    arr = np.sort(np.asarray(errors, dtype=float))
    n = len(arr)
    k = max(1, int(math.ceil((1.0 - alpha) * n)))  # number in the tail
    tail = arr[-k:]
    return float(np.mean(tail))


@dataclass
class LossComponents:
    cvar95: float
    penalty_down: float
    penalty_jerk: float
    penalty_drift: float
    penalty_smooth: float
    penalty_hi: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "cvar95": self.cvar95,
            "penalty_down": self.penalty_down,
            "penalty_jerk": self.penalty_jerk,
            "penalty_drift": self.penalty_drift,
            "penalty_smooth": self.penalty_smooth,
            "penalty_hi": self.penalty_hi,
            "total": self.total,
        }


def smoothness_penalty(theta: Theta, v_grid: np.ndarray) -> float:
    """
    Penalize adjacent relative jumps |Δ| / cap > 0.5 across grid.
    We use max of the two neighboring cap values in denominator to be conservative.
    """
    g = gamma_cap_speed_param(v_grid, theta)
    g = np.asarray(g, dtype=float)
    # guard tiny values to avoid division blow-up; but d_min already prevents zero
    eps = 1e-12
    rel = np.abs(np.diff(g)) / np.maximum(np.maximum(g[:-1], g[1:]), eps)
    excess = np.clip(rel - 0.5, 0.0, None)
    return float(np.sum(excess))


def hi_guard_penalty(theta: Theta, J_lat_min: float = 2.0, s: float = 1.05) -> float:
    """
    Penalize violation of the high-speed inertial guard: d < d_min.
    In theory mapping already enforces this (d >= d_min), so this remains zero.
    """
    d_min = s * (J_lat_min / max(theta.v_hi, 1e-3) ** 3)
    return float(max(0.0, d_min - theta.d))


@dataclass
class BatchEvalResult:
    errors: List[float]
    down_counts: List[int]
    jerk_pos_excess: List[float]
    jerk_neg_excess: List[float]
    drift_excess: List[float]
    # for debugging / reporting
    jerk_pos_list: List[float]
    jerk_neg_list: List[float]
    drift_pm_list: List[float]

    def cvar95(self) -> float:
        return cvar_at_alpha(self.errors, 0.95)

    def penalties(self) -> Tuple[float, float, float]:
        down = float(np.sum(self.down_counts))
        jerk = float(np.sum(self.jerk_pos_excess) + np.sum(self.jerk_neg_excess))
        drift = float(np.sum(self.drift_excess))
        return down, jerk, drift


# Worker function for parallel evaluation of a single φ over a list of ScenarioSpec
def _worker_eval_phi(args) -> BatchEvalResult:
    phi, v_hi, pm_clamp, specs = args
    # Install patch in this process
    occl = install_occlusion_patch(phi=np.asarray(phi, dtype=float), v_hi=float(v_hi), jerk_cap_guard=False)
    clamp = None
    try:
        if pm_clamp:
            clamp = install_margin_gate_patch(dt=0.05, jerk_limit_pos=2.5)
        # Build and evaluate scenarios
        errors = []
        down_counts = []
        jerk_pos_excess = []
        jerk_neg_excess = []
        drift_excess = []
        jerk_pos_list = []
        jerk_neg_list = []
        drift_pm_list = []
        for sp in specs:
            scn = _build_scenario(sp)
            ev = evaluate_one_scenario(scn)
            errors.append(ev.error_abs)
            down_counts.append(ev.down_pm_count)
            jerk_pos_excess.append(ev.jerk_pos_excess)
            jerk_neg_excess.append(ev.jerk_neg_excess)
            drift_excess.append(ev.drift_excess)
            jerk_pos_list.append(ev.jerk_pos)
            jerk_neg_list.append(ev.jerk_neg)
            drift_pm_list.append(ev.drift_pm)
        return BatchEvalResult(
            errors=errors,
            down_counts=down_counts,
            jerk_pos_excess=jerk_pos_excess,
            jerk_neg_excess=jerk_neg_excess,
            drift_excess=drift_excess,
            jerk_pos_list=jerk_pos_list,
            jerk_neg_list=jerk_neg_list,
            drift_pm_list=drift_pm_list,
        )
    finally:
        if clamp is not None:
            clamp.uninstall()
        occl.uninstall()


def evaluate_phi_on_specs(phi: np.ndarray, specs: List[ScenarioSpec], v_hi: float, pm_clamp: bool,
                          max_workers: int = 2) -> BatchEvalResult:
    """
    Evaluate φ over specs. We run in parallel by distributing the *entire* set once
    (SPSA uses two φ points per step; each is one job).
    """
    # For single evaluation, no need to split specs—just one worker is enough; but using a pool
    # keeps the code uniform for SPSA (+ and -).
    args = (np.asarray(phi, dtype=float), float(v_hi), bool(pm_clamp), specs)
    # Run in a separate process to ensure patch isolation from the main process
    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_worker_eval_phi, args)
        return fut.result()


def assemble_loss(phi: np.ndarray, specs: List[ScenarioSpec], v_hi: float, pm_clamp: bool,
                  smooth_grid: np.ndarray, weights: Dict[str, float],
                  max_workers: int = 2) -> Tuple[LossComponents, BatchEvalResult, Theta]:
    """
    Compute aggregate loss and return components along with BatchEvalResult and θ.
    """
    theta = phi_to_theta(phi, v_hi=v_hi)
    batch = evaluate_phi_on_specs(phi, specs, v_hi=v_hi, pm_clamp=pm_clamp, max_workers=max_workers)
    cvar95 = batch.cvar95()
    down, jerk, drift = batch.penalties()
    pen_smooth = smoothness_penalty(theta, smooth_grid)
    pen_hi = hi_guard_penalty(theta)
    total = (cvar95
             + weights["down"] * down
             + weights["jerk"] * jerk
             + weights["drift"] * drift
             + weights["smooth"] * pen_smooth
             + weights["hi"] * pen_hi)
    comps = LossComponents(cvar95=cvar95, penalty_down=down, penalty_jerk=jerk, penalty_drift=drift,
                           penalty_smooth=pen_smooth, penalty_hi=pen_hi, total=float(total))
    return comps, batch, theta


# --------------------------------------------------------------------------------------
# SPSA with tiny L∞ trust region
# --------------------------------------------------------------------------------------

@dataclass
class SPSAState:
    phi_best: np.ndarray
    loss_best: float
    theta_best: Theta
    rho: float
    streak_ok: int


def spsa_optimize(phi0: np.ndarray,
                  specs_active: List[ScenarioSpec],
                  specs_guard: List[ScenarioSpec],
                  v_hi: float,
                  pm_clamp: bool,
                  iters: int = 45,
                  seed: int = 123,
                  refresh_every: int = 4,
                  weights: Optional[Dict[str, float]] = None,
                  verbose: bool = True) -> Tuple[np.ndarray, Theta, LossComponents, Dict[str, Any]]:
    """
    SPSA with L∞ trust region and active-set refresh.
    """
    if weights is None:
        weights = {"down": 5.0, "jerk": 2.0, "drift": 2.0, "smooth": 1.0, "hi": 1.0}

    rng = np.random.RandomState(seed)
    phi = np.asarray(phi0, dtype=float).copy()
    # Smoothness grid
    v_grid = np.arange(5.0, 45.0 + 1e-6, 1.0)

    # Initial evaluation
    comps, batch, theta = assemble_loss(phi, specs_active, v_hi, pm_clamp, v_grid, weights, max_workers=2)
    state = SPSAState(phi_best=phi.copy(), loss_best=comps.total, theta_best=theta, rho=1.0, streak_ok=0)
    best_snapshot = {"loss": comps.to_dict(), "theta": theta.to_dict()}

    if verbose:
        print(f"[init] CVaR95={comps.cvar95:.4f} | penalties(down/jerk/drift/smooth/hi) = "
              f"{comps.penalty_down:.2f}/{comps.penalty_jerk:.4f}/{comps.penalty_drift:.4f}/"
              f"{comps.penalty_smooth:.4f}/{comps.penalty_hi:.4f} | total={comps.total:.4f}")
        print(f"[init] θ = a:{theta.a:.3e} b:{theta.b:.3e} c:{theta.c:.3f} d:{theta.d:.3e} (d_min={theta.d_min:.3e})")
        print(f"[init] trust-region ρ = {state.rho:.3f} | best total = {state.loss_best:.4f}")

    # SPSA loop
    history = []
    no_improve_steps = 0

    for k in range(1, iters + 1):
        # Trust region schedules
        rho = max(0.1, min(2.0, state.rho))
        c_k = max(0.03, 0.15 * rho)
        a_k = 0.2 * rho

        # Rademacher ±1 perturbation
        delta = rng.choice([-1.0, 1.0], size=4).astype(float)
        phi_center = state.phi_best.copy()

        phi_plus = phi_center + c_k * delta
        phi_minus = phi_center - c_k * delta

        # Parallel evaluate the two batched points
        with ProcessPoolExecutor(max_workers=2) as ex:
            fut_plus = ex.submit(assemble_loss, phi_plus, specs_active, v_hi, pm_clamp, v_grid, weights, 2)
            fut_minus = ex.submit(assemble_loss, phi_minus, specs_active, v_hi, pm_clamp, v_grid, weights, 2)
            comps_plus, batch_plus, theta_plus = fut_plus.result()
            comps_minus, batch_minus, theta_minus = fut_minus.result()

        # SPSA gradient estimate
        ghat = ((comps_plus.total - comps_minus.total) / (2.0 * c_k)) * delta

        # Proposed step (project inside L∞ trust box around center)
        phi_prop = phi_center - a_k * ghat
        phi_proj = np.minimum(np.maximum(phi_prop, phi_center - rho), phi_center + rho)

        # Evaluate projected
        comps_proj, batch_proj, theta_proj = assemble_loss(phi_proj, specs_active, v_hi, pm_clamp, v_grid, weights, 2)

        improved = comps_proj.total < state.loss_best
        if improved:
            state.phi_best = phi_proj.copy()
            state.loss_best = float(comps_proj.total)
            state.theta_best = theta_proj
            state.rho = min(2.0, rho * 1.20)
            no_improve_steps = 0
        else:
            state.rho = max(0.1, rho * 0.70)
            no_improve_steps += 1

        # Early stopping gates (on the active set)
        all_pen_zero = (comps_proj.penalty_down == 0.0 and comps_proj.penalty_jerk == 0.0
                        and comps_proj.penalty_drift == 0.0 and comps_proj.penalty_smooth == 0.0
                        and comps_proj.penalty_hi == 0.0)
        if comps_proj.cvar95 <= 1.0 and all_pen_zero:
            state.streak_ok += 1
        else:
            state.streak_ok = 0

        # Logging
        if improved:
            best_snapshot = {"loss": comps_proj.to_dict(), "theta": theta_proj.to_dict()}

        if (k % 1) == 0:
            print(f"[iter {k:02d}] total={comps_proj.total:.4f} (best {state.loss_best:.4f}) | "
                  f"CVaR95={comps_proj.cvar95:.4f} | pen(down/jerk/drift/smooth/hi)="
                  f"{comps_proj.penalty_down:.2f}/{comps_proj.penalty_jerk:.4f}/{comps_proj.penalty_drift:.4f}/"
                  f"{comps_proj.penalty_smooth:.4f}/{comps_proj.penalty_hi:.4f} | "
                  f"θ: a={theta_proj.a:.3e} b={theta_proj.b:.3e} c={theta_proj.c:.3f} d={theta_proj.d:.3e} | "
                  f"ρ={state.rho:.3f} | streak_ok={state.streak_ok}")

        history.append({
            "iter": k,
            "rho": state.rho,
            "phi_best": state.phi_best.tolist(),
            "theta_best": state.theta_best.to_dict(),
            "loss_best": state.loss_best,
            "current_total": comps_proj.total,
            "current_components": comps_proj.to_dict(),
        })

        # Active-set maintenance every 'refresh_every' steps: replace easiest 25% with hardest from guard pool
        if (k % refresh_every) == 0 and specs_guard:
            # Evaluate guard hardness under current best φ
            guard_eval = evaluate_phi_on_specs(state.phi_best, specs_guard, v_hi=v_hi, pm_clamp=pm_clamp, max_workers=2)
            # Easiest in active (lowest errors); hardest in guard (highest errors)
            active_errors = np.array(assemble_loss(state.phi_best, specs_active, v_hi, pm_clamp, v_grid, weights, 2)[1].errors)
            guard_errors = np.array(guard_eval.errors)
            n_replace = max(1, len(specs_active) // 4)

            # Indices
            active_sorted_idx = np.argsort(active_errors)  # ascending
            guard_sorted_idx = np.argsort(-guard_errors)   # descending (hardest first)

            # Replace keeping per-family coverage (don't drop below 3 per family)
            fam_counts: Dict[str, int] = {}
            for i, sp in enumerate(specs_active):
                fam_counts[sp.family] = fam_counts.get(sp.family, 0) + 1

            replaced = 0
            to_drop = []
            for idx in active_sorted_idx:
                f = specs_active[idx].family
                if fam_counts.get(f, 0) > 3:
                    to_drop.append(idx)
                    fam_counts[f] -= 1
                    replaced += 1
                if replaced >= n_replace:
                    break

            to_add = [specs_guard[i] for i in guard_sorted_idx[:replaced]]
            # Rebuild active and guard
            new_active = [sp for i, sp in enumerate(specs_active) if i not in to_drop] + to_add
            # Move dropped to guard (and remove added from guard)
            new_guard = []
            for i, sp in enumerate(specs_guard):
                if i in guard_sorted_idx[:replaced]:
                    continue
                new_guard.append(sp)
            new_guard += [specs_active[i] for i in to_drop]  # recycle old easy cases

            specs_active[:] = new_active
            specs_guard[:] = new_guard

            print(f"[refresh] Active set refreshed: replaced {replaced}, |A|={len(specs_active)} |G|={len(specs_guard)}")

        # Early stopping: run quick verifier trigger
        if state.streak_ok >= 5:
            print("[early-stop] Acceptance met on active set for 5 consecutive iters; stopping optimization.")
            break

    final_comps, _, final_theta = assemble_loss(state.phi_best, specs_active, v_hi, pm_clamp, v_grid, weights, 2)
    return state.phi_best, final_theta, final_comps, {
        "history": history,
        "best_snapshot": best_snapshot,
        "v_grid": v_grid.tolist(),
        "g_grid": gamma_cap_speed_param(v_grid, final_theta).tolist(),
    }


# --------------------------------------------------------------------------------------
# Verifier across full sweep
# --------------------------------------------------------------------------------------

@dataclass
class VerifierSummary:
    max_abs_error: float
    total_down_events: int
    jerk_pos_excess_sum: float
    jerk_neg_excess_sum: float
    drift_excess_sum: float
    smooth_penalty: float
    hi_penalty: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "max_abs_error": self.max_abs_error,
            "total_down_events": float(self.total_down_events),
            "jerk_pos_excess_sum": self.jerk_pos_excess_sum,
            "jerk_neg_excess_sum": self.jerk_neg_excess_sum,
            "drift_excess_sum": self.drift_excess_sum,
            "smooth_penalty": self.smooth_penalty,
            "hi_penalty": self.hi_penalty,
        }


def run_verifier(phi: np.ndarray, v_hi: float, pm_clamp: bool, full: bool = True) -> VerifierSummary:
    """
    Build a wide sweep of scenarios across 5..45 m/s (1 m/s grid), both margins, all families.
    """
    v_grid = np.arange(5.0, 45.0 + 1e-6, 1.0) if full else np.arange(10.0, 40.0 + 1e-6, 2.0)
    all_specs: List[ScenarioSpec] = []
    for v0 in v_grid:
        fam_specs = make_scenarios_for_speed(float(v0))
        all_specs.extend([sp for _, sp in fam_specs])

    # Evaluate in chunks for memory safety; patch is per-worker, so single worker per φ
    batch = evaluate_phi_on_specs(phi, all_specs, v_hi=v_hi, pm_clamp=pm_clamp, max_workers=1)

    theta = phi_to_theta(phi, v_hi=v_hi)
    smooth = smoothness_penalty(theta, np.arange(5.0, 45.0 + 1e-6, 1.0))
    hi = hi_guard_penalty(theta)

    summary = VerifierSummary(
        max_abs_error=float(np.max(batch.errors) if batch.errors else 0.0),
        total_down_events=int(np.sum(batch.down_counts)),
        jerk_pos_excess_sum=float(np.sum(batch.jerk_pos_excess)),
        jerk_neg_excess_sum=float(np.sum(batch.jerk_neg_excess)),
        drift_excess_sum=float(np.sum(batch.drift_excess)),
        smooth_penalty=float(smooth),
        hi_penalty=float(hi),
    )
    print("[verify] max|min_cmd-min_phys| = {:.4f} m/s | down={} | jerk_excess(pos/neg)={:.4f}/{:.4f} | drift_excess={:.4f} | smooth={:.4f} | hi={:.4f}".format(
        summary.max_abs_error, summary.total_down_events, summary.jerk_pos_excess_sum, summary.jerk_neg_excess_sum,
        summary.drift_excess_sum, summary.smooth_penalty, summary.hi_penalty
    ))
    return summary


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parametric tuner for VTSC speed-based gamma cap (SPSA + trust region).")
    ap.add_argument("--band", type=str, default="full", choices=["low", "mid", "high", "vhigh", "full"],
                    help="Speed band for scenarios (m/s).")
    ap.add_argument("--iters", type=int, default=45, help="SPSA iterations (default: 45).")
    ap.add_argument("--vhi", type=float, default=32.0, help="High-speed threshold for inertial guard (default: 32.0 m/s).")
    ap.add_argument("--pm-clamp", action="store_true", help="Enable optional jerk-safe PM clamp (off by default).")
    ap.add_argument("--save", type=str, default="docs/chauffeur/vtsc/testing/sweep/tuned_gamma_param_spsa.json",
                    help="Output JSON path (repo-relative).")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    ap.add_argument("--quick-verify", action="store_true", help="Run reduced verifier at the end.")
    ap.add_argument("--full-verify", action="store_true", help="Run full verifier at the end.")
    args = ap.parse_args()

    np.set_printoptions(precision=3, suppress=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initial φ (moderate, safe)
    phi0 = np.array([1.0, 0.5, 0.0, -2.0], dtype=float)  # produces small d near d_min, c ~ 1+softplus(0)=~1.693

    print("=== VTSC Parametric Gamma Cap Tuner (SPSA) ===")
    print(f"ROOT={ROOT}")
    print(f"Band={args.band} | Iters={args.iters} | v_hi={args.vhi:.1f} | pm-clamp={args.pm_clamp} | seed={args.seed}")
    print("Building initial scenario sets (active ~32, guard ~8)...")

    A, G, B = build_initial_sets(args.band, active_target=32, guard_target=8, seed=args.seed)
    print(f"Initial sets: |A|={len(A)} |G|={len(G)} |B|={len(B)}")

    # Optimize
    phi_best, theta_best, comps_best, aux = spsa_optimize(
        phi0=phi0, specs_active=A, specs_guard=G, v_hi=args.vhi, pm_clamp=args.pm_clamp,
        iters=args.iters, seed=args.seed, refresh_every=4, weights=None, verbose=True
    )

    # Verifier
    verifier_summary = None
    if args.quick_verify or args.full_verify:
        print("\n=== Verifier ===")
        verifier_summary = run_verifier(phi_best, v_hi=args.vhi, pm_clamp=args.pm_clamp, full=args.full_verify)

    # Final JSON summary
    out = {
        "phi": phi_best.tolist(),
        "theta": theta_best.to_dict(),
        "v_grid": aux["v_grid"],
        "g_grid": aux["g_grid"],
        "active_set_cvar95": comps_best.cvar95,
        "active_set_penalties": {
            "down": comps_best.penalty_down,
            "jerk": comps_best.penalty_jerk,
            "drift": comps_best.penalty_drift,
            "smooth": comps_best.penalty_smooth,
            "hi": comps_best.penalty_hi,
        },
        "active_set_total": comps_best.total,
        "verifier": (verifier_summary.to_dict() if verifier_summary is not None else None),
        "optimizer": {
            "best_snapshot": aux["best_snapshot"],
            "history_len": len(aux["history"]),
        },
        "meta": {
            "band": args.band,
            "iters": args.iters,
            "v_hi": args.vhi,
            "pm_clamp": bool(args.pm_clamp),
            "seed": args.seed,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    }

    # Repo-relative safety: ensure we write within repo
    save_path = args.save
    if not os.path.isabs(save_path):
        save_path = os.path.join(ROOT, save_path)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)
    print(f"\nSaved tuned parameters to: {save_path}")
    print("Final θ: a={:.6e}, b={:.6e}, c={:.6f}, d={:.6e} (d_min={:.6e})".format(
        theta_best.a, theta_best.b, theta_best.c, theta_best.d, theta_best.d_min
    ))
    print("Done.")


if __name__ == "__main__":
    main()
