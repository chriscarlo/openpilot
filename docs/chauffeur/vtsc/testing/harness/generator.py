#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np
from typing import Tuple, Dict

from .scenarios import Scenario, GeometryProfile, ConfidenceProfile, SpeedLimitProfile


def generate_timebase(scn: Scenario):
    t = np.arange(0.0, scn.duration_s + scn.dt/2, scn.dt)
    return t


def generate_curvature(scn: Scenario, t: np.ndarray) -> np.ndarray:
    g = scn.geometry
    if g.kind == 'constant':
        return np.ones_like(t) * float(g.kappa0)
    if g.kind == 'tightening':
        # Rise from kappa0 to kappa1 until apex then hold
        half = int(len(t) * 0.5)
        k = np.linspace(g.kappa0, g.kappa1, half)
        tail = np.ones(len(t) - half) * float(g.kappa1)
        return np.concatenate([k, tail])
    if g.kind == 'easing':
        # Rise to kappa1, then ease down to kappa0 (apex near mid)
        n = len(t)
        half = max(1, n // 2)
        up = np.linspace(g.kappa0, g.kappa1, half, endpoint=True)
        down = np.linspace(g.kappa1, g.kappa0, n - half, endpoint=True)
        return np.concatenate([up, down])[:n]
    if g.kind == 's_curve':
        # Right bend then short straight then left bend
        n = len(t)
        third = n // 3
        k1 = np.linspace(0.0, g.kappa0, third)
        k2 = np.zeros(max(1, int(scn.geometry.mid_straight_s / scn.dt)))
        k3 = np.linspace(0.0, -abs(g.kappa1), third)
        rest = n - (len(k1) + len(k2) + len(k3))
        if rest > 0:
            k3 = np.concatenate([k3, np.ones(rest) * (-abs(g.kappa1))])
        return np.concatenate([k1, k2, k3])[:n]
    if g.kind == 'multi_curve' and g.segments:
        # Piecewise constant curvature according to segments
        k = np.zeros_like(t)
        elapsed = 0.0
        idx = 0
        for seg in g.segments:
            dur = float(seg.get('duration_s', 0.0))
            # Allow either explicit curvature or radius-based specification
            if 'kappa' in seg:
                kappa = float(seg.get('kappa', 0.0))
            elif 'radius_m' in seg:
                r = max(1e-6, float(seg.get('radius_m', 1e6)))
                sign = float(seg.get('sign', 1.0))
                kappa = sign * (1.0 / r)
            else:
                kappa = 0.0
            end = elapsed + max(0.0, dur)
            while idx < len(t) and t[idx] <= end + 1e-6:
                k[idx] = kappa
                idx += 1
            elapsed = end
        # Fill any remaining time with last kappa
        if idx < len(t) and g.segments:
            k[idx:] = float(g.segments[-1].get('kappa', 0.0))
        return k
    return np.zeros_like(t)


def generate_confidence(scn: Scenario, t: np.ndarray) -> np.ndarray:
    c = scn.confidence
    if c.kind == 'stable':
        return np.ones_like(t) * float(c.value)
    if c.kind == 'window':
        conf = np.ones_like(t) * 0.9
        mask = (t >= c.window_start_s) & (t <= c.window_end_s)
        conf[mask] = float(c.value)
        return conf
    if c.kind == 'borderline_lpf':
        # Sinusoid around threshold with LPF-like easing
        raw = 0.5 * (1 + np.sin(2 * math.pi * c.freq_hz * t))
        raw = c.low + (c.high - c.low) * raw
        # Light first-order low-pass
        alpha = 0.2
        y = np.zeros_like(raw)
        y[0] = raw[0]
        for i in range(1, len(raw)):
            y[i] = (1 - alpha) * y[i-1] + alpha * raw[i]
        return y
    return np.ones_like(t) * 0.9


def generate_speed_limit(scn: Scenario, t: np.ndarray) -> np.ndarray:
    s = scn.speed_limit
    if s.kind == 'none':
        return np.ones_like(t) * float(s.start_mps)
    if s.kind == 'step':
        v = np.ones_like(t) * float(s.start_mps)
        v[t >= s.step_time_s] = float(s.step_to_mps)
        return v
    return np.ones_like(t) * 31.0


def find_apex_index(kappa: np.ndarray) -> int:
    # Approximate apex as peak abs curvature index; handle flat/constant gracefully
    if len(kappa) == 0:
        return 0
    abs_k = np.abs(kappa)
    max_val = np.max(abs_k)
    idxs = np.where(np.isclose(abs_k, max_val))[0]
    if idxs.size == 0:
        return int(np.argmax(abs_k))
    # If many maxima (e.g., constant curvature), choose mid index
    return int(idxs[len(idxs)//2])
