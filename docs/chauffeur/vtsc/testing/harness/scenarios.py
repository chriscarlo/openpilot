#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class GeometryProfile:
    kind: str  # 'constant', 'tightening', 'easing', 's_curve'
    kappa0: float = 0.0
    kappa1: float = 0.0
    length_s: float = 150.0  # meters horizon used for envelope checks
    # For S-curve
    mid_straight_s: float = 20.0
    # For multi-curve piecewise profiles
    segments: List[Dict[str, float]] = field(default_factory=list)  # [{"duration_s": float, "kappa": float}, ...]


@dataclass
class ConfidenceProfile:
    kind: str  # 'stable', 'borderline_lpf', 'window'
    value: float = 0.9
    low: float = 0.68
    high: float = 0.76
    freq_hz: float = 2.5
    window_start_s: float = 0.0
    window_end_s: float = 0.0


@dataclass
class SpeedLimitProfile:
    kind: str = 'none'  # 'none', 'step'
    start_mps: float = 31.0
    step_time_s: float = 999.0
    step_to_mps: float = 20.0


@dataclass
class VTSCParams:
    aggressiveness: float = 1.0
    alpha: float = 0.3
    hysteresis: float = 0.2
    safety_bias: float = 0.1


@dataclass
class Scenario:
    name: str
    duration_s: float = 10.0
    dt: float = 0.05
    v0_mps: float = 25.0
    geometry: GeometryProfile = field(default_factory=lambda: GeometryProfile(kind='constant', kappa0=0.002))
    confidence: ConfidenceProfile = field(default_factory=lambda: ConfidenceProfile(kind='stable', value=0.9))
    speed_limit: SpeedLimitProfile = field(default_factory=SpeedLimitProfile)
    latency_s: float = 0.0
    params: VTSCParams = field(default_factory=VTSCParams)
    # Per-scenario visibility barrier params
    vis_horizon_s: float = 1.4
    vis_margin_m: float = 10.0
    gamma_per_meter: float = 0.00035
    lat_jerk_cap: float | None = 2.0
    # Default to production-like lateral jerk cap


def load_scenario(path: str) -> Scenario:
    with open(path, 'r') as f:
        data = json.load(f)

    def to_geometry(d: Dict[str, Any]) -> GeometryProfile:
        return GeometryProfile(**d)

    def to_confidence(d: Dict[str, Any]) -> ConfidenceProfile:
        return ConfidenceProfile(**d)

    def to_speedlimit(d: Dict[str, Any]) -> SpeedLimitProfile:
        return SpeedLimitProfile(**d)

    def to_params(d: Dict[str, Any]) -> VTSCParams:
        return VTSCParams(**d)

    return Scenario(
        name=data.get('name', 'unnamed'),
        duration_s=float(data.get('duration_s', 10.0)),
        dt=float(data.get('dt', 0.05)),
        v0_mps=float(data.get('v0_mps', 25.0)),
        geometry=to_geometry(data.get('geometry', {'kind': 'constant', 'kappa0': 0.002})),
        confidence=to_confidence(data.get('confidence', {'kind': 'stable', 'value': 0.9})),
        speed_limit=to_speedlimit(data.get('speed_limit', {'kind': 'none'})),
        latency_s=float(data.get('latency_s', 0.0)),
        params=to_params(data.get('params', {})),
    )
