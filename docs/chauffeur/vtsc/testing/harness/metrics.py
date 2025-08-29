#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from typing import Dict, Any


def compute_metrics(t, v_cmd, v_clean, a_cmd, conf, occluded, good_th=0.75, apex_idx: int | None = None) -> Dict[str, Any]:
    t = np.asarray(t)
    v_cmd = np.asarray(v_cmd)
    v_clean = np.asarray(v_clean)
    a_cmd = np.asarray(a_cmd)
    occluded = np.asarray(occluded)
    conf = np.asarray(conf)

    dt = float(t[1] - t[0]) if len(t) > 1 else 0.05

    # Invariants
    pos_accel_while_occluded = float(np.max(np.maximum(a_cmd[occluded], 0.0))) if np.any(occluded) else 0.0

    # Integrated overslow after apex/reacquisition window: use entire trace as fallback
    # Start integration window at reacquisition if present; otherwise at apex if provided; else start
    start_idx = 0
    # Prefer reacquisition point (occluded→good transition)
    idxs = np.where((occluded[:-1] == True) & (occluded[1:] == False))[0]
    if idxs.size > 0:
        start_idx = int(idxs[0] + 1)
    elif apex_idx is not None:
        start_idx = int(apex_idx)
    overslow = np.maximum(v_clean - v_cmd, 0.0)
    integrated_overslow = float(np.sum(overslow[start_idx:]) * dt)

    # Reacquisition latency: first occluded->good transition to |v_cmd - v_clean| <= 0.2
    reacq_latency = None
    idxs = np.where((occluded[:-1] == True) & (occluded[1:] == False))[0]
    if idxs.size > 0:
        idx_good = int(idxs[0] + 1)
        for j in range(idx_good, len(t)):
            if abs(v_cmd[j] - v_clean[j]) <= 0.2:
                reacq_latency = float(t[j] - t[idx_good])
                break

    # Overshoot on recovery: post reacq
    overshoot = 0.0
    if reacq_latency is not None:
        start = idx_good
        diff = v_cmd[start:] - v_clean[start:]
        overshoot = float(np.max(diff)) if len(diff) > 0 else 0.0

    # Jerk caps
    jerk = np.diff(a_cmd) / dt if len(a_cmd) > 1 else np.array([0.0])
    jerk_pos = float(np.max(jerk)) if len(jerk) else 0.0
    jerk_neg = float(np.min(jerk)) if len(jerk) else 0.0

    return {
        'pos_accel_while_occluded': pos_accel_while_occluded,
        'integrated_overslow': integrated_overslow,
        'reacq_latency': reacq_latency,
        'overshoot_on_recovery': overshoot,
        'jerk_pos': jerk_pos,
        'jerk_neg': jerk_neg,
    }
