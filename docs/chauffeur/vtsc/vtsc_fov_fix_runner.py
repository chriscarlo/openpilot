#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from datetime import datetime


def gamma_cap_speed(v_mps: float) -> float:
    # Mirrors the speed-cap table used internally (piecewise) for occlusion growth
    sp = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0]
    gp = [8e-4, 6e-4, 5e-4, 4e-4, 3e-4, 1.8e-4, 1.2e-4, 1.0e-4]
    vv = max(0.0, float(v_mps))
    if vv <= sp[0]:
        return gp[0]
    if vv >= sp[-1]:
        return gp[-1]
    for i in range(len(sp) - 1):
        if sp[i] <= vv <= sp[i + 1]:
            t = (vv - sp[i]) / max(1e-6, (sp[i + 1] - sp[i]))
            return gp[i] + (gp[i + 1] - gp[i]) * t
    return gp[-1]


def audit_gamma_cap(out_dir: str):
    table = {f"{v:.1f}": gamma_cap_speed(v) for v in [x * 0.5 for x in range(0, 91)]}
    with open(os.path.join(out_dir, 'unit_audit_gamma_cap.json'), 'w') as f:
        json.dump(table, f, indent=2)
    # basic checks
    vals = [table[k] for k in sorted(table.keys(), key=lambda s: float(s))]
    monotone = all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))
    return {'monotone_nonincreasing': monotone, 'min': min(vals), 'max': max(vals)}


def sweep_fov_gate(out_dir: str, psi_fov=0.49, psi_margin=0.087):
    from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController
    res = []
    for k in [x * 1e-4 for x in range(0, 81)]:  # 0..0.008
        row = []
        for s in range(0, 301, 10):  # 0..300 m
            occ, st, reason, dbg = VisionTurnController.occlusion_gate(k, s, 0.9, psi_fov, psi_margin,
                                                                       state={'on_cnt': 4, 'off_cnt': 0, 'occluded': False})
            row.append(1 if occ else 0)
        res.append(row)
    with open(os.path.join(out_dir, 'fov_gate_sweep.json'), 'w') as f:
        json.dump({'kappa_grid_1e-4': [x for x in range(0, 81)],
                   's_visible_grid_m': [x for x in range(0, 301, 10)],
                   'occluded': res}, f)


def write_manifest(out_dir: str, extra: dict):
    manifest = {
        'created': datetime.utcnow().isoformat() + 'Z',
        'branch': os.popen('git rev-parse --abbrev-ref HEAD').read().strip(),
        'commit': os.popen('git rev-parse HEAD').read().strip(),
        'files': sorted(os.listdir(out_dir)),
    }
    manifest.update(extra)
    with open(os.path.join(out_dir, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('docs/chauffeur/vtsc', f'vtsc_fov_fix_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    gamma_stats = audit_gamma_cap(out_dir)
    sweep_fov_gate(out_dir)
    with open(os.path.join(out_dir, 'README.md'), 'w') as f:
        f.write('# VTSC FOV Gating Artifacts\n\n')
        f.write(f"- Gamma cap audit monotone: {gamma_stats['monotone_nonincreasing']}\n")
    write_manifest(out_dir, {'gamma_audit': gamma_stats})
    print('Artifacts written to', out_dir)


if __name__ == '__main__':
    main()
