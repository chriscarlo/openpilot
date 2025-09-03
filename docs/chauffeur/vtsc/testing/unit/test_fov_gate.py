#!/usr/bin/env python3

import pytest

from sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController


def test_fov_gate_onset_and_clear_hysteresis():
    # straight freeway: small curvature, long visibility, good conf
    occluded, st, reason, dbg = VisionTurnController.occlusion_gate(
        kappa_vis=1e-6, s_visible_m=200.0, path_conf=0.9,
        psi_fov_rad=0.49, psi_margin_rad=0.087,
        state={'on_cnt': 0, 'off_cnt': 0, 'occluded': False},
    )
    assert not occluded

    # approach FOV boundary: larger curvature and psi_vis above threshold
    st = {'on_cnt': 0, 'off_cnt': 0, 'occluded': False}
    for _ in range(5):
        occluded, st, reason, dbg = VisionTurnController.occlusion_gate(
            kappa_vis=3e-3, s_visible_m=200.0, path_conf=0.8,
            psi_fov_rad=0.49, psi_margin_rad=0.087,
            state=st,
        )
    assert occluded, "Should engage occlusion after N_on hysteresis"
    assert reason in ("fov_exit", "none")

    # clear after N_off under freeway-like conditions
    for _ in range(10):
        occluded, st, reason, dbg = VisionTurnController.occlusion_gate(
            kappa_vis=1e-6, s_visible_m=200.0, path_conf=0.9,
            psi_fov_rad=0.49, psi_margin_rad=0.087,
            state=st,
        )
    assert not occluded, "Should clear occlusion after N_off hysteresis"
