#!/usr/bin/env python3

import sys
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from docs.chauffeur.vtsc.testing.harness.scenarios import Scenario, GeometryProfile, ConfidenceProfile, SpeedLimitProfile, VTSCParams
from docs.chauffeur.vtsc.testing.harness.simulate import simulate
from opendbc.car.common.conversions import Conversions as CV


def mph(x: float) -> float:
    return float(x) * CV.MPH_TO_MS


def straight_freeway_scenario(v_mph: float = 70.0) -> Scenario:
    v0 = mph(v_mph)
    return Scenario(
        name="freeway_straight_failopen",
        duration_s=10.0,
        dt=0.05,
        v0_mps=v0,
        geometry=GeometryProfile(kind='constant', kappa0=0.0),
        confidence=ConfidenceProfile(kind='stable', value=0.9),
        speed_limit=SpeedLimitProfile(kind='none', start_mps=v0),
    )


def test_freeway_failopen_no_crawl():
    scn = straight_freeway_scenario(70.0)
    res = simulate(scn)
    set_speed = float(res.v_clean[-1])  # equals speed limit on straight
    v_cmd_min = float(np.min(res.v_cmd))
    # Expect no freeway crawl: keep at or near set speed
    assert v_cmd_min >= max(mph(45.0), 0.85 * set_speed), \
        f"Freeway crawl detected: v_cmd_min={v_cmd_min:.2f} m/s, set={set_speed:.2f} m/s"


    # Hidden-turn braking is validated by existing tests in
    # docs/chauffeur/vtsc/testing/integration/test_abrupt_hidden_turn.py
    # We keep this file focused on freeway no-crawl behavior.
