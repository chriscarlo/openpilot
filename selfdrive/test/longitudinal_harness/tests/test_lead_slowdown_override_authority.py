"""Regression contract for post-MPC lead-slowdown override authority.

The 2026-08-03 marked EV6 route reproduced repeated freeway commands where
the MPC requested only mild follow correction while the post-MPC slowdown
ceiling forced exactly -1.0 m/s^2.  Representative capture state was
vEgo=35.58 m/s, dRel=59.4 m, published vRel=-3.0 m/s, and aLeadK near zero.
The close-threat safety anchors were materially different: 7-10 m gaps,
1.7-2.5 s collision horizons, and measured lead deceleration.

This file owns that architectural boundary: the existing ceiling may keep its
state and release history, but ordinary follow and keep-up remain the MPC's
job unless independent urgent evidence authorizes the final override.
"""
from __future__ import annotations

import functools
from types import SimpleNamespace

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  get_lead_slowdown_accel_ceiling,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import arbitrate_lead_slowdown_ceiling
from openpilot.selfdrive.test.longitudinal_harness.closed_loop import SimulationResult, run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput


CAPTURE_V_EGO_MPS = 35.58
CAPTURE_D_REL_M = 59.40
CAPTURE_V_REL_MPS = -3.00
CAPTURE_A_LEAD_MPS2 = -0.005


def _lead(*, d_rel_m: float, v_ego_mps: float, v_rel_mps: float, a_lead_mps2: float):
  return SimpleNamespace(
    status=True,
    dRel=d_rel_m,
    vRel=v_rel_mps,
    vLead=v_ego_mps + v_rel_mps,
    aLeadK=a_lead_mps2,
  )


def test_far_marked_closure_has_no_post_mpc_authority() -> None:
  lead = _lead(
    d_rel_m=CAPTURE_D_REL_M,
    v_ego_mps=CAPTURE_V_EGO_MPS,
    v_rel_mps=CAPTURE_V_REL_MPS,
    a_lead_mps2=CAPTURE_A_LEAD_MPS2,
  )
  # The captured stateful release tail was exactly -1.0 m/s^2. A stateless
  # call to the underlying law cannot recreate that history, so inject the
  # recorded boundary value and test the final authority decision directly.
  raw_ceiling = -1.0
  effective_ceiling, debug = arbitrate_lead_slowdown_ceiling(
    v_ego=CAPTURE_V_EGO_MPS,
    t_follow=1.60,
    lead=lead,
    mpc_accel=-0.20,
    model_accel=0.02,
    slowdown_ceiling=raw_ceiling,
  )

  assert effective_ceiling == -0.20
  assert debug["reason"] == "uncorroborated_false_brake_mpc_authority"
  assert float(debug["collision_ttc_s"]) > 10.0
  assert float(debug["required_kinematic_decel_mps2"]) < 0.5


def test_lagged_closing_estimate_cannot_block_accelerating_lead_keepup() -> None:
  lead = _lead(
    d_rel_m=58.0,
    v_ego_mps=30.0,
    # The published velocity can lag the lead's positive acceleration.  The
    # secondary layer must not turn that stale close into any acceleration
    # cap; the MPC retains full authority to keep up.
    v_rel_mps=-1.2,
    a_lead_mps2=1.0,
  )
  raw_ceiling = get_lead_slowdown_accel_ceiling(
    v_ego=30.0,
    lead=lead,
    t_follow=1.60,
  )
  effective_ceiling, debug = arbitrate_lead_slowdown_ceiling(
    v_ego=30.0,
    t_follow=1.60,
    lead=lead,
    mpc_accel=0.45,
    model_accel=0.55,
    slowdown_ceiling=raw_ceiling,
  )

  assert raw_ceiling is not None
  assert effective_ceiling is None
  assert debug["reason"] == "departing_lead_mpc_authority"


def test_close_braking_lead_preserves_emergency_override_authority() -> None:
  lead = _lead(d_rel_m=10.41, v_ego_mps=6.83, v_rel_mps=-4.09, a_lead_mps2=-0.89)
  raw_ceiling = get_lead_slowdown_accel_ceiling(
    v_ego=6.83,
    lead=lead,
    t_follow=1.60,
  )
  effective_ceiling, debug = arbitrate_lead_slowdown_ceiling(
    v_ego=6.83,
    t_follow=1.60,
    lead=lead,
    mpc_accel=-0.40,
    model_accel=-0.80,
    slowdown_ceiling=raw_ceiling,
  )

  assert raw_ceiling is not None and raw_ceiling < -0.5
  assert effective_ceiling == raw_ceiling
  assert debug["urgent"] is True
  assert debug["reason"] == "lead_deceleration"
  assert float(debug["collision_ttc_s"]) < 3.0


def _accelerating_lead_steps() -> list[StepInput]:
  steps = []
  duration_s = 8.0
  initial_lead_speed_mps = 28.8
  lead_accel_mps2 = 1.0
  for idx in range(int(round(duration_s / DT_MDL))):
    t_s = idx * DT_MDL
    v_lead_mps = min(34.0, initial_lead_speed_mps + lead_accel_mps2 * t_s)
    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=34.5,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=v_lead_mps,
        model_prob_target=0.99,
        a_lead_k_mps2=lead_accel_mps2 if v_lead_mps < 34.0 else 0.0,
        d_rel_override_m=45.0 if idx == 0 else None,
        acquisition_reset=idx == 0,
      ),
      note="accelerating lead with initially lagged closing velocity",
    ))
  return steps


@functools.lru_cache(maxsize=1)
def _run_accelerating_lead() -> SimulationResult:
  return run_harness(
    vehicle_config=resolve_ev6_vehicle_config(),
    scenario_name="post_mpc_override_accelerating_lead",
    steps=_accelerating_lead_steps(),
    initial_speed_mps=30.0,
    noise_profile="off",
    seed=42,
    perception_filter="radard",
  )


def test_full_ev6_path_leaves_accelerating_lead_keepup_to_mpc() -> None:
  result = _run_accelerating_lead()
  lead_rows = [row for row in result.trace if row["planner_source"] == "lead0"]
  assert lead_rows

  nonurgent_rows = [
    row for row in lead_rows
    if row["lead_one_published_d_rel_m"] is not None
    and row["lead_one_published_d_rel_m"] > 35.0
  ]
  assert nonurgent_rows
  assert any(row["planner_accel_mps2"] > 0.20 for row in nonurgent_rows)
  assert any(
    row["mpc_acc_source_debug"]["lead_slowdown_arbitration"]["reason"] == "departing_lead_mpc_authority"
    for row in nonurgent_rows
  )
  assert all(
    row["mpc_acc_source_debug"]["lead_slowdown_arbitration"]["urgent"] is False
    and row["mpc_acc_source_debug"]["lead_slowdown_arbitration"]["reason"] in {
      "inactive", "departing_lead_mpc_authority", "legacy_ceiling_authority",
      "uncorroborated_false_brake_mpc_authority",
    }
    for row in nonurgent_rows
  )
