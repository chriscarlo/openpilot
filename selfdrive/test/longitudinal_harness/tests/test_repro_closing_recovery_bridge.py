"""Full-loop regression coverage for the RadarD calm-recovery brake-release bridge.

The fixture holds recorded ego state constant so candidate-versus-rollback can
isolate the perception/planner contract without plant feedback changing later
MPC inputs.  Raw model leads still pass through the real RadarD tracker.
"""
from __future__ import annotations

from functools import cache

import pytest

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.test.longitudinal_harness.closed_loop import run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import LeadDirective, StepInput


EGO_SPEED_MPS = 29.0
CRUISE_SPEED_MPS = 34.0
INITIAL_GAP_M = 50.4
TARGET_HEADWAY_S = 1.70
DURATION_S = 4.0
SLOWDOWN_START_S = 1.5
RECOVERY_START_S = 2.15
THREAT_START_S = 3.05
RECOVERY_BRIDGE_CAP_MPS2 = 0.10
BRAKE_AREA_WINDOW = (2.5, 3.5)

BRIDGE_PARAM = "Longitudinal.LiveTune.ClosingRecoveryBridgeMaxPositionClosingMps"

CALM = "calm"
RENEWED_BRAKING = "renewed_braking"
LATERAL_AMBIGUITY = "lateral_ambiguity"


def _timeline(variant: str) -> list[StepInput]:
  steps: list[StepInput] = []
  count = int(round(DURATION_S / DT_MDL))
  recovery_gap_at_threat = 49.23 + 0.7 * (THREAT_START_S - RECOVERY_START_S)

  for idx in range(count):
    t_s = idx * DT_MDL
    if t_s < SLOWDOWN_START_S:
      measured_d_rel_m = INITIAL_GAP_M
      measured_v_rel_mps = 0.0
      a_lead_mps2 = 0.0
    elif t_s < RECOVERY_START_S:
      measured_d_rel_m = INITIAL_GAP_M - 1.8 * (t_s - SLOWDOWN_START_S)
      measured_v_rel_mps = -1.2
      a_lead_mps2 = -0.7
    else:
      measured_d_rel_m = 49.23 + 0.7 * (t_s - RECOVERY_START_S)
      measured_v_rel_mps = -0.2
      a_lead_mps2 = 0.0

    d_path_m = 0.0
    if variant == RENEWED_BRAKING and t_s >= THREAT_START_S:
      measured_d_rel_m = recovery_gap_at_threat - 2.0 * (t_s - THREAT_START_S)
      measured_v_rel_mps = -2.0
      a_lead_mps2 = -1.0
    elif variant == LATERAL_AMBIGUITY and t_s >= THREAT_START_S:
      d_path_m = 1.6

    steps.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=CRUISE_SPEED_MPS,
      recorded_v_ego_mps=EGO_SPEED_MPS,
      recorded_a_ego_mps2=0.0,
      lead_one=LeadDirective(
        status=True,
        v_lead_mps=EGO_SPEED_MPS + measured_v_rel_mps,
        model_prob_target=0.99,
        d_rel_override_m=INITIAL_GAP_M if idx == 0 else None,
        measured_d_rel_m=measured_d_rel_m,
        measured_v_rel_mps=measured_v_rel_mps,
        a_lead_k_mps2=a_lead_mps2,
        d_path_m=d_path_m,
        v_lat_mps=0.0,
        exact_model_prob=True,
      ),
      note="real-RadarD calm-recovery bridge fixture",
    ))
  return steps


def _params(bridge_value: str | None) -> dict[str, str]:
  params = {
    f"VibeTune.Follow.Standard.Headway{idx}": f"{TARGET_HEADWAY_S:.2f}"
    for idx in range(4)
  }
  params["Longitudinal.LiveTune.ModelLeadFilterFastVRelTauS"] = "0.50"
  if bridge_value is not None:
    params[BRIDGE_PARAM] = bridge_value
  return params


@cache
def _run(variant: str, bridge_value: str | None):
  vehicle = resolve_ev6_vehicle_config(
    controller_mode="device",
    perception_filter="radard",
    param_overrides=_params(bridge_value),
  )
  return run_harness(
    vehicle_config=vehicle,
    scenario_name=f"closing_recovery_bridge_{variant}",
    steps=_timeline(variant),
    initial_speed_mps=EGO_SPEED_MPS,
    initial_accel_mps2=0.0,
    noise_profile="off",
    ego_replay_mode="recorded",
  )


UPSTREAM_KEYS = (
  "planner_source",
  "planner_should_stop",
  "planner_fcw",
  "mpc_crash_cnt",
  "planner_gap_reclaim_floor_mps2",
  "planner_lead_keepup_floor_mps2",
  "planner_lead_slowdown_ceiling_mps2",
  "planner_cutin_settle_floor_mps2",
  "planner_lead_brake_release_floor_mps2",
  "planner_lead_present_cruise_cap_mps2",
  "planner_accel_clip_min_mps2",
  "planner_accel_clip_max_mps2",
  "planner_input_v_ego_mps",
  "planner_input_a_ego_mps2",
  "planner_t_follow_s",
  "planner_v_ego_mps",
  "planner_active_obstacle_m",
  "planner_active_obstacle_gap_m",
  "planner_headway_gap_m",
  "planner_comfort_obstacle_distance_m",
  "planner_comfort_obstacle_surplus_m",
  "planner_lead_danger_factor",
  "planner_danger_obstacle_surplus_m",
  "planner_desired_true_gap_m",
  "lead_one_status",
  "lead_two_status",
  "lead_one_measured_d_rel_m",
  "lead_two_measured_d_rel_m",
  "lead_one_published_d_rel_m",
  "lead_two_published_d_rel_m",
  "lead_one_raw_d_rel_m",
  "lead_two_raw_d_rel_m",
  "lead_one_raw_v_rel_mps",
  "lead_two_raw_v_rel_mps",
  "lead_one_raw_v_lead_mps",
  "lead_two_raw_v_lead_mps",
  "lead_one_raw_a_lead_k_mps2",
  "lead_two_raw_a_lead_k_mps2",
  "lead_one_raw_model_prob",
  "lead_two_raw_model_prob",
  "lead_one_published_v_rel_mps",
  "lead_two_published_v_rel_mps",
  "lead_one_published_v_lead_mps",
  "lead_two_published_v_lead_mps",
  "lead_one_published_a_lead_k_mps2",
  "lead_two_published_a_lead_k_mps2",
  "lead_one_fcw_suppressed",
  "lead_two_fcw_suppressed",
  "lead_one_radard_debug",
  "lead_two_radard_debug",
  "mpc_acc_source_debug",
  "mpc_lead_role_debug",
  "mpc_steady_parity_debug",
  "mpc_cutin_settle_debug",
  "mpc_lead_preview_debug",
  "mpc_adjacent_awareness_preview_debug",
  "mpc_hyundai_virtual_lead_debug",
  "planner_cruise_reacquire_debug",
  "planner_relatch_blend_debug",
  "planner_handoff_limit_debug",
  "planner_comfort_jerk_debug",
)


def _without_bridge(debug: dict) -> dict:
  result = dict(debug)
  result.pop("closing_recovery_bridge", None)
  return result


def _assert_upstream_exact(rollback, candidate) -> None:
  assert len(candidate.trace) == len(rollback.trace)
  for idx, (rollback_row, candidate_row) in enumerate(zip(rollback.trace, candidate.trace, strict=True)):
    assert candidate_row["t_s"] == rollback_row["t_s"]
    for key in UPSTREAM_KEYS:
      assert candidate_row[key] == rollback_row[key], f"upstream drift at row {idx} t={rollback_row['t_s']:.2f}: {key}"
    assert _without_bridge(candidate_row["planner_lead_brake_release_debug"]) == \
      _without_bridge(rollback_row["planner_lead_brake_release_debug"]), \
      f"ordinary brake-release drift at row {idx} t={rollback_row['t_s']:.2f}"


def _bridge(row: dict) -> dict:
  return row["planner_lead_brake_release_debug"]["closing_recovery_bridge"]


def _brake_area(rows: list[dict], lo_s: float, hi_s: float) -> float:
  return sum(
    max(0.0, -float(row["planner_accel_mps2"])) * DT_CTRL
    for row in rows
    if lo_s <= float(row["t_s"]) < hi_s
  )


def test_default_off_is_exactly_explicit_rollback() -> None:
  implicit_default = _run(CALM, None)
  explicit_rollback = _run(CALM, "0")

  assert implicit_default.trace == explicit_rollback.trace
  assert not any(_bridge(row).get("candidate") or _bridge(row).get("applied") for row in implicit_default.trace)


def test_real_radard_recovery_proof_reduces_brake_area_without_upstream_drift() -> None:
  rollback = _run(CALM, "0")
  candidate = _run(CALM, "1.25")
  _assert_upstream_exact(rollback, candidate)

  applied_rows = [row for row in candidate.trace if _bridge(row).get("applied")]
  assert applied_rows, "real RadarD never produced an applied calm-recovery bridge frame"
  for row in applied_rows:
    bridge = _bridge(row)
    radard = row["lead_one_radard_debug"]
    assert radard["closing_governor_debug_exact"] is True
    assert radard["closing_governor_calm_recovery_mode"] is True
    assert radard["closing_governor_calm_recovery_applied"] is True
    assert radard["closing_governor_recovery_position_closing_mps"] is not None
    assert radard["closing_governor_recovery_vrel_floor_mps"] is not None
    assert bridge["position_closing_mps"] == pytest.approx(
      radard["closing_governor_recovery_position_closing_mps"])
    assert bridge["recovery_vrel_floor_mps"] == pytest.approx(
      radard["closing_governor_recovery_vrel_floor_mps"])

  deltas = [
    candidate_row["planner_accel_mps2"] - rollback_row["planner_accel_mps2"]
    for rollback_row, candidate_row in zip(rollback.trace, candidate.trace, strict=True)
  ]
  assert min(deltas) >= -1e-9
  assert max(deltas) <= RECOVERY_BRIDGE_CAP_MPS2 + 1e-9
  assert max(deltas) > 0.05

  lo_s, hi_s = BRAKE_AREA_WINDOW
  rollback_area = _brake_area(rollback.trace, lo_s, hi_s)
  candidate_area = _brake_area(candidate.trace, lo_s, hi_s)
  area_detail = f"calm-recovery brake area: rollback={rollback_area:.6f} candidate={candidate_area:.6f}"
  assert rollback_area - candidate_area >= 0.03, area_detail


@pytest.mark.parametrize(
  ("variant", "expected_reason"),
  [
    (RENEWED_BRAKING, "control_lead_braking"),
    (LATERAL_AMBIGUITY, "invalid_numeric_provenance"),
  ],
)
def test_same_frame_threat_revokes_to_exact_rollback(variant: str, expected_reason: str) -> None:
  rollback = _run(variant, "0")
  candidate = _run(variant, "1.25")
  _assert_upstream_exact(rollback, candidate)

  pre_threat = [row for row in candidate.trace if row["t_s"] < THREAT_START_S - 1e-6]
  assert any(_bridge(row).get("applied") for row in pre_threat), "fixture never armed recovery before the threat"

  post_pairs = [
    (rollback_row, candidate_row)
    for rollback_row, candidate_row in zip(rollback.trace, candidate.trace, strict=True)
    if rollback_row["t_s"] >= THREAT_START_S - 1e-6
  ]
  assert post_pairs
  first_rollback, first_candidate = post_pairs[0]
  assert first_candidate["t_s"] == pytest.approx(THREAT_START_S)
  assert _bridge(first_candidate)["reason"] == expected_reason
  assert _bridge(first_candidate)["candidate"] is False
  assert _bridge(first_candidate)["applied"] is False

  if variant == LATERAL_AMBIGUITY:
    assert first_candidate["lead_one_radard_debug"]["closing_governor_recovery_vrel_floor_mps"] is None
    assert first_candidate["lead_one_radard_debug"]["steady_parity_reason"] == "lateral_ambiguity"

  assert first_candidate["planner_accel_mps2"] == first_rollback["planner_accel_mps2"]
  assert all(
    candidate_row["planner_accel_mps2"] == rollback_row["planner_accel_mps2"]
    for rollback_row, candidate_row in post_pairs
  )
