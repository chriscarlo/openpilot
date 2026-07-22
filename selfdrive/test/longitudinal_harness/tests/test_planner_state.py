from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from openpilot.selfdrive.test.longitudinal_harness.closed_loop import run_harness
from openpilot.selfdrive.test.longitudinal_harness.config import resolve_ev6_vehicle_config
from openpilot.selfdrive.test.longitudinal_harness.inputs import StepInput
from openpilot.selfdrive.test.longitudinal_harness.planner_state import (
  build_route_start_initialization_claim,
  verify_route_start_initialization_claim,
  well_formed_route_start_initialization_claim,
)
from openpilot.selfdrive.test.longitudinal_harness.provenance import exact_planner_state_initialization
from openpilot.selfdrive.test.longitudinal_harness.route_extract import (
  _build_planner_route_start_provenance,
  CachedLongitudinalPlan,
  CachedLongitudinalPlanSP,
)


def _proof() -> dict:
  return {
    "status": "diagnostic",
    "version": 1,
    "formalFidelityEligible": False,
    "sourcePublicationCompletenessAttested": False,
    "loadedSegmentStart": 0,
    "loadedSegmentEnd": 1,
    "segmentsContiguous": True,
    "captureFirstLogMonoTimeNs": 100,
    "captureLastLogMonoTimeNs": 1_200,
    "startOfRouteMonoTimeNs": 200,
    "plannerProcessPid": 41,
    "plannerProcessStartMonoTimeNs": 150,
    "radardProcessStartMonoTimeNs": 160,
    "plannerProcessUnique": True,
    "radardProcessUnique": True,
    "plannerManagerIdentityExact": True,
    "plannerPublicationsPairedExactly": True,
    "plannerPublicationCount": 1,
    "firstPlannerPlanMonoTimeNs": 1_000,
    "firstPlannerModelMonoTimeNs": 900,
    "firstPlannerRadarStateMonoTimeNs": 800,
    "firstPlannerModelIsFirstLoggedModel": True,
    "reasons": [],
  }


def _steps() -> list[StepInput]:
  return [
    StepInput(
      t_s=0.0,
      cruise_speed_mps=20.0,
      recorded_perception_mode="published_seed",
      recorded_radar_state_log_mono_time_ns=800,
      planner_radar_resolution="missing",
      replay_reference={"logMonoTimeNs": 800},
    ),
    StepInput(
      t_s=0.05,
      cruise_speed_mps=20.0,
      recorded_perception_mode="radard",
      recorded_radar_state_log_mono_time_ns=850,
      planner_radar_state_log_mono_time_ns=800,
      planner_radar_state_candidates_ns=[800],
      planner_radar_resolution="exact",
      replay_reference={"logMonoTimeNs": 1_000},
    ),
  ]


def _vehicle():
  return resolve_ev6_vehicle_config(
    topology="lfa",
    controller_mode="passthrough",
    perception_filter="direct",
    livetune_snapshot=None,
  )


def test_route_start_diagnostic_hashes_observed_input_prefix_and_params() -> None:
  steps = _steps()
  vehicle = _vehicle()
  claim = build_route_start_initialization_claim(
    route_start_proof=_proof(),
    steps=steps,
    initial_speed_mps=0.0,
    initial_accel_mps2=0.0,
    params=vehicle.params,
  )

  assert well_formed_route_start_initialization_claim(claim)
  assert exact_planner_state_initialization(claim, restoration_verified=True) is False
  assert verify_route_start_initialization_claim(
    claim,
    steps=steps,
    initial_speed_mps=0.0,
    initial_accel_mps2=0.0,
    params=vehicle.params,
  )

  steps[1].cruise_speed_mps += 0.1
  assert not verify_route_start_initialization_claim(
    claim,
    steps=steps,
    initial_speed_mps=0.0,
    initial_accel_mps2=0.0,
    params=vehicle.params,
  )


def test_run_harness_verifies_route_start_claim_before_advancing_state() -> None:
  steps = _steps()
  vehicle = _vehicle()
  claim = build_route_start_initialization_claim(
    route_start_proof=_proof(),
    steps=steps,
    initial_speed_mps=0.0,
    initial_accel_mps2=0.0,
    params=vehicle.params,
  )
  for step in steps:
    step.replay_reference["plannerStateInitializationProvenance"] = deepcopy(claim)

  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="route_start_state",
    steps=steps,
    initial_speed_mps=20.0,
    noise_profile="off",
    perception_filter="direct",
  )

  assert result.planner_state_initialization_applied is True
  assert result.planner_state_restoration_verified is False
  assert result.planner_state_initialization_provenance == claim
  assert not any(row["planner_state_initialization_exact"] for row in result.trace)


def test_route_start_diagnostic_never_promotes_after_vehicle_config_drift() -> None:
  steps = _steps()
  vehicle = _vehicle()
  claim = build_route_start_initialization_claim(
    route_start_proof=_proof(),
    steps=steps,
    initial_speed_mps=0.0,
    initial_accel_mps2=0.0,
    params=vehicle.params,
  )
  for step in steps:
    step.replay_reference["plannerStateInitializationProvenance"] = deepcopy(claim)

  # This used to leave the route-start path formally "verified" because CP was
  # absent from its hash. It may remain a useful diagnostic, but never exact.
  vehicle.cp.vEgoStarting += 7.0
  result = run_harness(
    vehicle_config=vehicle,
    scenario_name="route_start_config_drift",
    steps=steps,
    initial_speed_mps=20.0,
    noise_profile="off",
    perception_filter="direct",
  )
  assert result.planner_state_initialization_applied is True
  assert result.planner_state_restoration_verified is False
  assert not any(row["planner_state_initialization_exact"] for row in result.trace)


def test_run_harness_rejects_truncated_or_edited_route_prefix() -> None:
  steps = _steps()
  vehicle = _vehicle()
  claim = build_route_start_initialization_claim(
    route_start_proof=_proof(),
    steps=steps,
    initial_speed_mps=0.0,
    initial_accel_mps2=0.0,
    params=vehicle.params,
  )
  for step in steps:
    step.replay_reference["plannerStateInitializationProvenance"] = deepcopy(claim)
  steps[1].cruise_speed_mps += 1.0

  with pytest.raises(ValueError, match="does not match the replay prefix"):
    run_harness(
      vehicle_config=vehicle,
      scenario_name="tampered_route_start_state",
      steps=steps,
      initial_speed_mps=20.0,
      noise_profile="off",
      perception_filter="direct",
    )


def test_route_start_claim_rejects_capture_that_did_not_precede_daemon_start() -> None:
  proof = _proof()
  proof["captureFirstLogMonoTimeNs"] = proof["plannerProcessStartMonoTimeNs"]

  with pytest.raises(ValueError, match="missing or invalid"):
    build_route_start_initialization_claim(
      route_start_proof=proof,
      steps=_steps(),
      initial_speed_mps=0.0,
      initial_accel_mps2=0.0,
      params={},
    )


def test_process_start_proof_requires_segment_zero_unique_daemons_and_exact_plan_pair() -> None:
  plan = CachedLongitudinalPlan(
    log_mono_time_ns=1_000,
    model_mono_time_ns=900,
    solver_execution_time_s=0.001,
    radar_state_mono_time_ns=800,
    v_cruise_deprecated_mps=20.0,
    a_target_mps2=0.0,
    source="cruise",
  )
  plan_sp = CachedLongitudinalPlanSP(
    log_mono_time_ns=1_005,
    slc_active=False,
    slc_state="inactive",
    slc_speed_limit_mps=0.0,
    slc_speed_limit_offset_mps=0.0,
    vtsc_state="disabled",
    vtsc_velocity_mps=0.0,
    object_hazard_active=False,
    replay_inputs_valid=True,
    replay_inputs_version=1,
    replay_effective_cruise_mps=20.0,
    replay_plan_log_mono_time_ns=1_000,
    replay_input_clocks_ns={"modelV2": 900, "radarState": 800},
  )
  paired = replace(plan, plan_sp=plan_sp)
  kwargs = {
    "segment_rows": [{"seg_idx": 0}, {"seg_idx": 1}],
    "capture_first_log_mono_time_ns": 100,
    "capture_last_log_mono_time_ns": 1_200,
    "start_of_route_mono_times_ns": [200],
    "process_identities": {"plannerd": {(41, 150)}, "radard": {(42, 160)}},
    "manager_plannerd_observations": [(300, 41, True, True)],
    "model_v2_mono_times_ns": [900],
    "plans": [plan],
    "paired_plans": [paired],
    "plan_sp_messages": [plan_sp],
  }

  exact = _build_planner_route_start_provenance(**kwargs)
  assert exact["status"] == "diagnostic"
  assert exact["formalFidelityEligible"] is False
  assert exact["sourcePublicationCompletenessAttested"] is False

  missing_segment_zero = _build_planner_route_start_provenance(
    **{**kwargs, "segment_rows": [{"seg_idx": 1}]},
  )
  assert missing_segment_zero["status"] == "missing"
  assert "segment-zero" in " ".join(missing_segment_zero["reasons"])

  restarted = _build_planner_route_start_provenance(
    **{
      **kwargs,
      "process_identities": {"plannerd": {(41, 150), (43, 700)}, "radard": {(42, 160)}},
    },
  )
  assert restarted["status"] == "missing"
  assert "non-unique" in " ".join(restarted["reasons"])
